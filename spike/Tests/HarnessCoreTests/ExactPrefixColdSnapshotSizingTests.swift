import XCTest
@testable import HarnessCore

/// TDD for the cold-snapshot BYTE plane — the modeled KV size of a persistable cold snapshot
/// (`docs/task-inbox/2026-07-12-exact-prefix-session-cache.md`, roadmap #3a). The token-axis
/// planner (`ExactPrefixColdSnapshotPlanner.plan`/`planRestore`) answers "how many blocks may be
/// persisted"; this answers the sibling question the SSD budget needs: "how many BYTES does a
/// persisted snapshot of N tokens cost". It is single-source-locked to `CapacityModel` (the one KV
/// formula) and pins fp16 — the runtime allocates fp16 KV, so a snapshot of live KV is fp16, and a
/// quant parameter here would fabricate a size the disk write never has.
///
/// Fail-closed spine: sizing succeeds ONLY for the arch classes whose cold-restore arithmetic is
/// audited safe (`.uniformGQA`, `.hybridLinear`) — exactly the classes `ServingSnapshotBridge`
/// marks reusable. Every other class throws rather than returning a fabricated or zero size that
/// could make an unservable snapshot look like it fits the budget.
final class ExactPrefixColdSnapshotSizingTests: XCTestCase {

    private func profile(
        _ modelType: ArchClass, nLayers: Int, nAttnLayers: Int, nKVHeads: Int, headDim: Int,
        fixedStateBytes: Int = 0, slidingWindow: Int? = nil
    ) -> ModelArchProfile {
        ModelArchProfile(
            id: "test-\(modelType.rawValue)", modelType: modelType, nLayers: nLayers,
            nAttnLayers: nAttnLayers, nKVHeads: nKVHeads, headDim: headDim,
            slidingWindow: slidingWindow, fixedStateBytes: fixedStateBytes,
            nativeMaxContext: 262_144, weightsBytes4bitEstimate: 1_000_000, license: "test")
    }

    // MARK: - supported-class arithmetic (independent hand-computed literals)

    /// uniformGQA every layer grows: 4 layers × 2 kv-heads × 8 head-dim × 2 (K and V) × 2 B (fp16)
    /// = 256 B/token; × 96 tokens = 24,576 B. Literal is hand-computed, NOT re-derived from the
    /// formula under test.
    func testUniformGQASnapshotBytes_matchesHandComputedGeometry() throws {
        let p = profile(.uniformGQA, nLayers: 4, nAttnLayers: 4, nKVHeads: 2, headDim: 8)
        let bytes = try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
            profile: p, persistedTokenCount: 96)
        XCTAssertEqual(bytes, 24_576, accuracy: 0.0)
    }

    /// hybridLinear grows only its attention-layer subset AND carries a fixed recurrent-state term:
    /// 2 attn-layers × 4 kv-heads × 16 head-dim × 2 × 2 B = 512 B/token; × 100 tokens = 51,200 B;
    /// + 2,048 B fixed state = 53,248 B. Uses a hand-made profile with NONZERO fixedStateBytes (the
    /// catalog Qwen3.5 entry is 0, which would not distinguish "includes fixed state" from "omits").
    func testHybridLinearSnapshotBytes_includesFixedState() throws {
        let p = profile(.hybridLinear, nLayers: 8, nAttnLayers: 2, nKVHeads: 4, headDim: 16,
                        fixedStateBytes: 2_048)
        let bytes = try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
            profile: p, persistedTokenCount: 100)
        XCTAssertEqual(bytes, 53_248, accuracy: 0.0)
    }

    /// Zero persisted tokens is a valid (empty) snapshot for a supported class: only the fixed state
    /// remains. hybridLinear with 2,048 B fixed state → 2,048 B; uniformGQA (no fixed state) → 0.
    func testZeroPersistedTokens_returnsFixedStateOnly() throws {
        let hybrid = profile(.hybridLinear, nLayers: 8, nAttnLayers: 2, nKVHeads: 4, headDim: 16,
                             fixedStateBytes: 2_048)
        XCTAssertEqual(
            try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                profile: hybrid, persistedTokenCount: 0), 2_048, accuracy: 0.0)
        let dense = profile(.uniformGQA, nLayers: 4, nAttnLayers: 4, nKVHeads: 2, headDim: 8)
        XCTAssertEqual(
            try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                profile: dense, persistedTokenCount: 0), 0, accuracy: 0.0)
    }

    // MARK: - fail-closed on every unsupported class

    /// interleavedSWA (RotatingKVCache — no arbitrary-boundary restore), mlaAsImplemented,
    /// hybridMamba2MoE, and novelCompressedUnsupported must THROW, never return 0 or a fabricated
    /// size. These are exactly the classes `predictedSnapshotReuse` returns `.unsupported` for.
    func testUnsupportedClassesThrow() {
        let unsupported: [ArchClass] = [
            .interleavedSWA, .mlaAsImplemented, .hybridMamba2MoE, .novelCompressedUnsupported,
        ]
        for cls in unsupported {
            let p = profile(cls, nLayers: 8, nAttnLayers: 4, nKVHeads: 2, headDim: 8,
                            slidingWindow: cls == .interleavedSWA ? 1024 : nil)
            XCTAssertThrowsError(
                try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                    profile: p, persistedTokenCount: 96),
                "sizing must fail closed for \(cls.rawValue)"
            ) { error in
                XCTAssertEqual(
                    error as? ExactPrefixColdSnapshotError,
                    .snapshotSizingUnsupported(cls))
            }
        }
    }

    /// A negative persisted-token count is not a valid snapshot — fail closed rather than compute a
    /// negative size.
    func testNegativeTokenCountThrows() {
        let p = profile(.uniformGQA, nLayers: 4, nAttnLayers: 4, nKVHeads: 2, headDim: 8)
        XCTAssertThrowsError(
            try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                profile: p, persistedTokenCount: -1)
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixColdSnapshotError, .invalidPersistedTokenCount(-1))
        }
    }

    // MARK: - single-source regression lock (guards a future hand-rolled reimplementation)

    /// Locks the sizer to the one KV formula: for every supported catalog profile, the modeled
    /// snapshot bytes equal `CapacityModel.kvBytesForContext` at fp16/concurrency-1. This is a
    /// regression lock against someone later inlining a divergent formula — NOT the correctness
    /// check (the hand-computed literals above are); it passes by construction today.
    func testSizingMatchesCapacityModelForSupportedCatalogProfiles() throws {
        let supported = ModelArchProfile.catalog.filter {
            $0.modelType == .uniformGQA || $0.modelType == .hybridLinear
        }
        XCTAssertFalse(supported.isEmpty, "catalog should contain supported profiles")
        for p in supported {
            for tokens in [0, 1, 512, 4096] {
                let modeled = try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                    profile: p, persistedTokenCount: tokens)
                let expected = CapacityModel.kvBytesForContext(
                    p, context: tokens, kvQuant: .fp16, concurrency: 1)
                XCTAssertEqual(modeled, expected, accuracy: 0.0, "\(p.id) @ \(tokens) tokens")
            }
        }
    }
}
