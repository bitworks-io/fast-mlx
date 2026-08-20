import XCTest
@testable import ServingSnapshotBridge
import HarnessCore
import ServingCore

/// TDD for the one tested mapping between ServingCore's snapshot-reuse classification and HarnessCore's
/// cold-plane restore granularity (roadmap #3a). The bridge's job is to make it impossible to pair a
/// model's cache layout with the wrong reuse arithmetic — and to fail closed, never defaulting, when a
/// layout has no audited reuse rule.
final class ServingSnapshotBridgeTests: XCTestCase {

    private func seq(_ n: Int) -> [Int] { Array(0..<n) }

    // MARK: - direct enum mapping

    func testBlockAligned_mapsToBlockAligned() throws {
        XCTAssertEqual(try ServingSnapshotBridge.coldSnapshotGranularity(for: .blockAligned), .blockAligned)
    }

    func testWholeSnapshotOnly_mapsToWholeSnapshotOnly() throws {
        XCTAssertEqual(try ServingSnapshotBridge.coldSnapshotGranularity(for: .wholeSnapshotOnly), .wholeSnapshotOnly)
    }

    /// `.unsupported` must THROW — never silently default to `.blockAligned` (the exact wrong-arithmetic
    /// bug the whole-snapshot rule exists to prevent).
    func testUnsupported_throws() {
        XCTAssertThrowsError(try ServingSnapshotBridge.coldSnapshotGranularity(for: .unsupported)) {
            XCTAssertEqual($0 as? ServingSnapshotBridgeError, .unsupportedSnapshotReuseLayout)
        }
    }

    /// Drift lock: every `ScalarServingSnapshotReuse` case has an explicit, asserted mapping. If a new
    /// case is added to the serving enum, the exhaustive `switch` in the bridge fails to compile; if a
    /// mapping is changed, this table breaks — either way the drift is caught, never silent.
    func testEveryServingReuseCaseIsExplicitlyMapped() {
        let expected: [(ScalarServingSnapshotReuse, ColdSnapshotReuseGranularity?)] = [
            (.blockAligned, .blockAligned),
            (.wholeSnapshotOnly, .wholeSnapshotOnly),
            (.unsupported, nil),   // nil == expected to throw
        ]
        for (reuse, mapping) in expected {
            if let mapping {
                XCTAssertEqual(try? ServingSnapshotBridge.coldSnapshotGranularity(for: reuse), mapping,
                    "\(reuse) should map to \(mapping)")
            } else {
                XCTAssertThrowsError(try ServingSnapshotBridge.coldSnapshotGranularity(for: reuse),
                    "\(reuse) must fail closed")
            }
        }
    }

    // MARK: - end-to-end from native cache layout (classifier + bridge)

    /// A dense layout classifies to `.blockAligned` and drives the block-floor arithmetic: the two
    /// whole blocks before a mid-snapshot divergence are reused.
    func testDenseLayout_planRestore_reusesPrecedingBlocks() throws {
        let stored = seq(64)                                   // 4 blocks @ bs 16
        var prompt = seq(40); prompt += Array(1000..<1024)     // diverges at token 40
        let plan = try ServingSnapshotBridge.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16,
            cacheKinds: [.denseAttention, .denseAttention])
        XCTAssertEqual(plan.restoredBlockCount, 2, "dense reuses the whole blocks before divergence")
        XCTAssertEqual(plan.restoredTokenCount, 32)
    }

    /// A hybrid layout containing `.recurrentState` (qwen3.5-style) classifies to `.wholeSnapshotOnly`,
    /// so the SAME mid-snapshot-divergence prompt reuses NOTHING — the conservative arithmetic is
    /// selected by construction, not by the caller remembering to pass the right flag.
    func testHybridRecurrentLayout_planRestore_reusesNothingOnMidDivergence() throws {
        let stored = seq(64)
        var prompt = seq(40); prompt += Array(1000..<1024)
        let plan = try ServingSnapshotBridge.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16,
            cacheKinds: [.denseAttention, .recurrentState, .denseAttention])
        XCTAssertEqual(plan.restoredBlockCount, 0, "recurrent state can't rewind to a mid boundary")
        XCTAssertEqual(plan.recomputeTokenCount, prompt.count)
    }

    /// An unsupported layout (any rotating/composite/unknown, or empty) throws before any reuse is
    /// planned — the composed path inherits the bridge's fail-closed stance.
    func testUnsupportedLayout_planRestore_throws() {
        for kinds in [[ScalarServingNativeCacheKind.rotatingAttention],
                      [.composite], [.unknown], [.denseAttention, .rotatingAttention], []] {
            XCTAssertThrowsError(try ServingSnapshotBridge.planRestore(
                storedPrefixTokens: seq(64), promptTokens: seq(64), blockSize: 16, cacheKinds: kinds)) {
                XCTAssertEqual($0 as? ServingSnapshotBridgeError, .unsupportedSnapshotReuseLayout,
                    "layout \(kinds) must fail closed")
            }
        }
    }

    /// The best-restore convenience routes through the same classification: on a hybrid layout a
    /// shorter fully-covered candidate beats a longer diverging one (whole-snapshot semantics), and an
    /// unsupported layout throws before any candidate is considered.
    func testHybridLayout_planBestRestore_prefersFullyCoveredCandidate() throws {
        let shortFull = seq(32)
        let longDiverging = seq(16) + Array(9000..<9048)       // 64 tokens, diverges at 16
        var prompt = seq(32); prompt += Array(80..<96)
        let sel = try ServingSnapshotBridge.planBestRestore(
            candidateStoredPrefixes: [longDiverging, shortFull], promptTokens: prompt, blockSize: 16,
            cacheKinds: [.recurrentState])
        XCTAssertEqual(sel?.candidateIndex, 1)
        XCTAssertEqual(sel?.plan.restoredTokenCount, 32)
    }

    func testUnsupportedLayout_planBestRestore_throws() {
        XCTAssertThrowsError(try ServingSnapshotBridge.planBestRestore(
            candidateStoredPrefixes: [seq(32)], promptTokens: seq(32), blockSize: 16,
            cacheKinds: [.unknown])) {
            XCTAssertEqual($0 as? ServingSnapshotBridgeError, .unsupportedSnapshotReuseLayout)
        }
    }

    /// Fail-closed planner inputs (non-whole-block stored snapshot) still surface as the planner's own
    /// error through the composed path — the bridge doesn't swallow them.
    func testDenseLayout_nonWholeBlockStored_surfacesPlannerError() {
        XCTAssertThrowsError(try ServingSnapshotBridge.planRestore(
            storedPrefixTokens: seq(30), promptTokens: seq(30), blockSize: 16,
            cacheKinds: [.denseAttention])) {
            XCTAssertEqual($0 as? ExactPrefixColdSnapshotError,
                .snapshotNotWholeBlock(storedTokenCount: 30, blockSize: 16))
        }
    }

    // MARK: - config-plane SAFE predictor (#3a seam)

    /// Every `ArchClass` case must have an explicit, asserted mapping — the exhaustive `switch` in
    /// `predictedSnapshotReuse` breaks compilation if a new arch class is added without a mapping.
    func testPredictedSnapshotReuse_everyArchClassMapped() {
        let expected: [(ArchClass, ScalarServingSnapshotReuse)] = [
            (.uniformGQA, .blockAligned),
            (.hybridLinear, .wholeSnapshotOnly),
            (.interleavedSWA, .unsupported),
            (.mlaAsImplemented, .unsupported),
            (.hybridMamba2MoE, .unsupported),
            (.novelCompressedUnsupported, .unsupported),
        ]
        for (archClass, mapping) in expected {
            XCTAssertEqual(ServingSnapshotBridge.predictedSnapshotReuse(for: archClass), mapping,
                "\(archClass) should predict \(mapping)")
        }
    }

    // MARK: - serve-startup announce line (roadmap #3a: surface the predictor at serve time)

    /// The announce line must (a) carry the exact predicted granularity rawValue for the arch class,
    /// (b) always be labeled `predicted` (never a measured claim), and (c) name the arch class. The
    /// value is delegated to `predictedSnapshotReuse`, so it can never drift from the fail-closed
    /// planner policy — this test pins the wire shape for every arch class.
    func testSnapshotReuseAnnounceLine_exactShapePerArchClass() {
        for archClass in ArchClass.allCases {
            let line = ServingSnapshotBridge.snapshotReuseAnnounceLine(for: archClass)
            let expectedReuse = ServingSnapshotBridge.predictedSnapshotReuse(for: archClass).rawValue
            XCTAssertEqual(
                line,
                "snapshot_reuse=\(expectedReuse) predicted arch_class=\(archClass.rawValue)",
                "announce line shape for \(archClass.rawValue)")
            XCTAssertTrue(line.contains(" predicted "),
                "announce must be labeled predicted, never a measured claim — got: \(line)")
        }
    }

    /// Consistency check: for the arch classes with an audited, known live cache-kind shape, the
    /// config-plane predictor agrees with the live classifier run over the representative kinds.
    func testPredictedSnapshotReuse_agreesWithLiveClassifier_forAuditedArchClasses() {
        XCTAssertEqual(
            ServingSnapshotBridge.predictedSnapshotReuse(for: .uniformGQA),
            snapshotReuseGranularity(for: [.denseAttention, .denseAttention]))
        XCTAssertEqual(
            ServingSnapshotBridge.predictedSnapshotReuse(for: .hybridLinear),
            snapshotReuseGranularity(for: [.recurrentState, .denseAttention]))
        XCTAssertEqual(
            ServingSnapshotBridge.predictedSnapshotReuse(for: .interleavedSWA),
            snapshotReuseGranularity(for: [.rotatingAttention, .denseAttention]))
    }

    /// Deliberate conservative divergence: MLA-as-implemented decompresses to dense per-head K/V
    /// before the cache write, so a dense-only LIVE layout would classify `.blockAligned` — but its
    /// cold-restore arithmetic is NOT yet audited off-box, so the config-plane predictor fails closed
    /// to `.unsupported` pending a live cache-kind audit on the M5. This asserts the divergence is
    /// intentional, not a bug: never apply unaudited arithmetic, even when the live shape looks safe.
    func testPredictedSnapshotReuse_mlaDivergesConservativelyFromDenseOnlyLiveLayout() {
        XCTAssertEqual(ServingSnapshotBridge.predictedSnapshotReuse(for: .mlaAsImplemented), .unsupported)
        XCTAssertEqual(snapshotReuseGranularity(for: [.denseAttention]), .blockAligned)
    }

    /// Cross-plane fail-closed lock: the cold-snapshot BYTE sizer must refuse EXACTLY the arch
    /// classes the reuse predictor refuses. A snapshot that can't be restored must never be sized
    /// (a budget/eviction decision on a byte figure for an unrestorable snapshot is a lie), and a
    /// restorable snapshot must always be sizeable. Iterating `ArchClass.allCases` (not a hand-list)
    /// means a newly added arch is covered automatically — the two supported-sets can't silently
    /// drift apart.
    func testSnapshotSizingRefusesExactlyTheClassesReuseRefuses() {
        for cls in ArchClass.allCases {
            let reuseSupported = ServingSnapshotBridge.predictedSnapshotReuse(for: cls) != .unsupported
            let p = ModelArchProfile(
                id: "parity-\(cls.rawValue)", modelType: cls, nLayers: 8, nAttnLayers: 4,
                nKVHeads: 2, headDim: 8, slidingWindow: cls == .interleavedSWA ? 1024 : nil,
                nativeMaxContext: 262_144, weightsBytes4bitEstimate: 1_000_000, license: "test")
            var sizingSucceeded = false
            do {
                _ = try ExactPrefixColdSnapshotPlanner.modeledSnapshotKVBytes(
                    profile: p, persistedTokenCount: 64)
                sizingSucceeded = true
            } catch {
                sizingSucceeded = false
            }
            XCTAssertEqual(
                sizingSucceeded, reuseSupported,
                "sizing-support (\(sizingSucceeded)) must match reuse-support (\(reuseSupported)) for \(cls.rawValue)")
        }
    }
}
