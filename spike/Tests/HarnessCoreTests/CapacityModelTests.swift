import XCTest
@testable import HarnessCore

/// TDD against the spec's CONFIRMED per-model numbers (`docs/superpowers/specs/
/// 2026-07-09-system-aware-context-operability.md` §2.2/§2.3). Tolerance is ±3% unless the
/// comment on a given assertion says otherwise — these are hand-computed reconciliations against
/// a real config's head geometry, not round-tripped from the implementation itself.
final class CapacityModelTests: XCTestCase {
    private let gib = 1024.0 * 1024.0 * 1024.0

    private func model(_ id: String) -> ModelArchProfile {
        guard let m = ModelArchProfile.catalog.first(where: { $0.id == id }) else {
            XCTFail("missing catalog entry \(id)"); fatalError("missing catalog entry \(id)")
        }
        return m
    }

    private func assertClose(_ actual: Double, _ expected: Double, tolerance: Double = 0.03, _ message: String) {
        let diff = abs(actual - expected) / expected
        XCTAssertLessThanOrEqual(diff, tolerance, "\(message): actual \(actual) vs expected \(expected) (off by \(diff * 100)%)")
    }

    // MARK: - §2.2 confirmed KV@32K numbers, per architecture class

    /// Qwen3-30B-A3B-2507 (uniform-GQA): KV @32K ≈ 3.0 GiB. Spec §2.2.
    func testUniformGQA_Qwen3_30B_KVAt32K() {
        let m = model("Qwen3-30B-A3B-2507")
        let kv = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(kv / gib, 3.0, "Qwen3-30B-A3B KV@32K")
    }

    /// Qwen3.6-35B-A3B (hybrid-linear): KV @32K ≈ 0.625 GiB, AND ≈4x below the naive
    /// all-layers-uniform calc (proves the dispatch — not just the class label — matters). Spec §2.2.
    func testHybridLinear_Qwen36_35B_KVAt32K_and4xBelowNaive() {
        let m = model("Qwen3.6-35B-A3B")
        let kv = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(kv / gib, 0.625, "Qwen3.6-35B-A3B KV@32K")

        // Naive: treat every layer as uniform-GQA (ignore the hybrid-linear dispatch entirely).
        let naive = ModelArchProfile(
            id: "naive-uniform-comparator", modelType: .uniformGQA, nLayers: m.nLayers, nAttnLayers: m.nLayers,
            nKVHeads: m.nKVHeads, headDim: m.headDim, nativeMaxContext: m.nativeMaxContext,
            weightsBytes4bitEstimate: m.weightsBytes4bitEstimate, license: m.license)
        let naiveKV = CapacityModel.kvBytesForContext(naive, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(naiveKV / kv, 4.0, "naive/dispatched ratio")
    }

    /// Gemma-3-27B (interleaved-SWA): KV @32K ≈ 2.91 GiB (global growth + local-capped fixed term). Spec §2.2.
    func testInterleavedSWA_Gemma3_27B_KVAt32K() {
        let m = model("Gemma-3-27B")
        let kv = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(kv / gib, 2.91, "Gemma-3-27B KV@32K")
    }

    /// Qwen3-32B (uniform-GQA): KV @32K ≈ 8.0 GiB. Spec §2.2.
    func testUniformGQA_Qwen3_32B_KVAt32K() {
        let m = model("Qwen3-32B")
        let kv = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(kv / gib, 8.0, "Qwen3-32B KV@32K")
    }

    /// DeepSeek-R1 (MLA-as-implemented, the honesty case): KV @32K ≈ 152.5 GiB, per-token ≈4.88 MiB.
    /// Spec §2.2. Per-token tolerance is looser (5%): the spec's own "4.88 MiB" figure is itself
    /// ~2.4% off the value that exactly reconciles KV@32K (see ModelArchProfile.swift comment) —
    /// treated here as a KiB/1000-vs/1024 rounding slip in the spec's prose, not a formula bug.
    func testMLAAsImplemented_DeepSeekR1_KVAt32KAndPerToken() {
        let m = model("DeepSeek-R1")
        let kv = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        assertClose(kv / gib, 152.5, "DeepSeek-R1 KV@32K")

        let perTokenBytes = CapacityModel.kvBytesPerToken(m, kvQuant: .fp16)
        let mib = 1024.0 * 1024.0
        assertClose(perTokenBytes / mib, 4.88, tolerance: 0.05, "DeepSeek-R1 KV/token")
    }

    // MARK: - §4 effective default context

    /// Phi-4's native max (16,384) sits below the 32K tunable default — effectiveDefault must be
    /// the native max, not 32K (spec §4/§8, the honesty case the design must not paper over).
    func testEffectiveDefaultContext_Phi4BelowDefault() {
        let m = model("Phi-4-14B")
        XCTAssertEqual(CapacityModel.effectiveDefaultContext(m), 16384)
    }

    /// A model whose native max clears 32K keeps the 32K default.
    func testEffectiveDefaultContext_AboveDefaultStaysAt32K() {
        let m = model("Qwen3-30B-A3B-2507")
        XCTAssertEqual(CapacityModel.effectiveDefaultContext(m), 32768)
    }

    // MARK: - §5 classify: green/yellow/red + binding constraint

    /// Qwen3-235B-A22B (~117.5 GiB weights + ~5.875 GiB KV @32K): red on the 128GB bench box
    /// (weights alone nearly exhaust the 115GB wired limit), green on a 256GB M3 Ultra. Spec §2.3.
    func testClassify_Qwen3_235B_RedOnM5Max128_GreenOnM3Ultra256() {
        let m = model("Qwen3-235B-A22B")
        let predOn128 = CapacityModel.predictPeakBytes(
            model: m, context: 32768, concurrency: 1, kvQuant: .fp16, profile: .m5Max128)
        let verdict128 = CapacityModel.classify(predOn128, profile: .m5Max128, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertEqual(verdict128.color, .red, "Qwen3-235B on m5Max128")

        let predOn256 = CapacityModel.predictPeakBytes(
            model: m, context: 32768, concurrency: 1, kvQuant: .fp16, profile: .m3Ultra256)
        let verdict256 = CapacityModel.classify(predOn256, profile: .m3Ultra256, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertEqual(verdict256.color, .green, "Qwen3-235B on m3Ultra256")
    }

    /// DeepSeek-R1 @32K: red even on a 512GB M3 Ultra, with bindingConstraint naming the KV
    /// formula itself (MLA-as-implemented) as the wall — not generic RAM pressure. Spec §5/§7/§8.
    func testClassify_DeepSeekR1_RedOnM3Ultra512_BindingConstraintIsMLA() {
        let m = model("DeepSeek-R1")
        let prediction = CapacityModel.predictPeakBytes(
            model: m, context: 32768, concurrency: 1, kvQuant: .fp16, profile: .m3Ultra512)
        let verdict = CapacityModel.classify(prediction, profile: .m3Ultra512, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertEqual(verdict.color, .red)
        XCTAssertEqual(verdict.bindingConstraint, .mlaAsImplemented)
    }

    // MARK: - concurrency + KV-quant tiers

    /// Concurrency multiplies KV linearly: 4x Qwen3-32B @32K KV ≈ 32 GiB. Spec §2 ("N_concurrent × KV").
    func testConcurrencyMultipliesKV() {
        let m = model("Qwen3-32B")
        let kv1 = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        let kv4 = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 4)
        XCTAssertEqual(kv4, kv1 * 4, accuracy: 1.0, "concurrency must multiply KV linearly")
        assertClose(kv4 / gib, 32.0, "4x Qwen3-32B KV@32K")
    }

    /// int8 KV tier is ~half of fp16 (1 byte/elem vs 2); turbo4 is strictly smaller than int8
    /// (0.5 byte/elem incl. documented metadata tax vs 1.0). Spec §6 mitigation #1.
    func testKVQuantTiers_Int8HalfOfFp16_Turbo4BelowInt8() {
        let m = model("Qwen3-32B")
        let fp16 = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .fp16, concurrency: 1)
        let int8 = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .int8, concurrency: 1)
        let turbo4 = CapacityModel.kvBytesForContext(m, context: 32768, kvQuant: .turbo4, concurrency: 1)
        assertClose(int8, fp16 / 2, "int8 vs fp16/2")
        XCTAssertLessThan(turbo4, int8, "turbo4 must be strictly below int8")
    }

    // MARK: - honesty: derivability + hard model-capability limits

    /// KV is derivable for confirmed arches; NOT derivable where the attention-layer count is an
    /// unconfirmed sentinel (Nemotron `nemotron_h`) or the arch is out of scope (V4-Flash). Spec §2.1/§8.
    func testIsKVDerivable() {
        XCTAssertTrue(model("Qwen3-30B-A3B-2507").isKVDerivable, "uniform-GQA is derivable")
        XCTAssertTrue(model("DeepSeek-R1").isKVDerivable, "MLA-as-implemented is derivable")
        XCTAssertFalse(model("Nemotron-3-Ultra").isKVDerivable, "unconfirmed attn-layer count → not derivable")
        XCTAssertFalse(model("DeepSeek-V4-Flash").isKVDerivable, "out-of-scope arch → not derivable")
    }

    /// A non-derivable model must classify as `.kvNotDerivable` — NOT a fit color built on a
    /// one-layer under-count or a fabricated zero (the silent-wrong this tool exists to prevent).
    func testClassify_NotDerivable_ReturnsKVNotDerivable() {
        for id in ["Nemotron-3-Ultra", "DeepSeek-V4-Flash"] {
            let m = model(id)
            let prediction = CapacityModel.predictPeakBytes(
                model: m, context: 32768, concurrency: 1, kvQuant: .fp16, profile: .m3Ultra512)
            let verdict = CapacityModel.classify(prediction, profile: .m3Ultra512, weightsBytes: Double(m.weightsBytes4bitEstimate))
            XCTAssertEqual(verdict.bindingConstraint, .kvNotDerivable, "\(id) KV must be flagged non-derivable, not under-counted")
            XCTAssertEqual(verdict.color, .red, "\(id): cannot claim a fit we can't derive")
        }
    }

    /// A context beyond the model's native max is red with `.modelNativeMax` regardless of memory —
    /// a hard model-capability limit, not a hardware one (spec §4). Even on a 512GB box.
    func testClassify_ContextExceedsNativeMax_BindingModelNativeMax() {
        let m = model("Qwen3-30B-A3B-2507") // native max 262,144
        let prediction = CapacityModel.predictPeakBytes(
            model: m, context: 300_000, concurrency: 1, kvQuant: .fp16, profile: .m3Ultra512)
        let verdict = CapacityModel.classify(prediction, profile: .m3Ultra512, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertEqual(verdict.color, .red)
        XCTAssertEqual(verdict.bindingConstraint, .modelNativeMax)
    }

    /// The default-selection advisory names `.nativeMaxBelowDefault` for a model that tops out below
    /// 32K (Phi-4), and `nil` for one that clears it (spec §4/§8).
    func testDefaultContextAdvisory() {
        XCTAssertEqual(CapacityModel.defaultContextAdvisory(model("Phi-4-14B")), .nativeMaxBelowDefault)
        XCTAssertNil(CapacityModel.defaultContextAdvisory(model("Qwen3-30B-A3B-2507")))
    }

    // MARK: - §3.1 cacheLimit invariant — explicit, far below MLX's 1.5x-wired-limit default.

    /// llmbench-shaped: 115 GiB wired limit -> 1/8th = 14.375 GiB, in-range (not the ~172 GiB the
    /// 1.5x default would allow) — the exact "7K wall" mechanism the invariant exists to prevent.
    func testRecommendedCacheLimitBytes_MidRange_IsOneEighthOfWiredLimit() {
        let wiredLimit = Int(115 * gib)
        let cacheLimit = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: wiredLimit)
        assertClose(Double(cacheLimit) / gib, 14.375, "115GiB wired limit -> cacheLimit")
    }

    /// A tiny box (8 GiB wired limit) floors at 4 GiB rather than shrinking to an unusably small
    /// cache (1 GiB via the raw 1/8th formula).
    func testRecommendedCacheLimitBytes_SmallBox_FloorsAt4GiB() {
        let wiredLimit = Int(8 * gib)
        let cacheLimit = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: wiredLimit)
        XCTAssertEqual(cacheLimit, 4 * Int(gib))
    }

    /// A huge box (512 GiB wired limit) caps at 24 GiB rather than letting the cache hoard 64 GiB
    /// via the raw 1/8th formula — the anti-hoard cap.
    func testRecommendedCacheLimitBytes_HugeBox_CapsAt24GiB() {
        let wiredLimit = Int(512 * gib)
        let cacheLimit = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: wiredLimit)
        XCTAssertEqual(cacheLimit, 24 * Int(gib))
    }
}
