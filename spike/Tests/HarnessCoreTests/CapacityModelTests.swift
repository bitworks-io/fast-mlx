import XCTest
@testable import HarnessCore

/// TDD against the reviewed system-aware context-operability specification's confirmed per-model
/// numbers (§2.2/§2.3). Tolerance is ±3% unless the
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

    /// Qwen3.6-35B-A3B (hybrid-linear). The spec §2.2 confirmed number is the GROWING attention term
    /// alone: KV @32K ≈ 0.625 GiB (20 KiB/tok × 32K), AND ≈4x below the naive all-layers-uniform calc
    /// (proves the dispatch — not just the class label — matters). The separately-reconciled fixed
    /// GatedDeltaNet recurrent+conv state (`fixedStateBytes`) is pinned on its own below so the
    /// spec-confirmed growing term keeps its exact provenance (`kvBytesForContext` adds both).
    func testHybridLinear_Qwen36_35B_KVAt32K_and4xBelowNaive() {
        let m = model("Qwen3.6-35B-A3B")
        // Spec §2.2 confirmed the growing attention term only; assert it on its own accessor so the
        // 0.625 GiB reconciliation stays exact independent of the fixed state.
        let growingKV = CapacityModel.kvBytesPerToken(m, kvQuant: .fp16) * 32768
        assertClose(growingKV / gib, 0.625, "Qwen3.6-35B-A3B growing KV@32K")

        // Fixed per-sequence GatedDeltaNet state, reconciled from the real text_config linear_* geometry
        // (30 linear layers × 2,146,304 B = conv 49,152 + ssm 2,097,152). Pinned as a constant, not
        // folded into the spec's growing number.
        XCTAssertEqual(m.fixedStateBytes, 64_389_120)

        // Naive: treat every layer as uniform-GQA (ignore the hybrid-linear dispatch entirely). Compare
        // growing-vs-growing so the fixed term stays orthogonal to the dispatch ratio.
        let naive = ModelArchProfile(
            id: "naive-uniform-comparator", modelType: .uniformGQA, nLayers: m.nLayers, nAttnLayers: m.nLayers,
            nKVHeads: m.nKVHeads, headDim: m.headDim, nativeMaxContext: m.nativeMaxContext,
            weightsBytes4bitEstimate: m.weightsBytes4bitEstimate, license: m.license)
        let naiveGrowingKV = CapacityModel.kvBytesPerToken(naive, kvQuant: .fp16) * 32768
        assertClose(naiveGrowingKV / growingKV, 4.0, "naive/dispatched ratio")
    }

    /// Qwen3.5-9B (hybrid-linear, live hybrid-serving checkpoint): 8 of 32 layers grow a KV cache
    /// (full_attention_interval 4), nKVHeads 4 × head_dim 256 → 32 KiB/tok → exactly 1.0 GiB @ 32K
    /// growing, AND 4x below the naive all-32-layers calc. Confirmed from the checkpoint's config.json.
    /// The 24 linear layers' fixed GatedDeltaNet state is pinned separately below.
    func testHybridLinear_Qwen35_9B_KVAt32K_and4xBelowNaive() {
        let m = model("Qwen3.5-9B")
        XCTAssertTrue(m.isKVDerivable, "Qwen3.5-9B KV must be derivable (nAttnLayers > 0)")
        // Growing attention term only (kvBytesForContext also adds fixedStateBytes); the spec-confirmed
        // 1.0 GiB is the growing term, held to tol 0.001 because 8×4×256×2×2 B reconciles exactly.
        let growingKV = CapacityModel.kvBytesPerToken(m, kvQuant: .fp16) * 32768
        assertClose(growingKV / gib, 1.0, tolerance: 0.001, "Qwen3.5-9B growing KV@32K")

        // Fixed per-sequence recurrent+conv state (24 linear layers × 2,146,304 B = conv 49,152 + ssm
        // 2,097,152), reconciled from the checkpoint's config.json linear_* geometry.
        XCTAssertEqual(m.fixedStateBytes, 51_511_296)

        // Naive: treat every layer as uniform-GQA (ignore the hybrid-linear dispatch entirely). Compare
        // growing-vs-growing so the fixed term stays orthogonal to the dispatch ratio.
        let naive = ModelArchProfile(
            id: "naive-uniform-comparator", modelType: .uniformGQA, nLayers: m.nLayers, nAttnLayers: m.nLayers,
            nKVHeads: m.nKVHeads, headDim: m.headDim, nativeMaxContext: m.nativeMaxContext,
            weightsBytes4bitEstimate: m.weightsBytes4bitEstimate, license: m.license)
        let naiveGrowingKV = CapacityModel.kvBytesPerToken(naive, kvQuant: .fp16) * 32768
        assertClose(naiveGrowingKV / growingKV, 4.0, "naive/dispatched ratio")
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

    // MARK: - dual-geometry SWA (Baichuan-M1): DIFFERENT geometry per layer class + fixed state

    /// Baichuan-M1-14B (`.dualGeometrySWA`): the growing term uses GLOBAL geometry, the window-capped
    /// term uses SWA geometry (2× the head product), and a per-layer conv state is added on top —
    /// all three simultaneously. Exact integers hand-computed from the source-verified 14B config
    /// (docs/task-inbox/2026-08-19-baichuan-m1-dual-geometry-arch-reach.md): 40 layers = 20 global
    /// (kv 2 × dim 256) + 20 SWA (kv 8 × dim 128), window 8192.
    private func baichuanM1() -> ModelArchProfile {
        ModelArchProfile(
            id: "Baichuan-M1-14B", modelType: .dualGeometrySWA, nLayers: 40, nAttnLayers: 20,
            nKVHeads: 2, headDim: 256, slidingWindow: 8192, fixedStateBytes: 122_880,
            nativeMaxContext: 32_768, weightsBytes4bitEstimate: 7 * 1024 * 1024 * 1024,
            license: "Apache-2.0", swaKVHeads: 8, swaHeadDim: 128)
    }

    func testDualGeometrySWA_BaichuanM1_growingTermUsesGlobalGeometry() {
        // 20 global layers × 2 kv × 256 dim × 2 (K+V) × 2 B (fp16) = 40,960 B/token.
        XCTAssertEqual(CapacityModel.kvBytesPerToken(baichuanM1(), kvQuant: .fp16), 40_960)
    }

    func testDualGeometrySWA_BaichuanM1_windowCapUsesSWAGeometryNotGlobal() {
        // 20 SWA layers × 8 kv × 128 dim × 2 (K+V) × 2 B × 8192-token window = 671,088,640 B.
        let swaCap = CapacityModel.swaFixedLocalBytes(baichuanM1(), kvQuant: .fp16)
        XCTAssertEqual(swaCap, 671_088_640)
        // Forcing the GLOBAL head product (512) instead of the SWA product (1024) would halve this to
        // 335,544,320 B — a 320 MiB/seq PHANTOM-GREEN under-count. Assert the correct term is exactly
        // 2× that, i.e. the SWA geometry is honored, not silently borrowed from the global layers.
        XCTAssertEqual(swaCap, 2 * 335_544_320)
    }

    func testDualGeometrySWA_BaichuanM1_perSequenceAddsAllThreeTerms() {
        // grow(40,960 × 32768 = 1,342,177,280) + swaCap(671,088,640) + fixedState(122,880).
        // Unlike .interleavedSWA, the fixed conv state is NOT dropped for this class.
        let perSeq = CapacityModel.kvBytesForContext(baichuanM1(), context: 32_768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(perSeq, 2_013_388_800)
        // concurrency multiplies the whole per-sequence total.
        let at4 = CapacityModel.kvBytesForContext(baichuanM1(), context: 32_768, kvQuant: .fp16, concurrency: 4)
        XCTAssertEqual(at4, 4 * 2_013_388_800)
    }

    /// Asymmetric K/V (mimo_v2_flash shape): K cached at `headDim`, V at a narrower `vHeadDim`, on both
    /// layer classes. The growing/cap terms size K and V separately (`headDim + vHeadDim`), never
    /// `2×headDim`. Constructed from the real MiMo-V2-Flash geometry (9 global + 39 SWA, K 192 / V 128).
    private func mimoV2Flash() -> ModelArchProfile {
        ModelArchProfile(
            id: "MiMo-V2-Flash", modelType: .dualGeometrySWA, nLayers: 48, nAttnLayers: 9,
            nKVHeads: 4, headDim: 192, slidingWindow: 128, fixedStateBytes: 0,
            nativeMaxContext: 262_144, weightsBytes4bitEstimate: 12 * 1024 * 1024 * 1024,
            license: "Apache-2.0", swaKVHeads: 8, swaHeadDim: 192,
            vHeadDim: 128, swaVHeadDim: 128)
    }

    func testDualGeometrySWA_MimoV2Flash_asymmetricKVSizesKAndVSeparately() {
        let m = mimoV2Flash()
        // grow: 9 global × 4 kv × (192 K + 128 V) × 2 B = 23,040 B/token
        XCTAssertEqual(CapacityModel.kvBytesPerToken(m, kvQuant: .fp16), 23_040)
        // SWA cap: 39 SWA × 8 kv × (192 + 128) × 2 B × 128 window = 25,559,040 B
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(m, kvQuant: .fp16), 25_559_040)
        // KV@32K/seq = grow(23,040)×32768 + cap(25,559,040) + fixedState(0) = 780,533,760 B;
        // concurrency multiplies it.
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(m, context: 32_768, kvQuant: .fp16, concurrency: 1), 780_533_760)
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(m, context: 32_768, kvQuant: .fp16, concurrency: 4), 4 * 780_533_760)
    }

    /// The `vHeadDim` generalization is NUMBER-PRESERVING for symmetric models: a profile with
    /// `vHeadDim == nil` and one with `vHeadDim == headDim` produce identical KV bytes (both recover
    /// `2×headDim`). This is why every existing catalog entry's assertions stayed green.
    func testKVBytes_vHeadDimNilEqualsExplicitSymmetric() {
        let base = model("Qwen3-32B")  // uniformGQA, vHeadDim nil
        let explicitSymmetric = ModelArchProfile(
            id: base.id, modelType: base.modelType, nLayers: base.nLayers, nAttnLayers: base.nAttnLayers,
            nKVHeads: base.nKVHeads, headDim: base.headDim, slidingWindow: base.slidingWindow,
            fixedStateBytes: base.fixedStateBytes, nativeMaxContext: base.nativeMaxContext,
            weightsBytes4bitEstimate: base.weightsBytes4bitEstimate, license: base.license,
            vHeadDim: base.headDim)  // explicitly V == K
        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(base, kvQuant: .fp16),
            CapacityModel.kvBytesPerToken(explicitSymmetric, kvQuant: .fp16),
            "nil vHeadDim must equal explicit vHeadDim == headDim")
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

    /// Qualified-host shape: 115 GiB wired limit -> 1/8th = 14.375 GiB, in-range (not the ~172 GiB
    /// the 1.5x default would allow) — the exact "7K wall" mechanism the invariant prevents.
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
