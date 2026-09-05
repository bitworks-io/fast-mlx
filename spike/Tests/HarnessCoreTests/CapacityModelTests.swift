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

    func testRecommendedCacheLimitBytes_BelowOrdinaryFloorStaysBeneathEffectiveCeiling() {
        let effectiveCeiling = 3 * Int(gib)
        let cacheLimit = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: effectiveCeiling)

        XCTAssertEqual(cacheLimit, effectiveCeiling / 2)
        XCTAssertLessThan(cacheLimit, effectiveCeiling)
    }

    // MARK: - shared-host effective memory ceiling

    func testSharedEffectiveCeilingLowerMeasuredWiredWinsAndHardwareHoldsDeductsReserveOnce() {
        let host = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 64 * Int(gib),
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: nil)

        XCTAssertEqual(host.effectiveMemoryCeiling.bytes, 64 * Int(gib))
        XCTAssertEqual(host.effectiveMemoryCeiling.source, .wiredLimit)
        XCTAssertEqual(
            host.hardwareHoldsBytes(weightsBytes: 10 * Int(gib), osReserveBytes: 4 * Int(gib)),
            50 * Int(gib),
            "hardwareHolds must subtract weights and reserve once from the effective ceiling")
    }

    func testNilZeroOrNegativeMetalRecommendationCannotRaiseSharedFallbackCeiling() {
        let fallback = 48 * Int(gib)
        for recommendedWorkingSetBytes in [nil, 0, -1] as [Int?] {
            let host = SystemProfile(
                chip: "test",
                totalRAMBytes: 64 * Int(gib),
                wiredLimitBytes: 50 * Int(gib),
                wiredLimitIsMeasured: true,
                recommendedWorkingSetBytes: recommendedWorkingSetBytes)

            XCTAssertEqual(host.effectiveMemoryCeiling.bytes, fallback)
            XCTAssertEqual(host.effectiveMemoryCeiling.source, .sharedPolicy)
        }
    }

    func testDedicatedServingIgnoresSharedPolicyAndMetalRecommendationForThisIncrement() {
        let host = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 64 * Int(gib),
            hostUse: .operatorAssertedDedicatedServing())

        XCTAssertEqual(host.effectiveMemoryCeiling.bytes, 115 * Int(gib))
        XCTAssertEqual(host.effectiveMemoryCeiling.source, .wiredLimit)
        XCTAssertEqual(
            host.hardwareHoldsBytes(weightsBytes: 10 * Int(gib), osReserveBytes: 4 * Int(gib)),
            101 * Int(gib))
    }

    // MARK: - Qwen3.8-Flash-Next (hybrid-linear): the catalog entry this increment adds.
    //
    // Ground truth is the real artifact's own config.json text_config and the 22 safetensors
    // headers of its published checkpoint, both read directly rather than assumed — see
    // ModelArchProfile.swift's entry comment for the full arithmetic, the vendored-cache
    // reconciliation, and the load-bearing caveat on the weights figure.

    /// The entry exists and is retrievable through the catalog's real lookup (`model(_:)` here
    /// wraps the same `ModelArchProfile.catalog.first(where:)` the production code uses).
    func testFlashNext_EntryExistsAndRetrievable() {
        let m = model("Qwen3.8-Flash-Next")
        XCTAssertEqual(m.modelType, .hybridLinear)
        XCTAssertEqual(m.nLayers, 48)
        XCTAssertEqual(m.nAttnLayers, 12)
        XCTAssertEqual(m.nKVHeads, 2)
        XCTAssertEqual(m.headDim, 256)
        XCTAssertEqual(m.nativeMaxContext, 262_144)
    }

    /// Growing KV/token has TWO terms on this family, not one:
    ///   standard K+V  = 12 full_attention layers × 2 kv heads × (256 K + 256 V) × 2 B (fp16) = 24,576
    ///   QSA indexer   = 12 full_attention layers × indexer_head_dim 128 × 2 B                =  3,072
    ///                                                                                   total = 27,648
    /// The indexer term was previously omitted here, under-counting by 12.5%. It is a SECOND growing
    /// per-attention-layer cache (`auxPerLayerKeyDim`): each full-attention layer's QSA indexer keeps
    /// its own `rawKeys` buffer, grown by `concatenated(..., axis: 1)` and NOT capped by
    /// `indexer_budget` — that budget caps sparse selection, not the stored raw keys. Verified against
    /// `spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpQSAIndexer.swift:332-341`.
    /// Produced by `CapacityModel.kvBytesPerToken`'s real hybrid-linear dispatch, not restated as a
    /// standalone literal computation in this test.
    func testFlashNext_GrowingKVPerToken() {
        let m = model("Qwen3.8-Flash-Next")
        let perToken = CapacityModel.kvBytesPerToken(m, kvQuant: .fp16)
        XCTAssertEqual(perToken, 27_648)
    }

    /// Growing KV at the model's full native max context (262,144 tokens), 1 sequence: 27,648 B/tok
    /// × 262,144 = 7,247,757,312 B (6.75 GiB) exactly. This is the growing attention term alone
    /// (kvBytesPerToken × context) — matching how the existing hybrid-linear entries above pin
    /// their growing term separately from `fixedStateBytes` (`kvBytesForContext` adds both).
    /// The 0.75 GiB above the previous 6.00 GiB figure is the QSA indexer `rawKeys` cache that this
    /// entry used to omit — it is real, resident, and grows with the sequence.
    func testFlashNext_KVAt262144Tokens() {
        let m = model("Qwen3.8-Flash-Next")
        let growingKV = CapacityModel.kvBytesPerToken(m, kvQuant: .fp16) * 262_144
        XCTAssertEqual(growingKV, 7_247_757_312)
        assertClose(growingKV / gib, 6.75, tolerance: 0.001, "Qwen3.8-Flash-Next growing KV@262,144")
    }

    /// Fixed per-sequence GatedDeltaNet state, reconciled against the vendored MambaCache
    /// allocation in the vendored gated-delta-net implementation: conv state
    /// (4−1)×10,240×2 B(bf16) = 61,440 +
    /// recurrent/SSM state 48×128×128×4 B(float32, enforced by `validateCache`) = 3,145,728 →
    /// 3,207,168 B/linear-layer × 36 linear layers (48 total − 12 full_attention) = 115,458,048 B.
    func testFlashNext_FixedStateBytesReconciled() {
        let m = model("Qwen3.8-Flash-Next")
        XCTAssertEqual(m.fixedStateBytes, 115_458_048)
    }

    /// End-to-end fit: PLE-offloaded weights (81,324,594,328 B measured) on a 128 GiB SHARED host
    /// (`.m5Max128`, `hostUse` defaults to `.shared`) at concurrency 1 / context 262,144 (the
    /// model's own native max) classifies GREEN, and the headroom `hardwareHoldsBytes` computes is
    /// ~16.26 GiB — the tool now reproduces the docs' hand-computed headroom figure. If this
    /// diverges from 16.26 GiB, that is a real discrepancy to report, not something to paper over
    /// by bending the assertion.
    func testFlashNext_Classify_GreenOn128GiBSharedHost_AtConcurrency1Context262144() {
        let m = model("Qwen3.8-Flash-Next")
        let profile = SystemProfile.m5Max128
        XCTAssertEqual(profile.hostUse.use, .shared, "must be a SHARED host, not dedicated-serving")

        let headroom = profile.hardwareHoldsBytes(
            weightsBytes: m.weightsBytes4bitEstimate, osReserveBytes: CapacityThresholds.default.osReserveBytes)
        XCTAssertEqual(headroom, 17_459_653_480)
        assertClose(Double(headroom) / gib, 16.26, tolerance: 0.01, "Qwen3.8-Flash-Next headroom on m5Max128 shared")

        let prediction = CapacityModel.predictPeakBytes(
            model: m, context: 262_144, concurrency: 1, kvQuant: .fp16, profile: profile)
        let verdict = CapacityModel.classify(
            prediction, profile: profile, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertEqual(verdict.color, .green, "Qwen3.8-Flash-Next @ concurrency 1 / context 262,144 on 128GiB shared")
        XCTAssertEqual(verdict.bindingConstraint, .fits)
    }

    /// Negative case: the SAME host/context, at a concurrency high enough that the classifier
    /// leaves GREEN — proving the catalog entry (not a hardcoded verdict) actually drives the
    /// classification. Concurrency 3 pushes the non-weights peak past the green threshold.
    func testFlashNext_Classify_HigherConcurrencyLeavesGreen() {
        let m = model("Qwen3.8-Flash-Next")
        let profile = SystemProfile.m5Max128

        let prediction = CapacityModel.predictPeakBytes(
            model: m, context: 262_144, concurrency: 3, kvQuant: .fp16, profile: profile)
        let verdict = CapacityModel.classify(
            prediction, profile: profile, weightsBytes: Double(m.weightsBytes4bitEstimate))
        XCTAssertNotEqual(verdict.color, .green, "concurrency 3 must push the same model/host/context off GREEN")
        XCTAssertEqual(verdict.color, .red)
    }
}
