import XCTest
@testable import HarnessCore

/// S2-prep of the hybrid continuous-batching build (design of record: the continuous-batching
/// heterogeneous-cache design). Pins the
/// per-layer cache GEOMETRY (dense KV + recurrent conv/SSM state) and its fail-closed derivation from
/// `config.json`, composed with the structural `HybridLayerKindMap` from S1. No MLX, no model load.
///
/// The geometry lives as a SEPARATE composed value type, NOT payloads on the shipped geometry-free
/// `LayerCacheKind` enum — see the decision record in
/// docs/task-inbox/2026-08-20-continuous-batching-s2-geometry-plan.md.
final class HybridCacheGeometryTests: XCTestCase {

    // MARK: Fixtures

    /// The real Qwen3.5-9B text geometry (from mlx-community/Qwen3.5-9B-MLX-4bit config.json,
    /// cross-referenced in ModelArchProfile.swift:153-167): 32 layers, interval 4 (8 full + 24 linear),
    /// GatedDeltaNet linear heads Hk=16 Hv=32, head dims Dk=Dv=128, conv kernel 4. Dense attention:
    /// num_key_value_heads 8, head_dim 128. dtype bfloat16.
    private func qwen35TextGeom() -> [String: Any] {
        [
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        ]
    }

    /// VL-wrapped: model_type at root, ALL text geometry (incl. dtype) under text_config only.
    private func qwen35VLConfig() -> Data {
        let root: [String: Any] = ["model_type": "qwen3_5", "text_config": qwen35TextGeom()]
        return try! JSONSerialization.data(withJSONObject: root)
    }

    /// Flat: model_type + geometry all at the root (no text_config wrapper).
    private func qwen35FlatConfig() -> Data {
        var root = qwen35TextGeom()
        root["model_type"] = "qwen3_5"
        return try! JSONSerialization.data(withJSONObject: root)
    }

    /// Synthetic asymmetric fixture (Hk=16,Dk=64, Hv=32,Dv=128) so that any transposition of the convDim
    /// formula `2·(Hk·Dk) + (Hv·Dv)` changes the value, and any K/V head-dim swap in the SSM term breaks.
    private func asymmetricConfig(layers: Int = 8, interval: Int = 4) -> Data {
        let root: [String: Any] = [
            "model_type": "qwen3_5",
            "num_hidden_layers": layers,
            "full_attention_interval": interval,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "torch_dtype": "float16",
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        ]
        return try! JSONSerialization.data(withJSONObject: root)
    }

    // MARK: 1. text_config unwrap (VL) + flat both decode

    func testVLWrappedAndFlatDeriveIdenticalGeometry() throws {
        let vl = try ModelConfigDecoder.qwen35HybridGeometry(configJSON: qwen35VLConfig())
        let flat = try ModelConfigDecoder.qwen35HybridGeometry(configJSON: qwen35FlatConfig())
        XCTAssertEqual(vl, flat, "VL text_config unwrap must yield the same geometry as the flat config")

        // Structure: 32 layers, interval-4 dense at [3,7,...,31], layer 0 recurrent.
        XCTAssertEqual(vl.map.layerCount, 32)
        XCTAssertEqual(vl.map.firstDenseLayerIndex, 3, "layer 0 is recurrent on qwen3_5")
        XCTAssertEqual(vl.map.denseLayerIndices, [3, 7, 11, 15, 19, 23, 27, 31])
        XCTAssertEqual(vl.map.recurrentLayerIndices.count, 24)

        // Dense geometry: 8 KV heads × 128 head dim × 2 bytes (bfloat16).
        XCTAssertEqual(vl.dense.kvHeads, 8)
        XCTAssertEqual(vl.dense.headDim, 128)
        XCTAssertEqual(vl.dense.elementBytes, 2)

        // Recurrent geometry: symmetric head dims here, Hk=16 Hv=32.
        XCTAssertEqual(vl.recurrent.valueHeads, 32)
        XCTAssertEqual(vl.recurrent.valueHeadDim, 128)
        XCTAssertEqual(vl.recurrent.keyHeadDim, 128)
        XCTAssertEqual(vl.recurrent.convKernelSize, 4)
        XCTAssertEqual(vl.recurrent.convElementBytes, 2)
        // convDim = 2·(Hk·Dk) + (Hv·Dv) = 2·(16·128) + (32·128) = 4096 + 4096 = 8192.
        XCTAssertEqual(vl.recurrent.convDim, 8192)
    }

    // MARK: 2. convDim transposition trap (asymmetric)

    func testConvDimAsymmetricTranspositionTrap() throws {
        let g = try ModelConfigDecoder.qwen35HybridGeometry(configJSON: asymmetricConfig())
        // keyDim = Hk·Dk = 16·64 = 1024; valueDim = Hv·Dv = 32·128 = 4096.
        // convDim = 2·keyDim + valueDim = 2048 + 4096 = 6144. Any swap (e.g. 2·valueDim + keyDim = 9216)
        // fails this exact assertion.
        XCTAssertEqual(g.recurrent.convDim, 6144)
        XCTAssertEqual(g.recurrent.keyHeadDim, 64)
        XCTAssertEqual(g.recurrent.valueHeadDim, 128)
        XCTAssertEqual(g.recurrent.valueHeads, 32)
    }

    // MARK: 3. Sizer-agreement byte cross-check (mirrors resolveHybridFixedStateBytes, MCD.swift:1042-1047)

    func testRecurrentBytesMatchAuditedSizerFormula() throws {
        let g = try ModelConfigDecoder.qwen35HybridGeometry(configJSON: asymmetricConfig())
        // Independent hand computation of the audited formula:
        //   convBytes = (kernel-1)·convDim·2 = 3·6144·2 = 36_864
        //   ssmBytes  = Hv·Dv·Dk·4 (fp32)     = 32·128·64·4 = 1_048_576
        let expectedConv = 3 * 6144 * 2
        let expectedSSM = 32 * 128 * 64 * 4
        XCTAssertEqual(g.recurrent.convBytesPerLayer, expectedConv)
        XCTAssertEqual(g.recurrent.ssmBytesPerLayer, expectedSSM, "SSM state is fp32 (×4) by vendored construction")
        XCTAssertEqual(g.recurrent.bytesPerLayer, expectedConv + expectedSSM)
        // Aggregate over the recurrent layers must equal the sizer's per-model fixed-state term.
        let nLinear = g.map.recurrentLayerIndices.count
        XCTAssertEqual(nLinear, 6, "8 layers interval 4 → dense {3,7} → 6 recurrent")
        XCTAssertEqual(g.recurrentBytesTotal, (expectedConv + expectedSSM) * nLinear)
    }

    // MARK: 4. Fail-closed derivation (admission-grade — must THROW where the sizer fails open with 0)

    func testDerivationFailsClosedOnMissingLinearKey() {
        for missing in [
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim", "num_key_value_heads", "head_dim",
            "num_hidden_layers",
        ] {
            var geom = qwen35TextGeom()
            geom[missing] = nil
            let root: [String: Any] = ["model_type": "qwen3_5", "text_config": geom]
            let data = try! JSONSerialization.data(withJSONObject: root)
            XCTAssertThrowsError(
                try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data),
                "missing \(missing) must fail closed (throw), never silently degrade")
        }
    }

    func testDerivationFailsClosedOnNonPositiveValue() {
        var geom = qwen35TextGeom()
        geom["linear_num_value_heads"] = 0
        let root: [String: Any] = ["model_type": "qwen3_5", "text_config": geom]
        let data = try! JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data))
    }

    func testDerivationFailsClosedOnWrongFamily() {
        var geom = qwen35TextGeom()
        let root: [String: Any] = ["model_type": "qwen3", "text_config": geom]
        _ = geom
        let data = try! JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data),
            "qwen35HybridGeometry is qwen3_5-scoped for v1 (GatedDeltaNet conv/SSM formula)")
    }

    func testDerivationFailsClosedOnUnknownDtype() {
        var geom = qwen35TextGeom()
        geom["torch_dtype"] = "int8"
        let root: [String: Any] = ["model_type": "qwen3_5", "text_config": geom]
        let data = try! JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data))
    }

    // MARK: 5-6. Index resolver agrees with the audited count resolver across all four config sources

    func testIndexResolverIntervalFourNotContainingZero() throws {
        let root: [String: Any] = [
            "model_type": "qwen3_5", "num_hidden_layers": 12, "full_attention_interval": 4,
        ]
        let data = try! JSONSerialization.data(withJSONObject: root)
        let idx = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(configJSON: data)
        XCTAssertEqual(idx, [3, 7, 11])
        XCTAssertFalse(idx.contains(0), "the layer-0 landmine: interval derivation never marks layer 0 dense")
        XCTAssertEqual(idx.count, 12 / 4, "index count agrees with the floor count resolver")
    }

    func testIndexResolverNonDivisibleInterval() throws {
        let root: [String: Any] = [
            "model_type": "qwen3_5", "num_hidden_layers": 10, "full_attention_interval": 4,
        ]
        let data = try! JSONSerialization.data(withJSONObject: root)
        let idx = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(configJSON: data)
        XCTAssertEqual(idx, [3, 7], "(i+1)%4==0 over 0..<10 → {3,7}; count 2 == floor(10/4)")
    }

    func testIndexResolverLayerTypesBothSpellings() throws {
        // lfm2/qwen spelling: "full_attention" vs "conv".
        let a: [String: Any] = [
            "model_type": "qwen3_5", "num_hidden_layers": 4,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
        ]
        let ai = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(
            configJSON: try! JSONSerialization.data(withJSONObject: a))
        XCTAssertEqual(ai, [2])
        // granitemoehybrid spelling: "attention" vs "mamba".
        let b: [String: Any] = [
            "model_type": "granitemoehybrid", "num_hidden_layers": 4,
            "layer_types": ["mamba", "attention", "mamba", "attention"],
        ]
        let bi = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(
            configJSON: try! JSONSerialization.data(withJSONObject: b))
        XCTAssertEqual(bi, [1, 3])
    }

    func testIndexResolverFullAttnIdxs() throws {
        let root: [String: Any] = [
            "model_type": "lfm2", "num_hidden_layers": 6, "full_attn_idxs": [1, 4],
        ]
        let idx = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(
            configJSON: try! JSONSerialization.data(withJSONObject: root))
        XCTAssertEqual(idx, [1, 4])
    }

    func testIndexResolverJambaPeriodOffset() throws {
        // jamba: attention layers are {i : i % period == offset}.
        let root: [String: Any] = [
            "model_type": "jamba", "num_hidden_layers": 8,
            "attn_layer_period": 4, "attn_layer_offset": 1,
        ]
        let idx = try ModelConfigDecoder.resolveHybridAttentionLayerIndices(
            configJSON: try! JSONSerialization.data(withJSONObject: root))
        XCTAssertEqual(idx, [1, 5])
    }

    func testIndexResolverFailsClosedWhenNoSource() {
        let root: [String: Any] = ["model_type": "qwen3_5", "num_hidden_layers": 8]
        XCTAssertThrowsError(
            try ModelConfigDecoder.resolveHybridAttentionLayerIndices(
                configJSON: try! JSONSerialization.data(withJSONObject: root)))
    }

    // MARK: HybridCacheGeometry composition invariants

    func testGeometryRejectsNonHeterogeneousMap() {
        // interval 1 → every layer dense → not heterogeneous → must route .fp16, never hybrid.
        let map = HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 1)!
        let dense = DenseKVGeometry(kvHeads: 8, headDim: 128, elementBytes: 2)
        let rec = RecurrentStateGeometry(
            convKernelSize: 4, convDim: 8192, valueHeads: 32, valueHeadDim: 128,
            keyHeadDim: 128, convElementBytes: 2)
        XCTAssertNil(HybridCacheGeometry(map: map, dense: dense, recurrent: rec),
                     "an all-dense map is not heterogeneous; hybrid geometry must refuse it")
    }

    // MARK: DenseKVGeometry aux term (QSA sparse-indexer rawKeys cache) — Increment A of
    // docs/task-inbox/2026-09-05-qwen4exp-fit-check-qsa-indexer-term-DECISION.md. Mirrors
    // `CapacityModel.kvBytesPerToken`'s `.hybridLinear` case, which applies the identical fixed-2-bytes,
    // never-scale-by-KV-quant rule to the same term on the sizer path.

    func testDenseKVGeometry_auxNil_matchesTodaysValue_regressionGuardForQwen35() {
        // qwen3_5 real dense geometry (8 KV heads, 128 head dim, bf16). qwen3_5 has no QSA indexer, so
        // leaving `auxPerLayerKeyDim` at its default `nil` must reproduce EXACTLY today's value —
        // byte-for-byte unchanged behavior for every existing hybrid family.
        let g = DenseKVGeometry(kvHeads: 8, headDim: 128, elementBytes: 2)
        XCTAssertNil(g.auxPerLayerKeyDim)
        XCTAssertEqual(g.bytesPerLayerPerToken, 2 * 8 * 128 * 2)
        XCTAssertEqual(g.bytesPerLayerPerToken, 4096)
    }

    /// Real Flash Next per-attention-layer geometry (`num_key_value_heads` 2, `head_dim` 256,
    /// `indexer_head_dim` 128) times the confirmed 12 full-attention layers of 48 (decision record +
    /// `ModelConfigDecoderTests.qwen4ExpLayerTypesJSON`) reproduces the DECISION record's exact
    /// aggregate literal: `12 · 2 · 2 · 256 · 2` (dense K+V) `+ 12 · 128 · 2` (aux) `= 24,576 + 3,072 =
    /// 27,648`. `bytesPerLayerPerToken` itself is a PER-LAYER quantity (as its name says); the ×12 here
    /// is applied explicitly in the test, matching how a real hybrid geometry sums `dense.bytesPerLayerPerToken`
    /// over each of its dense layer indices (see the cross-path agreement test for the full derivation).
    private static let flashNextAttentionLayers = 12

    func testDenseKVGeometry_withAux_addsFixedWidthTerm_exactLiteral() {
        let g = DenseKVGeometry(kvHeads: 2, headDim: 256, elementBytes: 2, auxPerLayerKeyDim: 128)
        XCTAssertEqual(g.bytesPerLayerPerToken, 2_304, "one layer: 2·2·256·2 (dense) + 128·2 (aux) = 2048+256")
        let total = g.bytesPerLayerPerToken * Self.flashNextAttentionLayers
        XCTAssertEqual(2 * 2 * 256 * 2 * Self.flashNextAttentionLayers, 24_576, "sanity: dense-only component")
        XCTAssertEqual(128 * DenseKVGeometry.auxElementBytes * Self.flashNextAttentionLayers, 3_072, "sanity: aux-only component")
        XCTAssertEqual(total, 27_648)
    }

    func testDenseKVGeometry_auxTerm_doesNotScaleWithElementBytes_pinsDecisionRecordRule() {
        // THE decisive assertion: at elementBytes 1 (e.g. an int8 KV-quant tier), the dense term halves
        // (24,576 → 12,288) but the aux term must NOT halve with it — it stays fixed at the indexer's
        // model-dtype width regardless of the KV-quant tier (12,288 + 3,072 = 15,360), NOT
        // 12,288 + 1,536. Scaling the aux term by `elementBytes` would UNDER-count the QSA indexer's
        // rawKeys cache under a lossy KV tier — the exact phantom-GREEN failure this term exists to
        // prevent (DECISION record, item 4: "Scaling it by bpe would under-count it under an int8 KV
        // tier — the same phantom-GREEN failure in a different place").
        let g = DenseKVGeometry(kvHeads: 2, headDim: 256, elementBytes: 1, auxPerLayerKeyDim: 128)
        let total = g.bytesPerLayerPerToken * Self.flashNextAttentionLayers
        XCTAssertEqual(total, 15_360)
        XCTAssertNotEqual(
            total, 12_288 + 128 * 1 * Self.flashNextAttentionLayers,
            "aux term must not scale down with elementBytes (would silently under-count under int8 KV)")
    }

    // MARK: Increment B — qwen4ExpHybridGeometry (Qwen3.8-Flash-Next), derived from config.json.
    // Field names/values (48 layers, 12 full-attention, `linear_*`, `indexer_head_dim`,
    // `indexer_kv_heads`) verified against the live fixtures in ModelConfigDecoderTests.swift
    // (`qwen4ExpLayerTypesJSON`, T1-T4e) before writing this fixture.

    /// 12 repetitions of [linear_attention, linear_attention, linear_attention, full_attention] — the
    /// CONFIRMED Flash Next layer pattern (48 layers, 12 full-attention), mirroring
    /// `ModelConfigDecoderTests.qwen4ExpLayerTypesJSON` (that helper is `private` to its own file, so
    /// this is an independent reproduction, not a shared import).
    private func qwen4ExpLayerTypes() -> [String] {
        var layerTypes: [String] = []
        for _ in 0..<12 { layerTypes += ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }
        return layerTypes
    }

    private func qwen4ExpGeom(dtype: String = "bfloat16") -> [String: Any] {
        [
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "torch_dtype": dtype,
            "layer_types": qwen4ExpLayerTypes(),
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
        ]
    }

    private func qwen4ExpFlatConfig(dtype: String = "bfloat16") -> Data {
        var root = qwen4ExpGeom(dtype: dtype)
        root["model_type"] = "qwen4_exp"
        return try! JSONSerialization.data(withJSONObject: root)
    }

    private func qwen4ExpNestedConfig(dtype: String = "bfloat16") -> Data {
        let root: [String: Any] = ["model_type": "qwen4_exp", "text_config": qwen4ExpGeom(dtype: dtype)]
        return try! JSONSerialization.data(withJSONObject: root)
    }

    // NAMING CONSTRAINT: these tests are named `testFlashNext_*`, not after the internal
    // implementation-family type name, because this file is part of the public projection and
    // `validate_public_repository.py` fails any projected file containing that marker. The
    // lowercase `qwen4_exp` config value and the `qwen4Exp*` helper/API spellings above are fine —
    // only the capitalised type-name form trips the scan. Match `CapacityModelTests`' existing
    // `testFlashNext_*` convention when adding cases here.
    func testFlashNext_flatAndNested_deriveIdenticalGeometry() throws {
        let flat = try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: qwen4ExpFlatConfig())
        let nested = try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: qwen4ExpNestedConfig())
        XCTAssertEqual(flat, nested)

        XCTAssertEqual(flat.map.layerCount, 48)
        XCTAssertEqual(flat.map.denseLayerIndices.count, 12, "12 full-attention layers of 48")
        XCTAssertEqual(flat.map.recurrentLayerIndices.count, 36)

        XCTAssertEqual(flat.dense.kvHeads, 2)
        XCTAssertEqual(flat.dense.headDim, 256)
        XCTAssertEqual(flat.dense.elementBytes, 2)
        XCTAssertEqual(flat.dense.auxPerLayerKeyDim, 128, "indexer_head_dim must populate the aux term")

        XCTAssertEqual(flat.recurrent.valueHeads, 48)
        XCTAssertEqual(flat.recurrent.valueHeadDim, 128)
        XCTAssertEqual(flat.recurrent.keyHeadDim, 128)
        XCTAssertEqual(flat.recurrent.convKernelSize, 4)
    }

    func testFlashNext_missingIndexerHeadDim_failsClosed() {
        var geom = qwen4ExpGeom()
        geom["indexer_head_dim"] = nil
        geom["model_type"] = "qwen4_exp"
        let data = try! JSONSerialization.data(withJSONObject: geom)
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data),
            "missing indexer_head_dim must fail closed, never silently omit the aux term")
    }

    func testFlashNext_indexerKVHeadsNotOne_failsClosed() {
        var geom = qwen4ExpGeom()
        geom["indexer_kv_heads"] = 2
        geom["model_type"] = "qwen4_exp"
        let data = try! JSONSerialization.data(withJSONObject: geom)
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data),
            "indexer_kv_heads != 1 must refuse rather than model an unverified tensor shape")
    }

    func testFlashNext_indexerKVHeadsAbsent_failsClosed() {
        var geom = qwen4ExpGeom()
        geom["indexer_kv_heads"] = nil
        geom["model_type"] = "qwen4_exp"
        let data = try! JSONSerialization.data(withJSONObject: geom)
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data),
            "indexer_kv_heads absent must fail closed, never silently default to 1")
    }

    func testFlashNext_wrongFamily_failsClosed() {
        var geom = qwen4ExpGeom()
        geom["model_type"] = "qwen3_5"
        let data = try! JSONSerialization.data(withJSONObject: geom)
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data),
            "qwen4ExpHybridGeometry is qwen4_exp/qwen4_exp_text-scoped")
    }

    func testFlashNext_missingLinearRecurrentKey_failsClosed() {
        for missing in [
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim",
        ] {
            var geom = qwen4ExpGeom()
            geom[missing] = nil
            geom["model_type"] = "qwen4_exp"
            let data = try! JSONSerialization.data(withJSONObject: geom)
            XCTAssertThrowsError(
                try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data),
                "missing \(missing) must fail closed (throw), never silently degrade")
        }
    }

    /// THE gate that matters: the continuous-batching geometry's per-token dense+aux total must agree
    /// with the INDEPENDENT, already-corrected sizer path (`ModelConfigDecoder.decode` →
    /// `ModelArchProfile.auxPerLayerKeyDim` → `CapacityModel.kvBytesPerToken`), which shares none of
    /// `qwen4ExpHybridGeometry`'s arithmetic assembly (only the JSON parse and the audited
    /// `requirePositiveInt`/`intOf`/`hybridAttentionLayerIndices` helpers are common). Both are also
    /// pinned against the hand-checked literal from the DECISION record, so this is not merely two
    /// paths agreeing with each other — see docs/task-inbox/2026-09-05-qwen4exp-fit-check-qsa-indexer-term-DECISION.md.
    func testFlashNext_crossPathAgreement_derivedIndependentlyFromSizerPath_pins27648() throws {
        let json = qwen4ExpFlatConfig()

        let geometry = try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: json)
        let geometryPerToken = geometry.dense.bytesPerLayerPerToken * geometry.map.denseLayerIndices.count

        let parsed = try ModelConfigDecoder.decode(configJSON: json, safetensorsBytes: 1, id: "cross-path-check")
        XCTAssertEqual(parsed.profile.auxPerLayerKeyDim, 128)
        let sizerPerToken = CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16)

        XCTAssertEqual(Double(geometryPerToken), sizerPerToken,
                       "continuous-batching geometry must agree with the independently-derived sizer path")
        XCTAssertEqual(sizerPerToken, 27_648, "hand-checked literal from the DECISION record")
        XCTAssertEqual(geometryPerToken, 27_648)

        // Recurrent side, hand-derived independently from the fixture's own linear_* values and the
        // `RecurrentStateGeometry` formula (NOT recomputed through the code under test):
        //   Hk=linear_num_key_heads=16, Dk=linear_key_head_dim=128,
        //   Hv=linear_num_value_heads=48, Dv=linear_value_head_dim=128, kernel=linear_conv_kernel_dim=4.
        //   convDim = 2*(Hk*Dk) + (Hv*Dv) = 2*(16*128) + (48*128) = 2*2048 + 6144 = 4096 + 6144 = 10_240.
        //   convBytesPerLayer = (kernel-1)*convDim*elementBytes(2, bfloat16) = 3*10_240*2 = 61_440.
        //   ssmBytesPerLayer  = Hv*Dv*Dk*4 (fp32) = 48*128*128*4 = 3_145_728.
        //   bytesPerLayer = 61_440 + 3_145_728 = 3_207_168.
        //   recurrentLayerIndices.count = 48 layers - 12 dense = 36.
        //   recurrentBytesTotal = 3_207_168 * 36 = 115_458_048.
        XCTAssertEqual(geometry.recurrent.convDim, 10_240)
        XCTAssertEqual(geometry.recurrent.convBytesPerLayer, 61_440)
        XCTAssertEqual(geometry.recurrent.ssmBytesPerLayer, 3_145_728)
        XCTAssertEqual(geometry.recurrent.bytesPerLayer, 3_207_168)
        XCTAssertEqual(geometry.map.recurrentLayerIndices.count, 36)
        XCTAssertEqual(geometry.recurrentBytesTotal, 115_458_048)
    }

    // MARK: qwen4ExpHybridGeometry — dtype and model_type spelling coverage (independent review gaps)

    func testFlashNext_float32Dtype_setsElementBytesFour() throws {
        let g = try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: qwen4ExpFlatConfig(dtype: "float32"))
        XCTAssertEqual(g.dense.elementBytes, 4)
    }

    func testFlashNext_unknownDtype_throws() {
        XCTAssertThrowsError(
            try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: qwen4ExpFlatConfig(dtype: "int4")),
            "an unrecognized torch_dtype must fail closed, never default to a guessed element width")
    }

    func testFlashNext_modelTypeTextVariant_isAccepted() throws {
        var root = qwen4ExpGeom()
        root["model_type"] = "qwen4_exp_text"
        let data = try! JSONSerialization.data(withJSONObject: root)
        let g = try ModelConfigDecoder.qwen4ExpHybridGeometry(configJSON: data)
        XCTAssertEqual(g.map.layerCount, 48)
        XCTAssertEqual(g.dense.auxPerLayerKeyDim, 128)
    }

    // MARK: bytesPerLayerPerTokenChecked() agrees with bytesPerLayerPerToken (Increment A structural fix)

    func testBytesPerLayerPerTokenChecked_agreesWithPlainProperty_forNormalGeometry() {
        let g = DenseKVGeometry(kvHeads: 2, headDim: 256, elementBytes: 2, auxPerLayerKeyDim: 128)
        XCTAssertEqual(g.bytesPerLayerPerTokenChecked(), g.bytesPerLayerPerToken)
        XCTAssertEqual(g.bytesPerLayerPerTokenChecked(), 2_304)

        let gNoAux = DenseKVGeometry(kvHeads: 8, headDim: 128, elementBytes: 2)
        XCTAssertEqual(gNoAux.bytesPerLayerPerTokenChecked(), gNoAux.bytesPerLayerPerToken)
        XCTAssertEqual(gNoAux.bytesPerLayerPerTokenChecked(), 4_096)
    }
}
