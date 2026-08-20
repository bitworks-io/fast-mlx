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
}
