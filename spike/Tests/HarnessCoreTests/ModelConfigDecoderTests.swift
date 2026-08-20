import XCTest
@testable import HarnessCore

/// TDD for the `config.json` → `ModelArchProfile` decoder (fit-checked-serve, differentiator #2).
/// The pure `decode(configJSON:safetensorsBytes:id:)` core is driven here with inline JSON blobs +
/// an injected safetensors byte total, so the whole parser is exercised off-box (no filesystem, no
/// MLX) — the live serve path only has to confirm the wired numbers match.
final class ModelConfigDecoderTests: XCTestCase {
    private func data(_ s: String) -> Data { Data(s.utf8) }

    // MARK: - uniform-GQA dense (top-level fields)

    /// A dense Qwen3-style config (top-level fields, no hybrid markers) classifies uniform-GQA with
    /// every growing layer = nLayers, and carries the summed safetensors bytes as measured weights.
    func testDenseQwen3_uniformGQA() throws {
        let json = """
        {
          "model_type": "qwen3",
          "num_hidden_layers": 64,
          "num_attention_heads": 64,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "hidden_size": 5120,
          "max_position_embeddings": 40960
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 16 * 1024 * 1024 * 1024, id: "test-qwen3-32b")
        XCTAssertEqual(parsed.profile.modelType, .uniformGQA)
        XCTAssertEqual(parsed.profile.nLayers, 64)
        XCTAssertEqual(parsed.profile.nAttnLayers, 64, "uniform-GQA: every layer grows")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 128)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 40960)
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, 16 * 1024 * 1024 * 1024)
        XCTAssertTrue(parsed.weightsAreMeasured)
        XCTAssertTrue(parsed.profile.isKVDerivable)
    }

    /// Seven vendored dense arches share the qwen3-style default `KVCacheDimensionProvider` (no
    /// `newCache` override → every layer allocates a plain growing `KVCacheSimple`) and read a
    /// SCALAR `num_key_value_heads` + `head_dim`, so `uniformGQA` is as-implemented-correct for all
    /// of them. `gemma2` is a deliberate over-count: HF's gemma2 is conceptually interleaved-SWA
    /// (sliding local layers), but the vendored `Gemma2.swift` has no `RotatingKVCache` and no
    /// `newCache` override, so as-implemented it grows a full KV cache on every layer — uniformGQA
    /// is the honest as-implemented model and fails toward RED (over-counts), never phantom-GREEN.
    func testUniformGQAReach_commonDenseFamilies() throws {
        let modelTypes = ["cohere", "gemma2", "starcoder2", "olmo2", "granite", "internlm2", "minicpm"]
        for modelType in modelTypes {
            let json = """
            {
              "model_type": "\(modelType)",
              "num_hidden_layers": 8,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "head_dim": 64,
              "max_position_embeddings": 4096
            }
            """
            let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-\(modelType)")
            XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "\(modelType) should classify uniformGQA")
            XCTAssertEqual(parsed.profile.nAttnLayers, 8, "\(modelType): every layer grows")
            XCTAssertEqual(parsed.profile.nKVHeads, 4, "\(modelType)")
            XCTAssertEqual(parsed.profile.headDim, 64, "\(modelType)")
            // 8 layers × 4 kv heads × 64 head_dim × 2 (K+V) × 2 bytes(fp16) = 8192 B/tok.
            XCTAssertEqual(
                CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
                "\(modelType): KV bytes/token cross-check")
        }
    }

    /// Set-C MoE-but-uniform-dense-attention families. Audited each vendored arch directly:
    ///  - `BailingMoe.swift`: no `newCache` override (default growing `KVCacheSimple` every layer),
    ///    scalar `num_key_value_heads`; head_dim is hard-computed as `hiddenSize / heads` (ignores any
    ///    `head_dim` field), so it agrees with the decoder ONLY when head_dim is derived — this test
    ///    omits `head_dim` and supplies `hidden_size` so every path derives `hidden/heads` identically.
    ///  - `OlmoE.swift`: no `newCache`, per-layer `kvHeads` is a uniform repeat of the scalar
    ///    `num_key_value_heads`; `resolvedHeadDimensions = head_dim ?? hidden/heads` — exact match.
    ///  - `Ernie4_5.swift`: no `newCache`, `headDim = head_dim ?? dim/num_attention_heads` — exact
    ///    match. None grows a sliding/rotating cache, so uniformGQA's every-layer formula is honest.
    func testUniformGQAReach_setC_moeUniformDense() throws {
        let modelTypes = ["bailing_moe", "olmoe", "ernie4_5"]
        for modelType in modelTypes {
            let json = """
            {
              "model_type": "\(modelType)",
              "num_hidden_layers": 8,
              "hidden_size": 512,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "max_position_embeddings": 4096
            }
            """
            let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-\(modelType)")
            XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "\(modelType) should classify uniformGQA")
            XCTAssertEqual(parsed.profile.nAttnLayers, 8, "\(modelType): every layer grows")
            XCTAssertEqual(parsed.profile.nKVHeads, 4, "\(modelType)")
            XCTAssertEqual(parsed.profile.headDim, 64, "\(modelType): head_dim derived hidden/heads = 512/8")
            // 8 layers × 4 kv heads × 64 head_dim × 2 (K+V) × 2 bytes(fp16) = 8192 B/tok.
            XCTAssertEqual(
                CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
                "\(modelType): KV bytes/token cross-check")
        }
    }

    /// Set-D: three more DENSE families whose vendored arches have NO `newCache` override (every layer
    /// grows a plain `KVCacheSimple`) and read a SCALAR `num_key_value_heads` — so the uniformGQA
    /// every-layer formula is exact as-implemented. Audited the vendored sources directly:
    ///  - gemma (Gemma.swift, the gemma1 base): explicit `head_dim` field, scalar kvHeads; same shape
    ///    as gemma2 (already uniformGQA) with no sliding window — plain growing KV every layer.
    ///  - acereason (backed by Qwen2.swift per the factory registry): a plain Qwen2 attention arch,
    ///    head_dim derived `hidden/heads`, scalar kvHeads; identical to qwen2 (already uniformGQA).
    ///  - smollm3 (SmolLM3.swift): `head_dim ?? hidden/heads`, scalar kvHeads; its `no_rope_layers` only
    ///    disable rotary on some layers (a positional detail) — the KV cache is uniform growing on every
    ///    layer, unchanged in bytes, so uniformGQA holds.
    /// Each was a fit-check GAP (→ `fit_check=skipped`, serve.sh flat limits); this pins the honest
    /// verdict. head_dim 64 is provided explicitly AND equals hidden/heads (512/8) so the assertion holds
    /// whether the decoder reads the field or derives it.
    func testUniformGQAReach_setD_moreDenseFamilies() throws {
        let modelTypes = ["gemma", "acereason", "smollm3"]
        for modelType in modelTypes {
            let json = """
            {
              "model_type": "\(modelType)",
              "num_hidden_layers": 8,
              "hidden_size": 512,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "head_dim": 64,
              "max_position_embeddings": 4096
            }
            """
            let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-\(modelType)")
            XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "\(modelType) should classify uniformGQA")
            XCTAssertEqual(parsed.profile.nAttnLayers, 8, "\(modelType): every layer grows")
            XCTAssertEqual(parsed.profile.nKVHeads, 4, "\(modelType)")
            XCTAssertEqual(parsed.profile.headDim, 64, "\(modelType)")
            // 8 layers × 4 kv heads × 64 head_dim × 2 (K+V) × 2 bytes(fp16) = 8192 B/tok.
            XCTAssertEqual(
                CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
                "\(modelType): KV bytes/token cross-check")
        }
    }

    /// Set-E: three more DENSE families audited to have NO `newCache` override (plain growing KV every
    /// layer) and a scalar `num_key_value_heads` → uniformGQA. bitnet's ternary weights don't touch KV
    /// geometry; nanochat/lille-130m derive head_dim = hidden/heads. Each was a fit-check GAP.
    func testUniformGQAReach_setE_moreDenseFamilies() throws {
        let modelTypes = ["bitnet", "nanochat", "lille-130m"]
        for modelType in modelTypes {
            let json = """
            {
              "model_type": "\(modelType)",
              "num_hidden_layers": 8,
              "hidden_size": 512,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "head_dim": 64,
              "max_position_embeddings": 4096
            }
            """
            let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-\(modelType)")
            XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "\(modelType) should classify uniformGQA")
            XCTAssertEqual(parsed.profile.nAttnLayers, 8, "\(modelType): every layer grows")
            XCTAssertEqual(parsed.profile.nKVHeads, 4, "\(modelType)")
            XCTAssertEqual(parsed.profile.headDim, 64, "\(modelType)")
            XCTAssertEqual(
                CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
                "\(modelType): KV bytes/token cross-check")
        }
    }

    /// Set-F: two more DENSE families audited to have NO `newCache` override (plain growing
    /// `KVCacheSimple` on every layer) and a SCALAR `num_key_value_heads`, with head_dim COMPUTED
    /// as hidden_size / num_attention_heads (neither vendored arch reads an explicit head_dim field):
    ///  - mimo (MiMo.swift): `headDim = hiddenSize / heads` (:33), scalar `kvHeads` (:31, uniform
    ///    per-layer repeat :170). Its MTP draft layers (`num_nextn_predict_layers`) are NOT part of
    ///    the main transformer cache enumeration (the kvHeads array spans `0..<num_hidden_layers`
    ///    only), so the every-layer uniformGQA formula over num_hidden_layers is exact.
    ///  - apertus (Apertus.swift): `headDim = hiddenSize / heads` (:177), scalar `numKeyValueHeads`
    ///    (:176, uniform per-layer repeat :343). Its QK-norm (RMSNorm on headDim) and xIELU
    ///    activation don't change stored K/V bytes. Each was a fit-check GAP (skipped → flat limits).
    func testUniformGQAReach_setF_computedHeadDimFamilies() throws {
        let modelTypes = ["mimo", "apertus"]
        for modelType in modelTypes {
            let json = """
            {
              "model_type": "\(modelType)",
              "num_hidden_layers": 8,
              "hidden_size": 512,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "max_position_embeddings": 4096
            }
            """
            let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-\(modelType)")
            XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "\(modelType) should classify uniformGQA")
            XCTAssertEqual(parsed.profile.nAttnLayers, 8, "\(modelType): every layer grows")
            XCTAssertEqual(parsed.profile.nKVHeads, 4, "\(modelType)")
            XCTAssertEqual(parsed.profile.headDim, 64, "\(modelType): head_dim = hidden/heads = 512/8")
            XCTAssertEqual(
                CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
                "\(modelType): KV bytes/token cross-check")
        }
    }

    /// OlmoE's vendored arch falls back to MHA (`kvHeads = num_key_value_heads ?? num_attention_heads`,
    /// OlmoE.swift:293-294) when `num_key_value_heads` is absent. The decoder does NOT silently mirror
    /// that fallback — it fails closed (`missingField`) rather than guess an MHA head count and risk a
    /// phantom-GREEN under/over-count. Honest refusal over a silent divergence.
    func testOlmoE_missingKVHeads_failsClosed() {
        let json = """
        {
          "model_type": "olmoe",
          "num_hidden_layers": 8,
          "hidden_size": 512,
          "num_attention_heads": 8,
          "max_position_embeddings": 4096
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "olmoe")) { error in
            guard case ModelConfigDecodeError.missingField = error else {
                return XCTFail("expected missingField for absent num_key_value_heads, got \(error)")
            }
        }
    }

    /// head_dim absent → derived from hidden_size / num_attention_heads.
    func testHeadDimFallbackFromHiddenSize() throws {
        let json = """
        {
          "model_type": "llama",
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "hidden_size": 4096,
          "max_position_embeddings": 131072
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-llama")
        XCTAssertEqual(parsed.profile.headDim, 128, "4096 / 32 = 128")
    }

    // MARK: - quantization bits (drives quant auto-pick, differentiator #2 full shape)

    /// An MLX 4-bit checkpoint declares `"quantization": {"bits": 4, ...}`; the decoder surfaces it
    /// so the auto-picker can rank candidates by the checkpoint's own declared quant, not a repo-name
    /// guess.
    func testQuantizationBitsParsed_topLevel() throws {
        let json = """
        {
          "model_type": "qwen3",
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "max_position_embeddings": 32768,
          "quantization": { "group_size": 64, "bits": 4 }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "q4")
        XCTAssertEqual(parsed.quantBits, 4)
    }

    /// mlx-lm also emits the bits under `quantization_config`; either key is honored.
    func testQuantizationBitsParsed_quantizationConfigKey() throws {
        let json = """
        {
          "model_type": "llama",
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "max_position_embeddings": 8192,
          "quantization_config": { "bits": 8 }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "q8")
        XCTAssertEqual(parsed.quantBits, 8)
    }

    /// An unquantized checkpoint (no quantization block) reports `nil` bits — the picker treats nil as
    /// highest-quality (unquantized) for its ranking tiebreak.
    func testNoQuantizationBlock_nilBits() throws {
        let json = """
        {
          "model_type": "mistral",
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "max_position_embeddings": 8192
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "base")
        XCTAssertNil(parsed.quantBits)
    }

    /// MLX mixed-precision recipes (e.g. DWQ / mixed-3-6bit) write a top-level default `bits`/
    /// `group_size` PLUS per-module override objects keyed by weight path. The decoder ranks by the
    /// top-level *declared default* bits and must ignore the nested override dicts — reading `bits`
    /// as the module-path values (`[String: Any]`) instead would crash or mis-rank. Selection now
    /// depends on this parse (QuantAutoPicker), so lock it: dominant bits == the top-level default,
    /// nested overrides never derail it.
    func testQuantizationBits_mixedPrecision_readsTopLevelDefault() throws {
        let json = """
        {
          "model_type": "qwen3",
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "max_position_embeddings": 32768,
          "quantization": {
            "group_size": 64, "bits": 3,
            "model.embed_tokens": { "group_size": 64, "bits": 6 },
            "model.layers.0.self_attn.q_proj": { "group_size": 64, "bits": 6 }
          }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "mixed-3-6")
        XCTAssertEqual(parsed.quantBits, 3, "ranks by the top-level default bits; per-module overrides are ignored, never crash the parse")
    }

    // MARK: - hybrid-linear qwen3_5 (nested text_config), mirrors the live checkpoint

    /// The real Qwen3.5-9B config: fields nest under `text_config`, layer_types marks 8 full_attention
    /// layers out of 32. Must match the hand-audited catalog entry (nAttnLayers = 8, headDim 256).
    func testQwen35Hybrid_nestedTextConfig_matchesCatalog() throws {
        // 32 layers: pattern of 3×linear then 1×full, repeated 8× → 8 full_attention layers.
        var layerTypes: [String] = []
        for _ in 0..<8 { layerTypes += ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5",
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": \(layerTypesJSON),
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
          }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 5_600_000_000, id: "Qwen3.5-9B-live")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nLayers, 32)
        XCTAssertEqual(parsed.profile.nAttnLayers, 8, "only the 8 full_attention layers grow KV")
        XCTAssertEqual(parsed.profile.nKVHeads, 4)
        XCTAssertEqual(parsed.profile.headDim, 256)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 262144)
        XCTAssertTrue(parsed.profile.isKVDerivable)

        // Cross-check against the hand-audited catalog entry: decoded KV@32K must match Qwen3.5-9B.
        // The catalog now carries the reconciled fixedStateBytes (51,511,296), so the fixture carries
        // the linear_* geometry too and both sides equal growing 1.0 GiB + fixed 49.13 MiB.
        guard let catalog = ModelArchProfile.catalog.first(where: { $0.id == "Qwen3.5-9B" }) else {
            return XCTFail("missing catalog entry Qwen3.5-9B")
        }
        let decodedKV = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        let catalogKV = CapacityModel.kvBytesForContext(catalog, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(decodedKV, catalogKV, "decoded qwen3_5 KV@32K must equal the hand catalog entry")
        XCTAssertEqual(decodedKV, 1_073_741_824 + 51_511_296, "growing 1.0 GiB @32K + fixed 49.13 MiB recurrent/conv state")
    }

    /// layer_types absent → nAttnLayers derived from nLayers / full_attention_interval.
    func testHybrid_nAttnLayersFromInterval_whenLayerTypesAbsent() throws {
        let json = """
        {
          "model_type": "qwen3_next",
          "num_hidden_layers": 40,
          "num_attention_heads": 16,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-hybrid")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nAttnLayers, 10, "40 / interval 4 = 10 full-attention layers")
    }

    /// qwen3_5_text (the text-only config variant of the Qwen3.5 family, e.g. a text-tower checkpoint
    /// whose root `model_type` is `qwen3_5_text` rather than the VL wrapper's `qwen3_5`). The vendored
    /// `Qwen35.swift newCache` (:582-589) is identical for all three siblings — `MambaCache()` on the
    /// gated-delta-net (linear) layers, `KVCacheSimple()` on the `full_attention_interval` attention
    /// layers — so it is `.hybridLinear` on the SAME default GatedDeltaNet path as `qwen3_5`/
    /// `qwen3_5_moe` (no family flag; the state resolver keys on the `linear_*` config fields). It was a
    /// fit-check GAP (→ `fit_check=skipped`); this pins the honest hybrid-linear verdict, flat root
    /// shape (no `text_config` nesting), 8 of 32 layers full-attention, plus the recurrent/conv state.
    func testQwen35TextHybrid_flatRootConfig_reusesHybridLinearPath() throws {
        var layerTypes: [String] = []
        for _ in 0..<8 { layerTypes += ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "qwen3_5_text",
          "num_hidden_layers": 32,
          "num_attention_heads": 16,
          "num_key_value_heads": 4,
          "head_dim": 256,
          "hidden_size": 4096,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4,
          "layer_types": \(layerTypesJSON),
          "linear_num_key_heads": 16,
          "linear_num_value_heads": 32,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 5_600_000_000, id: "Qwen3.5-text")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear, "qwen3_5_text reuses the qwen3_5 hybrid-linear class")
        XCTAssertEqual(parsed.profile.nLayers, 32)
        XCTAssertEqual(parsed.profile.nAttnLayers, 8, "only the 8 full_attention layers grow KV")
        XCTAssertEqual(parsed.profile.nKVHeads, 4)
        XCTAssertEqual(parsed.profile.headDim, 256)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 262144)
        // Growing KV @32K: 8 attn layers × 4 kv × 256 × 2(K+V) × 2 B × 32768 = 1.0 GiB, matching the
        // qwen3_5 sibling. The GatedDeltaNet recurrent/conv state (24 linear layers) is ADDED on top.
        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 32768,
            "8 × 4 × 256 × 2 × 2 = 32,768 B/token growing")
        XCTAssertGreaterThan(parsed.profile.fixedStateBytes, 0,
            "GatedDeltaNet conv+SSM recurrent state present on the 24 linear layers")
    }

    /// The real Qwen3.5-9B config carries the linear-attention geometry (`linear_num_*_heads`,
    /// `linear_*_head_dim`, `linear_conv_kernel_dim`); the decoder derives the fixed per-sequence
    /// recurrent+conv state from it, reconciled against the vendored MambaCache allocation
    /// (SSM `[Hv,Dv,Dk]` fp32 + conv `[kernel-1, 2·keyDim+valueDim]` bf16). 24 linear layers →
    /// 51,511,296 B ≈ 49.13 MiB/seq. This is ADDED to the growing KV, so the fit-check is strictly
    /// more conservative than the growing-only reconciliation.
    func testQwen35Hybrid_fixedStateBytesFromLinearConfig() throws {
        var layerTypes: [String] = []
        for _ in 0..<8 { layerTypes += ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5",
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": \(layerTypesJSON),
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
          }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 5_600_000_000, id: "Qwen3.5-9B-live")
        // 24 linear layers × (conv 49,152 B [ (4-1)×(2·2048+4096)×2 ] + ssm 2,097,152 B [ 32×128×128×4 ])
        // = 24 × 2,146,304 = 51,511,296 B.
        XCTAssertEqual(parsed.profile.fixedStateBytes, 51_511_296)
        // The growing term is unchanged (8 full-attn layers × 4 kv × 256 × 2 × 2 = 32 KiB/tok → 1.0 GiB
        // @ 32K); kvBytesForContext now adds the fixed state on top.
        let kv = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(kv, 1_073_741_824 + 51_511_296, "growing 1.0 GiB @32K + fixed 49.13 MiB recurrent/conv state")
    }

    /// Fail-OPEN: a hybrid-linear config WITHOUT the `linear_*` geometry fields keeps `fixedStateBytes`
    /// at 0 (the pre-existing behavior) rather than refusing to size the model — the growing-KV +
    /// weights terms that dominate are still counted. Documents why the catalog-matching test above
    /// (whose fixture omits the linear fields) stays green.
    func testHybrid_missingLinearFields_fixedStateZero() throws {
        let json = """
        {
          "model_type": "qwen3_next",
          "num_hidden_layers": 40,
          "num_attention_heads": 16,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "test-hybrid")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.fixedStateBytes, 0, "no linear_* fields → fail-open to 0, not a refusal")
    }

    /// LFM2 (Liquid `lfm2`, e.g. `LiquidAI/LFM2-1.2B`): hybrid-linear like qwen3_next, but with two
    /// family-specific config shapes. (1) Its attention layers are listed as explicit INDICES in
    /// `full_attn_idxs` rather than a `layer_types` array or `full_attention_interval`; the vendored
    /// `LFM2.swift` `newCache` (LFM2.swift:401-408) grows a `KVCacheSimple` on exactly those layers and
    /// a `MambaCache` (conv-only) on the rest. (2) Its recurrent state is a simple depthwise conv cache,
    /// NOT GatedDeltaNet SSM state — `LFM2.swift:220` allocates `zeros([B, conv_L_cache − 1, hidden])`
    /// at activation precision and uses only `cache[0]` (no ssm_state slot), so the qwen3_5 SSM+conv
    /// derivation does not apply. Geometry is the REAL published config of `LiquidAI/LFM2-1.2B`
    /// (huggingface.co/LiquidAI/LFM2-1.2B/resolve/main/config.json, fetched 2026-08-19): 16 layers,
    /// full_attn_idxs [2,5,8,10,12,14] (6 attention, 10 conv), hidden_size 2048, 32 attention heads
    /// (head_dim derived 2048/32 = 64), num_key_value_heads 8, conv_L_cache 3, max_position_embeddings
    /// 128000, no head_dim field.
    func testLFM2_realConfig_hybridLinearConvState() throws {
        let json = """
        {
          "model_type": "lfm2",
          "num_hidden_layers": 16,
          "hidden_size": 2048,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "conv_L_cache": 3,
          "full_attn_idxs": [2, 5, 8, 10, 12, 14],
          "max_position_embeddings": 128000
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "LFM2-1.2B")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nLayers, 16)
        XCTAssertEqual(parsed.profile.nAttnLayers, 6, "6 attention-layer indices in full_attn_idxs grow a KVCacheSimple")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 64, "no head_dim field → derived hidden_size/num_attention_heads = 2048/32")
        XCTAssertEqual(parsed.profile.nativeMaxContext, 128000)
        // 10 conv layers × (conv_L_cache−1 = 2) × hidden_size 2048 × 2 bytes (activation precision) = 81,920 B.
        XCTAssertEqual(parsed.profile.fixedStateBytes, 81_920, "conv-only recurrent state; no GatedDeltaNet SSM term")

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 12288,
            "6 attention layers × 8 kv heads × 64 head_dim × 2 (K+V) × 2 bytes (fp16) = 12 KiB/tok")
        let kv = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(kv, 402_653_184 + 81_920, "growing 12 KiB/tok @32K + 80 KiB fixed conv state")
    }

    /// LFM2MoE (Liquid `lfm2_moe`, e.g. `LiquidAI/LFM2-8B-A1B`): the MoE sibling of lfm2 — the vendored
    /// `LFM2MoE.swift` `newCache` (LFM2MoE.swift:495-503) is byte-for-byte the same cache shape as lfm2,
    /// growing a `KVCacheSimple` on the attention layers and a conv-only `MambaCache`
    /// (`zeros([B, conv_L_cache − 1, hidden])`, LFM2MoE.swift:228) on the rest. The only config-shape
    /// difference from lfm2 is that attention layers are enumerated in `layer_types` (`"full_attention"`
    /// entries — LFM2MoE derives `fullAttnIdxs` from them at LFM2MoE.swift:44-46) rather than an explicit
    /// `full_attn_idxs` index list; `resolveHybridAttnLayers` already reads `layer_types` first, so the
    /// same `.hybridLinear` + `resolveLfm2ConvStateBytes` path applies (the MoE router — `num_experts`,
    /// `moe_intermediate_size` — is orthogonal to KV/conv geometry). Geometry is the REAL published config
    /// of `LiquidAI/LFM2-8B-A1B` (huggingface.co/LiquidAI/LFM2-8B-A1B/raw/main/config.json, fetched
    /// 2026-08-19): 24 layers, layer_types with 6 `full_attention` (indices 2,6,10,14,18,21) + 18 `conv`,
    /// hidden_size 2048, 32 attention heads (head_dim derived 2048/32 = 64), num_key_value_heads 8,
    /// conv_L_cache 3, max_position_embeddings 128000, no head_dim field.
    func testLFM2MoE_realConfig_hybridLinearConvState() throws {
        let json = """
        {
          "model_type": "lfm2_moe",
          "num_hidden_layers": 24,
          "hidden_size": 2048,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "conv_L_cache": 3,
          "num_experts": 32,
          "num_experts_per_tok": 4,
          "moe_intermediate_size": 1792,
          "layer_types": [
            "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv",
            "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv",
            "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv"
          ],
          "max_position_embeddings": 128000
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "LFM2-8B-A1B")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nLayers, 24)
        XCTAssertEqual(parsed.profile.nAttnLayers, 6, "6 `full_attention` entries in layer_types grow a KVCacheSimple")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 64, "no head_dim field → derived hidden_size/num_attention_heads = 2048/32")
        XCTAssertEqual(parsed.profile.nativeMaxContext, 128000)
        // 18 conv layers × (conv_L_cache−1 = 2) × hidden_size 2048 × 2 bytes (activation precision) = 147,456 B.
        XCTAssertEqual(parsed.profile.fixedStateBytes, 147_456, "conv-only recurrent state (same shape as lfm2); no GatedDeltaNet SSM term")

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 12288,
            "6 attention layers × 8 kv heads × 64 head_dim × 2 (K+V) × 2 bytes (fp16) = 12 KiB/tok")
        let kv = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(kv, 402_653_184 + 147_456, "growing 12 KiB/tok @32K + 144 KiB fixed conv state")
    }

    /// jamba: the vendored `Jamba.swift` `newCache` (Jamba.swift:479-487) grows a plain `KVCacheSimple`
    /// on `layer.isAttn` layers and a 2-slot `MambaCache` on the rest — a hybrid-linear cache shape like
    /// lfm2/qwen3_5, but with a DIFFERENT recurrent term than either: a CLASSIC Mamba SSM state (NOT
    /// lfm2's conv-only cache, NOT qwen3_5's GatedDeltaNet). Two family-specific config shapes handled
    /// below: (1) no `layer_types`/`full_attn_idxs`/`full_attention_interval` and (for Jamba-v0.1) no
    /// `layers_block_type` — attention layers are `i % attn_layer_period == attn_layer_offset`; and
    /// (2) the Mamba inner dim is `mamba_expand × hidden_size` (`JambaMambaMixer` at Jamba.swift:213,
    /// which SHADOWS the config's MoE `intermediate_size`), NOT the config `intermediate_size` field.
    /// MambaCache holds `[0]` conv state `[B, mamba_d_conv-1, d_inner]` (activation precision, 2 B) and
    /// `[1]` SSM state `[B, d_inner, mamba_d_state]` (fp32, 4 B) — see `resolveJambaStateBytes`. Geometry
    /// is the REAL published config of `ai21labs/Jamba-v0.1`
    /// (huggingface.co/ai21labs/Jamba-v0.1/resolve/main/config.json).
    func testJamba_realJambaV01Config_hybridLinearMambaState() throws {
        // Real Jamba-v0.1 config (layers_block_type ABSENT → attention layers derived from
        // attn_layer_offset/period). MoE fields (num_experts etc.) do not affect KV/state geometry.
        let json = """
        {
          "model_type": "jamba",
          "hidden_size": 4096,
          "intermediate_size": 14336,
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "attn_layer_offset": 4,
          "attn_layer_period": 8,
          "expert_layer_offset": 1,
          "expert_layer_period": 2,
          "mamba_d_conv": 4,
          "mamba_d_state": 16,
          "mamba_expand": 2,
          "num_experts": 16,
          "num_experts_per_tok": 2,
          "max_position_embeddings": 262144
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "Jamba-v0.1")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nAttnLayers, 4, "i % 8 == 4 over 0..31 → layers {4,12,20,28}")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 128, "no head_dim field → hidden_size 4096 / num_attention_heads 32")
        XCTAssertEqual(parsed.profile.nativeMaxContext, 262144)

        // 28 mamba layers × (conv (4-1)×8192×2 = 49,152 B + ssm 8192×16×4 = 524,288 B) = 28 × 573,440
        // = 16,056,320 B. d_inner = mamba_expand 2 × hidden_size 4096 = 8192.
        XCTAssertEqual(parsed.profile.fixedStateBytes, 16_056_320,
            "classic Mamba conv+SSM state, NOT lfm2 conv-only and NOT qwen3_5 GatedDeltaNet")

        // Growing KV: 4 attention layers × 8 kv heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16).
        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 16384,
            "4 attention layers × 8 kv × 128 head_dim × 2 (K+V) × 2 bytes (fp16) = 16 KiB/tok")
        let kv = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(kv, 536_870_912 + 16_056_320, "growing 16 KiB/tok @32K (512 MiB) + fixed Mamba state")
    }

    /// LFM2 fails closed the same way every hybrid-linear config does when NO attention-layer signal is
    /// present (no `full_attn_idxs`, `layer_types`, or `full_attention_interval`) — the attention-layer
    /// count can't be derived, so refuse rather than guess.
    func testLFM2_missingAttnIdxs_failsClosed() {
        let json = """
        {
          "model_type": "lfm2",
          "num_hidden_layers": 16,
          "hidden_size": 2048,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "conv_L_cache": 3,
          "max_position_embeddings": 128000
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "LFM2-nofield")) { error in
            guard case ModelConfigDecodeError.missingField = error else {
                return XCTFail("expected .missingField for the un-derivable attention-layer count, got \(error)")
            }
        }
    }

    // MARK: - interleaved-SWA gemma3 (nested text_config, multimodal checkpoint)

    /// The real Gemma-3-27B-it multimodal config: geometry nests under `text_config`, global-attention
    /// layer count is derived from `sliding_window_pattern` (no `layer_types` on disk). Cross-checks
    /// KV bytes against `CapacityModel` to prove the 16×128 head geometry reconciles the catalog's
    /// hand-audited Gemma-3-27B numbers.
    func testGemma3Multimodal_interleavedSWA_matchesCapacityModel() throws {
        let json = """
        {
          "model_type": "gemma3",
          "text_config": {
            "model_type": "gemma3_text",
            "num_hidden_layers": 62,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 131072
          }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gemma-3-27b-it")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nLayers, 62)
        XCTAssertEqual(parsed.profile.nAttnLayers, 10, "floor(62/6) = 10 global layers")
        XCTAssertEqual(parsed.profile.nLocalLayers, 52)
        XCTAssertEqual(parsed.profile.slidingWindow, 1024)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 131072)

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 81920,
            "10 global layers × 16 kv heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16) = 80 KiB/tok")
        XCTAssertEqual(
            CapacityModel.swaFixedLocalBytes(parsed.profile, kvQuant: .fp16), 436_207_616,
            "52 local layers capped at the 1024-token sliding window ≈ 0.406 GiB fixed")
    }

    /// The text-only Gemma-3 variant (`gemma3_text`) keeps fields flat at the root, no `text_config`
    /// nesting — the decoder's text_config-first/root-fallback `geom()` helper must still resolve them.
    func testGemma3Text_flatConfig_interleavedSWA() throws {
        let json = """
        {
          "model_type": "gemma3_text",
          "num_hidden_layers": 26,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "sliding_window": 512,
          "sliding_window_pattern": 6,
          "max_position_embeddings": 32768
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gemma-3-text")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 4, "floor(26/6) = 4 global layers")
        XCTAssertEqual(parsed.profile.nLocalLayers, 22)
        XCTAssertEqual(parsed.profile.slidingWindow, 512)
    }

    /// `layer_types`, when present, is authoritative over `sliding_window_pattern` — mirrors the
    /// hybrid-linear decoder's override precedence.
    func testGemma3_layerTypesOverridesPattern() throws {
        // 12 layers, 3 marked full_attention, no sliding_window_pattern present at all.
        var layerTypes: [String] = []
        for i in 0..<12 { layerTypes.append(i % 4 == 3 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "gemma3_text",
          "num_hidden_layers": 12,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "sliding_window": 512,
          "max_position_embeddings": 32768,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gemma-3-layertypes")
        XCTAssertEqual(parsed.profile.nAttnLayers, 3, "layer_types counts exactly 3 full_attention entries")
        XCTAssertEqual(parsed.profile.nLocalLayers, 9)
    }

    /// gpt-oss: the vendored `GPTOSS.swift` (lines 515-524) allocates `StandardKVCache` (grows with
    /// context) for `full_attention` layers and `RotatingKVCache(maxSize: slidingWindow)` (capped at
    /// the window) for the rest — the same interleaved-SWA cache shape as Gemma-3, so `gpt_oss` fits
    /// the existing `interleavedSWATypes` allow-list rather than needing a new `ArchClass`. Flat
    /// config (no `text_config`), mirroring the gpt-oss-20b shape: 24 layers alternating
    /// sliding/full_attention, 12 full_attention entries.
    func testGptOss_flatConfig_interleavedSWA() throws {
        var layerTypes: [String] = []
        for i in 0..<24 { layerTypes.append(i % 2 == 0 ? "sliding_attention" : "full_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "gpt_oss",
          "num_hidden_layers": 24,
          "num_attention_heads": 64,
          "num_key_value_heads": 8,
          "head_dim": 64,
          "sliding_window": 128,
          "max_position_embeddings": 131072,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gpt-oss-20b")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 12, "12 full_attention entries")
        XCTAssertEqual(parsed.profile.slidingWindow, 128)
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 64)

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 24576,
            "12 global layers × 8 kv heads × 64 head_dim × 2 (K+V) × 2 bytes (fp16) = 24 KiB/tok")
        XCTAssertEqual(
            CapacityModel.swaFixedLocalBytes(parsed.profile, kvQuant: .fp16), 3_145_728,
            "12 local layers capped at the 128-token sliding window = 3 MiB fixed")
    }

    /// olmo3: the vendored `Olmo3.swift` `newCache` (Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/
    /// Olmo3.swift:226-234) allocates `KVCacheSimple()` (grows with context) for `full_attention`
    /// layers and `RotatingKVCache(maxSize: slidingWindow)` (capped at the window) for the rest —
    /// the same interleaved-SWA cache shape as Gemma-3/gpt-oss, driven by the same `layer_types`/
    /// `sliding_window` config fields and a scalar `num_key_value_heads`/`head_dim`, so `olmo3` fits
    /// the existing `interleavedSWATypes` allow-list rather than needing a new `ArchClass`.
    /// Representative olmo3-shape: 32 layers alternating on the vendored default `(i+1) % 4 == 0`
    /// full-attention cadence → 8 full_attention entries, 24 sliding-local.
    func testOlmo3_flatConfig_interleavedSWA() throws {
        var layerTypes: [String] = []
        for i in 0..<32 { layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "olmo3",
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "sliding_window": 4096,
          "max_position_embeddings": 65536,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "olmo-3-7b")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 8, "8 full_attention entries")
        XCTAssertEqual(parsed.profile.slidingWindow, 4096)
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 128)

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 32768,
            "8 global layers × 8 kv heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16) = 32 KiB/tok")
        XCTAssertEqual(
            CapacityModel.swaFixedLocalBytes(parsed.profile, kvQuant: .fp16), 402_653_184,
            "24 local layers × 8 kv × 128 head_dim × 2 (K+V) × 2 bytes × 4096-token window = 384 MiB fixed")
    }

    /// afmoe: the vendored `AfMoE.swift` `newCache` (AfMoE.swift:560-569) allocates
    /// `RotatingKVCache(maxSize: slidingWindow)` (capped at the window) for `sliding_attention` layers
    /// and `KVCacheSimple()` (grows with context) for the rest — the same interleaved-SWA cache shape
    /// as Gemma-3/gpt-oss/olmo3, driven by the same `layer_types`/`sliding_window` fields
    /// (`layerUsesSliding = layerTypes.map { $0 == "sliding_attention" }`, AfMoE.swift:510) and a scalar
    /// `num_key_value_heads`/`head_dim`. It caches ONLY standard K/V (no altup/laurel/mamba/conv state),
    /// so `afmoe` fits the existing `interleavedSWATypes` allow-list — a pure allow-list add, no new
    /// `ArchClass`. Geometry is the REAL published config of `arcee-ai/Trinity-Nano-Base`
    /// (huggingface.co/arcee-ai/Trinity-Nano-Base/resolve/main/config.json): 56 layers on the
    /// `global_attn_every_n_layers: 4` cadence (`(i+1) % 4 == 0` → 14 `full_attention`, 42 sliding),
    /// num_key_value_heads 2, head_dim 128, explicit sliding_window 2048, max_position_embeddings 131072.
    /// The MoE router fields (num_experts etc.) do not affect KV geometry and are omitted.
    func testAfmoe_realTrinityNanoConfig_interleavedSWA() throws {
        var layerTypes: [String] = []
        for i in 0..<56 { layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "afmoe",
          "num_hidden_layers": 56,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 128,
          "sliding_window": 2048,
          "global_attn_every_n_layers": 4,
          "max_position_embeddings": 131072,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "trinity-nano-base")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 14, "14 full_attention entries (every 4th of 56)")
        XCTAssertEqual(parsed.profile.slidingWindow, 2048)
        XCTAssertEqual(parsed.profile.nKVHeads, 2)
        XCTAssertEqual(parsed.profile.headDim, 128)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 131072)

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 14336,
            "14 global layers × 2 kv heads × 128 head_dim × 2 (K+V) × 2 bytes (fp16) = 14 KiB/tok")
        XCTAssertEqual(
            CapacityModel.swaFixedLocalBytes(parsed.profile, kvQuant: .fp16), 88_080_384,
            "42 local layers × 2 kv × 128 head_dim × 2 (K+V) × 2 bytes × 2048-token window = 84 MiB fixed")
    }

    /// gemma3n (Gemma-3n E2B/E4B, e.g. `mlx-community/gemma-3n-E2B-it-lm-4bit`): interleaved-SWA like
    /// Gemma-3, BUT the vendored `Gemma3nText.swift` `newCache` (Gemma3nText.swift:677-693) allocates
    /// caches for ONLY the first `num_hidden_layers − num_kv_shared_layers` layers — the shared tail
    /// reuses earlier layers' KV and allocates NOTHING. So the honest KV model is the interleaved-SWA
    /// formula over the CACHED PREFIX only: a plain allow-list add would over-count both the growing
    /// (`full_attention`) and the window-capped (`sliding_attention`) layers by including the shared
    /// tail. This is a config-SHAPE branch (like `mistral3ArchClass`), not an allow-list add: it reads
    /// `num_kv_shared_layers`, truncates `layer_types`/`nLayers` to the cached prefix, and counts
    /// `full_attention` within it. `newCache` allocates ONLY `StandardKVCache`/`RotatingKVCache` (no
    /// altup/laurel/per-layer-input state), so `fixedStateBytes` stays 0. Geometry is the REAL published
    /// config of `mlx-community/gemma-3n-E2B-it-lm-4bit`
    /// (huggingface.co/mlx-community/gemma-3n-E2B-it-lm-4bit/resolve/main/config.json, fetched
    /// 2026-08-19): text_config with num_hidden_layers 30, num_kv_shared_layers 10 (→ 20 cached),
    /// layer_types full_attention at every 5th index (`(i+1) % 5 == 0` → 6 of 30, 4 of the first 20),
    /// num_key_value_heads 2, head_dim 256, sliding_window 512, max_position_embeddings 32768.
    func testGemma3n_realE2BConfig_cachedPrefixInterleavedSWA() throws {
        var layerTypes: [String] = []
        for i in 0..<30 { layerTypes.append((i + 1) % 5 == 0 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "gemma3n",
          "text_config": {
            "model_type": "gemma3n_text",
            "num_hidden_layers": 30,
            "num_kv_shared_layers": 10,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "sliding_window": 512,
            "max_position_embeddings": 32768,
            "layer_types": \(layerTypesJSON)
          }
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gemma-3n-E2B")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nLayers, 20, "cached prefix = 30 − 10 shared; the 10-layer shared tail allocates no cache")
        XCTAssertEqual(parsed.profile.nAttnLayers, 4, "4 full_attention entries within the first 20 (cached) layers")
        XCTAssertEqual(parsed.profile.nLocalLayers, 16, "16 sliding_attention entries within the cached prefix")
        XCTAssertEqual(parsed.profile.slidingWindow, 512)
        XCTAssertEqual(parsed.profile.nKVHeads, 2)
        XCTAssertEqual(parsed.profile.headDim, 256)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 32768)
        XCTAssertEqual(parsed.profile.fixedStateBytes, 0, "newCache allocates only K/V caches — no recurrent/conv/altup state")

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
            "4 global layers × 2 kv heads × 256 head_dim × 2 (K+V) × 2 bytes (fp16) = 8 KiB/tok")
        XCTAssertEqual(
            CapacityModel.swaFixedLocalBytes(parsed.profile, kvQuant: .fp16), 16_777_216,
            "16 local layers × 2 kv × 256 head_dim × 2 (K+V) × 2 bytes × 512-token window = 16 MiB fixed")
    }

    /// gemma3n fails closed when `num_kv_shared_layers` is absent: the cached-prefix count can't be
    /// honestly derived, and silently assuming 0 shared layers would count the full 30-layer stack
    /// (over-counting KV). Never default the shared count (spec §5) — refuse instead.
    func testGemma3n_missingKvSharedLayers_failsClosed() {
        var layerTypes: [String] = []
        for i in 0..<30 { layerTypes.append((i + 1) % 5 == 0 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "gemma3n_text",
          "num_hidden_layers": 30,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 256,
          "sliding_window": 512,
          "max_position_embeddings": 32768,
          "layer_types": \(layerTypesJSON)
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "gemma-3n-nofield")) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected .missingField, got \(error)")
            }
            XCTAssertTrue(f.contains("num_kv_shared_layers"), "field name should point at the missing shared-layer count, got '\(f)'")
        }
    }

    /// Missing BOTH `sliding_window_pattern` and `layer_types` — no way to honestly derive the
    /// global-layer count — must fail closed, never default to transformers' pattern-6 assumption.
    func testGemma3_missingPatternAndLayerTypes_failsClosed() {
        let json = """
        {
          "model_type": "gemma3_text",
          "num_hidden_layers": 26,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "sliding_window": 512,
          "max_position_embeddings": 32768
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.missingField = error else {
                return XCTFail("expected missingField, got \(error)")
            }
        }
    }

    /// `sliding_window_pattern` present but `sliding_window` itself absent — the window size is
    /// required to derive the local-layer cap; must fail closed.
    func testGemma3_missingSlidingWindow_failsClosed() {
        let json = """
        {
          "model_type": "gemma3_text",
          "num_hidden_layers": 26,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "sliding_window_pattern": 6,
          "max_position_embeddings": 32768
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField, got \(error)")
            }
            XCTAssertEqual(f, "sliding_window")
        }
    }

    /// gemma2 is now classified `uniformGQA` (see `testUniformGQAReach_commonDenseFamilies`): the
    /// vendored `Gemma2.swift` has no `RotatingKVCache`/`newCache` override, so as-implemented it
    /// grows a full KV cache on every layer despite HF's conceptual interleaved-SWA design.
    func testGemma2_classifiesUniformGQA_asImplementedOverCount() throws {
        let json = """
        { "model_type": "gemma2", "num_hidden_layers": 26, "num_attention_heads": 8,
          "num_key_value_heads": 4, "head_dim": 128, "sliding_window": 4096,
          "max_position_embeddings": 8192 }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")
        XCTAssertEqual(parsed.profile.modelType, .uniformGQA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 26, "as-implemented: every layer grows (over-counts vs. HF's sliding design)")
    }

    // MARK: - MLA-as-implemented deepseek_v3 (decompressed per-head cache; the spec-§8 OOM case)

    /// A real DeepSeek-V3/R1-geometry config classifies `.mlaAsImplemented` and wires the four
    /// decompressed MLA dims so `CapacityModel` produces the catalog-reconciled KV rate. This turns
    /// the single most OOM-dangerous arch (R1: 152.5 GiB KV @32K) from `fit_check=skipped` into an
    /// honest RED/ceiling. Wires the decoder to the EXISTING `.mlaAsImplemented` formula — it builds
    /// no MLA cache (the compact "absorbed" cache stays spec-§7 backlog).
    func testDeepseekV3_mlaAsImplemented_matchesCatalogKVParity() throws {
        let json = """
        {
          "model_type": "deepseek_v3",
          "num_hidden_layers": 61,
          "num_attention_heads": 128,
          "qk_rope_head_dim": 64,
          "qk_nope_head_dim": 128,
          "v_head_dim": 128,
          "num_key_value_heads": 128,
          "max_position_embeddings": 163840
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "deepseek-r1")
        XCTAssertEqual(parsed.profile.modelType, .mlaAsImplemented)
        XCTAssertEqual(parsed.profile.nLayers, 61)
        XCTAssertEqual(parsed.profile.nAttnLayers, 61, "MLA: every layer caches decompressed per-head K/V")
        XCTAssertEqual(parsed.profile.nKVHeads, 0, "MLA repurposes num_key_value_heads; the growing rate is the decompressed dims")
        XCTAssertEqual(parsed.profile.headDim, 0)
        XCTAssertEqual(parsed.profile.mlaHeads, 128)
        XCTAssertEqual(parsed.profile.mlaRopeDim, 64)
        XCTAssertEqual(parsed.profile.mlaNopeDim, 128)
        XCTAssertEqual(parsed.profile.mlaVDim, 128)
        XCTAssertEqual(parsed.profile.nativeMaxContext, 163840)
        XCTAssertTrue(parsed.profile.isKVDerivable)
        // 61 × 128 × (64 + 128 + 128) × 2 B(fp16) = 4,997,120 B/tok — the catalog R1 entry's
        // reconciled 4880 KiB/tok exactly (parity with `ModelArchProfile.catalog`, not a new number).
        XCTAssertEqual(CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 4_997_120)
    }

    /// Field-wiring guard with all-distinct dims so a rope↔nope↔v transposition is caught (R1's
    /// nope == v == 128 can't). 3 × 8 × (1 + 2 + 4) × 2 = 336 B/tok.
    func testDeepseekV3_mlaFieldWiring_distinctDims() throws {
        let json = """
        {
          "model_type": "deepseek_v3",
          "num_hidden_layers": 3,
          "num_attention_heads": 8,
          "qk_rope_head_dim": 1,
          "qk_nope_head_dim": 2,
          "v_head_dim": 4,
          "max_position_embeddings": 4096
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "distinct")
        XCTAssertEqual(parsed.profile.mlaHeads, 8)
        XCTAssertEqual(parsed.profile.mlaRopeDim, 1)
        XCTAssertEqual(parsed.profile.mlaNopeDim, 2)
        XCTAssertEqual(parsed.profile.mlaVDim, 4)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 336)
    }

    /// A missing MLA dim must fail CLOSED (not fall through to a 0 that phantom-GREENs the model):
    /// `kvBytesPerToken` reads `mlaX ?? 0`, so an absent dim would collapse the KV rate.
    func testDeepseekV3_missingMLADim_failsClosed() {
        let json = """
        {
          "model_type": "deepseek_v3", "num_hidden_layers": 61, "num_attention_heads": 128,
          "qk_nope_head_dim": 128, "v_head_dim": 128, "max_position_embeddings": 163840
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField, got \(error)")
            }
            XCTAssertEqual(f, "qk_rope_head_dim")
        }
    }

    /// A present-but-non-positive MLA dim (`v_head_dim: 0`) must ALSO fail closed — the exact
    /// phantom-GREEN class the fit-check exists to prevent, on the MLA side.
    func testDeepseekV3_zeroMLADim_failsClosed() {
        let json = """
        {
          "model_type": "deepseek_v3", "num_hidden_layers": 61, "num_attention_heads": 128,
          "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 0,
          "max_position_embeddings": 163840
        }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.invalidField(let f) = error else {
                return XCTFail("expected invalidField, got \(error)")
            }
            XCTAssertEqual(f, "v_head_dim")
        }
    }

    /// `deepseek_v2` is NOT in the allow-list yet (factory registration unconfirmed) — must still
    /// fail closed, never borrow the deepseek_v3 MLA formula on an unverified family.
    func testDeepseekV2_stillUnsupported() {
        let json = """
        { "model_type": "deepseek_v2", "num_hidden_layers": 60, "num_attention_heads": 128,
          "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
          "max_position_embeddings": 163840 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.unsupportedModelType(let t) = error else {
                return XCTFail("expected unsupportedModelType, got \(error)")
            }
            XCTAssertEqual(t, "deepseek_v2")
        }
    }

    // MARK: - fail-closed + measured flags

    /// An unknown model_type must fail closed (throw), never silently mis-model as uniform-GQA.
    func testUnknownModelType_failsClosed() {
        let json = """
        { "model_type": "some_future_arch", "num_hidden_layers": 10, "num_key_value_heads": 4,
          "head_dim": 128, "max_position_embeddings": 8192 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.unsupportedModelType(let t) = error else {
                return XCTFail("expected unsupportedModelType, got \(error)")
            }
            XCTAssertEqual(t, "some_future_arch")
        }
    }

    /// baichuan_m1 (mlx-community/Baichuan-M1-14B-Instruct-4bit) is now SUPPORTED via the
    /// `.dualGeometrySWA` class — the dual head-geometry capability landed (new ArchClass +
    /// `swaKVHeads`/`swaHeadDim`, `CapacityModel` three-term assembly, a `sliding_window_layers`
    /// enumeration source). Audited directly against the vendored `BaichuanM1.swift` (`newCache`
    /// :265-273, per-layer geometry :70-77) + the real published config on 2026-08-19: global layers
    /// (20) read num_attention_heads/num_key_value_heads (20/2 → kv×dim = 2×256 = 512), sliding-window
    /// layers (20) read num_swa_attention_heads/num_swa_key_value_heads (40/8 → 8×128 = 1024). The
    /// growing term uses the GLOBAL geometry, the window cap uses the SWA geometry (2× the head
    /// product — forcing global would phantom-GREEN by ~320 MiB/seq), and a per-layer conv state is
    /// added on top. Exact integers hand-derived in
    /// docs/task-inbox/2026-08-19-baichuan-m1-dual-geometry-arch-reach.md.
    func testBaichuanM1_dualGeometrySWA_realConfig() throws {
        let json = """
        {
          "model_type": "baichuan_m1",
          "num_hidden_layers": 40,
          "hidden_size": 5120,
          "num_attention_heads": 20,
          "num_key_value_heads": 2,
          "num_swa_attention_heads": 40,
          "num_swa_key_value_heads": 8,
          "sliding_window": 8192,
          "sliding_window_layers": [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
          "conv_window": 2,
          "max_position_embeddings": 32768
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "Baichuan-M1-14B")
        let p = parsed.profile
        XCTAssertEqual(p.modelType, .dualGeometrySWA)
        XCTAssertEqual(p.nLayers, 40)
        XCTAssertEqual(p.nAttnLayers, 20, "40 − 20 sliding_window_layers = 20 global (growing) layers")
        XCTAssertEqual(p.nKVHeads, 2, "global-layer KV heads")
        XCTAssertEqual(p.headDim, 256, "global-layer head dim = hidden_size 5120 / num_attention_heads 20")
        XCTAssertEqual(p.swaKVHeads, 8, "SWA-layer KV heads = num_swa_key_value_heads")
        XCTAssertEqual(p.swaHeadDim, 128, "SWA-layer head dim = hidden_size 5120 / num_swa_attention_heads 40")
        XCTAssertEqual(p.slidingWindow, 8192)
        XCTAssertEqual(p.fixedStateBytes, 122_880, "20×2,048 (global conv) + 20×4,096 (SWA conv)")
        XCTAssertEqual(p.nativeMaxContext, 32_768)
        // Growing term uses GLOBAL geometry; window cap uses SWA geometry (not silently borrowed).
        XCTAssertEqual(CapacityModel.kvBytesPerToken(p, kvQuant: .fp16), 40_960,
            "20 global × 2 kv × 256 dim × 2 (K+V) × 2 B = 40,960 B/token")
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(p, kvQuant: .fp16), 671_088_640,
            "20 SWA × 8 kv × 128 dim × 2 × 2 B × 8192 window — 2× the global-geometry cap")
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(p, context: 32_768, kvQuant: .fp16, concurrency: 1),
            2_013_388_800, "grow(1,342,177,280) + swaCap(671,088,640) + fixedState(122,880)")
    }

    /// A dual-geometry-SWA checkpoint MISSING `sliding_window_layers` fails closed — the global/SWA
    /// split can't be honestly derived, and defaulting either way would mis-size one geometry term
    /// (the phantom-GREEN hazard this class exists to avoid). Fail with `.missingField`, never a guess.
    func testBaichuanM1_missingSlidingWindowLayers_failsClosed() {
        let json = """
        {
          "model_type": "baichuan_m1",
          "num_hidden_layers": 40,
          "hidden_size": 5120,
          "num_attention_heads": 20,
          "num_key_value_heads": 2,
          "num_swa_attention_heads": 40,
          "num_swa_key_value_heads": 8,
          "sliding_window": 8192,
          "conv_window": 2,
          "max_position_embeddings": 32768
        }
        """
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
            "missing sliding_window_layers → can't split global vs SWA layers; fail closed"
        ) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField for sliding_window_layers, got \(error)")
            }
            XCTAssertTrue(f.contains("sliding_window_layers"), "field was \(f)")
        }
    }

    /// When `num_swa_*` head counts are ABSENT, the SWA geometry falls back to the GLOBAL counts —
    /// NOT a fail-closed. This is honest, not a guess: the vendored `BaichuanM1.swift:72-74` runtime
    /// uses the same `?? global` fallback, so absence changes the actual cache allocation to the
    /// global geometry (unlike a sizer-only default that would diverge from what runs). This pins that
    /// behavior so a future "make num_swa_* required" change can't silently regress it. Here the SWA
    /// layers size identically to global (kv 2 × dim 256), so the window cap uses the global product.
    func testDualGeometrySWA_missingSwaHeadCounts_fallsBackToGlobalGeometry() throws {
        let json = """
        {
          "model_type": "baichuan_m1",
          "num_hidden_layers": 40,
          "hidden_size": 5120,
          "num_attention_heads": 20,
          "num_key_value_heads": 2,
          "sliding_window": 8192,
          "sliding_window_layers": [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
          "max_position_embeddings": 32768
        }
        """
        let p = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x").profile
        XCTAssertEqual(p.modelType, .dualGeometrySWA)
        XCTAssertEqual(p.swaKVHeads, 2, "absent num_swa_key_value_heads → falls back to num_key_value_heads")
        XCTAssertEqual(p.swaHeadDim, 256, "absent num_swa_attention_heads → head dim from global num_attention_heads")
        // Window cap now uses the global product (512), i.e. equals what the global geometry would give.
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(p, kvQuant: .fp16), 335_544_320,
            "20 SWA × 2 kv × 256 dim × 2 × 2 B × 8192 window (fallback = global geometry)")
    }

    /// Build MiMo-V2-Flash's `hybrid_layer_pattern`: 48 layers, `0` at the 9 global indices, `1` (SWA)
    /// elsewhere — the verbatim published array (global at {0,5,11,17,23,29,35,41,47}).
    private func mimoHybridLayerPattern() -> String {
        let globalIdx: Set<Int> = [0, 5, 11, 17, 23, 29, 35, 41, 47]
        return "[" + (0..<48).map { globalIdx.contains($0) ? "0" : "1" }.joined(separator: ",") + "]"
    }

    private func mimoV2FlashConfigJSON() -> String {
        """
        {
          "model_type": "mimo_v2_flash",
          "num_hidden_layers": 48,
          "hidden_size": 4096,
          "num_attention_heads": 64,
          "num_key_value_heads": 4,
          "head_dim": 192,
          "v_head_dim": 128,
          "swa_num_attention_heads": 64,
          "swa_num_key_value_heads": 8,
          "swa_head_dim": 192,
          "swa_v_head_dim": 128,
          "sliding_window": 128,
          "sliding_window_size": 128,
          "max_position_embeddings": 262144,
          "hybrid_layer_pattern": \(mimoHybridLayerPattern())
        }
        """
    }

    /// mimo_v2_flash (XiaomiMiMo/MiMo-V2-Flash): dual-geometry SWA with ASYMMETRIC K/V — K cached at
    /// head_dim 192, V at the narrower v_head_dim 128 on both layer classes (vendored
    /// `MiMoV2Flash.swift` projections :141-146, cache write :49). The KV formula sizes K and V
    /// separately (`headDim + vHeadDim`), never `2×headDim`. Global/SWA split from `hybrid_layer_pattern`
    /// (9 global, 39 SWA); no conv/recurrent state (`fixedStateBytes = 0`); window from
    /// `sliding_window_size`. Exact integers from
    /// docs/task-inbox/2026-08-19-mimo-v2-flash-dual-geometry-vheaddim.md.
    func testMimoV2Flash_dualGeometrySWA_asymmetricKV_realConfig() throws {
        let parsed = try ModelConfigDecoder.decode(
            configJSON: data(mimoV2FlashConfigJSON()), safetensorsBytes: 1, id: "MiMo-V2-Flash")
        let p = parsed.profile
        XCTAssertEqual(p.modelType, .dualGeometrySWA)
        XCTAssertEqual(p.nLayers, 48)
        XCTAssertEqual(p.nAttnLayers, 9, "9 zeros in hybrid_layer_pattern = 9 global (growing) layers")
        XCTAssertEqual(p.nKVHeads, 4, "global KV heads")
        XCTAssertEqual(p.headDim, 192, "global K head dim (explicit head_dim, not hidden/heads)")
        XCTAssertEqual(p.vHeadDim, 128, "global V head dim — narrower than K (asymmetric)")
        XCTAssertEqual(p.swaKVHeads, 8, "SWA KV heads = swa_num_key_value_heads")
        XCTAssertEqual(p.swaHeadDim, 192, "SWA K head dim = swa_head_dim")
        XCTAssertEqual(p.swaVHeadDim, 128, "SWA V head dim = swa_v_head_dim")
        XCTAssertEqual(p.slidingWindow, 128, "from sliding_window_size")
        XCTAssertEqual(p.fixedStateBytes, 0, "no MambaCache/conv state; sink biases are weights")
        XCTAssertEqual(p.nativeMaxContext, 262_144)
        // grow: 9 global × 4 kv × (192+128) × 2 B = 23,040 B/token
        XCTAssertEqual(CapacityModel.kvBytesPerToken(p, kvQuant: .fp16), 23_040)
        // SWA cap: 39 SWA × 8 kv × (192+128) × 2 B × 128 window = 25,559,040 B
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(p, kvQuant: .fp16), 25_559_040)
        // KV@32K/seq = 23,040 × 32,768 (754,974,720) + 25,559,040 + 0 = 780,533,760 B
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(p, context: 32_768, kvQuant: .fp16, concurrency: 1),
            780_533_760)
        // Phantom-GREEN witness: forcing the narrow V dim symmetric (128×2) under-counts the growing
        // term by ~20% — the exact mis-size the vHeadDim generalization prevents.
        let symmetricUndercount = Double(9 * 4 * 128 * 2) * 2  // 9×4×128×2(K+V)×2 B = 18,432
        XCTAssertEqual(symmetricUndercount, 18_432)
        XCTAssertGreaterThan(CapacityModel.kvBytesPerToken(p, kvQuant: .fp16), symmetricUndercount)
    }

    /// mimo_v2_flash geometry fields are non-optional in the vendored Codable — a config missing
    /// `v_head_dim` fails CLOSED (no fallback), unlike baichuan's `??`-to-global SWA head counts.
    func testMimoV2Flash_missingVHeadDim_failsClosed() {
        let json = mimoV2FlashConfigJSON().replacingOccurrences(of: "\"v_head_dim\": 128,", with: "")
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
            "missing v_head_dim → asymmetric cache can't be sized; fail closed"
        ) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField for v_head_dim, got \(error)")
            }
            XCTAssertTrue(f.contains("v_head_dim"), "field was \(f)")
        }
    }

    /// A `hybrid_layer_pattern` whose length ≠ num_hidden_layers can't honestly split global vs SWA —
    /// fail closed rather than mis-count the growing layers.
    func testMimoV2Flash_hybridPatternLengthMismatch_failsClosed() {
        let json = mimoV2FlashConfigJSON().replacingOccurrences(
            of: "\"hybrid_layer_pattern\": \(mimoHybridLayerPattern())",
            with: "\"hybrid_layer_pattern\": [0,1,0,1]")
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
            "hybrid_layer_pattern length 4 != 48 layers; fail closed"
        ) { error in
            guard case ModelConfigDecodeError.invalidField(let f) = error else {
                return XCTFail("expected invalidField for hybrid_layer_pattern, got \(error)")
            }
            XCTAssertTrue(f.contains("hybrid_layer_pattern"), "field was \(f)")
        }
    }

    // MARK: - gemma4 dual-geometry SWA + KV-layer-sharing (real mlx-community/gemma-4-e2b-it-4bit)

    /// MiMo-V2-Flash's `hybrid_layer_pattern` sibling for gemma4: the real e2b `layer_types` (35 entries,
    /// `full_attention` at indices {4,9,14,19,24,29,34}, `sliding_attention` elsewhere).
    private func gemma4E2BLayerTypes() -> String {
        let fullIdx: Set<Int> = [4, 9, 14, 19, 24, 29, 34]
        let types = (0..<35).map { fullIdx.contains($0) ? "\"full_attention\"" : "\"sliding_attention\"" }
        return "[" + types.joined(separator: ",") + "]"
    }

    /// The real `mlx-community/gemma-4-e2b-it-4bit` text geometry, nested under `text_config` (the
    /// `Gemma4ForConditionalGeneration` multimodal wrapper shape — root `model_type` "gemma4").
    private func gemma4E2BNestedConfigJSON(rootModelType: String = "gemma4") -> String {
        """
        {
          "model_type": "\(rootModelType)",
          "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "sliding_window": 512,
            "attention_k_eq_v": false,
            "max_position_embeddings": 131072,
            "layer_types": \(gemma4E2BLayerTypes())
          }
        }
        """
    }

    /// gemma4 (Google Gemma-4, real e2b config): dual-geometry SWA — global layers cache at
    /// `global_head_dim` 512, sliding at `head_dim` 256 — PLUS KV-layer-sharing (only the first
    /// `35 − 20 = 15` layers own a cache). Within that cached prefix: `full_attention` at {4,9,14} = 3
    /// growing (global) layers + 12 sliding (window-capped). The shared tail (15..34, incl. full layers
    /// 19/24/29/34) allocates NOTHING — counting it would phantom-RED. No conv/recurrent state. Exact
    /// integers from the real published config (fetched cycleU); see
    /// docs/task-inbox/2026-08-19-gemma4-dual-geometry-kv-shared-arch-reach.md.
    func testGemma4_e2b_realConfig_dualGeometryKVShared() throws {
        let parsed = try ModelConfigDecoder.decode(
            configJSON: data(gemma4E2BNestedConfigJSON()), safetensorsBytes: 1, id: "gemma-4-e2b")
        let p = parsed.profile
        XCTAssertEqual(p.modelType, .dualGeometrySWA)
        XCTAssertEqual(p.nLayers, 15, "cached prefix = num_hidden_layers 35 − num_kv_shared_layers 20")
        XCTAssertEqual(p.nAttnLayers, 3, "full_attention within the first 15 layers = indices {4,9,14}")
        XCTAssertEqual(p.nKVHeads, 1, "global KV heads (attention_k_eq_v false → num_key_value_heads)")
        XCTAssertEqual(p.headDim, 512, "GLOBAL cache dim = global_head_dim (NOT the sliding head_dim 256)")
        XCTAssertEqual(p.swaKVHeads, 1, "sliding KV heads = num_key_value_heads")
        XCTAssertEqual(p.swaHeadDim, 256, "sliding cache dim = head_dim")
        XCTAssertNil(p.vHeadDim, "symmetric K/V (global)")
        XCTAssertNil(p.swaVHeadDim, "symmetric K/V (sliding)")
        XCTAssertEqual(p.slidingWindow, 512)
        XCTAssertEqual(p.fixedStateBytes, 0, "KV-only caches; PLE embeddings are weights, not per-seq state")
        XCTAssertEqual(p.nativeMaxContext, 131072)
        // grow: 3 global × 1 kv × (512+512) × 2 B = 6,144 B/token
        XCTAssertEqual(CapacityModel.kvBytesPerToken(p, kvQuant: .fp16), 6_144)
        // SWA cap: 12 sliding × 1 kv × (256+256) × 2 B × 512 window = 6,291,456 B
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(p, kvQuant: .fp16), 6_291_456)
        // KV@32K/seq = 6,144 × 32,768 (201,326,592) + 6,291,456 + 0 = 207,618,048 B
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(p, context: 32_768, kvQuant: .fp16, concurrency: 1),
            207_618_048)
        // Phantom-RED witness: had we NOT truncated the shared tail, all 7 full layers (of 35) would
        // count → 7/3 × the growing term, a large false too-small-host refusal.
        XCTAssertLessThan(p.nAttnLayers, 7, "shared-tail full layers must not inflate the growing term")
    }

    /// gemma4/gemma4_unified/gemma4_text are ONE geometry: `gemma4_text` ships the text fields FLAT at
    /// the root (no `text_config` wrapper); the decoder's text-scope-then-root `geom()` fallback must
    /// yield the SAME profile as the nested wrapper. Also covers `gemma4_unified` as the root tag.
    func testGemma4_flatText_and_unified_equalNested() throws {
        let nested = try ModelConfigDecoder.decode(
            configJSON: data(gemma4E2BNestedConfigJSON()), safetensorsBytes: 1, id: "n").profile
        // Flat gemma4_text: hoist the text_config fields to the root.
        let flat = """
        {
          "model_type": "gemma4_text",
          "num_hidden_layers": 35, "hidden_size": 1536, "num_attention_heads": 8,
          "head_dim": 256, "global_head_dim": 512, "num_key_value_heads": 1,
          "num_kv_shared_layers": 20, "sliding_window": 512, "attention_k_eq_v": false,
          "max_position_embeddings": 131072, "layer_types": \(gemma4E2BLayerTypes())
        }
        """
        let flatP = try ModelConfigDecoder.decode(configJSON: data(flat), safetensorsBytes: 1, id: "f").profile
        XCTAssertEqual(flatP.modelType, .dualGeometrySWA)
        XCTAssertEqual(CapacityModel.kvBytesForContext(flatP, context: 32_768, kvQuant: .fp16, concurrency: 1),
            CapacityModel.kvBytesForContext(nested, context: 32_768, kvQuant: .fp16, concurrency: 1),
            "flat gemma4_text must size identically to the nested wrapper")
        // gemma4_unified root tag → same class + same numbers.
        let unified = try ModelConfigDecoder.decode(
            configJSON: data(gemma4E2BNestedConfigJSON(rootModelType: "gemma4_unified")),
            safetensorsBytes: 1, id: "u").profile
        XCTAssertEqual(unified.modelType, .dualGeometrySWA)
        XCTAssertEqual(unified.nAttnLayers, nested.nAttnLayers)
        XCTAssertEqual(unified.headDim, 512)
    }

    /// When `layer_types` is ABSENT, gemma4 synthesizes the vendored default (`Gemma4Text.swift:148-158`):
    /// within each block of `sliding_window_pattern`, the LAST index is full_attention (layer i full iff
    /// `i % pattern == pattern−1`), counted over the cached prefix. With pattern 5 and a 15-layer prefix,
    /// full at {4,9,14} = 3 — identical to the explicit e2b layer_types.
    func testGemma4_noLayerTypes_synthesizesFromSlidingWindowPattern() throws {
        let json = """
        {
          "model_type": "gemma4_text",
          "num_hidden_layers": 35, "hidden_size": 1536, "num_attention_heads": 8,
          "head_dim": 256, "global_head_dim": 512, "num_key_value_heads": 1,
          "num_kv_shared_layers": 20, "sliding_window": 512, "sliding_window_pattern": 5,
          "max_position_embeddings": 131072
        }
        """
        let p = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "syn").profile
        XCTAssertEqual(p.nAttnLayers, 3, "synthesized 5:1 pattern → full at {4,9,14} within the 15-layer prefix")
        XCTAssertEqual(CapacityModel.kvBytesPerToken(p, kvQuant: .fp16), 6_144)
    }

    /// gemma4 fail-closed matrix — every geometry field is required (its vendored Codable has a constant
    /// default, so absence must REFUSE, not guess a mis-size):
    ///  - num_kv_shared_layers absent → can't truncate the cached prefix (assuming 0 over-counts 20 phantom
    ///    layers) → missingField.
    ///  - global_head_dim absent → the global cache dim can't be sized (head_dim is the SLIDING dim) →
    ///    missingField.
    ///  - head_dim absent → the sliding cache dim can't be sized → missingField.
    ///  - layer_types AND sliding_window_pattern both absent → the global/SWA split can't be derived →
    ///    missingField.
    ///  - sliding_window absent → the window cap can't be derived → missingField.
    func testGemma4_missingGeometry_failsClosed() {
        let base = gemma4E2BNestedConfigJSON()
        let cases: [(String, String)] = [
            ("\"num_kv_shared_layers\": 20,", "num_kv_shared_layers"),
            ("\"global_head_dim\": 512,", "global_head_dim"),
            ("\"head_dim\": 256,", "head_dim"),
            ("\"sliding_window\": 512,", "sliding_window"),
        ]
        for (needle, field) in cases {
            let json = base.replacingOccurrences(of: needle, with: "")
            XCTAssertThrowsError(
                try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
                "missing \(field) must fail closed"
            ) { error in
                guard case ModelConfigDecodeError.missingField(let f) = error else {
                    return XCTFail("expected missingField(\(field)), got \(error)")
                }
                XCTAssertTrue(f.contains(field), "field was \(f), expected \(field)")
            }
        }
        // layer_types absent AND no sliding_window_pattern → the split can't be derived.
        let noSplit = """
        {
          "model_type": "gemma4_text",
          "num_hidden_layers": 35, "hidden_size": 1536, "num_attention_heads": 8,
          "head_dim": 256, "global_head_dim": 512, "num_key_value_heads": 1,
          "num_kv_shared_layers": 20, "sliding_window": 512, "max_position_embeddings": 131072
        }
        """
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(noSplit), safetensorsBytes: 1, id: "x"),
            "no layer_types and no sliding_window_pattern → fail closed"
        ) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField for the split, got \(error)")
            }
            XCTAssertTrue(f.contains("layer_types") || f.contains("sliding_window_pattern"), "field was \(f)")
        }
    }

    /// A cached prefix with NO full_attention layer would zero the growing-KV term (phantom-GREEN at long
    /// contexts, and trips the derivability sentinel). Fail closed. Constructed by making all of the first
    /// 15 layers sliding (full only in the shared tail).
    func testGemma4_noFullAttentionInPrefix_failsClosed() {
        // full_attention only at indices 20,25,30 (all in the shared tail 15..34); prefix 0..14 all sliding.
        let fullIdx: Set<Int> = [20, 25, 30]
        let types = (0..<35).map { fullIdx.contains($0) ? "\"full_attention\"" : "\"sliding_attention\"" }
        let lt = "[" + types.joined(separator: ",") + "]"
        let json = gemma4E2BNestedConfigJSON().replacingOccurrences(
            of: gemma4E2BLayerTypes(), with: lt)
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
            "no full_attention in the cached prefix → fail closed"
        ) { error in
            guard case ModelConfigDecodeError.invalidField(let f) = error else {
                return XCTFail("expected invalidField, got \(error)")
            }
            XCTAssertTrue(f.contains("layer_types"), "field was \(f)")
        }
    }

    /// ACCEPTANCE: decode the REAL published `config.json` from disk (checked into Fixtures), not a
    /// synthetic reconstruction. This exercises the genuine `Gemma4ForConditionalGeneration` multimodal
    /// wrapper — `audio_config`/`vision_config`/`quantization_config`/`generation_config` sibling blocks,
    /// the real `text_config` nesting, and real JSON number/bool types. Critically, the real
    /// `vision_config` carries its OWN `head_dim` (64 in e2b, distinct from the text tower's 256): this
    /// proves `geom()` reads the TEXT scope's `head_dim`, not a sibling block's — a shadow the synthetic
    /// fixtures don't contain. Both real checkpoints (e2b: 35L/20-shared/kv1; e4b: 42L/18-shared/kv2)
    /// map to the same `.dualGeometrySWA` KV-shared formula with their own exact integers. This is
    /// fable's stated acceptance criterion ("a live fit_check verdict, not skipped, on a Gemma-4
    /// checkpoint") proven at the decode boundary.
    func testGemma4_realPublishedConfigs_decodeFromDisk() throws {
        // e2b: cached prefix 15, full {4,9,14}=3 growing + 12 sliding, global 512 / sliding 256, kv 1.
        let e2bURL = try XCTUnwrap(Bundle.module.url(
            forResource: "gemma4-e2b-it-4bit-config", withExtension: "json"),
            "missing Fixtures/gemma4-e2b-it-4bit-config.json")
        let e2b = try ModelConfigDecoder.decode(
            configJSON: Data(contentsOf: e2bURL), safetensorsBytes: 1, id: "gemma-4-e2b-it-4bit").profile
        XCTAssertEqual(e2b.modelType, .dualGeometrySWA)
        XCTAssertEqual(e2b.nLayers, 15)
        XCTAssertEqual(e2b.nAttnLayers, 3)
        XCTAssertEqual(e2b.nKVHeads, 1)
        XCTAssertEqual(e2b.headDim, 512, "GLOBAL dim from text_config.global_head_dim — NOT vision_config.head_dim (64)")
        XCTAssertEqual(e2b.swaHeadDim, 256, "sliding dim from text_config.head_dim")
        XCTAssertEqual(e2b.swaKVHeads, 1)
        XCTAssertEqual(e2b.slidingWindow, 512)
        XCTAssertEqual(e2b.fixedStateBytes, 0)
        XCTAssertEqual(e2b.nativeMaxContext, 131072)
        XCTAssertEqual(CapacityModel.kvBytesForContext(e2b, context: 32_768, kvQuant: .fp16, concurrency: 1),
            207_618_048, "e2b KV@32K on the real config")

        // e4b: cached prefix 24, full {5,11,17,23}=4 growing + 20 sliding, global 512 / sliding 256, kv 2.
        let e4bURL = try XCTUnwrap(Bundle.module.url(
            forResource: "gemma4-e4b-it-4bit-config", withExtension: "json"),
            "missing Fixtures/gemma4-e4b-it-4bit-config.json")
        let e4b = try ModelConfigDecoder.decode(
            configJSON: Data(contentsOf: e4bURL), safetensorsBytes: 1, id: "gemma-4-e4b-it-4bit").profile
        XCTAssertEqual(e4b.modelType, .dualGeometrySWA)
        XCTAssertEqual(e4b.nLayers, 24, "42 − 18 shared")
        XCTAssertEqual(e4b.nAttnLayers, 4, "full_attention {5,11,17,23} within the 24-layer prefix")
        XCTAssertEqual(e4b.nKVHeads, 2)
        XCTAssertEqual(e4b.headDim, 512)
        XCTAssertEqual(e4b.swaHeadDim, 256)
        XCTAssertEqual(e4b.swaKVHeads, 2)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(e4b, kvQuant: .fp16), 16_384, "4 × 2 × (512+512) × 2 B")
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(e4b, kvQuant: .fp16), 20_971_520,
            "20 × 2 × (256+256) × 2 B × 512 window")
        XCTAssertEqual(CapacityModel.kvBytesForContext(e4b, context: 32_768, kvQuant: .fp16, concurrency: 1),
            557_842_432, "e4b KV@32K on the real config")
    }

    /// Falcon-H1 (tiiuae/Falcon-H1-*): a PARALLEL hybrid — the vendored `newCache`
    /// (`FalconH1.swift:799-801`) gives EVERY layer `CacheList(MambaCache(), attentionCache.copy())`,
    /// so all layers both grow an attention KV cache AND hold a Mamba-2 conv+SSM recurrent state (unlike
    /// the interval-select qwen3_5/jamba/lfm2, where only the non-attention layers are recurrent). Maps
    /// to `.hybridLinear` with nAttnLayers == nLayers; the fixed term is sized over ALL layers by
    /// `resolveFalconH1StateBytes`. Geometry + dtypes are source-verified against the real
    /// Falcon-H1-34B-Instruct config.json and `FalconH1.swift`/`SSM.swift`:
    ///  - conv state `[B, mamba_d_conv-1, convDim]` at activation precision (`FalconH1.swift:424-426`),
    ///    convDim = mamba_d_ssm + 2·mamba_n_groups·mamba_d_state (`:376`, where `intermediateSize =
    ///    args.mambaDSSM` at `:364` — NOT mamba_expand×hidden, which would be 10,240 and 2.5× the term).
    ///  - SSM state = mamba_d_ssm · mamba_d_state, stored at activation precision (2 B): `SSM.swift:167`
    ///    sets `stateType = state?.dtype ?? x.dtype` and `:219` casts `nextState.asType(stateType)`, so
    ///    the cache slot follows the activation dtype forever (same provenance as the conv slot — hence
    ///    ×2, not the ×4 conservative inference used for jamba where the dtype was unknown).
    /// Real Falcon-H1-34B: 72 layers, kv 4 × head_dim 128 (growing 144 KiB/tok ≈ 4.6 GiB @ 32K — the
    /// dominant term), mamba_d_ssm 4096, n_groups 2, d_state 256, d_conv 4. per_layer = conv
    /// (4-1)·5120·2 = 30,720 + ssm 4096·256·2 = 2,097,152 = 2,127,872; × 72 = 153,206,784 B ≈ 146.1 MiB/seq.
    func testFalconH1_parallelHybrid_mamba2State() throws {
        let json = """
        {
          "model_type": "falcon_h1",
          "num_hidden_layers": 72,
          "num_attention_heads": 20,
          "num_key_value_heads": 4,
          "head_dim": 128,
          "hidden_size": 5120,
          "mamba_d_ssm": 4096,
          "mamba_n_groups": 2,
          "mamba_d_state": 256,
          "mamba_d_conv": 4,
          "mamba_n_heads": 32,
          "mamba_d_head": 128,
          "max_position_embeddings": 262144
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 19_000_000_000, id: "Falcon-H1-34B")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nLayers, 72)
        // PARALLEL hybrid: EVERY layer grows a KV cache (degenerate nAttnLayers == nLayers). Locked so a
        // future refactor can never silently make the fixed-state resolver see 0 recurrent layers.
        XCTAssertEqual(parsed.profile.nAttnLayers, 72, "all layers grow attention KV (parallel hybrid)")
        XCTAssertEqual(parsed.profile.nKVHeads, 4)
        XCTAssertEqual(parsed.profile.headDim, 128)
        XCTAssertEqual(parsed.profile.fixedStateBytes, 153_206_784,
            "72 layers × (conv 30,720 + ssm 2,097,152), both at activation precision (2 B)")
        XCTAssertTrue(parsed.profile.isKVDerivable)
    }

    /// Falcon-H1 fail-CLOSED (not fail-open to 0) when `mamba_d_ssm` is absent: the vendored code defaults
    /// it to 1536 (`FalconH1.swift:149`), but mirroring a vendored default in the sizer is exactly what the
    /// afmoe precedent rejects — a default-relying checkpoint could then be sized against a guessed inner
    /// dim. Refuse instead so no phantom fixed-state term is emitted.
    func testFalconH1_missingMambaDSSM_failsClosed() {
        let json = """
        {
          "model_type": "falcon_h1", "num_hidden_layers": 72, "num_attention_heads": 20,
          "num_key_value_heads": 4, "head_dim": 128, "hidden_size": 5120,
          "mamba_n_groups": 2, "mamba_d_state": 256, "mamba_d_conv": 4, "max_position_embeddings": 262144
        }
        """
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 19_000_000_000, id: "x"),
            "falcon_h1 without mamba_d_ssm must refuse, not size against a vendored default"
        ) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField(mamba_d_ssm), got \(error)")
            }
            XCTAssertTrue(f.contains("mamba_d_ssm"), "missing field should name mamba_d_ssm, got \(f)")
        }
    }

    /// GraniteMoeHybrid (IBM `granitemoehybrid`, e.g. `ibm-granite/granite-4.0-h-micro`): an
    /// INTERVAL-SELECT hybrid (unlike falcon_h1's parallel shape) — the vendored `GraniteMoeHybrid.swift`
    /// `newCache` (:520-525) grows a plain `KVCacheSimple` on `layer_types == "attention"` layers and a
    /// `MambaCache` on the `"mamba"` layers, so the recurrent state lives on `nLayers − nAttnLayers` layers
    /// (not all layers). Its recurrent term is a CLASSIC Mamba-2 conv+SSM state sized from the mamba mixer
    /// geometry, source-verified against `GraniteMoeHybrid.swift` + the shared `SSM.swift`:
    ///  - mamba inner dim `intermediateSize = mamba_n_heads × mamba_d_head` (GraniteMoeHybrid.swift:79),
    ///    NOT `mamba_expand × hidden` (they coincide here: 64×64 = 2×2048 = 4096, but the arch uses the
    ///    heads×head_dim product).
    ///  - conv state `[B, mamba_d_conv−1, convDim]`, `convDim = intermediateSize + 2·mamba_n_groups·mamba_d_state`
    ///    (:83, :116), at activation precision (2 B).
    ///  - SSM state `intermediateSize × mamba_d_state` (from `state.reshaped(b,1,g,repeats,dh,d)` at
    ///    SSM.swift:209, g·repeats·dh = numHeads·headDim = intermediateSize), stored at the ACTIVATION dtype
    ///    (`stateType = state?.dtype ?? x.dtype`, SSM.swift:167/219) — ×2, the same shared SSM path falcon_h1
    ///    uses, NOT the ×4 fp32 width jamba conservatively assumes for an unknown SSM dtype.
    /// Attention layers are enumerated in `layer_types` as `"attention"` (not lfm2's `"full_attention"`),
    /// which `resolveHybridAttnLayers` now counts. The MoE router (`num_local_experts`,
    /// `shared_intermediate_size`) is orthogonal to KV/state geometry — the h-micro is dense
    /// (`num_local_experts: 0`) yet still tagged `granitemoehybrid`. Geometry is the REAL published config
    /// of `ibm-granite/granite-4.0-h-micro` (huggingface.co/ibm-granite/granite-4.0-h-micro/raw/main/config.json,
    /// fetched 2026-08-19): 40 layers, layer_types with 4 `attention` (indices 5,15,25,35) + 36 `mamba`,
    /// hidden 2048, 32 heads (head_dim derived 2048/32 = 64), 8 kv heads, mamba_n_heads 64, mamba_d_head 64,
    /// mamba_d_state 128, mamba_n_groups 1, mamba_d_conv 4, max_position_embeddings 131072.
    func testGraniteMoeHybrid_realConfig_intervalHybridMamba2State() throws {
        let json = """
        {
          "model_type": "granitemoehybrid",
          "num_hidden_layers": 40,
          "hidden_size": 2048,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "mamba_n_heads": 64,
          "mamba_d_head": 64,
          "mamba_d_state": 128,
          "mamba_n_groups": 1,
          "mamba_d_conv": 4,
          "mamba_expand": 2,
          "num_local_experts": 0,
          "shared_intermediate_size": 8192,
          "layer_types": [
            "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba"
          ],
          "max_position_embeddings": 131072
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "granite-4.0-h-micro")
        XCTAssertEqual(parsed.profile.modelType, .hybridLinear)
        XCTAssertEqual(parsed.profile.nLayers, 40)
        XCTAssertEqual(parsed.profile.nAttnLayers, 4, "4 `attention` entries in layer_types grow a KVCacheSimple; 36 mamba layers are recurrent")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 64, "no head_dim field → derived hidden_size/num_attention_heads = 2048/32")
        XCTAssertEqual(parsed.profile.nativeMaxContext, 131072)
        // inner = mamba_n_heads 64 × mamba_d_head 64 = 4096; convDim = 4096 + 2·1·128 = 4352.
        // Per mamba layer: conv (4−1)×4352×2 = 26,112 B + ssm 4096×128×2 = 1,048,576 B = 1,074,688 B.
        // × 36 mamba layers = 38,688,768 B.
        XCTAssertEqual(parsed.profile.fixedStateBytes, 38_688_768,
            "classic Mamba-2 conv+SSM state over 36 mamba layers, both at activation precision (2 B)")

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(parsed.profile, kvQuant: .fp16), 8192,
            "4 attention layers × 8 kv heads × 64 head_dim × 2 (K+V) × 2 bytes (fp16) = 8 KiB/tok")
        let kv = CapacityModel.kvBytesForContext(parsed.profile, context: 32768, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(kv, 268_435_456 + 38_688_768, "growing 8 KiB/tok @32K (256 MiB) + fixed Mamba-2 state")
    }

    /// GraniteMoeHybrid fails CLOSED (not fail-open to 0) when a mamba geometry field is absent: the
    /// recurrent term is MATERIAL (~1 MiB/layer × 36 ≈ 37 MiB/seq here, far more at scale), so a fail-open
    /// 0 would be a real under-count. Mirrors the falcon_h1 precedent.
    func testGraniteMoeHybrid_missingMambaGeometry_failsClosed() {
        let json = """
        {
          "model_type": "granitemoehybrid", "num_hidden_layers": 40, "hidden_size": 2048,
          "num_attention_heads": 32, "num_key_value_heads": 8,
          "mamba_d_head": 64, "mamba_d_state": 128, "mamba_n_groups": 1, "mamba_d_conv": 4,
          "layer_types": [
            "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention",
            "mamba", "mamba", "mamba", "mamba"
          ],
          "max_position_embeddings": 131072
        }
        """
        XCTAssertThrowsError(
            try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
            "granitemoehybrid without mamba_n_heads must refuse, not fail-open to a 0 recurrent term"
        ) { error in
            guard case ModelConfigDecodeError.missingField(let f) = error else {
                return XCTFail("expected missingField(mamba_n_heads), got \(error)")
            }
            XCTAssertTrue(f.contains("mamba_n_heads"), "missing field should name mamba_n_heads, got \(f)")
        }
    }

    /// Families that need a NEW `ArchClass` formula (not an allow-list add) must fail closed. openelm
    /// carries a PER-LAYER-VARYING `num_key_value_heads` array (not the scalar every audited class
    /// reads), so uniformGQA's `nKVHeads × head_dim × nAttnLayers` would use the wrong head count on
    /// most layers; mamba2 is a pure SSM with NO attention KV cache at all, so no per-token KV formula
    /// applies. Neither can be modeled by borrowing an existing class — refuse until a dedicated
    /// formula is audited and added.
    func testUnauditedArch_requiresNewFormula_stillUnsupported() {
        for family in ["openelm", "mamba2"] {
            let json = """
            { "model_type": "\(family)", "num_hidden_layers": 32, "num_attention_heads": 32,
              "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 32768 }
            """
            XCTAssertThrowsError(
                try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x"),
                "\(family) needs a dedicated formula; it must fail closed, not borrow an existing class"
            ) { error in
                guard case ModelConfigDecodeError.unsupportedModelType(let t) = error else {
                    return XCTFail("expected unsupportedModelType for \(family), got \(error)")
                }
                XCTAssertEqual(t, family)
            }
        }
    }

    func testMalformedJSON_throws() {
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data("{ not json"), safetensorsBytes: 1, id: "x"))
    }

    // MARK: - geometry sanity (present-but-garbage fields fail closed, not phantom-GREEN)

    /// A corrupt/hand-edited config with `num_key_value_heads: 0` must fail closed. Before this
    /// guard it decoded fine and `kvBytesPerToken` returned `nLayers × 0 × headDim × … = 0` → KV = 0
    /// → the model classified GREEN at any context (a phantom fit). Presence-only checks weren't
    /// enough on the geometry side.
    func testNonPositiveKVHeads_failsClosed() {
        let json = """
        { "model_type": "qwen3", "num_hidden_layers": 32, "num_attention_heads": 32,
          "num_key_value_heads": 0, "head_dim": 128, "max_position_embeddings": 32768 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.invalidField(let f) = error else {
                return XCTFail("expected invalidField, got \(error)")
            }
            XCTAssertEqual(f, "num_key_value_heads")
        }
    }

    /// `head_dim: 0` (or a hidden_size/num_attention_heads pair that derives to 0) must also fail
    /// closed — same phantom-GREEN class via the other factor of the per-token rate.
    func testZeroHeadDim_failsClosed() {
        let json = """
        { "model_type": "qwen3", "num_hidden_layers": 32, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 0, "max_position_embeddings": 32768 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.invalidField(let f) = error else {
                return XCTFail("expected invalidField, got \(error)")
            }
            XCTAssertEqual(f, "head_dim")
        }
    }

    /// Non-positive `num_hidden_layers` / `max_position_embeddings` fail closed too (a 0-layer or
    /// 0-context config is not servable and must not size as trivially-fits).
    func testZeroLayersOrContext_failClosed() {
        let zeroLayers = """
        { "model_type": "qwen3", "num_hidden_layers": 0, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 32768 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(zeroLayers), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.invalidField("num_hidden_layers") = error else {
                return XCTFail("expected invalidField(num_hidden_layers), got \(error)")
            }
        }
        let zeroContext = """
        { "model_type": "qwen3", "num_hidden_layers": 32, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 0 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(zeroContext), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.invalidField("max_position_embeddings") = error else {
                return XCTFail("expected invalidField(max_position_embeddings), got \(error)")
            }
        }
    }

    /// An out-of-`Int`-range JSON number (`1e300`) must fail closed, not sign-flip into a garbage
    /// (often negative) layer/head count via `NSNumber.intValue`. `intOf` rejects non-integral /
    /// out-of-range doubles so the value reads as absent → fail closed.
    func testOutOfRangeNumericField_failsClosed() {
        let json = """
        { "model_type": "qwen3", "num_hidden_layers": 1e300, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 32768 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            switch error {
            case ModelConfigDecodeError.missingField("num_hidden_layers"),
                 ModelConfigDecodeError.invalidField("num_hidden_layers"):
                break
            default:
                XCTFail("expected missing/invalid num_hidden_layers, got \(error)")
            }
        }
    }

    /// safetensorsBytes == 0 (unknown) → weightsAreMeasured is false (the fit-check caller must treat
    /// the weights figure as unverified rather than a measured fact).
    func testUnknownSafetensorsBytes_notMeasured() throws {
        let json = """
        { "model_type": "mistral", "num_hidden_layers": 40, "num_attention_heads": 32,
          "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 131072 }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 0, id: "x")
        XCTAssertFalse(parsed.weightsAreMeasured)
    }

    // MARK: - filesystem wrapper (the path the live serve uses)

    /// `decodeModelDirectory` reads config.json and sums real *.safetensors bytes from a temp dir —
    /// the end-to-end on-disk path the serving fit-check exercises.
    func testDecodeModelDirectory_readsConfigAndSumsSafetensors() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-fixture-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let config = """
        { "model_type": "qwen3", "num_hidden_layers": 48, "num_attention_heads": 40,
          "num_key_value_heads": 4, "head_dim": 128, "max_position_embeddings": 262144 }
        """
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        // Two shards: 1000 + 500 bytes → summed real weights = 1500.
        try Data(count: 1000).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data(count: 500).write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        // A non-safetensors file must be ignored.
        try Data(count: 9999).write(to: dir.appendingPathComponent("tokenizer.json"))

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")
        XCTAssertEqual(parsed.profile.modelType, .uniformGQA)
        XCTAssertEqual(parsed.profile.nLayers, 48)
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, 1500, "summed only the two safetensors shards")
        XCTAssertTrue(parsed.weightsAreMeasured)
        XCTAssertEqual(parsed.profile.id, "fixture")
    }

    /// Hugging Face snapshot dirs store *.safetensors as SYMLINKS into the blobs cache; the sizer
    /// must sum the real blob bytes, not the ~76-byte symlink. (Live smoke showed weights=0.00 GiB
    /// for a 5.5 GiB checkpoint because the symlink size was read instead of the target.)
    func testSumSafetensorsBytes_followsSymlinks() throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appendingPathComponent("fitcheck-symlink-\(UUID().uuidString)", isDirectory: true)
        let blobs = base.appendingPathComponent("blobs", isDirectory: true)
        let snapshot = base.appendingPathComponent("snapshot", isDirectory: true)
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: base) }

        // A 2000-byte blob, and a symlink to it named like a real shard.
        let blob = blobs.appendingPathComponent("abc123")
        try Data(count: 2000).write(to: blob)
        let link = snapshot.appendingPathComponent("model-00001-of-00001.safetensors")
        try fm.createSymbolicLink(at: link, withDestinationURL: blob)

        XCTAssertEqual(ModelConfigDecoder.sumSafetensorsBytes(in: snapshot), 2000,
            "must follow the symlink to the real blob size, not read the symlink itself")
    }

    /// A directory without config.json fails closed with `.missingConfigFile`.
    func testDecodeModelDirectory_missingConfig_throws() {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-empty-\(UUID().uuidString)", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        XCTAssertThrowsError(try ModelConfigDecoder.decodeModelDirectory(dir)) { error in
            guard case ModelConfigDecodeError.missingConfigFile = error else {
                return XCTFail("expected missingConfigFile, got \(error)")
            }
        }
    }

    /// A missing required geometry field fails closed rather than defaulting to a wrong number.
    func testMissingRequiredField_throws() {
        let json = """
        { "model_type": "qwen3", "num_hidden_layers": 32, "max_position_embeddings": 32768 }
        """
        XCTAssertThrowsError(try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "x")) { error in
            guard case ModelConfigDecodeError.missingField = error else {
                return XCTFail("expected missingField, got \(error)")
            }
        }
    }

    // MARK: - honest weight sizing (config.json present, shards partial/absent — the phantom-fit bug)

    private let qwen3Config = """
    { "model_type": "qwen3", "num_hidden_layers": 48, "num_attention_heads": 40,
      "num_key_value_heads": 4, "head_dim": 128, "max_position_embeddings": 262144 }
    """

    /// No `*.safetensors` shards on disk, but a `model.safetensors.index.json` declares the full
    /// checkpoint size — a metadata-only prefetch / interrupted `snapshot_download`. The declared
    /// size must be honored (not zero), but is NOT a measured fact.
    func testDeclaredWeightsFromIndex_whenNoShards() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-idxonly-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try Data(qwen3Config.utf8).write(to: dir.appendingPathComponent("config.json"))
        let index = """
        { "metadata": { "total_size": 5000000000 }, "weight_map": {} }
        """
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, 5_000_000_000)
        XCTAssertFalse(parsed.weightsAreMeasured)
        XCTAssertTrue(parsed.weightsAreDeclared, "size came from the index, not on-disk shards")
    }

    /// A complete download: on-disk shards sum MORE than the index's declared total (shard files
    /// include their own header overhead) → the real on-disk shard bytes win and count as measured.
    func testShardBytesWinOverIndex_completeDownload() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-complete-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try Data(qwen3Config.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data(count: 1000).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data(count: 500).write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        let index = """
        { "metadata": { "total_size": 1400 }, "weight_map": {} }
        """
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, 1500, "on-disk shards win over the declared total")
        XCTAssertTrue(parsed.weightsAreMeasured)
        XCTAssertFalse(parsed.weightsAreDeclared, "measured shards win — not a declared size")
    }

    /// A partial download: only one shard landed (700 bytes), but the index declares a larger total
    /// (1450) → the declared (larger, more conservative) size wins and is NOT measured.
    func testPartialShards_declaredWins() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-partial-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try Data(qwen3Config.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data(count: 700).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        let index = """
        { "metadata": { "total_size": 1450 }, "weight_map": {} }
        """
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, 1450, "declared full size wins over the partial on-disk shard")
        XCTAssertFalse(parsed.weightsAreMeasured)
        XCTAssertTrue(parsed.weightsAreDeclared, "partial download → declared index size, not measured")
    }

    /// No shards, no index at all — nothing to honestly size the weights from — must fail closed.
    func testNoShardsNoIndex_throwsWeightsUnknown() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-nothing-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try Data(qwen3Config.utf8).write(to: dir.appendingPathComponent("config.json"))

        XCTAssertThrowsError(try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")) { error in
            guard case ModelConfigDecodeError.weightsUnknown = error else {
                return XCTFail("expected weightsUnknown, got \(error)")
            }
        }
    }

    /// An index.json exists but has no `metadata.total_size` and there are no shards — the declared
    /// helper must return nil (not crash, not treat some other field as size) → fails closed.
    func testMalformedIndexMetadata_throwsWeightsUnknown() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitcheck-malformed-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try Data(qwen3Config.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data("""
        { "weight_map": {} }
        """.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        XCTAssertThrowsError(try ModelConfigDecoder.decodeModelDirectory(dir, id: "fixture")) { error in
            guard case ModelConfigDecodeError.weightsUnknown = error else {
                return XCTFail("expected weightsUnknown, got \(error)")
            }
        }
    }

    // MARK: - unservable-checkpoint classification (which decode failures must fail the serve closed)

    /// The serve path skips the fit-check on a decode error and proceeds with provided limits — the
    /// right call for an *unmodeled arch* (weights are fine, the KV formula just isn't audited yet).
    /// But `.weightsUnknown` means there are NO loadable weights on disk (interrupted/metadata-only
    /// download): the MLXLMCommon loader needs `*.safetensors`, so proceeding drives to a guaranteed
    /// load failure. This predicate lets the serve path fail closed on that case only, so the
    /// differentiator's fail-closed promise covers an incomplete checkpoint rather than crashing at load.
    func testIndicatesUnservableCheckpoint_forUnservableCases() {
        // A directory the MLXLMCommon loader itself cannot load — no *.safetensors, no config.json,
        // or a config.json that is not valid JSON — is unservable regardless of arch. The fit-check
        // must fail closed on these rather than skip-and-proceed to a guaranteed load-time crash.
        XCTAssertTrue(ModelConfigDecodeError.weightsUnknown("/x").indicatesUnservableCheckpoint,
            "no weights on disk → the load is doomed → the fit-check must fail closed, not skip")
        XCTAssertTrue(ModelConfigDecodeError.missingConfigFile("/x").indicatesUnservableCheckpoint,
            "no config.json → the MLX loader needs one → unservable → fail closed")
        XCTAssertTrue(ModelConfigDecodeError.malformedJSON.indicatesUnservableCheckpoint,
            "config.json is not valid JSON → the loader cannot parse it → unservable → fail closed")
        // A geometry field the DECODER rejects may still load: MLX reads its own config fields
        // independently, so a checkpoint rejected for a term this decoder happens to require MIGHT
        // still load. Keep these on skip-and-proceed until a live smoke proves otherwise. Likewise an
        // unmodeled arch has its weights on disk and can load — the skip path exists for it.
        XCTAssertFalse(ModelConfigDecodeError.unsupportedModelType("afmoe").indicatesUnservableCheckpoint)
        XCTAssertFalse(ModelConfigDecodeError.missingField("head_dim").indicatesUnservableCheckpoint)
        XCTAssertFalse(ModelConfigDecodeError.invalidField("num_key_value_heads").indicatesUnservableCheckpoint)
    }

    /// The refusal `reason=` token is case-appropriate and kept in lockstep with
    /// `indicatesUnservableCheckpoint`: exactly the fail-closed cases return a distinct token, and the
    /// skip cases return `nil`. Distinct tokens keep the serve-path refusal honest — an interrupted
    /// download is not the same failure mode as a missing or unparseable config.
    func testUnservableRefusalReason_distinctTokensForFailClosedCasesOnly() {
        XCTAssertEqual(ModelConfigDecodeError.weightsUnknown("/x").unservableRefusalReason, "incomplete_checkpoint")
        XCTAssertEqual(ModelConfigDecodeError.missingConfigFile("/x").unservableRefusalReason, "missing_config")
        XCTAssertEqual(ModelConfigDecodeError.malformedJSON.unservableRefusalReason, "malformed_config")
        XCTAssertNil(ModelConfigDecodeError.missingField("head_dim").unservableRefusalReason)
        XCTAssertNil(ModelConfigDecodeError.invalidField("num_key_value_heads").unservableRefusalReason)
        XCTAssertNil(ModelConfigDecodeError.unsupportedModelType("afmoe").unservableRefusalReason)
        // Lockstep invariant: a token exists iff the case fails closed.
        let cases: [ModelConfigDecodeError] = [
            .weightsUnknown("/x"), .missingConfigFile("/x"), .malformedJSON,
            .missingField("f"), .invalidField("f"), .unsupportedModelType("t"),
        ]
        for c in cases {
            XCTAssertEqual(c.unservableRefusalReason != nil, c.indicatesUnservableCheckpoint,
                "token presence must match the fail-closed predicate for \(c)")
        }
    }

    // MARK: - mistral3: the first CONFIG-SHAPE-DEPENDENT family

    /// mistral3 with an interleaved `layer_types` (sliding + full) AND a `sliding_window` classifies
    /// interleaved-SWA — matching the vendored `Mistral3Text.newCache`, which gives sliding layers a
    /// capped `RotatingKVCache` only when the window is present. nAttnLayers = the full_attention
    /// count (the layers that actually grow with full context); the rest are window-capped local.
    /// Shape modeled on Mistral-Small-3.1: 40 layers, a 5-sliding / 1-full interleave (here 32 full).
    func testMistral3_interleavedLayerTypes_withWindow_interleavedSWA() throws {
        var layerTypes: [String] = []
        for i in 0..<40 { layerTypes.append(i % 5 == 4 ? "full_attention" : "sliding_attention") }
        let layerTypesJSON = "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "mistral3",
          "num_hidden_layers": 40,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "sliding_window": 4096,
          "max_position_embeddings": 131072,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "mistral-small-3.1")
        XCTAssertEqual(parsed.profile.modelType, .interleavedSWA)
        XCTAssertEqual(parsed.profile.nLayers, 40)
        XCTAssertEqual(parsed.profile.nAttnLayers, 8, "8 full_attention entries grow with full context")
        XCTAssertEqual(parsed.profile.nKVHeads, 8)
        XCTAssertEqual(parsed.profile.headDim, 128)
        XCTAssertEqual(parsed.profile.slidingWindow, 4096)
        XCTAssertTrue(parsed.profile.isKVDerivable)
    }

    /// mistral3 with NO `layer_types` (a dense Mistral-Small-3.2-style config) classifies
    /// uniform-GQA: the vendored arch defaults every layer to `full_attention` → a growing
    /// `KVCacheSimple` on every layer, so every layer grows (nAttnLayers == nLayers).
    func testMistral3_noLayerTypes_uniformGQA() throws {
        let json = """
        {
          "model_type": "mistral3",
          "num_hidden_layers": 40,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "max_position_embeddings": 131072
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "mistral-small-3.2")
        XCTAssertEqual(parsed.profile.modelType, .uniformGQA)
        XCTAssertEqual(parsed.profile.nAttnLayers, 40, "no layer_types → all full_attention → every layer grows")
        XCTAssertNil(parsed.profile.slidingWindow)
        XCTAssertTrue(parsed.profile.isKVDerivable)
    }

    /// Subtle honest case: mistral3 declaring `sliding_attention` layers but OMITTING
    /// `sliding_window` classifies uniform-GQA, NOT interleaved-SWA. The vendored arch's cache
    /// allocation is `if layer.useSliding, let slidingWindow = args.slidingWindow { RotatingKVCache }
    /// else { KVCacheSimple }` — with no window it falls back to a growing `KVCacheSimple` on every
    /// layer. Classifying uniform-GQA (every layer grows) matches that fallback and fails toward RED
    /// (over-counts KV), never phantom-GREEN by modeling a window-cap the runtime won't apply.
    func testMistral3_slidingDeclaredButNoWindow_uniformGQA() throws {
        let layerTypesJSON = "[" + Array(repeating: "\"sliding_attention\"", count: 40).joined(separator: ",") + "]"
        let json = """
        {
          "model_type": "mistral3",
          "num_hidden_layers": 40,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "max_position_embeddings": 131072,
          "layer_types": \(layerTypesJSON)
        }
        """
        let parsed = try ModelConfigDecoder.decode(configJSON: data(json), safetensorsBytes: 1, id: "mistral3-windowless")
        XCTAssertEqual(parsed.profile.modelType, .uniformGQA, "no sliding_window → arch grows every layer")
        XCTAssertEqual(parsed.profile.nAttnLayers, 40)
        XCTAssertNil(parsed.profile.slidingWindow)
    }
}
