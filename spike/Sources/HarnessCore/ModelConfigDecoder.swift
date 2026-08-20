import Foundation

/// A `ModelArchProfile` decoded from a real on-disk model, paired with provenance flags the
/// fit-check surfaces (spec §5: never present an estimate as a measured fact).
public struct ParsedModelArch: Sendable {
    public let profile: ModelArchProfile
    /// `true` when `profile.weightsBytes4bitEstimate` was set from summed real safetensors bytes
    /// (not a param-count estimate). Pairs with `SystemProfile.wiredLimitIsMeasured` at the verdict
    /// surface so weights-vs-headroom can be labelled measured/modeled honestly.
    public let weightsAreMeasured: Bool
    /// `true` when `weightsBytes4bitEstimate` came from the checkpoint's own
    /// `model.safetensors.index.json` (`metadata.total_size`) rather than summed on-disk shards —
    /// a *declared* size (honest, checkpoint-authored, but not verified against bytes on disk). Lets
    /// the fit-check announce say "(declared)" instead of conflating it with a param-count
    /// "(estimated)" guess. Mutually exclusive with `weightsAreMeasured` (measured wins). Always
    /// `false` on the pure `decode(...)` path — only `decodeModelDirectory` can observe an index.
    public let weightsAreDeclared: Bool
    /// The checkpoint's own declared weight quantization width (`quantization.bits` /
    /// `quantization_config.bits`), or `nil` when the config carries no quantization block
    /// (an unquantized checkpoint). Drives the quant auto-picker's ranking (differentiator #2 full
    /// shape): candidates are compared by the bits the checkpoint actually ships, not a repo-name
    /// guess; `nil` ranks as highest quality (unquantized).
    public let quantBits: Int?

    public init(
        profile: ModelArchProfile, weightsAreMeasured: Bool,
        weightsAreDeclared: Bool = false, quantBits: Int? = nil
    ) {
        self.profile = profile
        self.weightsAreMeasured = weightsAreMeasured
        self.weightsAreDeclared = weightsAreDeclared
        self.quantBits = quantBits
    }
}

public enum ModelConfigDecodeError: Error, Equatable, CustomStringConvertible {
    case malformedJSON
    case missingConfigFile(String)
    case missingField(String)
    /// The `model_type` is not one the fit-check can model yet. Callers must fail closed (refuse) or
    /// skip the fit-check and warn — never silently mis-model it as uniform-GQA (which would
    /// under-count a hybrid/SWA/MLA KV footprint and let an unservable config look like it fits).
    case unsupportedModelType(String)
    /// A required geometry field is present but not a usable positive integer (e.g.
    /// `num_key_value_heads: 0`, a hand-edited/corrupt config, or a JSON number outside `Int`'s
    /// exactly-representable range). Distinct from `.missingField` (absent) so the surface can say
    /// "malformed" vs "missing". Fails closed: a non-positive geometry term would zero the KV rate
    /// and let an unservable config classify GREEN — the same phantom-fit hole `.weightsUnknown`
    /// closes for weights, on the geometry side.
    case invalidField(String)
    /// A checkpoint directory has config.json but no way to honestly size its weights: no
    /// `*.safetensors` shards on disk AND no readable `model.safetensors.index.json` declaring
    /// `metadata.total_size`. Sizing this as 0 would let an arbitrarily large phantom checkpoint
    /// (an interrupted/partial `snapshot_download`, or a metadata-only prefetch) classify GREEN.
    case weightsUnknown(String)

    public var description: String {
        switch self {
        case .malformedJSON: return "config.json is not valid JSON"
        case .missingConfigFile(let path): return "no config.json at \(path)"
        case .missingField(let f): return "config.json is missing required field '\(f)'"
        case .unsupportedModelType(let t): return "model_type '\(t)' is not fit-checkable yet (arch KV formula unknown)"
        case .invalidField(let f): return "config.json field '\(f)' is present but not a usable positive integer"
        case .weightsUnknown(let path):
            return "no weights found for '\(path)': directory has config.json but no *.safetensors shards and no readable model.safetensors.index.json (metadata.total_size) — finish the download before serving"
        }
    }

    /// True when the error means the checkpoint itself cannot be served, so the serve-path fit-check
    /// must FAIL CLOSED rather than skip-and-proceed. Three cases qualify — each names a directory the
    /// MLXLMCommon loader itself could never load, so skipping the fit-check only defers a guaranteed
    /// load-time crash to a worse place:
    ///   - `.weightsUnknown` — no `*.safetensors` on disk and no readable index; the loader requires
    ///     shards (interrupted / metadata-only download).
    ///   - `.missingConfigFile` — no `config.json`; the loader needs one to build the model.
    ///   - `.malformedJSON` — `config.json` is present but not valid JSON; the loader cannot parse it.
    /// `.unsupportedModelType` is deliberately excluded — the arch is merely not fit-checkable yet; its
    /// weights are present and the load can still succeed (the skip path exists exactly to not regress
    /// those). `.missingField`/`.invalidField` also stay on skip: MLX reads its own config fields
    /// independently of this decoder, so a checkpoint rejected here for a geometry term it happens to
    /// require MIGHT still load — kept on skip until a live smoke against a real offending checkpoint
    /// proves otherwise. Taxonomy + rationale:
    /// docs/task-inbox/2026-08-19-fit-check-skip-vs-failclosed-taxonomy.md.
    public var indicatesUnservableCheckpoint: Bool {
        switch self {
        case .weightsUnknown, .missingConfigFile, .malformedJSON: return true
        case .missingField, .invalidField, .unsupportedModelType: return false
        }
    }

    /// A stable, case-appropriate machine-readable `reason=` token for the serve-path refusal, or
    /// `nil` for the cases that stay on skip-and-proceed (kept in lockstep with
    /// `indicatesUnservableCheckpoint`: exactly the `true` cases return a token). Distinct tokens keep
    /// the refusal honest — an interrupted download is not the same failure as a missing or unparseable
    /// config — while the human-readable `description` line carries the detail.
    public var unservableRefusalReason: String? {
        switch self {
        case .weightsUnknown: return "incomplete_checkpoint"
        case .missingConfigFile: return "missing_config"
        case .malformedJSON: return "malformed_config"
        case .missingField, .invalidField, .unsupportedModelType: return nil
        }
    }
}

/// Parses a Hugging Face `config.json` into the sizer's `ModelArchProfile` so the serving path can
/// fit-check a *real on-disk checkpoint* against the host (fit-checked-serve, differentiator #2)
/// instead of only the hand-audited `ModelArchProfile.catalog`. Pure and MLX-free so the whole
/// parser is unit-tested off-box; the thin `decodeModelDirectory` wrapper adds the filesystem read
/// (config.json + summed `*.safetensors` bytes) that the live serve path uses.
///
/// **Fail-closed by design.** Only architectures whose KV growth formula this repo has audited
/// (`uniform-GQA` dense families; `hybrid-linear` qwen3_next/qwen3_5) are classified; every other
/// `model_type` throws `.unsupportedModelType` rather than defaulting to a formula that would
/// under-count its footprint.
public enum ModelConfigDecoder {

    /// Explicit allow-lists — deliberately narrow. Adding an entry is a claim that
    /// `CapacityModel`'s formula for that `ArchClass` is correct for the family.
    private static let uniformGQATypes: Set<String> = [
        "qwen3", "qwen3_moe", "qwen2", "qwen2_moe", "llama", "mistral", "mixtral",
        "glm4", "glm4_moe", "phi3", "phi3_5", "phi", "phimoe",
        // cohere, gemma2, starcoder2, olmo2, granite, internlm2, minicpm: audited the vendored
        // MLX-Swift arches directly — none of these has a `newCache` override, so all fall through
        // to the default `KVCacheDimensionProvider` (every layer allocates a plain growing
        // `KVCacheSimple`), and all read a SCALAR `num_key_value_heads` + `head_dim` (no per-layer
        // varying kvHeads array, unlike openelm, which stays excluded). uniformGQA's
        // nKVHeads × head_dim × nAttnLayers formula (all layers grow) is correct as-implemented.
        //
        // gemma2 special case: HF's gemma2 is conceptually interleaved-SWA (sliding local layers
        // alternating with global attention), but the vendored `Gemma2.swift` has no
        // `RotatingKVCache` and no `newCache` override — as-implemented it grows a full KV cache on
        // EVERY layer. uniformGQA is therefore the honest as-implemented model for gemma2 here, and
        // it OVER-counts vs. a hypothetical sliding-window implementation — conservative/safe: it
        // fails toward RED (too-small-host false negative), never a phantom-GREEN under-count.
        "cohere", "gemma2", "starcoder2", "olmo2", "granite", "internlm2", "minicpm",
        // Set-C: MoE routers with uniform DENSE attention. Audited each vendored arch — none has a
        // `newCache` override (all grow a plain `KVCacheSimple` on every layer) and all read a SCALAR
        // `num_key_value_heads`, so the uniformGQA every-layer formula holds:
        //  - bailing_moe (BailingMoe.swift): head_dim is hard-computed `hidden_size / heads`; agrees
        //    with the decoder's derived head_dim (fails toward the honest hidden/heads when no
        //    `head_dim` field is present, the on-disk norm for this family).
        //  - olmoe (OlmoE.swift): per-layer kvHeads is a uniform repeat of the scalar; the arch's
        //    absent-KV MHA fallback is intentionally NOT mirrored — the decoder fails closed instead.
        //  - ernie4_5 (Ernie4_5.swift): `head_dim ?? dim/num_attention_heads` matches the resolver.
        "bailing_moe", "olmoe", "ernie4_5",
        // Set-D: three more DENSE families, each audited in the vendored MLX-Swift arch to have NO
        // `newCache` override (every layer grows a plain `KVCacheSimple`) and a SCALAR
        // `num_key_value_heads`, so the every-layer formula is exact:
        //  - gemma (Gemma.swift, the gemma1 base): explicit `head_dim`, scalar kvHeads, no sliding
        //    window — same as gemma2 above with no window; plain growing KV every layer.
        //  - acereason (backed by Qwen2.swift in LLMModelFactory): a plain Qwen2 attention arch
        //    (head_dim = hidden/heads, scalar kvHeads); identical geometry to qwen2 above.
        //  - smollm3 (SmolLM3.swift): `head_dim ?? hidden/heads`, scalar kvHeads; its `no_rope_layers`
        //    only disable rotary on some layers (positional) — KV cache is uniform growing every layer,
        //    unchanged in bytes. Each was a fit-check GAP (fit_check=skipped → flat serve.sh limits).
        "gemma", "acereason", "smollm3",
        // Set-E: three more DENSE families, each audited in the vendored arch to have NO `newCache`
        // override (plain growing `KVCacheSimple` every layer) and a SCALAR `num_key_value_heads`:
        //  - bitnet (Bitnet.swift): `head_dim ?? hidden/heads`, `num_key_value_heads ?? num_attention_heads`
        //    (MHA fallback — but the decoder fails CLOSED on absence, like olmoe, rather than mirror the
        //    guess). BitNet's ternary WEIGHTS don't change KV geometry — the cache is standard K/V.
        //  - nanochat (NanoChat.swift): head_dim = hidden/heads, scalar kvHeads.
        //  - lille-130m (Lille130m.swift): head_dim = hidden/heads, scalar kvHeads (tiny 130M model,
        //    trivially servable). Each was a fit-check GAP (fit_check=skipped → flat serve.sh limits).
        "bitnet", "nanochat", "lille-130m",
        // Set-F: two more DENSE families, each audited in the vendored arch to have NO `newCache`
        // override (plain growing `KVCacheSimple` every layer) and a SCALAR `num_key_value_heads`,
        // with head_dim COMPUTED as hidden_size / num_attention_heads (neither reads an explicit
        // `head_dim` field — the decoder's hidden/heads fallback matches the arch exactly):
        //  - mimo (MiMo.swift): `headDim = hiddenSize / heads` (:33), scalar `kvHeads` (:31). Its MTP
        //    draft layers (`num_nextn_predict_layers`) are NOT in the main transformer cache — the
        //    per-layer kvHeads array spans `0..<num_hidden_layers` only (:170) — so the every-layer
        //    formula over num_hidden_layers is exact and does not over-count the draft head.
        //  - apertus (Apertus.swift): `headDim = hiddenSize / heads` (:177), scalar `numKeyValueHeads`
        //    (:176). Its QK-norm (RMSNorm on headDim, :186-187) and xIELU activation don't change
        //    stored K/V bytes. Each was a fit-check GAP (fit_check=skipped → flat serve.sh limits).
        "mimo", "apertus",
    ]
    /// lfm2 (Liquid LFM2, e.g. LiquidAI/LFM2-1.2B): hybrid-linear like qwen3_next — the vendored
    /// `LFM2.swift` `newCache` (LFM2.swift:401-408) grows a `KVCacheSimple` on the `full_attn_idxs`
    /// layers and a conv-only `MambaCache` on the rest — but with two family-specific config shapes
    /// handled below: attention layers are enumerated as INDICES in `full_attn_idxs` (folded into
    /// `resolveHybridAttnLayers`), and the recurrent state is a simple depthwise conv cache, not
    /// GatedDeltaNet SSM state (`resolveLfm2ConvStateBytes`, gated by `isLfm2`).
    private static let hybridLinearTypes: Set<String> = [
        // qwen3_5_text: the text-only config variant of the Qwen3.5 family (root `model_type` is
        // `qwen3_5_text` rather than the VL wrapper's `qwen3_5`). The vendored `Qwen35.swift newCache`
        // (:582-589) is identical across the siblings — MambaCache on gated-delta-net (linear) layers,
        // KVCacheSimple on the `full_attention_interval` attention layers — so it reuses the SAME default
        // GatedDeltaNet hybrid-linear path (no family flag; the state resolver keys on the `linear_*`
        // fields). Was a fit-check GAP (fit_check=skipped → flat serve.sh limits).
        "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "lfm2",
        // lfm2_moe (Liquid LFM2-MoE, e.g. LiquidAI/LFM2-8B-A1B): the MoE sibling of lfm2 — the vendored
        // `LFM2MoE.swift` `newCache` (LFM2MoE.swift:495-503) is the SAME cache shape as lfm2 (KVCacheSimple
        // on attention layers, conv-only `MambaCache` = `zeros([B, conv_L_cache − 1, hidden])` on the rest),
        // so it reuses `.hybridLinear` + the `isLfm2` conv-state path. Its only config-shape difference from
        // lfm2 is that attention layers are listed in `layer_types` (`"full_attention"`) rather than an
        // explicit `full_attn_idxs` index array (`resolveHybridAttnLayers` reads `layer_types` first). The
        // MoE router (`num_experts`, `moe_intermediate_size`) is orthogonal to KV/conv geometry.
        "lfm2_moe",
        // jamba (AI21 Jamba): the vendored `Jamba.swift` `newCache` (Jamba.swift:479-487) grows a plain
        // `KVCacheSimple` on `layer.isAttn` layers and a 2-slot `MambaCache` on the rest — the same
        // attention-grows + non-attention-fixed shape as lfm2, so it reuses `.hybridLinear`. Its recurrent
        // term is a CLASSIC Mamba conv+SSM state (NOT lfm2's conv-only, NOT qwen3_5's GatedDeltaNet), sized
        // by a dedicated `resolveJambaStateBytes` gated on `isJamba`. Attention layers are enumerated by
        // `attn_layer_offset`/`attn_layer_period` (no `layer_types`/`full_attn_idxs` on Jamba-v0.1), folded
        // into `resolveHybridAttnLayers`.
        "jamba",
        // falcon_h1 (tiiuae/Falcon-H1-*): a PARALLEL hybrid, unlike the interval-select families above.
        // The vendored `FalconH1.swift` `newCache` (:799-801) gives EVERY layer a
        // `CacheList(MambaCache(), attentionCache.copy())`, so all layers BOTH grow an attention KV
        // cache AND hold a Mamba-2 conv+SSM recurrent state. It reuses `.hybridLinear` but is handled
        // as its own shape in `decode` (`isFalconH1`): `nAttnLayers == nLayers` (every layer grows KV,
        // not a resolved subset) and the fixed state is sized over ALL layers by
        // `resolveFalconH1StateBytes` (not `nLayers − nAttnLayers`, which is 0 here).
        "falcon_h1",
        // granitemoehybrid (IBM Granite 4.0-H, e.g. ibm-granite/granite-4.0-h-micro): an INTERVAL-SELECT
        // hybrid (unlike falcon_h1's parallel shape) — the vendored `GraniteMoeHybrid.swift` `newCache`
        // (:520-525) grows a `KVCacheSimple` on `layer_types == "attention"` layers and a `MambaCache` on
        // the `"mamba"` layers, so the recurrent state lives on `nLayers − nAttnLayers` layers. Its
        // recurrent term is a CLASSIC Mamba-2 conv+SSM state sized by `resolveGraniteMoeHybridStateBytes`
        // (gated on `isGraniteMoeHybrid`), distinct from lfm2's conv-only cache and jamba's ×4 SSM width.
        // Its attention layers are enumerated in `layer_types` as `"attention"` (not `"full_attention"`),
        // which `resolveHybridAttnLayers` now also counts.
        "granitemoehybrid",
    ]
    /// gpt_oss (gpt-oss-20b/120b): audited the vendored `GPTOSS.swift` cache allocation directly
    /// (Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GPTOSS.swift lines 515-524) — `full_attention`
    /// layers get `StandardKVCache` (grows with context), other layers get
    /// `RotatingKVCache(maxSize: slidingWindow)` (capped at the window). Same interleaved-SWA cache
    /// shape as Gemma-3, driven by the same `layer_types`/`sliding_window` config fields, so it
    /// reuses this class rather than needing a new `ArchClass`. Weights are the on-disk MXFP4 bytes
    /// as measured from safetensors — no runtime-upcast claim is made here.
    ///
    /// olmo3: audited the vendored `Olmo3.swift` `newCache` (Vendor/mlx-swift-lm/Libraries/MLXLLM/
    /// Models/Olmo3.swift:226-234) — `full_attention` layers get `KVCacheSimple()` (grows with
    /// context), the rest get `RotatingKVCache(maxSize: slidingWindow)` (capped at the window). Same
    /// interleaved-SWA cache shape as Gemma-3/gpt-oss, driven by the same `layer_types`/
    /// `sliding_window` fields (Olmo3.swift:265-273) and a scalar `num_key_value_heads`/`head_dim`,
    /// so it reuses this class rather than needing a new `ArchClass`.
    ///
    /// afmoe (Arcee AfmoeForCausalLM, e.g. arcee-ai/Trinity-Nano-Base): audited the vendored
    /// `AfMoE.swift` `newCache` (AfMoE.swift:560-569) — `sliding_attention` layers get
    /// `RotatingKVCache(maxSize: slidingWindow)` (capped at the window), the rest get `KVCacheSimple()`
    /// (grows with context), driven by the same `layer_types`/`sliding_window` fields
    /// (`layerUsesSliding = layerTypes.map { $0 == "sliding_attention" }`, AfMoE.swift:510) and a scalar
    /// `num_key_value_heads`/`head_dim`. The cache holds ONLY standard K/V — no altup/laurel/mamba/conv
    /// per-layer state — so the interleaved-SWA formula (growing KV on full layers + a fixed window-cap
    /// on sliding layers) is exact as-implemented. Verified against the real published config on disk
    /// (14 full_attention of 56, window 2048); the MoE router is orthogonal to KV geometry.
    ///
    /// gemma3n (Gemma-3n E2B/E4B): interleaved-SWA like Gemma-3, but the vendored `Gemma3nText.swift`
    /// `newCache` (Gemma3nText.swift:677-693) allocates caches for ONLY the first
    /// `num_hidden_layers − num_kv_shared_layers` layers — the shared tail reuses earlier layers' KV
    /// and allocates nothing. It is therefore a config-SHAPE branch (see the `isGemma3n` handling in
    /// `decode`), not a pure allow-list add: the KV formula runs over the CACHED PREFIX only
    /// (`nLayers`/`layer_types` truncated to `num_hidden_layers − num_kv_shared_layers`), else it would
    /// over-count both growing and window-capped layers by the shared tail. `newCache` allocates only
    /// `StandardKVCache`/`RotatingKVCache` (no altup/laurel/per-layer-input state), so `fixedStateBytes`
    /// stays 0. Verified against the real published `mlx-community/gemma-3n-E2B-it-lm-4bit` config.
    private static let interleavedSWATypes: Set<String> = [
        "gemma3", "gemma3_text", "gpt_oss", "olmo3", "afmoe", "gemma3n", "gemma3n_text",
    ]
    /// MLA-as-implemented: the vendored MLX-Swift DeepSeek-V3 arch (`deepseek_v3`, which R1 also
    /// tags) decompresses the MLA latent back to per-head K/V before the cache write, so the cache
    /// holds full decompressed geometry (spec §8 OOM case). Narrow on purpose — `deepseek_v2`'s
    /// factory registration is unconfirmed, so it stays out until verified rather than borrowing
    /// this formula on an unaudited family.
    private static let mlaTypes: Set<String> = [
        "deepseek_v3",
    ]
    /// Dual-geometry SWA: global (growing) and SWA (window-capped) attention layers carry DIFFERENT
    /// head geometry, and every layer also holds a `MambaCache` conv state. `baichuan_m1`
    /// (mlx-community/Baichuan-M1-14B-Instruct-4bit): the vendored `BaichuanM1.swift newCache`
    /// (:265-273) gives every layer a `CacheList(MambaCache(), kvCache)` where `kvCache` is a
    /// `RotatingKVCache(maxSize: sliding_window)` on the `sliding_window_layers` and a
    /// `KVCacheSimple()` elsewhere; the SWA and global layers read different `num_swa_*`/`num_*`
    /// head counts (:70-77). Neither `.interleavedSWA` (single geometry, drops the conv state) nor
    /// `.hybridLinear` (drops the window cap) can express this — it needs the dedicated formula.
    /// See docs/task-inbox/2026-08-19-baichuan-m1-dual-geometry-arch-reach.md for the source audit.
    ///
    /// mimo_v2_flash (XiaomiMiMo/MiMo-V2-Flash): also dual-geometry SWA, but with ASYMMETRIC K/V —
    /// the vendored `MiMoV2Flash.swift` caches K at `head_dim` (192) and V at a narrower `v_head_dim`
    /// (128) on both layer classes (projections :141-146, cache write :49, `RotatingKVCache` holds K/V
    /// as independent arrays). It has NO conv/recurrent state (`fixedStateBytes = 0`), enumerates its
    /// SWA layers via `hybrid_layer_pattern` (1 = SWA, 0 = global; :353-358), reads `sliding_window_size`
    /// (:516), and its four head-dim fields are non-optional (fail-closed, no `??` fallback). Handled as
    /// its own shape below (`isMimoV2Flash`). See
    /// docs/task-inbox/2026-08-19-mimo-v2-flash-dual-geometry-vheaddim.md for the source + config audit.
    ///
    /// gemma4 / gemma4_unified / gemma4_text (Google Gemma-4, e.g. mlx-community/gemma-4-e2b-it-4bit):
    /// dual-geometry SWA where the two geometries differ in HEAD DIM — global/full layers cache at
    /// `global_head_dim` (512), sliding layers at `head_dim` (256) (vendored `Gemma4Text.swift:228-230`;
    /// `newCache` :755-767 gives full layers a growing `StandardKVCache`, sliding layers a
    /// `RotatingKVCache(sliding_window)`). It ALSO shares KV across layers like gemma3n: only the first
    /// `num_hidden_layers − num_kv_shared_layers` layers own a cache (`newCache` loops that prefix; the
    /// shared tail allocates nothing), so `nLayers` is truncated to that prefix (as for gemma3n) and the
    /// global/SWA split is counted within it. No conv/recurrent state (`fixedStateBytes = 0`); the
    /// `attention_k_eq_v` tie is weight-level only (V is still stored — forward :318-343 — so KV bytes
    /// are unchanged), but the flag gates the global KV-head predicate. Handled as its own shape below
    /// (`isGemma4`): global geometry reads `global_head_dim` + the k_eq_v KV-head predicate, SWA geometry
    /// reads `head_dim` + `num_key_value_heads`. See
    /// docs/task-inbox/2026-08-19-gemma4-dual-geometry-kv-shared-arch-reach.md for the source + config audit.
    private static let dualGeometrySWATypes: Set<String> = [
        "baichuan_m1", "mimo_v2_flash", "gemma4", "gemma4_unified", "gemma4_text",
    ]

    /// Pure core: decode a raw `config.json` payload plus a known summed safetensors byte total.
    /// `safetensorsBytes == 0` means "unknown" → `weightsAreMeasured == false`.
    public static func decode(
        configJSON: Data, safetensorsBytes: Int, id: String, license: String = "unknown (from config)"
    ) throws -> ParsedModelArch {
        guard let root = (try? JSONSerialization.jsonObject(with: configJSON)) as? [String: Any] else {
            throw ModelConfigDecodeError.malformedJSON
        }
        // VL / multimodal checkpoints nest the text decoder's geometry under `text_config`; dense
        // text models keep it at the root. Read geometry from the text scope, falling back to root.
        let textScope = (root["text_config"] as? [String: Any]) ?? [:]
        func geom(_ key: String) -> Any? { textScope[key] ?? root[key] }

        // model_type: prefer the root (the authoritative family tag, e.g. "qwen3_5" on a VL wrapper),
        // fall back to the text scope.
        guard let rawModelType = (root["model_type"] as? String) ?? (textScope["model_type"] as? String) else {
            throw ModelConfigDecodeError.missingField("model_type")
        }
        let modelType = rawModelType.lowercased()
        // gemma3n's KV cost depends on config SHAPE (its shared-layer tail allocates no cache), so it
        // needs the cached-prefix handling below rather than the plain interleaved-SWA field logic.
        let isGemma3n = (modelType == "gemma3n" || modelType == "gemma3n_text")
        // lfm2's fixed recurrent state is a depthwise conv cache, not GatedDeltaNet SSM state — a
        // distinct formula from the other hybrid-linear families (see `resolveLfm2ConvStateBytes`).
        // lfm2_moe shares lfm2's conv-only cache exactly (LFM2MoE.swift:495-503), so it takes the same path.
        let isLfm2 = (modelType == "lfm2" || modelType == "lfm2_moe")
        let isJamba = (modelType == "jamba")
        // falcon_h1 is a PARALLEL hybrid (every layer grows KV AND holds Mamba-2 state), so it takes a
        // distinct path from the interval-select hybrids: nAttnLayers == nLayers and a fixed-state term
        // sized over ALL layers (see `resolveFalconH1StateBytes`).
        let isFalconH1 = (modelType == "falcon_h1")
        // granitemoehybrid is an interval-select hybrid whose recurrent layers hold a classic Mamba-2
        // conv+SSM state (see `resolveGraniteMoeHybridStateBytes`); it enumerates attention layers in
        // `layer_types` as "attention".
        let isGraniteMoeHybrid = (modelType == "granitemoehybrid")
        // Both dual-geometry-SWA members dispatch on `archClass == .dualGeometrySWA` below, but their
        // config SHAPES differ (baichuan: `sliding_window_layers` + conv state + symmetric K/V + `??`
        // fallbacks; mimo: `hybrid_layer_pattern` + no conv state + asymmetric K/V `v_head_dim` +
        // non-optional geometry + `sliding_window_size`), so the `.dualGeometrySWA` cases branch on this.
        let isMimoV2Flash = (modelType == "mimo_v2_flash")
        // gemma4/gemma4_unified/gemma4_text: dual-geometry SWA (global cache dim `global_head_dim` vs
        // sliding `head_dim`) + KV-layer-sharing (only the first `num_hidden_layers − num_kv_shared_layers`
        // layers own a cache, like gemma3n). Its `.dualGeometrySWA` branches override the global head
        // geometry (config `head_dim` is the SLIDING dim, NOT the global slot) and count the global/SWA
        // split within the cached prefix. `gemma4`/`gemma4_unified` wrap the text tower in `text_config`
        // (the `geom()` text-scope-then-root fallback already unwraps it); `gemma4_text` is flat.
        let isGemma4 = (modelType == "gemma4" || modelType == "gemma4_unified" || modelType == "gemma4_text")

        let archClass: ArchClass
        if modelType == "mistral3" {
            // mistral3 is the first family whose KV class is not fixed by model_type alone — it
            // depends on config SHAPE. See `mistral3ArchClass`: the vendored Mistral3Text arch caps
            // a layer's cache only when it is `sliding_attention` AND a `sliding_window` is present;
            // otherwise every layer grows. Classify accordingly, then reuse the existing per-class
            // field logic (uniform-GQA nAttnLayers==nLayers, or SWA global-layer/window derivation).
            archClass = Self.mistral3ArchClass(geom: geom)
        } else if uniformGQATypes.contains(modelType) {
            archClass = .uniformGQA
        } else if hybridLinearTypes.contains(modelType) {
            archClass = .hybridLinear
        } else if interleavedSWATypes.contains(modelType) {
            archClass = .interleavedSWA
        } else if mlaTypes.contains(modelType) {
            archClass = .mlaAsImplemented
        } else if dualGeometrySWATypes.contains(modelType) {
            archClass = .dualGeometrySWA
        } else {
            throw ModelConfigDecodeError.unsupportedModelType(rawModelType)
        }

        // All classes: layer count and native context must be present AND positive. A 0/negative
        // (corrupt/hand-edited config, or an out-of-range JSON number `intOf` now rejects) would
        // either zero the KV total or make a non-servable config size as trivially-fits.
        // For gemma3n AND gemma4 the KV formula runs over the CACHED PREFIX only — the last
        // `num_kv_shared_layers` layers reuse earlier layers' caches and allocate nothing (vendored
        // `Gemma3nText.newCache` / `Gemma4Text.newCache` both loop `0 ..< numHiddenLayers −
        // numKvSharedLayers`). Truncate `nLayers` to that prefix so `nLocalLayers = nLayers − nAttnLayers`
        // counts only cached layers. Fail closed when `num_kv_shared_layers` is absent — never assume 0,
        // which would over-count the shared tail (phantom-RED).
        let nLayers: Int
        if isGemma3n || isGemma4 {
            let totalLayers = try requirePositiveInt(geom, "num_hidden_layers")
            guard let shared = intOf(geom("num_kv_shared_layers")), shared >= 0 else {
                throw ModelConfigDecodeError.missingField("num_kv_shared_layers")
            }
            let cached = totalLayers - shared
            guard cached > 0 else {
                throw ModelConfigDecodeError.invalidField("num_kv_shared_layers (>= num_hidden_layers leaves no cached layers)")
            }
            nLayers = cached
        } else {
            nLayers = try requirePositiveInt(geom, "num_hidden_layers")
        }
        let nativeMaxContext = try requirePositiveInt(geom, "max_position_embeddings")

        // Head geometry is class-specific. GQA/hybrid/SWA grow a per-head K/V cache sized by
        // num_key_value_heads × head_dim. MLA-as-implemented caches DECOMPRESSED per-head K/V, whose
        // rate comes from the four decompressed dims instead — `num_key_value_heads` on an MLA
        // checkpoint is a different, unrelated quantity, so reading it here would mis-size the cache.
        // The MLA dims fail closed on missing OR non-positive: `CapacityModel.kvBytesPerToken` reads
        // `mla* ?? 0`, so a nil/zero dim would collapse the KV rate to ~0 and phantom-GREEN the most
        // OOM-dangerous arch in the catalog (spec §8).
        let nKVHeads: Int
        let headDim: Int
        let mlaHeads: Int?, mlaRopeDim: Int?, mlaNopeDim: Int?, mlaVDim: Int?
        switch archClass {
        case .mlaAsImplemented:
            nKVHeads = 0
            headDim = 0
            mlaHeads = try requirePositiveInt(geom, "num_attention_heads")
            mlaRopeDim = try requirePositiveInt(geom, "qk_rope_head_dim")
            mlaNopeDim = try requirePositiveInt(geom, "qk_nope_head_dim")
            mlaVDim = try requirePositiveInt(geom, "v_head_dim")
        case .dualGeometrySWA where isGemma4:
            // gemma4 GLOBAL geometry. The trap: this profile slot `headDim` is the GLOBAL cache dim, but
            // gemma4's config field `head_dim` is the SLIDING dim — the generic `resolveHeadDim` (which
            // reads `head_dim` or `hidden/heads`) would put the sliding 256 (or 1536/8=192) in the global
            // slot and under-size global layers ~2× (phantom-GREEN). Read `global_head_dim` explicitly.
            // Global KV heads follow the vendored predicate (`Gemma4Text.swift:236-241`): the tie flag
            // `attention_k_eq_v` (full layers only) swaps to `num_global_key_value_heads` WHEN PRESENT,
            // else every layer uses `num_key_value_heads`. Naively coalescing `num_global_key_value_heads
            // ?? num_key_value_heads` is wrong when the field is present but the flag is false. The tie
            // itself is weight-level only (V is still cached — forward :318-343 — so KV bytes are
            // unchanged); the flag is read solely for this KV-head predicate.
            headDim = try requirePositiveInt(geom, "global_head_dim")
            let kEqV = (geom("attention_k_eq_v") as? Bool) ?? false
            let baseKV = try requirePositiveInt(geom, "num_key_value_heads")
            if kEqV, let g = intOf(geom("num_global_key_value_heads")) {
                guard g > 0 else {
                    throw ModelConfigDecodeError.invalidField("num_global_key_value_heads (gemma4 global KV heads)")
                }
                nKVHeads = g
            } else {
                nKVHeads = baseKV
            }
            mlaHeads = nil; mlaRopeDim = nil; mlaNopeDim = nil; mlaVDim = nil
        default:
            nKVHeads = try requirePositiveInt(geom, "num_key_value_heads")
            let hd = try resolveHeadDim(geom: geom)
            guard hd > 0 else { throw ModelConfigDecodeError.invalidField("head_dim") }
            headDim = hd
            mlaHeads = nil; mlaRopeDim = nil; mlaNopeDim = nil; mlaVDim = nil
        }

        // Dual-geometry SWA: the window-capped layers carry their OWN head geometry, distinct from the
        // global-layer `nKVHeads`/`headDim` resolved above. Under-counting this with the global
        // geometry is the 320 MiB/seq phantom-GREEN hazard the fit-check exists to prevent. The
        // `?? global` fallbacks MIRROR the vendored `BaichuanM1.swift:72-74` runtime allocation —
        // absence changes the actual cache allocation to the global counts, so falling back is honest
        // (unlike a sizer-only default). `hidden_size`/`num_attention_heads` fail closed (always
        // present in a real transformer config; needed to derive the per-head dim).
        let swaKVHeads: Int?
        let swaHeadDim: Int?
        let vHeadDim: Int?
        let swaVHeadDim: Int?
        switch archClass {
        case .dualGeometrySWA where isGemma4:
            // gemma4 SWA (sliding) geometry: `head_dim` (256, the config field the global slot above
            // deliberately did NOT read) and `num_key_value_heads`. Symmetric K/V (vHeadDim/swaVHeadDim
            // nil → the KV formula's `dim + dim` recovers `2·dim`). `head_dim` required here (fail closed;
            // its `hidden/heads` derivation would be silently wrong for the sliding layers too).
            swaHeadDim = try requirePositiveInt(geom, "head_dim")
            swaKVHeads = try requirePositiveInt(geom, "num_key_value_heads")
            vHeadDim = nil
            swaVHeadDim = nil
        case .dualGeometrySWA where isMimoV2Flash:
            // mimo_v2_flash: ASYMMETRIC K/V (V narrower than K) on both layer classes. All four
            // head-dim fields are non-optional in the vendored Codable — fail CLOSED on absence, no
            // fallback. `head_dim`/`swa_head_dim` are EXPLICIT config (not `hidden/heads`); the global
            // `headDim` resolved above already reads `head_dim`, here we add the value dims + SWA geom.
            vHeadDim = try requirePositiveInt(geom, "v_head_dim")
            swaKVHeads = try requirePositiveInt(geom, "swa_num_key_value_heads")
            swaHeadDim = try requirePositiveInt(geom, "swa_head_dim")
            swaVHeadDim = try requirePositiveInt(geom, "swa_v_head_dim")
        case .dualGeometrySWA:
            // baichuan_m1: SYMMETRIC K/V (vHeadDim nil → the KV formula's `headDim + headDim` recovers
            // `2·headDim`). SWA head counts fall back to global (`??` mirrors BaichuanM1.swift:72-74);
            // swaHeadDim derived from `hidden_size / num_swa_attention_heads`.
            guard let hidden = intOf(geom("hidden_size")), hidden > 0 else {
                throw ModelConfigDecodeError.missingField("hidden_size (dual-geometry-SWA SWA head dim)")
            }
            let globalAttnHeads = try requirePositiveInt(geom, "num_attention_heads")
            let swaAttnHeads = intOf(geom("num_swa_attention_heads")) ?? globalAttnHeads
            guard swaAttnHeads > 0 else {
                throw ModelConfigDecodeError.invalidField("num_swa_attention_heads (dual-geometry-SWA)")
            }
            swaHeadDim = hidden / swaAttnHeads
            swaKVHeads = intOf(geom("num_swa_key_value_heads")) ?? nKVHeads
            vHeadDim = nil
            swaVHeadDim = nil
        default:
            swaKVHeads = nil
            swaHeadDim = nil
            vHeadDim = nil
            swaVHeadDim = nil
        }

        let nAttnLayers: Int
        switch archClass {
        case .uniformGQA:
            nAttnLayers = nLayers
        case .hybridLinear:
            if isFalconH1 {
                // Parallel hybrid: EVERY layer grows an attention KV cache (degenerate nAttnLayers ==
                // nLayers), on top of its Mamba-2 recurrent state. Not a resolved subset.
                nAttnLayers = nLayers
            } else {
                nAttnLayers = try resolveHybridAttnLayers(geom: geom, nLayers: nLayers)
            }
        case .interleavedSWA:
            if isGemma3n {
                // Count `full_attention` only within the cached prefix (`nLayers`), matching the
                // vendored `newCache`'s `0 ..< firstKvSharedLayerIdx` cache-allocation loop.
                nAttnLayers = try resolveGemma3nGlobalLayers(geom: geom, cachedPrefixLen: nLayers)
            } else {
                nAttnLayers = try resolveSWAGlobalLayers(geom: geom, nLayers: nLayers)
            }
        case .mlaAsImplemented:
            // Every layer caches decompressed per-head K/V and grows with context (mirrors the
            // catalog DeepSeek-R1 entry, nAttnLayers == nLayers == 61).
            nAttnLayers = nLayers
        case .dualGeometrySWA:
            if isMimoV2Flash {
                // Global (growing) layers = count of `0`s in `hybrid_layer_pattern` (1 = SWA layer).
                nAttnLayers = try resolveMimoGlobalLayers(geom: geom, nLayers: nLayers)
            } else if isGemma4 {
                // Global (growing) = count of `full_attention` WITHIN the cached prefix (`nLayers` was
                // truncated to `num_hidden_layers − num_kv_shared_layers`); the local (SWA) count is
                // `nLayers − nAttnLayers`, matching the vendored `newCache`'s per-type split over the
                // `0 ..< firstKvShared` prefix.
                nAttnLayers = try resolveGemma4GlobalLayers(geom: geom, cachedPrefixLen: nLayers)
            } else {
                // baichuan: global (growing) = total − |sliding_window_layers| (the window-capped subset).
                nAttnLayers = try resolveDualGeometryGlobalLayers(geom: geom, nLayers: nLayers)
            }
        default:
            // Unreachable given the allow-list above, but keep the switch total.
            throw ModelConfigDecodeError.unsupportedModelType(rawModelType)
        }

        // Sliding-window size for interleaved-SWA local layers — fail closed, never default (spec
        // §5): a missing `sliding_window` means the local-layer cap can't be honestly derived.
        let slidingWindow: Int?
        switch archClass {
        case .interleavedSWA:
            guard let w = intOf(geom("sliding_window")), w > 0 else {
                throw ModelConfigDecodeError.missingField("sliding_window")
            }
            slidingWindow = w
        case .dualGeometrySWA:
            // The local layers are capped at the window — fail closed if absent. baichuan and gemma4
            // read `sliding_window`; mimo's vendored Codable reads `sliding_window_size` (:516).
            let key = isMimoV2Flash ? "sliding_window_size" : "sliding_window"
            guard let w = intOf(geom(key)), w > 0 else {
                throw ModelConfigDecodeError.missingField(key)
            }
            slidingWindow = w
        default:
            slidingWindow = nil
        }

        // Fixed per-sequence recurrent+conv state for hybrid-linear GatedDeltaNet layers
        // (qwen3_next/qwen3_5) — a small, always-present term the growing-KV formula omits. Derived
        // from the linear-attention config geometry and reconciled against the vendored MLX-Swift
        // arch's actual `MambaCache` allocation (see `resolveHybridFixedStateBytes`). `0` for every
        // other class, and `0` (fail-open) when the linear geometry fields are absent.
        let fixedStateBytes: Int
        switch archClass {
        case .hybridLinear:
            if isFalconH1 {
                // Parallel hybrid: the Mamba-2 state lives on ALL layers, so size it over nLayers (not
                // nLayers − nAttnLayers, which is 0). Fails CLOSED on a missing geometry field — this
                // term is material (~146 MiB/seq on 34B), so a fail-open 0 would materially under-count.
                fixedStateBytes = try resolveFalconH1StateBytes(geom: geom, nLayers: nLayers)
            } else if isGraniteMoeHybrid {
                // Interval-select Mamba-2 hybrid: the recurrent state lives on the mamba layers only
                // (nLayers − nAttnLayers). Fails CLOSED on a missing mamba field — the term is material.
                fixedStateBytes = try resolveGraniteMoeHybridStateBytes(geom: geom, nMambaLayers: nLayers - nAttnLayers)
            } else if isLfm2 {
                fixedStateBytes = resolveLfm2ConvStateBytes(geom: geom, nLinearLayers: nLayers - nAttnLayers)
            } else if isJamba {
                fixedStateBytes = resolveJambaStateBytes(geom: geom, nLinearLayers: nLayers - nAttnLayers)
            } else {
                fixedStateBytes = resolveHybridFixedStateBytes(geom: geom, nLinearLayers: nLayers - nAttnLayers)
            }
        case .dualGeometrySWA:
            if isMimoV2Flash || isGemma4 {
                // mimo_v2_flash and gemma4 cache K/V only (KVCacheSimple/StandardKVCache/RotatingKVCache);
                // no MambaCache/conv state anywhere. (mimo's `attention_sink_bias` and gemma4's PLE
                // per-layer embeddings are weights — captured in safetensors bytes — not per-seq state.)
                fixedStateBytes = 0
            } else {
                // Baichuan-M1 holds a `MambaCache` conv slot on EVERY layer (global + SWA), each sized by
                // that layer class's own head geometry. Small but always present — the growing-KV formula
                // omits it, so add it here. Both geometries are already resolved.
                fixedStateBytes = resolveBaichuanM1ConvStateBytes(
                    nGlobal: nAttnLayers, kvHeadsGlobal: nKVHeads, headDimGlobal: headDim,
                    nSWA: nLayers - nAttnLayers, kvHeadsSWA: swaKVHeads ?? nKVHeads, headDimSWA: swaHeadDim ?? headDim)
            }
        default:
            fixedStateBytes = 0
        }

        // Weight quantization width, when the checkpoint declares one. MLX writes it under
        // `quantization`; mlx-lm also emits `quantization_config`. Either form carries `bits`.
        let quantBlock = (root["quantization"] as? [String: Any]) ?? (root["quantization_config"] as? [String: Any])
        let quantBits = intOf(quantBlock?["bits"])

        let profile = ModelArchProfile(
            id: id, modelType: archClass, nLayers: nLayers, nAttnLayers: nAttnLayers,
            nKVHeads: nKVHeads, headDim: headDim, slidingWindow: slidingWindow, fixedStateBytes: fixedStateBytes,
            nativeMaxContext: nativeMaxContext, weightsBytes4bitEstimate: safetensorsBytes,
            license: license,
            mlaHeads: mlaHeads, mlaRopeDim: mlaRopeDim, mlaNopeDim: mlaNopeDim, mlaVDim: mlaVDim,
            swaKVHeads: swaKVHeads, swaHeadDim: swaHeadDim, vHeadDim: vHeadDim, swaVHeadDim: swaVHeadDim)
        return ParsedModelArch(
            profile: profile, weightsAreMeasured: safetensorsBytes > 0, quantBits: quantBits)
    }

    /// Filesystem wrapper: read `<directory>/config.json`, size the weights honestly, then decode.
    /// `id` defaults to the directory's last path component.
    ///
    /// Honest sizing takes the LARGER of two signals, because a partial download has real shards on
    /// disk that under-count the true footprint:
    ///  - the summed real on-disk `*.safetensors` bytes (0 if none present), and
    ///  - the `metadata.total_size` declared by `model.safetensors.index.json`, when present.
    ///
    /// A complete download's on-disk shards meet-or-exceed the declared total (shard files carry their
    /// own header overhead) → the measured shard bytes win. A partial/interrupted download, or a
    /// metadata-only prefetch with no shards at all, has a declared total that exceeds what's on disk
    /// → the declared (larger, more conservative) size wins but is marked NOT measured. If neither
    /// signal is available at all, fail closed with `.weightsUnknown` rather than silently sizing the
    /// checkpoint as 0 bytes (the exact honesty failure that let a phantom checkpoint classify GREEN).
    public static func decodeModelDirectory(_ directory: URL, id: String? = nil) throws -> ParsedModelArch {
        let configURL = directory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw ModelConfigDecodeError.missingConfigFile(configURL.path)
        }
        let resolvedID = id ?? directory.lastPathComponent
        let shardBytes = sumSafetensorsBytes(in: directory)
        let declared = declaredSafetensorsBytes(in: directory)

        if shardBytes == 0 && declared == nil {
            throw ModelConfigDecodeError.weightsUnknown(directory.path)
        }
        if let declared, declared > shardBytes {
            // Declared full size exceeds what's actually on disk: a partial/interrupted download or a
            // metadata-only prefetch. Use the declared size for an honest (conservative) fit-check, but
            // it is a claim from the index, not a measured fact — override measured to false.
            let parsed = try decode(configJSON: configData, safetensorsBytes: declared, id: resolvedID)
            return ParsedModelArch(
                profile: parsed.profile, weightsAreMeasured: false,
                weightsAreDeclared: true, quantBits: parsed.quantBits)
        }
        // shardBytes >= declared (or no index at all) and shardBytes > 0: trust the real on-disk shards.
        return try decode(configJSON: configData, safetensorsBytes: shardBytes, id: resolvedID)
    }

    /// Read `<directory>/model.safetensors.index.json` and return the declared `metadata.total_size`,
    /// only when it parses to a positive integer. Returns `nil` when the file is absent, unreadable,
    /// malformed, missing the field, or the value is non-positive — callers must treat `nil` as "no
    /// declared size available", never as 0.
    public static func declaredSafetensorsBytes(in directory: URL) -> Int? {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL) else { return nil }
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        guard let metadata = root["metadata"] as? [String: Any] else { return nil }
        guard let totalSize = intOf(metadata["total_size"]), totalSize > 0 else { return nil }
        return totalSize
    }

    /// Sum the byte sizes of every `*.safetensors` shard in `directory` — the real resident weight
    /// footprint (includes any vision tower on a VL checkpoint, which does load into memory).
    ///
    /// Hugging Face snapshot directories store each shard as a SYMLINK into the shared blobs cache,
    /// so the symlink resolves to the real weights on disk. Read the size of the resolved target,
    /// not the ~76-byte symlink (which would under-count weights to ≈0 and make an unservable model
    /// look like it fits — the exact honesty failure the fit-check exists to prevent).
    public static func sumSafetensorsBytes(in directory: URL) -> Int {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return 0 }
        var total = 0
        for url in entries where url.pathExtension == "safetensors" {
            let resolvedPath = url.resolvingSymlinksInPath().path
            if let size = (try? fm.attributesOfItem(atPath: resolvedPath))?[.size] as? Int {
                total += size
            }
        }
        return total
    }

    // MARK: - Field helpers

    /// Read a required geometry field that must be present AND a positive integer. Throws
    /// `.missingField` when absent/unparseable, `.invalidField` when present but ≤ 0. Used for terms
    /// that feed the KV formula directly, where a `0`/negative would zero the KV rate and let an
    /// unservable config classify GREEN (the phantom-fit hole, geometry side).
    private static func requirePositiveInt(_ geom: (String) -> Any?, _ key: String) throws -> Int {
        guard let v = intOf(geom(key)) else { throw ModelConfigDecodeError.missingField(key) }
        guard v > 0 else { throw ModelConfigDecodeError.invalidField(key) }
        return v
    }

    /// `head_dim` when present; otherwise `hidden_size / num_attention_heads` (the standard derived
    /// per-head dim). Throws `.missingField` when neither route is available.
    private static func resolveHeadDim(geom: (String) -> Any?) throws -> Int {
        if let headDim = intOf(geom("head_dim")) { return headDim }
        guard let hidden = intOf(geom("hidden_size")), let heads = intOf(geom("num_attention_heads")), heads > 0 else {
            throw ModelConfigDecodeError.missingField("head_dim (and no hidden_size/num_attention_heads to derive it)")
        }
        return hidden / heads
    }

    /// The count of full-attention (KV-growing) layers for a hybrid-linear model: authoritative from
    /// `layer_types` when present; otherwise `nLayers / full_attention_interval`.
    private static func resolveHybridAttnLayers(geom: (String) -> Any?, nLayers: Int) throws -> Int {
        if let layerTypes = geom("layer_types") as? [Any] {
            // Two spellings across families: lfm2/lfm2_moe tag KV-growing layers `"full_attention"` (vs
            // `"conv"`); granitemoehybrid tags them `"attention"` (vs `"mamba"`). Both denote a plain
            // KVCacheSimple-growing layer in the vendored arch, so count either. No hybrid-linear family
            // uses `"attention"` for a non-growing layer, so this cannot over-count.
            let full = layerTypes.compactMap { $0 as? String }
                .filter { $0 == "full_attention" || $0 == "attention" }.count
            if full > 0 { return full }
        }
        // LFM2 enumerates its attention layers as explicit INDICES (`full_attn_idxs`) rather than a
        // `layer_types` array or an interval; the count of indices is the attention-layer count. The
        // qwen3_next/qwen3_5 configs carry no `full_attn_idxs`, so this source is inert for them.
        if let fullAttnIdxs = geom("full_attn_idxs") as? [Any] {
            let count = fullAttnIdxs.compactMap { intOf($0) }.count
            if count > 0 { return count }
        }
        if let interval = intOf(geom("full_attention_interval")), interval > 0 {
            return nLayers / interval
        }
        // jamba (AI21) enumerates neither `layer_types`, `full_attn_idxs`, nor an interval — its
        // attention layers are the periodic set `i % attn_layer_period == attn_layer_offset`
        // (`JambaConfiguration` derives its `layers_block_type` from exactly this when absent,
        // Jamba.swift:104-108). Count that set over `nLayers`.
        if let period = intOf(geom("attn_layer_period")), period > 0 {
            let offset = intOf(geom("attn_layer_offset")) ?? 0
            let count = (0..<nLayers).filter { $0 % period == offset }.count
            if count > 0 { return count }
        }
        throw ModelConfigDecodeError.missingField(
            "layer_types, full_attn_idxs, full_attention_interval, or attn_layer_period "
                + "(hybrid-linear attention-layer count)")
    }

    /// Fixed per-sequence recurrent state (bytes) for jamba's Mamba layers, reconciled against the
    /// vendored `Jamba.swift` `MambaCache` allocation. The Mamba mixer's inner dim is `mamba_expand ×
    /// hidden_size` (`JambaMambaMixer` at Jamba.swift:213 — this SHADOWS the config's MoE
    /// `intermediate_size`, which sizes only the MLP). MambaCache holds two slots per mamba layer:
    /// `[0]` conv state `[B, mamba_d_conv − 1, d_inner]` at activation precision (bf16/fp16, 2 B) and
    /// `[1]` SSM state `[B, d_inner, mamba_d_state]` at fp32 (4 B). Per mamba layer:
    /// `(mamba_d_conv − 1) × d_inner × 2 + d_inner × mamba_d_state × 4`. This is a CLASSIC Mamba
    /// conv+SSM term — distinct from lfm2's conv-only cache and from qwen3_5's GatedDeltaNet. Returns
    /// `0` (fail-OPEN) when a required field is absent, matching the other hybrid-linear resolvers.
    private static func resolveJambaStateBytes(geom: (String) -> Any?, nLinearLayers: Int) -> Int {
        guard nLinearLayers > 0,
            let hidden = intOf(geom("hidden_size")), hidden > 0,
            let expand = intOf(geom("mamba_expand")), expand > 0,
            let dConv = intOf(geom("mamba_d_conv")), dConv >= 1,
            let dState = intOf(geom("mamba_d_state")), dState > 0
        else { return 0 }
        let dInner = expand * hidden
        let convBytes = (dConv - 1) * dInner * 2  // activation precision (bf16/fp16)
        let ssmBytes = dInner * dState * 4        // fp32 SSM state
        return (convBytes + ssmBytes) * nLinearLayers
    }

    /// Fixed per-sequence recurrent state (bytes) for Falcon-H1's Mamba-2 layers. Falcon-H1 is a
    /// PARALLEL hybrid: the vendored `FalconH1.swift` `newCache` (:799-801) gives EVERY layer a
    /// `CacheList(MambaCache(), attentionCache.copy())`, so all `nLayers` layers hold a conv+SSM
    /// recurrent state IN ADDITION to a growing attention KV cache — hence sized over ALL layers, not
    /// `nLayers − nAttnLayers` (which is 0 for this family). Both cache slots are at activation
    /// precision (2 B), source-verified against `FalconH1.swift`/`SSM.swift`:
    ///  - conv state `[B, mamba_d_conv−1, convDim]`, convDim = mamba_d_ssm + 2·mamba_n_groups·mamba_d_state
    ///    (`FalconH1.swift:376`; intermediateSize == mamba_d_ssm at :364, NOT mamba_expand×hidden).
    ///  - SSM state `mamba_d_ssm × mamba_d_state`, stored at the activation dtype: `SSM.swift:167` sets
    ///    `stateType = state?.dtype ?? x.dtype` and `:219` casts `nextState.asType(stateType)`, so the
    ///    slot follows the activation dtype forever — ×2, not the ×4 conservative width used for jamba
    ///    where the SSM dtype was unknown.
    /// Fails CLOSED (throws `.missingField`) when a required geometry field is absent. `mamba_d_ssm`
    /// especially: the vendored code defaults it to 1536 (`FalconH1.swift:149`), and mirroring a
    /// vendored default in the sizer is exactly what the afmoe precedent rejects — a default-relying
    /// checkpoint would then be sized against a guessed inner dim. Unlike the ~2%-of-total conv terms
    /// elsewhere (which fail OPEN to 0), this family's fixed state is MATERIAL (~146 MiB/seq on 34B), so
    /// a fail-open 0 on any missing field would be a real under-count, not a negligible omission.
    private static func resolveFalconH1StateBytes(geom: (String) -> Any?, nLayers: Int) throws -> Int {
        guard let dSSM = intOf(geom("mamba_d_ssm")), dSSM > 0 else {
            throw ModelConfigDecodeError.missingField(
                "mamba_d_ssm (falcon_h1 Mamba-2 inner dim; refused rather than sized against the vendored 1536 default)")
        }
        guard let nGroups = intOf(geom("mamba_n_groups")), nGroups > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_n_groups (falcon_h1 conv-state group dim)")
        }
        guard let dState = intOf(geom("mamba_d_state")), dState > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_d_state (falcon_h1 SSM state dim)")
        }
        guard let dConv = intOf(geom("mamba_d_conv")), dConv >= 1 else {
            throw ModelConfigDecodeError.missingField("mamba_d_conv (falcon_h1 conv kernel width)")
        }
        let convDim = dSSM + 2 * nGroups * dState
        let convBytes = (dConv - 1) * convDim * 2  // activation precision (bf16/fp16)
        let ssmBytes = dSSM * dState * 2            // activation precision (SSM.swift casts state to activation dtype)
        return (convBytes + ssmBytes) * nLayers
    }

    /// Fixed per-sequence recurrent state (bytes) for GraniteMoeHybrid's Mamba-2 layers. Unlike
    /// falcon_h1's PARALLEL shape, granitemoehybrid is INTERVAL-SELECT: only the `layer_types == "mamba"`
    /// layers hold recurrent state (the vendored `GraniteMoeHybrid.swift` `newCache` :520-525 gives them a
    /// `MambaCache` and the `"attention"` layers a plain `KVCacheSimple`), so this is sized over
    /// `nMambaLayers = nLayers − nAttnLayers`. Both cache slots are at activation precision (2 B),
    /// source-verified against `GraniteMoeHybrid.swift` + the shared `SSM.swift`:
    ///  - mamba inner dim `intermediateSize = mamba_n_heads × mamba_d_head` (GraniteMoeHybrid.swift:79) —
    ///    the arch uses the heads×head_dim product, NOT `mamba_expand × hidden_size` (they coincide on the
    ///    published configs but the product is authoritative).
    ///  - conv state `[B, mamba_d_conv−1, convDim]`, `convDim = intermediateSize + 2·mamba_n_groups·mamba_d_state`
    ///    (:83, :116).
    ///  - SSM state `intermediateSize × mamba_d_state` (`state.reshaped(b,1,g,repeats,dh,d)` at SSM.swift:209
    ///    ⇒ g·repeats·dh = numHeads·headDim = intermediateSize), stored at the ACTIVATION dtype
    ///    (`stateType = state?.dtype ?? x.dtype`, SSM.swift:167/219) — ×2, the same shared SSM path as
    ///    falcon_h1, NOT jamba's conservative ×4.
    /// Fails CLOSED (throws `.missingField`) on any absent mamba geometry field: this recurrent term is
    /// MATERIAL (~1 MiB/mamba-layer here), so a fail-open 0 would be a real under-count, matching the
    /// falcon_h1 precedent (and unlike the small conv-only lfm2 term, which fails open).
    private static func resolveGraniteMoeHybridStateBytes(geom: (String) -> Any?, nMambaLayers: Int) throws -> Int {
        guard nMambaLayers > 0 else { return 0 }
        guard let nHeads = intOf(geom("mamba_n_heads")), nHeads > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_n_heads (granitemoehybrid Mamba-2 head count)")
        }
        guard let dHead = intOf(geom("mamba_d_head")), dHead > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_d_head (granitemoehybrid Mamba-2 head dim)")
        }
        guard let nGroups = intOf(geom("mamba_n_groups")), nGroups > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_n_groups (granitemoehybrid conv-state group dim)")
        }
        guard let dState = intOf(geom("mamba_d_state")), dState > 0 else {
            throw ModelConfigDecodeError.missingField("mamba_d_state (granitemoehybrid SSM state dim)")
        }
        guard let dConv = intOf(geom("mamba_d_conv")), dConv >= 1 else {
            throw ModelConfigDecodeError.missingField("mamba_d_conv (granitemoehybrid conv kernel width)")
        }
        let inner = nHeads * dHead
        let convDim = inner + 2 * nGroups * dState
        let convBytes = (dConv - 1) * convDim * 2  // activation precision (bf16/fp16)
        let ssmBytes = inner * dState * 2          // activation precision (SSM.swift casts state to activation dtype)
        return (convBytes + ssmBytes) * nMambaLayers
    }

    /// Fixed per-sequence recurrent state (bytes) for LFM2's linear (conv) layers, reconciled against
    /// the vendored `LFM2.swift` conv-cache allocation (`LFM2.swift:220`: `zeros([B, conv_L_cache − 1,
    /// hidden_size])` at activation precision; the `MambaCache` uses only `cache[0]` — no SSM state
    /// slot). Per conv layer: `(conv_L_cache − 1) × hidden_size × 2` bytes (bf16/fp16 activation
    /// precision). Returns `0` when a required field is absent — fail-OPEN on this small term (it is a
    /// per-sequence constant dominated by weights + growing KV), matching `resolveHybridFixedStateBytes`.
    private static func resolveLfm2ConvStateBytes(geom: (String) -> Any?, nLinearLayers: Int) -> Int {
        guard nLinearLayers > 0,
            let convLCache = intOf(geom("conv_L_cache")), convLCache >= 1,
            let hidden = intOf(geom("hidden_size")), hidden > 0
        else { return 0 }
        let perLayer = (convLCache - 1) * hidden * 2  // activation precision (bf16/fp16)
        return perLayer * nLinearLayers
    }

    /// The count of full-attention (KV-growing, global) layers for interleaved-SWA: authoritative
    /// from `layer_types` when present (counts `"full_attention"` entries); otherwise
    /// `nLayers / sliding_window_pattern` (transformers' `(i+1) % pattern == 0` rule gives exactly
    /// `floor(nLayers / pattern)` global layers). Never defaults the pattern itself (spec §5) — a
    /// checkpoint with neither signal fails closed rather than assuming transformers' pattern-6.
    private static func resolveSWAGlobalLayers(geom: (String) -> Any?, nLayers: Int) throws -> Int {
        if let layerTypes = geom("layer_types") as? [Any] {
            let full = layerTypes.compactMap { $0 as? String }.filter { $0 == "full_attention" }.count
            if full > 0 { return full }
        }
        if let pattern = intOf(geom("sliding_window_pattern")), pattern > 0 {
            return nLayers / pattern
        }
        throw ModelConfigDecodeError.missingField(
            "layer_types or sliding_window_pattern (interleaved-SWA global-attention layer count)")
    }

    /// The count of GLOBAL (full-context-growing) attention layers for a dual-geometry-SWA model:
    /// `nLayers − |sliding_window_layers|`. Baichuan-M1 enumerates its window-capped layers as an
    /// explicit index array `sliding_window_layers` (`BaichuanM1.swift` reads it to decide which
    /// layers get a `RotatingKVCache`); every other layer grows a `KVCacheSimple`. Fails CLOSED when
    /// the array is absent or empty, or when it leaves no global layers — a missing/degenerate
    /// enumeration means the global/SWA split can't be honestly derived, and defaulting either way
    /// would mis-size one of the two geometry terms (the phantom-GREEN hazard this class exists to
    /// avoid). Not a `sliding_window_pattern` family — the layers are listed by index, not a period.
    private static func resolveDualGeometryGlobalLayers(geom: (String) -> Any?, nLayers: Int) throws -> Int {
        guard let swaLayers = geom("sliding_window_layers") as? [Any] else {
            throw ModelConfigDecodeError.missingField(
                "sliding_window_layers (dual-geometry-SWA window-capped layer indices)")
        }
        let nSWA = swaLayers.compactMap { intOf($0) }.count
        guard nSWA > 0 else {
            throw ModelConfigDecodeError.invalidField("sliding_window_layers (empty — no window-capped layers)")
        }
        let nGlobal = nLayers - nSWA
        guard nGlobal > 0 else {
            throw ModelConfigDecodeError.invalidField(
                "sliding_window_layers (\(nSWA) SWA layers leave \(nGlobal) global of \(nLayers))")
        }
        return nGlobal
    }

    /// The count of GLOBAL (full-context-growing) attention layers for mimo_v2_flash: the number of
    /// `0` entries in `hybrid_layer_pattern` (0 = global `KVCacheSimple`, 1 = SWA `RotatingKVCache`,
    /// per `MiMoV2Flash.swift:353-358`). Fails CLOSED on a missing array, a length ≠ `num_hidden_layers`
    /// (a truncated/wrong pattern would mis-split the layers), or a degenerate all-global/all-SWA split
    /// (a genuine dual-geometry model has both). Distinct from baichuan's `sliding_window_layers`
    /// index-array source — mimo enumerates a per-layer 0/1 map.
    private static func resolveMimoGlobalLayers(geom: (String) -> Any?, nLayers: Int) throws -> Int {
        guard let pattern = geom("hybrid_layer_pattern") as? [Any] else {
            throw ModelConfigDecodeError.missingField("hybrid_layer_pattern (mimo_v2_flash global/SWA layer map)")
        }
        let codes = pattern.compactMap { intOf($0) }
        guard codes.count == nLayers else {
            throw ModelConfigDecodeError.invalidField(
                "hybrid_layer_pattern (length \(codes.count) != num_hidden_layers \(nLayers))")
        }
        let nGlobal = codes.filter { $0 == 0 }.count
        guard nGlobal > 0, nGlobal < nLayers else {
            throw ModelConfigDecodeError.invalidField(
                "hybrid_layer_pattern (\(nGlobal) global of \(nLayers) — need a genuine global/SWA split)")
        }
        return nGlobal
    }

    /// Fixed per-sequence conv state (bytes) for Baichuan-M1's per-layer `MambaCache` conv slot,
    /// reconciled against the vendored `BaichuanM1.swift newCache` (:265-273): EVERY layer holds a
    /// `CacheList(MambaCache(), kvCache)`, so both the global and SWA layers carry a conv slot sized
    /// by that layer class's OWN head geometry. The slot is a single-token depthwise buffer: 2 slots
    /// (K&V) × 2 B activation precision per (kvHeads × headDim) element; `conv_window` does NOT appear
    /// (the slot is always 1 token — `customConvolution` reads only kernel taps 0 and 1). Both
    /// geometries are already resolved by the caller, so this takes them directly. On the real 14B
    /// checkpoint this is 20×2,048 (global) + 20×4,096 (SWA) = 122,880 B.
    private static func resolveBaichuanM1ConvStateBytes(
        nGlobal: Int, kvHeadsGlobal: Int, headDimGlobal: Int,
        nSWA: Int, kvHeadsSWA: Int, headDimSWA: Int
    ) -> Int {
        let perGlobal = 2 * 2 * kvHeadsGlobal * headDimGlobal  // 2 conv slots (K&V) × 2 B × (kv×dim)
        let perSWA = 2 * 2 * kvHeadsSWA * headDimSWA
        return max(0, nGlobal) * perGlobal + max(0, nSWA) * perSWA
    }

    /// The count of full-attention (KV-growing, global) layers WITHIN gemma3n's cached prefix — the
    /// first `cachedPrefixLen` (`num_hidden_layers − num_kv_shared_layers`) entries of `layer_types`,
    /// matching the vendored `Gemma3nText.newCache` allocation loop (`0 ..< firstKvSharedLayerIdx`).
    /// Requires `layer_types` (gemma3n always ships it; no `sliding_window_pattern` fallback exists for
    /// this family) with at least `cachedPrefixLen` entries. Fails closed if the prefix contains no
    /// `full_attention` layer — that would leave the growing-KV term at 0 and trip the `nAttnLayers > 0`
    /// derivability sentinel; a real gemma3n checkpoint always has global layers in the prefix.
    private static func resolveGemma3nGlobalLayers(geom: (String) -> Any?, cachedPrefixLen: Int) throws -> Int {
        guard let layerTypes = geom("layer_types") as? [Any] else {
            throw ModelConfigDecodeError.missingField("layer_types (gemma3n cached-prefix global-attention layer count)")
        }
        let types = layerTypes.compactMap { $0 as? String }
        guard types.count >= cachedPrefixLen else {
            throw ModelConfigDecodeError.invalidField("layer_types (fewer entries than the cached-prefix layer count)")
        }
        let full = types[0..<cachedPrefixLen].filter { $0 == "full_attention" }.count
        guard full > 0 else {
            throw ModelConfigDecodeError.invalidField("layer_types (no full_attention layer in gemma3n's cached prefix)")
        }
        return full
    }

    /// The count of full-attention (KV-growing, global) layers WITHIN gemma4's cached prefix — the first
    /// `cachedPrefixLen` (`num_hidden_layers − num_kv_shared_layers`) layers, matching the vendored
    /// `Gemma4Text.newCache` loop (`0 ..< firstKvShared`, full → `StandardKVCache`, else → rotating).
    /// Prefers the explicit `layer_types` (the real published configs ship it, length = total layers);
    /// requires at least `cachedPrefixLen` entries and counts `full_attention` in that prefix. When
    /// `layer_types` is ABSENT, synthesizes the vendored default pattern (`Gemma4Text.swift:148-158`):
    /// within each block of `sliding_window_pattern`, the LAST index is `full_attention` — i.e. layer `i`
    /// is full iff `i % pattern == pattern − 1` — counted over the cached prefix. Fails closed if BOTH
    /// sources are absent (never guess the split), or if the prefix contains no full layer (a zero
    /// growing-KV term would trip the `nAttnLayers > 0` derivability sentinel / phantom-GREEN long
    /// contexts). Distinct from gemma3n (which never synthesizes) by that `sliding_window_pattern` fallback.
    private static func resolveGemma4GlobalLayers(geom: (String) -> Any?, cachedPrefixLen: Int) throws -> Int {
        if let layerTypes = geom("layer_types") as? [Any] {
            let types = layerTypes.compactMap { $0 as? String }
            guard types.count >= cachedPrefixLen else {
                throw ModelConfigDecodeError.invalidField("layer_types (fewer entries than gemma4's cached-prefix layer count)")
            }
            let full = types[0..<cachedPrefixLen].filter { $0 == "full_attention" }.count
            guard full > 0 else {
                throw ModelConfigDecodeError.invalidField("layer_types (no full_attention layer in gemma4's cached prefix)")
            }
            return full
        }
        guard let pattern = intOf(geom("sliding_window_pattern")), pattern > 0 else {
            throw ModelConfigDecodeError.missingField(
                "layer_types or sliding_window_pattern (gemma4 cached-prefix global-attention layer count)")
        }
        // Vendored synthesis: layer i is full_attention iff i % pattern == pattern − 1.
        let full = (0..<cachedPrefixLen).filter { $0 % pattern == pattern - 1 }.count
        guard full > 0 else {
            throw ModelConfigDecodeError.invalidField(
                "sliding_window_pattern (\(pattern) leaves no full_attention layer in gemma4's \(cachedPrefixLen)-layer cached prefix)")
        }
        return full
    }

    /// Fixed per-sequence recurrent+conv state (bytes) for the linear-attention layers of a
    /// hybrid-linear model, reconciled against the vendored MLX-Swift `MambaCache` allocation
    /// (`GatedDelta.swift`: SSM state `[B, Hv, Dv, Dk]` hard-coded fp32; `Qwen3Next.swift`/`Qwen35.swift`:
    /// conv state `[B, kernel-1, 2·keyDim + valueDim]` at activation precision):
    ///  - SSM state per layer: `linear_num_value_heads × linear_value_head_dim × linear_key_head_dim × 4`
    ///    (fp32 unconditionally — never quantized, independent of the compute dtype or KV quant tier).
    ///  - conv state per layer: `(linear_conv_kernel_dim − 1) × (2·keyDim + valueDim) × 2`, where
    ///    `keyDim = linear_key_head_dim × linear_num_key_heads`,
    ///    `valueDim = linear_value_head_dim × linear_num_value_heads`. The `2` is the activation
    ///    precision (bf16/fp16), an established mlx-swift runtime convention rather than a config field;
    ///    it is a minor (~2%-of-per-layer) term, so its exact width barely moves the total.
    ///
    /// Returns `0` when any required linear field is absent — fail-OPEN on this small term rather than
    /// refusing to size an otherwise-decodable hybrid config. Unlike missing weights (a phantom-fit
    /// risk that must fail closed), omitting this term only mildly UNDER-counts a per-sequence constant
    /// that is dominated by weights + growing KV, so a fallback of 0 is the pre-existing behavior, not a
    /// new honesty hole. A present value makes the fit-check strictly MORE conservative.
    private static func resolveHybridFixedStateBytes(geom: (String) -> Any?, nLinearLayers: Int) -> Int {
        guard nLinearLayers > 0,
            let nKHeads = intOf(geom("linear_num_key_heads")),
            let nVHeads = intOf(geom("linear_num_value_heads")),
            let kHeadDim = intOf(geom("linear_key_head_dim")),
            let vHeadDim = intOf(geom("linear_value_head_dim")),
            let convKernel = intOf(geom("linear_conv_kernel_dim")), convKernel >= 1
        else { return 0 }
        let keyDim = kHeadDim * nKHeads
        let valueDim = vHeadDim * nVHeads
        let convDim = 2 * keyDim + valueDim
        let convBytes = (convKernel - 1) * convDim * 2  // activation precision (bf16/fp16)
        let ssmBytes = nVHeads * vHeadDim * kHeadDim * 4  // fp32, hard-coded in GatedDelta.swift
        return (convBytes + ssmBytes) * nLinearLayers
    }

    /// Tolerant int read: JSON numbers decode as `Int`, `Double`, or `NSNumber` depending on form.
    /// Fails closed (`nil`) on a non-integral or out-of-`Int`-range value rather than truncating or
    /// sign-flipping into garbage: `NSNumber.intValue` / `Int(_ d:)` on `1e300` yield a bogus (often
    /// negative) count that would silently corrupt the KV formula. `Int(exactly:)` rejects those.
    private static func intOf(_ value: Any?) -> Int? {
        if let i = value as? Int { return i }
        if let n = value as? NSNumber { return Int(exactly: n.doubleValue) }
        if let d = value as? Double { return Int(exactly: d) }
        if let s = value as? String { return Int(s) }
        return nil
    }

    /// mistral3's honest KV class from config SHAPE (not the type string). The vendored
    /// `Mistral3Text.newCache` allocates a window-capped `RotatingKVCache` for a layer ONLY when the
    /// layer is `sliding_attention` AND `sliding_window` is present; otherwise it allocates a growing
    /// `KVCacheSimple`. So:
    ///   - `.interleavedSWA` iff `layer_types` contains "sliding_attention" AND `sliding_window > 0`
    ///     (some layers are genuinely window-capped; the SWA path counts the `full_attention` layers
    ///     as the growing/global set and reads the window);
    ///   - `.uniformGQA` otherwise. Two sub-cases both grow every layer in the arch, so uniform-GQA
    ///     is as-implemented-correct: (a) `layer_types` absent → the arch defaults all layers to
    ///     `full_attention`; (b) `sliding_attention` declared but NO `sliding_window` → the arch's
    ///     `else { KVCacheSimple() }` fallback grows every layer. Choosing uniform-GQA here fails
    ///     toward RED (over-counts KV), never phantom-GREEN by modeling a cap the runtime won't apply.
    private static func mistral3ArchClass(geom: (String) -> Any?) -> ArchClass {
        guard let layerTypes = geom("layer_types") as? [Any],
              layerTypes.compactMap({ $0 as? String }).contains("sliding_attention"),
              let window = intOf(geom("sliding_window")), window > 0
        else {
            return .uniformGQA
        }
        return .interleavedSWA
    }
}

// MARK: - Hybrid continuous-batching geometry (S2)

/// Admission-grade derivation of the hybrid per-layer cache geometry from `config.json`, for the
/// continuous-batching-over-hybrid build (design of record: the continuous-batching
/// heterogeneous-cache design). Lives here (same
/// file) to reuse the audited `intOf`/`requirePositiveInt` helpers and the source-priority chain of the
/// private count resolver `resolveHybridAttnLayers`.
///
/// KEY CONTRACT — this path is admission-grade, so it FAILS CLOSED (throws) on any missing/malformed
/// field, unlike the sizer's `resolveHybridFixedStateBytes`, which fails OPEN (returns 0) because a
/// missing minor term only makes the fit-check more conservative. An admission proof must never inherit
/// that leniency: a hybrid cache built on a silently-zeroed geometry is a correctness bug, not a
/// conservative estimate.
public extension ModelConfigDecoder {

    /// The INDEX set of full-attention (KV-growing) layers for a hybrid-linear model — the index-valued
    /// sibling of the private count resolver `resolveHybridAttnLayers`, sharing its exact source-priority
    /// chain (`layer_types` → `full_attn_idxs` → `full_attention_interval` → `attn_layer_period`/offset).
    /// The count-resolver invariant `indices.count == resolveHybridAttnLayers` is pinned by tests.
    static func resolveHybridAttentionLayerIndices(configJSON: Data) throws -> Set<Int> {
        guard let root = (try? JSONSerialization.jsonObject(with: configJSON)) as? [String: Any] else {
            throw ModelConfigDecodeError.malformedJSON
        }
        let textScope = (root["text_config"] as? [String: Any]) ?? [:]
        func geom(_ key: String) -> Any? { textScope[key] ?? root[key] }
        let nLayers = try requirePositiveInt(geom, "num_hidden_layers")
        return try hybridAttentionLayerIndices(geom: geom, nLayers: nLayers)
    }

    /// Core index resolver over a `geom` accessor. Mirrors `resolveHybridAttnLayers` (MCD.swift:726-758)
    /// but returns the layer indices, not just their count.
    internal static func hybridAttentionLayerIndices(
        geom: (String) -> Any?, nLayers: Int
    ) throws -> Set<Int> {
        // `layer_types`: KV-growing layers are tagged "full_attention" (lfm2/qwen) or "attention"
        // (granitemoehybrid); everything else is recurrent. Only honor it when it names ≥1 growing layer.
        if let layerTypes = geom("layer_types") as? [Any] {
            let strings = layerTypes.compactMap { $0 as? String }
            if !strings.isEmpty {
                let idx = strings.indices.filter { strings[$0] == "full_attention" || strings[$0] == "attention" }
                if !idx.isEmpty { return Set(idx) }
            }
        }
        // `full_attn_idxs`: the attention layers ARE the enumerated indices (lfm2).
        if let fullAttnIdxs = geom("full_attn_idxs") as? [Any] {
            let idx = fullAttnIdxs.compactMap { intOf($0) }
            if !idx.isEmpty { return Set(idx) }
        }
        // `full_attention_interval`: layer i is attention iff (i+1) % interval == 0 — the SAME formula as
        // the vendored Qwen35 constructor and `HybridLayerKindMap.qwen35`, so layer 0 is recurrent. This
        // set's size is floor(nLayers/interval), agreeing with the count resolver's `nLayers / interval`.
        if let interval = intOf(geom("full_attention_interval")), interval > 0 {
            return Set((0..<nLayers).filter { ($0 + 1) % interval == 0 })
        }
        // jamba: {i : i % attn_layer_period == attn_layer_offset}.
        if let period = intOf(geom("attn_layer_period")), period > 0 {
            let offset = intOf(geom("attn_layer_offset")) ?? 0
            let idx = (0..<nLayers).filter { $0 % period == offset }
            if !idx.isEmpty { return Set(idx) }
        }
        throw ModelConfigDecodeError.missingField(
            "layer_types, full_attn_idxs, full_attention_interval, or attn_layer_period "
                + "(hybrid-linear attention-layer indices)")
    }

    /// Fail-closed derivation of the full `HybridCacheGeometry` for a `qwen3_5` checkpoint (VL-wrapped or
    /// flat). qwen3_5-scoped for v1 because the conv/SSM formula below is the GatedDeltaNet layout
    /// (`Qwen35.swift`/`GatedDelta.swift`); other hybrid families (lfm2 conv-only, jamba Mamba-2) have
    /// distinct recurrent state and are out of scope here.
    public static func qwen35HybridGeometry(configJSON: Data) throws -> HybridCacheGeometry {
        guard let root = (try? JSONSerialization.jsonObject(with: configJSON)) as? [String: Any] else {
            throw ModelConfigDecodeError.malformedJSON
        }
        let textScope = (root["text_config"] as? [String: Any]) ?? [:]
        func geom(_ key: String) -> Any? { textScope[key] ?? root[key] }

        // Family gate (root is authoritative for the VL wrapper; fall back to text scope).
        let rawModelType = (root["model_type"] as? String) ?? (textScope["model_type"] as? String)
        guard let modelType = rawModelType?.lowercased() else {
            throw ModelConfigDecodeError.missingField("model_type")
        }
        guard modelType == "qwen3_5" else {
            throw ModelConfigDecodeError.unsupportedModelType(
                "\(modelType) (qwen35HybridGeometry is qwen3_5-scoped for v1)")
        }

        // Structure.
        let nLayers = try requirePositiveInt(geom, "num_hidden_layers")
        let attnIndices = try hybridAttentionLayerIndices(geom: geom, nLayers: nLayers)
        guard let map = HybridLayerKindMap.from(attentionLayerIndices: attnIndices, layerCount: nLayers) else {
            throw ModelConfigDecodeError.invalidField(
                "attention layer indices \(attnIndices.sorted()) out of range for \(nLayers) layers")
        }

        // Dense K/V geometry + element width from dtype.
        let elementBytes: Int
        switch (geom("torch_dtype") as? String) ?? (geom("dtype") as? String) {
        case "float16", "bfloat16": elementBytes = 2
        case "float32": elementBytes = 4
        case let other:
            throw ModelConfigDecodeError.invalidField("torch_dtype/dtype (\(other ?? "absent"))")
        }
        let dense = DenseKVGeometry(
            kvHeads: try requirePositiveInt(geom, "num_key_value_heads"),
            headDim: try requirePositiveInt(geom, "head_dim"),
            elementBytes: elementBytes)

        // Recurrent conv/SSM geometry. convDim = 2·(Dk·Hk) + (Dv·Hv) (Qwen35.swift:195); conv tail at the
        // model/activation dtype (elementBytes); SSM state fp32 by construction (in RecurrentStateGeometry).
        let nKHeads = try requirePositiveInt(geom, "linear_num_key_heads")
        let nVHeads = try requirePositiveInt(geom, "linear_num_value_heads")
        let kHeadDim = try requirePositiveInt(geom, "linear_key_head_dim")
        let vHeadDim = try requirePositiveInt(geom, "linear_value_head_dim")
        let convKernel = try requirePositiveInt(geom, "linear_conv_kernel_dim")
        let convDim = 2 * (kHeadDim * nKHeads) + (vHeadDim * nVHeads)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: convKernel, convDim: convDim, valueHeads: nVHeads,
            valueHeadDim: vHeadDim, keyHeadDim: kHeadDim, convElementBytes: elementBytes)

        guard let geometry = HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent) else {
            throw ModelConfigDecodeError.invalidField(
                "qwen3_5 config resolves to an all-dense (non-heterogeneous) map — route .fp16, not hybrid")
        }
        return geometry
    }
}
