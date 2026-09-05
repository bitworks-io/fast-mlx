import Foundation

/// The KV-cache growth pattern class, dispatching the per-token formula (spec §2.1). A single
/// formula is wrong for 5 of 14 catalog models — off by 4x-71x — so the capacity model switches
/// on this instead of treating every model as uniform-GQA.
public enum ArchClass: String, Sendable, CaseIterable {
    /// Every transformer layer runs standard GQA attention and grows its KV cache every token.
    /// `KVCacheSimple`/`StandardKVCache`. Qwen3 dense/MoE, Llama, Mistral, GLM-4.5-Air/4.6, Phi-4,
    /// Qwen3-235B.
    case uniformGQA
    /// Only `nLayers / interval` layers are GQA attention (and grow); the rest are GatedDeltaNet
    /// linear-attention layers with a fixed recurrent state (`MambaCache`). `qwen3_next`/
    /// `qwen3_5_moe` (Qwen3.6-35B-A3B, Ornith).
    case hybridLinear
    /// A fixed ratio of global (`StandardKVCache`, grows with full context) to local
    /// (`RotatingKVCache`, capped at a sliding window) attention layers, interleaved. Gemma-3.
    case interleavedSWA
    /// Like `.interleavedSWA` (global growing + local window-capped layers) BUT the two layer
    /// classes carry DIFFERENT head geometry, AND every layer also holds a fixed recurrent/conv
    /// state. The growing term uses the global geometry (`nKVHeads`/`headDim` over `nAttnLayers`),
    /// the window-capped term uses the SWA geometry (`swaKVHeads`/`swaHeadDim` over `nLocalLayers`),
    /// and `fixedStateBytes` is added on top — all three simultaneously. Neither `.interleavedSWA`
    /// (one geometry, drops `fixedStateBytes`) nor `.hybridLinear` (drops the window cap) can
    /// express this. Baichuan-M1 (`baichuan_m1`).
    case dualGeometrySWA
    /// DeepSeek-V3/R1 as the vendored Swift arch actually caches it today: `kv_b_proj`
    /// decompresses MLA back to per-head K/V *before* the cache write, so the cache holds the
    /// full decompressed per-head geometry, not the compact MLA latent. Huge. (The compact
    /// "absorbed" MLA cache is 71x smaller but unbuilt — spec §7 backlog.)
    case mlaAsImplemented
    /// Mamba2 SSM layers (O(1) state) interleaved with a minority of "select" attention layers
    /// that grow, plus MoE-FFN layers with no attention at all. Nemotron-3-Ultra (`nemotron_h`).
    case hybridMamba2MoE
    /// CSA/HCA per-layer compressed attention (DeepSeek-V4-Flash `deepseek_v4`) — not MLX-servable
    /// today (no vendored arch; `ds4`-only). Out of scope; the tunable does not apply (spec §8).
    case novelCompressedUnsupported
}

/// Checked-in per-model catalog data (spec §2.2). Every field here is data, not computed — the
/// arithmetic lives in `CapacityModel`. ⚠️ markers on individual entries are carried verbatim from
/// the spec's research; do not "clean up" an inferred number into a confirmed-looking one.
public struct ModelArchProfile: Sendable {
    public let id: String
    public let modelType: ArchClass
    /// Total transformer/hybrid block count.
    public let nLayers: Int
    /// The count of layers whose KV cache actually grows with context: == `nLayers` for
    /// uniform-GQA; the attention-layer subset for hybrid-linear, interleaved-SWA (global layers
    /// only — see `nLocalLayers`), MLA, and hybrid-Mamba2+MoE.
    public let nAttnLayers: Int
    public let nKVHeads: Int
    public let headDim: Int
    /// Sliding-window size for interleaved-SWA local layers (`RotatingKVCache` cap). `nil` for
    /// every other class.
    public let slidingWindow: Int?
    /// Fixed per-sequence recurrent/SSM state in bytes (hybrid-linear `MambaCache`, hybrid-Mamba2
    /// state) — a per-sequence constant, not a per-token rate. `0` where the spec's reconciled
    /// KV@32K number required no additional term (documented per entry below), not a claim the
    /// true fixed cost is zero.
    public let fixedStateBytes: Int
    public let nativeMaxContext: Int
    /// 4-bit weights estimate in bytes (spec's "0.5B/param" rule, ⚠️ where flagged). Settable
    /// (internal-set only — every other field on this type stays `let`) SPECIFICALLY so
    /// `ModelSizer.scaledModel` can copy-and-modify a whole profile instead of re-listing every
    /// field to rebuild one. A field-by-field rebuild is what silently dropped `swaKVHeads`/
    /// `swaHeadDim`/`vHeadDim`/`swaVHeadDim`/`auxPerLayerKeyDim` from every sizer-report row —
    /// re-listing fields is a defect generator, not a fix.
    public internal(set) var weightsBytes4bitEstimate: Int
    public let license: String

    // MARK: MLA-as-implemented geometry (DeepSeek-R1 only; nil for every other class)
    /// Decompressed attention heads (128 for R1).
    public let mlaHeads: Int?
    /// Per-head RoPE-carrying query/key dim (64 for R1).
    public let mlaRopeDim: Int?
    /// Per-head non-RoPE query/key dim (128 for R1).
    public let mlaNopeDim: Int?
    /// Per-head value dim (128 for R1).
    public let mlaVDim: Int?

    // MARK: Dual-geometry SWA (`.dualGeometrySWA` only; nil for every other class)
    /// KV-head count for the SWA (window-capped) layers when it differs from the global-layer
    /// `nKVHeads`. `nil` outside `.dualGeometrySWA` (and `swaFixedLocalBytes` falls back to
    /// `nKVHeads` if unexpectedly nil, matching the vendored runtime's own head-count fallback).
    public let swaKVHeads: Int?
    /// Per-head dim for the SWA (window-capped) layers when it differs from the global-layer
    /// `headDim`. `nil` outside `.dualGeometrySWA` (falls back to `headDim`).
    public let swaHeadDim: Int?
    /// Per-head VALUE dim for the GLOBAL layers when K and V are cached at DIFFERENT per-head dims
    /// (asymmetric attention, e.g. mimo_v2_flash: K at `headDim`, V narrower). `nil` for the common
    /// symmetric case, where the KV formula's `headDim + (vHeadDim ?? headDim) = 2·headDim` recovers
    /// the standard `×2` — so every symmetric model's numbers are unchanged.
    public let vHeadDim: Int?
    /// Per-head VALUE dim for the SWA (window-capped) layers when it differs from `swaHeadDim`.
    /// `nil` falls back to `swaHeadDim` (symmetric). Only meaningful for `.dualGeometrySWA`.
    public let swaVHeadDim: Int?

    // MARK: Auxiliary growing per-layer cache (`.hybridLinear` only; nil for every other class and
    // every existing hybrid-linear member)
    /// A per-attention-layer auxiliary key cache width that GROWS with sequence length, on top of
    /// the standard K+V cache already counted by `nKVHeads`/`kvHeadDimSum`. `nil` for every family
    /// that has no such second growing cache (every hybrid-linear entry to date). Currently used
    /// only by the `qwen4_exp`/`qwen4_exp_text` (Qwen3.8-Flash-Next) QSA indexer: each full-attention
    /// layer's indexer keeps its own `rawKeys` cache of width `indexer_head_dim`, independent of
    /// `indexer_kv_heads`, NOT capped by `indexer_budget` (that budget caps sparse selection, not the
    /// stored raw keys) — verified directly against the vendored sparse-attention indexer
    /// implementation. Omitting this term under-counts Flash Next's KV footprint
    /// by ~12.5%.
    public let auxPerLayerKeyDim: Int?

    public init(
        id: String, modelType: ArchClass, nLayers: Int, nAttnLayers: Int, nKVHeads: Int, headDim: Int,
        slidingWindow: Int? = nil, fixedStateBytes: Int = 0, nativeMaxContext: Int,
        weightsBytes4bitEstimate: Int, license: String,
        mlaHeads: Int? = nil, mlaRopeDim: Int? = nil, mlaNopeDim: Int? = nil, mlaVDim: Int? = nil,
        swaKVHeads: Int? = nil, swaHeadDim: Int? = nil, vHeadDim: Int? = nil, swaVHeadDim: Int? = nil,
        auxPerLayerKeyDim: Int? = nil
    ) {
        self.id = id; self.modelType = modelType; self.nLayers = nLayers; self.nAttnLayers = nAttnLayers
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.slidingWindow = slidingWindow
        self.fixedStateBytes = fixedStateBytes; self.nativeMaxContext = nativeMaxContext
        self.weightsBytes4bitEstimate = weightsBytes4bitEstimate; self.license = license
        self.mlaHeads = mlaHeads; self.mlaRopeDim = mlaRopeDim; self.mlaNopeDim = mlaNopeDim; self.mlaVDim = mlaVDim
        self.swaKVHeads = swaKVHeads; self.swaHeadDim = swaHeadDim
        self.vHeadDim = vHeadDim; self.swaVHeadDim = swaVHeadDim
        self.auxPerLayerKeyDim = auxPerLayerKeyDim
    }

    /// K+V per-head dim sum for the GLOBAL layers: `headDim + (vHeadDim ?? headDim)`. Recovers the
    /// standard `2·headDim` when K and V are symmetric (`vHeadDim == nil`), so symmetric models are
    /// unchanged; models with a narrower/wider V (asymmetric attention) size their cache honestly.
    public var kvHeadDimSum: Int { headDim + (vHeadDim ?? headDim) }
    /// K+V per-head dim sum for the SWA (window-capped) layers: `swaHeadDim + (swaVHeadDim ?? swaHeadDim)`,
    /// falling back to the global sum when no SWA geometry is set.
    public var swaKVHeadDimSum: Int {
        let k = swaHeadDim ?? headDim
        let v = swaVHeadDim ?? swaHeadDim ?? vHeadDim ?? headDim
        return k + v
    }

    /// Local (rotating/capped) layer count for interleaved-SWA and dual-geometry-SWA:
    /// `nLayers - nAttnLayers`. Meaningless (and unused) for every other class.
    public var nLocalLayers: Int { nLayers - nAttnLayers }

    /// Whether this model's TOTAL KV cost is derivable from confirmed config. `false` when the
    /// growing-attention-layer count is an unconfirmed sentinel (`nAttnLayers == 0`, used for
    /// hybrid-Mamba2+MoE where the spec flags the count "unconfirmed — do not multiply blind") or
    /// the architecture is out of scope (novel-compressed, `ds4`-only). In those cases the capacity
    /// model MUST surface "not derivable" rather than return a one-layer under-count or a fabricated
    /// zero that could make an unservable model look like it fits (spec §2.1/§8, the honesty cases).
    /// Adapts automatically: confirm the attention-layer count later and the entry becomes derivable.
    public var isKVDerivable: Bool {
        modelType != .novelCompressedUnsupported && nAttnLayers > 0
    }

    private static let gib = 1024 * 1024 * 1024

    /// Full catalog, spec §2.2 table verbatim. GiB values are converted to bytes at load time;
    /// everything else (layer counts, head geometry) is transcribed directly.
    public static let catalog: [ModelArchProfile] = [
        // ✅ nativeMaxContext = 262,144 (the ⚠️ ~1M opt-in YaRN max is NOT encoded — shipped default only).
        // nKVHeads×headDim = 512 reconciles the spec's 96 KiB/tok @ 48 layers exactly.
        ModelArchProfile(
            id: "Qwen3-30B-A3B-2507", modelType: .uniformGQA, nLayers: 48, nAttnLayers: 48,
            nKVHeads: 4, headDim: 128, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(15.25 * Double(gib)), license: "Apache-2.0"
        ),
        // hybrid-linear: 10 of 40 layers are GQA attention (interval 4); rest are GatedDeltaNet.
        // GROWING term: nKVHeads×headDim = 512 reconciles 20 KiB/tok (naive all-40-layers would be
        // 80 KiB/tok — the 4x the spec calls out); the spec's own KV@32K = 0.625 GiB is this attention
        // term ALONE (20 KiB × 32768 = 0.625 GiB exactly). FIXED term: the 30 linear layers' per-sequence
        // GatedDeltaNet recurrent+conv state, previously left at 0 (unreconciled), is now source-verified
        // from the real config.json text_config (linear_num_key_heads 16, linear_num_value_heads 32,
        // linear_key_head_dim 128, linear_value_head_dim 128, linear_conv_kernel_dim 4): per_layer =
        // conv (4−1)·8192·2 = 49,152 + ssm 32·128·128·4 = 2,097,152 = 2,146,304; × 30 linear layers =
        // 64,389,120 B ≈ 61.41 MiB/seq (identical per-layer to Qwen3.5-9B; only the linear-layer count
        // differs, 30 vs 24). Attention geometry corrected from the same text_config to the real
        // num_key_value_heads 2 × head_dim 256 (was 4×128; product 512 unchanged, so KV/tok is identical
        // — an honesty fix, not a number change). ⚠️ license "verify" per spec.
        ModelArchProfile(
            id: "Qwen3.6-35B-A3B", modelType: .hybridLinear, nLayers: 40, nAttnLayers: 10,
            nKVHeads: 2, headDim: 256, fixedStateBytes: 64_389_120, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(17.5 * Double(gib)), license: "⚠️ verify"
        ),
        // hybrid-linear, the live hybrid-serving verification checkpoint
        // (mlx-community/Qwen3.5-9B-MLX-4bit). Confirmed from its config.json text_config:
        // 32 layers, layer_types = 8×full_attention + 24×linear_attention (full_attention_interval
        // 4), so nAttnLayers = 8 (only the full-attention layers grow a KV cache); nKVHeads = 4,
        // head_dim = 256. GROWING KV/tok = 8 × 4 × 256 × 2 × 2 B(fp16) = 32 KiB/tok → exactly 1.0 GiB @ 32K.
        // FIXED term: the 24 linear_attention layers carry recurrent conv/SSM state (linear_conv_kernel_dim
        // 4, 16 key / 32 value heads × 128 dim), source-verified from the checkpoint's config.json and
        // reconciled against the vendored MambaCache allocation: per_layer = conv (4−1)·8192·2 = 49,152 +
        // ssm 32·128·128·4 = 2,097,152 = 2,146,304; × 24 linear layers = 51,511,296 B ≈ 49.13 MiB/seq.
        // Previously 0 (unreconciled); now separately encoded so the off-box catalog matches the live
        // decoder. ⚠️ This checkpoint is a vision-language model (Qwen3_5ForConditionalGeneration); the
        // geometry above is the text decoder's and drives text-serving KV. weights estimate ⚠️ approx
        // (includes the vision tower). nativeMaxContext = 262,144 (max_position_embeddings).
        ModelArchProfile(
            id: "Qwen3.5-9B", modelType: .hybridLinear, nLayers: 32, nAttnLayers: 8,
            nKVHeads: 4, headDim: 256, fixedStateBytes: 51_511_296, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(5.5 * Double(gib)), license: "⚠️ verify"
        ),
        // nKVHeads×headDim = 1024 reconciles 184 KiB/tok @ 46 layers.
        ModelArchProfile(
            id: "GLM-4.5-Air", modelType: .uniformGQA, nLayers: 46, nAttnLayers: 46,
            nKVHeads: 8, headDim: 128, nativeMaxContext: 131_072,
            weightsBytes4bitEstimate: Int(53 * Double(gib)), license: "MIT"
        ),
        // interleaved-SWA: 10 global + 52 local layers (62 total), window 1024. nKVHeads×headDim =
        // 2048 reconciles BOTH the 80 KiB/tok global growth rate and the 0.406 GiB fixed local-cap
        // term (52 × 2048 × 2 × 2 bytes × 1024 window = 0.4063 GiB) against the spec's KV@32K =
        // 2.91 GiB (2.5 GiB growth + 0.406 GiB fixed). ⚠️ Gemma license needs legal sign-off per spec.
        ModelArchProfile(
            id: "Gemma-3-27B", modelType: .interleavedSWA, nLayers: 62, nAttnLayers: 10,
            nKVHeads: 8, headDim: 256, slidingWindow: 1024, nativeMaxContext: 131_072,
            weightsBytes4bitEstimate: Int(13.5 * Double(gib)), license: "⚠️ Gemma (legal sign-off)"
        ),
        // ✅ nativeMaxContext = 40,960 "field" value per spec (the 32,768 "native" figure is ⚠️).
        // nKVHeads×headDim = 1024 reconciles 256 KiB/tok @ 64 layers exactly (8.0 GiB @32K).
        ModelArchProfile(
            id: "Qwen3-32B", modelType: .uniformGQA, nLayers: 64, nAttnLayers: 64,
            nKVHeads: 8, headDim: 128, nativeMaxContext: 40_960,
            weightsBytes4bitEstimate: Int(16 * Double(gib)), license: "Apache-2.0"
        ),
        ModelArchProfile(
            id: "Llama-3.3-70B", modelType: .uniformGQA, nLayers: 80, nAttnLayers: 80,
            nKVHeads: 8, headDim: 128, nativeMaxContext: 131_072,
            weightsBytes4bitEstimate: Int(35 * Double(gib)), license: "⚠️ Llama Community (<700M MAU)"
        ),
        ModelArchProfile(
            id: "Mistral-Small-3.2-24B", modelType: .uniformGQA, nLayers: 40, nAttnLayers: 40,
            nKVHeads: 8, headDim: 128, nativeMaxContext: 131_072,
            weightsBytes4bitEstimate: Int(12 * Double(gib)), license: "Apache-2.0"
        ),
        // Hard ceiling 16,384 < the 32K tunable default — the honesty case spec §8 calls out;
        // nKVHeads×headDim = 1280 reconciles 200 KiB/tok @ 40 layers.
        ModelArchProfile(
            id: "Phi-4-14B", modelType: .uniformGQA, nLayers: 40, nAttnLayers: 40,
            nKVHeads: 10, headDim: 128, nativeMaxContext: 16_384,
            weightsBytes4bitEstimate: Int(7 * Double(gib)), license: "MIT"
        ),
        // ✅ nativeMaxContext = 40,960 "field" value (32,768 ⚠️ native; 262,144 ⚠️ Instruct-2507 —
        // neither of the latter two encoded). nKVHeads×headDim = 512 reconciles 188 KiB/tok @ 94 layers.
        ModelArchProfile(
            id: "Qwen3-235B-A22B", modelType: .uniformGQA, nLayers: 94, nAttnLayers: 94,
            nKVHeads: 4, headDim: 128, nativeMaxContext: 40_960,
            weightsBytes4bitEstimate: Int(117.5 * Double(gib)), license: "Apache-2.0"
        ),
        // MLA-as-implemented, the honesty case (spec §8): decompressed per-head cache, 128 heads ×
        // (rope 64 + nope 128 + v 128) × 2 bytes × 61 layers = 4880 KiB/tok, which reconciles the
        // spec's confirmed KV@32K = 152.5 GiB EXACTLY (4880 KiB × 32768 / 1024² = 152.51 GiB). The
        // spec's own "4.88 MiB/tok" figure is ~2.4% off this (4880 KiB = 4.766 MiB, not 4.88 MiB) —
        // almost certainly a KiB/1000-vs/1024 rounding slip in the spec's prose; the KV@32K number
        // is the one independently reconciled here, so it is treated as authoritative. ⚠️ weights
        // 335.5 GiB per spec.
        ModelArchProfile(
            id: "DeepSeek-R1", modelType: .mlaAsImplemented, nLayers: 61, nAttnLayers: 61,
            nKVHeads: 0, headDim: 0, nativeMaxContext: 163_840,
            weightsBytes4bitEstimate: Int(335.5 * Double(gib)), license: "MIT",
            mlaHeads: 128, mlaRopeDim: 64, mlaNopeDim: 128, mlaVDim: 128
        ),
        // nKVHeads×headDim = 1024 reconciles 368 KiB/tok @ 92 layers. ⚠️ weights ~177.5 GiB per spec.
        ModelArchProfile(
            id: "GLM-4.6", modelType: .uniformGQA, nLayers: 92, nAttnLayers: 92,
            nKVHeads: 8, headDim: 128, nativeMaxContext: 202_752,
            weightsBytes4bitEstimate: Int(177.5 * Double(gib)), license: "MIT"
        ),
        // novel-compressed, OUT OF SCOPE (spec §2.1/§8): no vendored `deepseek_v4` arch, not
        // MLX-servable. KV geometry "not derived" in the spec — encoded as zeros rather than a
        // fabricated formula; `CapacityModel` special-cases this class to return 0 and flags it as
        // not applicable, never silently computing a number.
        ModelArchProfile(
            id: "DeepSeek-V4-Flash", modelType: .novelCompressedUnsupported, nLayers: 0, nAttnLayers: 0,
            nKVHeads: 0, headDim: 0, nativeMaxContext: 1_048_576,
            weightsBytes4bitEstimate: Int(86.7 * Double(gib)), license: "unknown (out of scope, ds4-only)"
        ),
        // ⚠️ EVERYTHING here is inferred: the spec flags the attention-layer count as
        // "unconfirmed — do not multiply blind". nKVHeads×headDim = 256 is chosen ONLY to
        // reconcile the spec's "⚠️ 1 KiB/tok/attn-layer" figure (256 × 2 × 2 bytes = 1024 bytes =
        // 1 KiB); it is not read from a config. `nAttnLayers = 0` is a deliberate sentinel meaning
        // "unknown, do not trust a total" — `CapacityModel.kvBytesPerToken` special-cases this
        // class to return the PER-LAYER unit and surface non-derivability rather than multiplying
        // by this placeholder. ⚠️ OpenMDW-1.1 license needs legal sign-off per spec.
        ModelArchProfile(
            id: "Nemotron-3-Ultra", modelType: .hybridMamba2MoE, nLayers: 0, nAttnLayers: 0,
            nKVHeads: 2, headDim: 128, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(275 * Double(gib)), license: "⚠️ OpenMDW-1.1 (legal sign-off)"
        ),
        // ⚠️ hybrid-linear like Qwen3.6, but the spec marks the full-attention interval "assumed"
        // for Ornith. nLayers=60/interval=4→nAttnLayers=15 and nKVHeads×headDim=512 (same family
        // geometry as the Qwen3 line it's post-trained on) are chosen ONLY to reconcile the spec's
        // ⚠️ KV@32K = 0.9375 GiB (30 KiB/tok × 32768 = 0.9375 GiB exactly) — not independently
        // confirmed layer counts. ⚠️ MIT license "confirm at pull" per spec.
        ModelArchProfile(
            id: "Ornith-1.0-397B", modelType: .hybridLinear, nLayers: 60, nAttnLayers: 15,
            nKVHeads: 4, headDim: 128, fixedStateBytes: 0, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(198.5 * Double(gib)), license: "⚠️ MIT (confirm at pull)"
        ),
        // dual-geometry-SWA (baichuan_m1), the fit-check coverage entry for this ArchClass.
        // Geometry audited directly against the real published config (docs/task-inbox/
        // 2026-08-19-baichuan-m1-dual-geometry-arch-reach.md) and pinned exactly by
        // `ModelConfigDecoderTests.testBaichuanM1_dualGeometrySWA_realConfig` +
        // `DualGeometrySWACatalogTests` (decode-drift guard): 40 layers, 20 global (growing) + 20
        // sliding-window layers; global 2 kv × 256 dim, SWA 8 kv × 128 dim (2x the global product —
        // NOT symmetric); sliding_window 8192; fixedStateBytes 122,880 (per-layer conv state, both
        // layer classes); nativeMaxContext 32,768. KV@32K = 2,013,388,800 B (fp16) reconciled exactly.
        // weightsBytes4bitEstimate is MEASURED from the published mlx-community/Baichuan-M1-14B-
        // Instruct-4bit artifact (HF safetensors dtype breakdown, 2026-08-20): BF16 452,613,920×2
        // + U32 1,808,793,600×4 = 8,140,402,240 B = 7.58 GiB.
        ModelArchProfile(
            id: "Baichuan-M1-14B", modelType: .dualGeometrySWA, nLayers: 40, nAttnLayers: 20,
            nKVHeads: 2, headDim: 256, slidingWindow: 8192, fixedStateBytes: 122_880,
            nativeMaxContext: 32_768,
            weightsBytes4bitEstimate: Int(7.58 * Double(gib)), license: "Apache-2.0",
            swaKVHeads: 8, swaHeadDim: 128
        ),
        // dual-geometry-SWA with ASYMMETRIC K/V (mimo_v2_flash), the second fit-check coverage entry
        // for this ArchClass — exercises `vHeadDim`/`swaVHeadDim` (narrower cached V than K) which
        // Baichuan-M1 (symmetric) does not touch. Geometry audited against the real published config
        // (docs/task-inbox/2026-08-19-mimo-v2-flash-dual-geometry-vheaddim.md) and pinned exactly by
        // `ModelConfigDecoderTests.testMimoV2Flash_dualGeometrySWA_asymmetricKV_realConfig` +
        // `DualGeometrySWACatalogTests`: 48 layers, 9 global (growing, `hybrid_layer_pattern`) + 39
        // sliding-window layers; global 4 kv, K dim 192 / V dim 128; SWA 8 kv, K dim 192 / V dim 128;
        // sliding_window 128; fixedStateBytes 0 (no conv/recurrent state); nativeMaxContext 262,144.
        // KV@32K = 780,533,760 B (fp16) reconciled exactly.
        // ⚠️ MiMo-V2-Flash is a ~309B-parameter fp8-native MoE, NOT a small model. weightsBytes4bit-
        // Estimate is MEASURED from the published mlx-community/MiMo-V2-Flash-4bit artifact (HF
        // safetensors dtype breakdown, 2026-08-20): BF16 9,697,466,816×2 + U32 38,591,135,744×4 +
        // F32 12,032×4 = 173,759,524,736 B = 161.83 GiB. This EXCEEDS the 128 GB M5's RAM — a
        // weights-larger-than-RAM (#3b) case — so the fit-check correctly reds it on every current
        // box; that honest refusal is the differentiator working, not a bug. (An earlier proportional
        // guess of 17.5 GiB under-counted this by ~9x — the exact phantom-GREEN the fit-check prevents.)
        ModelArchProfile(
            id: "MiMo-V2-Flash", modelType: .dualGeometrySWA, nLayers: 48, nAttnLayers: 9,
            nKVHeads: 4, headDim: 192, slidingWindow: 128, fixedStateBytes: 0,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(161.83 * Double(gib)), license: "unknown (see model card)",
            swaKVHeads: 8, swaHeadDim: 192, vHeadDim: 128, swaVHeadDim: 128
        ),
        // hybrid-linear "Flash Next". Geometry confirmed directly from the real artifact's
        // config.json text_config: 48 layers, layer_types = 12x full_attention + 36x
        // linear_attention (full_attention_interval 4) -> nAttnLayers = 12 (only the
        // full-attention layers grow a KV cache); num_key_value_heads 2, head_dim 256 ->
        // GROWING KV/tok, K+V term only = 12 x 2 x (256+256) x 2 B(fp16) = 24,576 B/tok. Plus the
        // QSA indexer term below (12 x 128 x 2 B = 3,072 B/tok): TOTAL GROWING KV/tok = 24,576 +
        // 3,072 = 27,648 B/tok -> exactly 7,247,757,312 B (6.75 GiB) at the model's own native
        // max, 262,144 tokens (27,648 x 262,144 = 7,247,757,312). The earlier "24,576 B/tok ->
        // 6,442,450,944 B (6.00 GiB)" figure that stood here omitted the indexer term entirely —
        // corrected once that term was counted (see the indexer note below).
        // nativeMaxContext = 262,144 (max_position_embeddings).
        //
        // FIXED term: the 36 linear_attention layers' per-sequence gated-delta-net recurrent+conv
        // state, reconciled against the VENDORED cache allocation rather than config geometry
        // alone (per the Qwen3.5-9B convention above). convDim = keyDim*2 + valueDim =
        // (16x128)x2 + (48x128) = 4,096 + 6,144 = 10,240, from config's linear_num_key_heads 16 /
        // linear_key_head_dim 128 / linear_num_value_heads 48 / linear_value_head_dim 128. The
        // conv-state cache allocates [batch, convKernelSize-1, convDim] at the model's bf16/fp16
        // compute dtype (2 B/elem) = (4-1) x 10,240 x 2 = 61,440 B. The recurrent/SSM state's
        // shape is [batch, numVHeads, headVDim, headKDim] = [_, 48, 128, 128] and the vendored
        // code asserts its dtype is ALWAYS float32 regardless of compute dtype: 48 x 128 x 128 x
        // 4 B = 3,145,728 B. Per linear layer = 61,440 + 3,145,728 = 3,207,168 B; x 36 linear
        // layers (48 total - 12 full_attention) = 115,458,048 B (110.11 MiB/seq). Both shapes and
        // dtypes were confirmed against the vendored source, not inferred from config alone.
        //
        // weightsBytes4bitEstimate is the MEASURED resident-after-offload figure read from the
        // artifact's 22 safetensors headers: total 113,324,747,928 B (105.54 GiB) MINUS the
        // 32,000,153,600 B (29.80 GiB) per-layer-embedding n-gram lookup table =
        // 81,324,594,328 B (75.74 GiB). This is NOT a 0.5B/param estimate and NOT the full on-disk
        // size -- it EXCLUDES the n-gram table because that table is designed to be SSD-streamed
        // rather than held resident. WARNING, LOAD-BEARING: this number is only valid once that
        // offload is actually wired into the serving path. There is no production call site for it
        // yet, so any fit verdict consuming this figure describes the offload-enabled
        // configuration, not the as-shipped one -- loading the full 105.54 GiB would fail the green
        // verdict most of the current fleet gives it here.
        //
        // Consequence for the sizer matrix: the artifact is MIXED precision (base 4-bit/group-32
        // with 5- and 8-bit per-module overrides), so the `weightBits: 8` row the matrix emits by
        // doubling this figure is a mechanical extrapolation with no corresponding real build. It
        // is kept only for matrix uniformity; read the 4-bit row for this model.
        //
        // COUNTED as of the cycle-35 correction, via `auxPerLayerKeyDim: 128` below. This entry
        // previously said the sparse-attention indexer's cache was "indexer_budget 2048 x
        // indexer_kv_heads 1 x indexer_head_dim 128" and had no term for it. That description was
        // WRONG in the direction that hides cost, and is corrected here:
        //   - `indexer_budget` does NOT bound this cache. The budget caps sparse SELECTION -- how
        //     many tokens attention picks -- not the stored raw keys. Verified directly against the
        //     vendored sparse-attention indexer implementation:
        //     `concatenated([existingRawKeys, newRawKeys], axis: 1)` with no truncation on that
        //     path, and a subsequent assertion that `rawKeys.dim(1) == keyLength`, the FULL key
        //     length.
        //   - The width is exactly `indexer_head_dim` (128), independent of `indexer_kv_heads`:
        //     `newRawKeys = qkProjection[0..., 0..., split...].reshaped(batch, seq, headDim)`.
        // So it is a GROWING per-attention-layer term, not a fixed per-sequence one: 12 layers x 128
        // x 2 B = 3,072 B/token, i.e. ~805 MB at the 262,144 native context, not the ~262 KB the old
        // budget-based description implied -- roughly 3,000x larger. Sized at model dtype (2 B) and
        // NOT at the KV-quant bytes/element, because `rawKeys` is a projection output held at model
        // dtype rather than part of the quantizable KV cache; see `CapacityModel.kvBytesPerToken`.
        //
        // NOT counted by this entry or by `CapacityModel`: an MTP (multi-token-prediction) deploy
        // adds a 13th growing K+V cache PLUS a 13th growing indexer cache on top of the 12 that
        // `nAttnLayers = 12` models here. The MTP draft head builds its own `.qsa` decoder layer
        // with its own indexer and allocates its own growing cache (verified directly against the
        // vendored MTP draft-head implementation) — this entry has no attention-layer
        // count that includes it. Roughly +1/12 (~8%) more growing KV/tok than the 27,648 B/tok
        // figure above once MTP is in the serving path. Not modeled here; treat any fit verdict
        // for an MTP-enabled deploy as an under-count by that margin, not as already covering it.
        // WARNING: Qwen Community License 1.0 -- flagged for legal verification, not a confirmed
        // clearance (same convention as the other flagged entries above).
        ModelArchProfile(
            id: "Qwen3.8-Flash-Next", modelType: .hybridLinear, nLayers: 48, nAttnLayers: 12,
            nKVHeads: 2, headDim: 256, fixedStateBytes: 115_458_048, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: 81_324_594_328, license: "⚠️ Qwen Community License 1.0 (verify)",
            auxPerLayerKeyDim: 128
        ),
    ]
}
