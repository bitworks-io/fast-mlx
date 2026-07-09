import Foundation

/// The KV-cache growth pattern class, dispatching the per-token formula (spec §2.1). A single
/// formula is wrong for 5 of 14 catalog models — off by 4x-71x — so the capacity model switches
/// on this instead of treating every model as uniform-GQA.
public enum ArchClass: String, Sendable {
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
    /// 4-bit weights estimate in bytes (spec's "0.5B/param" rule, ⚠️ where flagged).
    public let weightsBytes4bitEstimate: Int
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

    public init(
        id: String, modelType: ArchClass, nLayers: Int, nAttnLayers: Int, nKVHeads: Int, headDim: Int,
        slidingWindow: Int? = nil, fixedStateBytes: Int = 0, nativeMaxContext: Int,
        weightsBytes4bitEstimate: Int, license: String,
        mlaHeads: Int? = nil, mlaRopeDim: Int? = nil, mlaNopeDim: Int? = nil, mlaVDim: Int? = nil
    ) {
        self.id = id; self.modelType = modelType; self.nLayers = nLayers; self.nAttnLayers = nAttnLayers
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.slidingWindow = slidingWindow
        self.fixedStateBytes = fixedStateBytes; self.nativeMaxContext = nativeMaxContext
        self.weightsBytes4bitEstimate = weightsBytes4bitEstimate; self.license = license
        self.mlaHeads = mlaHeads; self.mlaRopeDim = mlaRopeDim; self.mlaNopeDim = mlaNopeDim; self.mlaVDim = mlaVDim
    }

    /// Local (rotating/capped) layer count for interleaved-SWA: `nLayers - nAttnLayers`. Meaningless
    /// (and unused) for every other class.
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
        // nKVHeads×headDim = 512 reconciles 20 KiB/tok (naive all-40-layers would be 80 KiB/tok —
        // the 4x the spec calls out). fixedStateBytes = 0: the spec's own KV@32K = 0.625 GiB
        // reconciles from the attention term ALONE (20 KiB × 32768 = 0.625 GiB exactly), so no
        // additional recurrent-state term was needed to hit the confirmed number; the true
        // GatedDeltaNet state cost is not zero, just not separately reconciled here. ⚠️ license
        // "verify" per spec.
        ModelArchProfile(
            id: "Qwen3.6-35B-A3B", modelType: .hybridLinear, nLayers: 40, nAttnLayers: 10,
            nKVHeads: 4, headDim: 128, fixedStateBytes: 0, nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: Int(17.5 * Double(gib)), license: "⚠️ verify"
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
    ]
}
