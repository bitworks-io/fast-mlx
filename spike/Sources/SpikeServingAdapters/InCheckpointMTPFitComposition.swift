import Foundation

import HarnessCore

/// Composes the fit-check profile for a `qwen4_exp` (Qwen3.8-Flash-Next) checkpoint served with the
/// in-checkpoint MTP (multi-token-prediction) drafter enabled (`--qwen4exp-mtp`).
///
/// The catalog entry `ModelArchProfile`'s own doc comment (see `ModelArchProfile.catalog`, the
/// "Qwen3.8-Flash-Next" entries) states plainly what it does NOT model: an MTP-enabled deploy adds a
/// 13th growing K+V cache PLUS a 13th growing QSA-indexer `rawKeys` cache on top of the 12 that
/// `nAttnLayers = 12` counts. That is because the in-checkpoint MTP draft-head predictor builds
/// exactly one additional `.qsa` decoder layer (its own QSA indexer, same geometry as the target
/// model's full-attention layers) and `newCache()` allocates one full-attention KV cache for it —
/// verified directly against the vendored MTP draft-head implementation in `Vendor/mlx-swift-lm`.
/// This type turns that documented gap into a counted one: it composes a fit profile with
/// `nAttnLayers` incremented by exactly one, so `CapacityModel.kvBytesPerToken`'s existing
/// `.hybridLinear` formula — already multiplying BOTH the standard K+V term and the
/// `auxPerLayerKeyDim` indexer term by `nAttnLayers` — picks up the drafter's cache for free, with
/// no separate drafter-specific term to keep in sync.
///
/// Deliberately does NOT touch `weightsBytes4bitEstimate`: the 76 in-checkpoint MTP tensors live in
/// shard 22 of the real artifact and are already included in the measured/declared whole-file total
/// `ModelConfigDecoder.decodeModelDirectory` produces, so the resident-weights figure is already
/// honest with no adjustment. (Composing on top of `NGramOffloadFitComposition` stays correct for the
/// same reason: that composition subtracts only `ple.ple_embedding.ngram_embedding.shard_N.*` keys,
/// which the drafter's tensors are not, so the drafter's weight bytes survive the offload subtraction
/// unaffected either way.)
///
/// Mirrors `NGramOffloadFitComposition.make`'s field-by-field profile rebuild discipline: every field
/// of `base` is copied explicitly except `id` (relabeled to name this composition) and `nAttnLayers`
/// (incremented). A field-by-field rebuild that silently drops one — as a prior composition in this
/// codebase once dropped `auxPerLayerKeyDim` — would silently under-count a term a later reader has
/// no way to notice from this call site alone.
public enum InCheckpointMTPFitComposition {
    /// Builds the corrected `ModelArchProfile` for a `qwen4_exp` checkpoint served with the
    /// in-checkpoint MTP drafter enabled. Pure and non-throwing: unlike `NGramOffloadFitComposition`
    /// and `ExactQwen35MTPCompositeFitProfile`, this composition reads no additional on-disk state.
    /// It DOES have one failure mode to fail closed against, and it is checked first: `base.nAttnLayers
    /// == 0` is `ModelArchProfile`'s deliberate sentinel for "growing-attention-layer count
    /// unconfirmed — do not multiply blind" (see `ModelArchProfile.isKVDerivable`'s doc comment),
    /// which `CapacityModel.classify` turns into an honest RED `.kvNotDerivable` rather than a
    /// fabricated fit. Incrementing that sentinel unconditionally (0 -> 1) would silently convert
    /// "not derivable" into "derivable" and hand `classify` a computed — but fabricated — fit color
    /// for an architecture whose attention-layer count was never confirmed. So `make` refuses to
    /// touch a sentinel profile at all: it returns `base` completely unchanged, INCLUDING `id` (not
    /// relabeled — relabeling would claim a composition happened when none did), leaving
    /// `isKVDerivable` and the resulting `.kvNotDerivable` verdict exactly as honest as they were
    /// before this composition ran.
    ///
    /// Deliberately NOT modeled by this composition, for the record rather than as a future TODO:
    /// MTP-on also holds a transient duplicate of the GatedDeltaNet recurrent state.
    /// `checkpointSpeculativePromptCacheBeforeAppend` (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/
    /// KVCache.swift:2953-2981`) snapshots every recurrent cache entry once per speculative round, and
    /// the `qwen4_exp` per-layer cache's own speculative checkpoint additionally copies the PLE
    /// continuation alongside it. That duplicate is bounded by another fixed-size term — a second
    /// `fixedStateBytes` (115,458,048 B ≈ 110 MiB) — and does not move any context ceiling (it is
    /// context-independent, unlike the `nAttnLayers` term this composition does correct). It is
    /// recorded here rather than added as a term because this composition's contract is the growing
    /// per-attention-layer correction only.
    public static func make(base: ModelArchProfile) -> ModelArchProfile {
        guard base.nAttnLayers > 0 else { return base }
        return ModelArchProfile(
            id: "\(base.id)+qwen4exp-mtp-composition",
            modelType: base.modelType,
            nLayers: base.nLayers,
            nAttnLayers: base.nAttnLayers + 1,
            nKVHeads: base.nKVHeads,
            headDim: base.headDim,
            slidingWindow: base.slidingWindow,
            fixedStateBytes: base.fixedStateBytes,
            nativeMaxContext: base.nativeMaxContext,
            weightsBytes4bitEstimate: base.weightsBytes4bitEstimate,
            license: base.license,
            mlaHeads: base.mlaHeads,
            mlaRopeDim: base.mlaRopeDim,
            mlaNopeDim: base.mlaNopeDim,
            mlaVDim: base.mlaVDim,
            swaKVHeads: base.swaKVHeads,
            swaHeadDim: base.swaHeadDim,
            vHeadDim: base.vHeadDim,
            swaVHeadDim: base.swaVHeadDim,
            auxPerLayerKeyDim: base.auxPerLayerKeyDim)
    }
}
