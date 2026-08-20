public enum ScalarServingNativeCacheKind: String, Codable, Equatable, Sendable {
    case denseAttention = "dense-attention"
    case rotatingAttention = "rotating-attention"
    case recurrentState = "recurrent-state"
    case composite
    case unknown
}

public enum ScalarServingCacheLayoutError: Error, Equatable, Sendable {
    case emptyCacheLayout
    case unsupportedCacheLayout(
        index: Int,
        kind: ScalarServingNativeCacheKind)
}

/// Accept only cache layouts that the scalar compiled decoder can replace safely.
///
/// Sliding, recurrent, composite, and unknown native layouts need model-specific state and
/// must never be silently replaced with a dense compiled KV cache.
public func validateScalarServingCacheLayout(
    _ kinds: [ScalarServingNativeCacheKind]
) throws {
    guard !kinds.isEmpty else {
        throw ScalarServingCacheLayoutError.emptyCacheLayout
    }
    for (index, kind) in kinds.enumerated() {
        switch kind {
        case .denseAttention:
            continue
        case .rotatingAttention, .recurrentState, .composite, .unknown:
            throw ScalarServingCacheLayoutError.unsupportedCacheLayout(
                index: index,
                kind: kind)
        }
    }
}

/// Accept the cache layouts the CONTINUOUS-batch route can serve.
///
/// Stricter than the scalar route by default: `.recurrentState` (Qwen3.5 hybrid's GatedDeltaNet-linear
/// layers, `MambaCache`) is admitted ONLY when `hybridAdmitted` is true — i.e. the operator opted in
/// with `--allow-hybrid-qwen35` AND the continuous proof certified the qwen3_5 hybrid geometry. With
/// `hybridAdmitted: false` this is byte-identical to `validateScalarServingCacheLayout`, so the
/// flag-OFF continuous path keeps today's fail-closed behavior exactly (a hybrid model rejects here and
/// the executable falls back to scalar serving).
///
/// `.rotatingAttention` (sliding-window), `.composite`, and `.unknown` stay unsupported on BOTH
/// branches — no audited continuous-route correctness for those families, so opting hybrid in never
/// relaxes them.
public func validateContinuousServingCacheLayout(
    _ kinds: [ScalarServingNativeCacheKind],
    hybridAdmitted: Bool
) throws {
    guard !kinds.isEmpty else {
        throw ScalarServingCacheLayoutError.emptyCacheLayout
    }
    for (index, kind) in kinds.enumerated() {
        switch kind {
        case .denseAttention:
            continue
        case .recurrentState where hybridAdmitted:
            continue
        case .recurrentState, .rotatingAttention, .composite, .unknown:
            throw ScalarServingCacheLayoutError.unsupportedCacheLayout(
                index: index,
                kind: kind)
        }
    }
}

/// Decoder route for the scalar serving path, decided from the model's native per-layer cache
/// shape. `.compiled` uses `CompiledMLXDecoder`'s uniform dense cache (fast path, unchanged
/// behavior). `.nativeHeterogeneous` uses the non-compiled `MLXDecoder`, which consumes the
/// model's own `newCache(parameters:)` directly and so preserves per-layer heterogeneity —
/// required for hybrid architectures (e.g. Qwen3.5's alternating GatedDeltaNet-linear/full-
/// attention layers, `.recurrentState` == `MambaCache`).
///
/// `.rotatingAttention` (sliding-window interleaved, e.g. Gemma-3), `.composite`, and `.unknown`
/// remain unsupported here — different architecture families whose correctness on the
/// non-compiled path has not been independently verified — and still fail closed with the same
/// error as `validateScalarServingCacheLayout`.
public enum ScalarServingDecoderRoute: String, Equatable, Sendable {
    case compiled
    case nativeHeterogeneous
}

/// Model families the continuous-batch route rejects but the scalar serving route
/// has been LIVE-PROVEN to handle. When the executable is launched on the default
/// continuous path (serve.sh appends `--continuous-batch-no-spec`) and the proof
/// rejects the family, membership here — and only here — authorizes an automatic
/// fallback to scalar serving so the operator's one-command launch still works.
///
/// Fail-closed by construction: this is a *serving-proof* allowlist, deliberately
/// narrower than the sizer's hybrid-linear classification (`ModelConfigDecoder`,
/// which also tags `qwen3_next` and the human-gated MoE `qwen3_5_moe`). Only
/// `qwen3_5` has a recorded live scalar-serving proof (generation + `.xmlFunction`
/// tool calls); every other family — proven or not by the sizer — rethrows and
/// leaves `DenseContinuousBatchModelProof.verifying` the sole authority on support.
private let scalarHybridServingFamilies: Set<String> = ["qwen3_5"]

/// Whether `family` (a raw `config.json` `model_type`) is eligible for the
/// continuous→scalar serving fallback. Exact-match only; no case folding or
/// trimming, so a mislabeled tag fails closed rather than silently serving.
public func isScalarHybridServingFamily(_ family: String) -> Bool {
    scalarHybridServingFamilies.contains(family)
}

/// Operator-facing announce line written to stderr when the continuous route
/// rejects a family that `isScalarHybridServingFamily` admits and the executable
/// falls back to scalar serving. Machine-readable with a fixed key order so the
/// live gate (and evidence tooling) can assert the fallback happened, mirroring
/// the fit-check announce style. This is the ONLY place the fallback is announced;
/// the startup line keeps `mode=scalar` parity with a native scalar launch.
public func scalarHybridFallbackAnnounceLine(modelType: String) -> String {
    "fastmlx-serve continuous_fallback=scalar model_type=\(modelType) "
        + "reason=unsupported_continuous_family"
}

/// Operator-facing announce line written to stderr when `--allow-hybrid-qwen35` admits a hybrid
/// family onto the CONTINUOUS route (the proof carried it through instead of throwing
/// `unsupportedModelFamily`). Machine-readable with a fixed key order, mirroring the fallback line,
/// so a misconfigured opt-in doesn't silently mask as a plain scalar fallback: the continuous route
/// announces the admission explicitly. Emitted only when the hybrid family was actually admitted.
public func hybridQwen35ContinuousAdmissionAnnounceLine(modelType: String) -> String {
    "fastmlx-serve continuous_admitted=hybrid model_type=\(modelType) "
        + "reason=allow_hybrid_qwen35"
}

public func classifyScalarServingDecoderRoute(
    _ kinds: [ScalarServingNativeCacheKind]
) throws -> ScalarServingDecoderRoute {
    guard !kinds.isEmpty else {
        throw ScalarServingCacheLayoutError.emptyCacheLayout
    }
    for (index, kind) in kinds.enumerated() {
        switch kind {
        case .denseAttention, .recurrentState:
            continue
        case .rotatingAttention, .composite, .unknown:
            throw ScalarServingCacheLayoutError.unsupportedCacheLayout(
                index: index,
                kind: kind)
        }
    }
    return kinds.contains(.recurrentState) ? .nativeHeterogeneous : .compiled
}

/// Which cold-prefix-snapshot reuse rule a model's native cache layout permits — the serving-side
/// answer a future restore orchestration consults so it never applies dense block-floor arithmetic
/// to a layout that can't support it. Mirrors the `ColdSnapshotReuseGranularity` cases in HarnessCore
/// but is declared LOCALLY so ServingCore stays zero-dependency; a future consumer maps between them.
///
/// - `.blockAligned`: all-`.denseAttention`. Each whole block's K/V is independent, so a divergent
///   prompt still reuses every block before the first divergence.
/// - `.wholeSnapshotOnly`: the layout contains `.recurrentState` (`MambaCache`/GatedDeltaNet). The
///   recurrent state exists only at the stored token count and can't be rewound to an earlier
///   boundary, so a snapshot is reusable only in full.
/// - `.unsupported`: any `.rotatingAttention` (a sliding window can't be restored at an arbitrary
///   boundary either), `.composite`, `.unknown`, or an empty layout — no audited reuse rule, fail
///   closed. `.unsupported` dominates: a layout mixing recurrent AND rotating state is unsupported.
public enum ScalarServingSnapshotReuse: String, Equatable, Sendable {
    case blockAligned = "block-aligned"
    case wholeSnapshotOnly = "whole-snapshot-only"
    case unsupported
}

/// Classify a model's per-layer native cache layout into the snapshot-reuse rule it permits. Total
/// and non-throwing (unlike the sibling classifiers): `.unsupported` is an explicit RESULT here, so
/// an unaudited or empty layout returns `.unsupported` rather than raising — the caller branches on
/// the value. Priority: any rotating/composite/unknown (or empty) → `.unsupported`; else any
/// recurrent → `.wholeSnapshotOnly`; else all dense → `.blockAligned`.
public func snapshotReuseGranularity(
    for kinds: [ScalarServingNativeCacheKind]
) -> ScalarServingSnapshotReuse {
    guard !kinds.isEmpty else { return .unsupported }
    for kind in kinds {
        switch kind {
        case .denseAttention, .recurrentState:
            continue
        case .rotatingAttention, .composite, .unknown:
            return .unsupported
        }
    }
    return kinds.contains(.recurrentState) ? .wholeSnapshotOnly : .blockAligned
}
