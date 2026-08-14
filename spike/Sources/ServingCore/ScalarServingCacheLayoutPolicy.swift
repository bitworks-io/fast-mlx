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
