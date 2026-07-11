import Foundation
import MLX
import MLXLMCommon

/// The cache contract `CompiledMLXDecoder` actually depends on, beyond MLXLMCommon's
/// `KVCache`: compile-capturable state (`Updatable`), chunked growth between steps (one
/// retrace per chunk), and identity-preserving in-place reset. `CompiledKVCache` (fp16)
/// and `TurboQuantKVCache` (quantized codes, materialize-then-attend) both conform, so
/// the decoder is selected per KV tier instead of hardcoding the fp16 type.
public protocol CompiledCache: AnyObject, KVCache, Updatable {
    var capacity: Int { get }
    func grow(by chunk: Int)
    func resetInPlace()
    /// Roll the cached-token count back to `newLength` — speculative-decoding rollback of
    /// rows a rejected verify forward wrote. Identity-preserving (`_updateInternal`), like
    /// `resetInPlace`, so compiled-step bindings survive.
    func truncate(to newLength: Int)
}

extension CompiledKVCache: CompiledCache {}
extension TurboQuantKVCache: CompiledCache {}

/// Which KV-cache implementation a decode/scoring path runs with — selected from the
/// harness's `RunConfig.kvQuant` tier string.
public enum KVCacheKind: Sendable, Hashable {
    case fp16
    case turboQuant(TurboQuantTier)

    /// Maps a `RunConfig.kvQuant` string. `nil`/`"fp16"` → `.fp16`; TurboQuant tiers accept
    /// both the harness recording slot ("tq2.5"/"tq3.5") and the honest tier name
    /// ("tqB2"/"tqB3"). Unknown strings return nil — callers must fail loudly, never
    /// silently fall back to fp16 (a "measurement" must measure what was asked for).
    public init?(kvQuant: String?) {
        switch kvQuant {
        case nil, "fp16":
            self = .fp16
        case let s?:
            guard
                let tier = TurboQuantTier.allCases.first(where: {
                    $0.harnessSlot == s || $0.rawValue == s
                })
            else { return nil }
            self = .turboQuant(tier)
        }
    }

    /// Build one layer's cache. TurboQuant params resolve lazily from the first update's
    /// head_dim; the fixed seed means every layer derives the identical global Π/S/codebook
    /// (the paper's "global parameters").
    public func makeCache(capacity: Int) -> any CompiledCache {
        switch self {
        case .fp16:
            CompiledKVCache(capacity: capacity)
        case .turboQuant(let tier):
            TurboQuantKVCache(capacity: capacity, tier: tier)
        }
    }
}
