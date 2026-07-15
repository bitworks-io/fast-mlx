import Foundation
import MLX
import MLXLMCommon

/// The cache contract `CompiledMLXDecoder` actually depends on, beyond MLXLMCommon's
/// `KVCache`: compile-capturable state (`Updatable`), chunked growth between steps (one
/// retrace per chunk), and identity-preserving in-place reset. `CompiledKVCache` (fp16)
/// `AffineKVCache`, and `TurboQuantKVCache` all conform, so the decoder is selected per KV
/// tier instead of hardcoding the fp16 type.
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
extension AffineKVCache: CompiledCache {}
extension KVarNKVCache: CompiledCache {}

public enum KVCacheExecutionMode: String, Equatable, Sendable {
    case compiled
    case uncompiledCorrectness = "uncompiled-correctness"
}

/// Which KV-cache implementation a decode/scoring path runs with — selected from the
/// harness's `RunConfig.kvQuant` tier string.
public enum KVCacheKind: Sendable, Hashable {
    case fp16
    case affine(AffineKVTier)
    case turboQuant(TurboQuantTier)
    case kvarn(KVarNKVTier)

    /// Maps a `RunConfig.kvQuant` string. `nil`/`"fp16"` → `.fp16`; affine cells accept
    /// only `AffineKVTier`'s canonical raw values, while TurboQuant accepts both the harness
    /// recording slot ("tq2.5"/"tq3.5") and honest tier name ("tqB2"/"tqB3"). Unknown
    /// strings return nil — callers must fail loudly, never silently fall back to fp16.
    public init?(kvQuant: String?) {
        switch kvQuant {
        case nil, "fp16":
            self = .fp16
        case let s?:
            if let tier = AffineKVTier(rawValue: s) {
                self = .affine(tier)
                return
            }
            if let tier = KVarNKVTier(rawValue: s) {
                self = .kvarn(tier)
                return
            }
            guard
                let tier = TurboQuantTier.allCases.first(where: {
                    $0.harnessSlot == s || $0.rawValue == s
                })
            else { return nil }
            self = .turboQuant(tier)
        }
    }

    /// Build one layer's cache. Native affine geometry is fixed by its named tier. TurboQuant
    /// params resolve lazily from the first update's head_dim; the fixed seed means every layer
    /// derives the identical global Π/S/codebook (the paper's "global parameters").
    public func makeCache(capacity: Int) -> any CompiledCache {
        switch self {
        case .fp16:
            CompiledKVCache(capacity: capacity)
        case .affine(let tier):
            AffineKVCache(capacity: capacity, configuration: tier.configuration)
        case .turboQuant(let tier):
            TurboQuantKVCache(capacity: capacity, tier: tier)
        case .kvarn(let tier):
            KVarNKVCache(
                capacity: capacity, tier: tier,
                iterations: tier.matrixIterationCount)
        }
    }

    /// Resolve the actual step mode once, before the decoder builds its closure. KVarN's host
    /// tile-boundary mutation is not compile-capturable, so a caller cannot accidentally turn it
    /// into a stale compiled trace by requesting compilation.
    public func executionMode(requestingCompilation: Bool) -> KVCacheExecutionMode {
        guard requestingCompilation else { return .uncompiledCorrectness }
        switch self {
        case .kvarn:
            return .uncompiledCorrectness
        case .fp16, .affine, .turboQuant:
            return .compiled
        }
    }

    /// Speculation is qualified independently from a cache's standalone decode path. Until a
    /// lossy cache proves byte-identical temp-0 A/B behavior (including rollback), the engine
    /// rejects the combination at its own API boundary instead of relying on CLI validation.
    public var supportsSpecDecode: Bool {
        switch self {
        case .fp16:
            return true
        case .affine, .turboQuant, .kvarn:
            return false
        }
    }
}
