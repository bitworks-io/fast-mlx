import Foundation
import HarnessCore
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

public enum KVCacheKindError: Error, Equatable, Sendable {
    case layerCountMismatch(expected: Int, actual: Int)
    case invalidKVTunerConfiguration(layer: Int)
}

/// Which KV-cache implementation a decode/scoring path runs with — selected from the
/// harness's `RunConfig.kvQuant` tier string.
public enum KVCacheKind: Sendable, Hashable {
    case fp16
    case affine(AffineKVTier)
    case kvtuner(KVTunerRuntimeSelection)
    /// Preselection-only heterogeneous policy. There is deliberately no `kvQuant` parser route:
    /// only the authenticated KVTuner qualification workflow may construct and execute it.
    case kvtunerCandidate(KVTunerCandidateRuntimePolicy)
    case turboQuant(TurboQuantTier)
    case kvarn(KVarNKVRuntimeCell)

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
            if let cell = KVarNKVRuntimeCell(rawValue: s) {
                self = .kvarn(cell)
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
    public func makeCache(
        capacity: Int,
        affineAttentionMode: AffineKVAttentionMode = .materialize,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) -> any CompiledCache {
        switch self {
        case .fp16:
            CompiledKVCache(capacity: capacity)
        case .affine(let tier):
            AffineKVCache(
                capacity: capacity,
                configuration: tier.configuration,
                attentionMode: affineAttentionMode)
        case .kvtuner, .kvtunerCandidate:
            preconditionFailure(
                "KVTuner requires the layer-aware makeCaches factory")
        case .turboQuant(let tier):
            TurboQuantKVCache(capacity: capacity, tier: tier)
        case .kvarn(let cell):
            KVarNKVCache(
                capacity: capacity, tier: cell.tier,
                iterations: cell.iterations,
                attentionMode: kvarnAttentionMode,
                storageDType: kvarnStorageDType)
        }
    }

    /// Build the complete cache list for one model. Uniform formats repeat one cache kind;
    /// KVTuner consumes its immutable authenticated layer policy in canonical index order.
    /// A count or configuration mismatch is an explicit error and never becomes fp16.
    public func makeCaches(
        layerCount: Int,
        capacity: Int,
        affineAttentionMode: AffineKVAttentionMode = .materialize,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) throws -> [any CompiledCache] {
        switch self {
        case .kvtuner(let selection):
            return try Self.makeKVTunerCaches(
                layers: selection.layers,
                groupSize: selection.groupSize,
                layerCount: layerCount,
                capacity: capacity,
                attentionMode: affineAttentionMode)
        case .kvtunerCandidate(let policy):
            return try Self.makeKVTunerCaches(
                layers: policy.layers,
                groupSize: policy.groupSize,
                layerCount: layerCount,
                capacity: capacity,
                attentionMode: affineAttentionMode)
        case .fp16, .affine, .turboQuant, .kvarn:
            return (0 ..< layerCount).map { _ in
                makeCache(
                    capacity: capacity,
                    affineAttentionMode: affineAttentionMode,
                    kvarnAttentionMode: kvarnAttentionMode,
                    kvarnStorageDType: kvarnStorageDType)
            }
        }
    }

    /// Resolve the actual step mode once, before the decoder builds its closure. KVarN's
    /// materialized path has host tile-boundary mutation that is not compile-capturable; the
    /// direct attention path keeps execution in graph state and may compile when requested.
    public func executionMode(
        requestingCompilation: Bool,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize
    ) -> KVCacheExecutionMode {
        guard requestingCompilation else { return .uncompiledCorrectness }
        switch self {
        case .kvarn:
            return kvarnAttentionMode == .splitQuantizedMM
                ? .compiled
                : .uncompiledCorrectness
        case .fp16, .affine, .kvtuner, .kvtunerCandidate, .turboQuant:
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
        case .affine, .kvtuner, .kvtunerCandidate, .turboQuant, .kvarn:
            return false
        }
    }

    private static func makeKVTunerCaches(
        layers: [KVTunerRuntimeLayerPolicy],
        groupSize: Int,
        layerCount: Int,
        capacity: Int,
        attentionMode: AffineKVAttentionMode
    ) throws -> [any CompiledCache] {
        guard layers.count == layerCount else {
            throw KVCacheKindError.layerCountMismatch(
                expected: layers.count, actual: layerCount)
        }
        return try layers.enumerated().map { position, policy in
            guard policy.layer == position else {
                throw KVCacheKindError.invalidKVTunerConfiguration(
                    layer: position)
            }
            let configuration: AffineKVCacheConfiguration
            do {
                configuration = try AffineKVCacheConfiguration(
                    keyBits: policy.keyBits,
                    valueBits: policy.valueBits,
                    keyGroupSize: groupSize,
                    valueGroupSize: groupSize)
            } catch {
                throw KVCacheKindError.invalidKVTunerConfiguration(
                    layer: position)
            }
            return AffineKVCache(
                capacity: capacity,
                configuration: configuration,
                attentionMode: attentionMode)
        }
    }
}
