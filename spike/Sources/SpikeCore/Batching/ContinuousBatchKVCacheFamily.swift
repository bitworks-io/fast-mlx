import Foundation
import HarnessCore
import MLX
import MLXLMCommon

struct ContinuousBatchKVGeometry: Equatable {
    let keyValueHeadCount: Int
    let keyHeadDimension: Int
    let valueHeadDimension: Int
    let keyElementBytes: Int
    let valueElementBytes: Int
}

/// Kind-agnostic per-request row: the members every continuous-batch scalar row shares regardless of
/// cache kind (dense KV or recurrent state). `continuousLogicalOffset` is the row's committed token
/// count — for a dense row the compiled offset, for a recurrent row `BaseKVCache.offset`. NOTE: the
/// vendored `MambaCache.advance(_:)` (`ArraysCache.advance`) adjusts only the transient `lengths`/
/// `leftPadding` and never `BaseKVCache.offset`, so a recurrent row's `offset` is DRIVER-maintained: the
/// continuous-batch runtime must set it from the row's committed token count (S3). Deliberately does NOT
/// inherit `CompiledCache`: a recurrent row
/// is a fixed-size recurrent state, not a growing-KV cache, and must not be forced to conform.
protocol ContinuousScalarRowCache: AnyObject, KVCache, Updatable {
    var continuousLogicalOffset: Int { get }
    /// Grow the row's allocation between steps. A provable no-op for a recurrent row (fixed-size
    /// state, never grows with context) — mirrors the batched-side seam's `grow(by:)`, so capacity
    /// growth loops can treat every layer uniformly. Dense rows satisfy it via `CompiledCache`.
    func grow(by chunk: Int)
}

/// The dense-attention refinement: a scalar row that is additionally a `CompiledCache` (fixed-capacity
/// growing KV) and can report its 4-dim KV geometry. This is the unchanged surface the existing
/// fp16/affine continuous-batch path consumes; the super-protocol above is the seam the hybrid recurrent
/// row joins on without pulling in `CompiledCache`.
protocol ContinuousScalarKVCache: ContinuousScalarRowCache, CompiledCache {
    var continuousKVGeometry: ContinuousBatchKVGeometry? { get }
}

extension CompiledKVCache: ContinuousScalarKVCache {
    var continuousLogicalOffset: Int { Int(offsetArr.item(Int32.self)) }

    var continuousKVGeometry: ContinuousBatchKVGeometry? {
        guard let keysBuf, let valuesBuf,
            keysBuf.ndim == 4, valuesBuf.ndim == 4
        else { return nil }
        return ContinuousBatchKVGeometry(
            keyValueHeadCount: keysBuf.dim(1),
            keyHeadDimension: keysBuf.dim(3),
            valueHeadDimension: valuesBuf.dim(3),
            keyElementBytes: keysBuf.itemSize,
            valueElementBytes: valuesBuf.itemSize)
    }
}

extension AffineKVCache: ContinuousScalarKVCache {
    var continuousLogicalOffset: Int { Int(offsetArr.item(Int32.self)) }

    var continuousKVGeometry: ContinuousBatchKVGeometry? {
        guard let state = packedBatchState() else { return nil }
        func scalarBytes(_ dtype: DType) -> Int? {
            switch dtype {
            case .float16, .bfloat16: 2
            case .float32: 4
            default: nil
            }
        }
        guard let keyBytes = scalarBytes(state.keyOutputDType),
            let valueBytes = scalarBytes(state.valueOutputDType)
        else { return nil }
        return ContinuousBatchKVGeometry(
            keyValueHeadCount: state.kPayload.dim(1),
            keyHeadDimension: state.keyDimension,
            valueHeadDimension: state.valueDimension,
            keyElementBytes: keyBytes,
            valueElementBytes: valueBytes)
    }
}

/// Kind-agnostic batched-row seam: the members every merged batched cache shares regardless of cache
/// kind (dense growing-KV or recurrent fixed-state). This is the batched-side mirror of the scalar-side
/// `ContinuousScalarRowCache` split (c69e866): the hybrid runtime holds one
/// `[any ContinuousBatchedRowCache]` per layer and keys dense-only capacity checks off
/// `HybridLayerKindMap.firstDenseLayerIndex` instead of assuming layer 0 is dense. Deliberately does NOT
/// inherit `KVCache`/`Updatable`/`capacity`: a recurrent batched state is a fixed-size recurrent state,
/// not a growing-KV cache, and must not be forced to present a capacity to pad to.
protocol ContinuousBatchedRowCache: AnyObject {
    /// Per-row committed token count, in row order (dense: batched logical offsets; recurrent:
    /// `MambaCache.offset` per row).
    var continuousLogicalOffsets: [Int] { get }
    /// Highest logical token count across rows — the accounting frontier (recurrent state has no physical
    /// write frontier of its own; it mirrors this).
    var continuousPhysicalWrittenEnd: Int { get }
    /// Grow the batched allocation. A provable no-op for recurrent state (fixed-size).
    func grow(by chunk: Int)
    /// Slice one independent scalar row out of the batched state — kind-agnostic return so the seam works
    /// for both dense (`CompiledKVCache`/`AffineKVCache`) and recurrent (`MambaCache`) rows.
    func extractContinuousRow(slot: Int) throws -> any ContinuousScalarRowCache
    /// The object the model must see at THIS layer's index in its `[any KVCache]` array. A dense batched
    /// cache is itself the `KVCache` the model attends through; a recurrent batched state returns its owned
    /// inner `MambaCache` (which the model downcasts to via `cache as? MambaCache`). The S3 hybrid runtime
    /// builds the model's per-layer cache array by mapping `\.modelCache` over the batched rows.
    ///
    /// `Updatable` as well as `KVCache` because the runtime reuses these same objects as
    /// `compile(inputs:outputs:)` state — the compiled decode step must re-read their `innerState()`
    /// per call so recurrent SSM/conv arrays are tracked, not frozen. Both concrete model caches
    /// (dense `ContinuousBatchedKVCache`, recurrent inner `MambaCache`) satisfy it.
    var modelCache: any (KVCache & Updatable) { get }
}

extension ContinuousBatchedKVCache {
    /// A dense batched cache is a growing `KVCache` in its own right, so it IS the model-visible cache.
    var modelCache: any (KVCache & Updatable) { self }
}

/// The dense-attention refinement: a batched cache that is additionally a growing `KVCache`/`Updatable`
/// with a capacity. This is the unchanged surface the existing fp16/affine continuous-batch path
/// consumes; the super-protocol above is the seam the recurrent batched state joins without pulling in
/// `KVCache`/`capacity`.
protocol ContinuousBatchedKVCache: ContinuousBatchedRowCache, KVCache, Updatable {
    var capacity: Int { get }
    func extractContinuous(slot: Int) throws -> any ContinuousScalarKVCache
}

extension ContinuousBatchedKVCache {
    /// Dense caches satisfy the kind-agnostic seam via their existing `extractContinuous`; the returned
    /// `ContinuousScalarKVCache` upcasts to `ContinuousScalarRowCache` for free.
    func extractContinuousRow(slot: Int) throws -> any ContinuousScalarRowCache {
        try extractContinuous(slot: slot)
    }
}

extension BatchedCompiledKVCache: ContinuousBatchedKVCache {
    var continuousLogicalOffsets: [Int] {
        batchOffset.asArray(Int32.self).map(Int.init)
    }

    var continuousPhysicalWrittenEnd: Int {
        continuousLogicalOffsets.max() ?? 0
    }

    func extractContinuous(slot: Int) throws -> any ContinuousScalarKVCache {
        try extract(slot: slot)
    }
}

extension BatchedAffineKVCache: ContinuousBatchedKVCache {
    var continuousLogicalOffsets: [Int] {
        batchOffset.asArray(Int32.self).map(Int.init)
    }

    var continuousPhysicalWrittenEnd: Int { physicalWrittenEnd }

    func extractContinuous(slot: Int) throws -> any ContinuousScalarKVCache {
        try extract(slot: slot)
    }
}

enum ContinuousBatchKVCacheFamilyError: Error, Equatable {
    case unsupportedCacheKind
    case invalidLayerConfiguration
    case incompatibleScalarCache(layer: Int)
}

/// Cache-family factory shared by the existing continuous scheduler/runtime state machine.
/// The fp16 case remains the default. Affine and KVTuner reuse the same actor-confined lifecycle
/// only after explicit compressed-attention admission has selected their scalar cache kind.
enum ContinuousBatchKVCacheFamily {
    case fp16
    case affine(
        configurations: [AffineKVCacheConfiguration],
        attentionMode: AffineKVAttentionMode)
    /// Heterogeneous per-layer caches for a hybrid model (qwen3_5): dense fp16 KV at attention layers,
    /// fixed-size recurrent state at linear layers, keyed by the verified `HybridCacheGeometry`. Selected
    /// from the runtime's `verifiedHybridGeometry` (NOT from a `KVCacheKind` — there is no hybrid kind);
    /// the `init(cacheKind:...)` below is unchanged. Uses the kind-agnostic `makeScalarRows`/`mergeRow`
    /// seam, never the dense-typed `makeScalarCaches`/`merge`.
    case hybridFP16(HybridCacheGeometry)

    init(
        cacheKind: KVCacheKind,
        layerCount: Int,
        affineAttentionMode: AffineKVAttentionMode
    ) throws {
        switch cacheKind {
        case .fp16:
            self = .fp16
        case .affine, .kvtuner:
            let prototypes: [any CompiledCache]
            do {
                prototypes = try cacheKind.makeCaches(
                    layerCount: layerCount,
                    capacity: 1,
                    affineAttentionMode: affineAttentionMode)
            } catch {
                throw ContinuousBatchKVCacheFamilyError.invalidLayerConfiguration
            }
            let affine = prototypes.compactMap { $0 as? AffineKVCache }
            guard affine.count == layerCount else {
                throw ContinuousBatchKVCacheFamilyError.invalidLayerConfiguration
            }
            self = .affine(
                configurations: affine.map(\.configuration),
                attentionMode: affineAttentionMode)
        case .turboQuant, .kvarn, .kvtunerCandidate:
            throw ContinuousBatchKVCacheFamilyError.unsupportedCacheKind
        }
    }

    /// Hybrid family from a verified `HybridCacheGeometry` (heterogeneous by construction — the geometry's
    /// failable init already rejected all-dense maps). Kept separate from `init(cacheKind:...)` because a
    /// hybrid model is not selected by a `KVCacheKind`.
    init(hybridGeometry: HybridCacheGeometry) {
        self = .hybridFP16(hybridGeometry)
    }

    var affineConfigurations: [AffineKVCacheConfiguration]? {
        guard case .affine(let configurations, _) = self else { return nil }
        return configurations
    }

    /// The verified geometry when this is the hybrid family; nil otherwise.
    var hybridGeometry: HybridCacheGeometry? {
        guard case .hybridFP16(let geometry) = self else { return nil }
        return geometry
    }

    func makeScalarCaches(layerCount: Int, capacity: Int)
        -> [any ContinuousScalarKVCache]
    {
        switch self {
        case .fp16:
            return (0 ..< layerCount).map { _ in CompiledKVCache(capacity: capacity) }
        case .affine(let configurations, let attentionMode):
            precondition(
                configurations.count == layerCount,
                "affine continuous-batch layer configuration changed")
            return configurations.map {
                AffineKVCache(
                    capacity: capacity,
                    configuration: $0,
                    attentionMode: attentionMode)
            }
        case .hybridFP16:
            preconditionFailure(
                "hybrid family is heterogeneous; use makeScalarRows/mergeRow, not the dense-typed API")
        }
    }

    /// Kind-agnostic per-layer scalar rows. For fp16/affine this upcasts the dense rows; for hybrid it
    /// returns a dense `CompiledKVCache` at attention layers and a fresh recurrent `MambaCache` at linear
    /// layers, keyed by the geometry. This is the seam the S3 hybrid runtime consumes.
    func makeScalarRows(layerCount: Int, capacity: Int) -> [any ContinuousScalarRowCache] {
        switch self {
        case .fp16, .affine:
            return makeScalarCaches(layerCount: layerCount, capacity: capacity)
                .map { $0 as any ContinuousScalarRowCache }
        case .hybridFP16(let geometry):
            precondition(
                geometry.map.layerCount == layerCount,
                "hybrid continuous-batch layer count disagrees with the verified geometry")
            return (0 ..< layerCount).map { index in
                switch geometry.kind(atLayer: index) {
                case .denseAttention:
                    return CompiledKVCache(capacity: capacity) as any ContinuousScalarRowCache
                case .recurrentState:
                    return MambaCache() as any ContinuousScalarRowCache
                }
            }
        }
    }

    func merge(
        layer: Int,
        rows: [any ContinuousScalarKVCache],
        lengths: [Int]
    ) throws -> any ContinuousBatchedKVCache {
        switch self {
        case .fp16:
            let dense = rows.compactMap { $0 as? CompiledKVCache }
            guard dense.count == rows.count else {
                throw ContinuousBatchKVCacheFamilyError.incompatibleScalarCache(layer: layer)
            }
            return try BatchedCompiledKVCache.merging(dense, lengths: lengths)
        case .affine:
            let affine = rows.compactMap { $0 as? AffineKVCache }
            guard affine.count == rows.count else {
                throw ContinuousBatchKVCacheFamilyError.incompatibleScalarCache(layer: layer)
            }
            return try BatchedAffineKVCache.merging(affine, lengths: lengths)
        case .hybridFP16:
            preconditionFailure(
                "hybrid family is heterogeneous; use mergeRow(layer:rows:lengths:), not the dense-typed merge")
        }
    }

    /// Kind-agnostic per-layer merge for the S3 hybrid runtime. Dispatches by the layer's cache kind: a
    /// dense attention layer merges `CompiledKVCache` rows into a `BatchedCompiledKVCache`; a recurrent
    /// linear layer merges `MambaCache` rows into a `BatchedRecurrentStateCache`. A row of the wrong kind
    /// for the layer fails closed with `incompatibleScalarCache(layer:)` — never a silent cross-kind merge.
    func mergeRow(
        layer: Int,
        rows: [any ContinuousScalarRowCache],
        lengths: [Int]
    ) throws -> any ContinuousBatchedRowCache {
        switch self {
        case .fp16, .affine:
            // Uniform dense model: every layer is a dense KV row. Re-narrow to the dense seam and reuse
            // the byte-stable dense merge so fp16/affine behavior is unchanged.
            let dense = rows.compactMap { $0 as? any ContinuousScalarKVCache }
            guard dense.count == rows.count else {
                throw ContinuousBatchKVCacheFamilyError.incompatibleScalarCache(layer: layer)
            }
            return try merge(layer: layer, rows: dense, lengths: lengths)
        case .hybridFP16(let geometry):
            switch geometry.kind(atLayer: layer) {
            case .denseAttention:
                let dense = rows.compactMap { $0 as? CompiledKVCache }
                guard dense.count == rows.count else {
                    throw ContinuousBatchKVCacheFamilyError.incompatibleScalarCache(layer: layer)
                }
                return try BatchedCompiledKVCache.merging(dense, lengths: lengths)
            case .recurrentState:
                let recurrent = rows.compactMap { $0 as? RecurrentScalarRowCache }
                guard recurrent.count == rows.count else {
                    throw ContinuousBatchKVCacheFamilyError.incompatibleScalarCache(layer: layer)
                }
                return try BatchedRecurrentStateCache.merging(recurrent, lengths: lengths)
            }
        }
    }
}
