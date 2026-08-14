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

protocol ContinuousScalarKVCache: CompiledCache {
    var continuousLogicalOffset: Int { get }
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

protocol ContinuousBatchedKVCache: AnyObject, KVCache, Updatable {
    var capacity: Int { get }
    var continuousLogicalOffsets: [Int] { get }
    var continuousPhysicalWrittenEnd: Int { get }
    func grow(by chunk: Int)
    func extractContinuous(slot: Int) throws -> any ContinuousScalarKVCache
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

    var affineConfigurations: [AffineKVCacheConfiguration]? {
        guard case .affine(let configurations, _) = self else { return nil }
        return configurations
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
        }
    }
}
