import Foundation
import MLX
import MLXLMCommon

public enum KVarNMLXError: Error, Equatable, Sendable {
    case invalidConfiguration
    case invalidTileShape
    case nonFiniteInput
    case invalidRecord
    case nonFiniteOutput
}

/// MLX-side form of the pinned KVarN tile contract. Runtime tiers remain narrower than this
/// math-level configuration; the small power-of-two shapes exist so the official fixture can
/// lock packing and fp16 metadata before a model cache is admitted.
public struct KVarNMLXConfiguration: Equatable, Sendable {
    public let headDimension: Int
    public let groupSize: Int
    public let keyBits: Int
    public let valueBits: Int
    public let iterations: Int

    public init(
        headDimension: Int, groupSize: Int,
        keyBits: Int, valueBits: Int, iterations: Int
    ) throws {
        guard headDimension > 1, Self.isPowerOfTwo(headDimension),
            groupSize > 1, Self.isPowerOfTwo(groupSize),
            headDimension <= 512, groupSize <= 128,
            [2, 4].contains(keyBits), [2, 4].contains(valueBits),
            iterations > 0,
            groupSize.isMultiple(of: 8 / keyBits),
            headDimension.isMultiple(of: 8 / valueBits)
        else { throw KVarNMLXError.invalidConfiguration }
        let (_, elementOverflow) = headDimension.multipliedReportingOverflow(by: groupSize)
        let (_, hadamardOverflow) = headDimension.multipliedReportingOverflow(by: headDimension)
        guard !elementOverflow, !hadamardOverflow else {
            throw KVarNMLXError.invalidConfiguration
        }
        self.headDimension = headDimension
        self.groupSize = groupSize
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.iterations = iterations
    }

    private static func isPowerOfTwo(_ value: Int) -> Bool {
        value > 0 && (value & (value - 1)) == 0
    }
}

/// Native MLX arrays for one complete KVarN tile per batch/head row. The arrays intentionally
/// remain non-Sendable and must stay inside their inference actor.
public struct KVarNMLXRecord {
    public let configuration: KVarNMLXConfiguration
    public let batchSize: Int
    public let headCount: Int
    public let keyDType: DType
    public let valueDType: DType
    public let keyPayload: MLXArray
    public let keyAbsorbedScale: MLXArray
    public let keyAbsorbedBias: MLXArray
    public let keyTokenScale: MLXArray
    public let valuePayload: MLXArray
    public let valueChannelScale: MLXArray
    public let valueAbsorbedScale: MLXArray
    public let valueAbsorbedBias: MLXArray
}

public struct KVarNMLXReconstruction {
    public let keys: MLXArray
    public let values: MLXArray
}

/// Correctness-first MLX port of the pinned KVarN tile transform. This is deliberately a full
/// tile codec, not a per-token affine approximation: K uses channel-by-token orientation, V uses
/// token-by-channel orientation, and both retain the variance-normalization scales specified by
/// the source lock.
public enum KVarNMLXCodec {
    private static let stdMinimum: Float = 1e-3
    private static let stdMaximum: Float = 1e3
    private static let logScaleMinimum: Float = -0.3
    private static let logScaleMaximum: Float = 10
    private static let supportedInputDTypes: Set<DType> = [
        .float16, .bfloat16, .float32,
    ]

    public static func quantize(
        keys: MLXArray, values: MLXArray,
        configuration: KVarNMLXConfiguration
    ) throws -> KVarNMLXRecord {
        let d = configuration.headDimension
        let g = configuration.groupSize
        guard keys.shape.count == 4, values.shape.count == 4,
            keys.dim(0) == values.dim(0), keys.dim(1) == values.dim(1),
            keys.dim(2) == g, values.dim(2) == g,
            keys.dim(3) == d, values.dim(3) == d
        else { throw KVarNMLXError.invalidTileShape }
        guard supportedInputDTypes.contains(keys.dtype),
            supportedInputDTypes.contains(values.dtype)
        else { throw KVarNMLXError.invalidTileShape }
        guard isFinite(keys).all().item(Bool.self),
            isFinite(values).all().item(Bool.self)
        else { throw KVarNMLXError.nonFiniteInput }

        let batch = keys.dim(0)
        let heads = keys.dim(1)
        let rows = batch * heads
        let hadamard = normalizedHadamard(dimension: d)
        let keyRotatedTokenMajor = matmul(
            keys.asType(.float32).reshaped([rows, g, d]), hadamard)
        let valueRotated = matmul(
            values.asType(.float32).reshaped([rows, g, d]), hadamard)
        let keyRotated = keyRotatedTokenMajor.transposed(0, 2, 1)

        let keyBalanced = varianceNormalize(
            keyRotated, rows: d, columns: g,
            iterations: configuration.iterations)
        let valueBalanced = varianceNormalize(
            valueRotated, rows: g, columns: d,
            iterations: configuration.iterations)
        let keyRTN = asymmetricRTN(keyBalanced.values, bits: configuration.keyBits)
        let valueRTN = asymmetricRTN(valueBalanced.values, bits: configuration.valueBits)

        let keyAbsorbedScale = (keyBalanced.rowScale * keyRTN.scale).asType(.float16)
        let keyAbsorbedBias = (keyBalanced.rowScale * keyRTN.bias).asType(.float16)
        let keyTokenScale = keyBalanced.columnScale.asType(.float16)
        let valueChannelScale = valueBalanced.columnScale.asType(.float16)
        let valueAbsorbedScale = (valueBalanced.rowScale * valueRTN.scale).asType(.float16)
        let valueAbsorbedBias = (valueBalanced.rowScale * valueRTN.bias).asType(.float16)

        let finiteOutputs = [
            keyAbsorbedScale, keyAbsorbedBias, keyTokenScale,
            valueChannelScale, valueAbsorbedScale, valueAbsorbedBias,
        ]
        guard finiteOutputs.allSatisfy({ isFinite($0).all().item(Bool.self) }) else {
            throw KVarNMLXError.nonFiniteOutput
        }
        return KVarNMLXRecord(
            configuration: configuration,
            batchSize: batch,
            headCount: heads,
            keyDType: keys.dtype,
            valueDType: values.dtype,
            keyPayload: pack(keyRTN.quantized, bits: configuration.keyBits),
            keyAbsorbedScale: keyAbsorbedScale,
            keyAbsorbedBias: keyAbsorbedBias,
            keyTokenScale: keyTokenScale,
            valuePayload: pack(valueRTN.quantized, bits: configuration.valueBits),
            valueChannelScale: valueChannelScale,
            valueAbsorbedScale: valueAbsorbedScale,
            valueAbsorbedBias: valueAbsorbedBias)
    }

    /// Returns an equivalent record whose arrays are backed by independent, already-materialized
    /// storage. This is primarily useful for measuring decode without retaining the encode graph.
    package static func detachedStorageCopy(
        of record: KVarNMLXRecord
    ) throws -> KVarNMLXRecord {
        guard isValidRecord(record) else { throw KVarNMLXError.invalidRecord }

        func payload(_ array: MLXArray) -> MLXArray {
            MLXArray(array.asArray(UInt8.self)).reshaped(array.shape)
        }
        func metadata(_ array: MLXArray) -> MLXArray {
            MLXArray(array.asArray(Float16.self)).reshaped(array.shape)
        }

        return KVarNMLXRecord(
            configuration: record.configuration,
            batchSize: record.batchSize,
            headCount: record.headCount,
            keyDType: record.keyDType,
            valueDType: record.valueDType,
            keyPayload: payload(record.keyPayload),
            keyAbsorbedScale: metadata(record.keyAbsorbedScale),
            keyAbsorbedBias: metadata(record.keyAbsorbedBias),
            keyTokenScale: metadata(record.keyTokenScale),
            valuePayload: payload(record.valuePayload),
            valueChannelScale: metadata(record.valueChannelScale),
            valueAbsorbedScale: metadata(record.valueAbsorbedScale),
            valueAbsorbedBias: metadata(record.valueAbsorbedBias))
    }

    public static func dequantize(
        _ record: KVarNMLXRecord
    ) throws -> KVarNMLXReconstruction {
        let configuration = record.configuration
        let d = configuration.headDimension
        let g = configuration.groupSize
        guard isValidRecord(record) else { throw KVarNMLXError.invalidRecord }

        let keyQuantized = unpack(
            record.keyPayload, columns: g, bits: configuration.keyBits)
        let valueQuantized = unpack(
            record.valuePayload, columns: d, bits: configuration.valueBits)
        let keyRotated = (
            keyQuantized.asType(.float32)
                * record.keyAbsorbedScale.asType(.float32).expandedDimensions(axis: -1)
                + record.keyAbsorbedBias.asType(.float32).expandedDimensions(axis: -1)
        ) * record.keyTokenScale.asType(.float32).expandedDimensions(axis: 1)
        let valueRotated = (
            valueQuantized.asType(.float32)
                * record.valueAbsorbedScale.asType(.float32).expandedDimensions(axis: -1)
                + record.valueAbsorbedBias.asType(.float32).expandedDimensions(axis: -1)
        ) * record.valueChannelScale.asType(.float32).expandedDimensions(axis: 1)

        let hadamard = normalizedHadamard(dimension: d)
        let keys = matmul(keyRotated.transposed(0, 2, 1), hadamard)
            .reshaped([record.batchSize, record.headCount, g, d])
            .asType(record.keyDType)
        let values = matmul(valueRotated, hadamard)
            .reshaped([record.batchSize, record.headCount, g, d])
            .asType(record.valueDType)
        guard isFinite(keys).all().item(Bool.self),
            isFinite(values).all().item(Bool.self)
        else { throw KVarNMLXError.nonFiniteOutput }
        return KVarNMLXReconstruction(keys: keys, values: values)
    }

    private static func isValidRecord(_ record: KVarNMLXRecord) -> Bool {
        let configuration = record.configuration
        let d = configuration.headDimension
        let g = configuration.groupSize
        guard record.batchSize > 0, record.headCount > 0 else { return false }
        let (rows, rowOverflow) = record.batchSize.multipliedReportingOverflow(
            by: record.headCount)
        guard !rowOverflow else { return false }
        let metadata = [
            record.keyAbsorbedScale, record.keyAbsorbedBias, record.keyTokenScale,
            record.valueChannelScale, record.valueAbsorbedScale, record.valueAbsorbedBias,
        ]
        return record.keyPayload.shape == [rows, d, g * configuration.keyBits / 8]
            && record.keyAbsorbedScale.shape == [rows, d]
            && record.keyAbsorbedBias.shape == [rows, d]
            && record.keyTokenScale.shape == [rows, g]
            && record.valuePayload.shape == [rows, g, d * configuration.valueBits / 8]
            && record.valueChannelScale.shape == [rows, d]
            && record.valueAbsorbedScale.shape == [rows, g]
            && record.valueAbsorbedBias.shape == [rows, g]
            && record.keyPayload.dtype == .uint8
            && record.valuePayload.dtype == .uint8
            && metadata.allSatisfy { $0.dtype == .float16 }
            && supportedInputDTypes.contains(record.keyDType)
            && supportedInputDTypes.contains(record.valueDType)
    }

    private struct BalancedTile {
        let values: MLXArray
        let columnScale: MLXArray
        let rowScale: MLXArray
    }

    private struct RTNRecord {
        let quantized: MLXArray
        let scale: MLXArray
        let bias: MLXArray
    }

    private static func normalizedHadamard(dimension: Int) -> MLXArray {
        var matrix: [Float] = [1]
        var size = 1
        while size < dimension {
            let nextSize = size * 2
            var next = [Float](repeating: 0, count: nextSize * nextSize)
            for row in 0 ..< size {
                for column in 0 ..< size {
                    let value = matrix[row * size + column]
                    next[row * nextSize + column] = value
                    next[row * nextSize + column + size] = value
                    next[(row + size) * nextSize + column] = value
                    next[(row + size) * nextSize + column + size] = -value
                }
            }
            matrix = next
            size = nextSize
        }
        let normalization = Float(1 / sqrt(Double(dimension)))
        return MLXArray(matrix.map { $0 * normalization }).reshaped([dimension, dimension])
    }

    private static func varianceNormalize(
        _ input: MLXArray, rows: Int, columns: Int, iterations: Int
    ) -> BalancedTile {
        let batch = input.dim(0)
        var logColumnScale = MLXArray.zeros([batch, columns], dtype: .float32)
        var logRowScale = MLXArray.zeros([batch, rows], dtype: .float32)
        var current = input
        var bestImbalance = imbalance(current)
        var bestColumnScale = MLXArray.ones([batch, columns], dtype: .float32)
        var bestRowScale = MLXArray.ones([batch, rows], dtype: .float32)

        for _ in 0 ..< iterations {
            let columnStd = std(current, axis: 1, ddof: 1)
            logColumnScale = clip(
                logColumnScale + log(clip(
                    columnStd, min: stdMinimum, max: stdMaximum)),
                min: logScaleMinimum, max: logScaleMaximum)
            current = divideByScales(
                input, columnScale: exp(logColumnScale), rowScale: exp(logRowScale))

            let rowStd = std(current, axis: 2, ddof: 1)
            logRowScale = clip(
                logRowScale + log(clip(
                    rowStd, min: stdMinimum, max: stdMaximum)),
                min: logScaleMinimum, max: logScaleMaximum)
            let columnScale = exp(logColumnScale)
            let rowScale = exp(logRowScale)
            current = divideByScales(
                input, columnScale: columnScale, rowScale: rowScale)

            let nextImbalance = imbalance(current)
            let improved = nextImbalance .<= bestImbalance
            bestImbalance = which(improved, nextImbalance, bestImbalance)
            bestColumnScale = which(
                improved.expandedDimensions(axis: -1), columnScale, bestColumnScale)
            bestRowScale = which(
                improved.expandedDimensions(axis: -1), rowScale, bestRowScale)
        }
        return BalancedTile(
            values: divideByScales(
                input, columnScale: bestColumnScale, rowScale: bestRowScale),
            columnScale: bestColumnScale,
            rowScale: bestRowScale)
    }

    private static func divideByScales(
        _ input: MLXArray, columnScale: MLXArray, rowScale: MLXArray
    ) -> MLXArray {
        input / columnScale.expandedDimensions(axis: 1)
            / rowScale.expandedDimensions(axis: -1)
    }

    private static func imbalance(_ input: MLXArray) -> MLXArray {
        let columnStd = std(input, axis: 1, ddof: 1)
        let rowStd = std(input, axis: 2, ddof: 1)
        let columnRatio = max(columnStd, axis: 1)
            / maximum(min(columnStd, axis: 1), 1e-8)
        let rowRatio = max(rowStd, axis: 1)
            / maximum(min(rowStd, axis: 1), 1e-8)
        return columnRatio + rowRatio
    }

    private static func asymmetricRTN(_ input: MLXArray, bits: Int) -> RTNRecord {
        let qmax = Float((1 << bits) - 1)
        let low = min(input, axis: 2)
        let high = max(input, axis: 2)
        let scale = maximum((high - low) / qmax, 1e-10)
        let quantized = clip(
            round((input - low.expandedDimensions(axis: -1))
                / scale.expandedDimensions(axis: -1)),
            min: 0, max: qmax).asType(.uint32)
        return RTNRecord(quantized: quantized, scale: scale, bias: low)
    }

    private static func pack(_ input: MLXArray, bits: Int) -> MLXArray {
        let valuesPerByte = 8 / bits
        let columns = input.dim(2)
        precondition(columns.isMultiple(of: valuesPerByte))
        let shifts = MLXArray(
            (0 ..< valuesPerByte).map { UInt8($0 * bits) }
        ).reshaped([1, 1, 1, valuesPerByte])
        return leftShift(
            input.asType(.uint8).reshaped([
                input.dim(0), input.dim(1), columns / valuesPerByte, valuesPerByte,
            ]), shifts
        ).sum(axis: -1).asType(.uint8)
    }

    private static func unpack(
        _ input: MLXArray, columns: Int, bits: Int
    ) -> MLXArray {
        let valuesPerByte = 8 / bits
        guard input.dim(2) * valuesPerByte == columns else {
            preconditionFailure("KVarN packed width does not match its configuration")
        }
        let shifts = MLXArray(
            (0 ..< valuesPerByte).map { UInt8($0 * bits) }
        ).reshaped([1, 1, valuesPerByte])
        return bitwiseAnd(
            rightShift(input.expandedDimensions(axis: -1), shifts),
            UInt8((1 << bits) - 1)
        ).reshaped([input.dim(0), input.dim(1), columns])
    }
}

/// The first runtime-admitted KVarN storage cell. The tier name fixes the format geometry;
/// balancing iterations remain an explicit cache/evidence field because eight and sixteen are
/// separate audit cells even though they share the same packed layout.
public enum KVarNKVTier: String, CaseIterable, Sendable, Hashable {
    case k4v2G128 = "kvarn-k4v2-g128"

    public var keyBits: Int { 4 }
    public var valueBits: Int { 2 }
    public var groupSize: Int { 128 }
    public var sinkTokens: Int { 128 }
    public var alignment: Int { 8 }
    public var matrixIterationCount: Int { 8 }
}

/// Actual persistent MLX arrays owned by one KVarN layer cache plus the full K/V pair returned
/// to attention. `materializationWorkspaceBytes` deliberately does not claim to include the
/// codec's transient float32 transform scratch; that is measured and gated separately before a
/// promotion result can be written.
public struct KVarNKVCacheStorageSnapshot: Equatable, Sendable {
    public let tier: KVarNKVTier
    public let iterations: Int
    public let capacityTokens: Int
    public let packedTileSlots: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int

    public var formatPersistentBytes: Int {
        payloadBytes + metadataBytes + alignmentPaddingBytes + fp16SinkBytes + fp16TailBytes
    }

    public var totalPersistentBytes: Int { formatPersistentBytes + controlBytes }

    public var storageAndMaterializationBytes: Int {
        formatPersistentBytes + materializationWorkspaceBytes
    }
}

/// Actor-safe scalar telemetry aggregated from every KVarN layer cache after a run. Persistent
/// terms sum all layers; materialization is one layer's sequential K/V output pair. Transient
/// float32 codec scratch remains a separately measured promotion prerequisite.
public struct KVarNKVCacheTelemetry: Equatable, Sendable {
    public let tier: KVarNKVTier
    public let iterations: Int
    public let executionMode: KVCacheExecutionMode
    public let cachedTokens: Int
    public let layerCount: Int
    public let capacityTokens: Int
    public let packedTileSlots: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int

    public var formatPersistentBytes: Int {
        payloadBytes + metadataBytes + alignmentPaddingBytes + fp16SinkBytes + fp16TailBytes
    }

    public var totalPersistentBytes: Int { formatPersistentBytes + controlBytes }

    public var storageAndMaterializationBytes: Int {
        formatPersistentBytes + materializationWorkspaceBytes
    }

    public static func capture(caches: [KVarNKVCache]) -> KVarNKVCacheTelemetry {
        precondition(!caches.isEmpty, "KVarN telemetry requires at least one layer cache")
        let snapshots = caches.map { cache -> KVarNKVCacheStorageSnapshot in
            guard let snapshot = cache.storageSnapshot() else {
                preconditionFailure("KVarN cache did not allocate before telemetry capture")
            }
            return snapshot
        }
        let first = snapshots[0]
        precondition(
            snapshots.dropFirst().allSatisfy {
                $0.tier == first.tier
                    && $0.iterations == first.iterations
                    && $0.capacityTokens == first.capacityTokens
                    && $0.packedTileSlots == first.packedTileSlots
                    && $0.sequences == first.sequences
                    && $0.kvHeadCount == first.kvHeadCount
                    && $0.headDimension == first.headDimension
                    && $0.metadataScalarBytes == first.metadataScalarBytes
                    && $0.materializationWorkspaceBytes
                        == first.materializationWorkspaceBytes
            },
            "KVarN layer-cache geometry is inconsistent")
        let cachedTokens = Int(caches[0].offsetArr.item(Int32.self))
        precondition(
            caches.dropFirst().allSatisfy {
                Int($0.offsetArr.item(Int32.self)) == cachedTokens
            },
            "KVarN layer-cache offsets are inconsistent")

        func sum(_ values: [Int]) -> Int {
            values.reduce(into: 0) { result, value in
                let (next, overflow) = result.addingReportingOverflow(value)
                precondition(!overflow, "KVarN telemetry byte count overflow")
                result = next
            }
        }
        return KVarNKVCacheTelemetry(
            tier: first.tier, iterations: first.iterations,
            executionMode: .uncompiledCorrectness,
            cachedTokens: cachedTokens, layerCount: caches.count,
            capacityTokens: first.capacityTokens,
            packedTileSlots: first.packedTileSlots,
            sequences: first.sequences, kvHeadCount: first.kvHeadCount,
            headDimension: first.headDimension,
            metadataScalarBytes: first.metadataScalarBytes,
            payloadBytes: sum(snapshots.map(\.payloadBytes)),
            metadataBytes: sum(snapshots.map(\.metadataBytes)),
            alignmentPaddingBytes: sum(snapshots.map(\.alignmentPaddingBytes)),
            fp16SinkBytes: sum(snapshots.map(\.fp16SinkBytes)),
            fp16TailBytes: sum(snapshots.map(\.fp16TailBytes)),
            controlBytes: sum(snapshots.map(\.controlBytes)),
            materializationWorkspaceBytes: first.materializationWorkspaceBytes)
    }
}

/// Correctness-first fixed-capacity KVarN cache.
///
/// The first `G` tokens remain in an explicit fp16 sink. Later tokens accumulate in an explicit
/// fp16 tail and are encoded only when the whole tile is present. Completed records are stored in
/// native low-bit payload arrays plus fp16 metadata, then materialized before the existing
/// attention path. Host-side tile branching makes this version intentionally uncompiled; the
/// decoder integration must preserve that execution-mode boundary.
///
/// This type intentionally does not conform to `Sendable`. Its MLX state must remain confined to
/// the inference actor.
public final class KVarNKVCache: KVCache, Updatable {
    public private(set) var capacity: Int
    public let tier: KVarNKVTier
    public let iterations: Int

    // Packed records are flattened inside each [B, H, slot, bytes] row. The codec fixes K's
    // [D,G] and V's [G,D] orientations before flattening; metadata keeps each axis separate.
    var kPayload: MLXArray?
    var kAbsorbedScales: MLXArray?
    var kAbsorbedBiases: MLXArray?
    var kTokenScales: MLXArray?
    var vPayload: MLXArray?
    var vChannelScales: MLXArray?
    var vAbsorbedScales: MLXArray?
    var vAbsorbedBiases: MLXArray?
    var sinkKeys: MLXArray?
    var sinkValues: MLXArray?
    var tailKeys: MLXArray?
    var tailValues: MLXArray?
    var offsetArr: MLXArray = MLXArray([Int32(0)])

    private var batchSize: Int?
    private var headCount: Int?
    private var headDimension: Int?
    private var keyOutputDType: DType?
    private var valueOutputDType: DType?
    private var keyOutputScalarBytes: Int?
    private var valueOutputScalarBytes: Int?

    public private(set) var offset: Int = 0

    static func supportsExactSinkAndTail(
        keyDType: DType, valueDType: DType
    ) -> Bool {
        keyDType == .float16 && valueDType == .float16
    }

    public var completedTileCount: Int {
        Swift.max(0, offset - tier.sinkTokens) / tier.groupSize
    }

    public init(
        capacity: Int, tier: KVarNKVTier, iterations: Int
    ) {
        precondition(capacity > 0, "capacity must be positive")
        precondition(
            iterations == 8 || iterations == 16,
            "KVarN runtime supports only the declared 8- and 16-iteration cells")
        guard (try? KVarNMLXConfiguration(
            headDimension: 128, groupSize: tier.groupSize,
            keyBits: tier.keyBits, valueBits: tier.valueBits,
            iterations: iterations)) != nil
        else { preconditionFailure("named KVarN tier has invalid codec geometry") }
        self.capacity = capacity
        self.tier = tier
        self.iterations = iterations
    }

    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] {
        [
            kPayload, kAbsorbedScales, kAbsorbedBiases, kTokenScales,
            vPayload, vChannelScales, vAbsorbedScales, vAbsorbedBiases,
            sinkKeys, sinkValues, tailKeys, tailValues,
        ].compactMap { $0 } + [offsetArr]
    }

    public var ropeOffset: RoPEOffset { .batch(offsetArr) }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        validateInput(keys: keys, values: values)
        if kPayload == nil {
            allocate(keys: keys, values: values)
        }

        let tokenCount = keys.dim(2)
        var sourceStart = 0
        var nextOffset = offset

        if nextOffset < tier.sinkTokens {
            let count = Swift.min(tokenCount, tier.sinkTokens - nextOffset)
            let sourceRange = sourceStart ..< (sourceStart + count)
            sinkKeys = scatterRows(
                into: sinkKeys!, start: nextOffset,
                values: keys[0..., 0..., sourceRange, 0...].asType(.float16))
            sinkValues = scatterRows(
                into: sinkValues!, start: nextOffset,
                values: values[0..., 0..., sourceRange, 0...].asType(.float16))
            sourceStart += count
            nextOffset += count
        }

        while sourceStart < tokenCount {
            let postSinkOffset = nextOffset - tier.sinkTokens
            let slot = postSinkOffset / tier.groupSize
            let tailStart = postSinkOffset % tier.groupSize
            let count = Swift.min(
                tokenCount - sourceStart, tier.groupSize - tailStart)
            let sourceRange = sourceStart ..< (sourceStart + count)
            tailKeys = scatterRows(
                into: tailKeys!, start: tailStart,
                values: keys[0..., 0..., sourceRange, 0...].asType(.float16))
            tailValues = scatterRows(
                into: tailValues!, start: tailStart,
                values: values[0..., 0..., sourceRange, 0...].asType(.float16))
            sourceStart += count
            nextOffset += count

            if tailStart + count == tier.groupSize {
                let record: KVarNMLXRecord
                do {
                    record = try KVarNMLXCodec.quantize(
                        keys: tailKeys!, values: tailValues!,
                        configuration: runtimeConfiguration())
                } catch {
                    preconditionFailure("KVarN tile encoding failed: \(error)")
                }
                store(record: record, at: slot)
                tailKeys = MLXArray.zeros(tailKeys!.shape, dtype: .float16)
                tailValues = MLXArray.zeros(tailValues!.shape, dtype: .float16)
            }
        }

        offset = nextOffset
        offsetArr._updateInternal(MLXArray([Int32(nextOffset)]))
        return materialize()
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by KVarNKVCache")
        let keyPositions = MLXArray(Int32(0) ..< Int32(capacity))
            .reshaped([1, 1, 1, capacity])
        let queryPositions = (offsetArr + MLXArray(Int32(0) ..< Int32(n)))
            .reshaped([1, 1, n, 1])
        return .array(keyPositions .<= queryPositions)
    }

    public func grow(by chunk: Int) {
        precondition(chunk > 0, "growth chunk must be positive")
        let oldSlots = packedTileSlots(for: capacity)
        let newCapacity = capacity + chunk
        let newSlots = packedTileSlots(for: newCapacity)
        guard kPayload != nil else {
            capacity = newCapacity
            return
        }
        let addedSlots = newSlots - oldSlots
        if addedSlots > 0 {
            func grown(_ buffer: MLXArray) -> MLXArray {
                let padding = MLXArray.zeros(
                    [buffer.dim(0), buffer.dim(1), addedSlots, buffer.dim(3)],
                    dtype: buffer.dtype)
                return concatenated([buffer, padding], axis: 2)
            }
            kPayload = grown(kPayload!)
            kAbsorbedScales = grown(kAbsorbedScales!)
            kAbsorbedBiases = grown(kAbsorbedBiases!)
            kTokenScales = grown(kTokenScales!)
            vPayload = grown(vPayload!)
            vChannelScales = grown(vChannelScales!)
            vAbsorbedScales = grown(vAbsorbedScales!)
            vAbsorbedBiases = grown(vAbsorbedBiases!)
        }
        capacity = newCapacity
    }

    /// Roll back only state that has not been absorbed into a packed record. The harness rejects
    /// KVarN+spec-decode, so crossing a completed tile is a caller bug rather than a lossy
    /// re-quantization path.
    public func truncate(to newLength: Int) {
        precondition(
            newLength >= 0 && newLength <= offset,
            "truncate target outside the cached prefix")
        let packedBoundary = tier.sinkTokens + completedTileCount * tier.groupSize
        if completedTileCount > 0 {
            precondition(
                newLength >= packedBoundary,
                "KVarN cannot truncate into an already packed tile")
        }
        offsetArr._updateInternal(MLXArray([Int32(newLength)]))
        offset = newLength
    }

    public func resetInPlace() {
        for buffer in [
            kPayload, kAbsorbedScales, kAbsorbedBiases, kTokenScales,
            vPayload, vChannelScales, vAbsorbedScales, vAbsorbedBiases,
            sinkKeys, sinkValues, tailKeys, tailValues,
        ] {
            if let buffer {
                buffer._updateInternal(MLXArray.zeros(buffer.shape, dtype: buffer.dtype))
            }
        }
        offsetArr._updateInternal(MLXArray([Int32(0)]))
        offset = 0
    }

    public func storageSnapshot() -> KVarNKVCacheStorageSnapshot? {
        guard let kPayload, let kAbsorbedScales, let kAbsorbedBiases,
            let kTokenScales, let vPayload, let vChannelScales,
            let vAbsorbedScales, let vAbsorbedBiases,
            let sinkKeys, let sinkValues, let tailKeys, let tailValues,
            let batchSize, let headCount, let headDimension,
            let keyOutputScalarBytes, let valueOutputScalarBytes
        else { return nil }

        let metadataArrays = [
            kAbsorbedScales, kAbsorbedBiases, kTokenScales,
            vChannelScales, vAbsorbedScales, vAbsorbedBiases,
        ]
        precondition(
            metadataArrays.allSatisfy { $0.itemSize == 2 },
            "KVarN metadata must remain fp16")
        let payloadBytes = checkedSum([kPayload.nbytes, vPayload.nbytes])
        let metadataBytes = checkedSum(metadataArrays.map(\.nbytes))
        let sinkBytes = checkedSum([sinkKeys.nbytes, sinkValues.nbytes])
        let tailBytes = checkedSum([tailKeys.nbytes, tailValues.nbytes])
        let workspaceBytes = checkedProduct([
            capacity, batchSize, headCount, headDimension,
            keyOutputScalarBytes + valueOutputScalarBytes,
        ])
        let slots = packedTileSlots(for: capacity)
        let headSequences = checkedProduct([batchSize, headCount])
        let rawUnitBytes = slots == 0
            ? 0
            : (payloadBytes + metadataBytes) / checkedProduct([slots, headSequences])
        let alignedUnitBytes = slots == 0
            ? 0
            : ((rawUnitBytes + tier.alignment - 1) / tier.alignment) * tier.alignment
        let alignmentPaddingBytes = checkedProduct([
            alignedUnitBytes - rawUnitBytes, slots, headSequences,
        ])
        return KVarNKVCacheStorageSnapshot(
            tier: tier, iterations: iterations,
            capacityTokens: capacity, packedTileSlots: slots,
            sequences: batchSize, kvHeadCount: headCount,
            headDimension: headDimension, metadataScalarBytes: 2,
            payloadBytes: payloadBytes, metadataBytes: metadataBytes,
            alignmentPaddingBytes: alignmentPaddingBytes,
            fp16SinkBytes: sinkBytes, fp16TailBytes: tailBytes,
            controlBytes: offsetArr.nbytes,
            materializationWorkspaceBytes: workspaceBytes)
    }

    private func runtimeConfiguration() -> KVarNMLXConfiguration {
        guard let headDimension,
            let configuration = try? KVarNMLXConfiguration(
                headDimension: headDimension, groupSize: tier.groupSize,
                keyBits: tier.keyBits, valueBits: tier.valueBits,
                iterations: iterations)
        else { preconditionFailure("allocated KVarN cache has invalid codec geometry") }
        return configuration
    }

    private func validateInput(keys: MLXArray, values: MLXArray) {
        precondition(keys.shape.count == 4 && values.shape.count == 4, "K/V must be rank 4")
        precondition(
            keys.dim(0) == values.dim(0) && keys.dim(1) == values.dim(1)
                && keys.dim(2) == values.dim(2) && keys.dim(3) == values.dim(3),
            "K/V geometry must match")
        precondition(keys.dim(0) == 1, "the first KVarN frontier is batch-1 only")
        precondition(keys.dim(2) > 0, "K/V update must contain at least one token")
        precondition(offset + keys.dim(2) <= capacity, "K/V update exceeds cache capacity")
        precondition(
            [128, 256, 512].contains(keys.dim(3)),
            "KVarN runtime head dimension is unsupported")
        precondition(
            Self.supportsExactSinkAndTail(
                keyDType: keys.dtype, valueDType: values.dtype),
            "the first KVarN runtime requires fp16 K/V for exact sink and tail storage")
        precondition(
            batchSize == nil || batchSize == keys.dim(0), "K/V batch size changed")
        precondition(headCount == nil || headCount == keys.dim(1), "K/V head count changed")
        precondition(
            headDimension == nil || headDimension == keys.dim(3),
            "K/V head dimension changed")
        precondition(
            isFinite(keys).all().item(Bool.self)
                && isFinite(values).all().item(Bool.self),
            "KVarN input must be finite")
    }

    private func allocate(keys: MLXArray, values: MLXArray) {
        let batch = keys.dim(0)
        let heads = keys.dim(1)
        let dimension = keys.dim(3)
        let slots = packedTileSlots(for: capacity)
        let keyPayloadBytes = dimension * tier.groupSize * tier.keyBits / 8
        let valuePayloadBytes = tier.groupSize * dimension * tier.valueBits / 8

        kPayload = MLXArray.zeros([batch, heads, slots, keyPayloadBytes], dtype: .uint8)
        kAbsorbedScales = MLXArray.zeros([batch, heads, slots, dimension], dtype: .float16)
        kAbsorbedBiases = MLXArray.zeros([batch, heads, slots, dimension], dtype: .float16)
        kTokenScales = MLXArray.zeros(
            [batch, heads, slots, tier.groupSize], dtype: .float16)
        vPayload = MLXArray.zeros([batch, heads, slots, valuePayloadBytes], dtype: .uint8)
        vChannelScales = MLXArray.zeros([batch, heads, slots, dimension], dtype: .float16)
        vAbsorbedScales = MLXArray.zeros(
            [batch, heads, slots, tier.groupSize], dtype: .float16)
        vAbsorbedBiases = MLXArray.zeros(
            [batch, heads, slots, tier.groupSize], dtype: .float16)
        sinkKeys = MLXArray.zeros(
            [batch, heads, tier.sinkTokens, dimension], dtype: .float16)
        sinkValues = MLXArray.zeros(
            [batch, heads, tier.sinkTokens, dimension], dtype: .float16)
        tailKeys = MLXArray.zeros(
            [batch, heads, tier.groupSize, dimension], dtype: .float16)
        tailValues = MLXArray.zeros(
            [batch, heads, tier.groupSize, dimension], dtype: .float16)
        batchSize = batch
        headCount = heads
        headDimension = dimension
        keyOutputDType = keys.dtype
        valueOutputDType = values.dtype
        keyOutputScalarBytes = keys.itemSize
        valueOutputScalarBytes = values.itemSize
    }

    private func scatterRows(
        into buffer: MLXArray, start: Int, values: MLXArray
    ) -> MLXArray {
        let count = values.dim(2)
        let positions = MLXArray(Int32(start) ..< Int32(start + count))
        let indices = positions.reshaped([1, 1, count, 1])
        return putAlong(buffer, indices, values: values, axis: 2)
    }

    private func store(record: KVarNMLXRecord, at slot: Int) {
        let batch = batchSize!
        let heads = headCount!
        let dimension = headDimension!
        let indices = MLXArray([Int32(slot)]).reshaped([1, 1, 1, 1])

        func put(_ buffer: MLXArray, _ values: MLXArray, width: Int) -> MLXArray {
            putAlong(
                buffer, indices,
                values: values.reshaped([batch, heads, 1, width]), axis: 2)
        }
        kPayload = put(kPayload!, record.keyPayload, width: kPayload!.dim(3))
        kAbsorbedScales = put(
            kAbsorbedScales!, record.keyAbsorbedScale, width: dimension)
        kAbsorbedBiases = put(
            kAbsorbedBiases!, record.keyAbsorbedBias, width: dimension)
        kTokenScales = put(
            kTokenScales!, record.keyTokenScale, width: tier.groupSize)
        vPayload = put(vPayload!, record.valuePayload, width: vPayload!.dim(3))
        vChannelScales = put(
            vChannelScales!, record.valueChannelScale, width: dimension)
        vAbsorbedScales = put(
            vAbsorbedScales!, record.valueAbsorbedScale, width: tier.groupSize)
        vAbsorbedBiases = put(
            vAbsorbedBiases!, record.valueAbsorbedBias, width: tier.groupSize)
    }

    private func storedRecord(at slot: Int) -> KVarNMLXRecord {
        let batch = batchSize!
        let heads = headCount!
        let rows = batch * heads
        let dimension = headDimension!
        let keyPackedColumns = tier.groupSize * tier.keyBits / 8
        let valuePackedColumns = dimension * tier.valueBits / 8

        func slotValues(_ buffer: MLXArray, width: Int) -> MLXArray {
            buffer[0..., 0..., slot ..< (slot + 1), 0...]
                .reshaped([rows, width])
        }
        return KVarNMLXRecord(
            configuration: runtimeConfiguration(),
            batchSize: batch, headCount: heads,
            keyDType: keyOutputDType!, valueDType: valueOutputDType!,
            keyPayload: kPayload![0..., 0..., slot ..< (slot + 1), 0...]
                .reshaped([rows, dimension, keyPackedColumns]),
            keyAbsorbedScale: slotValues(kAbsorbedScales!, width: dimension),
            keyAbsorbedBias: slotValues(kAbsorbedBiases!, width: dimension),
            keyTokenScale: slotValues(kTokenScales!, width: tier.groupSize),
            valuePayload: vPayload![0..., 0..., slot ..< (slot + 1), 0...]
                .reshaped([rows, tier.groupSize, valuePackedColumns]),
            valueChannelScale: slotValues(vChannelScales!, width: dimension),
            valueAbsorbedScale: slotValues(vAbsorbedScales!, width: tier.groupSize),
            valueAbsorbedBias: slotValues(vAbsorbedBiases!, width: tier.groupSize))
    }

    private func materialize() -> (MLXArray, MLXArray) {
        var keyPieces = [sinkKeys!.asType(keyOutputDType!)]
        var valuePieces = [sinkValues!.asType(valueOutputDType!)]
        for slot in 0 ..< completedTileCount {
            let reconstruction: KVarNMLXReconstruction
            do {
                reconstruction = try KVarNMLXCodec.dequantize(storedRecord(at: slot))
            } catch {
                preconditionFailure("KVarN tile reconstruction failed: \(error)")
            }
            keyPieces.append(reconstruction.keys)
            valuePieces.append(reconstruction.values)
        }
        if tier.sinkTokens + completedTileCount * tier.groupSize < capacity {
            keyPieces.append(tailKeys!.asType(keyOutputDType!))
            valuePieces.append(tailValues!.asType(valueOutputDType!))
        }

        var keys = concatenated(keyPieces, axis: 2)
        var values = concatenated(valuePieces, axis: 2)
        if keys.dim(2) < capacity {
            let padding = capacity - keys.dim(2)
            keys = concatenated([
                keys,
                MLXArray.zeros(
                    [batchSize!, headCount!, padding, headDimension!],
                    dtype: keyOutputDType!),
            ], axis: 2)
            values = concatenated([
                values,
                MLXArray.zeros(
                    [batchSize!, headCount!, padding, headDimension!],
                    dtype: valueOutputDType!),
            ], axis: 2)
        }
        return (
            keys[0..., 0..., 0 ..< capacity, 0...],
            values[0..., 0..., 0 ..< capacity, 0...])
    }

    private func packedTileSlots(for capacity: Int) -> Int {
        let remaining = Swift.max(0, capacity - tier.sinkTokens)
        return (remaining + tier.groupSize - 1) / tier.groupSize
    }

    private func checkedProduct(_ values: [Int]) -> Int {
        values.reduce(into: 1) { result, value in
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            precondition(!overflow, "KVarN byte count overflow")
            result = next
        }
    }

    private func checkedSum(_ values: [Int]) -> Int {
        values.reduce(into: 0) { result, value in
            let (next, overflow) = result.addingReportingOverflow(value)
            precondition(!overflow, "KVarN byte count overflow")
            result = next
        }
    }

    // MARK: - KVCache protocol surface unused by fast-mlx's decode path

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("KVarNKVCache state restore not supported in the spike") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("KVarNKVCache.copy() not supported in the spike")
    }
}
