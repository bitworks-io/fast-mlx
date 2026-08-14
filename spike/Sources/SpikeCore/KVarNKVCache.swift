import Foundation
import MLX
import MLXLMCommon

public enum KVarNMLXError: Error, Equatable, Sendable {
    case invalidConfiguration
    case invalidTileShape
    case nonFiniteInput
    case invalidRecord
    case nonFiniteOutput
    case directAttentionUnavailable
    case unsupportedInputDType
    case inputDTypeMismatch
    case inputDTypeChanged
}

/// Closed scalar dtype identity used by KVarN ingress and evidence. Keeping MLX's `DType` behind
/// this actor-safe enum makes the authenticated model policy and post-run receipts explicit.
public enum KVarNKVScalarDType:
    String, Codable, Equatable, Hashable, Sendable
{
    case float16
    case bfloat16
    case float32

    var mlxDType: DType {
        switch self {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
    }

    init?(mlxDType: DType) {
        switch mlxDType {
        case .float16: self = .float16
        case .bfloat16: self = .bfloat16
        case .float32: self = .float32
        default: return nil
        }
    }

    var isNative16Bit: Bool { self == .float16 || self == .bfloat16 }
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

/// Actor-confined K-side view of one packed KVarN record. Keeping the exact arrays explicit lets
/// the direct-attention implementation validate and consume every persisted byte class without
/// exposing a reconstructed K tensor.
struct KVarNMLXPackedKeyOperand {
    let configuration: KVarNMLXConfiguration
    let batchSize: Int
    let headCount: Int
    let outputDType: DType
    let payload: MLXArray
    let absorbedScale: MLXArray
    let absorbedBias: MLXArray
    let tokenScale: MLXArray
}

/// Actor-confined V-side view of one packed KVarN record. K and V remain separate because their
/// transforms, metadata axes, and bit widths are intentionally asymmetric.
struct KVarNMLXPackedValueOperand {
    let configuration: KVarNMLXConfiguration
    let batchSize: Int
    let headCount: Int
    let outputDType: DType
    let payload: MLXArray
    let channelScale: MLXArray
    let absorbedScale: MLXArray
    let absorbedBias: MLXArray
}

extension KVarNMLXRecord {
    var keyOperand: KVarNMLXPackedKeyOperand {
        KVarNMLXPackedKeyOperand(
            configuration: configuration,
            batchSize: batchSize,
            headCount: headCount,
            outputDType: keyDType,
            payload: keyPayload,
            absorbedScale: keyAbsorbedScale,
            absorbedBias: keyAbsorbedBias,
            tokenScale: keyTokenScale)
    }

    var valueOperand: KVarNMLXPackedValueOperand {
        KVarNMLXPackedValueOperand(
            configuration: configuration,
            batchSize: batchSize,
            headCount: headCount,
            outputDType: valueDType,
            payload: valuePayload,
            channelScale: valueChannelScale,
            absorbedScale: valueAbsorbedScale,
            absorbedBias: valueAbsorbedBias)
    }
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

    static func directKeyScores(
        queries: MLXArray,
        key: KVarNMLXPackedKeyOperand
    ) throws -> MLXArray {
        guard isValid(key: key) else { throw KVarNMLXError.invalidRecord }
        let d = key.configuration.headDimension
        guard queries.ndim == 4,
            queries.dim(0) == key.batchSize,
            queries.dim(1).isMultiple(of: key.headCount),
            queries.dim(3) == d,
            supportedInputDTypes.contains(queries.dtype)
        else { throw KVarNMLXError.invalidTileShape }
        guard isFinite(queries).all().item(Bool.self) else {
            throw KVarNMLXError.nonFiniteInput
        }
        return directKeyScoresUnchecked(queries: queries, key: key)
    }

    static func directValueProduct(
        weights: MLXArray,
        value: KVarNMLXPackedValueOperand
    ) throws -> MLXArray {
        guard isValid(value: value) else { throw KVarNMLXError.invalidRecord }
        let g = value.configuration.groupSize
        guard weights.ndim == 4,
            weights.dim(0) == value.batchSize,
            weights.dim(1).isMultiple(of: value.headCount),
            weights.dim(3) == g,
            supportedInputDTypes.contains(weights.dtype)
        else { throw KVarNMLXError.invalidTileShape }
        guard isFinite(weights).all().item(Bool.self) else {
            throw KVarNMLXError.nonFiniteInput
        }
        return directValueProductUnchecked(weights: weights, value: value)
    }

    /// Direct packed K algebra used by the cache-owned attention path. Validation is separated
    /// from graph construction because compiled replay supplies tracer arrays whose finiteness
    /// cannot be synchronously read back while the graph is being captured.
    fileprivate static func directKeyScoresUnchecked(
        queries: MLXArray,
        key: KVarNMLXPackedKeyOperand
    ) -> MLXArray {
        let configuration = key.configuration
        let d = configuration.headDimension
        let g = configuration.groupSize
        let batch = key.batchSize
        let kvHeads = key.headCount
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let repeats = queryHeads / kvHeads
        let hadamard = normalizedHadamard(dimension: d)
        var rotatedQueries = matmul(
            queries.asType(.float32), hadamard
        ).asType(queries.dtype)
        var payload = key.payload.view(dtype: .uint32).reshaped([
            batch, kvHeads, d, g * configuration.keyBits / 32,
        ])
        var scales = key.absorbedScale.reshaped([
            batch, kvHeads, d, 1,
        ])
        var biases = key.absorbedBias.reshaped([
            batch, kvHeads, d, 1,
        ])
        var tokenScale = key.tokenScale.reshaped([
            batch, kvHeads, 1, g,
        ])
        if repeats > 1 {
            rotatedQueries = rotatedQueries.reshaped([
                batch, kvHeads, repeats, queryTokens, d,
            ])
            payload = payload.expandedDimensions(axis: 2)
            scales = scales.expandedDimensions(axis: 2)
            biases = biases.expandedDimensions(axis: 2)
            tokenScale = tokenScale.expandedDimensions(axis: 2)
        }
        var scores = quantizedMM(
            rotatedQueries,
            payload,
            scales: scales,
            biases: biases,
            transpose: false,
            groupSize: g,
            bits: configuration.keyBits,
            mode: .affine)
        scores = scores * tokenScale.asType(scores.dtype)
        if repeats > 1 {
            scores = scores.reshaped([
                batch, queryHeads, queryTokens, g,
            ])
        }
        return scores.asType(key.outputDType)
    }

    /// Direct packed V algebra. Qwen's D=128 runtime uses MLX's quantized matmul directly. The
    /// math-level fixture also admits D=256/512 so the layout contract can be tested; those wider
    /// rows use a tile-local unpacked fallback because MLX's native affine kernel is limited to
    /// the admitted 128-element group. Neither branch reconstructs a capacity-wide V cache.
    fileprivate static func directValueProductUnchecked(
        weights: MLXArray,
        value: KVarNMLXPackedValueOperand
    ) -> MLXArray {
        let configuration = value.configuration
        let d = configuration.headDimension
        let g = configuration.groupSize
        let batch = value.batchSize
        let kvHeads = value.headCount
        let queryHeads = weights.dim(1)
        let queryTokens = weights.dim(2)
        let repeats = queryHeads / kvHeads
        var groupedWeights = weights
        var rotated: MLXArray
        if d == 128 {
            var payload = value.payload.view(dtype: .uint32).reshaped([
                batch, kvHeads, g, d * configuration.valueBits / 32,
            ])
            var scales = value.absorbedScale.reshaped([
                batch, kvHeads, g, 1,
            ])
            var biases = value.absorbedBias.reshaped([
                batch, kvHeads, g, 1,
            ])
            if repeats > 1 {
                groupedWeights = groupedWeights.reshaped([
                    batch, kvHeads, repeats, queryTokens, g,
                ])
                payload = payload.expandedDimensions(axis: 2)
                scales = scales.expandedDimensions(axis: 2)
                biases = biases.expandedDimensions(axis: 2)
            }
            rotated = quantizedMM(
                groupedWeights,
                payload,
                scales: scales,
                biases: biases,
                transpose: false,
                groupSize: d,
                bits: configuration.valueBits,
                mode: .affine)
        } else {
            let rows = batch * kvHeads
            let quantizedValues = unpack(
                value.payload, columns: d, bits: configuration.valueBits)
            var denseRotated = (
                quantizedValues.asType(.float32)
                    * value.absorbedScale.asType(.float32)
                        .expandedDimensions(axis: -1)
                    + value.absorbedBias.asType(.float32)
                        .expandedDimensions(axis: -1)
            ).reshaped([batch, kvHeads, g, d])
            if repeats > 1 {
                groupedWeights = groupedWeights.reshaped([
                    batch, kvHeads, repeats, queryTokens, g,
                ])
                denseRotated = denseRotated.expandedDimensions(axis: 2)
            }
            _ = rows // Keep the flattened record geometry explicit in this fallback.
            rotated = matmul(
                groupedWeights.asType(.float32), denseRotated)
        }
        var channelScale = value.channelScale.reshaped([
            batch, kvHeads, 1, d,
        ])
        if repeats > 1 {
            channelScale = channelScale.expandedDimensions(axis: 2)
        }
        rotated = rotated * channelScale.asType(rotated.dtype)
        var output = matmul(
            rotated.asType(.float32), normalizedHadamard(dimension: d)
        ).asType(value.outputDType)
        if repeats > 1 {
            output = output.reshaped([
                batch, queryHeads, queryTokens, d,
            ])
        }
        return output
    }

    private static func isValid(key: KVarNMLXPackedKeyOperand) -> Bool {
        let c = key.configuration
        guard key.batchSize > 0, key.headCount > 0,
            supportedInputDTypes.contains(key.outputDType)
        else { return false }
        let rows = key.batchSize * key.headCount
        return key.payload.shape == [rows, c.headDimension, c.groupSize * c.keyBits / 8]
            && key.payload.dtype == .uint8
            && key.absorbedScale.shape == [rows, c.headDimension]
            && key.absorbedBias.shape == [rows, c.headDimension]
            && key.tokenScale.shape == [rows, c.groupSize]
            && [key.absorbedScale, key.absorbedBias, key.tokenScale]
                .allSatisfy {
                    $0.dtype == .float16
                        && isFinite($0).all().item(Bool.self)
                }
    }

    private static func isValid(value: KVarNMLXPackedValueOperand) -> Bool {
        let c = value.configuration
        guard value.batchSize > 0, value.headCount > 0,
            supportedInputDTypes.contains(value.outputDType)
        else { return false }
        let rows = value.batchSize * value.headCount
        return value.payload.shape == [rows, c.groupSize, c.headDimension * c.valueBits / 8]
            && value.payload.dtype == .uint8
            && value.channelScale.shape == [rows, c.headDimension]
            && value.absorbedScale.shape == [rows, c.groupSize]
            && value.absorbedBias.shape == [rows, c.groupSize]
            && [value.channelScale, value.absorbedScale, value.absorbedBias]
                .allSatisfy {
                    $0.dtype == .float16
                        && isFinite($0).all().item(Bool.self)
                }
    }

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

        let record = quantizeUnchecked(
            keys: keys, values: values, configuration: configuration)
        let finiteOutputs = [
            record.keyAbsorbedScale, record.keyAbsorbedBias,
            record.keyTokenScale, record.valueChannelScale,
            record.valueAbsorbedScale, record.valueAbsorbedBias,
        ]
        guard finiteOutputs.allSatisfy({ isFinite($0).all().item(Bool.self) }) else {
            throw KVarNMLXError.nonFiniteOutput
        }
        return record
    }

    /// Graph-only tile encoder. Callers must validate shape, dtype, and finiteness before state
    /// mutation. Keeping synchronous readback out of this body makes tile finalization traceable.
    fileprivate static func quantizeUnchecked(
        keys: MLXArray, values: MLXArray,
        configuration: KVarNMLXConfiguration
    ) -> KVarNMLXRecord {
        let d = configuration.headDimension
        let g = configuration.groupSize
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

    fileprivate static func normalizedHadamard(dimension: Int) -> MLXArray {
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

/// The first runtime-admitted KVarN storage geometry. Runtime balancing work is represented by
/// `KVarNKVRuntimeCell`: eight and sixteen iterations share these exact persistent arrays but
/// are separate user-visible speed/quality cells.
public enum KVarNKVTier: String, CaseIterable, Sendable, Hashable {
    case k4v2G128 = "kvarn-k4v2-g128"

    public var keyBits: Int { 4 }
    public var valueBits: Int { 2 }
    public var groupSize: Int { 128 }
    public var sinkTokens: Int { 128 }
    public var alignment: Int { 8 }
}

/// Closed runtime cells for the admitted KVarN geometry. The eight-iteration cell keeps the
/// original canonical spelling; the explicit `-i16` suffix prevents evidence for the slower
/// balancing pass from being mislabeled as the faster cell.
public enum KVarNKVRuntimeCell: String, CaseIterable, Sendable, Hashable {
    case k4v2G128I8 = "kvarn-k4v2-g128"
    case k4v2G128I16 = "kvarn-k4v2-g128-i16"

    public var tier: KVarNKVTier { .k4v2G128 }

    public var iterations: Int {
        switch self {
        case .k4v2G128I8: 8
        case .k4v2G128I16: 16
        }
    }
}

/// Explicit KVarN attention selection. The default retains the previously qualified
/// materialize-then-attend behavior; the direct route remains inert until the shared attention
/// protocol and telemetry prove actual packed-state consumption.
public enum KVarNKVAttentionMode:
    String, Codable, Equatable, Hashable, Sendable
{
    case materialize
    case splitQuantizedMM = "split-quantized-mm"
}

/// The KVarN cache read path most recently observed while building an attention graph. Requested
/// mode is deliberately separate: an experimental direct request is not engagement evidence until
/// the shared router actually consumes the packed representation.
public enum KVarNKVAttentionOperation:
    String, Codable, Equatable, Hashable, Sendable
{
    case materializedKV = "materialized-kv"
    case splitQuantizedMM = "split-kvarn-quantized-mm"
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
    public let sourceKeyDType: KVarNKVScalarDType
    public let sourceValueDType: KVarNKVScalarDType
    public let storageKeyDType: KVarNKVScalarDType
    public let storageValueDType: KVarNKVScalarDType
    public let ingressNormalizationApplied: Bool
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int
    public let normalizationWorkspaceBytes: Int
    public let attentionWorkspaceBytes: Int
    public let workspaceBytes: Int
    public let attentionOperation: KVarNKVAttentionOperation

    public var formatPersistentBytes: Int {
        payloadBytes + metadataBytes + alignmentPaddingBytes + fp16SinkBytes + fp16TailBytes
    }

    public var totalPersistentBytes: Int { formatPersistentBytes + controlBytes }

    public var storageAndMaterializationBytes: Int {
        formatPersistentBytes + materializationWorkspaceBytes
    }

    public var storageAndWorkspaceBytes: Int {
        formatPersistentBytes + workspaceBytes
    }
}

enum KVarNKVCacheTelemetryError: Error, Equatable, Sendable {
    case emptySnapshots
    case inconsistentLayerGeometry(layerIndex: Int)
    case inconsistentStorageDType(layerIndex: Int)
    case inconsistentAttentionOperation(layerIndex: Int)
    case invalidIngressDType(layerIndex: Int)
    case inconsistentWorkspace(layerIndex: Int)
    case invalidCompressionState
    case byteCountOverflow
}

/// Actor-safe scalar telemetry aggregated from every KVarN layer cache after a run. Persistent
/// terms sum all layers; source dtype sets preserve legitimate layer-to-layer ingress variation,
/// while storage dtype remains one authenticated model-native contract. Workspace components are
/// copied from the same highest-workspace layer because layers execute sequentially. Transient
/// codec scratch remains a separately measured promotion prerequisite.
public struct KVarNKVCacheTelemetry: Equatable, Sendable {
    public let tier: KVarNKVTier
    public let iterations: Int
    public let executionMode: KVCacheExecutionMode
    public let cachedTokens: Int
    /// Complete post-sink tiles that have actually passed through the KVarN codec. Packed
    /// capacity is preallocated, so allocation bytes alone are not proof of engagement.
    public let completedTileCount: Int
    public let compressedTokens: Int
    public let layerCount: Int
    public let capacityTokens: Int
    public let packedTileSlots: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let sourceKeyDTypes: Set<KVarNKVScalarDType>
    public let sourceValueDTypes: Set<KVarNKVScalarDType>
    public let storageKeyDType: KVarNKVScalarDType
    public let storageValueDType: KVarNKVScalarDType
    public let ingressNormalizationApplied: Bool
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int
    public let normalizationWorkspaceBytes: Int
    public let attentionWorkspaceBytes: Int
    public let workspaceBytes: Int
    public let attentionOperation: KVarNKVAttentionOperation

    public var formatPersistentBytes: Int {
        payloadBytes + metadataBytes + alignmentPaddingBytes + fp16SinkBytes + fp16TailBytes
    }

    public var totalPersistentBytes: Int { formatPersistentBytes + controlBytes }

    public var storageAndMaterializationBytes: Int {
        formatPersistentBytes + materializationWorkspaceBytes
    }

    public var storageAndWorkspaceBytes: Int {
        formatPersistentBytes + workspaceBytes
    }

    public static func capture(caches: [KVarNKVCache]) -> KVarNKVCacheTelemetry {
        precondition(!caches.isEmpty, "KVarN telemetry requires at least one layer cache")
        let snapshots = caches.map { cache -> KVarNKVCacheStorageSnapshot in
            guard let snapshot = cache.storageSnapshot() else {
                preconditionFailure("KVarN cache did not allocate before telemetry capture")
            }
            return snapshot
        }
        let cachedTokens = Int(caches[0].offsetArr.item(Int32.self))
        let completedTileCount = caches[0].completedTileCount
        precondition(
            caches.dropFirst().allSatisfy {
                Int($0.offsetArr.item(Int32.self)) == cachedTokens
                    && $0.completedTileCount == completedTileCount
            },
            "KVarN layer-cache compression state is inconsistent")
        do {
            return try aggregate(
                snapshots: snapshots, cachedTokens: cachedTokens,
                completedTileCount: completedTileCount)
        } catch {
            preconditionFailure("KVarN telemetry is inconsistent: \(error)")
        }
    }

    static func aggregate(
        snapshots: [KVarNKVCacheStorageSnapshot], cachedTokens: Int,
        completedTileCount: Int
    ) throws -> KVarNKVCacheTelemetry {
        guard let first = snapshots.first else {
            throw KVarNKVCacheTelemetryError.emptySnapshots
        }
        guard cachedTokens >= 0, completedTileCount >= 0 else {
            throw KVarNKVCacheTelemetryError.invalidCompressionState
        }

        for (layerIndex, snapshot) in snapshots.enumerated() {
            guard snapshot.tier == first.tier,
                snapshot.iterations == first.iterations,
                snapshot.capacityTokens == first.capacityTokens,
                snapshot.packedTileSlots == first.packedTileSlots,
                snapshot.sequences == first.sequences,
                snapshot.kvHeadCount == first.kvHeadCount,
                snapshot.headDimension == first.headDimension,
                snapshot.metadataScalarBytes == first.metadataScalarBytes
            else {
                throw KVarNKVCacheTelemetryError.inconsistentLayerGeometry(
                    layerIndex: layerIndex)
            }
            guard snapshot.storageKeyDType == first.storageKeyDType,
                snapshot.storageValueDType == first.storageValueDType,
                snapshot.storageKeyDType == snapshot.storageValueDType,
                snapshot.storageKeyDType.isNative16Bit
            else {
                throw KVarNKVCacheTelemetryError.inconsistentStorageDType(
                    layerIndex: layerIndex)
            }
            guard snapshot.attentionOperation == first.attentionOperation else {
                throw KVarNKVCacheTelemetryError.inconsistentAttentionOperation(
                    layerIndex: layerIndex)
            }
            let normalized = snapshot.sourceKeyDType != snapshot.storageKeyDType
                || snapshot.sourceValueDType != snapshot.storageValueDType
            guard snapshot.sourceKeyDType == snapshot.sourceValueDType,
                snapshot.sourceKeyDType == snapshot.storageKeyDType
                    || snapshot.sourceKeyDType == .float32,
                snapshot.ingressNormalizationApplied == normalized,
                normalized
                    ? snapshot.normalizationWorkspaceBytes > 0
                    : snapshot.normalizationWorkspaceBytes == 0
            else {
                throw KVarNKVCacheTelemetryError.invalidIngressDType(
                    layerIndex: layerIndex)
            }
            let workspaceComponents = [
                snapshot.materializationWorkspaceBytes,
                snapshot.normalizationWorkspaceBytes,
                snapshot.attentionWorkspaceBytes,
            ]
            guard workspaceComponents.allSatisfy({ $0 >= 0 }) else {
                throw KVarNKVCacheTelemetryError.inconsistentWorkspace(
                    layerIndex: layerIndex)
            }
            var workspaceTotal = 0
            for component in workspaceComponents {
                let (next, overflow) = workspaceTotal.addingReportingOverflow(
                    component)
                guard !overflow else {
                    throw KVarNKVCacheTelemetryError.byteCountOverflow
                }
                workspaceTotal = next
            }
            guard workspaceTotal == snapshot.workspaceBytes else {
                throw KVarNKVCacheTelemetryError.inconsistentWorkspace(
                    layerIndex: layerIndex)
            }
        }

        let highWater = snapshots.dropFirst().reduce(first) { current, candidate in
            candidate.workspaceBytes > current.workspaceBytes ? candidate : current
        }
        let normalizationObserved = snapshots.contains(
            where: \.ingressNormalizationApplied)
        guard highWater.ingressNormalizationApplied == normalizationObserved else {
            throw KVarNKVCacheTelemetryError.inconsistentWorkspace(
                layerIndex: snapshots.firstIndex(of: highWater) ?? 0)
        }
        let (compressedTokens, compressedTokensOverflow) = completedTileCount
            .multipliedReportingOverflow(by: first.tier.groupSize)
        guard !compressedTokensOverflow else {
            throw KVarNKVCacheTelemetryError.byteCountOverflow
        }

        func sum(_ values: [Int]) throws -> Int {
            try values.reduce(into: 0) { result, value in
                guard value >= 0 else {
                    throw KVarNKVCacheTelemetryError.byteCountOverflow
                }
                let (next, overflow) = result.addingReportingOverflow(value)
                guard !overflow else {
                    throw KVarNKVCacheTelemetryError.byteCountOverflow
                }
                result = next
            }
        }

        return KVarNKVCacheTelemetry(
            tier: first.tier, iterations: first.iterations,
            // A bare cache has no authority to claim its caller compiled the model step. The
            // compiled decoder upgrades this receipt after it resolves and executes its closure;
            // scoring paths that call the model directly remain honestly uncompiled.
            executionMode: .uncompiledCorrectness,
            cachedTokens: cachedTokens,
            completedTileCount: completedTileCount,
            compressedTokens: compressedTokens,
            layerCount: snapshots.count,
            capacityTokens: first.capacityTokens,
            packedTileSlots: first.packedTileSlots,
            sequences: first.sequences, kvHeadCount: first.kvHeadCount,
            headDimension: first.headDimension,
            sourceKeyDTypes: Set(snapshots.map(\.sourceKeyDType)),
            sourceValueDTypes: Set(snapshots.map(\.sourceValueDType)),
            storageKeyDType: first.storageKeyDType,
            storageValueDType: first.storageValueDType,
            ingressNormalizationApplied:
                highWater.ingressNormalizationApplied,
            metadataScalarBytes: first.metadataScalarBytes,
            payloadBytes: try sum(snapshots.map(\.payloadBytes)),
            metadataBytes: try sum(snapshots.map(\.metadataBytes)),
            alignmentPaddingBytes: try sum(
                snapshots.map(\.alignmentPaddingBytes)),
            fp16SinkBytes: try sum(snapshots.map(\.fp16SinkBytes)),
            fp16TailBytes: try sum(snapshots.map(\.fp16TailBytes)),
            controlBytes: try sum(snapshots.map(\.controlBytes)),
            materializationWorkspaceBytes:
                highWater.materializationWorkspaceBytes,
            normalizationWorkspaceBytes:
                highWater.normalizationWorkspaceBytes,
            attentionWorkspaceBytes: highWater.attentionWorkspaceBytes,
            workspaceBytes: highWater.workspaceBytes,
            attentionOperation: first.attentionOperation)
    }
}

/// Correctness-first fixed-capacity KVarN cache.
///
/// The first `G` tokens remain in an explicit native 16-bit sink (fp16 or bfloat16). Later tokens
/// accumulate in an explicit native 16-bit tail and are encoded only when the whole tile is
/// present. Completed records are stored in native low-bit payload arrays plus fp16 metadata.
/// The default compatibility path materializes the cache and remains uncompiled; the explicit
/// direct path consumes packed records with graph-state tile transitions and can be compiled.
///
/// This type intentionally does not conform to `Sendable`. Its MLX state must remain confined to
/// the inference actor.
public final class KVarNKVCache: AttentionKVCacheProtocol, Updatable {
    public private(set) var capacity: Int
    public let tier: KVarNKVTier
    public let iterations: Int
    public let attentionMode: KVarNKVAttentionMode
    /// Authenticated model-native dtype required when runtime projections arrive in float32.
    /// `nil` preserves the legacy exact-native contract for direct cache/unit construction: only
    /// already-fp16/bf16 K/V are accepted and no dtype is guessed.
    public let storageDType: KVarNKVScalarDType?

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
    private var sourceKeyDType: KVarNKVScalarDType?
    private var sourceValueDType: KVarNKVScalarDType?
    private var materializationWorkspaceBytes: Int?
    private var normalizationWorkspaceBytes: Int?
    private var attentionWorkspaceBytes: Int?
    private var attentionOperation: KVarNKVAttentionOperation?

    /// The MLX scalar is authoritative because compiled replay updates it in graph. Reading the
    /// logical offset synchronizes that state instead of exposing a host mirror captured once
    /// while the graph was traced.
    public var offset: Int { Int(offsetArr.item(Int32.self)) }

    static func supportsExactSinkAndTail(
        keyDType: DType, valueDType: DType
    ) -> Bool {
        (keyDType == .float16 || keyDType == .bfloat16)
            && (valueDType == .float16 || valueDType == .bfloat16)
    }

    static func matchesEstablishedSinkAndTailDTypes(
        keyDType: DType, valueDType: DType,
        establishedKeyDType: DType?, establishedValueDType: DType?
    ) -> Bool {
        (establishedKeyDType == nil || keyDType == establishedKeyDType)
            && (establishedValueDType == nil || valueDType == establishedValueDType)
    }

    public var completedTileCount: Int {
        Swift.max(
            0,
            Int(offsetArr.item(Int32.self)) - tier.sinkTokens
        ) / tier.groupSize
    }

    public init(
        capacity: Int, tier: KVarNKVTier, iterations: Int,
        attentionMode: KVarNKVAttentionMode = .materialize,
        storageDType: KVarNKVScalarDType? = nil
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
        precondition(
            storageDType == nil || storageDType!.isNative16Bit,
            "KVarN persistent sink/tail storage must remain fp16 or bfloat16")
        self.capacity = capacity
        self.tier = tier
        self.iterations = iterations
        self.attentionMode = attentionMode
        self.storageDType = storageDType
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
        let prepared = requirePreparedStorageInputs(
            keys: keys, values: values)
        validateInput(keys: prepared.keys, values: prepared.values)
        recordIngress(prepared)
        if kPayload == nil {
            allocate(keys: prepared.keys, values: prepared.values)
        }

        storeHost(keys: prepared.keys, values: prepared.values)
        let materialized = materialize()
        materializationWorkspaceBytes = checkedSum([
            materialized.0.nbytes, materialized.1.nbytes,
        ])
        attentionWorkspaceBytes = 0
        attentionOperation = .materializedKV
        return materialized
    }

    public func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch attentionMode {
        case .materialize:
            let (cachedKeys, cachedValues) = update(keys: keys, values: values)
            let attentionKeys = cachedKeys.dtype == queries.dtype
                ? cachedKeys : cachedKeys.asType(queries.dtype)
            let attentionValues = cachedValues.dtype == queries.dtype
                ? cachedValues : cachedValues.asType(queries.dtype)
            materializationWorkspaceBytes = checkedSum([
                cachedKeys.nbytes, cachedValues.nbytes,
                attentionKeys === cachedKeys ? 0 : attentionKeys.nbytes,
                attentionValues === cachedValues ? 0 : attentionValues.nbytes,
            ])
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: attentionKeys,
                values: attentionValues,
                scale: scale,
                mask: mask)
        case .splitQuantizedMM:
            let prepared = requirePreparedStorageInputs(
                keys: keys, values: values)
            precondition(
                directGeometryIsValid(
                    queries: queries,
                    keys: prepared.keys,
                    values: prepared.values,
                    scale: scale, mask: mask),
                "invalid direct KVarN attention geometry")
            let inputsFinite = isFinite(queries).all()
                & isFinite(prepared.keys).all()
                & isFinite(prepared.values).all()
            let hasCapacity = offsetArr + MLXArray(Int32(keys.dim(2)))
                .<= MLXArray(Int32(capacity))
            let inputsValid = inputsFinite & hasCapacity
            // Fresh allocation, every multi-token prefill, and the first call after an in-place
            // reset are deliberately synchronized. The reset case prevents a rejected raw
            // single-token call from repopulating only the host workspace receipt while leaving
            // the in-graph offset at zero. Normal compiled single-token replay reaches this path
            // with a receipt established by its valid prefill and keeps the predicate in graph.
            let requiresSynchronousValidation = kPayload == nil
                || keys.dim(2) > 1
                || (attentionWorkspaceBytes ?? 0) == 0
            if requiresSynchronousValidation,
                !inputsValid.item(Bool.self)
            {
                return invalidDirectAttentionOutput(like: queries)
            }
            recordIngress(prepared)
            if kPayload == nil {
                allocate(keys: prepared.keys, values: prepared.values)
            }
            if keys.dim(2) == 1 {
                storeSingleGraph(
                    keys: prepared.keys, values: prepared.values,
                    when: inputsValid)
            } else {
                // Long prefill is deliberately uncompiled and chunked by the decoder. The host
                // loop finalizes each complete tile before the direct attention graph consumes it.
                validateInput(keys: prepared.keys, values: prepared.values)
                storeHost(keys: prepared.keys, values: prepared.values)
            }
            materializationWorkspaceBytes = 0
            attentionOperation = .splitQuantizedMM
            let output = packedAttention(
                queries: queries, scale: scale, mask: mask)
            return MLX.where(
                inputsValid,
                output,
                invalidDirectAttentionOutput(like: output))
        }
    }

    /// Throwing preflight used by tests and evidence producers that must prove a malformed direct
    /// request fails before allocation or mutation. Production model calls use the nonthrowing
    /// protocol method after the same structural contract has been fixed by model geometry.
    func checkedUpdateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> MLXArray {
        guard attentionMode == .splitQuantizedMM else {
            throw KVarNMLXError.directAttentionUnavailable
        }
        let prepared = try preparedStorageInputs(keys: keys, values: values)
        guard directGeometryIsValid(
            queries: queries,
            keys: prepared.keys,
            values: prepared.values,
            scale: scale, mask: mask)
        else { throw KVarNMLXError.invalidTileShape }
        guard isFinite(queries).all().item(Bool.self),
            isFinite(prepared.keys).all().item(Bool.self),
            isFinite(prepared.values).all().item(Bool.self)
        else { throw KVarNMLXError.nonFiniteInput }
        guard offset + keys.dim(2) <= capacity else {
            throw KVarNMLXError.invalidTileShape
        }
        return updateAndAttend(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask)
    }

    private struct PreparedStorageInputs {
        let keys: MLXArray
        let values: MLXArray
        let sourceKeyDType: KVarNKVScalarDType
        let sourceValueDType: KVarNKVScalarDType
        let storageKeyDType: KVarNKVScalarDType
        let storageValueDType: KVarNKVScalarDType
        let normalizationWorkspaceBytes: Int

        var normalizationApplied: Bool {
            sourceKeyDType != storageKeyDType
                || sourceValueDType != storageValueDType
        }
    }

    private func requirePreparedStorageInputs(
        keys: MLXArray, values: MLXArray
    ) -> PreparedStorageInputs {
        do {
            return try preparedStorageInputs(keys: keys, values: values)
        } catch {
            preconditionFailure("invalid KVarN K/V ingress dtype: \(error)")
        }
    }

    /// Resolve the exact storage arrays without mutating cache or telemetry state. Float32 K/V
    /// are admitted only when an authenticated model-native 16-bit target was supplied at cache
    /// construction; already-native legacy inputs retain their exact dtype.
    private func preparedStorageInputs(
        keys: MLXArray, values: MLXArray
    ) throws -> PreparedStorageInputs {
        guard let sourceKey = KVarNKVScalarDType(mlxDType: keys.dtype),
            let sourceValue = KVarNKVScalarDType(mlxDType: values.dtype)
        else { throw KVarNMLXError.unsupportedInputDType }
        if let established = sourceKeyDType, established != sourceKey {
            throw KVarNMLXError.inputDTypeChanged
        }
        if let established = sourceValueDType, established != sourceValue {
            throw KVarNMLXError.inputDTypeChanged
        }

        let storageKey: KVarNKVScalarDType
        let storageValue: KVarNKVScalarDType
        if let storageDType {
            guard sourceKey == sourceValue else {
                throw KVarNMLXError.inputDTypeMismatch
            }
            guard sourceKey == storageDType || sourceKey == .float32 else {
                throw KVarNMLXError.inputDTypeMismatch
            }
            storageKey = storageDType
            storageValue = storageDType
        } else {
            guard sourceKey.isNative16Bit, sourceValue.isNative16Bit else {
                throw KVarNMLXError.unsupportedInputDType
            }
            storageKey = sourceKey
            storageValue = sourceValue
        }

        let preparedKeys = sourceKey == storageKey
            ? keys : keys.asType(storageKey.mlxDType)
        let preparedValues = sourceValue == storageValue
            ? values : values.asType(storageValue.mlxDType)
        let workspace = checkedSum([
            preparedKeys === keys ? 0 : preparedKeys.nbytes,
            preparedValues === values ? 0 : preparedValues.nbytes,
        ])
        return PreparedStorageInputs(
            keys: preparedKeys,
            values: preparedValues,
            sourceKeyDType: sourceKey,
            sourceValueDType: sourceValue,
            storageKeyDType: storageKey,
            storageValueDType: storageValue,
            normalizationWorkspaceBytes: workspace)
    }

    private func recordIngress(_ prepared: PreparedStorageInputs) {
        sourceKeyDType = prepared.sourceKeyDType
        sourceValueDType = prepared.sourceValueDType
        normalizationWorkspaceBytes = Swift.max(
            normalizationWorkspaceBytes ?? 0,
            prepared.normalizationWorkspaceBytes)
    }

    private func storeHost(keys: MLXArray, values: MLXArray) {

        let tokenCount = keys.dim(2)
        var sourceStart = 0
        var nextOffset = offset

        if nextOffset < tier.sinkTokens {
            let count = Swift.min(tokenCount, tier.sinkTokens - nextOffset)
            let sourceRange = sourceStart ..< (sourceStart + count)
            sinkKeys = scatterRows(
                into: sinkKeys!, start: nextOffset,
                values: keys[0..., 0..., sourceRange, 0...])
            sinkValues = scatterRows(
                into: sinkValues!, start: nextOffset,
                values: values[0..., 0..., sourceRange, 0...])
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
                values: keys[0..., 0..., sourceRange, 0...])
            tailValues = scatterRows(
                into: tailValues!, start: tailStart,
                values: values[0..., 0..., sourceRange, 0...])
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
                tailKeys = MLXArray.zeros(
                    tailKeys!.shape, dtype: keyOutputDType!)
                tailValues = MLXArray.zeros(
                    tailValues!.shape, dtype: valueOutputDType!)
            }
        }

        offsetArr._updateInternal(MLXArray([Int32(nextOffset)]))
    }

    /// One fixed-shape decode update expressed entirely with array predicates. Every state array
    /// keeps its identity; only its internal graph value changes. The same captured closure can
    /// therefore move from sink row 127 to tail row 0 and across later tile boundaries.
    private func storeSingleGraph(
        keys: MLXArray,
        values: MLXArray,
        when inputIsValid: MLXArray
    ) {
        let sinkLimit = MLXArray(Int32(tier.sinkTokens))
        let group = MLXArray(Int32(tier.groupSize))
        let position = offsetArr
        let isSink = position .< sinkLimit
        let writeSink = inputIsValid & isSink

        let sinkPosition = minimum(
            position, MLXArray(Int32(tier.sinkTokens - 1)))
        let sinkIndices = sinkPosition.reshaped([1, 1, 1, 1])
        let nextSinkKeys = putAlong(
            sinkKeys!, sinkIndices, values: keys, axis: 2)
        let nextSinkValues = putAlong(
            sinkValues!, sinkIndices, values: values, axis: 2)
        sinkKeys!._updateInternal(MLX.where(
            writeSink, nextSinkKeys, sinkKeys!))
        sinkValues!._updateInternal(MLX.where(
            writeSink, nextSinkValues, sinkValues!))

        let postSink = maximum(position - sinkLimit, MLXArray(Int32(0)))
        let tailPosition = remainder(postSink, group)
        let tailIndices = tailPosition.reshaped([1, 1, 1, 1])
        let nextTailKeys = putAlong(
            tailKeys!, tailIndices, values: keys, axis: 2)
        let nextTailValues = putAlong(
            tailValues!, tailIndices, values: values, axis: 2)
        let isPostSink = inputIsValid & logicalNot(isSink)
        let updatedTailKeys = MLX.where(isPostSink, nextTailKeys, tailKeys!)
        let updatedTailValues = MLX.where(isPostSink, nextTailValues, tailValues!)
        let completesTile = isPostSink
            & (tailPosition .== MLXArray(Int32(tier.groupSize - 1)))

        let slotCount = packedTileSlots(for: capacity)
        if slotCount > 0 {
            let record = KVarNMLXCodec.quantizeUnchecked(
                keys: updatedTailKeys,
                values: updatedTailValues,
                configuration: runtimeConfiguration())
            // `putAlong` still constructs its indexed graph when `condition` is false. Clamp the
            // rejected full-cache position so overflow remains a non-mutating in-graph failure.
            let slot = minimum(
                floorDivide(postSink, group),
                MLXArray(Int32(slotCount - 1)))
            storeGraph(record: record, at: slot, when: completesTile)
        }

        tailKeys!._updateInternal(MLX.where(
            completesTile,
            MLXArray.zeros(tailKeys!.shape, dtype: tailKeys!.dtype),
            updatedTailKeys))
        tailValues!._updateInternal(MLX.where(
            completesTile,
            MLXArray.zeros(tailValues!.shape, dtype: tailValues!.dtype),
            updatedTailValues))
        offsetArr._updateInternal(MLX.where(
            inputIsValid,
            offsetArr + MLXArray([Int32(1)]),
            offsetArr))
    }

    private func invalidDirectAttentionOutput(
        like output: MLXArray
    ) -> MLXArray {
        MLXArray.full(
            output.shape,
            values: MLXArray(Float.nan).asType(output.dtype))
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
        materializationWorkspaceBytes = 0
        normalizationWorkspaceBytes = 0
        attentionWorkspaceBytes = 0
    }

    public func storageSnapshot() -> KVarNKVCacheStorageSnapshot? {
        guard let kPayload, let kAbsorbedScales, let kAbsorbedBiases,
            let kTokenScales, let vPayload, let vChannelScales,
            let vAbsorbedScales, let vAbsorbedBiases,
            let sinkKeys, let sinkValues, let tailKeys, let tailValues,
            let batchSize, let headCount, let headDimension,
            let sourceKeyDType, let sourceValueDType,
            let storageKeyDType = KVarNKVScalarDType(
                mlxDType: sinkKeys.dtype),
            let storageValueDType = KVarNKVScalarDType(
                mlxDType: sinkValues.dtype),
            let materializationWorkspaceBytes,
            let normalizationWorkspaceBytes,
            let attentionWorkspaceBytes,
            let attentionOperation
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
        if Int(offsetArr.item(Int32.self)) == 0 {
            precondition(
                materializationWorkspaceBytes == 0
                    && normalizationWorkspaceBytes == 0
                    && attentionWorkspaceBytes == 0,
                "reset KVarN cache must clear workspace high-water marks")
        } else {
            switch attentionOperation {
            case .materializedKV:
                precondition(
                    materializationWorkspaceBytes > 0,
                    "materialized KVarN attention must report output workspace")
                precondition(
                    attentionWorkspaceBytes == 0,
                    "materialized KVarN attention cannot report direct-attention workspace")
            case .splitQuantizedMM:
                precondition(
                    materializationWorkspaceBytes == 0,
                    "direct KVarN attention cannot report materialization workspace")
                precondition(
                    attentionWorkspaceBytes > 0,
                    "direct KVarN attention must report score/weight workspace")
            }
        }
        let workspaceBytes = checkedSum([
            materializationWorkspaceBytes,
            normalizationWorkspaceBytes,
            attentionWorkspaceBytes,
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
            headDimension: headDimension,
            sourceKeyDType: sourceKeyDType,
            sourceValueDType: sourceValueDType,
            storageKeyDType: storageKeyDType,
            storageValueDType: storageValueDType,
            ingressNormalizationApplied:
                sourceKeyDType != storageKeyDType
                    || sourceValueDType != storageValueDType,
            metadataScalarBytes: 2,
            payloadBytes: payloadBytes, metadataBytes: metadataBytes,
            alignmentPaddingBytes: alignmentPaddingBytes,
            fp16SinkBytes: sinkBytes, fp16TailBytes: tailBytes,
            controlBytes: offsetArr.nbytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            normalizationWorkspaceBytes: normalizationWorkspaceBytes,
            attentionWorkspaceBytes: attentionWorkspaceBytes,
            workspaceBytes: workspaceBytes,
            attentionOperation: attentionOperation)
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
            "the first KVarN runtime requires fp16 or bfloat16 K/V for exact sink and tail storage")
        precondition(
            Self.matchesEstablishedSinkAndTailDTypes(
                keyDType: keys.dtype, valueDType: values.dtype,
                establishedKeyDType: keyOutputDType,
                establishedValueDType: valueOutputDType),
            "K/V dtypes changed after KVarN sink and tail allocation")
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
            [batch, heads, tier.sinkTokens, dimension], dtype: keys.dtype)
        sinkValues = MLXArray.zeros(
            [batch, heads, tier.sinkTokens, dimension], dtype: values.dtype)
        tailKeys = MLXArray.zeros(
            [batch, heads, tier.groupSize, dimension], dtype: keys.dtype)
        tailValues = MLXArray.zeros(
            [batch, heads, tier.groupSize, dimension], dtype: values.dtype)
        batchSize = batch
        headCount = heads
        headDimension = dimension
        keyOutputDType = keys.dtype
        valueOutputDType = values.dtype
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
        kPayload!._updateInternal(
            put(kPayload!, record.keyPayload, width: kPayload!.dim(3)))
        kAbsorbedScales!._updateInternal(put(
            kAbsorbedScales!, record.keyAbsorbedScale, width: dimension))
        kAbsorbedBiases!._updateInternal(put(
            kAbsorbedBiases!, record.keyAbsorbedBias, width: dimension))
        kTokenScales!._updateInternal(put(
            kTokenScales!, record.keyTokenScale, width: tier.groupSize))
        vPayload!._updateInternal(
            put(vPayload!, record.valuePayload, width: vPayload!.dim(3)))
        vChannelScales!._updateInternal(put(
            vChannelScales!, record.valueChannelScale, width: dimension))
        vAbsorbedScales!._updateInternal(put(
            vAbsorbedScales!, record.valueAbsorbedScale, width: tier.groupSize))
        vAbsorbedBiases!._updateInternal(put(
            vAbsorbedBiases!, record.valueAbsorbedBias, width: tier.groupSize))
    }

    private func storeGraph(
        record: KVarNMLXRecord,
        at slot: MLXArray,
        when condition: MLXArray
    ) {
        let batch = batchSize!
        let heads = headCount!
        let dimension = headDimension!
        let indices = slot.reshaped([1, 1, 1, 1])

        func next(
            _ buffer: MLXArray, _ values: MLXArray, width: Int
        ) -> MLXArray {
            let stored = putAlong(
                buffer,
                indices,
                values: values.reshaped([batch, heads, 1, width]),
                axis: 2)
            return MLX.where(condition, stored, buffer)
        }
        kPayload!._updateInternal(next(
            kPayload!, record.keyPayload, width: kPayload!.dim(3)))
        kAbsorbedScales!._updateInternal(next(
            kAbsorbedScales!, record.keyAbsorbedScale, width: dimension))
        kAbsorbedBiases!._updateInternal(next(
            kAbsorbedBiases!, record.keyAbsorbedBias, width: dimension))
        kTokenScales!._updateInternal(next(
            kTokenScales!, record.keyTokenScale, width: tier.groupSize))
        vPayload!._updateInternal(next(
            vPayload!, record.valuePayload, width: vPayload!.dim(3)))
        vChannelScales!._updateInternal(next(
            vChannelScales!, record.valueChannelScale, width: dimension))
        vAbsorbedScales!._updateInternal(next(
            vAbsorbedScales!, record.valueAbsorbedScale, width: tier.groupSize))
        vAbsorbedBiases!._updateInternal(next(
            vAbsorbedBiases!, record.valueAbsorbedBias, width: tier.groupSize))
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

    private func directGeometryIsValid(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        guard scale.isFinite, scale > 0,
            capacity > tier.sinkTokens,
            queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
            queries.dim(0) == 1, keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(0) == values.dim(0),
            keys.dim(1) == values.dim(1), keys.dim(1) > 0,
            queries.dim(1).isMultiple(of: keys.dim(1)),
            queries.dim(2) == keys.dim(2), keys.dim(2) == values.dim(2),
            keys.dim(2) > 0,
            queries.dim(3) == keys.dim(3), keys.dim(3) == values.dim(3),
            [128, 256, 512].contains(keys.dim(3)),
            [DType.float16, .bfloat16].contains(keys.dtype),
            [DType.float16, .bfloat16].contains(values.dtype),
            [DType.float16, .bfloat16, .float32].contains(queries.dtype),
            keys.dim(2) <= capacity,
            Self.matchesEstablishedSinkAndTailDTypes(
                keyDType: keys.dtype,
                valueDType: values.dtype,
                establishedKeyDType: keyOutputDType,
                establishedValueDType: valueOutputDType),
            batchSize == nil || batchSize == keys.dim(0),
            headCount == nil || headCount == keys.dim(1),
            headDimension == nil || headDimension == keys.dim(3)
        else { return false }

        func validMaskArray(_ array: MLXArray) -> Bool {
            guard [2, 4].contains(array.ndim),
                array.dim(-1) == capacity
            else { return false }
            let queryRows = array.dim(-2)
            guard queryRows == 1 || queryRows == queries.dim(2) else {
                return false
            }
            guard array.ndim == 4 else { return true }
            let batches = array.dim(-4)
            let heads = array.dim(-3)
            return (batches == 1 || batches == queries.dim(0))
                && (heads == 1 || heads == queries.dim(1))
        }
        switch mask {
        case .none, .causal:
            return true
        case .array(let array):
            return validMaskArray(array)
        case .arrays(let arrays):
            return arrays.count <= 1
                && (arrays.first.map(validMaskArray) ?? true)
        }
    }

    /// Attend over a fixed logical source layout: sink followed by one segment per packed slot.
    /// For each post-sink slot, an in-graph liveness predicate selects either its completed packed
    /// record, the one live fp16 tail, or negative infinity. This prevents both double counting
    /// and inactive zero slots from stealing softmax mass.
    private func packedAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard headDimension == 128 else {
            return packedAttentionBySlot(
                queries: queries, scale: scale, mask: mask)
        }

        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let dimension = queries.dim(3)
        let kvHeads = headCount!
        let repeats = queryHeads / kvHeads
        let slots = packedTileSlots(for: capacity)
        let groupSize = tier.groupSize
        let scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        let sinkScores = groupedDenseScores(
            queries: scaledQueries,
            keys: sinkKeys!)
        let tailScores = groupedDenseScores(
            queries: scaledQueries,
            keys: tailKeys!)

        let rotatedQueries = matmul(
            scaledQueries.asType(.float32),
            KVarNMLXCodec.normalizedHadamard(dimension: dimension)
        ).asType(scaledQueries.dtype).reshaped([
            batch, kvHeads, repeats, 1, queryTokens, dimension,
        ])
        let keyPayload = kPayload!.view(dtype: .uint32).reshaped([
            batch, kvHeads, slots, dimension,
            groupSize * tier.keyBits / 32,
        ]).expandedDimensions(axis: 2)
        let keyScales = kAbsorbedScales!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: -1)
        let keyBiases = kAbsorbedBiases!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: -1)
        let keyTokenScales = kTokenScales!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: 4)
        var packedScores = quantizedMM(
            rotatedQueries,
            keyPayload,
            scales: keyScales,
            biases: keyBiases,
            transpose: false,
            groupSize: groupSize,
            bits: tier.keyBits,
            mode: .affine)
        packedScores = packedScores
            * keyTokenScales.asType(packedScores.dtype)

        let postSink = maximum(
            offsetArr - MLXArray(Int32(tier.sinkTokens)),
            MLXArray(Int32(0)))
        let completed = floorDivide(
            postSink, MLXArray(Int32(tier.groupSize)))
        let liveTailCount = remainder(
            postSink, MLXArray(Int32(tier.groupSize)))
        let negativeInfinity = MLXArray(-Float.infinity).asType(queries.dtype)
        let slotIndices = MLXArray(Int32(0) ..< Int32(slots))
            .reshaped([1, 1, 1, slots, 1, 1])
        let isPacked = slotIndices .< completed
        let isLiveTail = (slotIndices .== completed)
            & (liveTailCount .> MLXArray(Int32(0)))
        let groupedTailScores = tailScores.reshaped([
            batch, kvHeads, repeats, queryTokens, groupSize,
        ]).expandedDimensions(axis: 3)
        let selectedScores = MLX.where(
            isPacked,
            packedScores,
            MLX.where(isLiveTail, groupedTailScores, negativeInfinity))
        let postSinkScores = selectedScores
            .transposed(0, 1, 2, 4, 3, 5)
            .reshaped([
                batch, queryHeads, queryTokens, slots * groupSize,
            ])
        var scores = concatenated([sinkScores, postSinkScores], axis: -1)
        scores = scores[0..., 0..., 0..., 0 ..< capacity]
        scores = apply(mask: mask, to: scores)
        let weights = softmax(scores, axis: -1, precise: true)
        let workspace = checkedSum([
            scores.nbytes,
            weights.nbytes,
            denseReadConversionWorkspaceBytes(to: queries.dtype),
        ])
        attentionWorkspaceBytes = Swift.max(
            attentionWorkspaceBytes ?? 0,
            workspace)

        let sinkWidth = Swift.min(tier.sinkTokens, capacity)
        var output = groupedDenseValueProduct(
            weights: paddedWeights(
                weights, start: 0, count: sinkWidth,
                targetCount: tier.sinkTokens),
            values: sinkValues!)
        let postSinkCount = capacity - sinkWidth
        let groupedWeights = paddedWeights(
            weights,
            start: sinkWidth,
            count: postSinkCount,
            targetCount: slots * groupSize
        ).reshaped([
            batch, kvHeads, repeats, queryTokens, slots, groupSize,
        ]).transposed(0, 1, 2, 4, 3, 5)
        let valuePayload = vPayload!.view(dtype: .uint32).reshaped([
            batch, kvHeads, slots, groupSize,
            dimension * tier.valueBits / 32,
        ]).expandedDimensions(axis: 2)
        let valueScales = vAbsorbedScales!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: -1)
        let valueBiases = vAbsorbedBiases!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: -1)
        var packedOutput = quantizedMM(
            groupedWeights,
            valuePayload,
            scales: valueScales,
            biases: valueBiases,
            transpose: false,
            groupSize: dimension,
            bits: tier.valueBits,
            mode: .affine)
        let valueChannelScales = vChannelScales!
            .expandedDimensions(axis: 2)
            .expandedDimensions(axis: 4)
        packedOutput = packedOutput
            * valueChannelScales.asType(packedOutput.dtype)
        packedOutput = matmul(
            packedOutput.asType(.float32),
            KVarNMLXCodec.normalizedHadamard(dimension: dimension)
        ).asType(groupedWeights.dtype)

        let tailOutput = matmul(
            groupedWeights,
            tailValues!
                .asType(groupedWeights.dtype)
                .expandedDimensions(axis: 2)
                .expandedDimensions(axis: 3))
        let selectedOutput = MLX.where(
            isPacked,
            packedOutput,
            MLX.where(
                isLiveTail,
                tailOutput,
                MLXArray.zeros(packedOutput.shape, dtype: packedOutput.dtype)))
        output = output + selectedOutput.sum(axis: 3).reshaped([
            batch, queryHeads, queryTokens, dimension,
        ])
        return output
    }

    /// Compatibility route for the math-level D=256/512 fixtures. The loaded model path is
    /// D=128 and uses the slot-vectorized graph above so compiled graph width does not grow with
    /// cache capacity.
    private func packedAttentionBySlot(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        let sinkScores = groupedDenseScores(
            queries: scaledQueries,
            keys: sinkKeys!)
        let tailScores = groupedDenseScores(
            queries: scaledQueries,
            keys: tailKeys!)
        let postSink = maximum(
            offsetArr - MLXArray(Int32(tier.sinkTokens)),
            MLXArray(Int32(0)))
        let completed = floorDivide(
            postSink, MLXArray(Int32(tier.groupSize)))
        let liveTailCount = remainder(
            postSink, MLXArray(Int32(tier.groupSize)))
        let negativeInfinity = MLXArray(-Float.infinity).asType(queries.dtype)

        var scorePieces = [sinkScores]
        let slots = packedTileSlots(for: capacity)
        for slot in 0 ..< slots {
            let record = storedRecord(at: slot)
            let packedScores = KVarNMLXCodec.directKeyScoresUnchecked(
                queries: scaledQueries,
                key: record.keyOperand)
            let slotValue = MLXArray(Int32(slot))
            let isPacked = slotValue .< completed
            let isLiveTail = (slotValue .== completed)
                & (liveTailCount .> MLXArray(Int32(0)))
            scorePieces.append(MLX.where(
                isPacked,
                packedScores,
                MLX.where(isLiveTail, tailScores, negativeInfinity)))
        }
        var scores = concatenated(scorePieces, axis: -1)
        scores = scores[0..., 0..., 0..., 0 ..< capacity]
        scores = apply(mask: mask, to: scores)
        let weights = softmax(scores, axis: -1, precise: true)
        let workspace = checkedSum([
            scores.nbytes,
            weights.nbytes,
            denseReadConversionWorkspaceBytes(to: queries.dtype),
        ])
        attentionWorkspaceBytes = Swift.max(
            attentionWorkspaceBytes ?? 0,
            workspace)

        let sinkWidth = Swift.min(tier.sinkTokens, capacity)
        var output = groupedDenseValueProduct(
            weights: paddedWeights(
                weights, start: 0, count: sinkWidth,
                targetCount: tier.sinkTokens),
            values: sinkValues!)
        for slot in 0 ..< slots {
            let start = tier.sinkTokens + slot * tier.groupSize
            guard start < capacity else { break }
            let count = Swift.min(tier.groupSize, capacity - start)
            let segmentWeights = paddedWeights(
                weights, start: start, count: count,
                targetCount: tier.groupSize)
            let record = storedRecord(at: slot)
            let packedOutput = KVarNMLXCodec.directValueProductUnchecked(
                weights: segmentWeights,
                value: record.valueOperand).asType(output.dtype)
            let tailOutput = groupedDenseValueProduct(
                weights: segmentWeights,
                values: tailValues!)
            let slotValue = MLXArray(Int32(slot))
            let isPacked = slotValue .< completed
            let isLiveTail = (slotValue .== completed)
                & (liveTailCount .> MLXArray(Int32(0)))
            output = output + MLX.where(
                isPacked,
                packedOutput,
                MLX.where(
                    isLiveTail,
                    tailOutput,
                    MLXArray.zeros(output.shape, dtype: output.dtype)))
        }
        return output
    }

    private func groupedDenseScores(
        queries: MLXArray,
        keys: MLXArray
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let dimension = queries.dim(3)
        let typedKeys = keys.dtype == queries.dtype
            ? keys : keys.asType(queries.dtype)
        let kvHeads = typedKeys.dim(1)
        let repeats = queryHeads / kvHeads
        guard repeats > 1 else {
            return matmul(
                queries, typedKeys.transposed(0, 1, 3, 2))
        }
        return matmul(
            queries.reshaped([
                batch, kvHeads, repeats, queryTokens, dimension,
            ]),
            typedKeys.expandedDimensions(axis: 2)
                .transposed(0, 1, 2, 4, 3)
        ).reshaped([
            batch, queryHeads, queryTokens, typedKeys.dim(2),
        ])
    }

    private func groupedDenseValueProduct(
        weights: MLXArray,
        values: MLXArray
    ) -> MLXArray {
        let batch = weights.dim(0)
        let queryHeads = weights.dim(1)
        let queryTokens = weights.dim(2)
        let typedValues = values.dtype == weights.dtype
            ? values : values.asType(weights.dtype)
        let kvHeads = typedValues.dim(1)
        let dimension = typedValues.dim(3)
        let repeats = queryHeads / kvHeads
        guard repeats > 1 else { return matmul(weights, typedValues) }
        return matmul(
            weights.reshaped([
                batch, kvHeads, repeats, queryTokens, weights.dim(3),
            ]),
            typedValues.expandedDimensions(axis: 2)
        ).reshaped([batch, queryHeads, queryTokens, dimension])
    }

    private func denseReadConversionWorkspaceBytes(to dtype: DType) -> Int {
        [sinkKeys, tailKeys, sinkValues, tailValues].compactMap { $0 }
            .reduce(into: 0) { total, array in
                guard array.dtype != dtype else { return }
                let bytes = checkedProduct([array.size, dtype.size])
                let (next, overflow) = total.addingReportingOverflow(bytes)
                precondition(
                    !overflow,
                    "KVarN dense-read conversion workspace overflow")
                total = next
            }
    }

    private func paddedWeights(
        _ weights: MLXArray,
        start: Int,
        count: Int,
        targetCount: Int
    ) -> MLXArray {
        let selected = weights[0..., 0..., 0..., start ..< (start + count)]
        guard count < targetCount else { return selected }
        return concatenated([
            selected,
            MLXArray.zeros([
                weights.dim(0), weights.dim(1), weights.dim(2),
                targetCount - count,
            ], dtype: weights.dtype),
        ], axis: -1)
    }

    private func apply(
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        to scores: MLXArray
    ) -> MLXArray {
        let masked: MLXArray
        switch mask {
        case .none:
            masked = scores
        case .causal:
            let queryTokens = scores.dim(-2)
            let queryPositions = MLXArray(0 ..< queryTokens)
                + offsetArr - MLXArray([Int32(queryTokens)])
            let keyPositions = MLXArray(0 ..< scores.dim(-1))
            masked = MLX.where(
                queryPositions.expandedDimensions(axis: -1)
                    .>= keyPositions.expandedDimensions(axis: -2),
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        case .array(let array):
            masked = apply(maskArray: array, to: scores)
        case .arrays(let arrays):
            precondition(arrays.count <= 1, "only one attention mask array is supported")
            masked = arrays.first.map { apply(maskArray: $0, to: scores) }
                ?? scores
        }
        let written = MLXArray(0 ..< scores.dim(-1)) .< offsetArr
        return MLX.where(
            written,
            masked,
            MLXArray(-Float.infinity).asType(scores.dtype))
    }

    private func apply(maskArray: MLXArray, to scores: MLXArray) -> MLXArray {
        if maskArray.dtype == .bool {
            return MLX.where(
                maskArray,
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        }
        return scores + maskArray.asType(scores.dtype)
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
