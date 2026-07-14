import Foundation

public enum KVarNReferenceError: Error, Equatable, Sendable {
    case invalidConfig
    case invalidTileShape
    case nonFiniteInput
    case invalidRecord
    case nonFiniteOutput
}

public struct KVarNReferenceConfig: Equatable, Sendable {
    public let headDimension: Int
    public let groupSize: Int
    public let keyBits: Int
    public let valueBits: Int
    public let iterations: Int

    public init(
        headDimension: Int, groupSize: Int, keyBits: Int, valueBits: Int,
        iterations: Int
    ) {
        self.headDimension = headDimension
        self.groupSize = groupSize
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.iterations = iterations
    }
}

/// Packed KVarN record for one complete token tile and one KV head. Metadata names mirror the
/// pinned official implementation; every metadata value is fp16 before it becomes cache state.
public struct KVarNReferenceRecord: Equatable, Sendable {
    public let config: KVarNReferenceConfig
    public let keyPacked: [UInt8]
    public let keyAbsorbedScale: [Float16]
    public let keyAbsorbedBias: [Float16]
    public let keyTokenScale: [Float16]
    public let valuePacked: [UInt8]
    public let valueChannelScale: [Float16]
    public let valueAbsorbedScale: [Float16]
    public let valueAbsorbedBias: [Float16]
}

public struct KVarNReferenceReconstruction: Equatable, Sendable {
    public let keysTokenMajor: [Float]
    public let valuesTokenMajor: [Float]
}

/// Correctness-first scalar implementation of the pinned KVarN PyTorch reference. This is not a
/// runtime kernel: it is the fixture oracle and design contract consumed by MLX-side tests.
public enum KVarNReference {
    private static let stdMinimum: Float = 1e-3
    private static let stdMaximum: Float = 1e3
    private static let logScaleMinimum: Float = -0.3
    private static let logScaleMaximum: Float = 10

    public static func quantize(
        keysTokenMajor: [Float], valuesTokenMajor: [Float],
        config: KVarNReferenceConfig
    ) throws -> KVarNReferenceRecord {
        let elementCount = try validatedElementCount(config)
        guard keysTokenMajor.count == elementCount,
            valuesTokenMajor.count == elementCount
        else { throw KVarNReferenceError.invalidTileShape }
        guard keysTokenMajor.allSatisfy(\.isFinite),
            valuesTokenMajor.allSatisfy(\.isFinite)
        else { throw KVarNReferenceError.nonFiniteInput }

        let d = config.headDimension
        let g = config.groupSize
        let hadamard = normalizedHadamard(dimension: d)
        let keyRotatedTokenMajor = matmulRight(
            keysTokenMajor, rows: g, columns: d, right: hadamard)
        let valueRotated = matmulRight(
            valuesTokenMajor, rows: g, columns: d, right: hadamard)
        let keyRotated = transpose(keyRotatedTokenMajor, rows: g, columns: d)

        let keyBalanced = varianceNormalize(
            keyRotated, rows: d, columns: g, iterations: config.iterations)
        let valueBalanced = varianceNormalize(
            valueRotated, rows: g, columns: d, iterations: config.iterations)
        guard keyRotated.allSatisfy(\.isFinite), valueRotated.allSatisfy(\.isFinite),
            keyBalanced.values.allSatisfy(\.isFinite),
            keyBalanced.columnScale.allSatisfy(\.isFinite),
            keyBalanced.rowScale.allSatisfy(\.isFinite),
            valueBalanced.values.allSatisfy(\.isFinite),
            valueBalanced.columnScale.allSatisfy(\.isFinite),
            valueBalanced.rowScale.allSatisfy(\.isFinite)
        else { throw KVarNReferenceError.nonFiniteOutput }
        let keyRTN = try asymmetricRTN(
            keyBalanced.values, rows: d, columns: g, bits: config.keyBits)
        let valueRTN = try asymmetricRTN(
            valueBalanced.values, rows: g, columns: d, bits: config.valueBits)

        let keyAbsorbedScale = try finiteFP16(zip(
            keyBalanced.rowScale, keyRTN.scale).map { $0.0 * $0.1 })
        let keyAbsorbedBias = try finiteFP16(zip(
            keyBalanced.rowScale, keyRTN.bias).map { $0.0 * $0.1 })
        let keyTokenScale = try finiteFP16(keyBalanced.columnScale)
        let valueChannelScale = try finiteFP16(valueBalanced.columnScale)
        let valueAbsorbedScale = try finiteFP16(zip(
            valueBalanced.rowScale, valueRTN.scale).map { $0.0 * $0.1 })
        let valueAbsorbedBias = try finiteFP16(zip(
            valueBalanced.rowScale, valueRTN.bias).map { $0.0 * $0.1 })

        return KVarNReferenceRecord(
            config: config,
            keyPacked: pack(keyRTN.quantized, rows: d, columns: g, bits: config.keyBits),
            keyAbsorbedScale: keyAbsorbedScale,
            keyAbsorbedBias: keyAbsorbedBias,
            keyTokenScale: keyTokenScale,
            valuePacked: pack(
                valueRTN.quantized, rows: g, columns: d, bits: config.valueBits),
            valueChannelScale: valueChannelScale,
            valueAbsorbedScale: valueAbsorbedScale,
            valueAbsorbedBias: valueAbsorbedBias)
    }

    public static func dequantize(
        _ record: KVarNReferenceRecord
    ) throws -> KVarNReferenceReconstruction {
        let config = record.config
        _ = try validatedElementCount(config)
        let d = config.headDimension
        let g = config.groupSize
        guard record.keyAbsorbedScale.count == d,
            record.keyAbsorbedBias.count == d,
            record.keyTokenScale.count == g,
            record.valueChannelScale.count == d,
            record.valueAbsorbedScale.count == g,
            record.valueAbsorbedBias.count == g
        else { throw KVarNReferenceError.invalidRecord }

        let keyQuantized = try unpack(
            record.keyPacked, rows: d, columns: g, bits: config.keyBits)
        let valueQuantized = try unpack(
            record.valuePacked, rows: g, columns: d, bits: config.valueBits)
        var keyRotated = [Float](repeating: 0, count: d * g)
        for row in 0 ..< d {
            let scale = Float(record.keyAbsorbedScale[row])
            let bias = Float(record.keyAbsorbedBias[row])
            for column in 0 ..< g {
                keyRotated[row * g + column] = (
                    Float(keyQuantized[row * g + column]) * scale + bias
                ) * Float(record.keyTokenScale[column])
            }
        }
        var valueRotated = [Float](repeating: 0, count: g * d)
        for row in 0 ..< g {
            let scale = Float(record.valueAbsorbedScale[row])
            let bias = Float(record.valueAbsorbedBias[row])
            for column in 0 ..< d {
                valueRotated[row * d + column] = (
                    Float(valueQuantized[row * d + column]) * scale + bias
                ) * Float(record.valueChannelScale[column])
            }
        }

        let hadamard = normalizedHadamard(dimension: d)
        let keysTokenRotated = transpose(keyRotated, rows: d, columns: g)
        let keys = matmulRight(keysTokenRotated, rows: g, columns: d, right: hadamard)
        let values = matmulRight(valueRotated, rows: g, columns: d, right: hadamard)
        guard keys.allSatisfy(\.isFinite), values.allSatisfy(\.isFinite) else {
            throw KVarNReferenceError.nonFiniteOutput
        }
        return KVarNReferenceReconstruction(
            keysTokenMajor: keys, valuesTokenMajor: values)
    }

    private struct BalancedTile {
        let values: [Float]
        let columnScale: [Float]
        let rowScale: [Float]
    }

    private struct RTNRecord {
        let quantized: [UInt8]
        let scale: [Float]
        let bias: [Float]
    }

    private static func validatedElementCount(_ config: KVarNReferenceConfig) throws -> Int {
        guard config.headDimension > 1, isPowerOfTwo(config.headDimension),
            config.groupSize > 1, isPowerOfTwo(config.groupSize),
            [2, 4].contains(config.keyBits), [2, 4].contains(config.valueBits),
            config.iterations > 0,
            config.groupSize.isMultiple(of: 8 / config.keyBits),
            config.headDimension.isMultiple(of: 8 / config.valueBits)
        else { throw KVarNReferenceError.invalidConfig }
        let (count, overflow) = config.headDimension.multipliedReportingOverflow(
            by: config.groupSize)
        guard !overflow else { throw KVarNReferenceError.invalidConfig }
        let (_, hadamardOverflow) = config.headDimension.multipliedReportingOverflow(
            by: config.headDimension)
        guard !hadamardOverflow else { throw KVarNReferenceError.invalidConfig }
        return count
    }

    private static func normalizedHadamard(dimension: Int) -> [Float] {
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
        return matrix.map { $0 * normalization }
    }

    /// `[rows, columns] @ [columns, columns]`, row-major, with the same ascending reduction
    /// order used by the fixture's small CPU reference.
    private static func matmulRight(
        _ left: [Float], rows: Int, columns: Int, right: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                var sum: Float = 0
                for inner in 0 ..< columns {
                    sum += left[row * columns + inner] * right[inner * columns + column]
                }
                output[row * columns + column] = sum
            }
        }
        return output
    }

    private static func transpose(_ input: [Float], rows: Int, columns: Int) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                output[column * rows + row] = input[row * columns + column]
            }
        }
        return output
    }

    private static func varianceNormalize(
        _ input: [Float], rows: Int, columns: Int, iterations: Int
    ) -> BalancedTile {
        var logColumnScale = [Float](repeating: 0, count: columns)
        var logRowScale = [Float](repeating: 0, count: rows)
        var current = input
        var bestImbalance = imbalance(current, rows: rows, columns: columns)
        var bestColumnScale = [Float](repeating: 1, count: columns)
        var bestRowScale = [Float](repeating: 1, count: rows)

        for _ in 0 ..< iterations {
            let columnStd = standardDeviationByColumn(
                current, rows: rows, columns: columns)
            for column in 0 ..< columns {
                let clamped = min(stdMaximum, max(stdMinimum, columnStd[column]))
                logColumnScale[column] = min(
                    logScaleMaximum,
                    max(logScaleMinimum, logColumnScale[column] + log(clamped)))
            }
            current = divideByScales(
                input, rows: rows, columns: columns,
                columnScale: logColumnScale.map(exp), rowScale: logRowScale.map(exp))

            let rowStd = standardDeviationByRow(current, rows: rows, columns: columns)
            for row in 0 ..< rows {
                let clamped = min(stdMaximum, max(stdMinimum, rowStd[row]))
                logRowScale[row] = min(
                    logScaleMaximum,
                    max(logScaleMinimum, logRowScale[row] + log(clamped)))
            }
            let columnScale = logColumnScale.map(exp)
            let rowScale = logRowScale.map(exp)
            current = divideByScales(
                input, rows: rows, columns: columns,
                columnScale: columnScale, rowScale: rowScale)

            let nextImbalance = imbalance(current, rows: rows, columns: columns)
            if nextImbalance <= bestImbalance {
                bestImbalance = nextImbalance
                bestColumnScale = columnScale
                bestRowScale = rowScale
            }
        }
        return BalancedTile(
            values: divideByScales(
                input, rows: rows, columns: columns,
                columnScale: bestColumnScale, rowScale: bestRowScale),
            columnScale: bestColumnScale,
            rowScale: bestRowScale)
    }

    private static func divideByScales(
        _ input: [Float], rows: Int, columns: Int,
        columnScale: [Float], rowScale: [Float]
    ) -> [Float] {
        var output = input
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let byColumn = input[row * columns + column] / columnScale[column]
                output[row * columns + column] = byColumn / rowScale[row]
            }
        }
        return output
    }

    private static func imbalance(_ input: [Float], rows: Int, columns: Int) -> Float {
        let columnStd = standardDeviationByColumn(input, rows: rows, columns: columns)
        let rowStd = standardDeviationByRow(input, rows: rows, columns: columns)
        return (columnStd.max()! / max(columnStd.min()!, 1e-8))
            + (rowStd.max()! / max(rowStd.min()!, 1e-8))
    }

    /// Matches `torch.std`'s default correction=1 (unbiased sample standard deviation).
    private static func standardDeviation(_ values: [Float]) -> Float {
        let mean = values.reduce(0, +) / Float(values.count)
        var squared: Float = 0
        for value in values {
            let delta = value - mean
            squared += delta * delta
        }
        return sqrt(squared / Float(values.count - 1))
    }

    private static func standardDeviationByColumn(
        _ input: [Float], rows: Int, columns: Int
    ) -> [Float] {
        (0 ..< columns).map { column in
            standardDeviation((0 ..< rows).map { input[$0 * columns + column] })
        }
    }

    private static func standardDeviationByRow(
        _ input: [Float], rows: Int, columns: Int
    ) -> [Float] {
        (0 ..< rows).map { row in
            standardDeviation(Array(input[(row * columns) ..< ((row + 1) * columns)]))
        }
    }

    private static func asymmetricRTN(
        _ input: [Float], rows: Int, columns: Int, bits: Int
    ) throws -> RTNRecord {
        let qmax = (1 << bits) - 1
        var quantized = [UInt8](repeating: 0, count: input.count)
        var scales = [Float](repeating: 0, count: rows)
        var biases = [Float](repeating: 0, count: rows)
        for row in 0 ..< rows {
            let values = input[(row * columns) ..< ((row + 1) * columns)]
            let low = values.min()!
            let high = values.max()!
            let range = high - low
            guard range.isFinite else { throw KVarNReferenceError.nonFiniteOutput }
            let scale = max(range / Float(qmax), 1e-10)
            guard scale.isFinite else { throw KVarNReferenceError.nonFiniteOutput }
            scales[row] = scale
            biases[row] = low
            for column in 0 ..< columns {
                let raw = ((input[row * columns + column] - low) / scale)
                    .rounded(.toNearestOrEven)
                guard raw.isFinite else { throw KVarNReferenceError.nonFiniteOutput }
                quantized[row * columns + column] = UInt8(
                    min(Float(qmax), max(0, raw)))
            }
        }
        return RTNRecord(quantized: quantized, scale: scales, bias: biases)
    }

    private static func pack(
        _ input: [UInt8], rows: Int, columns: Int, bits: Int
    ) -> [UInt8] {
        let valuesPerByte = 8 / bits
        var output = [UInt8](
            repeating: 0, count: rows * (columns / valuesPerByte))
        for row in 0 ..< rows {
            for byteColumn in 0 ..< (columns / valuesPerByte) {
                var byte: UInt8 = 0
                for index in 0 ..< valuesPerByte {
                    byte |= input[row * columns + byteColumn * valuesPerByte + index]
                        << UInt8(index * bits)
                }
                output[row * (columns / valuesPerByte) + byteColumn] = byte
            }
        }
        return output
    }

    private static func unpack(
        _ input: [UInt8], rows: Int, columns: Int, bits: Int
    ) throws -> [UInt8] {
        let valuesPerByte = 8 / bits
        guard input.count == rows * (columns / valuesPerByte) else {
            throw KVarNReferenceError.invalidRecord
        }
        let mask = UInt8((1 << bits) - 1)
        var output = [UInt8](repeating: 0, count: rows * columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let byte = input[row * (columns / valuesPerByte) + column / valuesPerByte]
                output[row * columns + column] = (
                    byte >> UInt8((column % valuesPerByte) * bits)) & mask
            }
        }
        return output
    }

    private static func finiteFP16(_ values: [Float]) throws -> [Float16] {
        let converted = values.map(Float16.init)
        guard converted.allSatisfy(\.isFinite) else {
            throw KVarNReferenceError.nonFiniteOutput
        }
        return converted
    }

    private static func isPowerOfTwo(_ value: Int) -> Bool {
        value > 0 && (value & (value - 1)) == 0
    }
}
