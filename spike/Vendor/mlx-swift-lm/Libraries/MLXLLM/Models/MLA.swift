import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MultiLinear

class MultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads

        let scale = sqrt(Float(1.0) / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numHeads, outputDims, inputDims]
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        let rhs = transpose ? weight.swappedAxes(-1, -2) : weight
        return x.matmul(rhs)
    }

    // MARK: - Quantizable conformance

    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        return QuantizedMultiLinear(
            weight: weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

func callMultiLinear(_ module: Module, _ x: MLXArray, transpose: Bool = true) -> MLXArray {
    if let multiLinear = module as? MultiLinear {
        return multiLinear(x, transpose: transpose)
    } else if let quantized = module as? QuantizedMultiLinear {
        return quantized(x, transpose: transpose)
    } else {
        fatalError("Module must be MultiLinear or QuantizedMultiLinear")
    }
}

// MARK: - QuantizedMultiLinear

/// Quantized version of MultiLinear that handles packed 4-bit weights.
class QuantizedMultiLinear: Module, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    /// Initialize from non-quantized weights.
    init(
        weight: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode
        )
        _weight.wrappedValue = quantizedWeight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    /// Initialize with pre-quantized weights and scales.
    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        _weight.wrappedValue = weight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        return quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: transpose,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}
