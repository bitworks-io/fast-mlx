// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v3.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV3Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }
}

private func yarnFindCorrectionDim(
    numRotations: Float, dim: Float, base: Float = 10000, maxPositionEmbeddings: Float = 2048
) -> Float {
    return (dim * log(maxPositionEmbeddings / (numRotations * 2 * Float.pi))) / (2 * log(base))
}

private func yarnFindCorrectionRange(
    lowRot: Float, highRot: Float, dim: Float, base: Float = 10000,
    maxPositionEmbeddings: Float = 2048
) -> (Float, Float) {
    let low = floor(
        yarnFindCorrectionDim(
            numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings
        ))
    let high = ceil(
        yarnFindCorrectionDim(
            numRotations: highRot, dim: dim, base: base,
            maxPositionEmbeddings: maxPositionEmbeddings))
    return (max(low, 0), min(high, dim - 1))
}

private func clippedSilu(_ x: MLXArray) -> MLXArray {
    clip(x * sigmoid(x), min: -100, max: 100)
}

private func absorbedMLAAttentionMask(
    rotaryScores: MLXArray,
    baseMask: MLXFast.ScaledDotProductAttentionMaskMode,
    queryLength: Int,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let negativeInfinity = MLXArray(-Float.infinity).asType(rotaryScores.dtype)

    func merge(_ array: MLXArray) -> MLXArray {
        if array.dtype == .bool {
            return MLX.where(array, rotaryScores, negativeInfinity)
        }
        return rotaryScores + array.asType(rotaryScores.dtype)
    }

    switch baseMask {
    case .none:
        return .array(rotaryScores)
    case .causal:
        precondition(
            keyLength >= queryLength,
            "MLA attention key length cannot be shorter than query length")
        let causal = createCausalMask(
            n: queryLength, offset: keyLength - queryLength)
        return .array(merge(causal))
    case .array(let array):
        return .array(merge(array))
    case .arrays(let arrays):
        precondition(
            arrays.count <= 1,
            "DeepSeek-V3 absorbed MLA supports at most one attention mask array")
        return .array(arrays.first.map(merge) ?? rotaryScores)
    }
}

private struct DeepseekV3MLAQuantizationGeometry: Equatable {
    let bits: Int
    let groupSize: Int
}

func deepseekV3MLASupportsAffineBits(_ bits: Int) -> Bool {
    [2, 3, 4, 5, 6, 8].contains(bits)
}

enum DeepseekV3MLAWeightConversionError: Error, Equatable, CustomStringConvertible {
    case incompletePackedInt4(base: String)
    case mixedPackedInt4(base: String)
    case missingProjection(layer: Int)
    case mixedProjection(layer: Int)
    case mixedConvertedQuantization(layer: Int)
    case incompleteQuantization(layer: Int, projection: String)
    case invalidShape(layer: Int, projection: String, expected: [Int], actual: [Int])
    case invalidQuantization(layer: Int, projection: String)
    case missingLoaderQuantization(layer: Int, projection: String)
    case loaderQuantizationMismatch(
        layer: Int,
        projection: String,
        expectedBits: Int,
        expectedGroupSize: Int,
        actualBits: Int,
        actualGroupSize: Int)
    case unsupportedLoaderQuantizationMode(layer: Int, projection: String)

    var description: String {
        switch self {
        case .incompletePackedInt4(let base):
            return "incomplete packed-int4 wrapper for \(base)"
        case .mixedPackedInt4(let base):
            return "packed-int4 and normalized weights are both present for \(base)"
        case .missingProjection(let layer):
            return "missing MLA projection weights for layer \(layer)"
        case .mixedProjection(let layer):
            return "source and converted MLA projection weights are both present for layer \(layer)"
        case .mixedConvertedQuantization(let layer):
            return
                "converted MLA projections use different affine quantization geometry for layer \(layer)"
        case .incompleteQuantization(let layer, let projection):
            return "incomplete affine metadata for layer \(layer) \(projection)"
        case .invalidShape(let layer, let projection, let expected, let actual):
            return
                "invalid layer \(layer) \(projection) shape \(actual); expected \(expected)"
        case .invalidQuantization(let layer, let projection):
            return "invalid affine geometry for layer \(layer) \(projection)"
        case .missingLoaderQuantization(let layer, let projection):
            return
                "missing loader quantization policy for affine layer \(layer) \(projection)"
        case .loaderQuantizationMismatch(
            let layer,
            let projection,
            let expectedBits,
            let expectedGroupSize,
            let actualBits,
            let actualGroupSize):
            return
                "loader quantization for layer \(layer) \(projection) is bits=\(actualBits), group=\(actualGroupSize); checkpoint requires bits=\(expectedBits), group=\(expectedGroupSize)"
        case .unsupportedLoaderQuantizationMode(let layer, let projection):
            return
                "loader quantization for layer \(layer) \(projection) must use affine mode"
        }
    }
}

enum DeepseekV3MLACachePolicyError: Error, Equatable, CustomStringConvertible {
    case rotatingCacheUnsupported
    case quantizedCacheUnsupported
    case unsupportedCacheType(String)
    case incorrectCacheCount(expected: Int, actual: Int)
    case mismatchedStorageDTypes(latent: String, rotary: String)
    case unsupportedStorageDType(String)

    var description: String {
        switch self {
        case .rotatingCacheUnsupported:
            return "DeepSeek-V3 absorbed MLA does not support a rotating KV cache"
        case .quantizedCacheUnsupported:
            return "DeepSeek-V3 absorbed MLA requires fp16/bfloat16 KV-cache storage"
        case .unsupportedCacheType(let type):
            return
                "DeepSeek-V3 absorbed MLA cache type \(type) does not preserve independent latent and rotary state"
        case .incorrectCacheCount(let expected, let actual):
            return
                "DeepSeek-V3 absorbed MLA requires exactly \(expected) layer caches, got \(actual)"
        case .mismatchedStorageDTypes(let latent, let rotary):
            return
                "DeepSeek-V3 absorbed MLA cache dtypes differ: latent=\(latent) rotary=\(rotary)"
        case .unsupportedStorageDType(let dtype):
            return
                "DeepSeek-V3 absorbed MLA cache requires float16 or bfloat16 storage, got \(dtype)"
        }
    }
}

func validateDeepseekV3MLACacheInstance(
    _ cache: any KVCache
) throws {
    guard cache is any MLALatentKVCache else {
        throw DeepseekV3MLACachePolicyError
            .unsupportedCacheType(String(describing: type(of: cache)))
    }
}

func validateDeepseekV3MLACacheInstances(
    _ caches: [KVCache]?,
    expectedLayerCount: Int
) throws {
    guard let caches else {
        return
    }
    guard caches.count == expectedLayerCount else {
        throw DeepseekV3MLACachePolicyError.incorrectCacheCount(
            expected: expectedLayerCount,
            actual: caches.count)
    }
    for cache in caches {
        try validateDeepseekV3MLACacheInstance(cache)
    }
}

func validateDeepseekV3MLACacheParameters(
    _ parameters: GenerateParameters?
) throws {
    guard parameters?.maxKVSize == nil else {
        throw DeepseekV3MLACachePolicyError.rotatingCacheUnsupported
    }
    guard parameters?.kvBits == nil, parameters?.kvScheme == nil else {
        throw DeepseekV3MLACachePolicyError.quantizedCacheUnsupported
    }
}

func validateDeepseekV3MLACacheStorage(
    latent: MLXArray,
    rotary: MLXArray
) throws {
    guard latent.dtype == rotary.dtype else {
        throw DeepseekV3MLACachePolicyError.mismatchedStorageDTypes(
            latent: "\(latent.dtype)",
            rotary: "\(rotary.dtype)")
    }
    guard latent.dtype == .float16 || latent.dtype == .bfloat16 else {
        throw DeepseekV3MLACachePolicyError
            .unsupportedStorageDType("\(latent.dtype)")
    }
}

func normalizeDeepseekV3PackedInt4Weights(
    _ weights: [String: MLXArray]
) throws -> [String: MLXArray] {
    let wrapperSuffixes = ["weight_shape", "weight_packed", "weight_scale"]
    var bases = Set<String>()
    for key in weights.keys {
        for suffix in wrapperSuffixes where key.hasSuffix(suffix) {
            bases.insert(String(key.dropLast(suffix.count)))
        }
    }

    var normalized = weights
    for base in bases.sorted() {
        let wrapperKeys = wrapperSuffixes.map { base + $0 }
        let presentWrapperCount = wrapperKeys.filter { weights[$0] != nil }.count
        guard presentWrapperCount == wrapperKeys.count else {
            throw DeepseekV3MLAWeightConversionError
                .incompletePackedInt4(base: base)
        }
        let normalizedKeys = [base + "weight", base + "scales", base + "biases"]
        guard normalizedKeys.allSatisfy({ weights[$0] == nil }) else {
            throw DeepseekV3MLAWeightConversionError
                .mixedPackedInt4(base: base)
        }

        let packed = weights[base + "weight_packed"]!
        let scales = weights[base + "weight_scale"]!
        normalized[base + "weight"] = packed.view(dtype: .uint32)
        normalized[base + "scales"] = scales
        normalized[base + "biases"] = scales * -8
        for key in wrapperKeys {
            normalized.removeValue(forKey: key)
        }
    }
    return normalized
}

func deepseekV3MLAWeightQuantizationSourcePath(
    for sanitizedPath: String
) -> String? {
    for derivedName in ["embed_q", "unembed_out"] {
        let suffix = ".self_attn.\(derivedName)"
        guard sanitizedPath.hasSuffix(suffix) else {
            continue
        }
        return String(sanitizedPath.dropLast(derivedName.count))
            + "kv_b_proj"
    }
    return nil
}

private func validateDeepseekV3ConvertedProjection(
    weights: [String: MLXArray],
    layer: Int,
    prefix: String,
    name: String,
    expectedShape: [Int]
) throws -> DeepseekV3MLAQuantizationGeometry? {
    let weightKey = "\(prefix).\(name).weight"
    let scalesKey = "\(prefix).\(name).scales"
    let biasesKey = "\(prefix).\(name).biases"
    guard let weight = weights[weightKey] else {
        throw DeepseekV3MLAWeightConversionError
            .missingProjection(layer: layer)
    }
    let scales = weights[scalesKey]
    let biases = weights[biasesKey]
    guard (scales == nil) == (biases == nil) else {
        throw DeepseekV3MLAWeightConversionError
            .incompleteQuantization(layer: layer, projection: name)
    }

    guard let scales, let biases else {
        guard weight.shape == expectedShape else {
            throw DeepseekV3MLAWeightConversionError.invalidShape(
                layer: layer,
                projection: name,
                expected: expectedShape,
                actual: weight.shape)
        }
        return nil
    }

    guard weight.ndim == expectedShape.count,
        scales.ndim == expectedShape.count,
        biases.shape == scales.shape,
        Array(weight.shape.dropLast()) == Array(expectedShape.dropLast()),
        Array(scales.shape.dropLast()) == Array(expectedShape.dropLast()),
        scales.dim(-1) > 0,
        expectedShape.last! % scales.dim(-1) == 0
    else {
        throw DeepseekV3MLAWeightConversionError
            .invalidQuantization(layer: layer, projection: name)
    }
    let bitNumerator = weight.dim(-1) * 32
    guard bitNumerator % expectedShape.last! == 0 else {
        throw DeepseekV3MLAWeightConversionError
            .invalidQuantization(layer: layer, projection: name)
    }
    let bits = bitNumerator / expectedShape.last!
    let groupSize = expectedShape.last! / scales.dim(-1)
    guard deepseekV3MLASupportsAffineBits(bits), groupSize > 0 else {
        throw DeepseekV3MLAWeightConversionError
            .invalidQuantization(layer: layer, projection: name)
    }
    let unpacked = dequantized(
        weight,
        scales: scales,
        biases: biases,
        groupSize: groupSize,
        bits: bits)
    guard unpacked.shape == expectedShape else {
        throw DeepseekV3MLAWeightConversionError.invalidShape(
            layer: layer,
            projection: name,
            expected: expectedShape,
            actual: unpacked.shape)
    }
    return DeepseekV3MLAQuantizationGeometry(
        bits: bits,
        groupSize: groupSize)
}

func convertDeepseekV3MLAProjectionWeights(
    _ weights: [String: MLXArray],
    configuration: DeepseekV3Configuration
) throws -> [String: MLXArray] {
    var converted = weights
    let headDimension = configuration.qkNopeHeadDim + configuration.vHeadDim
    let sourceShape = [
        configuration.numAttentionHeads * headDimension,
        configuration.kvLoraRank,
    ]
    let embedShape = [
        configuration.numAttentionHeads,
        configuration.kvLoraRank,
        configuration.qkNopeHeadDim,
    ]
    let unembedShape = [
        configuration.numAttentionHeads,
        configuration.vHeadDim,
        configuration.kvLoraRank,
    ]

    for layer in 0 ..< configuration.numHiddenLayers {
        let prefix = "model.layers.\(layer).self_attn"
        let sourceKeys = [
            "\(prefix).kv_b_proj.weight",
            "\(prefix).kv_b_proj.scales",
            "\(prefix).kv_b_proj.biases",
        ]
        let convertedKeys = [
            "\(prefix).embed_q.weight",
            "\(prefix).embed_q.scales",
            "\(prefix).embed_q.biases",
            "\(prefix).unembed_out.weight",
            "\(prefix).unembed_out.scales",
            "\(prefix).unembed_out.biases",
        ]
        let hasSource = sourceKeys.contains { converted[$0] != nil }
        let hasConverted = convertedKeys.contains { converted[$0] != nil }
        guard !(hasSource && hasConverted) else {
            throw DeepseekV3MLAWeightConversionError
                .mixedProjection(layer: layer)
        }
        guard hasSource || hasConverted else {
            throw DeepseekV3MLAWeightConversionError
                .missingProjection(layer: layer)
        }

        if hasConverted {
            let embedGeometry = try validateDeepseekV3ConvertedProjection(
                weights: converted,
                layer: layer,
                prefix: prefix,
                name: "embed_q",
                expectedShape: embedShape)
            let unembedGeometry = try validateDeepseekV3ConvertedProjection(
                weights: converted,
                layer: layer,
                prefix: prefix,
                name: "unembed_out",
                expectedShape: unembedShape)
            guard embedGeometry == unembedGeometry else {
                throw DeepseekV3MLAWeightConversionError
                    .mixedConvertedQuantization(layer: layer)
            }
            continue
        }

        guard var projection = converted["\(prefix).kv_b_proj.weight"] else {
            throw DeepseekV3MLAWeightConversionError
                .missingProjection(layer: layer)
        }
        let scales = converted["\(prefix).kv_b_proj.scales"]
        let biases = converted["\(prefix).kv_b_proj.biases"]
        guard (scales == nil) == (biases == nil) else {
            throw DeepseekV3MLAWeightConversionError
                .incompleteQuantization(layer: layer, projection: "kv_b_proj")
        }

        var inferredBits: Int?
        var inferredGroupSize: Int?
        if let scales, let biases {
            guard projection.ndim == 2,
                scales.ndim == 2,
                biases.shape == scales.shape,
                projection.dim(0) == sourceShape[0],
                scales.dim(0) == sourceShape[0],
                scales.dim(-1) > 0,
                configuration.kvLoraRank % scales.dim(-1) == 0
            else {
                throw DeepseekV3MLAWeightConversionError
                    .invalidQuantization(layer: layer, projection: "kv_b_proj")
            }
            let bitNumerator = projection.dim(-1) * 32
            guard bitNumerator % configuration.kvLoraRank == 0 else {
                throw DeepseekV3MLAWeightConversionError
                    .invalidQuantization(layer: layer, projection: "kv_b_proj")
            }
            let bits = bitNumerator / configuration.kvLoraRank
            let groupSize = configuration.kvLoraRank / scales.dim(-1)
            guard deepseekV3MLASupportsAffineBits(bits), groupSize > 0 else {
                throw DeepseekV3MLAWeightConversionError
                    .invalidQuantization(layer: layer, projection: "kv_b_proj")
            }
            inferredBits = bits
            inferredGroupSize = groupSize
            projection = dequantized(
                projection,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits)
        }

        guard projection.shape == sourceShape else {
            throw DeepseekV3MLAWeightConversionError.invalidShape(
                layer: layer,
                projection: "kv_b_proj",
                expected: sourceShape,
                actual: projection.shape)
        }

        projection = projection.reshaped(
            configuration.numAttentionHeads,
            headDimension,
            configuration.kvLoraRank)
        var embedWeight = contiguous(
            projection[
                0..., ..<configuration.qkNopeHeadDim, 0...
            ].swappedAxes(-1, -2))
        var unembedWeight = contiguous(
            projection[
                0..., configuration.qkNopeHeadDim..., 0...
            ])

        if let bits = inferredBits, let groupSize = inferredGroupSize {
            let (qEmbed, embedScales, embedBiases) = MLX.quantized(
                embedWeight, groupSize: groupSize, bits: bits)
            let (qUnembed, unembedScales, unembedBiases) = MLX.quantized(
                unembedWeight, groupSize: groupSize, bits: bits)
            embedWeight = qEmbed
            unembedWeight = qUnembed
            converted["\(prefix).embed_q.scales"] = embedScales
            converted["\(prefix).embed_q.biases"] = embedBiases
            converted["\(prefix).unembed_out.scales"] = unembedScales
            converted["\(prefix).unembed_out.biases"] = unembedBiases
        }

        for key in sourceKeys {
            converted.removeValue(forKey: key)
        }
        converted["\(prefix).embed_q.weight"] = embedWeight
        converted["\(prefix).unembed_out.weight"] = unembedWeight
    }

    return converted
}

func validateDeepseekV3MLALoaderQuantization(
    weights: [String: MLXArray],
    configuration: DeepseekV3Configuration,
    quantization: BaseConfiguration.Quantization?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) throws {
    let projections: [(name: String, shape: [Int])] = [
        (
            "embed_q",
            [
                configuration.numAttentionHeads,
                configuration.kvLoraRank,
                configuration.qkNopeHeadDim,
            ]
        ),
        (
            "unembed_out",
            [
                configuration.numAttentionHeads,
                configuration.vHeadDim,
                configuration.kvLoraRank,
            ]
        ),
    ]

    for layer in 0 ..< configuration.numHiddenLayers {
        let prefix = "model.layers.\(layer).self_attn"
        for projection in projections {
            guard
                let geometry = try validateDeepseekV3ConvertedProjection(
                    weights: weights,
                    layer: layer,
                    prefix: prefix,
                    name: projection.name,
                    expectedShape: projection.shape)
            else {
                continue
            }

            let sanitizedPath = "\(prefix).\(projection.name)"
            let sourcePath = deepseekV3MLAWeightQuantizationSourcePath(
                for: sanitizedPath)
            let selected: BaseConfiguration.Quantization?
            if let perLayerQuantization {
                let quantizationPath = resolvedWeightQuantizationPath(
                    sanitizedPath: sanitizedPath,
                    sourcePath: sourcePath,
                    perLayerQuantization: perLayerQuantization)
                selected = perLayerQuantization.quantization(
                    layer: quantizationPath)
            } else {
                selected = quantization
            }

            guard let selected else {
                throw DeepseekV3MLAWeightConversionError
                    .missingLoaderQuantization(
                        layer: layer,
                        projection: projection.name)
            }
            guard case .affine = selected.mode else {
                throw DeepseekV3MLAWeightConversionError
                    .unsupportedLoaderQuantizationMode(
                        layer: layer,
                        projection: projection.name)
            }
            guard selected.bits == geometry.bits,
                selected.groupSize == geometry.groupSize
            else {
                throw DeepseekV3MLAWeightConversionError
                    .loaderQuantizationMismatch(
                        layer: layer,
                        projection: projection.name,
                        expectedBits: geometry.bits,
                        expectedGroupSize: geometry.groupSize,
                        actualBits: selected.bits,
                        actualGroupSize: selected.groupSize)
            }
        }
    }
}

class DeepseekV3Attention: Module {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var numHeads: Int
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var kvLoraRank: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = qLoraRank {
            self._qAProj.wrappedValue = Linear(
                hiddenSize, qLoraRank, bias: config.attentionBias
            )
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false
            )
        } else {
            self._qProj.wrappedValue = Linear(hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias
        )
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim,
            outputDims: kvLoraRank,
            numHeads: numHeads
        )
        self._unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank,
            outputDims: vHeadDim,
            numHeads: numHeads
        )
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    self.scale = self.scale * s * s
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q: MLXArray
        if qLoraRank == nil {
            q = self.qProj!(x)
        } else {
            q = self.qBProj!(self.qALayerNorm!(self.qAProj!(x)))
        }

        q = q.reshaped(B, L, self.numHeads, self.qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])
        var compressedKv = self.kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, self.qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kvLatent = expandedDimensions(kvLatent, axis: 1)
        if let cache = cache {
            do {
                try validateDeepseekV3MLACacheInstance(cache)
                try validateDeepseekV3MLACacheStorage(
                    latent: kvLatent,
                    rotary: kPe)
            } catch {
                preconditionFailure(
                    "DeepSeek-V3 cache mutation rejected: \(error)")
            }
            (kvLatent, kPe) = cache.update(keys: kvLatent, values: kPe)
        }

        let rotaryScores = matmul(
            qPe * scale,
            kPe.swappedAxes(-1, -2)
        )
        let absorbedMask = absorbedMLAAttentionMask(
            rotaryScores: rotaryScores,
            baseMask: mask,
            queryLength: L,
            keyLength: kvLatent.dim(2)
        )

        let keys: MLXArray
        let values: MLXArray
        if L == 1 {
            qNope = callMultiLinear(embedQ, qNope)
            keys = kvLatent
            values = kvLatent
        } else {
            keys = callMultiLinear(embedQ, kvLatent, transpose: false)
            values = callMultiLinear(unembedOut, kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope,
            keys: keys,
            values: values,
            scale: scale,
            mask: absorbedMask
        )
        if L == 1 {
            output = callMultiLinear(unembedOut, output)
        }
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return self.oProj(output)
    }
}

class DeepseekV3MLP: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var hiddenSize: Int
    var intermediateSize: Int
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: DeepseekV3Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.config = config
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.downProj(silu(self.gateProj(x)) * self.upProj(x))
    }
}

class MoEGate: Module {
    var config: DeepseekV3Configuration
    var topK: Int?
    var normTopkProb: Bool
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var nGroup: Int
    var topkGroup: Int?

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup
        self.weight = zeros([self.nRoutedExperts ?? 1, config.hiddenSize])
        self.e_score_correction_bias = zeros([self.nRoutedExperts ?? 1])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        let hiddenStates = x.matmul(weight.T)
        var scores = sigmoid(hiddenStates)
        let scoresForChoice = scores + e_score_correction_bias
        let groupScores = scoresForChoice.reshaped(bsz, seqLen, self.nGroup, -1)
        let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        var k = nGroup - (topkGroup ?? 1)
        var groupIdx = argPartition(topKGroup, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, (nRoutedExperts ?? 1) / nGroup])
        scores = putAlong(groupScores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)

        k = topK ?? 1
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(scores, inds, axis: -1)
        if topK ?? 1 > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true) + 1e-20
            scores = scores / denominator
            scores = scores * routedScalingFactor
        }

        return (inds, scores)
    }
}

class DeepseekV3MoE: Module, UnaryLayer {
    var config: DeepseekV3Configuration
    var numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1

        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1,
            activation: clippedSilu
        )

        self.gate = MoEGate(config: config)

        if let sharedExpertCount = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * sharedExpertCount
            self._sharedExperts.wrappedValue = DeepseekV3MLP(
                config: config, intermediateSize: intermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores)

        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

class DeepseekV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV3Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3MoE(config: config)
        } else {
            self.mlp = DeepseekV3MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class DeepseekV3ModelInner: Module {
    var config: DeepseekV3Configuration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV3DecoderLayer]
    var startIdx: Int
    var endIdx: Int
    var numLayers: Int
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var pipelineRank: Int
    var pipelineSize: Int

    init(config: DeepseekV3Configuration) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3DecoderLayer(config: config, layerIdx: $0)
        }
        self.startIdx = 0
        self.endIdx = layers.count
        self.numLayers = endIdx
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.pipelineRank = 0
        self.pipelineSize = 1
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        do {
            try validateDeepseekV3MLACacheInstances(
                cache,
                expectedLayerCount: layers.count)
        } catch {
            preconditionFailure(
                "DeepSeek-V3 cache admission rejected: \(error)")
        }

        var h = embedTokens(x)

        let attentionMask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: attentionMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class DeepseekV3Model:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    WeightQuantizationPathResolver, SanitizedWeightQuantizationValidator
{
    public let kvHeads: [Int]

    var args: DeepseekV3Configuration
    public var model: DeepseekV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV3Configuration) {
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers)
        self.args = args
        self.model = DeepseekV3ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        do {
            try validateDeepseekV3MLACacheParameters(parameters)
        } catch {
            preconditionFailure("\(error)")
        }
        return (0 ..< args.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sourceWeightQuantizationPath(
        for sanitizedPath: String
    ) -> String? {
        deepseekV3MLAWeightQuantizationSourcePath(
            for: sanitizedPath)
    }

    public func validateSanitizedWeightQuantization(
        weights: [String: MLXArray],
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        try validateDeepseekV3MLALoaderQuantization(
            weights: weights,
            configuration: args,
            quantization: quantization,
            perLayerQuantization: perLayerQuantization)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray]
        do {
            newWeights = try normalizeDeepseekV3PackedInt4Weights(weights)
        } catch {
            fatalError("DeepSeek-V3 weight conversion failed: \(error)")
        }
        let normalizedWeights = newWeights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in normalizedWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = normalizedWeights[weightKey] {
                    let dequantized = dequant(weight: weight, scaleInv: value)
                    newWeights[weightKey] = dequantized
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if newWeights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            newWeights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        do {
            newWeights = try convertDeepseekV3MLAProjectionWeights(
                newWeights, configuration: args)
        } catch {
            fatalError("DeepSeek-V3 weight conversion failed: \(error)")
        }

        return newWeights.filter { key, _ in
            !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
