import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import MLXNN

public enum KVTunerSensitivityCaptureError:
    Error, Equatable, Sendable
{
    case malformedModelConfig
    case unsupportedModelType(String)
    case invalidModelGeometry
    case invalidPrompt(Int)
    case invalidGroupSize(Int)
    case invalidPrecisionPair(KVTunerPrecisionPair)
    case missingModelModule(String)
    case modelReplayMismatch
    case nonFiniteMetric(prompt: Int, layer: Int, pair: Int)
}

/// Scalar reductions for the KVTuner-v5 sensitivity protocol. Arrays stay in the caller's MLX
/// isolation region; only `Double` values leave these helpers.
public enum KVTunerSensitivityMetrics {
    public static func elementwiseRelativeMeanAbsoluteError(
        reference: MLXArray,
        candidate: MLXArray,
        denominatorEpsilon: Float
    ) -> Double {
        precondition(reference.shape == candidate.shape)
        precondition(denominatorEpsilon > 0)
        let reference = reference.asType(.float32)
        let candidate = candidate.asType(.float32)
        let denominator = MLX.maximum(
            reference.abs(), MLXArray(denominatorEpsilon))
        return Double(
            ((reference - candidate).abs() / denominator)
                .mean().item(Float.self))
    }

    public static func meanAbsoluteError(
        reference: MLXArray,
        candidate: MLXArray
    ) -> Double {
        precondition(reference.shape == candidate.shape)
        return Double(
            (reference.asType(.float32) - candidate.asType(.float32))
                .abs().mean().item(Float.self))
    }
}

/// Qwen3-only offline sensitivity capture. The caller must invoke this from the actor that owns
/// `model`; no MLX array, module, cache, or tokenizer crosses the boundary. Every prompt starts
/// from the same full-precision model state, and candidate K/V round trips affect only the four
/// recorded metrics for that layer—not the hidden state forwarded to the next layer.
public enum KVTunerSensitivityCapture {
    public static let denominatorEpsilon: Float = 1e-8

    public static func capture(
        model: any LanguageModel,
        exactModelConfigData: Data,
        promptTokenIDs: [[Int]],
        groupSize: Int,
        precisionPairs: [KVTunerPrecisionPair]
    ) throws -> [KVTunerSensitivitySample] {
        let configuration: Qwen3CaptureConfiguration
        do {
            configuration = try JSONDecoder().decode(
                Qwen3CaptureConfiguration.self,
                from: exactModelConfigData)
        } catch {
            throw KVTunerSensitivityCaptureError.malformedModelConfig
        }
        guard configuration.modelType == "qwen3" else {
            throw KVTunerSensitivityCaptureError.unsupportedModelType(
                configuration.modelType)
        }
        guard [64, 128].contains(groupSize) else {
            throw KVTunerSensitivityCaptureError.invalidGroupSize(groupSize)
        }
        guard configuration.hiddenLayers > 0,
            configuration.hiddenSize > 0,
            configuration.attentionHeads > 0,
            configuration.kvHeads > 0,
            configuration.attentionHeads.isMultiple(
                of: configuration.kvHeads),
            configuration.headDimension > 0,
            configuration.headDimension.isMultiple(of: groupSize)
        else {
            throw KVTunerSensitivityCaptureError.invalidModelGeometry
        }
        for pair in precisionPairs {
            guard [2, 4, 8].contains(pair.keyBits),
                [2, 4, 8].contains(pair.valueBits)
            else {
                throw KVTunerSensitivityCaptureError.invalidPrecisionPair(
                    pair)
            }
        }
        for (ordinal, prompt) in promptTokenIDs.enumerated()
            where prompt.isEmpty
        {
            throw KVTunerSensitivityCaptureError.invalidPrompt(ordinal)
        }

        let pipeline = try Qwen3SensitivityPipeline(
            model: model, configuration: configuration)
        var samples: [KVTunerSensitivitySample] = []
        samples.reserveCapacity(
            promptTokenIDs.count * configuration.hiddenLayers
                * precisionPairs.count)
        for (promptIndex, prompt) in promptTokenIDs.enumerated() {
            let promptSamples = try pipeline.capture(
                tokenIDs: prompt,
                promptIndex: promptIndex,
                groupSize: groupSize,
                precisionPairs: precisionPairs,
                validateModelReplay: promptIndex == 0)
            samples.append(contentsOf: promptSamples)
        }
        return samples
    }
}

private struct Qwen3CaptureConfiguration: Decodable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDimension: Int
    let rmsNormEpsilon: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let maximumPositionEmbeddings: Int
    let tieWordEmbeddings: Bool

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case rmsNormEpsilon = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maximumPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: any Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try values.decode(String.self, forKey: .modelType)
        hiddenSize = try values.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try values.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try values.decode(
            Int.self, forKey: .intermediateSize)
        attentionHeads = try values.decode(
            Int.self, forKey: .attentionHeads)
        kvHeads = try values.decode(Int.self, forKey: .kvHeads)
        headDimension = try values.decode(Int.self, forKey: .headDimension)
        rmsNormEpsilon = try values.decode(
            Float.self, forKey: .rmsNormEpsilon)
        vocabularySize = try values.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try values.decodeIfPresent(
            Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeScaling = try values.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maximumPositionEmbeddings = try values.decodeIfPresent(
            Int.self, forKey: .maximumPositionEmbeddings) ?? 32_768
        tieWordEmbeddings = try values.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

private final class Qwen3SensitivityPipeline {
    private let model: any LanguageModel
    private let configuration: Qwen3CaptureConfiguration
    private let modules: [String: Module]
    private let rope: any RoPELayer

    init(
        model: any LanguageModel,
        configuration: Qwen3CaptureConfiguration
    ) throws {
        self.model = model
        self.configuration = configuration
        self.modules = Dictionary(
            uniqueKeysWithValues: model.namedModules())
        self.rope = initializeRope(
            dims: configuration.headDimension,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: configuration.ropeScaling,
            maxPositionEmbeddings:
                configuration.maximumPositionEmbeddings)

        _ = try unary("model.embed_tokens")
        _ = try unary("model.norm")
        for layer in 0 ..< configuration.hiddenLayers {
            for suffix in [
                "input_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                _ = try unary("model.layers.\(layer).\(suffix)")
            }
        }
        if !configuration.tieWordEmbeddings {
            _ = try unary("lm_head")
        }
    }

    func capture(
        tokenIDs: [Int],
        promptIndex: Int,
        groupSize: Int,
        precisionPairs: [KVTunerPrecisionPair],
        validateModelReplay: Bool
    ) throws -> [KVTunerSensitivitySample] {
        let tokenArray = MLXArray(tokenIDs).reshaped([1, tokenIDs.count])
        var hidden = try apply("model.embed_tokens", to: tokenArray)
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            tokenIDs.count == 1 ? .none : .causal
        let additiveMask = tokenIDs.count == 1
            ? nil
            : MultiHeadAttention.createAdditiveCausalMask(
                tokenIDs.count, dtype: .float32)
        var samples: [KVTunerSensitivitySample] = []
        samples.reserveCapacity(
            configuration.hiddenLayers * precisionPairs.count)

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            let normalized = try apply(
                "\(prefix).input_layernorm", to: hidden)
            let batch = normalized.dim(0)
            let length = normalized.dim(1)
            var queries = try apply(
                "\(prefix).self_attn.q_proj", to: normalized)
            var keys = try apply(
                "\(prefix).self_attn.k_proj", to: normalized)
            let values = try apply(
                "\(prefix).self_attn.v_proj", to: normalized)
            queries = try apply(
                "\(prefix).self_attn.q_norm",
                to: queries.reshaped(
                    batch, length, configuration.attentionHeads, -1))
                .transposed(0, 2, 1, 3)
            keys = try apply(
                "\(prefix).self_attn.k_norm",
                to: keys.reshaped(
                    batch, length, configuration.kvHeads, -1))
                .transposed(0, 2, 1, 3)
            var shapedValues = values.reshaped(
                batch, length, configuration.kvHeads, -1)
                .transposed(0, 2, 1, 3)
            queries = applyRope(queries)
            keys = applyRope(keys)

            let reference = attention(
                queries: queries,
                keys: keys,
                values: shapedValues,
                additiveMask: additiveMask)
            for (pairIndex, pair) in precisionPairs.enumerated() {
                let candidateKeys = affineRoundTrip(
                    keys, bits: pair.keyBits, groupSize: groupSize)
                let candidateValues = affineRoundTrip(
                    shapedValues,
                    bits: pair.valueBits, groupSize: groupSize)
                let candidate = attention(
                    queries: queries,
                    keys: candidateKeys,
                    values: candidateValues,
                    additiveMask: additiveMask)
                let sample = KVTunerSensitivitySample(
                    promptIndex: promptIndex,
                    layer: layer,
                    keyBits: pair.keyBits,
                    valueBits: pair.valueBits,
                    relativeKeyError: KVTunerSensitivityMetrics
                        .elementwiseRelativeMeanAbsoluteError(
                            reference: keys,
                            candidate: candidateKeys,
                            denominatorEpsilon:
                                KVTunerSensitivityCapture
                                    .denominatorEpsilon),
                    relativeValueError: KVTunerSensitivityMetrics
                        .elementwiseRelativeMeanAbsoluteError(
                            reference: shapedValues,
                            candidate: candidateValues,
                            denominatorEpsilon:
                                KVTunerSensitivityCapture
                                    .denominatorEpsilon),
                    attentionScoreError: KVTunerSensitivityMetrics
                        .meanAbsoluteError(
                            reference: reference.scores,
                            candidate: candidate.scores),
                    relativeAttentionOutputError: KVTunerSensitivityMetrics
                        .elementwiseRelativeMeanAbsoluteError(
                            reference: reference.output,
                            candidate: candidate.output,
                            denominatorEpsilon:
                                KVTunerSensitivityCapture
                                    .denominatorEpsilon))
                guard [
                    sample.relativeKeyError,
                    sample.relativeValueError,
                    sample.attentionScoreError,
                    sample.relativeAttentionOutputError,
                ].allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                    throw KVTunerSensitivityCaptureError.nonFiniteMetric(
                        prompt: promptIndex,
                        layer: layer,
                        pair: pairIndex)
                }
                samples.append(sample)
            }

            let attentionOutput = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: shapedValues,
                scale: pow(
                    Float(configuration.headDimension), -0.5),
                mask: attentionMask)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)
            let projected = try apply(
                "\(prefix).self_attn.o_proj", to: attentionOutput)
            let afterAttention = hidden + projected
            let postAttention = try apply(
                "\(prefix).post_attention_layernorm",
                to: afterAttention)
            let gate = try apply(
                "\(prefix).mlp.gate_proj", to: postAttention)
            let up = try apply(
                "\(prefix).mlp.up_proj", to: postAttention)
            shapedValues = try apply(
                "\(prefix).mlp.down_proj", to: silu(gate) * up)
            hidden = afterAttention + shapedValues
        }

        if validateModelReplay {
            let manualLogits = try outputLogits(hidden: hidden)
            let modelLogits = model(tokenArray, cache: nil)
            let maximumDifference = (
                manualLogits.asType(.float32)
                    - modelLogits.asType(.float32)
            ).abs().max().item(Float.self)
            guard maximumDifference <= 1e-4 else {
                throw KVTunerSensitivityCaptureError.modelReplayMismatch
            }
        }
        return samples
    }

    private func unary(_ path: String) throws -> any UnaryLayer {
        guard let module = modules[path] as? any UnaryLayer else {
            throw KVTunerSensitivityCaptureError.missingModelModule(path)
        }
        return module
    }

    private func apply(
        _ path: String,
        to input: MLXArray
    ) throws -> MLXArray {
        let layer = try unary(path)
        return layer(input)
    }

    private func applyRope(_ input: MLXArray) -> MLXArray {
        rope(input, offset: 0)
    }

    private func outputLogits(hidden: MLXArray) throws -> MLXArray {
        let normalized = try apply("model.norm", to: hidden)
        if configuration.tieWordEmbeddings {
            guard let embedding = modules["model.embed_tokens"] as? Embedding
            else {
                throw KVTunerSensitivityCaptureError.missingModelModule(
                    "model.embed_tokens")
            }
            return embedding.asLinear(normalized)
        }
        return try apply("lm_head", to: normalized)
    }

    private func affineRoundTrip(
        _ input: MLXArray,
        bits: Int,
        groupSize: Int
    ) -> MLXArray {
        let dimension = input.dim(-1)
        let flat = input.reshaped([-1, dimension])
        let code = quantized(
            flat,
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
        return dequantized(
            code.wq,
            scales: code.scales,
            biases: code.biases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: input.dtype)
            .reshaped(input.shape)
    }

    private func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        additiveMask: MLXArray?
    ) -> (scores: MLXArray, output: MLXArray) {
        let repeatCount = configuration.attentionHeads
            / configuration.kvHeads
        let expandedKeys = repeated(
            keys, count: repeatCount, axis: 1).asType(.float32)
        let expandedValues = repeated(
            values, count: repeatCount, axis: 1).asType(.float32)
        var logits = matmul(
            queries.asType(.float32)
                * pow(Float(configuration.headDimension), -0.5),
            expandedKeys.transposed(0, 1, 3, 2))
        if let additiveMask {
            logits = logits + additiveMask
        }
        let scores = softmax(logits, axis: -1, precise: true)
        return (scores, matmul(scores, expandedValues))
    }
}
