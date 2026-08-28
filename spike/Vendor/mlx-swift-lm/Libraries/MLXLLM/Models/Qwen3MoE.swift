//
//  Qwen3MoE.swift
//  LLM
//
//  Created by John Mai on 2025/4/30.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_moe.py

class Qwen3MoEAttention: Module {
    let args: Qwen3MoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Qwen3MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Qwen3MoESparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU?

    init(_ args: Qwen3MoEConfiguration, pagedExperts: Bool = false) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        if !pagedExperts {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: numExperts
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let inds = MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = MLX.takeAlong(softGates, inds, axis: -1)

        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }

        guard let switchMLP else {
            preconditionFailure("paged Qwen3MoE experts require the throwing lazy runtime")
        }
        let y = switchMLP(x, inds)
        return weightedExpertSum(y, scores)
    }

    func callPaged(
        _ x: MLXArray,
        layer: Int,
        runtime: Qwen3MoELazyExpertRuntime
    ) throws -> MLXArray {
        let gates = gate(x)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)
        let inds = MLX.argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = MLX.takeAlong(softGates, inds, axis: -1)
        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }
        let y = try runtime.switchOutput(layer: layer, input: x, globalIndices: inds)
        let output = weightedExpertSum(y, scores)
        eval(output)
        return output
    }
}

/// Package-internal proof seam for expert pages. The production `LLMModel` path deliberately
/// remains non-throwing and continues to use `SwitchGLU`; disk-backed callers must own errors at
/// this separate boundary until loader and model identity plumbing can propagate them safely.
struct Qwen3MoEPagedSwitchQuantization {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
}

struct Qwen3MoEPagedSwitchFetchResult {
    let metrics: Qwen3MoEExpertFetchMetrics
    let page: Qwen3MoEExpertPage
    let switchGLU: Qwen3MoEPagedSwitchGLU
}

final class Qwen3MoEPagedSwitchGLU {
    private let page: Qwen3MoEExpertPage
    private let inputDims: Int
    private let hiddenDims: Int
    private let localExpertByGlobalID: [Int: Int]
    private let quantization: Qwen3MoEPagedSwitchQuantization?

    init(
        page: Qwen3MoEExpertPage,
        inputDims: Int,
        hiddenDims: Int,
        quantization: Qwen3MoEPagedSwitchQuantization? = nil
    ) throws {
        guard inputDims > 0, hiddenDims > 0, !page.globalExpertIDs.isEmpty else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        guard page.globalExpertIDs == page.globalExpertIDs.sorted(),
            Set(page.globalExpertIDs).count == page.globalExpertIDs.count
        else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "expert page ids must be unique and sorted"
            )
        }
        let requiredComponents: Set<String>
        if let quantization {
            let (inputBitWidth, inputBitWidthOverflow) = inputDims.multipliedReportingOverflow(
                by: quantization.bits
            )
            let (hiddenBitWidth, hiddenBitWidthOverflow) = hiddenDims.multipliedReportingOverflow(
                by: quantization.bits
            )
            guard quantization.groupSize > 0,
                (2 ... 8).contains(quantization.bits),
                case .affine = quantization.mode,
                inputDims.isMultiple(of: quantization.groupSize),
                hiddenDims.isMultiple(of: quantization.groupSize),
                !inputBitWidthOverflow,
                !hiddenBitWidthOverflow,
                inputBitWidth.isMultiple(of: 32),
                hiddenBitWidth.isMultiple(of: 32)
            else {
                throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                    "invalid affine paged-switch quantization"
                )
            }
            requiredComponents = ["weight", "scales", "biases"]
        } else {
            requiredComponents = ["weight"]
        }
        for projection in Qwen3MoEExpertProjection.allCases {
            guard let components = page.arrays[projection],
                Set(components.keys) == requiredComponents
            else {
                let component = page.arrays[projection]?.keys.sorted().first ?? "weight"
                throw Qwen3MoEExpertResidencyError.unsupportedComponent(
                    tensor: "qwen3_moe paged \(projection.tensorComponent)",
                    component: component
                )
            }
        }
        let count = page.globalExpertIDs.count
        if let quantization {
            for projection in Qwen3MoEExpertProjection.allCases {
                let tensorPrefix = "qwen3_moe paged \(projection.tensorComponent)"
                let weight = try page.array(projection: projection)
                guard weight.dtype == .uint32 else {
                    throw Qwen3MoEExpertResidencyError.componentDTypeMismatch(
                        tensor: "\(tensorPrefix).weight",
                        expected: "U32",
                        actual: qwen3MoESafetensorsDTypeName(weight.dtype)
                    )
                }
                let scales = try page.array(projection: projection, component: "scales")
                guard scales.dtype == .float16 || scales.dtype == .bfloat16 else {
                    throw Qwen3MoEExpertResidencyError.componentDTypeMismatch(
                        tensor: "\(tensorPrefix).scales",
                        expected: "F16 or BF16",
                        actual: qwen3MoESafetensorsDTypeName(scales.dtype)
                    )
                }
                let biases = try page.array(projection: projection, component: "biases")
                guard biases.dtype == scales.dtype else {
                    throw Qwen3MoEExpertResidencyError.componentDTypeMismatch(
                        tensor: "\(tensorPrefix).biases",
                        expected: qwen3MoESafetensorsDTypeName(scales.dtype),
                        actual: qwen3MoESafetensorsDTypeName(biases.dtype)
                    )
                }
            }
            let packedInput = inputDims * quantization.bits / 32
            let packedHidden = hiddenDims * quantization.bits / 32
            let inputGroups = inputDims / quantization.groupSize
            let hiddenGroups = hiddenDims / quantization.groupSize
            for projection in [Qwen3MoEExpertProjection.gate, .up] {
                guard try page.array(projection: projection).shape
                    == [count, hiddenDims, packedInput],
                    try page.array(projection: projection, component: "scales").shape
                        == [count, hiddenDims, inputGroups],
                    try page.array(projection: projection, component: "biases").shape
                        == [count, hiddenDims, inputGroups]
                else {
                    throw Qwen3MoEExpertResidencyError.invalidShape(
                        "qwen3_moe paged \(projection.tensorComponent)"
                    )
                }
            }
            guard try page.array(projection: .down).shape
                == [count, inputDims, packedHidden],
                try page.array(projection: .down, component: "scales").shape
                    == [count, inputDims, hiddenGroups],
                try page.array(projection: .down, component: "biases").shape
                    == [count, inputDims, hiddenGroups]
            else {
                throw Qwen3MoEExpertResidencyError.invalidShape(
                    "qwen3_moe paged down_proj"
                )
            }
        } else {
            let gate = try page.array(projection: .gate)
            let up = try page.array(projection: .up)
            let down = try page.array(projection: .down)
            for (projection, weight) in [
                (Qwen3MoEExpertProjection.gate, gate),
                (.up, up),
                (.down, down),
            ] where weight.dtype != .float16 && weight.dtype != .bfloat16 {
                throw Qwen3MoEExpertResidencyError.componentDTypeMismatch(
                    tensor: "qwen3_moe paged \(projection.tensorComponent).weight",
                    expected: "F16 or BF16",
                    actual: qwen3MoESafetensorsDTypeName(weight.dtype)
                )
            }
            guard gate.shape == [count, hiddenDims, inputDims],
                up.shape == [count, hiddenDims, inputDims],
                down.shape == [count, inputDims, hiddenDims]
            else {
                throw Qwen3MoEExpertResidencyError.invalidShape("qwen3_moe paged switch")
            }
        }

        self.page = page
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.quantization = quantization
        self.localExpertByGlobalID = Dictionary(
            uniqueKeysWithValues: page.globalExpertIDs.enumerated().map { ($0.element, $0.offset) }
        )
    }

    func callAsFunction(_ input: MLXArray, _ globalIndices: MLXArray) throws -> MLXArray {
        guard input.dim(-1) == inputDims else {
            throw Qwen3MoEExpertResidencyError.invalidShape("qwen3_moe paged input")
        }
        eval(globalIndices)
        let globalIDs = globalIndices.asType(.int32).asArray(Int32.self).map(Int.init)
        let localIDs = try globalIDs.map { globalID in
            guard let localID = localExpertByGlobalID[globalID] else {
                throw Qwen3MoEExpertResidencyError.expertNotResident(globalID)
            }
            return Int32(localID)
        }
        let localIndices = MLXArray(localIDs, globalIndices.shape)

        var x = MLX.expandedDimensions(input, axes: [-2, -3])
        let doSort = localIndices.size >= 64
        var indices = localIndices
        var inverseOrder = MLXArray()
        if doSort {
            (x, indices, inverseOrder) = gatherSort(x: x, indices: localIndices)
        }

        func project(_ projection: Qwen3MoEExpertProjection, _ value: MLXArray) throws -> MLXArray {
            let weight = try page.array(projection: projection)
            if let quantization {
                return MLX.gatherQuantizedMM(
                    value,
                    weight,
                    scales: try page.array(projection: projection, component: "scales"),
                    biases: try page.array(projection: projection, component: "biases"),
                    rhsIndices: indices,
                    transpose: true,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits,
                    mode: quantization.mode,
                    sortedIndices: doSort
                )
            }
            return MLX.gatherMM(
                value,
                weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: doSort
            )
        }

        let up = try project(.up, x)
        let gate = try project(.gate, x)
        x = try project(.down, compiledSiluProduct(gate, up))
        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: localIndices.shape)
        }
        return MLX.squeezed(x, axis: -2)
    }
}

private func qwen3MoESafetensorsDTypeName(_ dtype: DType) -> String {
    switch dtype {
    case .float16: "F16"
    case .bfloat16: "BF16"
    case .uint32: "U32"
    case .uint8: "U8"
    default: String(describing: dtype)
    }
}

extension Qwen3MoEExpertResidency {
    func fetchPagedSwitch(
        layer: Int,
        routedExperts: [Int],
        reader: Qwen3MoEExpertRangeReading,
        inputDims: Int,
        hiddenDims: Int,
        quantization: Qwen3MoEPagedSwitchQuantization? = nil,
        cancellationCheck: () throws -> Void = {}
    ) throws -> Qwen3MoEPagedSwitchFetchResult {
        guard !routedExperts.isEmpty else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        var candidatePage: Qwen3MoEExpertPage?
        var candidateSwitch: Qwen3MoEPagedSwitchGLU?
        let fetched = try fetch(
            layer: layer,
            routedExperts: routedExperts,
            reader: reader,
            cancellationCheck: cancellationCheck,
            materializationCheck: { result in
                let page = try Qwen3MoEExpertPage.materialize(
                    layer: layer,
                    fetchResult: result
                )
                let switchGLU = try Qwen3MoEPagedSwitchGLU(
                    page: page,
                    inputDims: inputDims,
                    hiddenDims: hiddenDims,
                    quantization: quantization
                )
                candidatePage = page
                candidateSwitch = switchGLU
            }
        )
        guard let page = candidatePage, let switchGLU = candidateSwitch else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        return Qwen3MoEPagedSwitchFetchResult(
            metrics: fetched.metrics,
            page: page,
            switchGLU: switchGLU
        )
    }
}

class Qwen3MoeDecoderLayer: Module {
    let args: Qwen3MoEConfiguration
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3MoEAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3MoEConfiguration, layerIdx: Int, pagedExperts: Bool = false) {
        self.args = args
        self.layerIdx = layerIdx

        _selfAttn.wrappedValue = Qwen3MoEAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if !args.mlpOnlyLayers.contains(layerIdx),
            args.numExperts > 0, (layerIdx + 1) % args.decoderSparseStep == 0
        {
            self.mlp = Qwen3MoESparseMoeBlock(args, pagedExperts: pagedExperts)
        } else {
            self.mlp = Qwen3MoEMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }

    func callPaged(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        runtime: Qwen3MoELazyExpertRuntime
    ) throws -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let normalized = postAttentionLayerNorm(h)
        if let sparse = mlp as? Qwen3MoESparseMoeBlock {
            r = try sparse.callPaged(normalized, layer: layerIdx, runtime: runtime)
        } else {
            r = mlp(normalized)
        }
        return h + r
    }
}

public class Qwen3MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3MoeDecoderLayer]
    let norm: RMSNorm
    let args: Qwen3MoEConfiguration

    init(_ args: Qwen3MoEConfiguration, pagedExperts: Bool = false) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { i in
                Qwen3MoeDecoderLayer(args, layerIdx: i, pagedExperts: pagedExperts)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    func callPaged(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        runtime: Qwen3MoELazyExpertRuntime
    ) throws -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = try layer.callPaged(h, mask: mask, cache: cache?[i], runtime: runtime)
        }
        return norm(h)
    }
}

public class Qwen3MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3MoEModelInner
    let configuration: Qwen3MoEConfiguration
    private let usesPagedExperts: Bool

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public convenience init(_ args: Qwen3MoEConfiguration) {
        self.init(args, pagedExperts: false)
    }

    init(_ args: Qwen3MoEConfiguration, pagedExperts: Bool) {
        self.configuration = args
        self.usesPagedExperts = pagedExperts
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3MoEModelInner(args, pagedExperts: pagedExperts)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        precondition(!usesPagedExperts, "paged Qwen3MoE experts require the throwing lazy runtime")
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    func callPaged(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        runtime: Qwen3MoELazyExpertRuntime
    ) throws -> MLXArray {
        precondition(usesPagedExperts, "eager Qwen3MoE models do not use the lazy runtime")
        var out = try model.callPaged(inputs, cache: cache, runtime: runtime)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        if sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                if sanitizedWeights["\(prefix).mlp.experts.0.\(n).weight"] != nil {
                    let toJoin = (0 ..< configuration.numExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).mlp.experts.\(e).\(n).weight")!
                    }
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitizedWeights
    }
}

public struct Qwen3MoEConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_moe"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var decoderSparseStep: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDim: Int
    var ropeTheta: Float = 1_000_000
    var tieWordEmbeddings: Bool = false
    var maxPositionEmbeddings: Int = 32768
    var normTopkProb: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_moe"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.decoderSparseStep = try container.decode(Int.self, forKey: .decoderSparseStep)
        self.mlpOnlyLayers = try container.decode([Int].self, forKey: .mlpOnlyLayers)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

extension Qwen3MoEConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try validateRoPEConfiguration(ropeScaling, context: "Qwen3MoEConfiguration.rope_scaling")
    }
}

// MARK: - LoRA

extension Qwen3MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
