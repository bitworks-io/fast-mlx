import Foundation
import HarnessCore
import MLX
import MLXLMCommon

enum CompressedAttentionProbeRunnerError: Error, Equatable {
    case missingAffineBias
    case unsupportedLayout
    case byteCountOverflow
    case nonFiniteOutput
}

struct CompressedAttentionProbeNumericResult: Equatable, Sendable {
    let outputShape: [Int]
    let valuesFinite: Bool
    let structuralEquivalent: Bool
    let top1Matches: Bool
    let maxAbsoluteError: Double
    let meanAbsoluteError: Double
    let persistentBytes: Int
    let materializationWorkspaceBytes: Int
}

/// Owns every MLX array used by the profiling fixture. Only scalar evidence leaves the actor.
actor CompressedAttentionProbeRunner {
    private typealias QuantizedTuple = (
        weights: MLXArray, scales: MLXArray, biases: MLXArray?
    )

    private struct AffinePair {
        let keys: QuantizedTuple
        let values: QuantizedTuple
        let keyDimension: Int
        let valueDimension: Int
        let keyBits: Int
        let valueBits: Int
        let keyGroupSize: Int
        let valueGroupSize: Int
    }

    private struct ProbeMask {
        let mode: MLXFast.ScaledDotProductAttentionMaskMode
        let additive: MLXArray?
    }

    func runFixture(
        plan: CompressedAttentionProbePlan
    ) throws -> CompressedAttentionProbeNumericResult {
        guard supportsFixture(plan) else {
            throw CompressedAttentionProbeRunnerError.unsupportedLayout
        }
        let dtype = mlxDType(plan.dtype)
        let queryShape = [
            plan.batchSize, plan.queryHeadCount,
            plan.queryTokens, plan.headDimension,
        ]
        let cacheShape = [
            plan.batchSize, plan.kvHeadCount,
            plan.contextTokens, plan.headDimension,
        ]
        let queries = MLXRandom.normal(
            queryShape, key: MLXRandom.key(UInt64(plan.seed)))
            .asType(dtype)
        let keys = MLXRandom.normal(
            cacheShape, key: MLXRandom.key(UInt64(plan.seed) &+ 1))
            .asType(dtype)
        let values = MLXRandom.normal(
            cacheShape, key: MLXRandom.key(UInt64(plan.seed) &+ 2))
            .asType(dtype)
        let scale = Float(1 / sqrt(Double(plan.headDimension)))
        let mask = mlxMask(
            plan.mask,
            queryTokens: plan.queryTokens,
            contextTokens: plan.contextTokens,
            dtype: dtype)

        let output: MLXArray
        let oracle: MLXArray
        let persistentBytes: Int
        let materializationWorkspaceBytes: Int

        switch (plan.operation, plan.layout) {
        case (.fp16SDPA, .fp16):
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask.mode)
            oracle = referenceAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                additiveMask: mask.additive)
            persistentBytes = try checkedSum([keys.nbytes, values.nbytes])
            materializationWorkspaceBytes = 0

        case let (
            .swiftLMQuantizedAttention,
            .affine(keyBits, valueBits, keyGroupSize, valueGroupSize)):
            let pair = try affinePair(
                keys: keys, values: values,
                keyBits: keyBits, valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
            output = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: pair.keys,
                quantizedValues: pair.values,
                scale: scale,
                mask: mask.mode,
                groupSize: keyGroupSize,
                bits: keyBits,
                mode: .affine)
            let materialized = dequantize(pair: pair, dtype: dtype)
            oracle = referenceAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale,
                additiveMask: mask.additive)
            persistentBytes = try affinePersistentBytes(pair)
            materializationWorkspaceBytes = 0

        case let (
            .materializeThenSDPA,
            .affine(keyBits, valueBits, keyGroupSize, valueGroupSize)):
            let pair = try affinePair(
                keys: keys, values: values,
                keyBits: keyBits, valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
            let materialized = dequantize(pair: pair, dtype: dtype)
            output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale, mask: mask.mode)
            oracle = referenceAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale,
                additiveMask: mask.additive)
            persistentBytes = try affinePersistentBytes(pair)
            materializationWorkspaceBytes = try checkedSum([
                materialized.keys.nbytes,
                materialized.values.nbytes,
            ])

        default:
            throw CompressedAttentionProbeRunnerError.unsupportedLayout
        }

        eval(output, oracle)
        let finite = isFinite(output).all().item(Bool.self)
            && isFinite(oracle).all().item(Bool.self)
        guard finite else {
            throw CompressedAttentionProbeRunnerError.nonFiniteOutput
        }
        let difference = abs(
            output.asType(.float32) - oracle.asType(.float32))
        let maxError = max(difference).item(Float.self)
        let meanError = mean(difference).item(Float.self)
        let tolerance: (relative: Double, absolute: Double) =
            plan.operation == .fp16SDPA
            ? (1e-4, 1e-5)
            : (2e-3, 2e-3)
        let equivalent = output.allClose(
            oracle,
            rtol: tolerance.relative,
            atol: tolerance.absolute).item(Bool.self)
        let top1Matches = argMax(output).item(Int.self)
            == argMax(oracle).item(Int.self)

        return CompressedAttentionProbeNumericResult(
            outputShape: output.shape,
            valuesFinite: finite,
            structuralEquivalent: equivalent,
            top1Matches: top1Matches,
            maxAbsoluteError: Double(maxError),
            meanAbsoluteError: Double(meanError),
            persistentBytes: persistentBytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes)
    }

    /// A tiny semantic canary for lower-right causal alignment. With zero queries/keys the
    /// attention weights are uniform over allowed positions, so the two outputs are independently
    /// checkable means: (1 + 2 + 3) / 3 and (1 + 2 + 3 + 100) / 4.
    func runCausalAlignmentFixture() throws -> [Float] {
        let queries = MLXArray.zeros([1, 1, 2, 1], dtype: .float16)
        let keys = MLXArray.zeros([1, 1, 4, 1], dtype: .float16)
        let values = MLXArray([Float(1), 2, 3, 100])
            .asType(.float16)
            .reshaped([1, 1, 4, 1])
        let mask = mlxMask(
            .causal,
            queryTokens: 2,
            contextTokens: 4,
            dtype: .float16)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1,
            mask: mask.mode)
        eval(output)
        guard isFinite(output).all().item(Bool.self) else {
            throw CompressedAttentionProbeRunnerError.nonFiniteOutput
        }
        return output.asType(.float32).asArray(Float.self)
    }

    private func supportsFixture(_ plan: CompressedAttentionProbePlan) -> Bool {
        switch (plan.operation, plan.layout) {
        case (.fp16SDPA, .fp16),
            (.swiftLMQuantizedAttention, .affine),
            (.materializeThenSDPA, .affine):
            return true
        default:
            return false
        }
    }

    /// An independent reference for the stock/fused operations. This deliberately expands the
    /// scaled QK matmul, precise softmax, and AV matmul instead of calling stock SDPA, so a shared
    /// mask, scale, GQA, or layout bug cannot make the structural control pass by self-comparison.
    private func referenceAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        additiveMask: MLXArray?
    ) -> MLXArray {
        let repeatCount = queries.dim(1) / keys.dim(1)
        let expandedKeys = repeatCount > 1
            ? repeated(keys, count: repeatCount, axis: 1)
            : keys
        let expandedValues = repeatCount > 1
            ? repeated(values, count: repeatCount, axis: 1)
            : values
        var logits = matmul(
            queries * scale,
            expandedKeys.transposed(0, 1, 3, 2))
        if let additiveMask {
            logits = logits + additiveMask.asType(logits.dtype)
        }
        let weights = softmax(logits, axis: -1, precise: true)
        return matmul(weights, expandedValues)
    }

    private func affinePair(
        keys: MLXArray,
        values: MLXArray,
        keyBits: Int,
        valueBits: Int,
        keyGroupSize: Int,
        valueGroupSize: Int
    ) throws -> AffinePair {
        let keyTuple = try quantize(
            keys, bits: keyBits, groupSize: keyGroupSize)
        let valueTuple = try quantize(
            values, bits: valueBits, groupSize: valueGroupSize)
        eval([
            keyTuple.weights, keyTuple.scales, keyTuple.biases!,
            valueTuple.weights, valueTuple.scales, valueTuple.biases!,
        ])
        return AffinePair(
            keys: keyTuple,
            values: valueTuple,
            keyDimension: keys.dim(-1),
            valueDimension: values.dim(-1),
            keyBits: keyBits,
            valueBits: valueBits,
            keyGroupSize: keyGroupSize,
            valueGroupSize: valueGroupSize)
    }

    private func quantize(
        _ array: MLXArray,
        bits: Int,
        groupSize: Int
    ) throws -> QuantizedTuple {
        let batch = array.dim(0)
        let heads = array.dim(1)
        let tokens = array.dim(2)
        let dimension = array.dim(3)
        let code = MLX.quantized(
            array.reshaped([-1, dimension]),
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
        guard let biases = code.biases else {
            throw CompressedAttentionProbeRunnerError.missingAffineBias
        }
        return (
            code.wq.reshaped([batch, heads, tokens, code.wq.dim(-1)]),
            code.scales.reshaped([
                batch, heads, tokens, code.scales.dim(-1),
            ]),
            biases.reshaped([batch, heads, tokens, biases.dim(-1)]))
    }

    private func dequantize(
        pair: AffinePair,
        dtype: DType
    ) -> (keys: MLXArray, values: MLXArray) {
        (
            dequantize(
                pair.keys,
                dimension: pair.keyDimension,
                bits: pair.keyBits,
                groupSize: pair.keyGroupSize,
                dtype: dtype),
            dequantize(
                pair.values,
                dimension: pair.valueDimension,
                bits: pair.valueBits,
                groupSize: pair.valueGroupSize,
                dtype: dtype)
        )
    }

    private func dequantize(
        _ tuple: QuantizedTuple,
        dimension: Int,
        bits: Int,
        groupSize: Int,
        dtype: DType
    ) -> MLXArray {
        let batch = tuple.weights.dim(0)
        let heads = tuple.weights.dim(1)
        let tokens = tuple.weights.dim(2)
        let packedWidth = tuple.weights.dim(3)
        let metadataWidth = tuple.scales.dim(3)
        return MLX.dequantized(
            tuple.weights.reshaped([-1, packedWidth]),
            scales: tuple.scales.reshaped([-1, metadataWidth]),
            biases: tuple.biases?.reshaped([-1, metadataWidth]),
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: dtype
        ).reshaped([batch, heads, tokens, dimension])
    }

    private func affinePersistentBytes(_ pair: AffinePair) throws -> Int {
        try checkedSum([
            pair.keys.weights.nbytes,
            pair.keys.scales.nbytes,
            pair.keys.biases?.nbytes ?? 0,
            pair.values.weights.nbytes,
            pair.values.scales.nbytes,
            pair.values.biases?.nbytes ?? 0,
        ])
    }

    private func checkedSum(_ values: [Int]) throws -> Int {
        var result = 0
        for value in values {
            let (next, overflow) = result.addingReportingOverflow(value)
            guard !overflow else {
                throw CompressedAttentionProbeRunnerError.byteCountOverflow
            }
            result = next
        }
        return result
    }

    private func mlxDType(
        _ dtype: CompressedAttentionProbeDType
    ) -> DType {
        switch dtype {
        case .float16: .float16
        case .bfloat16: .bfloat16
        }
    }

    private func mlxMask(
        _ mask: CompressedAttentionProbeMask,
        queryTokens: Int,
        contextTokens: Int,
        dtype: DType
    ) -> ProbeMask {
        switch mask {
        case .none:
            return ProbeMask(mode: .none, additive: nil)
        case .causal:
            // The pinned Swift-LM helper's symbolic `.causal` branch substitutes
            // `Float.leastNormalMagnitude` for masked scores. That value is positive, so
            // prefill-shaped queries can give future positions softmax mass. An additive
            // lower-right mask preserves the requested causal contract for both the packed
            // helper and its stock-SDPA oracle.
            let queryPositions = MLXArray(0 ..< queryTokens)
                + MLXArray(contextTokens - queryTokens)
            let keyPositions = MLXArray(0 ..< contextTokens)
            let allowed = greaterEqual(
                queryPositions.expandedDimensions(axis: -1),
                keyPositions.expandedDimensions(axis: -2))
            let additive = MLX.where(
                allowed,
                MLXArray(Float(0)),
                MLXArray(-Float.infinity)).asType(dtype)
            return ProbeMask(mode: .array(additive), additive: additive)
        }
    }
}
