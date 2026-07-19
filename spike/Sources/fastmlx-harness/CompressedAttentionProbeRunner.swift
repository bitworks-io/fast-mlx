import CryptoKit
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
    let maxRelativeError: Double
    let maximumToleranceRatio: Double
    let meanAbsoluteError: Double
    let outputTop1Index: Int
    let oracleTop1Index: Int
    let attentionSeconds: Double
    let payloadBytes: Int
    let scaleBytes: Int
    let biasBytes: Int
    let controlBytes: Int
    let alignmentPaddingBytes: Int
    let fp16ResidentBytes: Int
    let persistentBytes: Int
    let materializationWorkspaceBytes: Int
    let sourceKVTensorSHA256: String
    let packedKVTensorSHA256: String
    let queryTensorSHA256: String
    let outputTensorSHA256: String
}

struct CompressedAttentionProbeHashCoverageResult: Equatable, Sendable {
    let baseTensorSHA256: String
    let middleMutationSHA256: String
    let packedWordASHA256: String
    let packedWordBSHA256: String
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

    func configureMemory(memoryLimitBytes: Int, cacheLimitBytes: Int) {
        Memory.memoryLimit = memoryLimitBytes
        Memory.cacheLimit = cacheLimitBytes
    }

    func resetPeakMemory() {
        Memory.peakMemory = 0
    }

    func memorySnapshot() -> CompressedAttentionProbeMLXMemorySnapshot {
        let snapshot = Memory.snapshot()
        return CompressedAttentionProbeMLXMemorySnapshot(
            activeBytes: snapshot.activeMemory,
            cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory)
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
        eval(queries, keys, values)
        let sourceKVTensorSHA256 = tensorSHA256([keys, values])
        let queryTensorSHA256 = tensorSHA256([queries])
        let scale = Float(1 / sqrt(Double(plan.headDimension)))
        let mask = mlxMask(
            plan.mask,
            queryTokens: plan.queryTokens,
            contextTokens: plan.contextTokens,
            dtype: dtype)

        let output: MLXArray
        let oracle: MLXArray
        let attentionSeconds: Double
        let payloadBytes: Int
        let scaleBytes: Int
        let biasBytes: Int
        let controlBytes: Int
        let alignmentPaddingBytes: Int
        let fp16ResidentBytes: Int
        let persistentBytes: Int
        let materializationWorkspaceBytes: Int
        let packedKVTensorSHA256: String

        switch (plan.operation, plan.layout) {
        case (.fp16SDPA, .fp16):
            let startedAt = ProcessInfo.processInfo.systemUptime
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask.mode)
            eval(output)
            attentionSeconds = ProcessInfo.processInfo.systemUptime
                - startedAt
            oracle = referenceAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                additiveMask: mask.additive)
            payloadBytes = 0
            scaleBytes = 0
            biasBytes = 0
            controlBytes = 0
            alignmentPaddingBytes = 0
            fp16ResidentBytes = try checkedSum([
                keys.nbytes, values.nbytes,
            ])
            persistentBytes = fp16ResidentBytes
            materializationWorkspaceBytes = 0
            packedKVTensorSHA256 = sourceKVTensorSHA256

        case let (
            .swiftLMQuantizedAttention,
            .affine(keyBits, valueBits, keyGroupSize, valueGroupSize)):
            let pair = try affinePair(
                keys: keys, values: values,
                keyBits: keyBits, valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
            let startedAt = ProcessInfo.processInfo.systemUptime
            output = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: pair.keys,
                quantizedValues: pair.values,
                scale: scale,
                mask: mask.mode,
                groupSize: keyGroupSize,
                bits: keyBits,
                mode: .affine)
            eval(output)
            attentionSeconds = ProcessInfo.processInfo.systemUptime
                - startedAt
            let materialized = dequantize(pair: pair, dtype: dtype)
            oracle = referenceAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale,
                additiveMask: mask.additive)
            let components = try affinePersistentComponents(pair)
            payloadBytes = components.payload
            scaleBytes = components.scales
            biasBytes = components.biases
            controlBytes = 0
            alignmentPaddingBytes = 0
            fp16ResidentBytes = 0
            persistentBytes = components.total
            materializationWorkspaceBytes = 0
            packedKVTensorSHA256 = tensorSHA256(
                affineArrays(pair))

        case let (
            .materializeThenSDPA,
            .affine(keyBits, valueBits, keyGroupSize, valueGroupSize)):
            let pair = try affinePair(
                keys: keys, values: values,
                keyBits: keyBits, valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
            let startedAt = ProcessInfo.processInfo.systemUptime
            let materialized = dequantize(pair: pair, dtype: dtype)
            output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale, mask: mask.mode)
            eval(output)
            attentionSeconds = ProcessInfo.processInfo.systemUptime
                - startedAt
            oracle = referenceAttention(
                queries: queries,
                keys: materialized.keys,
                values: materialized.values,
                scale: scale,
                additiveMask: mask.additive)
            let components = try affinePersistentComponents(pair)
            payloadBytes = components.payload
            scaleBytes = components.scales
            biasBytes = components.biases
            controlBytes = 0
            alignmentPaddingBytes = 0
            fp16ResidentBytes = 0
            persistentBytes = components.total
            materializationWorkspaceBytes = try checkedSum([
                materialized.keys.nbytes,
                materialized.values.nbytes,
            ])
            packedKVTensorSHA256 = tensorSHA256(
                affineArrays(pair))

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
        let relativeError = difference / MLX.maximum(
            abs(oracle.asType(.float32)), MLXArray(Float(1e-6)))
        let maxRelativeError = max(relativeError).item(Float.self)
        let meanError = mean(difference).item(Float.self)
        let tolerance: (relative: Double, absolute: Double) =
            plan.operation == .fp16SDPA
            ? (
                CompressedAttentionProbeEvidence.fp16RTolerance,
                CompressedAttentionProbeEvidence.fp16ATolerance)
            : (
                CompressedAttentionProbeEvidence.packedRTolerance,
                CompressedAttentionProbeEvidence.packedATolerance)
        let toleranceRatio = difference / (
            MLXArray(Float(tolerance.absolute))
                + MLXArray(Float(tolerance.relative))
                    * abs(oracle.asType(.float32)))
        let maximumToleranceRatio = max(toleranceRatio).item(Float.self)
        let equivalent = output.allClose(
            oracle,
            rtol: tolerance.relative,
            atol: tolerance.absolute).item(Bool.self)
        let outputTop1Index = argMax(output).item(Int.self)
        let oracleTop1Index = argMax(oracle).item(Int.self)
        let top1Matches = outputTop1Index == oracleTop1Index
        let outputTensorSHA256 = tensorSHA256([output])

        return CompressedAttentionProbeNumericResult(
            outputShape: output.shape,
            valuesFinite: finite,
            structuralEquivalent: equivalent,
            top1Matches: top1Matches,
            maxAbsoluteError: Double(maxError),
            maxRelativeError: Double(maxRelativeError),
            maximumToleranceRatio: Double(maximumToleranceRatio),
            meanAbsoluteError: Double(meanError),
            outputTop1Index: outputTop1Index,
            oracleTop1Index: oracleTop1Index,
            attentionSeconds: attentionSeconds,
            payloadBytes: payloadBytes,
            scaleBytes: scaleBytes,
            biasBytes: biasBytes,
            controlBytes: controlBytes,
            alignmentPaddingBytes: alignmentPaddingBytes,
            fp16ResidentBytes: fp16ResidentBytes,
            persistentBytes: persistentBytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            sourceKVTensorSHA256: sourceKVTensorSHA256,
            packedKVTensorSHA256: packedKVTensorSHA256,
            queryTensorSHA256: queryTensorSHA256,
            outputTensorSHA256: outputTensorSHA256)
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

    func runTensorHashCoverageFixture()
        -> CompressedAttentionProbeHashCoverageResult
    {
        let base = MLXArray((0 ..< 8).map(Float.init))
            .asType(.float16)
            .reshaped([1, 1, 8, 1])
        var middleValues = (0 ..< 8).map(Float.init)
        middleValues[4] = 99
        let middleMutation = MLXArray(middleValues)
            .asType(.float16)
            .reshaped([1, 1, 8, 1])
        // These adjacent uint32 values collapse to the same float32 value. Native-byte hashing
        // must still distinguish them or packed KV authentication is lossy.
        let packedA = MLXArray([UInt32(16_777_216)])
        let packedB = MLXArray([UInt32(16_777_217)])
        return CompressedAttentionProbeHashCoverageResult(
            baseTensorSHA256: tensorSHA256([base]),
            middleMutationSHA256: tensorSHA256([middleMutation]),
            packedWordASHA256: tensorSHA256([packedA]),
            packedWordBSHA256: tensorSHA256([packedB]))
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
        // Match pinned MLX SDPA's scalar semantics: its fallback constructs the scale in the
        // query dtype before multiplication. A bare Swift `Float` promotes fp16 queries to
        // float32 and silently turns this into a different oracle.
        let typedScale = MLXArray(scale).asType(queries.dtype)
        var scaledQueries = queries * typedScale
        var oracleKeys = keys
        var oracleValues = values
        if repeatCount > 1 {
            // Mirror pinned MLX's GQA fallback layout exactly. Materially repeating K/V changes
            // the fp16 matmul route enough to exceed the backend's own tolerance at some shapes.
            scaledQueries = unflatten(
                scaledQueries,
                axis: 1,
                shape: [keys.dim(1), repeatCount])
            oracleKeys = keys.expandedDimensions(axis: 2)
            oracleValues = values.expandedDimensions(axis: 2)
        }
        var logits = matmul(
            scaledQueries,
            oracleKeys.swappedAxes(-1, -2))
        if let additiveMask {
            logits = logits + additiveMask.asType(logits.dtype)
        }
        let weights = softmax(logits, axis: -1, precise: true)
        let output = matmul(weights, oracleValues)
        return repeatCount > 1
            ? flatten(output, startAxis: 1, endAxis: 2)
            : output
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

    private func affineArrays(_ pair: AffinePair) -> [MLXArray] {
        [
            pair.keys.weights,
            pair.keys.scales,
            pair.keys.biases,
            pair.values.weights,
            pair.values.scales,
            pair.values.biases,
        ].compactMap { $0 }
    }

    private func affinePersistentComponents(
        _ pair: AffinePair
    ) throws -> (payload: Int, scales: Int, biases: Int, total: Int) {
        let payload = try checkedSum([
            pair.keys.weights.nbytes,
            pair.values.weights.nbytes,
        ])
        let scales = try checkedSum([
            pair.keys.scales.nbytes,
            pair.values.scales.nbytes,
        ])
        let biases = try checkedSum([
            pair.keys.biases?.nbytes ?? 0,
            pair.values.biases?.nbytes ?? 0,
        ])
        return (
            payload: payload,
            scales: scales,
            biases: biases,
            total: try checkedSum([payload, scales, biases]))
    }

    /// Exact SHA-256 over every tensor's geometry, dtype, and native contiguous bytes. Hashing is
    /// outside the timed attention interval and uses no-copy access for contiguous MLX storage,
    /// so 128K evidence authenticates the middle of each tensor without retaining a host clone.
    private func tensorSHA256(_ arrays: [MLXArray]) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(
            "fastmlx-compressed-attention-tensor-v1\n".utf8))
        for (index, array) in arrays.enumerated() {
            let bytes = array.asData(access: .noCopyIfContiguous).data
            hasher.update(data: Data(
                "index=\(index);shape=\(array.shape);dtype=\(array.dtype);bytes=\(bytes.count)\n"
                    .utf8))
            hasher.update(data: bytes)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
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
