import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Recoverable failures at scalar-affine/batched-affine membership and append boundaries.
public enum BatchedAffineKVCacheError: Error, Equatable {
    case emptyBatch
    case lengthCount(expected: Int, actual: Int)
    case lengthMismatch(slot: Int, expected: Int, actual: Int)
    case uninitializedSlot(Int)
    case invalidLength(slot: Int, length: Int, capacity: Int)
    case incompatibleConfiguration(slot: Int)
    case incompatibleAttentionMode(slot: Int)
    case incompatibleShape(slot: Int)
    case incompatibleDType(slot: Int)
    case invalidAdditionalTokens(Int)
    case insufficientCapacity(required: Int, capacity: Int)
    case emptySelection
    case invalidSelection([Int])
    case invalidSlot(index: Int, batchSize: Int)
    case invalidAppendGeometry
    case nonFiniteInput
    case corruptedMetadata
}

/// End-aligned packed affine KV cache for a ragged shared decode batch.
///
/// Logical offsets drive per-row RoPE. A separate graph-resident physical end drives shared
/// packed writes and deliberately survives filtering the longest row:
///
/// ```text
/// logical lengths       32, 48, 24
/// physical written end  48
/// packed rows            [16 pad | 32 live], [48 live], [24 pad | 24 live]
/// ```
///
/// Removing the 48-token row keeps the physical end at 48, so the next shared append begins at
/// 48 rather than overwriting either survivor at logical column 32 or 24. Every MLX array remains
/// actor-confined; the class intentionally does not conform to `Sendable`.
public final class BatchedAffineKVCache:
    AttentionKVCacheProtocol, Updatable, BatchPositionedKVCache
{
    public private(set) var capacity: Int
    public let configuration: AffineKVCacheConfiguration
    public let attentionMode: AffineKVAttentionMode

    var kPayload: MLXArray
    var kScales: MLXArray
    var kBiases: MLXArray
    var vPayload: MLXArray
    var vScales: MLXArray
    var vBiases: MLXArray
    public private(set) var batchOffset: MLXArray
    var physicalEndArr: MLXArray

    private let keyDimension: Int
    private let valueDimension: Int
    private let keyOutputDType: DType
    private let valueOutputDType: DType
    private var materializationWorkspaceBytes: Int?
    private var attentionWorkspaceBytes: Int?
    private var attentionOperation: AffineKVAttentionOperation?

    /// Host mirror used by uncompiled orchestration only. Graph paths read `batchOffset`.
    public private(set) var offset: Int

    public var batchSize: Int { batchOffset.dim(0) }
    public var maxSize: Int? { nil }

    var physicalWrittenEnd: Int {
        Int(physicalEndArr.item(Int32.self))
    }

    var leftPadding: [Int] {
        let physicalEnd = physicalWrittenEnd
        return batchOffset.asArray(Int32.self).map { physicalEnd - Int($0) }
    }

    private init(
        capacity: Int,
        configuration: AffineKVCacheConfiguration,
        attentionMode: AffineKVAttentionMode,
        kPayload: MLXArray,
        kScales: MLXArray,
        kBiases: MLXArray,
        vPayload: MLXArray,
        vScales: MLXArray,
        vBiases: MLXArray,
        logicalOffsets: [Int],
        physicalWrittenEnd: Int,
        keyDimension: Int,
        valueDimension: Int,
        keyOutputDType: DType,
        valueOutputDType: DType,
        materializationWorkspaceBytes: Int?,
        attentionWorkspaceBytes: Int?,
        attentionOperation: AffineKVAttentionOperation?
    ) {
        self.capacity = capacity
        self.configuration = configuration
        self.attentionMode = attentionMode
        self.kPayload = kPayload
        self.kScales = kScales
        self.kBiases = kBiases
        self.vPayload = vPayload
        self.vScales = vScales
        self.vBiases = vBiases
        self.batchOffset = MLXArray(logicalOffsets.map(Int32.init))
        self.physicalEndArr = MLXArray([Int32(physicalWrittenEnd)])
        self.keyDimension = keyDimension
        self.valueDimension = valueDimension
        self.keyOutputDType = keyOutputDType
        self.valueOutputDType = valueOutputDType
        self.materializationWorkspaceBytes = materializationWorkspaceBytes
        self.attentionWorkspaceBytes = attentionWorkspaceBytes
        self.attentionOperation = attentionOperation
        self.offset = logicalOffsets.max() ?? 0
    }

    /// Merge initialized scalar affine rows without mutating them.
    ///
    /// Optional lengths are a scheduler assertion and must equal authoritative graph offsets.
    public static func merging(
        _ caches: [AffineKVCache],
        lengths explicitLengths: [Int]? = nil
    ) throws -> BatchedAffineKVCache {
        guard !caches.isEmpty else {
            throw BatchedAffineKVCacheError.emptyBatch
        }
        if let explicitLengths, explicitLengths.count != caches.count {
            throw BatchedAffineKVCacheError.lengthCount(
                expected: caches.count,
                actual: explicitLengths.count)
        }

        var states: [AffineKVCache.PackedBatchState] = []
        states.reserveCapacity(caches.count)
        for (slot, cache) in caches.enumerated() {
            guard let state = cache.packedBatchState() else {
                throw BatchedAffineKVCacheError.uninitializedSlot(slot)
            }
            states.append(state)
        }
        let authoritativeLengths = states.map(\.logicalOffset)
        if let explicitLengths {
            for slot in states.indices
            where explicitLengths[slot] != authoritativeLengths[slot] {
                throw BatchedAffineKVCacheError.lengthMismatch(
                    slot: slot,
                    expected: authoritativeLengths[slot],
                    actual: explicitLengths[slot])
            }
        }

        let first = states[0]
        for (slot, state) in states.enumerated() {
            guard state.logicalOffset >= 0, state.logicalOffset <= state.capacity else {
                throw BatchedAffineKVCacheError.invalidLength(
                    slot: slot,
                    length: state.logicalOffset,
                    capacity: state.capacity)
            }
            guard state.configuration == first.configuration else {
                throw BatchedAffineKVCacheError.incompatibleConfiguration(slot: slot)
            }
            guard state.attentionMode == first.attentionMode else {
                throw BatchedAffineKVCacheError.incompatibleAttentionMode(slot: slot)
            }
            guard state.keyDimension == first.keyDimension,
                state.valueDimension == first.valueDimension,
                compatibleShapes(state, first: first)
            else {
                throw BatchedAffineKVCacheError.incompatibleShape(slot: slot)
            }
            guard state.keyOutputDType == first.keyOutputDType,
                state.valueOutputDType == first.valueOutputDType,
                compatibleDTypes(state, first: first)
            else {
                throw BatchedAffineKVCacheError.incompatibleDType(slot: slot)
            }
        }

        let layout: CompressedKVBatchLayout
        do {
            layout = try CompressedKVBatchLayout(
                logicalOffsets: authoritativeLengths,
                capacities: states.map(\.capacity))
        } catch {
            throw BatchedAffineKVCacheError.corruptedMetadata
        }

        func merged(_ array: (AffineKVCache.PackedBatchState) -> MLXArray) -> MLXArray {
            concatenated(
                states.enumerated().map { slot, state in
                    endAlignedRow(
                        array(state),
                        validLength: state.logicalOffset,
                        leftPadding: layout.leftPadding[slot],
                        capacity: layout.allocationCapacity)
                },
                axis: 0)
        }
        func maximum(_ values: [Int?]) -> Int? {
            let present = values.compactMap { $0 }
            return present.count == values.count ? present.max() : nil
        }
        let operations = states.compactMap(\.attentionOperation)
        let operation = operations.count == states.count && Set(operations).count == 1
            ? operations[0] : nil

        return BatchedAffineKVCache(
            capacity: layout.allocationCapacity,
            configuration: first.configuration,
            attentionMode: first.attentionMode,
            kPayload: merged(\.kPayload),
            kScales: merged(\.kScales),
            kBiases: merged(\.kBiases),
            vPayload: merged(\.vPayload),
            vScales: merged(\.vScales),
            vBiases: merged(\.vBiases),
            logicalOffsets: layout.logicalOffsets,
            physicalWrittenEnd: layout.physicalWrittenEnd,
            keyDimension: first.keyDimension,
            valueDimension: first.valueDimension,
            keyOutputDType: first.keyOutputDType,
            valueOutputDType: first.valueOutputDType,
            materializationWorkspaceBytes: maximum(
                states.map(\.materializationWorkspaceBytes)),
            attentionWorkspaceBytes: maximum(states.map(\.attentionWorkspaceBytes)),
            attentionOperation: operation)
    }

    private static func compatibleShapes(
        _ state: AffineKVCache.PackedBatchState,
        first: AffineKVCache.PackedBatchState
    ) -> Bool {
        let arrays = [
            state.kPayload, state.kScales, state.kBiases,
            state.vPayload, state.vScales, state.vBiases,
        ]
        let firstArrays = [
            first.kPayload, first.kScales, first.kBiases,
            first.vPayload, first.vScales, first.vBiases,
        ]
        return zip(arrays, firstArrays).allSatisfy { array, reference in
            array.ndim == 4 && array.dim(0) == 1
                && array.dim(1) == reference.dim(1)
                && array.dim(2) == state.capacity
                && array.dim(3) == reference.dim(3)
        }
    }

    private static func compatibleDTypes(
        _ state: AffineKVCache.PackedBatchState,
        first: AffineKVCache.PackedBatchState
    ) -> Bool {
        let arrays = [
            state.kPayload, state.kScales, state.kBiases,
            state.vPayload, state.vScales, state.vBiases,
        ]
        let firstArrays = [
            first.kPayload, first.kScales, first.kBiases,
            first.vPayload, first.vScales, first.vBiases,
        ]
        return zip(arrays, firstArrays).allSatisfy { $0.dtype == $1.dtype }
    }

    private static func endAlignedRow(
        _ source: MLXArray,
        validLength: Int,
        leftPadding: Int,
        capacity: Int
    ) -> MLXArray {
        var pieces: [MLXArray] = []
        if leftPadding > 0 {
            pieces.append(
                MLXArray.zeros(
                    [1, source.dim(1), leftPadding, source.dim(3)],
                    dtype: source.dtype))
        }
        if validLength > 0 {
            pieces.append(source[0..., 0..., 0 ..< validLength, 0...])
        }
        let tail = capacity - leftPadding - validLength
        if tail > 0 {
            pieces.append(
                MLXArray.zeros(
                    [1, source.dim(1), tail, source.dim(3)],
                    dtype: source.dtype))
        }
        return concatenated(pieces, axis: 2)
    }

    public func innerState() -> [MLXArray] {
        [
            kPayload, kScales, kBiases,
            vPayload, vScales, vBiases,
            batchOffset, physicalEndArr,
        ]
    }

    /// Recoverable preflight for actor orchestration. Compiled replay uses the same already-proven
    /// fixed geometry without synchronizing graph state inside the traced closure.
    public func validateAppend(keys: MLXArray, values: MLXArray) throws {
        guard keys.ndim == 4, values.shape == keys.shape,
            keys.dim(0) == batchSize,
            keys.dim(1) == kPayload.dim(1),
            keys.dim(3) == keyDimension,
            values.dim(3) == valueDimension,
            keys.dtype == keyOutputDType,
            values.dtype == valueOutputDType,
            keys.dim(2) > 0
        else {
            throw BatchedAffineKVCacheError.invalidAppendGeometry
        }
        try requireCapacity(for: keys.dim(2))
        guard isFinite(keys).all().item(Bool.self),
            isFinite(values).all().item(Bool.self)
        else {
            throw BatchedAffineKVCacheError.nonFiniteInput
        }
    }

    public func requireCapacity(for additionalTokens: Int) throws {
        guard additionalTokens >= 0 else {
            throw BatchedAffineKVCacheError.invalidAdditionalTokens(additionalTokens)
        }
        let (required, overflow) = physicalWrittenEnd.addingReportingOverflow(additionalTokens)
        guard !overflow, required <= Int(Int32.max) else {
            throw BatchedAffineKVCacheError.corruptedMetadata
        }
        guard required <= capacity else {
            throw BatchedAffineKVCacheError.insufficientCapacity(
                required: required,
                capacity: capacity)
        }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        store(keys: keys, values: values)
        let materialized = materializedStoredKV()
        let (workspace, overflow) = materialized.0.nbytes.addingReportingOverflow(
            materialized.1.nbytes)
        precondition(!overflow, "affine batch materialization workspace overflow")
        materializationWorkspaceBytes = workspace
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
        precondition(scale.isFinite && scale > 0, "attention scale must be finite and positive")
        switch attentionMode {
        case .materialize:
            let (cachedKeys, cachedValues) = update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask)
        case .splitQuantizedMM:
            store(keys: keys, values: values)
            materializationWorkspaceBytes = 0
            attentionOperation = .splitQuantizedMM
            return packedAttention(queries: queries, scale: scale, mask: mask)
        }
    }

    private struct StoredCode {
        let payload: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private func store(keys: MLXArray, values: MLXArray) {
        precondition(keys.ndim == 4 && values.shape == keys.shape, "K/V geometry must match")
        precondition(keys.dim(0) == batchSize, "batched affine update has wrong batch size")
        precondition(
            keys.dim(1) == kPayload.dim(1) && keys.dim(3) == keyDimension
                && values.dim(3) == valueDimension,
            "batched affine update has incompatible KV geometry")
        precondition(
            keys.dtype == keyOutputDType && values.dtype == valueOutputDType,
            "batched affine update changed K/V dtype")
        let tokenCount = keys.dim(2)
        precondition(tokenCount > 0, "batched affine update must contain a token")
        precondition(tokenCount <= capacity, "batched affine update exceeds fixed capacity")

        let keyCode = encode(
            keys,
            bits: configuration.keyBits,
            groupSize: configuration.keyGroupSize)
        let valueCode = encode(
            values,
            bits: configuration.valueBits,
            groupSize: configuration.valueGroupSize)
        let positions = physicalEndArr + MLXArray(Int32(0) ..< Int32(tokenCount))
        let indices = broadcast(
            positions.reshaped([1, 1, tokenCount, 1]),
            to: [batchSize, 1, tokenCount, 1])
        kPayload = putAlong(kPayload, indices, values: keyCode.payload, axis: 2)
        kScales = putAlong(kScales, indices, values: keyCode.scales, axis: 2)
        kBiases = putAlong(kBiases, indices, values: keyCode.biases, axis: 2)
        vPayload = putAlong(vPayload, indices, values: valueCode.payload, axis: 2)
        vScales = putAlong(vScales, indices, values: valueCode.scales, axis: 2)
        vBiases = putAlong(vBiases, indices, values: valueCode.biases, axis: 2)
        physicalEndArr = physicalEndArr + MLXArray([Int32(tokenCount)])
        batchOffset = batchOffset + Int32(tokenCount)
        offset += tokenCount
    }

    private func encode(_ input: MLXArray, bits: Int, groupSize: Int) -> StoredCode {
        let batch = input.dim(0)
        let heads = input.dim(1)
        let tokens = input.dim(2)
        let dimension = input.dim(3)
        let code = quantized(
            input.reshaped([-1, dimension]),
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
        guard let biases = code.biases else {
            preconditionFailure("native affine quantization did not return biases")
        }
        return StoredCode(
            payload: code.wq.reshaped([batch, heads, tokens, code.wq.dim(1)]),
            scales: code.scales.reshaped([batch, heads, tokens, code.scales.dim(1)]),
            biases: biases.reshaped([batch, heads, tokens, biases.dim(1)]))
    }

    func materializedStoredKV() -> (MLXArray, MLXArray) {
        (
            materialize(
                payload: kPayload,
                scales: kScales,
                biases: kBiases,
                dimension: keyDimension,
                bits: configuration.keyBits,
                groupSize: configuration.keyGroupSize,
                dtype: keyOutputDType),
            materialize(
                payload: vPayload,
                scales: vScales,
                biases: vBiases,
                dimension: valueDimension,
                bits: configuration.valueBits,
                groupSize: configuration.valueGroupSize,
                dtype: valueOutputDType))
    }

    private func materialize(
        payload: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        dimension: Int,
        bits: Int,
        groupSize: Int,
        dtype: DType
    ) -> MLXArray {
        dequantized(
            payload.reshaped([-1, payload.dim(3)]),
            scales: scales.reshaped([-1, scales.dim(3)]),
            biases: biases.reshaped([-1, biases.dim(3)]),
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: dtype
        ).reshaped([batchSize, payload.dim(1), capacity, dimension])
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by BatchedAffineKVCache")
        precondition(n > 0 && n <= capacity, "invalid batched affine mask width")
        let keyPositions = MLXArray(Int32(0) ..< Int32(capacity))
            .reshaped([1, 1, 1, capacity])
        let leftPadding = (physicalEndArr - batchOffset)
            .reshaped([batchSize, 1, 1, 1])
        let queryPositions = (
            physicalEndArr + MLXArray(Int32(0) ..< Int32(n))
        ).reshaped([1, 1, n, 1])
        return .array(
            (keyPositions .>= leftPadding)
                & (keyPositions .<= queryPositions))
    }

    private func packedAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        precondition(queries.ndim == 4, "queries must be rank 4")
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let kvHeads = kPayload.dim(1)
        precondition(
            queries.dim(0) == batchSize && kvHeads > 0
                && queryHeads.isMultiple(of: kvHeads),
            "query and packed-cache batch/head geometry is incompatible")
        precondition(
            queries.dim(3) == keyDimension,
            "query dimension does not match packed key dimension")

        let repeats = queryHeads / kvHeads
        var scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        var keyWeights = kPayload
        var keyScales = kScales
        var keyBiases: MLXArray? = kBiases
        var valueWeights = vPayload
        var valueScales = vScales
        var valueBiases: MLXArray? = vBiases
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped([
                batchSize, kvHeads, repeats, queryTokens, keyDimension,
            ])
            keyWeights = keyWeights.expandedDimensions(axis: -3)
            keyScales = keyScales.expandedDimensions(axis: -3)
            keyBiases = keyBiases?.expandedDimensions(axis: -3)
            valueWeights = valueWeights.expandedDimensions(axis: -3)
            valueScales = valueScales.expandedDimensions(axis: -3)
            valueBiases = valueBiases?.expandedDimensions(axis: -3)
        }

        var scores = quantizedMM(
            scaledQueries,
            keyWeights,
            scales: keyScales,
            biases: keyBiases,
            transpose: true,
            groupSize: configuration.keyGroupSize,
            bits: configuration.keyBits,
            mode: .affine)
        scores = apply(mask: mask, to: scores, groupedQueryRepeats: repeats)
        let weights = softmax(scores, axis: -1, precise: true)
        let (workspace, overflow) = scores.nbytes.addingReportingOverflow(weights.nbytes)
        precondition(!overflow, "split batch attention workspace overflow")
        attentionWorkspaceBytes = max(attentionWorkspaceBytes ?? 0, workspace)
        var output = quantizedMM(
            weights,
            valueWeights,
            scales: valueScales,
            biases: valueBiases,
            transpose: false,
            groupSize: configuration.valueGroupSize,
            bits: configuration.valueBits,
            mode: .affine)
        if repeats > 1 {
            output = output.reshaped([
                batchSize, queryHeads, queryTokens, valueDimension,
            ])
        }
        return output
    }

    private func apply(
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        to scores: MLXArray,
        groupedQueryRepeats: Int
    ) -> MLXArray {
        switch mask {
        case .none:
            return maskOutsideWrittenRows(in: scores)
        case .causal:
            let queryTokens = scores.dim(-2)
            let keyTokens = scores.dim(-1)
            var queryShape = Array(repeating: 1, count: scores.ndim)
            queryShape[scores.ndim - 2] = queryTokens
            var keyShape = Array(repeating: 1, count: scores.ndim)
            keyShape[scores.ndim - 1] = keyTokens
            var batchShape = Array(repeating: 1, count: scores.ndim)
            batchShape[0] = batchSize
            let queryPositions = (
                physicalEndArr - Int32(queryTokens)
                    + MLXArray(Int32(0) ..< Int32(queryTokens))
            ).reshaped(queryShape)
            let keyPositions = MLXArray(Int32(0) ..< Int32(keyTokens))
                .reshaped(keyShape)
            let leftPadding = (physicalEndArr - batchOffset)
                .reshaped(batchShape)
            return MLX.where(
                (keyPositions .>= leftPadding) & (keyPositions .<= queryPositions),
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        case .array(let array):
            return maskOutsideWrittenRows(
                in: apply(
                    array: array,
                    to: scores,
                    groupedQueryRepeats: groupedQueryRepeats))
        case .arrays(let arrays):
            precondition(arrays.count <= 1, "only one attention mask array is supported")
            guard let array = arrays.first else {
                return maskOutsideWrittenRows(in: scores)
            }
            return maskOutsideWrittenRows(
                in: apply(
                    array: array,
                    to: scores,
                    groupedQueryRepeats: groupedQueryRepeats))
        }
    }

    private func maskOutsideWrittenRows(in scores: MLXArray) -> MLXArray {
        var keyShape = Array(repeating: 1, count: scores.ndim)
        keyShape[scores.ndim - 1] = scores.dim(-1)
        var batchShape = Array(repeating: 1, count: scores.ndim)
        batchShape[0] = batchSize
        let keyPositions = MLXArray(Int32(0) ..< Int32(scores.dim(-1)))
            .reshaped(keyShape)
        let leftPadding = (physicalEndArr - batchOffset)
            .reshaped(batchShape)
        let physicalEnd = physicalEndArr.reshaped(
            Array(repeating: 1, count: scores.ndim))
        return MLX.where(
            (keyPositions .>= leftPadding) & (keyPositions .< physicalEnd),
            scores,
            MLXArray(-Float.infinity).asType(scores.dtype))
    }

    private func apply(
        array: MLXArray,
        to scores: MLXArray,
        groupedQueryRepeats: Int
    ) -> MLXArray {
        var array = array
        if groupedQueryRepeats > 1 && array.ndim == 4 {
            array = array.expandedDimensions(axis: 2)
        }
        let validRanks = groupedQueryRepeats > 1 ? [2, 5] : [2, 4]
        precondition(validRanks.contains(array.ndim), "unsupported attention-mask rank")
        if array.dtype == .bool {
            return MLX.where(
                array,
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        }
        return scores + array.asType(scores.dtype)
    }

    public func grow(by chunk: Int) {
        precondition(chunk > 0, "cache growth must be positive")
        func grown(_ buffer: MLXArray) -> MLXArray {
            concatenated([
                buffer,
                MLXArray.zeros(
                    [batchSize, buffer.dim(1), chunk, buffer.dim(3)],
                    dtype: buffer.dtype),
            ], axis: 2)
        }
        kPayload = grown(kPayload)
        kScales = grown(kScales)
        kBiases = grown(kBiases)
        vPayload = grown(vPayload)
        vScales = grown(vScales)
        vBiases = grown(vBiases)
        capacity += chunk
    }

    public func filter(keeping indices: [Int]) throws {
        guard !indices.isEmpty else {
            throw BatchedAffineKVCacheError.emptySelection
        }
        guard Set(indices).count == indices.count,
            indices.allSatisfy({ $0 >= 0 && $0 < batchSize })
        else {
            throw BatchedAffineKVCacheError.invalidSelection(indices)
        }
        let selection = MLXArray(indices.map(Int32.init))
        kPayload = kPayload[selection]
        kScales = kScales[selection]
        kBiases = kBiases[selection]
        vPayload = vPayload[selection]
        vScales = vScales[selection]
        vBiases = vBiases[selection]
        batchOffset = batchOffset[selection]
        offset = batchOffset.asArray(Int32.self).map(Int.init).max() ?? 0
    }

    public func extract(slot: Int) throws -> AffineKVCache {
        guard slot >= 0, slot < batchSize else {
            throw BatchedAffineKVCacheError.invalidSlot(index: slot, batchSize: batchSize)
        }
        let logical = Int(batchOffset[slot].item(Int32.self))
        let physicalEnd = physicalWrittenEnd
        guard logical >= 0, logical <= physicalEnd, physicalEnd <= capacity else {
            throw BatchedAffineKVCacheError.corruptedMetadata
        }
        let validRange = (physicalEnd - logical) ..< physicalEnd
        func scalarRow(_ source: MLXArray) -> MLXArray {
            var pieces: [MLXArray] = []
            if logical > 0 {
                pieces.append(source[slot ..< (slot + 1), 0..., validRange, 0...])
            }
            if capacity > logical {
                pieces.append(
                    MLXArray.zeros(
                        [1, source.dim(1), capacity - logical, source.dim(3)],
                        dtype: source.dtype))
            }
            return concatenated(pieces, axis: 2)
        }
        return AffineKVCache.restoringPackedBatchState(
            AffineKVCache.PackedBatchState(
                capacity: capacity,
                configuration: configuration,
                attentionMode: attentionMode,
                logicalOffset: logical,
                kPayload: scalarRow(kPayload),
                kScales: scalarRow(kScales),
                kBiases: scalarRow(kBiases),
                vPayload: scalarRow(vPayload),
                vScales: scalarRow(vScales),
                vBiases: scalarRow(vBiases),
                keyDimension: keyDimension,
                valueDimension: valueDimension,
                keyOutputDType: keyOutputDType,
                valueOutputDType: valueOutputDType,
                materializationWorkspaceBytes: materializationWorkspaceBytes,
                attentionWorkspaceBytes: attentionWorkspaceBytes,
                attentionOperation: attentionOperation))
    }

    // MARK: - KVCache protocol surface unused by continuous-batch orchestration

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("BatchedAffineKVCache state restore is not supported") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("BatchedAffineKVCache.copy() is not supported")
    }
}
