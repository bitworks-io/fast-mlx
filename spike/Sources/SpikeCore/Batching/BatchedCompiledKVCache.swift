import Foundation
import MLX
import MLXLMCommon

/// Validation failures at scalar-cache/batched-cache membership boundaries.
///
/// These are recoverable scheduler/driver errors, not preconditions: malformed or
/// unsupported batches fail closed before any request cache is mutated.
public enum BatchedCompiledKVCacheError: Error, Equatable {
    case emptyBatch
    case lengthCount(expected: Int, actual: Int)
    case lengthMismatch(slot: Int, expected: Int, actual: Int)
    case uninitializedSlot(Int)
    case invalidLength(slot: Int, length: Int, capacity: Int)
    case invalidAdditionalTokens(Int)
    case insufficientCapacity(required: Int, capacity: Int)
    case incompatibleShape(slot: Int, expected: [Int], actual: [Int])
    case incompatibleDType(slot: Int)
    case emptySelection
    case invalidSelection([Int])
    case invalidSlot(index: Int, batchSize: Int)
    case corruptedMetadata(slot: Int)
}

/// Fixed-capacity dense KV cache for a shared one-token decode forward.
///
/// Request histories can have different logical lengths. Because every row already has
/// the same fixed capacity, each retains the scalar cache's right-padded layout and
/// `batchOffset` is both its next physical write column and true logical RoPE position:
///
/// ```text
/// logical lengths   3, 1
/// physical rows     [a b c · ·], [x · · · ·]
/// next write column        ^       ^
/// RoPE next offset  3, 1
/// ```
///
/// The model snapshots `ropeOffset` before `update`. The same per-row array drives RoPE,
/// scatter positions, and prefix masks, preserving the scalar attention reduction layout
/// while rejecting the fixed-capacity tail.
///
/// Like `CompiledKVCache`, all per-step mutable state lives in `MLXArray`s so the
/// cache is legal as `MLX.compile` input/output state. Membership changes are performed
/// outside compiled calls and require the driver to create/retrace its batch-shape step.
public final class BatchedCompiledKVCache: KVCache, Updatable, BatchPositionedKVCache {
    public private(set) var capacity: Int

    var keysBuf: MLXArray?
    var valuesBuf: MLXArray?
    public private(set) var batchOffset: MLXArray

    /// Host-side mirror of the maximum logical length. As with `CompiledKVCache`, this
    /// is current only for uncompiled calls; transition code reads `batchOffset`.
    public private(set) var offset: Int

    public var batchSize: Int { batchOffset.dim(0) }
    public var maxSize: Int? { nil }

    private init(
        capacity: Int,
        keys: MLXArray,
        values: MLXArray,
        logicalOffsets: [Int]
    ) {
        self.capacity = capacity
        self.keysBuf = keys
        self.valuesBuf = values
        self.batchOffset = MLXArray(logicalOffsets.map(Int32.init))
        self.offset = logicalOffsets.max() ?? 0
    }

    /// Merge initialized scalar caches without modifying them.
    ///
    /// Optional `lengths` are an assertion from a driver that already tracks committed
    /// positions. They must equal each scalar cache's authoritative in-graph offset.
    public static func merging(
        _ caches: [CompiledKVCache], lengths explicitLengths: [Int]? = nil
    ) throws -> BatchedCompiledKVCache {
        guard !caches.isEmpty else {
            throw BatchedCompiledKVCacheError.emptyBatch
        }
        if let explicitLengths, explicitLengths.count != caches.count {
            throw BatchedCompiledKVCacheError.lengthCount(
                expected: caches.count, actual: explicitLengths.count)
        }

        let authoritativeLengths = caches.map {
            Int($0.offsetArr.item(Int32.self))
        }
        if let explicitLengths {
            for slot in caches.indices where explicitLengths[slot] != authoritativeLengths[slot] {
                throw BatchedCompiledKVCacheError.lengthMismatch(
                    slot: slot,
                    expected: authoritativeLengths[slot],
                    actual: explicitLengths[slot])
            }
        }
        let lengths = authoritativeLengths
        let capacity = caches.map(\.capacity).max() ?? 0

        guard let firstKeys = caches[0].keysBuf, let firstValues = caches[0].valuesBuf else {
            throw BatchedCompiledKVCacheError.uninitializedSlot(0)
        }
        guard firstKeys.shape.count == 4, firstValues.shape == firstKeys.shape,
            firstKeys.dim(0) == 1, firstKeys.dim(2) == caches[0].capacity
        else {
            throw BatchedCompiledKVCacheError.incompatibleShape(
                slot: 0,
                expected: [1, firstKeys.dim(1), caches[0].capacity, firstKeys.dim(3)],
                actual: firstKeys.shape)
        }

        var keyRows: [MLXArray] = []
        var valueRows: [MLXArray] = []
        keyRows.reserveCapacity(caches.count)
        valueRows.reserveCapacity(caches.count)

        for (slot, cache) in caches.enumerated() {
            guard let keys = cache.keysBuf, let values = cache.valuesBuf else {
                throw BatchedCompiledKVCacheError.uninitializedSlot(slot)
            }
            let expectedShape = [1, firstKeys.dim(1), cache.capacity, firstKeys.dim(3)]
            guard keys.shape == expectedShape, values.shape == expectedShape else {
                throw BatchedCompiledKVCacheError.incompatibleShape(
                    slot: slot, expected: expectedShape, actual: keys.shape)
            }
            guard keys.dtype == firstKeys.dtype, values.dtype == firstValues.dtype else {
                throw BatchedCompiledKVCacheError.incompatibleDType(slot: slot)
            }

            let length = lengths[slot]
            guard length >= 0, length <= cache.capacity else {
                throw BatchedCompiledKVCacheError.invalidLength(
                    slot: slot, length: length, capacity: cache.capacity)
            }
            keyRows.append(
                paddedRow(
                    keys, validLength: length, capacity: capacity))
            valueRows.append(
                paddedRow(
                    values, validLength: length, capacity: capacity))
        }

        return BatchedCompiledKVCache(
            capacity: capacity,
            keys: concatenated(keyRows, axis: 0),
            values: concatenated(valueRows, axis: 0),
            logicalOffsets: lengths)
    }

    private static func paddedRow(
        _ source: MLXArray,
        validLength: Int,
        capacity: Int
    ) -> MLXArray {
        let rowShape = [1, source.dim(1), 0, source.dim(3)]
        var pieces: [MLXArray] = []
        if validLength > 0 {
            pieces.append(source[0..., 0..., 0 ..< validLength, 0...])
        }
        let tail = capacity - validLength
        if tail > 0 {
            var shape = rowShape
            shape[2] = tail
            pieces.append(MLXArray.zeros(shape, dtype: source.dtype))
        }
        return concatenated(pieces, axis: 2)
    }

    public func innerState() -> [MLXArray] {
        [keysBuf, valuesBuf].compactMap { $0 }
            + [batchOffset]
    }

    /// Validate headroom from authoritative in-graph offsets between compiled calls.
    /// Production drivers may avoid this synchronous read by proving the same bound from
    /// their actor-confined per-request lengths, but must do so before invoking `update`.
    public func requireCapacity(for additionalTokens: Int) throws {
        guard additionalTokens >= 0 else {
            throw BatchedCompiledKVCacheError.invalidAdditionalTokens(additionalTokens)
        }
        let required = (batchOffset.asArray(Int32.self).map(Int.init).max() ?? 0)
            + additionalTokens
        guard required <= capacity else {
            throw BatchedCompiledKVCacheError.insufficientCapacity(
                required: required, capacity: capacity)
        }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        guard let currentKeys = keysBuf, let currentValues = valuesBuf else {
            preconditionFailure("BatchedCompiledKVCache update before initialization")
        }
        let n = keys.dim(2)
        precondition(n > 0, "BatchedCompiledKVCache requires a non-empty update")
        precondition(keys.shape == values.shape, "batched key/value shapes must match")
        precondition(keys.shape.count == 4, "batched key/value arrays must be rank 4")
        precondition(keys.dim(0) == batchSize, "batched update has wrong batch size")
        precondition(
            keys.dim(1) == currentKeys.dim(1) && keys.dim(3) == currentKeys.dim(3),
            "batched update has incompatible KV dimensions")
        // `batchOffset` is a graph value while this method is traced, so a host precondition
        // here would be stale on replay. The actor driver must prove headroom before the call
        // (via tracked lengths, or `requireCapacity(for:)` outside the compiled closure).
        precondition(n <= capacity, "batched update width exceeds cache capacity")

        let positions = batchOffset[0..., .newAxis]
            + MLXArray(Int32(0) ..< Int32(n))[.newAxis]
        let indices = positions.reshaped([batchSize, 1, n, 1])
        keysBuf = putAlong(currentKeys, indices, values: keys, axis: 2)
        valuesBuf = putAlong(currentValues, indices, values: values, axis: 2)
        batchOffset = batchOffset + Int32(n)
        offset += n
        return (keysBuf!, valuesBuf!)
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by BatchedCompiledKVCache")
        let keyPosition = MLXArray(Int32(0) ..< Int32(capacity))
            .reshaped([1, 1, 1, capacity])
        let queryPosition = (
            batchOffset[0..., .newAxis]
                + MLXArray(Int32(0) ..< Int32(n))[.newAxis]
        ).reshaped([batchSize, 1, n, 1])
        return .array(keyPosition .<= queryPosition)
    }

    /// Increase fixed capacity between compiled calls. Logical/physical positions and
    /// membership metadata do not change; the driver retraces on the new state shape.
    public func grow(by chunk: Int) {
        precondition(chunk > 0, "cache growth must be positive")
        guard let keys = keysBuf, let values = valuesBuf else {
            preconditionFailure("BatchedCompiledKVCache grow before initialization")
        }
        keysBuf = concatenated(
            [
                keys,
                MLXArray.zeros(
                    [batchSize, keys.dim(1), chunk, keys.dim(3)], dtype: keys.dtype),
            ], axis: 2)
        valuesBuf = concatenated(
            [
                values,
                MLXArray.zeros(
                    [batchSize, values.dim(1), chunk, values.dim(3)], dtype: values.dtype),
            ], axis: 2)
        capacity += chunk
    }

    /// Keep a non-empty, unique subset of rows in the requested order.
    /// The driver must discard/rebuild the compiled closure after this shape change.
    public func filter(keeping indices: [Int]) throws {
        guard !indices.isEmpty else {
            throw BatchedCompiledKVCacheError.emptySelection
        }
        guard Set(indices).count == indices.count,
            indices.allSatisfy({ $0 >= 0 && $0 < batchSize })
        else {
            throw BatchedCompiledKVCacheError.invalidSelection(indices)
        }
        guard let keys = keysBuf, let values = valuesBuf else {
            throw BatchedCompiledKVCacheError.uninitializedSlot(0)
        }

        let selection = MLXArray(indices.map(Int32.init))
        keysBuf = keys[selection]
        valuesBuf = values[selection]
        batchOffset = batchOffset[selection]
    }

    /// Return one independent scalar cache with the same right-padded layout.
    /// Reads authoritative array metadata so compiled batch steps are reflected exactly.
    public func extract(slot: Int) throws -> CompiledKVCache {
        guard slot >= 0, slot < batchSize else {
            throw BatchedCompiledKVCacheError.invalidSlot(index: slot, batchSize: batchSize)
        }
        guard let keys = keysBuf, let values = valuesBuf else {
            throw BatchedCompiledKVCacheError.uninitializedSlot(slot)
        }

        let logical = Int(batchOffset[slot].item(Int32.self))
        guard logical >= 0, logical <= capacity else {
            throw BatchedCompiledKVCacheError.corruptedMetadata(slot: slot)
        }

        let scalar = CompiledKVCache(capacity: capacity)
        scalar.keysBuf = scalarRow(
            keys, slot: slot, validRange: 0 ..< logical, logicalLength: logical)
        scalar.valuesBuf = scalarRow(
            values, slot: slot, validRange: 0 ..< logical, logicalLength: logical)
        scalar.truncate(to: logical)
        return scalar
    }

    private func scalarRow(
        _ source: MLXArray,
        slot: Int,
        validRange: Range<Int>,
        logicalLength: Int
    ) -> MLXArray {
        var pieces: [MLXArray] = []
        if logicalLength > 0 {
            pieces.append(source[slot ..< (slot + 1), 0..., validRange, 0...])
        }
        let tail = capacity - logicalLength
        if tail > 0 {
            pieces.append(
                MLXArray.zeros([1, source.dim(1), tail, source.dim(3)], dtype: source.dtype))
        }
        return concatenated(pieces, axis: 2)
    }

    // MARK: - KVCache protocol operations not used by the continuous-batching driver

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("BatchedCompiledKVCache state restore is not supported") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("BatchedCompiledKVCache.copy() is not supported")
    }
}
