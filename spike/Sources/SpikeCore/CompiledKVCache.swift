import Foundation
import MLX
import MLXLMCommon

public enum CompiledKVCacheSnapshotError: Error, Equatable, Sendable {
    case uninitializedCache
    case invalidTokenCount
    case offsetMismatch
    case unsupportedDType
    case partialSnapshot
    case invalidMetadata
    case shapeMismatch
    case dtypeMismatch
    case byteCountMismatch
    case insufficientCapacity
    case nonFiniteValues
    case invalidControlState
}

/// Detached logical-prefix storage for one dense scalar KV-cache layer.
///
/// The payload arrays are intentionally actor-confined and are not exposed publicly. Capture
/// round-trips the logical prefix through host-owned `Data`, so a later mutation of the live
/// cache cannot change this snapshot. Phase 1 is deliberately restricted to the dense native-half
/// route (`float16` or `bfloat16`); compressed, sliding, recurrent, batched, and other cache
/// families require their own complete checkpoint contracts.
public struct CompiledKVCacheSnapshot {
    public let rank: Int
    public let batchSize: Int
    public let kvHeadCount: Int
    public let tokenCount: Int
    public let headDimension: Int
    public let keyDType: DType
    public let valueDType: DType
    public let keyNBytes: Int
    public let valueNBytes: Int
    public let totalNBytes: Int

    let keys: MLXArray?
    let values: MLXArray?

    init(
        rank: Int,
        batchSize: Int,
        kvHeadCount: Int,
        tokenCount: Int,
        headDimension: Int,
        keyDType: DType,
        valueDType: DType,
        keyNBytes: Int,
        valueNBytes: Int,
        keys: MLXArray?,
        values: MLXArray?
    ) {
        self.rank = rank
        self.batchSize = batchSize
        self.kvHeadCount = kvHeadCount
        self.tokenCount = tokenCount
        self.headDimension = headDimension
        self.keyDType = keyDType
        self.valueDType = valueDType
        self.keyNBytes = keyNBytes
        self.valueNBytes = valueNBytes
        let (totalNBytes, overflow) = keyNBytes.addingReportingOverflow(valueNBytes)
        self.totalNBytes = overflow ? -1 : totalNBytes
        self.keys = keys
        self.values = values
    }
}

/// Fully validated, evaluated restore payload for one live dense cache.
///
/// Decoder-level restore prepares every layer before applying any of them, so a malformed
/// later layer cannot leave an earlier layer partially restored. The owner identity prevents
/// accidentally applying a plan to a different cache while keeping the actual mutation
/// nonthrowing.
struct CompiledKVCacheRestorePlan {
    let owner: ObjectIdentifier
    let liveKeys: MLXArray
    let liveValues: MLXArray
    let rebuiltKeys: MLXArray
    let rebuiltValues: MLXArray
    let tokenCount: Int
    let targetCapacity: Int
}

/// KV cache that is legal *inside* `MLX.compile`.
///
/// The stock `KVCacheSimple` bakes Swift `Int` offsets into every traced op (slice
/// assignment bounds, `ropeOffset`, mask sizes), so a compiled decode step would either
/// retrace every token or silently reuse a stale position. This cache keeps ALL
/// per-step state in MLXArrays instead:
///
/// - `keysBuf` / `valuesBuf`: fixed-shape `[B, kvHeads, capacity, headDim]` buffers,
///   zero-padded past the current length (fixed shapes -> the compiled graph replays
///   without retracing; growth is chunked and triggers one retrace per chunk).
/// - `offsetArr`: `[1]` int32 position array, advanced *in-graph* by `update`, exposed
///   to RoPE via `.batch(offsetArr)` (`mlx_fast_rope_dynamic`) and to attention via a
///   boolean mask computed from it — no Swift-side value ever changes between steps.
///
/// The padded tail is masked out of attention (scores -inf -> softmax weight 0), and
/// the padded V rows are finite (zeros / stale reals), so it contributes exactly 0.
public final class CompiledKVCache: KVCache, Updatable {
    public private(set) var capacity: Int

    // Buffers are allocated lazily on the first (uncompiled, prefill) update so the
    // kv-head count / head dim / dtype come from the model itself. They exist before
    // the step function is compiled, so `innerState()` is stable across compiled calls.
    var keysBuf: MLXArray?
    var valuesBuf: MLXArray?
    var offsetArr: MLXArray = MLXArray([Int32(0)])

    /// Host-side mirror of the cached-token count. Only meaningful during *uncompiled*
    /// use (prefill): compiled-step replays advance `offsetArr` in-graph without
    /// executing this Swift code, so the compiled driver must track position itself.
    public private(set) var offset: Int = 0

    public init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] {
        [keysBuf, valuesBuf].compactMap { $0 } + [offsetArr]
    }

    /// Dynamic (array) RoPE offset — recorded as a graph input, not a baked constant.
    public var ropeOffset: RoPEOffset { .batch(offsetArr) }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let n = keys.dim(2)
        if keysBuf == nil {
            keysBuf = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), capacity, keys.dim(3)], dtype: keys.dtype)
            valuesBuf = MLXArray.zeros(
                [values.dim(0), values.dim(1), capacity, values.dim(3)], dtype: values.dtype)
        }
        // Scatter the new rows at positions offset..<offset+n along the sequence axis.
        let positions = offsetArr + MLXArray(Int32(0) ..< Int32(n)) // [n]
        let indices = positions.reshaped([1, 1, n, 1])
        keysBuf = putAlong(keysBuf!, indices, values: keys, axis: 2)
        valuesBuf = putAlong(valuesBuf!, indices, values: values, axis: 2)
        offsetArr = offsetArr + MLXArray([Int32(n)])
        offset += n
        return (keysBuf!, valuesBuf!)
    }

    /// Boolean mask over the whole fixed-size buffer: query at position offset+i may
    /// attend to key positions j <= offset+i. Always an array mask (never `.none`),
    /// because the buffer's padded tail must be excluded even for single-token steps.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by CompiledKVCache")
        let kPos = MLXArray(Int32(0) ..< Int32(capacity)).reshaped([1, 1, 1, capacity])
        let qPos = (offsetArr + MLXArray(Int32(0) ..< Int32(n))).reshaped([1, 1, n, 1])
        return .array(kPos .<= qPos)
    }

    /// Grow the buffers by `chunk` sequence slots (call only from the *uncompiled*
    /// driver, between steps). The state-shape change makes the next compiled call
    /// retrace automatically — one retrace per chunk, amortized.
    public func grow(by chunk: Int) {
        guard let k = keysBuf, let v = valuesBuf else {
            capacity += chunk
            return
        }
        keysBuf = concatenated(
            [k, MLXArray.zeros([k.dim(0), k.dim(1), chunk, k.dim(3)], dtype: k.dtype)], axis: 2)
        valuesBuf = concatenated(
            [v, MLXArray.zeros([v.dim(0), v.dim(1), chunk, v.dim(3)], dtype: v.dtype)], axis: 2)
        capacity += chunk
    }

    /// Roll the cached-token count BACK to `newLength` (speculative-decoding rollback:
    /// a verify forward wrote K+1 rows, the accept-walk rejected a suffix). The offset is
    /// the cache's single source of truth — the attention mask and RoPE offset both derive
    /// from `offsetArr` — so rolling it back makes the rejected rows unreachable (masked
    /// out) and the next `update` overwrites them in place. IN PLACE (`_updateInternal`),
    /// mirroring `resetInPlace`: the MLXArray identities a compiled step is bound to are
    /// preserved, and no buffer is reallocated.
    ///
    /// NOTE: not validated against the host `offset` mirror — compiled-step replays
    /// advance `offsetArr` in-graph without touching the mirror, so the mirror may be
    /// smaller than the true in-graph position (see `offset`'s doc). Callers (the spec-
    /// decode driver) track the true position themselves.
    public func truncate(to newLength: Int) {
        precondition(newLength >= 0 && newLength <= capacity, "truncate target outside the buffer")
        offsetArr._updateInternal(MLXArray([Int32(newLength)]))
        offset = newLength
    }

    /// Reset to empty IN PLACE: writes fresh contents into the *same* MLXArray objects
    /// via `_updateInternal`, preserving the identity that an already-compiled step
    /// function is bound to (so a reset does not force a rebuild or retrace).
    public func resetInPlace() {
        if let k = keysBuf { k._updateInternal(MLXArray.zeros(k.shape, dtype: k.dtype)) }
        if let v = valuesBuf { v._updateInternal(MLXArray.zeros(v.shape, dtype: v.dtype)) }
        offsetArr._updateInternal(MLXArray([Int32(0)]))
        offset = 0
    }

    /// Capture the exact logical prefix into independently backed, evaluated dense-half arrays.
    ///
    /// `logicalTokenCount` is authoritative because compiled replay advances `offsetArr` without
    /// executing the Swift `offset` mirror. Capture still requires the in-graph offset to agree,
    /// so partial or stale evidence cannot enter the hot cache.
    public func captureSnapshot(
        logicalTokenCount: Int
    ) throws -> CompiledKVCacheSnapshot {
        guard let keys = keysBuf, let values = valuesBuf else {
            throw CompiledKVCacheSnapshotError.uninitializedCache
        }
        guard logicalTokenCount > 0,
            logicalTokenCount <= capacity,
            logicalTokenCount <= Int(Int32.max)
        else {
            throw CompiledKVCacheSnapshotError.invalidTokenCount
        }
        try validateControlState()
        guard Int(offsetArr.item(Int32.self)) == logicalTokenCount else {
            throw CompiledKVCacheSnapshotError.offsetMismatch
        }
        let geometry = try Self.validateLiveBuffers(
            keys: keys, values: values, capacity: capacity)
        guard Self.isSupportedSnapshotDType(keys.dtype),
            Self.isSupportedSnapshotDType(values.dtype)
        else {
            throw CompiledKVCacheSnapshotError.unsupportedDType
        }

        let logicalKeys = keys[0..., 0..., 0 ..< logicalTokenCount, 0...]
        let logicalValues = values[0..., 0..., 0 ..< logicalTokenCount, 0...]
        guard Self.arraysAreFinite([logicalKeys, logicalValues]) else {
            throw CompiledKVCacheSnapshotError.nonFiniteValues
        }

        let detachedKeys = Self.detachedCopy(of: logicalKeys)
        let detachedValues = Self.detachedCopy(of: logicalValues)
        guard Self.arraysAreFinite([detachedKeys, detachedValues]) else {
            throw CompiledKVCacheSnapshotError.nonFiniteValues
        }
        let keyNBytes = try Self.exactNBytes(
            shape: detachedKeys.shape, dtype: detachedKeys.dtype)
        let valueNBytes = try Self.exactNBytes(
            shape: detachedValues.shape, dtype: detachedValues.dtype)
        guard detachedKeys.nbytes == keyNBytes,
            detachedValues.nbytes == valueNBytes
        else {
            throw CompiledKVCacheSnapshotError.byteCountMismatch
        }

        return CompiledKVCacheSnapshot(
            rank: 4,
            batchSize: geometry.batchSize,
            kvHeadCount: geometry.kvHeadCount,
            tokenCount: logicalTokenCount,
            headDimension: geometry.headDimension,
            keyDType: detachedKeys.dtype,
            valueDType: detachedValues.dtype,
            keyNBytes: keyNBytes,
            valueNBytes: valueNBytes,
            keys: detachedKeys,
            values: detachedValues)
    }

    /// Restore a detached dense prefix into the existing compiled-cache wrappers.
    ///
    /// Every metadata, payload, live-buffer, and control-state check completes before mutation.
    /// The rebuilt buffers cover the full live capacity and zero the unwritten tail; only then are
    /// the existing MLXArray wrappers updated in place and both graph/host offsets advanced to the
    /// same logical length.
    public func restoreInPlace(
        from snapshot: CompiledKVCacheSnapshot
    ) throws {
        let plan = try prepareRestore(
            from: snapshot, targetCapacity: capacity)
        applyPreparedRestore(plan)
    }

    /// Validate and rebuild a detached prefix without mutating live cache state.
    ///
    /// This is internal so `CompiledMLXDecoder` can prepare every layer first and then apply
    /// the plans as one actor-confined state transition.
    func prepareRestore(
        from snapshot: CompiledKVCacheSnapshot,
        targetCapacity: Int
    ) throws -> CompiledKVCacheRestorePlan {
        guard let liveKeys = keysBuf, let liveValues = valuesBuf else {
            throw CompiledKVCacheSnapshotError.uninitializedCache
        }
        guard let snapshotKeys = snapshot.keys, let snapshotValues = snapshot.values else {
            throw CompiledKVCacheSnapshotError.partialSnapshot
        }
        try validateControlState()
        let liveGeometry = try Self.validateLiveBuffers(
            keys: liveKeys, values: liveValues, capacity: capacity)
        guard targetCapacity >= capacity,
            targetCapacity <= Int(Int32.max)
        else {
            throw CompiledKVCacheSnapshotError.invalidMetadata
        }
        guard snapshot.rank == 4,
            snapshot.batchSize > 0,
            snapshot.kvHeadCount > 0,
            snapshot.tokenCount > 0,
            snapshot.headDimension > 0,
            snapshot.tokenCount <= Int(Int32.max),
            snapshot.keyNBytes > 0,
            snapshot.valueNBytes > 0,
            snapshot.totalNBytes > 0
        else {
            throw CompiledKVCacheSnapshotError.invalidMetadata
        }
        guard snapshot.tokenCount <= targetCapacity else {
            throw CompiledKVCacheSnapshotError.insufficientCapacity
        }
        guard Self.isSupportedSnapshotDType(snapshot.keyDType),
            Self.isSupportedSnapshotDType(snapshot.valueDType)
        else {
            throw CompiledKVCacheSnapshotError.unsupportedDType
        }
        guard snapshotKeys.dtype == snapshot.keyDType,
            snapshotValues.dtype == snapshot.valueDType
        else {
            throw CompiledKVCacheSnapshotError.dtypeMismatch
        }

        let expectedShape = [
            snapshot.batchSize,
            snapshot.kvHeadCount,
            snapshot.tokenCount,
            snapshot.headDimension,
        ]
        guard snapshotKeys.shape == expectedShape,
            snapshotValues.shape == expectedShape
        else {
            throw CompiledKVCacheSnapshotError.shapeMismatch
        }
        let exactKeyNBytes = try Self.exactNBytes(
            shape: expectedShape, dtype: snapshot.keyDType)
        let exactValueNBytes = try Self.exactNBytes(
            shape: expectedShape, dtype: snapshot.valueDType)
        let (exactTotalNBytes, totalOverflow) = exactKeyNBytes.addingReportingOverflow(
            exactValueNBytes)
        guard !totalOverflow,
            snapshot.keyNBytes == exactKeyNBytes,
            snapshot.valueNBytes == exactValueNBytes,
            snapshot.totalNBytes == exactTotalNBytes,
            snapshotKeys.nbytes == exactKeyNBytes,
            snapshotValues.nbytes == exactValueNBytes
        else {
            throw CompiledKVCacheSnapshotError.byteCountMismatch
        }
        guard liveGeometry.batchSize == snapshot.batchSize,
            liveGeometry.kvHeadCount == snapshot.kvHeadCount,
            liveGeometry.headDimension == snapshot.headDimension
        else {
            throw CompiledKVCacheSnapshotError.shapeMismatch
        }
        guard liveKeys.dtype == snapshot.keyDType,
            liveValues.dtype == snapshot.valueDType
        else {
            throw CompiledKVCacheSnapshotError.dtypeMismatch
        }
        guard Self.arraysAreFinite([snapshotKeys, snapshotValues]) else {
            throw CompiledKVCacheSnapshotError.nonFiniteValues
        }

        let tailCount = targetCapacity - snapshot.tokenCount
        let rebuiltKeys: MLXArray
        let rebuiltValues: MLXArray
        if tailCount == 0 {
            rebuiltKeys = snapshotKeys
            rebuiltValues = snapshotValues
        } else {
            let keyTail = MLXArray.zeros(
                [
                    snapshot.batchSize, snapshot.kvHeadCount,
                    tailCount, snapshot.headDimension,
                ],
                dtype: snapshot.keyDType)
            let valueTail = MLXArray.zeros(
                [
                    snapshot.batchSize, snapshot.kvHeadCount,
                    tailCount, snapshot.headDimension,
                ],
                dtype: snapshot.valueDType)
            rebuiltKeys = concatenated([snapshotKeys, keyTail], axis: 2)
            rebuiltValues = concatenated([snapshotValues, valueTail], axis: 2)
        }
        eval([rebuiltKeys, rebuiltValues])
        let targetShape = [
            liveGeometry.batchSize,
            liveGeometry.kvHeadCount,
            targetCapacity,
            liveGeometry.headDimension,
        ]
        guard rebuiltKeys.shape == targetShape,
            rebuiltValues.shape == targetShape,
            rebuiltKeys.dtype == liveKeys.dtype,
            rebuiltValues.dtype == liveValues.dtype,
            Self.arraysAreFinite([rebuiltKeys, rebuiltValues])
        else {
            throw CompiledKVCacheSnapshotError.shapeMismatch
        }

        return CompiledKVCacheRestorePlan(
            owner: ObjectIdentifier(self),
            liveKeys: liveKeys,
            liveValues: liveValues,
            rebuiltKeys: rebuiltKeys,
            rebuiltValues: rebuiltValues,
            tokenCount: snapshot.tokenCount,
            targetCapacity: targetCapacity)
    }

    /// Apply a previously prepared restore while preserving the live MLXArray identities.
    func applyPreparedRestore(_ plan: CompiledKVCacheRestorePlan) {
        precondition(
            plan.owner == ObjectIdentifier(self),
            "dense KV restore plan belongs to a different cache")
        precondition(
            plan.targetCapacity >= capacity,
            "dense KV restore plan cannot shrink live capacity")
        if plan.targetCapacity == capacity {
            plan.liveKeys._updateInternal(plan.rebuiltKeys)
            plan.liveValues._updateInternal(plan.rebuiltValues)
        } else {
            precondition(
                keysBuf === plan.liveKeys && valuesBuf === plan.liveValues,
                "dense KV cache changed after restore preparation")
            keysBuf = plan.rebuiltKeys
            valuesBuf = plan.rebuiltValues
            capacity = plan.targetCapacity
        }
        offsetArr._updateInternal(MLXArray([Int32(plan.tokenCount)]))
        offset = plan.tokenCount
    }

    private struct DenseGeometry {
        let batchSize: Int
        let kvHeadCount: Int
        let headDimension: Int
    }

    private func validateControlState() throws {
        guard offsetArr.shape == [1],
            offsetArr.dtype == .int32
        else {
            throw CompiledKVCacheSnapshotError.invalidControlState
        }
        let graphOffset = Int(offsetArr.item(Int32.self))
        guard graphOffset >= 0, graphOffset <= capacity else {
            throw CompiledKVCacheSnapshotError.invalidControlState
        }
    }

    private static func validateLiveBuffers(
        keys: MLXArray,
        values: MLXArray,
        capacity: Int
    ) throws -> DenseGeometry {
        guard keys.shape.count == 4,
            values.shape.count == 4,
            keys.dim(0) > 0,
            keys.dim(1) > 0,
            keys.dim(2) == capacity,
            keys.dim(3) > 0,
            values.shape == keys.shape
        else {
            throw CompiledKVCacheSnapshotError.shapeMismatch
        }
        return DenseGeometry(
            batchSize: keys.dim(0),
            kvHeadCount: keys.dim(1),
            headDimension: keys.dim(3))
    }

    private static func arraysAreFinite(_ arrays: [MLXArray]) -> Bool {
        arrays.allSatisfy { array in
            guard isSupportedSnapshotDType(array.dtype) else {
                return false
            }
            return isFinite(array).all().item(Bool.self)
        }
    }

    private static func isSupportedSnapshotDType(
        _ dtype: DType
    ) -> Bool {
        dtype == .float16 || dtype == .bfloat16
    }

    private static func detachedCopy(of array: MLXArray) -> MLXArray {
        let hostCopy = array.asData(access: .copy)
        return MLXArray(hostCopy.data, hostCopy.shape, dtype: hostCopy.dType)
    }

    private static func exactNBytes(
        shape: [Int],
        dtype: DType
    ) throws -> Int {
        guard isSupportedSnapshotDType(dtype), !shape.isEmpty else {
            throw CompiledKVCacheSnapshotError.unsupportedDType
        }
        var elements = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw CompiledKVCacheSnapshotError.invalidMetadata
            }
            let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw CompiledKVCacheSnapshotError.byteCountMismatch
            }
            elements = next
        }
        let (nbytes, overflow) = elements.multipliedReportingOverflow(by: dtype.size)
        guard !overflow else {
            throw CompiledKVCacheSnapshotError.byteCountMismatch
        }
        return nbytes
    }

    // MARK: - KVCache protocol leftovers (not used by the spike's decode path)

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("CompiledKVCache state restore not supported in the spike") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("CompiledKVCache.copy() not supported in the spike")
    }
}
