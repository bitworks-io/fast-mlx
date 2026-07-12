import Foundation
import MLX
import MLXLMCommon

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
