import Foundation
import MLX
import MLXLMCommon

/// KV cache that stores TurboQuant `_prod` codes per token and materializes full-precision
/// K/V on read — "materialize-then-attend" (plan Phase 2 Task 6): the existing attention
/// path receives ordinary `[B, kvHeads, capacity, headDim]` tensors and needs no changes.
///
/// Storage mirrors `CompiledKVCache`'s compile-legal shape discipline exactly: fixed-shape
/// buffers zero-padded past the current length, a `[1]` int32 offset advanced in-graph,
/// chunked `grow(by:)` between steps (one retrace per chunk — the same guard against the
/// O(context²) allocator churn), and identity-preserving `resetInPlace()`. Instead of fp16
/// rows, each token holds the Task-6a normalized `_prod` code, one fixed `TurboQuantParams`
/// per head_dim (the paper's global parameters):
///
/// - `idx`    `[B, H, capacity, d]`  base-bit Lloyd-Max centroid indices
/// - `signs`  `[B, H, capacity, d]`  QJL residual signs (±1)
/// - `gamma`  `[B, H, capacity, 1]`  residual norm γ = ‖r‖₂
/// - `xNorm`  `[B, H, capacity, 1]`  input norm ‖x‖ (non-unit-norm handling, paper §1.1)
///
/// Padded rows materialize to exactly 0 because their stored ‖x‖ is 0 — the same zero
/// padding `CompiledKVCache` presents, and the mask excludes them from attention anyway.
///
/// Memory policy: like `CompiledKVCache`, this type never touches `Memory.cacheLimit`;
/// that bound belongs to the driver (see `CapacityModel.cacheLimit` and the harness
/// drivers). The cache's job is keeping buffer shapes fixed so allocations stay bounded.
///
/// Not `Sendable` (holds `MLXArray`s) — confined to the inference actor like all MLX state
/// in SpikeCore.
public final class TurboQuantKVCache: KVCache, Updatable {
    public private(set) var capacity: Int
    let params: TurboQuantParams

    // Code buffers, allocated lazily on the first (uncompiled, prefill) update so batch,
    // kv-head count, and dtypes come from the model itself — same lifecycle as
    // CompiledKVCache, so `innerState()` is stable before the step function compiles.
    var kIdx: MLXArray?
    var kSigns: MLXArray?
    var kGamma: MLXArray?
    var kXNorm: MLXArray?
    var vIdx: MLXArray?
    var vSigns: MLXArray?
    var vGamma: MLXArray?
    var vXNorm: MLXArray?
    var offsetArr: MLXArray = MLXArray([Int32(0)])

    /// Dtype the decode path expects back (recorded from the first update's inputs);
    /// codec math runs in float32 and the materialized K/V is cast back to this.
    private var outDType: DType?

    /// Host-side mirror of the cached-token count — only meaningful during *uncompiled*
    /// use (prefill), exactly as documented on `CompiledKVCache.offset`.
    public private(set) var offset: Int = 0

    public init(capacity: Int, params: TurboQuantParams) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.params = params
    }

    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] {
        [kIdx, kSigns, kGamma, kXNorm, vIdx, vSigns, vGamma, vXNorm].compactMap { $0 }
            + [offsetArr]
    }

    /// Dynamic (array) RoPE offset — a graph input, not a baked constant.
    public var ropeOffset: RoPEOffset { .batch(offsetArr) }

    /// Quantize the incoming `[B, H, n, d]` K/V, scatter their code fields at
    /// positions offset..<offset+n, and return the dequantized FULL buffers.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(
            keys.dim(3) == params.headDim && values.dim(3) == params.headDim,
            "TurboQuantKVCache params are per head_dim (\(params.headDim))")
        let n = keys.dim(2)
        let kCode = encode(keys)
        let vCode = encode(values)
        if kIdx == nil {
            allocate(like: keys, idxDType: kCode.idx.dtype)
            outDType = keys.dtype
        }

        let positions = offsetArr + MLXArray(Int32(0) ..< Int32(n))  // [n]
        let indices = positions.reshaped([1, 1, n, 1])
        kIdx = putAlong(kIdx!, indices, values: kCode.idx, axis: 2)
        kSigns = putAlong(kSigns!, indices, values: kCode.signs, axis: 2)
        kGamma = putAlong(kGamma!, indices, values: kCode.gamma, axis: 2)
        kXNorm = putAlong(kXNorm!, indices, values: kCode.xNorm, axis: 2)
        vIdx = putAlong(vIdx!, indices, values: vCode.idx, axis: 2)
        vSigns = putAlong(vSigns!, indices, values: vCode.signs, axis: 2)
        vGamma = putAlong(vGamma!, indices, values: vCode.gamma, axis: 2)
        vXNorm = putAlong(vXNorm!, indices, values: vCode.xNorm, axis: 2)
        offsetArr = offsetArr + MLXArray([Int32(n)])
        offset += n

        let mk = materialize(idx: kIdx!, signs: kSigns!, gamma: kGamma!, xNorm: kXNorm!)
        let mv = materialize(idx: vIdx!, signs: vSigns!, gamma: vGamma!, xNorm: vXNorm!)
        return (mk, mv)
    }

    /// Boolean mask over the whole fixed-size buffer — identical to `CompiledKVCache`:
    /// query at offset+i attends to key positions j <= offset+i; the padded tail is
    /// always excluded.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by TurboQuantKVCache")
        let kPos = MLXArray(Int32(0) ..< Int32(capacity)).reshaped([1, 1, 1, capacity])
        let qPos = (offsetArr + MLXArray(Int32(0) ..< Int32(n))).reshaped([1, 1, n, 1])
        return .array(kPos .<= qPos)
    }

    /// Grow every code buffer by `chunk` sequence slots (uncompiled driver only, between
    /// steps) — the state-shape change retraces the compiled step once per chunk.
    public func grow(by chunk: Int) {
        guard kIdx != nil else {
            capacity += chunk
            return
        }
        func grown(_ buf: MLXArray) -> MLXArray {
            let pad = MLXArray.zeros(
                [buf.dim(0), buf.dim(1), chunk, buf.dim(3)], dtype: buf.dtype)
            return concatenated([buf, pad], axis: 2)
        }
        kIdx = grown(kIdx!)
        kSigns = grown(kSigns!)
        kGamma = grown(kGamma!)
        kXNorm = grown(kXNorm!)
        vIdx = grown(vIdx!)
        vSigns = grown(vSigns!)
        vGamma = grown(vGamma!)
        vXNorm = grown(vXNorm!)
        capacity += chunk
    }

    /// Reset to empty IN PLACE (`_updateInternal`), preserving every MLXArray identity a
    /// compiled step function is bound to — zeroed ‖x‖ makes all rows materialize to 0.
    public func resetInPlace() {
        for buf in [kIdx, kSigns, kGamma, kXNorm, vIdx, vSigns, vGamma, vXNorm] {
            if let b = buf { b._updateInternal(MLXArray.zeros(b.shape, dtype: b.dtype)) }
        }
        offsetArr._updateInternal(MLXArray([Int32(0)]))
        offset = 0
    }

    // MARK: - Codec plumbing

    /// A quantized `[B, H, n, ·]` block ready to scatter into the code buffers.
    private struct StoredCode {
        let idx: MLXArray  // [B, H, n, d]
        let signs: MLXArray  // [B, H, n, d]
        let gamma: MLXArray  // [B, H, n, 1]
        let xNorm: MLXArray  // [B, H, n, 1]
    }

    /// Flatten `[B, H, n, d]` to codec rows, quantize (float32 math), reshape the code
    /// fields back to scatter shape.
    private func encode(_ x: MLXArray) -> StoredCode {
        let (b, h, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let code = TurboQuantCodec.quantizeProdNormalized(
            x.asType(.float32).reshaped([-1, d]), params: params)
        return StoredCode(
            idx: code.idx.reshaped([b, h, n, d]),
            signs: code.signs.reshaped([b, h, n, d]),
            gamma: code.norms.reshaped([b, h, n, 1]),
            xNorm: code.xNorm!.reshaped([b, h, n, 1]))
    }

    /// Dequantize a full `[B, H, capacity, ·]` buffer set back to `[B, H, capacity, d]`
    /// in the decode path's dtype. Padded rows come out exactly 0 (stored ‖x‖ = 0).
    private func materialize(
        idx: MLXArray, signs: MLXArray, gamma: MLXArray, xNorm: MLXArray
    ) -> MLXArray {
        let (b, h, cap, d) = (idx.dim(0), idx.dim(1), idx.dim(2), idx.dim(3))
        let code = TurboQuantCode(
            idx: idx.reshaped([-1, d]),
            signs: signs.reshaped([-1, d]),
            norms: gamma.reshaped([-1, 1]),
            xNorm: xNorm.reshaped([-1, 1]))
        return TurboQuantCodec.dequantizeProdNormalized(code, params: params)
            .reshaped([b, h, cap, d])
            .asType(outDType ?? .float32)
    }

    private func allocate(like keys: MLXArray, idxDType: DType) {
        let (b, h) = (keys.dim(0), keys.dim(1))
        let d = params.headDim
        kIdx = MLXArray.zeros([b, h, capacity, d], dtype: idxDType)
        kSigns = MLXArray.zeros([b, h, capacity, d], dtype: .float32)
        kGamma = MLXArray.zeros([b, h, capacity, 1], dtype: .float32)
        kXNorm = MLXArray.zeros([b, h, capacity, 1], dtype: .float32)
        vIdx = MLXArray.zeros([b, h, capacity, d], dtype: idxDType)
        vSigns = MLXArray.zeros([b, h, capacity, d], dtype: .float32)
        vGamma = MLXArray.zeros([b, h, capacity, 1], dtype: .float32)
        vXNorm = MLXArray.zeros([b, h, capacity, 1], dtype: .float32)
    }

    // MARK: - KVCache protocol leftovers (not used by the spike's decode path)

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("TurboQuantKVCache state restore not supported in the spike") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("TurboQuantKVCache.copy() not supported in the spike")
    }
}
