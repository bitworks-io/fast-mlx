import Foundation
import MLX
import MLXLMCommon

/// Decoder whose single-token step is wrapped in `MLX.compile`.
///
/// Why: Instruments showed the baseline `MLXDecoder`'s 6.7ms/step "submit" phase is
/// ~82% inside `mlx_async_eval` -> C++ `eval_impl` (per-primitive graph traversal +
/// Metal kernel encoding) and ~18% Swift graph construction — repeated for every
/// token. Compiling the step replays a cached, fused graph instead: the Swift forward
/// runs once per (shape signature), elementwise chains fuse into fewer kernels, and
/// per-step traversal shrinks with the node count.
///
/// The KV-cache position is carried by the caches as in-graph state (array offset +
/// fixed-size buffers), which is what makes the trace valid for every step. The cache
/// implementation is selected by `KVCacheKind` — fp16 `CompiledKVCache` or a TurboQuant
/// tier's `TurboQuantKVCache`; both satisfy `CompiledCache`, the contract this decoder
/// needs (Updatable state, chunked grow, identity-preserving reset).
/// Keeps MLXDecoder's submit-first lookahead: the next step's compiled call is
/// submitted (asyncEval) BEFORE the current token's blocking `.item()` readback.
public struct CompiledMLXDecoder: Decoder {
    private let model: any LanguageModel
    private let kvCacheKind: KVCacheKind
    /// Escape hatch for cache implementations whose update ops fail to trace under
    /// `MLX.compile`: `false` runs the same step closure uncompiled (correctness path;
    /// per-token graph construction returns, so decode perf drops — flagged, never silent).
    private let compileStepEnabled: Bool
    private let chunk = 256
    /// Decode headroom preallocated beyond the prompt (rounded up to `chunk`).
    private let reserve: Int

    private var caches: [any CompiledCache] = []
    // Plain (non-@Sendable) function type: `compile` returns a @Sendable function (a
    // subtype, assigns fine), while the uncompiled fallback closure captures the
    // non-Sendable model/caches and stays confined to this decoder like all MLX state.
    private var compiledStep: (([MLXArray]) -> [MLXArray])?
    private var pendingNext: MLXArray? // [1] token id for the current position (lazy)
    private var cachedTokens = 0 // host-side position; caches' host mirror goes stale

    public init(
        model: any LanguageModel, reserve: Int = 384,
        kvCache: KVCacheKind = .fp16, compileStep: Bool = true
    ) {
        self.model = model
        self.reserve = reserve
        self.kvCacheKind = kvCache
        self.compileStepEnabled = compileStep
    }

    public mutating func prefill(_ promptTokens: [Int]) -> Int {
        let promptLength = promptTokens.count
        if caches.isEmpty {
            let layerCount = model.newCache(parameters: nil).count
            let cap = ((promptLength + reserve + chunk - 1) / chunk) * chunk
            caches = (0 ..< layerCount).map { _ in kvCacheKind.makeCache(capacity: cap) }
        }
        while promptLength + 1 > caches[0].capacity {
            for cache in caches { cache.grow(by: chunk) }
        }

        let ids = MLXArray(promptTokens).reshaped([1, promptLength])
        let logits = model(ids, cache: caches) // uncompiled prefill, dynamic cache ops
        let first = argMax(logits[0..., -1, 0...], axis: -1) // [1]
        cachedTokens = promptLength

        if compiledStep == nil {
            // Weights are captured as constants; ONLY the caches are mutable state.
            let model = self.model
            let caches = self.caches
            let step: ([MLXArray]) -> [MLXArray] = { args in
                let y = args[0].reshaped([1, 1])
                let logits = model(y, cache: caches)
                return [argMax(logits[0..., -1, 0...], axis: -1)]
            }
            // Uncompiled variant is the same closure, graph-built fresh every call.
            compiledStep = compileStepEnabled ? compile(inputs: caches, outputs: caches, step) : step
        }

        // submit-first: the compiled forward for the position after `first` is in
        // flight before we block reading `first` back to the CPU.
        let next = compiledStep!([first])[0]
        asyncEval(next)
        pendingNext = next
        cachedTokens += 1
        return first.item(Int.self)
    }

    public mutating func step(last: Int) -> Int {
        guard let next = pendingNext, let compiledStep else {
            fatalError("CompiledMLXDecoder.step called before prefill")
        }
        if cachedTokens + 1 > caches[0].capacity {
            for cache in caches { cache.grow(by: chunk) } // next call retraces once
        }
        let following = compiledStep([next])[0]
        asyncEval(following)
        pendingNext = following
        cachedTokens += 1
        return next.item(Int.self)
    }

    /// Reset caches IN PLACE (same MLXArray identities) so the compiled step function
    /// stays valid across bench runs — no rebuild, no retrace, no cross-actor resend.
    public mutating func reset() {
        for cache in caches { cache.resetInPlace() }
        pendingNext = nil
        cachedTokens = 0
    }

    /// Engagement probe: the IN-GRAPH cached-token count from layer 0's TurboQuant cache.
    /// `offsetArr` advances inside the (compiled) step itself, so this is proof the
    /// quantized cache's update actually ran — a silently-substituted fp16 cache returns
    /// nil. Forces a sync readback; call after a run, never in the hot loop.
    public func turboQuantCachedTokens() -> Int? {
        guard let cache = caches.first as? TurboQuantKVCache else { return nil }
        return Int(cache.offsetArr.item(Int32.self))
    }
}
