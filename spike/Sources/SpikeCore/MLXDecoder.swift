import Foundation
import MLX
import MLXLMCommon

/// Real decoder. Greedy (temp=0). KEY CONSTRAINT (spec §5, backlog "lazy pipeline"):
/// keep a one-step lookahead — submit the NEXT forward with asyncEval BEFORE reading the
/// current token to CPU, so GPU compute overlaps the CPU-side .item() readback. Never call
/// a blocking eval()+.item() in the hot path with nothing else in flight (that is the 7.3x
/// stall this spike exists to avoid).
public struct MLXDecoder: Decoder {
    private let model: any LanguageModel
    private var cache: [KVCache]
    private var pendingLogits: MLXArray?

    public init(model: any LanguageModel, cache: [KVCache]) {
        self.model = model
        self.cache = cache
    }

    public mutating func prefill(_ promptTokens: [Int]) -> Int {
        let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        let logits = model(ids, cache: cache) // [1, seqLen, vocab]
        let last = logits[0..., -1, 0...] // [1, vocab]
        let next = argMax(last, axis: -1) // [1] on GPU

        // submit-first: kick the next forward before we read `next` to CPU
        let nextIds = next.reshaped([1, 1])
        let nextLogits = model(nextIds, cache: cache)
        asyncEval(nextLogits) // overlap GPU with the readback below
        pendingLogits = nextLogits

        return next.item(Int.self) // readback overlaps the pending forward
    }

    public mutating func step(last: Int) -> Int {
        // pendingLogits already computed for the position after `last`
        guard let logits = pendingLogits else {
            fatalError("MLXDecoder.step called before prefill")
        }
        let next = argMax(logits[0..., -1, 0...], axis: -1)
        let nextIds = next.reshaped([1, 1])
        let nextLogits = model(nextIds, cache: cache) // submit next
        asyncEval(nextLogits)
        pendingLogits = nextLogits
        return next.item(Int.self)
    }

    /// Rebuild the KV cache from `model` (already owned by this decoder, so this never
    /// needs to cross the actor boundary with a fresh non-Sendable reference) and drop
    /// any pending lookahead. Used between bench runs so each run starts with an empty
    /// cache instead of seeing the previous run's tokens as false history.
    public mutating func reset() {
        cache = model.newCache(parameters: nil)
        pendingLogits = nil
    }
}
