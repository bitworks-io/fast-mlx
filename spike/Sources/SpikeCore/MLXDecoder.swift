import Foundation
import MLX
import MLXLMCommon

/// Real decoder. Token selection is argmax (greedy) by default, or sampled when configured via
/// `setSampling` (the vendored `TopPSampler` honoring temperature/top-p/top-k/min-p + optional
/// seed). KEY CONSTRAINT (spec §5, backlog "lazy pipeline"): keep a one-step lookahead — submit
/// the NEXT forward with asyncEval BEFORE reading the current token to CPU, so GPU compute
/// overlaps the CPU-side .item() readback. Never call a blocking eval()+.item() in the hot path
/// with nothing else in flight (that is the 7.3x stall this spike exists to avoid).
public struct MLXDecoder: Decoder {
    private let model: any LanguageModel
    private var cache: [KVCache]
    private var pendingLogits: MLXArray?
    /// Token selection for the current generation. Default argmax (greedy) is byte-identical to
    /// the prior behavior; `setSampling` swaps in a `TopPSampler` for a `.sampled` request.
    private var sampler: any LogitSampler = ArgMaxSampler()
    /// Optional logit penalties (presence/frequency/repetition) applied to the logits BEFORE the
    /// sampler. `nil` = no penalty (byte-identical to the prior behavior). Built by `setPenalties`.
    private var processor: (any LogitProcessor)?

    public init(model: any LanguageModel, cache: [KVCache]) {
        self.model = model
        self.cache = cache
    }

    public mutating func prefill(_ promptTokens: [Int]) -> Int {
        let promptArray = MLXArray(promptTokens)
        processor?.prompt(promptArray) // seed the penalty context with the prompt tokens
        let ids = promptArray.reshaped([1, promptTokens.count])
        let logits = model(ids, cache: cache) // [1, seqLen, vocab]
        let last = logits[0..., -1, 0...] // [1, vocab]
        let processed = processor?.process(logits: last) ?? last // penalties before selection
        let next = sampler.sample(logits: processed) // [1] on GPU — argmax (greedy) or sampled
        processor?.didSample(token: next)

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
        let lastLogits = logits[0..., -1, 0...]
        let processed = processor?.process(logits: lastLogits) ?? lastLogits
        let next = sampler.sample(logits: processed)
        processor?.didSample(token: next)
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
        sampler = ArgMaxSampler()
        processor = nil
    }

    /// Configure token selection for the next generation. `.greedy` restores argmax (the default);
    /// `.sampled` builds a `TopPSampler` honoring temperature/top-p/top-k/min-p and, when supplied,
    /// a seed for reproducible draws. `TopPSampler`'s `RandomState` is a reference type, so it
    /// advances across `step` calls — each token is an independent draw.
    public mutating func setSampling(_ sampling: DecoderSampling) {
        switch sampling {
        case .greedy:
            sampler = ArgMaxSampler()
        case let .sampled(temperature, topP, topK, minP, seed):
            sampler = TopPSampler(
                temperature: Float(temperature),
                topP: Float(topP),
                topK: topK ?? 0,
                minP: Float(minP ?? 0),
                seed: seed.map { UInt64(bitPattern: $0) })
        }
    }

    /// Configure logit penalties for the next generation. Empty penalties clear the processor
    /// (byte-identical to no-penalty decode). Otherwise build the vendored `PenaltyProcessor` via
    /// `GenerateParameters.processor()`, honoring presence/frequency/repetition penalties.
    public mutating func setPenalties(_ penalties: DecoderPenalties) {
        if penalties.isEmpty {
            processor = nil
            return
        }
        let params = GenerateParameters(
            repetitionPenalty: penalties.repetitionPenalty.map { Float($0) },
            presencePenalty: penalties.presencePenalty.map { Float($0) },
            frequencyPenalty: penalties.frequencyPenalty.map { Float($0) })
        processor = params.processor()
    }
}
