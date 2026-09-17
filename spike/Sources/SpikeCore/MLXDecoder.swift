import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Real decoder. Token selection is argmax (greedy) by default, or sampled when configured via
/// `setSampling` (the vendored `TopPSampler` honoring temperature/top-p/top-k/min-p + optional
/// seed). KEY CONSTRAINT (spec §5, backlog "lazy pipeline"): keep a one-step lookahead — submit
/// the NEXT forward with asyncEval BEFORE reading the current token to CPU, so GPU compute
/// overlaps the CPU-side .item() readback. Never call a blocking eval()+.item() in the hot path
/// with nothing else in flight (that is the 7.3x stall this spike exists to avoid).
public struct MLXDecoder: Decoder, LogprobDecoding {
    /// Prompt tokens per prefill forward. `prefill` runs the prompt through this many chunks
    /// rather than one whole-prompt forward, bounding the transient working set — for the
    /// `qwen4_exp` family the QSA indexer's O(Q·K) intermediate from a whole-prompt forward
    /// derives to tens of GB at an 8K prompt, a deterministic OOM (task-inbox
    /// 2026-09-05-qwen4exp-item3-split-prefill-oom-DECISION, item 3b).
    ///
    /// This MUST stay equal to `CapacityModel.predictPeakBytes`'s `chunkTokens: 2048` default
    /// (`spike/Sources/HarnessCore/CapacityModel.swift`): that capacity fit already prices
    /// prefill as if it were chunked at 2048 tokens. If the two constants drift apart, the fit
    /// check stops describing what the runtime actually does.
    ///
    /// SECOND CONSUMER: `MTPSpeculativeDecoder.buildParameters()` also reads this constant, to set
    /// `GenerateParameters.prefillStepSize` for the MTP speculative serving route. That route does
    /// not otherwise share this decoder's prefill loop, so without that assignment it would
    /// silently prefill at the vendored `GenerateParameters` default (512) instead — a
    /// monolithic-vs-chunked geometry difference between the two serving routes unrelated to
    /// speculation. Changing this value changes BOTH routes' prefill chunking, not just this one's.
    public static let defaultPrefillChunkSize = 2048

    private let model: any LanguageModel
    /// Rebuilds the same cache family selected at load for every request reset. The default
    /// initializer uses `model.newCache`; explicit runtime KV tiers inject their own factory so a
    /// reset can never silently change the storage format back to the model's native fp16 cache.
    private let cacheFactory: () -> [KVCache]
    private var cache: [KVCache]
    private var pendingLogits: MLXArray?
    /// Token selection for the current generation. Default argmax (greedy) is byte-identical to
    /// the prior behavior; `setSampling` swaps in a `TopPSampler` for a `.sampled` request.
    private var sampler: any LogitSampler = ArgMaxSampler()
    /// Optional logit penalties (presence/frequency/repetition) applied to the logits BEFORE the
    /// sampler. `nil` = no penalty (byte-identical to the prior behavior). Built by `setPenalties`.
    private var processor: (any LogitProcessor)?
    /// Optional trailing logit processor applied AFTER `processor` (penalties), configured via
    /// `setResponseFormatConstraint`. A SEPARATE slot from `processor` deliberately: unlike
    /// penalties (a value-typed `PenaltyProcessor` built fresh per request), a response-format
    /// constraint also needs the `ConstraintProcessorFailureReporting` failure-surfacing contract
    /// `selectSampleAndAdvance` checks after every `didSample` call — see that method and
    /// `ConstraintProcessorFailureReporting`'s doc comment for why this composes LAST (mask applied
    /// to the already-penalized logits) per the response-format design's "penalties first, grammar
    /// mask last" contract.
    private var constraintProcessor: (any LogitProcessor)?
    /// Per-instance override of `defaultPrefillChunkSize`. Production call sites use the default
    /// parameter value and get `defaultPrefillChunkSize`; tests use small values to exercise
    /// multi-chunk prefill without multi-thousand-token fixtures.
    private let prefillChunkSize: Int

    /// `setSampling` genuinely swaps in a `TopPSampler` honoring the requested distribution (see
    /// its doc comment) — this is a real opt-in, not a stub. See `Decoder.supportsSampling`'s doc
    /// comment for why the default must stay `false` and only conformers that verify this may
    /// override it.
    public var supportsSampling: Bool { true }
    /// `setPenalties` genuinely builds the vendored `PenaltyProcessor` via
    /// `GenerateParameters.processor()` and applies it before token selection (see its doc
    /// comment) — a real opt-in, not a stub.
    public var supportsPenalties: Bool { true }
    /// `setResponseFormatConstraint` genuinely composes the injected processor AFTER `processor`
    /// (penalties) inside `selectSampleAndAdvance` and surfaces a recorded mask violation as a real
    /// thrown error (see that method) — a real opt-in, not a stub.
    public var supportsResponseFormatConstraint: Bool { true }

    public init(
        model: any LanguageModel, cache: [KVCache],
        prefillChunkSize: Int = MLXDecoder.defaultPrefillChunkSize
    ) {
        self.model = model
        self.cacheFactory = { model.newCache(parameters: nil) }
        self.cache = cache
        self.prefillChunkSize = prefillChunkSize
    }

    /// Construct a decoder whose initial cache and every later reset come from one storage-bound
    /// factory. This is the load-bearing initializer for explicit quantized-KV serving.
    public init(
        model: any LanguageModel,
        cacheFactory: @escaping () -> [KVCache],
        prefillChunkSize: Int = MLXDecoder.defaultPrefillChunkSize
    ) {
        self.model = model
        self.cacheFactory = cacheFactory
        self.cache = cacheFactory()
        self.prefillChunkSize = prefillChunkSize
    }

    public mutating func prefill(_ promptTokens: [Int]) throws -> Int {
        let last = try prefillRawLogits(promptTokens)
        return try selectSampleAndAdvance(rawLogits: last)
    }

    /// Prefill the prompt through `selectSampleAndAdvance`'s SHARED token-selection code, then
    /// additionally compute the selected token's real logprob from the SAME raw (pre-processor)
    /// logits `prefill` itself would have used — see `LogprobDecoding`'s doc comment for why this
    /// must never diverge in what gets selected, only in what gets reported alongside it.
    public mutating func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        let raw = try prefillRawLogits(promptTokens)
        let token = try selectSampleAndAdvance(rawLogits: raw)
        return (token, Self.decodedTokenLogprob(rawLogits: raw, sampledToken: token, topN: topN))
    }

    /// Given the last token, produce the next plus its real logprob — the `step` sibling of
    /// `prefillWithLogprob`. Shares `selectSampleAndAdvance` with `step` for the same reason.
    public mutating func stepWithLogprob(
        last: Int, topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        guard let logits = pendingLogits else {
            fatalError("MLXDecoder.stepWithLogprob called before prefill")
        }
        let raw = logits[0..., -1, 0...]
        let token = try selectSampleAndAdvance(rawLogits: raw)
        return (token, Self.decodedTokenLogprob(rawLogits: raw, sampledToken: token, topN: topN))
    }

    /// Runs the chunked prefill forward and returns the FINAL chunk's last-position RAW logits
    /// (`[1, vocab]`, before any processor/penalty runs) — the shared first half of both `prefill`
    /// and `prefillWithLogprob`.
    private mutating func prefillRawLogits(_ promptTokens: [Int]) throws -> MLXArray {
        let promptArray = MLXArray(promptTokens)
        processor?.prompt(promptArray) // seed the penalty context with the prompt tokens (full prompt, unchanged)

        // Chunked prefill: run the prompt through `prefillChunkSize`-token forwards instead of
        // one whole-prompt forward (see the constant's doc comment for why, and
        // `CompiledMLXDecoder.prefillCore` for the established idiom this follows). Only the
        // FINAL chunk's logits feed sampling below; earlier chunks exist solely to advance the
        // cache. A prompt at or below `prefillChunkSize` runs exactly one loop iteration over
        // the whole prompt — byte-identical in control flow to the prior single forward.
        var lastChunkLogits: MLXArray!
        var start = 0
        while start < promptTokens.count {
            let end = min(start + prefillChunkSize, promptTokens.count)
            let chunkIds = MLXArray(Array(promptTokens[start..<end])).reshaped([1, end - start])
            // Throwing seam (see `LanguageModel.evaluateThrowing`): a model whose own step-time
            // validation fails (e.g. `qwen4_exp`'s cache/PLE-ownership checks) reports that as an
            // ordinary Swift error here instead of aborting the process via `preconditionFailure`
            // inside `callAsFunction`. Every other conformer takes the default implementation,
            // which forwards straight to `callAsFunction` — semantically identical to the direct
            // call this replaces.
            let chunkLogits = try model.evaluateThrowing(
                LMInput.Text(tokens: chunkIds), cache: cache, state: nil
            ).logits // [1, chunkLen, vocab]
            // Force this chunk's cache mutation to materialize before the next chunk reads the
            // in-graph offset and packed buffers, and drop the reference to any prior chunk's
            // logits so intermediate activations are not retained across the loop (mirrors
            // CompiledMLXDecoder.prefillCore).
            eval(chunkLogits)
            lastChunkLogits = chunkLogits
            start = end
        }

        // An empty prompt never enters the loop. Fail with the reason rather than on an implicit
        // nil-unwrap; callers already reject empty prompts upstream (`emptyRenderedPrompt`), so
        // reaching this is a caller bug worth naming. Mirrors CompiledMLXDecoder.prefillCore.
        guard let lastChunkLogits else {
            preconditionFailure("prefill requires at least one prompt token")
        }
        return lastChunkLogits[0..., -1, 0...] // [1, vocab] — final chunk only
    }

    /// Select (argmax or sampled), apply the pending penalty processor, submit the next forward
    /// (submit-first lookahead), and return the readback token — the code SHARED by `prefill`/
    /// `step` and their `*WithLogprob` siblings. `rawLogits` is `[1, vocab]`, BEFORE any processor
    /// runs; callers that also need a logprob capture this same array before calling in, so the
    /// logprob they report is guaranteed to describe the exact distribution this function selected
    /// from.
    private mutating func selectSampleAndAdvance(rawLogits: MLXArray) throws -> Int {
        let afterPenalties = processor?.process(logits: rawLogits) ?? rawLogits // penalties first
        // Grammar mask LAST, on the already-penalized logits — the response-format design's
        // ordering contract (see `constraintProcessor`'s doc comment).
        let processed = constraintProcessor?.process(logits: afterPenalties) ?? afterPenalties
        let next = sampler.sample(logits: processed) // [1] on GPU — argmax (greedy) or sampled
        processor?.didSample(token: next)

        // submit-first: kick the next forward BEFORE reading `next` back to the CPU for ANY
        // purpose — including `constraintProcessor?.didSample`'s own readback below. The vendored
        // `JSONObjectMaskingLogitProcessor.didSample` calls `token.item(Int.self)` to advance the
        // automaton; doing that HERE (before this forward is submitted) would stall the pipeline
        // exactly like calling `next.item(Int.self)` early would — the same "7.3x stall" this
        // decoder exists to avoid (see the type's own doc comment), just introduced via the
        // constraint processor instead of the return statement.
        let nextIds = next.reshaped([1, 1])
        let nextLogits = try model.evaluateThrowing(
            LMInput.Text(tokens: nextIds), cache: cache, state: nil
        ).logits
        asyncEval(nextLogits) // overlap GPU with the readback below
        pendingLogits = nextLogits

        // Constraint advance + failure check run AFTER the next forward is already in flight (see
        // above), so `didSample`'s host readback overlaps that forward instead of stalling ahead
        // of it. The vendored `LogitProcessor.didSample` cannot itself throw, so a constraint
        // violation (a sampled token the mask should have made impossible) is recorded on the
        // processor instance and turned into a real thrown error HERE — still before this call
        // returns a token to a caller that would otherwise treat it as legitimate output — rather
        // than let generation silently continue past a disallowed token. See
        // `ConstraintProcessorFailureReporting`'s doc comment. A `nil` `constraintProcessor` (every
        // request without `response_format`) takes neither branch here, so that path's behavior
        // and timing are unchanged.
        constraintProcessor?.didSample(token: next)
        if let reporter = constraintProcessor as? ConstraintProcessorFailureReporting,
            let failure = reporter.recordedFailure {
            throw failure
        }

        return next.item(Int.self) // readback overlaps the pending forward
    }

    public mutating func step(last: Int) throws -> Int {
        // pendingLogits already computed for the position after `last`
        guard let logits = pendingLogits else {
            fatalError("MLXDecoder.step called before prefill")
        }
        let lastLogits = logits[0..., -1, 0...]
        return try selectSampleAndAdvance(rawLogits: lastLogits)
    }

    /// Computes the REAL per-step logprob from the RAW (pre-processor) logits — see
    /// `LogprobDecoding`'s doc comment for the contract this mirrors one level up
    /// (`ServingCore.ServingTokenLogprob`). `rawLogits` is `[1, vocab]`, the SAME slice
    /// `selectSampleAndAdvance` was called with; `sampledToken` is the token IT ALREADY SELECTED —
    /// this function runs strictly AFTER that selection, purely to REPORT that choice's raw
    /// logprob, never to influence it.
    ///
    /// One `eval`, one small host transfer: only the sampled token's own logprob plus up to
    /// `topN` candidate (id, logprob) pairs ever leave the GPU — never the full vocabulary row.
    private static func decodedTokenLogprob(
        rawLogits: MLXArray, sampledToken: Int, topN: Int
    ) -> DecodedTokenLogprob {
        // float32 before logSoftmax: models may emit float16/bfloat16 logits, and a half-precision
        // log-softmax would report visibly rounded logprobs.
        let flat = rawLogits.reshaped([rawLogits.size]).asType(.float32) // [vocab]
        let vocab = flat.dim(0)
        let logp = logSoftmax(flat, axis: -1) // [vocab], log-probabilities under the raw distribution
        let n = max(0, min(topN, vocab))

        let sampledLogprob = logp[sampledToken]
        var topIDs: MLXArray?
        var topVals: MLXArray?
        if n > 0 {
            // `argPartition(kth: vocab - n)` puts the top `n` values (by index, unsorted) at
            // indices `[vocab - n, vocab)` — see its doc comment. Gathering just those `n`
            // indices/values, rather than sorting the whole vocabulary, is what keeps this a
            // small host transfer regardless of vocabulary size.
            let partitioned = argPartition(logp, kth: vocab - n, axis: -1)
            let ids = partitioned[(vocab - n)...]
            topIDs = ids
            topVals = logp[ids]
        }

        // ONE eval call for everything this function needs on the host, whether or not `topN > 0`.
        if let topVals, let topIDs {
            eval(sampledLogprob, topVals, topIDs)
        } else {
            eval(sampledLogprob)
        }

        let sampledValue = sampledLogprob.item(Float.self)
        var candidates: [DecodedTokenLogprobCandidate] = []
        if let topVals, let topIDs {
            let ids = topIDs.asArray(Int32.self)
            let values = topVals.asArray(Float.self)
            candidates =
                zip(ids, values)
                .map { DecodedTokenLogprobCandidate(tokenID: Int($0), logprob: $1) }
                .filter { $0.logprob.isFinite } // -infinity marks a masked-out vocabulary entry
                .sorted { $0.logprob > $1.logprob }
        }
        return DecodedTokenLogprob(tokenID: sampledToken, logprob: sampledValue, top: candidates)
    }

    /// Rebuild the selected KV cache family and drop any pending lookahead. The factory is already
    /// owned by this decoder, so this never crosses the actor boundary with a fresh non-Sendable
    /// model reference. Used between runs so each request starts empty without changing KV storage.
    public mutating func reset() {
        cache = cacheFactory()
        pendingLogits = nil
        sampler = ArgMaxSampler()
        processor = nil
        constraintProcessor = nil
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

    /// Configure a per-request `response_format: json_object` grammar mask for the next generation.
    /// `nil` clears any previously configured constraint (mirrors `setPenalties(nil-equivalent)`'s
    /// clearing behavior). See `constraintProcessor`'s doc comment for composition order.
    public mutating func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        constraintProcessor = constraint
    }
}
