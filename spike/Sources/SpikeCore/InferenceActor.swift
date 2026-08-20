import Foundation

/// Per-request token-selection policy handed to the decoder before a bounded generation.
///
/// `.greedy` is argmax (temperature 0 or unspecified) — the default, byte-identical to the
/// prior behavior. `.sampled` carries the resolved knobs the vendored `TopPSampler` honors
/// (temperature, top-p, top-k, min-p, seed). This is the runtime-side mirror of ServingCore's
/// `ServingSamplingPolicy`; the serving adapter bridges one to the other. Repetition/presence/
/// frequency penalties are modeled separately as `DecoderPenalties` (they compose with greedy or
/// sampled decode alike).
public enum DecoderSampling: Equatable, Sendable {
    case greedy
    case sampled(temperature: Double, topP: Double, topK: Int?, minP: Double?, seed: Int64?)
}

/// Per-request logit penalties, applied to the logits BEFORE token selection so they compose with
/// either greedy or sampled decode. `nil`/zero means no penalty (the default, unchanged behavior).
/// OpenAI-style `presence`/`frequency` in [-2, 2]; HF-style `repetition` > 0 (1.0 = none). Wired to
/// the vendored `PenaltyProcessor` via `GenerateParameters.processor()`.
public struct DecoderPenalties: Equatable, Sendable {
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var repetitionPenalty: Double?

    public static let none = DecoderPenalties()

    public init(
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
    }

    /// True when no penalty is requested — the decoder then uses no logit processor (byte-identical
    /// to the prior behavior). A zero penalty counts as "none" (matches the vendored `processor()`).
    public var isEmpty: Bool {
        (presencePenalty ?? 0) == 0 && (frequencyPenalty ?? 0) == 0 && (repetitionPenalty ?? 0) == 0
    }
}

/// Abstraction over "one decode step" so the actor's loop is testable without MLX.
public protocol Decoder {
    /// Prefill the prompt and return the first token id.
    mutating func prefill(_ promptTokens: [Int]) -> Int
    /// Given the last token, produce the next.
    mutating func step(last: Int) -> Int
    /// Discard any per-conversation state (e.g. KV cache) so the next `prefill` starts
    /// fresh, without reconstructing the decoder (and re-crossing the actor boundary with
    /// a fresh non-Sendable model reference — see MLXDecoder.reset()).
    mutating func reset()
    /// Configure token selection for the NEXT generation. The default is a no-op, so a
    /// decoder that only supports greedy decode (e.g. the compiled path) stays greedy and
    /// existing conformers need no change. A decoder that ignores a `.sampled` request must
    /// never be reached by one — the serving layer rejects sampling on unsupported routes
    /// rather than silently downgrading to greedy.
    mutating func setSampling(_ sampling: DecoderSampling)
    /// Configure logit penalties for the NEXT generation (applied before token selection). Default
    /// no-op, so decoders that don't support penalties are unchanged; the serving layer only routes
    /// penalized requests to a decoder that honors them.
    mutating func setPenalties(_ penalties: DecoderPenalties)
}

extension Decoder {
    public mutating func setSampling(_ sampling: DecoderSampling) {}
    public mutating func setPenalties(_ penalties: DecoderPenalties) {}
}

/// Test double: replays a fixed script.
public struct ScriptedDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    public init(script: [Int], eos: Int) { self.script = script; self.eos = eos }
    public mutating func prefill(_ p: [Int]) -> Int { defer { i += 1 }; return script[i] }
    public mutating func step(last: Int) -> Int { defer { i += 1 }; return script[i] }
    public mutating func reset() { i = 0 }
}

public enum InferenceActorError: Error, Equatable, Sendable {
    case emptyPrompt
    case generationAlreadyActive
    case invalidEndOfSequence
    case invalidMaximumTokens
    case invalidTokenID(Int)
}

public enum InferenceTokenDisposition: Equatable, Sendable {
    case continueGeneration
    case stopGeneration
}

public enum InferenceRunFinishReason: Equatable, Sendable {
    case consumerStop
    case endOfSequence
    case length
}

public struct InferenceRunSummary: Equatable, Sendable {
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    public let finishReason: InferenceRunFinishReason

    public init(
        promptTokenCount: Int,
        generatedTokenCount: Int,
        finishReason: InferenceRunFinishReason
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.finishReason = finishReason
    }
}

/// Single-owner actor: owns the decoder (and, transitively, all MLX state) and streams
/// generated token ids to callers without ever exposing MLX types across the actor boundary.
public actor InferenceActor {
    private var decoder: any Decoder
    private var boundedGenerationActive = false

    public init(decoder: sending any Decoder) { self.decoder = decoder }

    /// Discard per-conversation state (KV cache) so a subsequent `submit` starts fresh.
    /// Lets one actor/decoder/model be reused across bench runs without crossing the
    /// actor boundary again with a non-Sendable model reference (see `MLXDecoder`).
    public func resetForNewRun() throws {
        guard !boundedGenerationActive else {
            throw InferenceActorError.generationAlreadyActive
        }
        decoder.reset()
    }

    /// Non-blocking: returns a stream immediately; decode runs inside the actor.
    public func submit(promptTokens: [Int], maxTokens: Int, eos: Int = 2) -> AsyncThrowingStream<Int, Error> {
        guard !boundedGenerationActive else {
            return AsyncThrowingStream { continuation in
                continuation.finish(
                    throwing: InferenceActorError.generationAlreadyActive)
            }
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                self.run(promptTokens, maxTokens, eos, continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Generate one scalar request through a suspending consumer callback.
    ///
    /// Unlike `submit`, this path cannot run ahead into an unbounded `AsyncThrowingStream`:
    /// the decoder advances only after the consumer accepts the current token. The callback is
    /// therefore the production backpressure seam used by serving adapters. Actor reentrancy is
    /// explicit and fail-closed: while the callback is suspended, another bounded generation is
    /// rejected without touching decoder state.
    public func generateBounded(
        promptTokens: [Int],
        maxTokens: Int,
        eos: Int,
        sampling: DecoderSampling = .greedy,
        penalties: DecoderPenalties = .none,
        consume: @escaping @Sendable (Int) async throws -> InferenceTokenDisposition
    ) async throws -> InferenceRunSummary {
        try await generateBounded(
            promptTokens: promptTokens,
            maxTokens: maxTokens,
            stopTokenIDs: [eos],
            sampling: sampling,
            penalties: penalties,
            consume: consume)
    }

    /// Generate one scalar request and stop before publishing any configured stop token.
    public func generateBounded(
        promptTokens: [Int],
        maxTokens: Int,
        stopTokenIDs: Set<Int>,
        sampling: DecoderSampling = .greedy,
        penalties: DecoderPenalties = .none,
        consume: @escaping @Sendable (Int) async throws -> InferenceTokenDisposition
    ) async throws -> InferenceRunSummary {
        guard !promptTokens.isEmpty else {
            throw InferenceActorError.emptyPrompt
        }
        guard maxTokens > 0 else {
            throw InferenceActorError.invalidMaximumTokens
        }
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw InferenceActorError.invalidEndOfSequence
        }
        guard !boundedGenerationActive else {
            throw InferenceActorError.generationAlreadyActive
        }

        boundedGenerationActive = true
        decoder.reset()
        decoder.setSampling(sampling)
        decoder.setPenalties(penalties)
        defer {
            decoder.reset()
            decoder.setSampling(.greedy)
            decoder.setPenalties(.none)
            boundedGenerationActive = false
        }

        try Task.checkCancellation()
        var token = decoder.prefill(promptTokens)
        var generatedTokenCount = 0

        while true {
            try Task.checkCancellation()
            if stopTokenIDs.contains(token) {
                return InferenceRunSummary(
                    promptTokenCount: promptTokens.count,
                    generatedTokenCount: generatedTokenCount,
                    finishReason: .endOfSequence)
            }
            guard token >= 0 else {
                throw InferenceActorError.invalidTokenID(token)
            }

            generatedTokenCount += 1
            let disposition = try await consume(token)
            if disposition == .stopGeneration {
                return InferenceRunSummary(
                    promptTokenCount: promptTokens.count,
                    generatedTokenCount: generatedTokenCount,
                    finishReason: .consumerStop)
            }
            if generatedTokenCount == maxTokens {
                return InferenceRunSummary(
                    promptTokenCount: promptTokens.count,
                    generatedTokenCount: generatedTokenCount,
                    finishReason: .length)
            }

            try Task.checkCancellation()
            token = decoder.step(last: token)
        }
    }

    private func run(
        _ prompt: [Int], _ maxTokens: Int, _ eos: Int,
        _ cont: AsyncThrowingStream<Int, Error>.Continuation
    ) {
        guard !boundedGenerationActive else {
            cont.finish(
                throwing: InferenceActorError.generationAlreadyActive)
            return
        }
        var tok = decoder.prefill(prompt)
        var n = 0
        while n < maxTokens {
            if tok == eos { break }
            cont.yield(tok)
            n += 1
            if Task.isCancelled { break }
            tok = decoder.step(last: tok)
        }
        cont.finish()
    }
}
