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
    /// `repetitionPenalty` also treats `1` as "none": the vendored `RepetitionContext` is a
    /// MULTIPLICATIVE penalty (`x < 0 ? x * penalty : x / penalty`), so `1` -- not `0` -- is its
    /// neutral element, and both HF-recommended presets for the deployed model send
    /// `repetition_penalty: 1.0`.
    public var isEmpty: Bool {
        (presencePenalty ?? 0) == 0 && (frequencyPenalty ?? 0) == 0
            && DecoderPenalties.repetitionPenaltyIsNeutral(repetitionPenalty)
    }

    /// Duplicated (rather than shared) against `GenerateParameters.repetitionPenaltyIsNeutral`:
    /// this type stores `Double` while the vendored parameters use `Float`, and this file doesn't
    /// otherwise import `MLXLMCommon`. Same invariant, same neutral set: `{nil, 0, 1}`.
    public static func repetitionPenaltyIsNeutral(_ repetitionPenalty: Double?) -> Bool {
        repetitionPenalty == nil || repetitionPenalty == 0 || repetitionPenalty == 1
    }
}

/// Abstraction over "one decode step" so the actor's loop is testable without MLX.
public protocol Decoder {
    /// Prefill the prompt and return the first token id. Throws when model evaluation itself
    /// fails validation (e.g. `qwen4_exp`'s cache/PLE-ownership checks) — a genuine caller bug
    /// (empty prompt) still aborts via `preconditionFailure`, never via this throw.
    mutating func prefill(_ promptTokens: [Int]) throws -> Int
    /// Given the last token, produce the next. Throws for the same reason as `prefill`; calling
    /// `step` before `prefill` remains a caller bug and still aborts via `fatalError`.
    mutating func step(last: Int) throws -> Int
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

/// Point-in-time speculative-decoding counters. `Sendable`/`Equatable` so a snapshot can cross
/// the actor boundary and be compared directly in tests. `passthroughReason` is `nil` when the
/// decoder speculated for the entire observed lifetime (across resets — see
/// `MTPSpeculativeDecoder.reset()`), and set once sticky passthrough engages.
public struct SpeculativeTelemetrySnapshot: Equatable, Sendable {
    public let proposedCount: Int
    public let acceptedCount: Int
    /// Number of speculative verify rounds run (one `speculateRound()` call each, regardless of
    /// how many draft tokens that round accepted — including a round that accepted zero). Sourced
    /// from `MTPSpeculativeTokenIterator.speculativeDecodingTelemetry?.roundCount`, which is `nil`
    /// exactly when `roundCount == 0` (a legitimate "no rounds yet" state, not an error) — callers
    /// populating this field must coalesce that `nil` to `0`, never let it collapse this whole
    /// snapshot to absent.
    public let verifyRoundCount: Int
    public let passthroughReason: String?
    /// Per-request count of completed requests that genuinely speculated this serve. In contrast
    /// to `passthroughReason` (which only ever moves from `nil` to a value and then LATCHES
    /// forever, per `MTPSpeculativeDecoder.reset()`'s doc comment), this is a plain running total
    /// that keeps incrementing for every request that speculates, including ones after
    /// `passthroughReason` has already gone non-nil on an earlier request. Defaulted to `0` so
    /// existing call sites/conformers predating this field keep compiling unchanged.
    public let speculativeRequestCount: Int
    /// Per-request count of completed requests that passed through this serve. Sibling of
    /// `speculativeRequestCount`; the pair gives a caller a passthrough RATE
    /// (`passthroughRequestCount / (speculativeRequestCount + passthroughRequestCount)`), which the
    /// sticky `passthroughReason`/gauge alone cannot express. Defaulted to `0` for the same
    /// compile-compatibility reason as `speculativeRequestCount`.
    public let passthroughRequestCount: Int

    public init(
        proposedCount: Int, acceptedCount: Int, verifyRoundCount: Int, passthroughReason: String?,
        speculativeRequestCount: Int = 0, passthroughRequestCount: Int = 0
    ) {
        self.proposedCount = proposedCount
        self.acceptedCount = acceptedCount
        self.verifyRoundCount = verifyRoundCount
        self.passthroughReason = passthroughReason
        self.speculativeRequestCount = speculativeRequestCount
        self.passthroughRequestCount = passthroughRequestCount
    }
}

/// Optional capability a `Decoder` may add to expose speculative-decoding telemetry. Kept
/// separate from `Decoder` itself (rather than widening every conformer) because most decoders
/// (`MLXDecoder`, `ScriptedDecoder`, ...) have no speculative counters to report.
public protocol SpeculativeTelemetryProviding {
    /// Cumulative counters, safe to read at any time — including after the owning decoder's
    /// per-request state (e.g. its `MTPSpeculativeTokenIterator`) has been torn down by `reset()`.
    var speculativeTelemetrySnapshot: SpeculativeTelemetrySnapshot { get }

    /// The CURRENT request's own passthrough reason, with NO sticky/cumulative fallback — `nil`
    /// whenever the live per-request state (e.g. the current `MTPSpeculativeTokenIterator`) is
    /// itself not in passthrough, even if an earlier request on this same decoder was. Distinct
    /// from `speculativeTelemetrySnapshot.passthroughReason`, which is deliberately sticky/
    /// cumulative for its existing callers (see `MTPSpeculativeDecoder`'s doc comments) — this
    /// accessor exists so `InferenceActor.runSummary` can report a genuinely per-request reason
    /// without changing that cumulative snapshot's meaning. Defaulted to `nil` so an existing or
    /// future conformer with no such notion (or none at all) keeps compiling unchanged.
    var currentRequestPassthroughReason: String? { get }
}

extension SpeculativeTelemetryProviding {
    public var currentRequestPassthroughReason: String? { nil }
}

/// Test double: replays a fixed script.
public struct ScriptedDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    public init(script: [Int], eos: Int) { self.script = script; self.eos = eos }
    public mutating func prefill(_ p: [Int]) -> Int { defer { i += 1 }; return script[i] }
    public mutating func step(last: Int) -> Int { defer { i += 1 }; return script[i] }
    // Non-throwing overrides satisfy the `Decoder` protocol's throwing requirements: a script
    // replay never fails model evaluation, so this conformer stays byte-identical to before.
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

/// Per-request speculative-decoding telemetry, computed as a DELTA between the decoder's
/// cumulative `SpeculativeTelemetrySnapshot` sampled immediately before this request's generation
/// began and immediately after it ended. The underlying counters are cumulative across the
/// decoder's whole life and deliberately SURVIVE `reset()` (see `MTPSpeculativeDecoder.reset()`),
/// so without this delta a caller reading `InferenceActor.speculativeTelemetry()` after N requests
/// would see the lifetime total, not this request's contribution to it.
///
/// `InferenceRunSummary.speculativeDelta` is `nil` when the decoder is not speculative at all
/// (does not conform to `SpeculativeTelemetryProviding`) — distinct from a speculative decoder
/// that proposed and accepted exactly zero draft tokens this request, which is a legitimate,
/// meaningful outcome represented by a non-nil value with `proposedDraftTokens == 0`.
public struct InferenceRunSpeculativeDelta: Equatable, Sendable {
    public let proposedDraftTokens: Int
    public let acceptedDraftTokens: Int
    /// The decoder's sticky passthrough reason as of the END of this request — not itself a delta.
    /// `passthroughReason` only ever moves from `nil` to a value and then stays there (see
    /// `SpeculativeTelemetrySnapshot`'s doc comment), so a non-nil value here means passthrough was
    /// in effect by the time this request finished, whether it engaged during this request or an
    /// earlier one.
    public let passthroughReason: String?

    public init(proposedDraftTokens: Int, acceptedDraftTokens: Int, passthroughReason: String?) {
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.passthroughReason = passthroughReason
    }
}

public struct InferenceRunSummary: Equatable, Sendable {
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    public let finishReason: InferenceRunFinishReason
    /// `nil` for a non-speculative decoder; see `InferenceRunSpeculativeDelta`'s doc comment for
    /// why absent and zero must stay distinguishable. Defaults to `nil` so existing call sites
    /// that predate this field keep compiling unchanged.
    public let speculativeDelta: InferenceRunSpeculativeDelta?

    public init(
        promptTokenCount: Int,
        generatedTokenCount: Int,
        finishReason: InferenceRunFinishReason,
        speculativeDelta: InferenceRunSpeculativeDelta? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.finishReason = finishReason
        self.speculativeDelta = speculativeDelta
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

    /// Actor-isolated read of the current decoder's speculative-decoding telemetry, or `nil` when
    /// the decoder does not conform to `SpeculativeTelemetryProviding` (e.g. `MLXDecoder`,
    /// `ScriptedDecoder`). This is the only sanctioned way to read those counters: the decoder
    /// (and any non-Sendable model it owns) never leaves the actor, so this method — not a direct
    /// reference to the decoder or its iterator — is what a caller or test awaits.
    public func speculativeTelemetry() -> SpeculativeTelemetrySnapshot? {
        (decoder as? SpeculativeTelemetryProviding)?.speculativeTelemetrySnapshot
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

        // Sampled BEFORE this request's `prefill`, so the delta computed in `runSummary` below
        // reflects only what THIS request contributed to the decoder's cumulative counters — not
        // the lifetime total. `nil` here (a non-speculative decoder) propagates straight through
        // to `InferenceRunSummary.speculativeDelta == nil`.
        let telemetryBefore = (decoder as? SpeculativeTelemetryProviding)?.speculativeTelemetrySnapshot

        // Single computed helper shared by every non-throwing exit below, so a future new return
        // point cannot silently omit the telemetry the way past bugs in this codebase have shipped
        // a field that was silently absent on one path among several.
        func runSummary(
            generatedTokenCount: Int, finishReason: InferenceRunFinishReason
        ) -> InferenceRunSummary {
            let speculativeDelta: InferenceRunSpeculativeDelta?
            if let telemetryBefore,
                let telemetryAfter =
                    (decoder as? SpeculativeTelemetryProviding)?.speculativeTelemetrySnapshot {
                // `passthroughReason` is read from the CURRENT request's own live state, NOT from
                // `telemetryAfter` (the cumulative, sticky snapshot) — see
                // `docs/task-inbox/2026-09-08-mtp-passthrough-reason-sticky-leak.md`. Reading
                // `telemetryAfter.passthroughReason` here would report the decoder's LIFETIME
                // sticky reason, mislabeling every later speculating request once any earlier
                // request on this same (load-once, reused) decoder had passed through even once.
                // A naive `telemetryAfter != telemetryBefore` delta is also wrong: two consecutive
                // passthrough requests with the SAME reason would compare equal and the second
                // would be falsely reported as speculating.
                speculativeDelta = InferenceRunSpeculativeDelta(
                    proposedDraftTokens: telemetryAfter.proposedCount - telemetryBefore.proposedCount,
                    acceptedDraftTokens: telemetryAfter.acceptedCount - telemetryBefore.acceptedCount,
                    passthroughReason: (decoder as? SpeculativeTelemetryProviding)?
                        .currentRequestPassthroughReason)
            } else {
                speculativeDelta = nil
            }
            return InferenceRunSummary(
                promptTokenCount: promptTokens.count,
                generatedTokenCount: generatedTokenCount,
                finishReason: finishReason,
                speculativeDelta: speculativeDelta)
        }

        try Task.checkCancellation()
        var token = try decoder.prefill(promptTokens)
        var generatedTokenCount = 0

        while true {
            try Task.checkCancellation()
            if stopTokenIDs.contains(token) {
                return runSummary(
                    generatedTokenCount: generatedTokenCount, finishReason: .endOfSequence)
            }
            guard token >= 0 else {
                throw InferenceActorError.invalidTokenID(token)
            }

            generatedTokenCount += 1
            let disposition = try await consume(token)
            if disposition == .stopGeneration {
                return runSummary(
                    generatedTokenCount: generatedTokenCount, finishReason: .consumerStop)
            }
            if generatedTokenCount == maxTokens {
                return runSummary(
                    generatedTokenCount: generatedTokenCount, finishReason: .length)
            }

            try Task.checkCancellation()
            token = try decoder.step(last: token)
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
        // `decoder.prefill`/`.step` can now throw when model evaluation itself fails validation
        // (e.g. `qwen4_exp`'s cache/PLE-ownership checks). This path predates `AsyncThrowingStream`
        // adoption of a suspending consumer; report the failure the same way every other error on
        // this stream is reported — finish the continuation with the thrown error — rather than
        // letting it propagate out of a non-throwing function and abort the process.
        do {
            var tok = try decoder.prefill(prompt)
            var n = 0
            while n < maxTokens {
                if tok == eos { break }
                cont.yield(tok)
                n += 1
                if Task.isCancelled { break }
                tok = try decoder.step(last: tok)
            }
            cont.finish()
        } catch {
            cont.finish(throwing: error)
        }
    }
}
