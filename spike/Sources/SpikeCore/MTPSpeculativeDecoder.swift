import Foundation
import MLX
import MLXLMCommon

/// Errors specific to the speculative `Decoder` bridge. Distinct from `InferenceActorError` and
/// `MLXDecoder`'s bare `preconditionFailure`s because both failure modes here are "fail closed at
/// a specific, load-bearing boundary" rather than an ordinary model-evaluation error.
public enum MTPSpeculativeDecoderError: Error, Equatable, Sendable {
    /// Thrown at construction when `blockSize != MTPSpeculativeDecoder.servingBlockSize`. See the
    /// doc comment on `servingBlockSize` for why any other value is unsafe, not merely unsupported.
    case unsupportedBlockSize(Int)
    /// `MTPSpeculativeTokenIterator.nextThrowing()` returned `nil` (its budget, `maxTokens`, was
    /// reached — see `prefill`'s doc comment on why that budget is always `nil` here). A `nil`
    /// here must never be silently turned into a token or a fabricated EOS.
    case iteratorExhausted
}

/// Speculative `Decoder` conformer (sibling of `MLXDecoder`) that owns a target + drafter pair and
/// wraps `MTPSpeculativeTokenIterator`, per
/// `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md` ("Option A"). `InferenceActor`
/// `sending`-consumes this the same way it consumes `MLXDecoder` — the non-Sendable target and
/// drafter never leave the actor once constructed here.
///
/// Sampler divergence (documented per the decision table, not hidden): `MLXDecoder.setSampling`
/// always builds a `TopPSampler`. `GenerateParameters.sampler()` — the only entry point this
/// iterator's initializer accepts — returns a plain `CategoricalSampler` whenever top-p/top-k/
/// min-p are all inert (`Evaluate.swift`'s `sampler()`). Same seed, same temperature, but a
/// `CategoricalSampler` and a `TopPSampler` consume their RNG differently, so a `.sampled` request
/// run through this decoder can select different tokens than the same request run through
/// `MLXDecoder`, even though both are "correct" samples from the same distribution. The iterator
/// exposes no sampler-injection seam, so this divergence is unavoidable through this initializer.
public struct MTPSpeculativeDecoder: Decoder, SpeculativeTelemetryProviding {
    /// The only blockSize this bridge accepts. Pinned to the offloaded hybrid family's native
    /// rewind depth (`maximumNativeTargetCacheRewind == 2`): its gated-delta-net layers are not
    /// trimmable, so the iterator depends entirely on native rewind, and
    /// `canUseNativeSpeculativeRewind(..., tokenCount: blockSize - 1, maximumDepth: 2)` only holds
    /// for `blockSize <= 3`. A larger value fails at `MTPSpeculativeTokenIterator.init` for every
    /// request, not merely degrades — so this bridge rejects it at construction instead
    /// (blocker 2 in the decision doc).
    public static let servingBlockSize = 3

    private let target: any LanguageModel
    private let drafter: any MTPDrafterModel
    /// Rebuilds the cache family selected at load, mirroring `MLXDecoder.cacheFactory` — never the
    /// model's default cache, so a reset cannot silently change KV storage format.
    private let cacheFactory: () -> [KVCache]
    private let blockSize: Int

    private var iterator: MTPSpeculativeTokenIterator?
    private var lastReturnedToken: Int?

    /// Stashed by `setSampling`/`setPenalties`, consumed by `prefill` to build the
    /// `GenerateParameters` the iterator is constructed with. Never hardcoded to greedy: a
    /// `.sampled` request is not refused (the iterator handles it — sticky passthrough when
    /// `requiresGreedySampling` — merely unaccelerated, not incorrect).
    private var sampling: DecoderSampling = .greedy
    private var penalties: DecoderPenalties = .none

    /// Cumulative telemetry across every iterator this decoder has owned (i.e. survives `reset()`
    /// — see its doc comment). `passthroughReason` only ever moves from `nil` to a value; it is
    /// never cleared once observed, mirroring the iterator's own "sticky" semantics.
    private var accumulatedProposedCount = 0
    private var accumulatedAcceptedCount = 0
    /// See `SpeculativeTelemetrySnapshot.verifyRoundCount`'s doc comment: the iterator's own
    /// `speculativeDecodingTelemetry` is `nil` exactly when its `roundCount == 0` (a legitimate
    /// "no rounds run yet" state), so every read of it below is coalesced with `?? 0` rather than
    /// propagated as `nil` — a live iterator with zero rounds must read as `0`, not fall back to
    /// `stickyPassthroughReason`-style "unknown".
    private var accumulatedVerifyRoundCount = 0
    private var stickyPassthroughReason: String?

    public var speculativeTelemetrySnapshot: SpeculativeTelemetrySnapshot {
        SpeculativeTelemetrySnapshot(
            proposedCount: accumulatedProposedCount + (iterator?.proposedDraftTokens ?? 0),
            acceptedCount: accumulatedAcceptedCount + (iterator?.acceptedDraftTokens ?? 0),
            verifyRoundCount: accumulatedVerifyRoundCount
                + (iterator?.speculativeDecodingTelemetry?.roundCount ?? 0),
            passthroughReason: iterator?.passthroughReason ?? stickyPassthroughReason)
    }

    /// - Parameters:
    ///   - target: The main (verifying) model. Retained non-Sendable; must stay actor-confined.
    ///   - drafter: The MTP drafter proposing candidate blocks against `target`.
    ///   - cacheFactory: Builds a fresh target KV cache family for every `prefill`/`reset`.
    ///   - blockSize: MUST equal `servingBlockSize` (3) or this throws — see that constant's doc
    ///     comment. Defaulted so ordinary call sites don't need to name it.
    public init(
        target: any LanguageModel,
        drafter: any MTPDrafterModel,
        cacheFactory: @escaping () -> [KVCache],
        blockSize: Int = MTPSpeculativeDecoder.servingBlockSize
    ) throws {
        guard blockSize == MTPSpeculativeDecoder.servingBlockSize else {
            throw MTPSpeculativeDecoderError.unsupportedBlockSize(blockSize)
        }
        self.target = target
        self.drafter = drafter
        self.cacheFactory = cacheFactory
        self.blockSize = blockSize
    }

    /// Build the `GenerateParameters` the iterator is constructed with, from whatever `setSampling`
    /// / `setPenalties` most recently stashed. `maxTokens` is always `nil` — see `prefill`'s doc
    /// comment for why `InferenceActor` must be the sole budget authority.
    private func buildParameters() -> GenerateParameters {
        var parameters: GenerateParameters
        switch sampling {
        case .greedy:
            parameters = GenerateParameters(temperature: 0)
        case let .sampled(temperature, topP, topK, minP, seed):
            parameters = GenerateParameters(
                temperature: Float(temperature),
                topP: Float(topP),
                topK: topK ?? 0,
                minP: Float(minP ?? 0),
                seed: seed.map { UInt64(bitPattern: $0) })
        }
        parameters.repetitionPenalty = penalties.repetitionPenalty.map { Float($0) }
        parameters.presencePenalty = penalties.presencePenalty.map { Float($0) }
        parameters.frequencyPenalty = penalties.frequencyPenalty.map { Float($0) }
        // Otherwise this silently inherits `GenerateParameters`' own default of 512
        // (`Evaluate.swift:134`), while `MLXDecoder.defaultPrefillChunkSize` (2048) is what
        // `CapacityModel.predictPeakBytes`'s fit check prices AND what the scalar serving route
        // actually chunks at. A mismatch here means the two serving routes take a
        // monolithic-vs-chunked prefill path difference unrelated to speculation on any prompt
        // longer than whichever chunk size is smaller — a real geometry divergence this repo has
        // already measured (max|Δ| 0.546875 from sequence-chunking alone on real weights), not a
        // hypothetical one. One named source of truth; do not reintroduce a second `2048` literal.
        parameters.prefillStepSize = MLXDecoder.defaultPrefillChunkSize
        return parameters
    }

    /// Construct a fresh iterator over `promptTokens` and return its first token (the prepare-time
    /// bonus). `parameters.maxTokens` is always `nil`: `InferenceActor.run` (the `submit` path)
    /// calls `step` once more than it consumes, so a real budget here would hit the iterator's own
    /// `nil` exactly at that extra call, and — because a `nil` from `nextThrowing()` throws rather
    /// than fabricating a token — that mismatch is safe (it surfaces as `.iteratorExhausted`
    /// instead of a corrupted `.stop`/EOS report). `InferenceActor.generateBounded`'s own
    /// `maxTokens` loop is therefore the sole budget authority; this decoder never independently
    /// bounds generation.
    public mutating func prefill(_ promptTokens: [Int]) throws -> Int {
        let parameters = buildParameters()
        let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
        var newIterator = try MTPSpeculativeTokenIterator(
            input: input,
            mainModel: target,
            drafter: drafter,
            mainCache: cacheFactory(),
            parameters: parameters,
            blockSize: blockSize)
        guard let token = try newIterator.nextThrowing() else {
            throw MTPSpeculativeDecoderError.iteratorExhausted
        }
        iterator = newIterator
        lastReturnedToken = token
        return token
    }

    /// `last` is not consumed by the iterator (it advances from its own internal pending/round
    /// state), but the coupling `generateBounded` relies on — that it always feeds back exactly the
    /// token this decoder most recently returned (`InferenceActor.swift`'s main loop) — is real, so
    /// it is asserted here rather than left implicit.
    public mutating func step(last: Int) throws -> Int {
        assert(
            last == lastReturnedToken,
            "MTPSpeculativeDecoder.step(last:) was fed \(last), but this decoder last returned "
                + "\(String(describing: lastReturnedToken)) — generateBounded must feed back "
                + "exactly the token it was given.")
        guard var currentIterator = iterator else {
            fatalError("MTPSpeculativeDecoder.step called before prefill")
        }
        defer { iterator = currentIterator }
        guard let token = try currentIterator.nextThrowing() else {
            throw MTPSpeculativeDecoderError.iteratorExhausted
        }
        lastReturnedToken = token
        return token
    }

    /// Drop the iterator and its target KV cache so the next `prefill` starts fresh, restoring
    /// greedy/no-penalty defaults exactly like `MLXDecoder.reset()`. Snapshots this iterator's
    /// telemetry into the decoder-owned accumulators FIRST — otherwise it dies here along with the
    /// iterator, and `generateBounded`'s `defer { decoder.reset() }` would make every request's
    /// telemetry unobservable by the time a caller could read it.
    ///
    /// Deliberately does NOT call `iterator.finalizeGeneration()`: that method holds hard
    /// `precondition`s on the native-rewind path that would abort the process, and there is nothing
    /// left to finalize — the cache this iterator was tracking is being discarded in the same call.
    public mutating func reset() {
        if let iterator {
            accumulatedProposedCount += iterator.proposedDraftTokens
            accumulatedAcceptedCount += iterator.acceptedDraftTokens
            accumulatedVerifyRoundCount += iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            if let reason = iterator.passthroughReason {
                stickyPassthroughReason = reason
            }
        }
        iterator = nil
        lastReturnedToken = nil
        sampling = .greedy
        penalties = .none
    }

    /// Configure token selection for the NEXT `prefill`. Stashed, not applied immediately: the
    /// iterator only reads `GenerateParameters` at construction, which happens inside `prefill`.
    /// `.sampled` is never refused here — see this type's header doc comment on the sampler
    /// divergence this implies.
    public mutating func setSampling(_ sampling: DecoderSampling) {
        self.sampling = sampling
    }

    /// Configure logit penalties for the NEXT `prefill`. Stashed for the same reason as
    /// `setSampling`. Mirrors `MLXDecoder.setPenalties`'s use of `GenerateParameters` fields, but
    /// via mutation of the parameters this decoder itself builds rather than
    /// `GenerateParameters.processor()` directly, since the iterator owns processor construction.
    public mutating func setPenalties(_ penalties: DecoderPenalties) {
        self.penalties = penalties
    }
}
