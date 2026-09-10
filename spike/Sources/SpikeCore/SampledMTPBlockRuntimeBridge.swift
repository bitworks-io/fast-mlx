import CryptoKit
import Foundation
import HarnessCore
import MLX
import MLXLMCommon

public enum SampledMTPBlockRuntimeEntropyDomain: String, Sendable, Equatable, Hashable {
    case proposal
    case acceptance
    case residual
    case bonus
}

public typealias SampledMTPBlockRuntimeEntropy = @Sendable (
    SampledMTPBlockRuntimeEntropyDomain
) -> Double

public struct SampledMTPBlockRuntimeDrawPlan: Sendable, Equatable {
    public let proposalUniforms: [Double]
    public let acceptanceUniforms: [Double]
    public let terminalDraw: SampledMTPBlockTerminalDraw

    public init(
        proposalUniforms: [Double],
        acceptanceUniforms: [Double],
        terminalDraw: SampledMTPBlockTerminalDraw
    ) {
        self.proposalUniforms = proposalUniforms
        self.acceptanceUniforms = acceptanceUniforms
        self.terminalDraw = terminalDraw
    }
}

public enum SampledMTPBlockRuntimeBridgeError: Error, Sendable, Equatable {
    case missingPlan
    case proposalCaptureFailed
    case proposalCountMismatch
    case proposalTokenMismatch
    case targetRowCountMismatch
}

public enum SeededSampledMTPBlockRuntimeProviderError: Error, Sendable, Equatable {
    case proposalCaptureFailed
    case proposalCountMismatch(expected: Int, actual: Int)
    case proposalTokenMismatch(index: Int, expected: Int, actual: Int)
    case invalidProposalToken(token: Int, vocabularyCount: Int)
    case targetRowCountMismatch(expected: Int, actual: Int)
    case vocabularyWidthMismatch(expected: Int, actual: Int)
    case invalidLogits
}

public enum NondeterministicSampledMTPBlockRuntimeProviderError: Error, Sendable, Equatable {
    case proposalCaptureFailed
    case proposalCountMismatch(expected: Int, actual: Int)
    case proposalTokenMismatch(index: Int, expected: Int, actual: Int)
    case invalidProposalToken(token: Int, vocabularyCount: Int)
    case targetRowCountMismatch(expected: Int, actual: Int)
    case vocabularyWidthMismatch(expected: Int, actual: Int)
    case invalidEntropy(domain: SampledMTPBlockRuntimeEntropyDomain, value: Double)
    case invalidLogits
}

public struct SeededSampledMTPBlockRuntimeDrawTrace: Sendable, Equatable {
    public let blockIndex: Int
    public let proposalUniforms: [Double]
    public let acceptanceUniforms: [Double]
    public let terminalDraw: SampledMTPBlockTerminalDraw
    public let outputTokens: [Int]
    public let acceptedDraftCount: Int

    public var terminalUniform: Double {
        switch terminalDraw {
        case let .residual(value), let .bonus(value):
            return value
        }
    }

    public init(
        blockIndex: Int,
        proposalUniforms: [Double],
        acceptanceUniforms: [Double],
        terminalDraw: SampledMTPBlockTerminalDraw,
        outputTokens: [Int],
        acceptedDraftCount: Int
    ) {
        self.blockIndex = blockIndex
        self.proposalUniforms = proposalUniforms
        self.acceptanceUniforms = acceptanceUniforms
        self.terminalDraw = terminalDraw
        self.outputTokens = outputTokens
        self.acceptedDraftCount = acceptedDraftCount
    }
}

/// The four `GenerateParameters` fields that determine the *target*
/// distribution `p` a truncated-sampling request actually draws from:
/// temperature, top-p, top-k, and min-p.
///
/// Each sampled-MTP block runtime provider stores one of these at
/// construction and cross-checks it against the incoming request's own
/// truncation on every `supports(parameters:)` call (see
/// `sharedSampledMTPSupportsPredicate` and each provider's `supports`
/// below). A mismatch is refused outright, so a forgotten or wrong wiring
/// degrades to "no speculation" (the caller falls back to ordinary
/// sampling), never to "speculation that silently emits tokens drawn from
/// the wrong distribution."
///
/// Where that cross-check can and cannot fire is worth stating precisely,
/// because the two cases are not alike. On the serving path it CANNOT fire:
/// `MTPSpeculativeDecoder.prefill` derives the truncation from the very same
/// `GenerateParameters` value it hands the iterator, so the comparison is a
/// tautology there and the guarantee comes from that shared derivation
/// rather than from this check. It earns its keep for providers constructed
/// somewhere OTHER than the request they will serve -- the measurement CLIs
/// and tests -- where the two really can diverge, and where being refused is
/// how such a provider avoids reporting a measurement of a distribution it
/// was not actually sampling."
public struct SampledMTPSamplingTruncation: Sendable, Equatable {
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float

    public init(temperature: Float, topP: Float, topK: Int, minP: Float) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
    }

    /// Reads the four truncation fields off the request's own parameters, so
    /// a provider's stored truncation can be compared against whatever a
    /// given request actually asked for.
    public init(parameters: GenerateParameters) {
        self.init(
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP)
    }

    /// The identity truncation: `truncatedSamplingProbabilities` at these
    /// four values is exactly `softmax(logits, axis: -1)`, matching this
    /// file's pre-truncation behavior. This is also every provider's
    /// default, which keeps existing untruncated construction sites
    /// compiling unchanged.
    public static let untruncated = SampledMTPSamplingTruncation(
        temperature: 1, topP: 1, topK: 0, minP: 0)
}

/// The sampling-shape predicate shared by every sampled-MTP block runtime
/// provider's `supports(parameters:)`.
///
/// `temperature > 0 && temperature.isFinite` admits any finite positive
/// temperature, not only `temperature == 1`, because the exactness this
/// predicate exists to protect does not depend on temperature at all.
/// `truncatedSamplingProbabilities` (`Evaluate.swift:407-441`, vendored)
/// applies the top-p/min-p/top-k filter chain to `logSoftmax(logits)` FIRST
/// and only afterward computes `softmax(logprobs * (1 / temperature))`.
/// `GenerateParameters.sampler()` (`Evaluate.swift:157-170`) picks between
/// two branches, and BOTH land on that identical order:
///   - untruncated (`topP == 1 && topK == 0 && minP == 0`): `sampler()`
///     returns `CategoricalSampler(temperature:)`, which draws
///     `categorical(logits * (1 / temperature))` -- i.e. `softmax(logits *
///     (1 / temperature))`. At that same truncation,
///     `truncatedSamplingProbabilities` skips every filter and returns
///     `softmax(logSoftmax(logits) * (1 / temperature))`. These are the same
///     law: `logSoftmax` only subtracts a per-row constant (`log-sum-exp`),
///     and softmax is invariant to a constant additive shift of its input.
///     (Not bit-exact -- two rounding passes versus one; this file already
///     documents that same distinction at `normalizedProbabilities`.)
///   - truncated (`usesTopP || usesTopK || usesMinP`): `sampler()` returns
///     `TopPSampler(temperature:topP:topK:minP:)`, whose `sample(logits:)`
///     applies the identical top-p -> min-p -> top-k filter chain to
///     `logSoftmax(logits)` and then draws `categorical(logprobs * (1 /
///     temperature))` -- line-for-line the same order and the same final
///     `softmax(logprobs * (1 / temperature))` `truncatedSamplingProbabilities`
///     computes.
/// So for every temperature this predicate admits, the target law `p` this
/// file computes tracks `parameters.sampler()`'s actual draw law exactly,
/// not only at `temperature == 1`.
///
/// `temperature > 0` is what does the excluding, not a separate `!= 0`
/// check: it is false for `temperature == 0` (correctly -- `sampler()`
/// returns `ArgMaxSampler` there, and `truncatedSamplingProbabilities` would
/// divide by zero, i.e. `softmax(logprobs * (1 / 0))`) AND false for
/// `temperature == .nan` (`NaN > 0` is `false` under IEEE-754, so `NaN` is
/// refused by construction with no separate `.isNaN` check needed). Negative
/// temperatures are refused by the same `> 0` term. `.isFinite` excludes
/// `+infinity` (and would exclude `-infinity`, already excluded by `> 0`):
/// `1 / .infinity` is `0`, which would silently flatten `p` to a uniform
/// distribution over the filtered support rather than refusing a request
/// whose target law this predicate cannot state as "the request's actual
/// sampler," so admitting it would be exactly the kind of silent-wrong-law
/// case this predicate exists to prevent.
///
/// `topP`/`topK` accept the deployed thinking preset's truncated range
/// (rather than requiring the untruncated `topP == 1, topK == 0`); `topP >
/// 0` stays strict because the vendored `GenerateParameters.sampler()` maps
/// `topP == 0` to "no filter", and accepting `topP >= 0` here while that
/// helper treats `0` as no-filter would be a silent divergence between what
/// this predicate admits and what the target sampler actually does.
///
/// This predicate no longer excludes `presencePenalty`/`frequencyPenalty`/
/// non-neutral `repetitionPenalty`. Penalties are handled OUTSIDE this
/// provider now: `MTPSpeculativeTokenIterator` penalizes the target verify
/// rows and the bonus row from a scratch copy of the request's canonical
/// `LogitProcessor` -- advanced by the drafted tokens -- before handing them
/// to `decide(proposedTokens:targetLogits:bonusTargetLogits:)`. Those rows
/// have already been through `LogitProcessor.process`, so the target law `p`
/// this provider computes from them is exactly the penalized law
/// `parameters.sampler()` would draw from after `processor.process` on the
/// live decode path -- the same exactness argument this predicate already
/// relies on for temperature and truncation, just applied one layer
/// upstream of this file.
///
/// The draft law `q` this provider records is deliberately NOT penalized,
/// and must stay that way: speculative sampling is distribution-preserving
/// for ANY proposal law `q`, provided `q` is the law the drafter actually
/// sampled from and the residual `max(p - q, 0)` covers the gap. Penalizing
/// the draft is an acceptance-rate optimization, never an exactness
/// requirement -- the drafter has no seam to apply a penalty consistently
/// across a block boundary anyway (see the block-N-proposed-at-end-of-block-
/// (N-1) note on `MTPSpeculativeTokenIterator`'s `commitDrafterState` call).
///
/// This predicate alone does not decide whether a given provider supports a
/// given request: every call site also cross-checks its own stored
/// `SampledMTPSamplingTruncation` against
/// `SampledMTPSamplingTruncation(parameters:)`. See the doc comment on
/// `SampledMTPSamplingTruncation` for why that cross-check is required.
private func sharedSampledMTPSupportsPredicate(_ parameters: GenerateParameters) -> Bool {
    parameters.temperature > 0 && parameters.temperature.isFinite
        && parameters.topP > 0 && parameters.topP <= 1
        && parameters.topK >= 0
        && parameters.minP == 0
}

/// Default-off production-shaped sampled MTP provider. It mirrors the seeded
/// diagnostic provider's validation and commit-on-success behavior, but draws
/// proposal, acceptance, residual, and bonus uniforms from caller-supplied
/// runtime entropy. Serving/admission stays unchanged unless a caller
/// explicitly constructs and passes this provider to the iterator.
public final class NondeterministicSampledMTPBlockRuntimeProvider:
    SampledMTPBlockRuntimeDeciding
{
    private let entropy: SampledMTPBlockRuntimeEntropy
    private let recorder: NondeterministicSampledMTPProposalSampler
    private let truncation: SampledMTPSamplingTruncation
    private var blockIndex = 0

    public private(set) var drawTraces: [SeededSampledMTPBlockRuntimeDrawTrace] = []
    public var proposalSampler: any LogitSampler { recorder }

    public init(
        entropy: @escaping SampledMTPBlockRuntimeEntropy = { _ in
            Double.random(in: 0 ..< 1)
        },
        truncation: SampledMTPSamplingTruncation = .untruncated
    ) {
        self.entropy = entropy
        self.recorder = NondeterministicSampledMTPProposalSampler(entropy: entropy)
        self.truncation = truncation
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        sharedSampledMTPSupportsPredicate(parameters)
            && truncation == SampledMTPSamplingTruncation(parameters: parameters)
    }

    public func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        guard !recorder.failed else {
            throw NondeterministicSampledMTPBlockRuntimeProviderError.proposalCaptureFailed
        }
        let pendingCaptureCount = recorder.pendingCaptureCount
        guard proposedTokens.count == pendingCaptureCount else {
            throw NondeterministicSampledMTPBlockRuntimeProviderError.proposalCountMismatch(
                expected: pendingCaptureCount,
                actual: proposedTokens.count)
        }
        guard targetLogits.count == proposedTokens.count else {
            throw NondeterministicSampledMTPBlockRuntimeProviderError.targetRowCountMismatch(
                expected: proposedTokens.count,
                actual: targetLogits.count)
        }

        let proposals = try recorder.peek(count: proposedTokens.count)
        let vocabularyCount = try validateRuntimeVocabularyWidth(proposals: proposals)
        let bonusTargetDistribution = try runtimeNormalizedProbabilities(
            bonusTargetLogits, truncation: truncation)
        try validateRuntimeWidth(bonusTargetDistribution.count, expected: vocabularyCount)

        var steps = [SampledMTPBlockStep]()
        steps.reserveCapacity(proposedTokens.count)
        for (index, proposedToken) in proposedTokens.enumerated() {
            try validateRuntimeProposalToken(proposedToken, vocabularyCount: vocabularyCount)
            let proposal = proposals[index]
            try validateRuntimeProposalToken(proposal.token, vocabularyCount: vocabularyCount)
            guard proposal.token == proposedToken else {
                throw NondeterministicSampledMTPBlockRuntimeProviderError.proposalTokenMismatch(
                    index: index,
                    expected: proposal.token,
                    actual: proposedToken)
            }

            let targetDistribution = try runtimeNormalizedProbabilities(
                targetLogits[index], truncation: truncation)
            try validateRuntimeWidth(targetDistribution.count, expected: vocabularyCount)
            steps.append(SampledMTPBlockStep(
                targetDistribution: targetDistribution,
                draftDistribution: proposal.probabilities,
                proposedToken: proposedToken))
        }

        var acceptanceDraws = [Double]()
        acceptanceDraws.reserveCapacity(steps.count)
        var terminalDraw: SampledMTPBlockTerminalDraw?
        for step in steps {
            let acceptanceUniform = try drawUniform(domain: .acceptance)
            acceptanceDraws.append(acceptanceUniform)
            let acceptanceProbability = try SampledMTPResidualCorrection.acceptanceProbability(
                target: step.targetDistribution,
                draft: step.draftDistribution,
                proposedToken: step.proposedToken)
            if acceptanceUniform >= acceptanceProbability {
                terminalDraw = .residual(try drawUniform(domain: .residual))
                break
            }
        }
        if terminalDraw == nil {
            terminalDraw = .bonus(try drawUniform(domain: .bonus))
        }

        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: acceptanceDraws,
            terminalDraws: terminalDraw.map { [$0] } ?? [],
            bonusTargetDistribution: bonusTargetDistribution)
        recorder.commit(count: proposedTokens.count)
        drawTraces.append(SeededSampledMTPBlockRuntimeDrawTrace(
            blockIndex: blockIndex,
            proposalUniforms: proposals.map(\.uniform),
            acceptanceUniforms: acceptanceDraws,
            terminalDraw: terminalDraw!,
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount))
        blockIndex += 1
        return SampledMTPBlockRuntimeDecision(
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount)
    }

    private func drawUniform(
        domain: SampledMTPBlockRuntimeEntropyDomain
    ) throws -> Double {
        let uniform = entropy(domain)
        guard uniform.isFinite, (0 ..< 1).contains(uniform) else {
            throw NondeterministicSampledMTPBlockRuntimeProviderError.invalidEntropy(
                domain: domain,
                value: uniform)
        }
        return uniform
    }
}

/// Caller-seeded diagnostic provider for the default-off sampled MTP block
/// runtime seam. It intentionally supports only production-shaped unfiltered
/// temperature-one sampling with no penalties.
public final class SeededSampledMTPBlockRuntimeProvider: SampledMTPBlockRuntimeDeciding {
    private let recorder: SeededSampledMTPProposalSampler
    private var acceptanceUniforms: SeededSampledMTPUniformSource
    private var residualUniforms: SeededSampledMTPUniformSource
    private var bonusUniforms: SeededSampledMTPUniformSource
    private let truncation: SampledMTPSamplingTruncation
    private var blockIndex = 0

    public private(set) var drawTraces: [SeededSampledMTPBlockRuntimeDrawTrace] = []
    public var proposalSampler: any LogitSampler { recorder }

    /// Test-only visibility into the proposal sampler's retained
    /// full-vocabulary capture count. Deliberately not `private`, for the
    /// same reason as `categoricalSample` below: it is the only honest seam
    /// for the regression test covering `SeededSampledMTPProposalSampler
    /// .commit`'s capture-trimming fix to observe that retained memory
    /// actually stays bounded. `pendingCaptureCount` cannot serve this role
    /// -- it is a DIFFERENCE (`captures.count - consumedCaptureCount`), so
    /// it reads small even if the underlying `captures` array itself grows
    /// without bound. `internal` access keeps this out of the public API
    /// while staying reachable from `@testable import SpikeCore`.
    var retainedProposalCaptureCount: Int { recorder.retainedCaptureCount }

    public init(seed: UInt64, truncation: SampledMTPSamplingTruncation = .untruncated) {
        self.recorder = SeededSampledMTPProposalSampler(seed: seed)
        self.acceptanceUniforms = SeededSampledMTPUniformSource(
            seed: seed,
            domain: .acceptance)
        self.residualUniforms = SeededSampledMTPUniformSource(
            seed: seed,
            domain: .residual)
        self.bonusUniforms = SeededSampledMTPUniformSource(
            seed: seed,
            domain: .bonus)
        self.truncation = truncation
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        sharedSampledMTPSupportsPredicate(parameters)
            && truncation == SampledMTPSamplingTruncation(parameters: parameters)
    }

    public func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        guard !recorder.failed else {
            throw SeededSampledMTPBlockRuntimeProviderError.proposalCaptureFailed
        }
        let pendingCaptureCount = recorder.pendingCaptureCount
        guard proposedTokens.count == pendingCaptureCount else {
            throw SeededSampledMTPBlockRuntimeProviderError.proposalCountMismatch(
                expected: pendingCaptureCount,
                actual: proposedTokens.count)
        }
        guard targetLogits.count == proposedTokens.count else {
            throw SeededSampledMTPBlockRuntimeProviderError.targetRowCountMismatch(
                expected: proposedTokens.count,
                actual: targetLogits.count)
        }

        let proposals = try recorder.peek(count: proposedTokens.count)
        let vocabularyCount = try validateVocabularyWidth(proposals: proposals)
        let bonusTargetDistribution = try validatingNormalizedProbabilities(
            bonusTargetLogits, truncation: truncation)
        try validateWidth(bonusTargetDistribution.count, expected: vocabularyCount)

        var steps = [SampledMTPBlockStep]()
        steps.reserveCapacity(proposedTokens.count)
        for (index, proposedToken) in proposedTokens.enumerated() {
            try validateProposalToken(proposedToken, vocabularyCount: vocabularyCount)
            let proposal = proposals[index]
            try validateProposalToken(proposal.token, vocabularyCount: vocabularyCount)
            guard proposal.token == proposedToken else {
                throw SeededSampledMTPBlockRuntimeProviderError.proposalTokenMismatch(
                    index: index,
                    expected: proposal.token,
                    actual: proposedToken)
            }

            let targetDistribution = try validatingNormalizedProbabilities(
                targetLogits[index], truncation: truncation)
            try validateWidth(targetDistribution.count, expected: vocabularyCount)
            steps.append(SampledMTPBlockStep(
                targetDistribution: targetDistribution,
                draftDistribution: proposal.probabilities,
                proposedToken: proposedToken))
        }

        var acceptanceDraws = [Double]()
        acceptanceDraws.reserveCapacity(steps.count)
        var terminalDraw: SampledMTPBlockTerminalDraw?
        var nextAcceptanceUniforms = acceptanceUniforms
        var nextResidualUniforms = residualUniforms
        var nextBonusUniforms = bonusUniforms
        for step in steps {
            let acceptanceUniform = nextAcceptanceUniforms.next()
            acceptanceDraws.append(acceptanceUniform)
            let acceptanceProbability = try SampledMTPResidualCorrection.acceptanceProbability(
                target: step.targetDistribution,
                draft: step.draftDistribution,
                proposedToken: step.proposedToken)
            if acceptanceUniform >= acceptanceProbability {
                terminalDraw = .residual(nextResidualUniforms.next())
                break
            }
        }
        if terminalDraw == nil {
            terminalDraw = .bonus(nextBonusUniforms.next())
        }

        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: acceptanceDraws,
            terminalDraws: terminalDraw.map { [$0] } ?? [],
            bonusTargetDistribution: bonusTargetDistribution)
        recorder.commit(count: proposedTokens.count)
        acceptanceUniforms = nextAcceptanceUniforms
        residualUniforms = nextResidualUniforms
        bonusUniforms = nextBonusUniforms
        drawTraces.append(SeededSampledMTPBlockRuntimeDrawTrace(
            blockIndex: blockIndex,
            proposalUniforms: proposals.map(\.uniform),
            acceptanceUniforms: acceptanceDraws,
            terminalDraw: terminalDraw!,
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount))
        blockIndex += 1
        return SampledMTPBlockRuntimeDecision(
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount)
    }
}

/// Default-off adapter from the MLX iterator callback to HarnessCore's
/// accepted ordered probability-ratio/residual-correction contract.
///
/// This first bounded runtime seam intentionally accepts only unfiltered,
/// temperature-one sampling with no penalties. Serving does not construct it.
public final class SampledMTPBlockRuntimeBridge: SampledMTPBlockRuntimeDeciding {
    private let plans: [SampledMTPBlockRuntimeDrawPlan]
    private let recorder: FixedUniformProposalSampler
    private let truncation: SampledMTPSamplingTruncation
    private var planIndex = 0

    public var proposalSampler: any LogitSampler { recorder }

    public init(
        plans: [SampledMTPBlockRuntimeDrawPlan],
        truncation: SampledMTPSamplingTruncation = .untruncated
    ) {
        self.plans = plans
        self.recorder = FixedUniformProposalSampler(
            uniforms: plans.flatMap(\.proposalUniforms))
        self.truncation = truncation
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        !plans.isEmpty
            && sharedSampledMTPSupportsPredicate(parameters)
            && truncation == SampledMTPSamplingTruncation(parameters: parameters)
    }

    public func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        guard plans.indices.contains(planIndex) else {
            throw SampledMTPBlockRuntimeBridgeError.missingPlan
        }
        let plan = plans[planIndex]
        guard proposedTokens.count == plan.proposalUniforms.count else {
            throw SampledMTPBlockRuntimeBridgeError.proposalCountMismatch
        }
        guard targetLogits.count == proposedTokens.count else {
            throw SampledMTPBlockRuntimeBridgeError.targetRowCountMismatch
        }
        let proposals = try recorder.consume(count: proposedTokens.count)
        guard proposals.map(\.token) == proposedTokens else {
            throw SampledMTPBlockRuntimeBridgeError.proposalTokenMismatch
        }

        let steps = zip(zip(proposedTokens.indices, proposedTokens), proposals).map {
            indexed, proposal in
            let (index, token) = indexed
            return SampledMTPBlockStep(
                targetDistribution: normalizedProbabilities(
                    targetLogits[index], truncation: truncation),
                draftDistribution: proposal.probabilities,
                proposedToken: token,
            )
        }
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: plan.acceptanceUniforms,
            terminalDraws: [plan.terminalDraw],
            bonusTargetDistribution: normalizedProbabilities(
                bonusTargetLogits, truncation: truncation))
        planIndex += 1
        return SampledMTPBlockRuntimeDecision(
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount)
    }
}

private struct CapturedProposal {
    let token: Int
    let uniform: Double
    let probabilities: [Double]
}

private final class FixedUniformProposalSampler: LogitSampler {
    private let uniforms: [Double]
    private var uniformIndex = 0
    private var consumedCaptureCount = 0
    private var captures = [CapturedProposal]()
    private var failed = false

    init(uniforms: [Double]) {
        self.uniforms = uniforms
    }

    // DO NOT thread `truncation`/temperature into this `normalizedProbabilities(logits)` call.
    // This is the DRAFT distribution `q` -- the law the drafter itself actually sampled the
    // proposed token from -- and it must stay `.untruncated` (temperature 1, no filters)
    // regardless of what temperature the REQUEST asks for. `SampledMTPResidualCorrection`'s
    // accept/reject test and its residual correction (`(target - min(target, ratio*draft)) /
    // (1 - sum(min))`) are only valid when `q` is the distribution actually used to draw the
    // proposed token; if this call instead used the request's temperature, `q` would no longer
    // match the token that was actually drawn, silently breaking the distribution-preservation
    // guarantee that makes the accept/reject/residual scheme produce exactly `p` in expectation.
    // Only the TARGET distribution (`targetLogits`/`bonusTargetLogits`, below) truncates at the
    // request's temperature -- see `sharedSampledMTPSupportsPredicate`'s doc comment for why that
    // one tracks the request's `parameters.sampler()` law exactly at every admitted temperature.
    func sample(logits: MLXArray) -> MLXArray {
        guard uniforms.indices.contains(uniformIndex) else {
            failed = true
            return MLXArray([Int32(0)])
        }
        let uniform = uniforms[uniformIndex]
        uniformIndex += 1
        let probabilities = normalizedProbabilities(logits)
        guard uniform.isFinite, (0 ..< 1).contains(uniform),
            let token = categoricalSample(probabilities, uniform: uniform)
        else {
            failed = true
            return MLXArray([Int32(0)])
        }
        captures.append(CapturedProposal(
            token: token,
            uniform: uniform,
            probabilities: probabilities))
        return MLXArray([Int32(token)])
    }

    func consume(count: Int) throws -> [CapturedProposal] {
        guard !failed, count >= 0,
            consumedCaptureCount + count <= captures.count
        else {
            throw SampledMTPBlockRuntimeBridgeError.proposalCaptureFailed
        }
        let result = Array(captures[consumedCaptureCount ..< consumedCaptureCount + count])
        consumedCaptureCount += count
        // `CapturedProposal.probabilities` is a full-vocabulary `[Double]`
        // row (~1.2 MB at this model's ~151,936-wide vocabulary). Without
        // this trim, every consumed capture would sit in `captures` for the
        // life of this sampler, which is now reachable on the serving path
        // (`MTPSpeculativeDecoder.prefill` constructs one per request) --
        // a multi-thousand-block completion would otherwise retain gigabytes
        // of already-consumed rows for no reason. Drop the consumed prefix
        // immediately and reset the index rather than merely advancing it.
        // Do NOT "simplify" this back to a plain `consumedCaptureCount +=
        // count` index bump.
        if consumedCaptureCount > 0 {
            captures.removeFirst(consumedCaptureCount)
            consumedCaptureCount = 0
        }
        return result
    }
}

private final class SeededSampledMTPProposalSampler: LogitSampler {
    private var proposalUniforms: SeededSampledMTPUniformSource
    private var consumedCaptureCount = 0
    private var captures = [CapturedProposal]()
    private(set) var failed = false

    var pendingCaptureCount: Int {
        captures.count - consumedCaptureCount
    }

    /// The raw retained array length, as opposed to `pendingCaptureCount`'s
    /// difference -- see the doc comment on
    /// `SeededSampledMTPBlockRuntimeProvider.retainedProposalCaptureCount`
    /// for why the distinction matters for testing the trim fix.
    var retainedCaptureCount: Int { captures.count }

    init(seed: UInt64) {
        self.proposalUniforms = SeededSampledMTPUniformSource(seed: seed, domain: .proposal)
    }

    // DO NOT thread `truncation`/temperature into this call -- this is the draft distribution
    // `q`; see the pin comment on `FixedUniformProposalSampler.sample` above for why it must
    // stay `.untruncated` regardless of the request's temperature.
    func sample(logits: MLXArray) -> MLXArray {
        do {
            let probabilities = try validatingNormalizedProbabilities(logits)
            let uniform = proposalUniforms.next()
            guard let token = categoricalSample(probabilities, uniform: uniform) else {
                failed = true
                return MLXArray([Int32(0)])
            }
            captures.append(CapturedProposal(
                token: token,
                uniform: uniform,
                probabilities: probabilities))
            return MLXArray([Int32(token)])
        } catch {
            failed = true
            return MLXArray([Int32(0)])
        }
    }

    func peek(count: Int) throws -> [CapturedProposal] {
        guard !failed, count >= 0,
            consumedCaptureCount + count <= captures.count
        else {
            throw SeededSampledMTPBlockRuntimeProviderError.proposalCaptureFailed
        }
        return Array(captures[consumedCaptureCount ..< consumedCaptureCount + count])
    }

    func commit(count: Int) {
        consumedCaptureCount += count
        // `CapturedProposal.probabilities` is a full-vocabulary `[Double]`
        // row (~1.2 MB at this model's ~151,936-wide vocabulary). Without
        // this trim, every committed capture would sit in `captures` for
        // the life of this sampler, which is now reachable on the serving
        // path (`MTPSpeculativeDecoder.prefill` constructs one per
        // request) -- a multi-thousand-block completion would otherwise
        // retain gigabytes of already-committed rows for no reason. Drop
        // the committed prefix immediately and reset the index rather than
        // merely advancing it. Do NOT "simplify" this back to a plain
        // `consumedCaptureCount += count` index bump.
        if consumedCaptureCount > 0 {
            captures.removeFirst(consumedCaptureCount)
            consumedCaptureCount = 0
        }
    }
}

private final class NondeterministicSampledMTPProposalSampler: LogitSampler {
    private let entropy: SampledMTPBlockRuntimeEntropy
    private var consumedCaptureCount = 0
    private var captures = [CapturedProposal]()
    private(set) var failed = false

    var pendingCaptureCount: Int {
        captures.count - consumedCaptureCount
    }

    init(entropy: @escaping SampledMTPBlockRuntimeEntropy) {
        self.entropy = entropy
    }

    // DO NOT thread `truncation`/temperature into this call -- this is the draft distribution
    // `q`; see the pin comment on `FixedUniformProposalSampler.sample` above for why it must
    // stay `.untruncated` regardless of the request's temperature.
    func sample(logits: MLXArray) -> MLXArray {
        do {
            let probabilities = try runtimeNormalizedProbabilities(logits)
            let uniform = entropy(.proposal)
            guard uniform.isFinite, (0 ..< 1).contains(uniform),
                let token = categoricalSample(probabilities, uniform: uniform)
            else {
                failed = true
                return MLXArray([Int32(0)])
            }
            captures.append(CapturedProposal(
                token: token,
                uniform: uniform,
                probabilities: probabilities))
            return MLXArray([Int32(token)])
        } catch {
            failed = true
            return MLXArray([Int32(0)])
        }
    }

    func peek(count: Int) throws -> [CapturedProposal] {
        guard !failed, count >= 0,
            consumedCaptureCount + count <= captures.count
        else {
            throw NondeterministicSampledMTPBlockRuntimeProviderError.proposalCaptureFailed
        }
        return Array(captures[consumedCaptureCount ..< consumedCaptureCount + count])
    }

    func commit(count: Int) {
        consumedCaptureCount += count
        // `CapturedProposal.probabilities` is a full-vocabulary `[Double]`
        // row (~1.2 MB at this model's ~151,936-wide vocabulary). Without
        // this trim, every committed capture would sit in `captures` for
        // the life of this sampler, which is now reachable on the serving
        // path (`MTPSpeculativeDecoder.prefill` constructs one per
        // request) -- a multi-thousand-block completion would otherwise
        // retain gigabytes of already-committed rows for no reason. Drop
        // the committed prefix immediately and reset the index rather than
        // merely advancing it. Do NOT "simplify" this back to a plain
        // `consumedCaptureCount += count` index bump.
        if consumedCaptureCount > 0 {
            captures.removeFirst(consumedCaptureCount)
            consumedCaptureCount = 0
        }
    }
}

private enum SeededSampledMTPUniformDomain: UInt8 {
    case proposal = 1
    case acceptance = 2
    case residual = 3
    case bonus = 4

    var label: String {
        switch self {
        case .proposal:
            return "proposal"
        case .acceptance:
            return "acceptance"
        case .residual:
            return "residual"
        case .bonus:
            return "bonus"
        }
    }
}

private struct SeededSampledMTPUniformSource {
    private let seed: UInt64
    private let domain: SeededSampledMTPUniformDomain
    private var counter: UInt64 = 0

    init(seed: UInt64, domain: SeededSampledMTPUniformDomain) {
        self.seed = seed
        self.domain = domain
    }

    mutating func next() -> Double {
        defer { counter &+= 1 }
        var data = Data("fast-mlx.sampled-mtp.seeded-provider.v1".utf8)
        data.append(domain.rawValue)
        data.append(contentsOf: domain.label.utf8)
        data.appendBigEndian(seed)
        data.appendBigEndian(counter)
        let digest = SHA256.hash(data: data)
        var raw = UInt64(0)
        for byte in digest.prefix(8) {
            raw = (raw << 8) | UInt64(byte)
        }
        let mantissa = raw >> 11
        return (Double(mantissa) + 0.5) / 9_007_199_254_740_992.0
    }
}

private func normalizedProbabilities(
    _ logits: MLXArray,
    truncation: SampledMTPSamplingTruncation = .untruncated
) -> [Double] {
    // `truncatedSamplingProbabilities` requires `ndim >= 2` (it partitions
    // the last two dimensions for its top-k filter); reshape to a 2D array
    // first. Using `[-1, logits.dim(-1)]` rather than `[1, -1]` preserves
    // the row boundary for a `[B, V]` or `[T, V]` input: `[1, -1]` would
    // silently flatten such an input into ONE joint softmax over `B * V`
    // instead of one softmax per row, and the result would still come back
    // with the right total length, so the width validators elsewhere in
    // this file could not catch the mismatch. Every call site today passes
    // a single `[V]`-or-`[1, V]` row, so this is latent, not live -- kept
    // here as free insurance against a future multi-row call site.
    //
    // At `.untruncated` (temperature 1, topP 1, topK 0, minP 0) this is
    // mathematically identical to `softmax(logits, axis: -1)` (matching
    // this function's pre-truncation behavior), though NOT necessarily
    // bit-exact: `truncatedSamplingProbabilities` computes it as
    // `softmax(logSoftmax(x))`, two rounding passes rather than one.
    // An EMPTY logits row is a legitimate fail-closed input: callers feed one
    // deliberately to prove this path refuses rather than fabricates a
    // distribution, and the pre-truncation implementation returned `[]` for it
    // (its `softmax(...).flattened()` produced an empty array, which the
    // `sum > 0` guard below then rejected). `reshaped` cannot infer a dimension
    // from an empty array and raises an MLX FATAL error rather than throwing,
    // which kills the whole process instead of failing this one call closed.
    // So the empty case must be answered before any reshape, not after it.
    guard logits.size > 0 else { return [] }
    let row = logits.asType(.float32).reshaped([-1, logits.dim(-1)])
    let probabilities = truncatedSamplingProbabilities(
        logits: row,
        temperature: truncation.temperature,
        topP: truncation.topP,
        topK: truncation.topK,
        minP: truncation.minP
    ).flattened()
    eval(probabilities)
    let raw = probabilities.asArray(Float.self).map(Double.init)
    let sum = raw.reduce(0, +)
    guard sum.isFinite, sum > 0 else { return [] }
    var normalized = raw.map { $0 / sum }
    guard let largest = normalized.indices.max(by: {
        normalized[$0] < normalized[$1]
    }) else { return normalized }
    normalized[largest] += 1 - normalized.reduce(0, +)
    return normalized
}

private func validatingNormalizedProbabilities(
    _ logits: MLXArray,
    truncation: SampledMTPSamplingTruncation = .untruncated
) throws -> [Double] {
    let rawLogits = logits.flattened()
    eval(rawLogits)
    // A raw `-infinity` logit is legitimate: it is how a model expresses a
    // masked/unsupported token. Some multimodal checkpoints mask their
    // unsupported media-sentinel indices to `-Float.infinity` on every
    // forward, so this is the ordinary shape of a real logits row, not a
    // corrupt one, and `softmax` maps it to exactly `0.0`. A finiteness
    // check therefore belongs on the OUTPUT of normalization, never on the
    // raw logits going in.
    //
    // `NaN` and `+infinity` are NOT legitimate: both make `softmax` produce
    // `NaN`. They are rejected here explicitly and fail-fast rather than by
    // relying on `NaN` propagating through `softmax`'s sum. Note that the
    // post-normalization checks below (the per-element `isFinite && >= 0`
    // loop and the `abs(sum - 1) <= 1e-12` check) already catch them
    // independently, and are what still catches a row that genuinely
    // degenerates -- including an all-`-inf` row, whose softmax is `NaN`.
    // This guard is therefore a documented early-out, not the enforcing
    // gate: do NOT remove those checks on the belief that it covers them.
    guard rawLogits.asArray(Float.self).allSatisfy({ !$0.isNaN && $0 != Float.infinity }) else {
        throw SeededSampledMTPBlockRuntimeProviderError.invalidLogits
    }
    let distribution = normalizedProbabilities(logits, truncation: truncation)
    guard !distribution.isEmpty else {
        throw SeededSampledMTPBlockRuntimeProviderError.invalidLogits
    }
    for value in distribution {
        guard value.isFinite, value >= 0 else {
            throw SeededSampledMTPBlockRuntimeProviderError.invalidLogits
        }
    }
    let sum = distribution.reduce(0, +)
    guard sum.isFinite, abs(sum - 1) <= 1e-12 else {
        throw SeededSampledMTPBlockRuntimeProviderError.invalidLogits
    }
    return distribution
}

private func validateVocabularyWidth(proposals: [CapturedProposal]) throws -> Int {
    guard let first = proposals.first else { return 0 }
    let vocabularyCount = first.probabilities.count
    for proposal in proposals {
        try validateWidth(proposal.probabilities.count, expected: vocabularyCount)
    }
    return vocabularyCount
}

private func validateWidth(_ actual: Int, expected: Int) throws {
    guard actual == expected else {
        throw SeededSampledMTPBlockRuntimeProviderError.vocabularyWidthMismatch(
            expected: expected,
            actual: actual)
    }
}

private func validateProposalToken(_ token: Int, vocabularyCount: Int) throws {
    guard token >= 0, token < vocabularyCount else {
        throw SeededSampledMTPBlockRuntimeProviderError.invalidProposalToken(
            token: token,
            vocabularyCount: vocabularyCount)
    }
}

private func runtimeNormalizedProbabilities(
    _ logits: MLXArray,
    truncation: SampledMTPSamplingTruncation = .untruncated
) throws -> [Double] {
    do {
        return try validatingNormalizedProbabilities(logits, truncation: truncation)
    } catch {
        throw NondeterministicSampledMTPBlockRuntimeProviderError.invalidLogits
    }
}

private func validateRuntimeVocabularyWidth(proposals: [CapturedProposal]) throws -> Int {
    guard let first = proposals.first else { return 0 }
    let vocabularyCount = first.probabilities.count
    for proposal in proposals {
        try validateRuntimeWidth(proposal.probabilities.count, expected: vocabularyCount)
    }
    return vocabularyCount
}

private func validateRuntimeWidth(_ actual: Int, expected: Int) throws {
    guard actual == expected else {
        throw NondeterministicSampledMTPBlockRuntimeProviderError.vocabularyWidthMismatch(
            expected: expected,
            actual: actual)
    }
}

private func validateRuntimeProposalToken(_ token: Int, vocabularyCount: Int) throws {
    guard token >= 0, token < vocabularyCount else {
        throw NondeterministicSampledMTPBlockRuntimeProviderError.invalidProposalToken(
            token: token,
            vocabularyCount: vocabularyCount)
    }
}

// Deliberately not `private`: the rounding-fallthrough branch below is
// unreachable through this file's public surface once a distribution has
// passed through `normalizedProbabilities`'s largest-bin correction (its
// cumulative sum comes back bit-exact `1.0` in practice), so the only
// honest way to cover the branch this file's fix actually changed is a
// direct call with a hand-built distribution and uniform. `internal` access
// keeps it out of the public API while staying reachable from
// `@testable import SpikeCore`.
func categoricalSample(_ probabilities: [Double], uniform: Double) -> Int? {
    guard !probabilities.isEmpty else { return nil }
    var cumulative = 0.0
    var lastSupportedToken: Int?
    for (index, probability) in probabilities.enumerated() {
        if probability > 0 {
            lastSupportedToken = index
        }
        cumulative += probability
        if uniform < cumulative { return index }
    }
    // Rounding fallthrough: floating-point summation left the cumulative
    // sum just short of `uniform` even though probabilities are normalized
    // to (approximately) 1. This fix aligns `categoricalSample` with
    // `SampledMTPResidualCorrection.sample(distribution:uniform:)`, which
    // already returns the last index with POSITIVE probability rather than
    // `probabilities.indices.last` for exactly this reason.
    //
    // All three call sites in this file sample the DRAFT distribution,
    // which is deliberately left `.untruncated`, so today this function is
    // never actually invoked on a truncated distribution with exact-zero
    // trailing entries. The scenario in which the distinction currently
    // matters is a model whose logits carry `-infinity` at masked indices
    // (this repo has a recorded fact that this model's logits do exactly
    // that at media-sentinel indices on every forward): those indices are
    // exact `0.0` after softmax, so the same rounding fallthrough could
    // still hand back a zero-mass token from an otherwise-untruncated
    // distribution. That token would fall outside `supp(q')`,
    // `acceptanceProbability` would throw `zeroDraftMass`, and the caller
    // would degrade to sticky passthrough for the rest of the stream: a
    // one-in-a-billion rounding artifact silently converted into a
    // permanent throughput cliff. Returning the last index with POSITIVE
    // probability instead keeps the fallback inside the support of the
    // distribution actually being sampled from.
    // Do not revert this to `probabilities.indices.last`.
    return lastSupportedToken
}

private extension Data {
    mutating func appendBigEndian(_ value: UInt64) {
        var bigEndian = value.bigEndian
        Swift.withUnsafeBytes(of: &bigEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
}
