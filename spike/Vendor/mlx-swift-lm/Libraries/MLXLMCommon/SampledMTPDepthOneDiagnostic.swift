// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Branch forced by the Release-only sampled-MTP diagnostic. This is a model-backed
/// transaction probe, not a serving sampler; ordinary generation never calls it.
public enum SampledMTPDepthOneDiagnosticBranch: String, Sendable, Equatable {
    case accepted
    case rejected
}

public enum SampledMTPDepthOneDiagnosticError: Error, Sendable, Equatable {
    case targetDoesNotExposePromptHiddenState
    case drafterIsNotStateful
    case promptPreparationUnavailable
    case promptPreparationDidNotReturnLogits
    case missingProposal
    case distributionsMatch
    case recurrentCheckpointUnavailable
    case recurrentRewindFailed
}

/// Complete, live inputs and finalized caches for one depth-one probability-ratio
/// transaction. The caller owns evidence serialization and byte-level cache comparison.
public struct SampledMTPDepthOneDiagnosticResult {
    public let branch: SampledMTPDepthOneDiagnosticBranch
    public let promptTokenCount: Int
    public let bonusToken: Int
    public let proposedToken: Int
    public let emittedToken: Int
    public let proposalUniform: Double
    public let acceptanceUniform: Double
    public let residualUniform: Double?
    public let targetProbabilities: [Double]
    public let draftProbabilities: [Double]
    public let candidateCache: [KVCache]
    public let scalarCache: [KVCache]

    public init(
        branch: SampledMTPDepthOneDiagnosticBranch,
        promptTokenCount: Int,
        bonusToken: Int,
        proposedToken: Int,
        emittedToken: Int,
        proposalUniform: Double,
        acceptanceUniform: Double,
        residualUniform: Double?,
        targetProbabilities: [Double],
        draftProbabilities: [Double],
        candidateCache: [KVCache],
        scalarCache: [KVCache]
    ) {
        self.branch = branch
        self.promptTokenCount = promptTokenCount
        self.bonusToken = bonusToken
        self.proposedToken = proposedToken
        self.emittedToken = emittedToken
        self.proposalUniform = proposalUniform
        self.acceptanceUniform = acceptanceUniform
        self.residualUniform = residualUniform
        self.targetProbabilities = targetProbabilities
        self.draftProbabilities = draftProbabilities
        self.candidateCache = candidateCache
        self.scalarCache = scalarCache
    }
}

/// Run one authenticated caller-owned target/drafter pair through both the real Qwen
/// prompt-hidden priming path and the real hybrid-cache transaction machinery.
///
/// The accepted lane uses a fixed proposal draw and an acceptance draw of zero. The
/// rejected lane deterministically selects a positive `q - p` token, derives a proposal
/// draw from that token's q-CDF interval, uses an acceptance draw immediately below one,
/// and samples the residual with a fixed draw. The pure project gate independently
/// reconstructs every choice from the returned distributions and draws.
public func runSampledMTPDepthOneDiagnostic(
    input: LMInput,
    target: any LanguageModel,
    drafter: any MTPDrafterModel,
    branch: SampledMTPDepthOneDiagnosticBranch
) throws -> SampledMTPDepthOneDiagnosticResult {
    guard let promptTarget = target as? any MTPPromptHiddenStatePreparingModel else {
        throw SampledMTPDepthOneDiagnosticError.targetDoesNotExposePromptHiddenState
    }
    guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
        throw SampledMTPDepthOneDiagnosticError.drafterIsNotStateful
    }

    let candidateCache = target.newCache(parameters: nil)
    guard let preparation = try promptTarget.prepareForMTP(
        input,
        cache: candidateCache,
        windowSize: 512)
    else {
        throw SampledMTPDepthOneDiagnosticError.promptPreparationUnavailable
    }
    guard case .logits(let prefillOutput) = preparation.result else {
        throw SampledMTPDepthOneDiagnosticError.promptPreparationDidNotReturnLogits
    }

    let bonus = argMax(prefillOutput.logits[0..., -1, 0...], axis: -1)
    eval(bonus)
    let bonusToken = bonus.item(Int.self)
    var bonusState = prefillOutput.state ?? LMOutput.State()
    bonusState[mtpEmitFlagKey] = true
    let bonusOutput = target(
        LMInput.Text(tokens: bonus)[text: .newAxis],
        cache: candidateCache,
        state: bonusState)
    let targetProbabilities = normalizedProbabilities(
        bonusOutput.logits[0..., -1, 0...])

    let sampler = DiagnosticProposalSampler(
        targetProbabilities: targetProbabilities,
        branch: branch)
    var drafterState = statefulDrafter.makeState(parameters: nil)
    statefulDrafter.prepareDrafterState(
        target: target,
        promptTokens: input.text.tokens,
        targetHidden: preparation.targetHidden,
        firstBonus: bonus,
        positionDeltas: prefillOutput.state?[mtpPositionDeltasKey],
        state: &drafterState,
        sampler: sampler)
    guard let proposal = drafterState.seedToken else {
        throw SampledMTPDepthOneDiagnosticError.missingProposal
    }
    eval(proposal)
    guard let draftProbabilities = sampler.probabilities,
        let proposalUniform = sampler.proposalUniform
    else {
        throw sampler.failure ?? SampledMTPDepthOneDiagnosticError.missingProposal
    }
    let proposedToken = proposal.item(Int.self)

    let acceptanceUniform: Double
    let residualUniform: Double?
    let emittedToken: Int
    switch branch {
    case .accepted:
        acceptanceUniform = 0
        residualUniform = nil
        emittedToken = proposedToken
        _ = target(
            LMInput.Text(tokens: proposal.flattened())[text: .newAxis],
            cache: candidateCache,
            state: bonusOutput.state)
    case .rejected:
        acceptanceUniform = Double(1).nextDown
        residualUniform = 0.75
        guard checkpointSpeculativePromptCacheBeforeAppend(candidateCache, tokenCount: 1) else {
            throw SampledMTPDepthOneDiagnosticError.recurrentCheckpointUnavailable
        }
        let rejectedOutput = target(
            LMInput.Text(tokens: proposal.flattened())[text: .newAxis],
            cache: candidateCache,
            state: bonusOutput.state)
        eval(rejectedOutput.logits, candidateCache)
        guard rewindSpeculativePromptCache(candidateCache, numTokens: 1) == 1 else {
            throw SampledMTPDepthOneDiagnosticError.recurrentRewindFailed
        }
        emittedToken = residualSample(
            target: targetProbabilities,
            draft: draftProbabilities,
            uniform: residualUniform!)
        _ = target(
            LMInput.Text(tokens: MLXArray([emittedToken]))[text: .newAxis],
            cache: candidateCache,
            state: bonusOutput.state)
        discardSpeculativePromptCacheCheckpoints(candidateCache)
    }

    // The control lane deliberately uses the ordinary LanguageModel preparation contract,
    // independently of MTP hidden-state extraction. Cache equality therefore compares the
    // speculative transaction with the scalar generation path users actually run.
    let scalarCache = target.newCache(parameters: nil)
    let scalarPrefillOutput: LMOutput
    switch try target.prepare(input, cache: scalarCache, windowSize: 512) {
    case .logits(let output):
        scalarPrefillOutput = output
    case .tokens(let tokens):
        scalarPrefillOutput = target(
            tokens[text: .newAxis], cache: scalarCache, state: nil)
    }
    let scalarBonusOutput = target(
        LMInput.Text(tokens: bonus)[text: .newAxis],
        cache: scalarCache,
        state: scalarPrefillOutput.state)
    _ = target(
        LMInput.Text(tokens: MLXArray([emittedToken]))[text: .newAxis],
        cache: scalarCache,
        state: scalarBonusOutput.state)
    eval(candidateCache, scalarCache)

    return SampledMTPDepthOneDiagnosticResult(
        branch: branch,
        promptTokenCount: input.text.tokens.size,
        bonusToken: bonusToken,
        proposedToken: proposedToken,
        emittedToken: emittedToken,
        proposalUniform: proposalUniform,
        acceptanceUniform: acceptanceUniform,
        residualUniform: residualUniform,
        targetProbabilities: targetProbabilities,
        draftProbabilities: draftProbabilities,
        candidateCache: candidateCache,
        scalarCache: scalarCache)
}

private final class DiagnosticProposalSampler: LogitSampler {
    let targetProbabilities: [Double]
    let branch: SampledMTPDepthOneDiagnosticBranch
    private(set) var probabilities: [Double]?
    private(set) var proposalUniform: Double?
    private(set) var failure: SampledMTPDepthOneDiagnosticError?

    init(
        targetProbabilities: [Double],
        branch: SampledMTPDepthOneDiagnosticBranch
    ) {
        self.targetProbabilities = targetProbabilities
        self.branch = branch
    }

    func sample(logits: MLXArray) -> MLXArray {
        let probabilities = normalizedProbabilities(logits)
        self.probabilities = probabilities

        let token: Int
        switch branch {
        case .accepted:
            let uniform = 0.25
            proposalUniform = uniform
            token = categoricalSample(probabilities, uniform: uniform)
        case .rejected:
            var selected: Int?
            var greatestPositiveDifference = 0.0
            for index in probabilities.indices {
                let difference = probabilities[index] - targetProbabilities[index]
                if difference > greatestPositiveDifference {
                    greatestPositiveDifference = difference
                    selected = index
                }
            }
            guard let selected else {
                failure = .distributionsMatch
                return MLXArray([0])
            }
            token = selected
            let lower = probabilities[..<selected].reduce(0, +)
            proposalUniform = lower + probabilities[selected] / 2
        }
        return MLXArray([token])
    }
}

private func normalizedProbabilities(_ logits: MLXArray) -> [Double] {
    let probabilities = softmax(logits.asType(.float32), axis: -1).flattened()
    eval(probabilities)
    let raw = probabilities.asArray(Float.self).map(Double.init)
    let sum = raw.reduce(0, +)
    return raw.map { $0 / sum }
}

private func categoricalSample(_ distribution: [Double], uniform: Double) -> Int {
    var cumulative = 0.0
    for (index, probability) in distribution.enumerated() {
        cumulative += probability
        if uniform < cumulative { return index }
    }
    return distribution.index(before: distribution.endIndex)
}

private func residualSample(
    target: [Double],
    draft: [Double],
    uniform: Double
) -> Int {
    let residual = zip(target, draft).map { max($0 - $1, 0) }
    let mass = residual.reduce(0, +)
    return categoricalSample(residual.map { $0 / mass }, uniform: uniform)
}
