// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Forced outcomes for the Release-only sampled two-proposal diagnostic.
/// Ordinary generation never calls this surface.
public enum SampledMTPBlockDiagnosticOutcome: String, Sendable, Equatable {
    case rejectFirst
    case rejectSecond
    case acceptAll
}

public enum SampledMTPBlockDiagnosticTerminalDraw: Sendable, Equatable {
    case residual(Double)
    case bonus(Double)
}

public struct SampledMTPBlockDiagnosticStep: Sendable, Equatable {
    public let stepIndex: Int
    public let proposedToken: Int
    public let proposalUniform: Double
    public let targetProbabilities: [Double]
    public let draftProbabilities: [Double]

    public init(
        stepIndex: Int,
        proposedToken: Int,
        proposalUniform: Double,
        targetProbabilities: [Double],
        draftProbabilities: [Double]
    ) {
        self.stepIndex = stepIndex
        self.proposedToken = proposedToken
        self.proposalUniform = proposalUniform
        self.targetProbabilities = targetProbabilities
        self.draftProbabilities = draftProbabilities
    }
}

public struct SampledMTPBlockDiagnosticDecision: Sendable, Equatable {
    public let outputTokens: [Int]
    public let acceptedDraftCount: Int
    public let acceptedDraftEndIndex: Int?

    public init(
        outputTokens: [Int],
        acceptedDraftCount: Int,
        acceptedDraftEndIndex: Int?
    ) {
        self.outputTokens = outputTokens
        self.acceptedDraftCount = acceptedDraftCount
        self.acceptedDraftEndIndex = acceptedDraftEndIndex
    }
}

public struct SampledMTPBlockDiagnosticResult {
    public let outcome: SampledMTPBlockDiagnosticOutcome
    public let promptTokenCount: Int
    public let initialBonusToken: Int
    public let steps: [SampledMTPBlockDiagnosticStep]
    public let acceptanceUniforms: [Double]
    public let terminalDraw: SampledMTPBlockDiagnosticTerminalDraw
    public let bonusTargetProbabilities: [Double]
    public let decision: SampledMTPBlockDiagnosticDecision
    public let candidateCache: [KVCache]
    public let scalarCache: [KVCache]

    public init(
        outcome: SampledMTPBlockDiagnosticOutcome,
        promptTokenCount: Int,
        initialBonusToken: Int,
        steps: [SampledMTPBlockDiagnosticStep],
        acceptanceUniforms: [Double],
        terminalDraw: SampledMTPBlockDiagnosticTerminalDraw,
        bonusTargetProbabilities: [Double],
        decision: SampledMTPBlockDiagnosticDecision,
        candidateCache: [KVCache],
        scalarCache: [KVCache]
    ) {
        self.outcome = outcome
        self.promptTokenCount = promptTokenCount
        self.initialBonusToken = initialBonusToken
        self.steps = steps
        self.acceptanceUniforms = acceptanceUniforms
        self.terminalDraw = terminalDraw
        self.bonusTargetProbabilities = bonusTargetProbabilities
        self.decision = decision
        self.candidateCache = candidateCache
        self.scalarCache = scalarCache
    }
}

public enum SampledMTPBlockDiagnosticError: Error, Sendable, Equatable {
    case targetDoesNotExposePromptHiddenState
    case drafterIsNotStateful
    case promptPreparationUnavailable
    case promptPreparationDidNotReturnLogits
    case missingProposal
    case wrongProposalCount
    case missingProposalDistribution
    case distributionsMatch(stepIndex: Int)
    case secondStepProbeDrift
    case recurrentCheckpointUnavailable
    case recurrentRewindFailed
    case invalidDecision
}

/// Execute one real two-proposal Qwen MTP transaction and finalize its hybrid
/// cache using a caller-owned probability-ratio decision. The callback keeps
/// the vendored model layer independent from the project's pure acceptance gate.
public func runSampledMTPBlockDiagnostic(
    input: LMInput,
    target: any LanguageModel,
    drafter: any MTPDrafterModel,
    outcome: SampledMTPBlockDiagnosticOutcome,
    decide: (
        _ steps: [SampledMTPBlockDiagnosticStep],
        _ acceptanceUniforms: [Double],
        _ terminalDraw: SampledMTPBlockDiagnosticTerminalDraw,
        _ bonusTargetProbabilities: [Double]
    ) throws -> SampledMTPBlockDiagnosticDecision
) throws -> SampledMTPBlockDiagnosticResult {
    guard target is any MTPPromptHiddenStatePreparingModel else {
        throw SampledMTPBlockDiagnosticError.targetDoesNotExposePromptHiddenState
    }
    guard drafter is any StatefulMTPDrafterModel else {
        throw SampledMTPBlockDiagnosticError.drafterIsNotStateful
    }

    let secondRejectionTarget: [Double]?
    if outcome == .rejectSecond {
        let probe = try makeSampledMTPBlockPass(
            input: input,
            target: target,
            drafter: drafter,
            outcome: .acceptAll,
            secondRejectionTarget: nil)
        secondRejectionTarget = probe.steps[1].targetProbabilities
    } else {
        secondRejectionTarget = nil
    }

    let pass = try makeSampledMTPBlockPass(
        input: input,
        target: target,
        drafter: drafter,
        outcome: outcome,
        secondRejectionTarget: secondRejectionTarget)
    if let secondRejectionTarget,
        pass.steps[1].targetProbabilities != secondRejectionTarget
    {
        throw SampledMTPBlockDiagnosticError.secondStepProbeDrift
    }

    let acceptanceUniforms: [Double]
    let terminalDraw: SampledMTPBlockDiagnosticTerminalDraw
    switch outcome {
    case .rejectFirst:
        acceptanceUniforms = [Double(1).nextDown]
        terminalDraw = .residual(0.75)
    case .rejectSecond:
        acceptanceUniforms = [0, Double(1).nextDown]
        terminalDraw = .residual(0.75)
    case .acceptAll:
        acceptanceUniforms = [0, 0]
        terminalDraw = .bonus(0.25)
    }
    let decision = try decide(
        pass.steps, acceptanceUniforms, terminalDraw, pass.bonusTargetProbabilities)
    guard decision.acceptedDraftCount == outcome.acceptedDraftCount,
        decision.acceptedDraftEndIndex
            == (outcome.acceptedDraftCount > 0 ? outcome.acceptedDraftCount - 1 : nil),
        decision.outputTokens.count == decision.acceptedDraftCount + 1,
        Array(decision.outputTokens.prefix(decision.acceptedDraftCount))
            == Array(pass.proposedTokens.prefix(decision.acceptedDraftCount)),
        decision.outputTokens.allSatisfy({ $0 >= 0 })
    else {
        throw SampledMTPBlockDiagnosticError.invalidDecision
    }

    var candidateState = pass.bonusState
    if decision.acceptedDraftCount < 2 {
        guard rewindSpeculativePromptCache(pass.cache, numTokens: 2) == 2 else {
            throw SampledMTPBlockDiagnosticError.recurrentRewindFailed
        }
        if decision.acceptedDraftCount > 0 {
            let prefix = Array(pass.proposedTokens.prefix(decision.acceptedDraftCount))
            let replay = target(
                LMInput.Text(tokens: MLXArray(prefix))[text: .newAxis],
                cache: pass.cache,
                state: candidateState)
            candidateState = replay.state
        }
    } else {
        candidateState = pass.verifyState
    }
    let terminal = decision.outputTokens[decision.acceptedDraftCount]
    _ = target(
        LMInput.Text(tokens: MLXArray([terminal]))[text: .newAxis],
        cache: pass.cache,
        state: candidateState)
    discardSpeculativePromptCacheCheckpoints(pass.cache)

    let scalarCache = target.newCache(parameters: nil)
    let scalarPrefill: LMOutput
    switch try target.prepare(input, cache: scalarCache, windowSize: 512) {
    case .logits(let output):
        scalarPrefill = output
    case .tokens(let tokens):
        scalarPrefill = target(tokens[text: .newAxis], cache: scalarCache, state: nil)
    }
    var scalarState = target(
        LMInput.Text(tokens: MLXArray([pass.initialBonusToken]))[text: .newAxis],
        cache: scalarCache,
        state: scalarPrefill.state).state
    for token in decision.outputTokens {
        scalarState = target(
            LMInput.Text(tokens: MLXArray([token]))[text: .newAxis],
            cache: scalarCache,
            state: scalarState).state
    }
    eval(pass.cache, scalarCache)

    return SampledMTPBlockDiagnosticResult(
        outcome: outcome,
        promptTokenCount: input.text.tokens.size,
        initialBonusToken: pass.initialBonusToken,
        steps: pass.steps,
        acceptanceUniforms: acceptanceUniforms,
        terminalDraw: terminalDraw,
        bonusTargetProbabilities: pass.bonusTargetProbabilities,
        decision: decision,
        candidateCache: pass.cache,
        scalarCache: scalarCache)
}

private struct SampledMTPBlockPass {
    let cache: [KVCache]
    let bonusState: LMOutput.State?
    let verifyState: LMOutput.State?
    let initialBonusToken: Int
    let proposedTokens: [Int]
    let steps: [SampledMTPBlockDiagnosticStep]
    let bonusTargetProbabilities: [Double]
}

private func makeSampledMTPBlockPass(
    input: LMInput,
    target: any LanguageModel,
    drafter: any MTPDrafterModel,
    outcome: SampledMTPBlockDiagnosticOutcome,
    secondRejectionTarget: [Double]?
) throws -> SampledMTPBlockPass {
    guard let promptTarget = target as? any MTPPromptHiddenStatePreparingModel else {
        throw SampledMTPBlockDiagnosticError.targetDoesNotExposePromptHiddenState
    }
    guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
        throw SampledMTPBlockDiagnosticError.drafterIsNotStateful
    }
    let cache = target.newCache(parameters: nil)
    guard let preparation = try promptTarget.prepareForMTP(input, cache: cache, windowSize: 512) else {
        throw SampledMTPBlockDiagnosticError.promptPreparationUnavailable
    }
    guard case .logits(let prefillOutput) = preparation.result else {
        throw SampledMTPBlockDiagnosticError.promptPreparationDidNotReturnLogits
    }

    let initialBonus = argMax(prefillOutput.logits[0..., -1, 0...], axis: -1)
    eval(initialBonus)
    var bonusInputState = prefillOutput.state ?? LMOutput.State()
    bonusInputState[mtpEmitFlagKey] = true
    let bonusOutput = target(
        LMInput.Text(tokens: initialBonus)[text: .newAxis],
        cache: cache,
        state: bonusInputState)
    let firstTarget = sampledBlockNormalizedProbabilities(
        bonusOutput.logits[0..., -1, 0...])

    let sampler = SampledMTPBlockProposalSampler(
        firstTarget: firstTarget,
        rejectFirst: outcome == .rejectFirst,
        secondRejectionTarget: secondRejectionTarget)
    var drafterState = statefulDrafter.makeState(parameters: nil)
    statefulDrafter.prepareDrafterState(
        target: target,
        promptTokens: input.text.tokens,
        targetHidden: preparation.targetHidden,
        firstBonus: initialBonus,
        positionDeltas: prefillOutput.state?[mtpPositionDeltasKey],
        state: &drafterState,
        sampler: sampler)
    guard drafterState.seedToken != nil else {
        throw sampler.failure ?? SampledMTPBlockDiagnosticError.missingProposal
    }
    let lastHidden = preparation.targetHidden[0..., (-1)..., 0...]
    let proposed = statefulDrafter.draftBlock(
        target: target,
        lastToken: initialBonus,
        lastHidden: lastHidden,
        sharedKV: bonusOutput.state?[mtpSharedKVStatesKey] ?? [:],
        positionDeltas: bonusOutput.state?[mtpPositionDeltasKey],
        queryOffset: cache.first?.offset ?? input.text.tokens.size,
        blockSize: 3,
        state: &drafterState,
        sampler: sampler).flattened()
    eval(proposed)
    let proposedTokens = proposed.asArray(Int.self)
    guard proposedTokens.count == 2 else {
        throw SampledMTPBlockDiagnosticError.wrongProposalCount
    }
    guard sampler.probabilities.count == 2, sampler.uniforms.count == 2 else {
        throw sampler.failure ?? SampledMTPBlockDiagnosticError.missingProposalDistribution
    }

    guard checkpointSpeculativePromptCacheBeforeAppend(cache, tokenCount: 2) else {
        throw SampledMTPBlockDiagnosticError.recurrentCheckpointUnavailable
    }
    let verify = target(
        LMInput.Text(tokens: proposed)[text: .newAxis],
        cache: cache,
        state: bonusOutput.state)
    let secondTarget = sampledBlockNormalizedProbabilities(verify.logits[0..., 0, 0...])
    let bonusTarget = sampledBlockNormalizedProbabilities(verify.logits[0..., 1, 0...])
    let targets = [firstTarget, secondTarget]
    let steps = (0 ..< 2).map { index in
        SampledMTPBlockDiagnosticStep(
            stepIndex: index,
            proposedToken: proposedTokens[index],
            proposalUniform: sampler.uniforms[index],
            targetProbabilities: targets[index],
            draftProbabilities: sampler.probabilities[index])
    }
    return SampledMTPBlockPass(
        cache: cache,
        bonusState: bonusOutput.state,
        verifyState: verify.state,
        initialBonusToken: initialBonus.item(Int.self),
        proposedTokens: proposedTokens,
        steps: steps,
        bonusTargetProbabilities: bonusTarget)
}

private final class SampledMTPBlockProposalSampler: LogitSampler {
    private let firstTarget: [Double]
    private let rejectFirst: Bool
    private let secondRejectionTarget: [Double]?
    private(set) var probabilities = [[Double]]()
    private(set) var uniforms = [Double]()
    private(set) var failure: SampledMTPBlockDiagnosticError?

    init(firstTarget: [Double], rejectFirst: Bool, secondRejectionTarget: [Double]?) {
        self.firstTarget = firstTarget
        self.rejectFirst = rejectFirst
        self.secondRejectionTarget = secondRejectionTarget
    }

    func sample(logits: MLXArray) -> MLXArray {
        let draft = sampledBlockNormalizedProbabilities(logits)
        let index = probabilities.count
        probabilities.append(draft)

        let rejectionTarget: [Double]?
        if index == 0, rejectFirst {
            rejectionTarget = firstTarget
        } else if index == 1 {
            rejectionTarget = secondRejectionTarget
        } else {
            rejectionTarget = nil
        }
        let uniform: Double
        if let rejectionTarget {
            guard let value = sampledBlockRejectionUniform(
                target: rejectionTarget, draft: draft)
            else {
                failure = .distributionsMatch(stepIndex: index)
                uniforms.append(0.25)
                return MLXArray([0])
            }
            uniform = value
        } else {
            uniform = 0.25
        }
        uniforms.append(uniform)
        return MLXArray([sampledBlockCategoricalSample(draft, uniform: uniform)])
    }
}

private extension SampledMTPBlockDiagnosticOutcome {
    var acceptedDraftCount: Int {
        switch self {
        case .rejectFirst: 0
        case .rejectSecond: 1
        case .acceptAll: 2
        }
    }
}

private func sampledBlockNormalizedProbabilities(_ logits: MLXArray) -> [Double] {
    let probabilities = softmax(logits.asType(.float32), axis: -1).flattened()
    eval(probabilities)
    let raw = probabilities.asArray(Float.self).map(Double.init)
    return normalizeSampledMTPBlockProbabilities(raw)
}

func normalizeSampledMTPBlockProbabilities(_ raw: [Double]) -> [Double] {
    let sum = raw.reduce(0, +)
    var normalized = raw.map { $0 / sum }
    guard let largest = normalized.indices.max(by: {
        normalized[$0] < normalized[$1]
    }) else { return normalized }
    normalized[largest] += 1 - normalized.reduce(0, +)
    return normalized
}

private func sampledBlockRejectionUniform(target: [Double], draft: [Double]) -> Double? {
    var selected: Int?
    var greatestDifference = 0.0
    for index in draft.indices {
        let difference = draft[index] - target[index]
        if difference > greatestDifference {
            greatestDifference = difference
            selected = index
        }
    }
    guard let selected else { return nil }
    return draft[..<selected].reduce(0, +) + draft[selected] / 2
}

private func sampledBlockCategoricalSample(_ distribution: [Double], uniform: Double) -> Int {
    var cumulative = 0.0
    for (index, probability) in distribution.enumerated() {
        cumulative += probability
        if uniform < cumulative { return index }
    }
    return distribution.index(before: distribution.endIndex)
}
