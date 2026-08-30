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
    private var blockIndex = 0

    public private(set) var drawTraces: [SeededSampledMTPBlockRuntimeDrawTrace] = []
    public var proposalSampler: any LogitSampler { recorder }

    public init(
        entropy: @escaping SampledMTPBlockRuntimeEntropy = { _ in
            Double.random(in: 0 ..< 1)
        }
    ) {
        self.entropy = entropy
        self.recorder = NondeterministicSampledMTPProposalSampler(entropy: entropy)
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        parameters.temperature == 1
            && parameters.topP == 1
            && parameters.topK == 0
            && parameters.minP == 0
            && parameters.repetitionPenalty == nil
            && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
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
        let bonusTargetDistribution = try runtimeNormalizedProbabilities(bonusTargetLogits)
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

            let targetDistribution = try runtimeNormalizedProbabilities(targetLogits[index])
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
    private var blockIndex = 0

    public private(set) var drawTraces: [SeededSampledMTPBlockRuntimeDrawTrace] = []
    public var proposalSampler: any LogitSampler { recorder }

    public init(seed: UInt64) {
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
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        parameters.temperature == 1
            && parameters.topP == 1
            && parameters.topK == 0
            && parameters.minP == 0
            && parameters.repetitionPenalty == nil
            && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
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
        let bonusTargetDistribution = try validatingNormalizedProbabilities(bonusTargetLogits)
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

            let targetDistribution = try validatingNormalizedProbabilities(targetLogits[index])
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
    private var planIndex = 0

    public var proposalSampler: any LogitSampler { recorder }

    public init(plans: [SampledMTPBlockRuntimeDrawPlan]) {
        self.plans = plans
        self.recorder = FixedUniformProposalSampler(
            uniforms: plans.flatMap(\.proposalUniforms))
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        !plans.isEmpty
            && parameters.temperature == 1
            && parameters.topP == 1
            && parameters.topK == 0
            && parameters.minP == 0
            && parameters.repetitionPenalty == nil
            && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
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
                targetDistribution: normalizedProbabilities(targetLogits[index]),
                draftDistribution: proposal.probabilities,
                proposedToken: token,
            )
        }
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: plan.acceptanceUniforms,
            terminalDraws: [plan.terminalDraw],
            bonusTargetDistribution: normalizedProbabilities(bonusTargetLogits))
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

    init(seed: UInt64) {
        self.proposalUniforms = SeededSampledMTPUniformSource(seed: seed, domain: .proposal)
    }

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

private func normalizedProbabilities(_ logits: MLXArray) -> [Double] {
    let probabilities = softmax(logits.asType(.float32), axis: -1).flattened()
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

private func validatingNormalizedProbabilities(_ logits: MLXArray) throws -> [Double] {
    let rawLogits = logits.flattened()
    eval(rawLogits)
    guard rawLogits.asArray(Float.self).allSatisfy(\.isFinite) else {
        throw SeededSampledMTPBlockRuntimeProviderError.invalidLogits
    }
    let distribution = normalizedProbabilities(logits)
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

private func runtimeNormalizedProbabilities(_ logits: MLXArray) throws -> [Double] {
    do {
        return try validatingNormalizedProbabilities(logits)
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

private func categoricalSample(_ probabilities: [Double], uniform: Double) -> Int? {
    guard !probabilities.isEmpty else { return nil }
    var cumulative = 0.0
    for (index, probability) in probabilities.enumerated() {
        cumulative += probability
        if uniform < cumulative { return index }
    }
    return probabilities.indices.last
}

private extension Data {
    mutating func appendBigEndian(_ value: UInt64) {
        var bigEndian = value.bigEndian
        Swift.withUnsafeBytes(of: &bigEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
}
