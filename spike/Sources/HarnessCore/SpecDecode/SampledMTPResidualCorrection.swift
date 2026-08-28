import Foundation

public enum SampledMTPResidualCorrectionError: Error, Equatable, Sendable {
    case emptyDistribution
    case mismatchedDistributionSizes(targetCount: Int, draftCount: Int)
    case invalidTargetProbability(index: Int)
    case invalidDraftProbability(index: Int)
    case nonNormalizedTarget(sum: Double)
    case nonNormalizedDraft(sum: Double)
    case invalidProposalToken(token: Int, vocabularyCount: Int)
    case zeroDraftMass(token: Int)
    case invalidAcceptanceUniform
    case invalidResidualUniform
    case missingCorrectionUniform
    case zeroResidualMass
}

public enum SampledMTPUniformPurpose: Equatable, Sendable {
    case acceptance
    case residual
}

public struct SampledMTPConsumedUniform: Equatable, Sendable {
    public let purpose: SampledMTPUniformPurpose
    public let value: Double

    public init(purpose: SampledMTPUniformPurpose, value: Double) {
        self.purpose = purpose
        self.value = value
    }
}

public enum SampledMTPResidualCorrectionDecision: Equatable, Sendable {
    case accepted(token: Int)
    case rejected(correctionToken: Int)
}

public struct SampledMTPResidualCorrectionTrace: Equatable, Sendable {
    public let proposedToken: Int
    public let acceptanceProbability: Double
    public let acceptanceUniform: Double
    public let residualUniform: Double?
    public let residualDistribution: [Double]?
    public let decision: SampledMTPResidualCorrectionDecision
    public let consumedUniforms: [SampledMTPConsumedUniform]

    public init(
        proposedToken: Int,
        acceptanceProbability: Double,
        acceptanceUniform: Double,
        residualUniform: Double?,
        residualDistribution: [Double]?,
        decision: SampledMTPResidualCorrectionDecision,
        consumedUniforms: [SampledMTPConsumedUniform])
    {
        self.proposedToken = proposedToken
        self.acceptanceProbability = acceptanceProbability
        self.acceptanceUniform = acceptanceUniform
        self.residualUniform = residualUniform
        self.residualDistribution = residualDistribution
        self.decision = decision
        self.consumedUniforms = consumedUniforms
    }
}

public enum SampledMTPResidualCorrection: Equatable, Sendable {
    public static func acceptanceProbability(
        target: [Double],
        draft: [Double],
        proposedToken: Int) throws -> Double
    {
        try validateDistributions(target: target, draft: draft)
        try validateProposalToken(proposedToken, vocabularyCount: target.count)
        let draftMass = draft[proposedToken]
        guard draftMass > 0.0 else {
            throw SampledMTPResidualCorrectionError.zeroDraftMass(token: proposedToken)
        }
        return min(1.0, target[proposedToken] / draftMass)
    }

    public static func residualDistribution(
        target: [Double],
        draft: [Double]) throws -> [Double]
    {
        try validateDistributions(target: target, draft: draft)
        let residual = zip(target, draft).map { max($0 - $1, 0.0) }
        let residualMass = residual.reduce(0.0, +)
        guard residualMass > 0.0 else {
            throw SampledMTPResidualCorrectionError.zeroResidualMass
        }
        return residual.map { $0 / residualMass }
    }

    public static func decide(
        target: [Double],
        draft: [Double],
        proposedToken: Int,
        acceptanceUniform: Double,
        residualUniform: Double? = nil) throws -> SampledMTPResidualCorrectionTrace
    {
        let acceptanceProbability = try acceptanceProbability(
            target: target,
            draft: draft,
            proposedToken: proposedToken)
        guard isValidUniform(acceptanceUniform) else {
            throw SampledMTPResidualCorrectionError.invalidAcceptanceUniform
        }

        let acceptanceDraw = SampledMTPConsumedUniform(
            purpose: .acceptance,
            value: acceptanceUniform)
        guard acceptanceUniform >= acceptanceProbability else {
            return SampledMTPResidualCorrectionTrace(
                proposedToken: proposedToken,
                acceptanceProbability: acceptanceProbability,
                acceptanceUniform: acceptanceUniform,
                residualUniform: nil,
                residualDistribution: nil,
                decision: .accepted(token: proposedToken),
                consumedUniforms: [acceptanceDraw])
        }

        guard let residualUniform else {
            throw SampledMTPResidualCorrectionError.missingCorrectionUniform
        }
        guard isValidUniform(residualUniform) else {
            throw SampledMTPResidualCorrectionError.invalidResidualUniform
        }

        let correctionDistribution = try residualDistribution(target: target, draft: draft)
        let correctionToken = sample(distribution: correctionDistribution, uniform: residualUniform)
        return SampledMTPResidualCorrectionTrace(
            proposedToken: proposedToken,
            acceptanceProbability: acceptanceProbability,
            acceptanceUniform: acceptanceUniform,
            residualUniform: residualUniform,
            residualDistribution: correctionDistribution,
            decision: .rejected(correctionToken: correctionToken),
            consumedUniforms: [
                acceptanceDraw,
                SampledMTPConsumedUniform(purpose: .residual, value: residualUniform),
            ])
    }

    private static func validateDistributions(target: [Double], draft: [Double]) throws {
        guard !target.isEmpty, !draft.isEmpty else {
            throw SampledMTPResidualCorrectionError.emptyDistribution
        }
        guard target.count == draft.count else {
            throw SampledMTPResidualCorrectionError.mismatchedDistributionSizes(
                targetCount: target.count,
                draftCount: draft.count)
        }

        for (index, value) in target.enumerated() {
            guard value.isFinite, value >= 0.0 else {
                throw SampledMTPResidualCorrectionError.invalidTargetProbability(index: index)
            }
        }
        for (index, value) in draft.enumerated() {
            guard value.isFinite, value >= 0.0 else {
                throw SampledMTPResidualCorrectionError.invalidDraftProbability(index: index)
            }
        }

        let targetSum = target.reduce(0.0, +)
        guard abs(targetSum - 1.0) <= normalizationTolerance else {
            throw SampledMTPResidualCorrectionError.nonNormalizedTarget(sum: targetSum)
        }
        let draftSum = draft.reduce(0.0, +)
        guard abs(draftSum - 1.0) <= normalizationTolerance else {
            throw SampledMTPResidualCorrectionError.nonNormalizedDraft(sum: draftSum)
        }
    }

    private static func validateProposalToken(_ token: Int, vocabularyCount: Int) throws {
        guard token >= 0, token < vocabularyCount else {
            throw SampledMTPResidualCorrectionError.invalidProposalToken(
                token: token,
                vocabularyCount: vocabularyCount)
        }
    }

    private static func isValidUniform(_ value: Double) -> Bool {
        value.isFinite && value >= 0.0 && value < 1.0
    }

    private static func sample(distribution: [Double], uniform: Double) -> Int {
        var cumulative = 0.0
        var lastSupportedToken = 0
        for (token, probability) in distribution.enumerated() {
            if probability > 0.0 {
                lastSupportedToken = token
            }
            cumulative += probability
            if uniform < cumulative {
                return token
            }
        }
        return lastSupportedToken
    }

    private static let normalizationTolerance = 1e-12
}
