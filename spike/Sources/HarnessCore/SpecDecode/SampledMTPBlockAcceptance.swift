import Foundation

public struct SampledMTPBlockStep: Equatable, Sendable {
    public let targetDistribution: [Double]
    public let draftDistribution: [Double]
    public let proposedToken: Int

    public init(
        targetDistribution: [Double],
        draftDistribution: [Double],
        proposedToken: Int)
    {
        self.targetDistribution = targetDistribution
        self.draftDistribution = draftDistribution
        self.proposedToken = proposedToken
    }
}

public enum SampledMTPBlockUniformPurpose: Equatable, Sendable {
    case acceptance
    case residual
    case bonus
}

public struct SampledMTPBlockConsumedUniform: Equatable, Sendable {
    public let purpose: SampledMTPBlockUniformPurpose
    public let value: Double

    public init(purpose: SampledMTPBlockUniformPurpose, value: Double) {
        self.purpose = purpose
        self.value = value
    }
}

public enum SampledMTPBlockTerminalPurpose: Equatable, Sendable {
    case residual
    case bonus
}

public enum SampledMTPBlockTerminalDraw: Equatable, Sendable {
    case residual(Double)
    case bonus(Double)

    public var purpose: SampledMTPBlockTerminalPurpose {
        switch self {
        case .residual:
            return .residual
        case .bonus:
            return .bonus
        }
    }
}

public enum SampledMTPBlockOutcome: Equatable, Sendable {
    case rejected(stepIndex: Int, correctionToken: Int)
    case acceptedAll(bonusToken: Int)
}

public struct SampledMTPBlockTrace: Equatable, Sendable {
    public let stepIndex: Int
    public let residualTrace: SampledMTPResidualCorrectionTrace

    public init(stepIndex: Int, residualTrace: SampledMTPResidualCorrectionTrace) {
        self.stepIndex = stepIndex
        self.residualTrace = residualTrace
    }
}

public struct SampledMTPBlockDecision: Equatable, Sendable {
    public let tokens: [Int]
    public let acceptedDraftCount: Int
    public let acceptedDraftEndIndex: Int?
    public let outcome: SampledMTPBlockOutcome
    public let traces: [SampledMTPBlockTrace]
    public let consumedUniforms: [SampledMTPBlockConsumedUniform]

    public init(
        tokens: [Int],
        acceptedDraftCount: Int,
        acceptedDraftEndIndex: Int?,
        outcome: SampledMTPBlockOutcome,
        traces: [SampledMTPBlockTrace],
        consumedUniforms: [SampledMTPBlockConsumedUniform])
    {
        self.tokens = tokens
        self.acceptedDraftCount = acceptedDraftCount
        self.acceptedDraftEndIndex = acceptedDraftEndIndex
        self.outcome = outcome
        self.traces = traces
        self.consumedUniforms = consumedUniforms
    }
}

public enum SampledMTPBlockAcceptanceError: Error, Equatable, Sendable {
    case emptySteps
    case missingAcceptanceUniform(index: Int)
    case extraAcceptanceUniforms(expected: Int, actual: Int)
    case missingTerminalDraw(expected: SampledMTPBlockTerminalPurpose)
    case extraTerminalDraws(expected: Int, actual: Int)
    case wrongTerminalDraw(expected: SampledMTPBlockTerminalPurpose, actual: SampledMTPBlockTerminalPurpose)
    case emptyBonusDistribution
    case invalidBonusProbability(index: Int)
    case nonNormalizedBonusDistribution(sum: Double)
    case invalidBonusUniform
}

public enum SampledMTPBlockAcceptance: Equatable, Sendable {
    public static func decide(
        steps: [SampledMTPBlockStep],
        acceptanceUniforms: [Double],
        terminalDraws: [SampledMTPBlockTerminalDraw],
        bonusTargetDistribution: [Double]) throws -> SampledMTPBlockDecision
    {
        guard !steps.isEmpty else {
            throw SampledMTPBlockAcceptanceError.emptySteps
        }
        guard terminalDraws.count <= 1 else {
            throw SampledMTPBlockAcceptanceError.extraTerminalDraws(
                expected: 1,
                actual: terminalDraws.count)
        }

        let terminalDraw = terminalDraws.first
        var tokens: [Int] = []
        var traces: [SampledMTPBlockTrace] = []
        var consumedUniforms: [SampledMTPBlockConsumedUniform] = []

        for (stepIndex, step) in steps.enumerated() {
            guard stepIndex < acceptanceUniforms.count else {
                throw SampledMTPBlockAcceptanceError.missingAcceptanceUniform(index: stepIndex)
            }

            let residualUniform: Double?
            switch terminalDraw {
            case let .residual(value):
                residualUniform = value
            case .bonus, nil:
                residualUniform = nil
            }

            let residualTrace: SampledMTPResidualCorrectionTrace
            do {
                residualTrace = try SampledMTPResidualCorrection.decide(
                    target: step.targetDistribution,
                    draft: step.draftDistribution,
                    proposedToken: step.proposedToken,
                    acceptanceUniform: acceptanceUniforms[stepIndex],
                    residualUniform: residualUniform)
            } catch SampledMTPResidualCorrectionError.missingCorrectionUniform {
                throw terminalErrorForRejectedStep(terminalDraw)
            }

            let blockTrace = SampledMTPBlockTrace(
                stepIndex: stepIndex,
                residualTrace: residualTrace)
            traces.append(blockTrace)
            consumedUniforms.append(contentsOf: residualTrace.consumedUniforms.map(blockConsumedUniform))

            switch residualTrace.decision {
            case let .accepted(token):
                tokens.append(token)
            case let .rejected(correctionToken):
                let expectedAcceptanceCount = stepIndex + 1
                guard acceptanceUniforms.count == expectedAcceptanceCount else {
                    throw SampledMTPBlockAcceptanceError.extraAcceptanceUniforms(
                        expected: expectedAcceptanceCount,
                        actual: acceptanceUniforms.count)
                }
                tokens.append(correctionToken)
                return SampledMTPBlockDecision(
                    tokens: tokens,
                    acceptedDraftCount: stepIndex,
                    acceptedDraftEndIndex: acceptedDraftEndIndex(count: stepIndex),
                    outcome: .rejected(stepIndex: stepIndex, correctionToken: correctionToken),
                    traces: traces,
                    consumedUniforms: consumedUniforms)
            }
        }

        guard acceptanceUniforms.count == steps.count else {
            throw SampledMTPBlockAcceptanceError.extraAcceptanceUniforms(
                expected: steps.count,
                actual: acceptanceUniforms.count)
        }
        guard let terminalDraw else {
            throw SampledMTPBlockAcceptanceError.missingTerminalDraw(expected: .bonus)
        }
        guard case let .bonus(bonusUniform) = terminalDraw else {
            throw SampledMTPBlockAcceptanceError.wrongTerminalDraw(
                expected: .bonus,
                actual: terminalDraw.purpose)
        }

        try validateBonusDistribution(bonusTargetDistribution)
        guard isValidUniform(bonusUniform) else {
            throw SampledMTPBlockAcceptanceError.invalidBonusUniform
        }

        let bonusToken = sample(distribution: bonusTargetDistribution, uniform: bonusUniform)
        tokens.append(bonusToken)
        consumedUniforms.append(.init(purpose: .bonus, value: bonusUniform))
        return SampledMTPBlockDecision(
            tokens: tokens,
            acceptedDraftCount: steps.count,
            acceptedDraftEndIndex: acceptedDraftEndIndex(count: steps.count),
            outcome: .acceptedAll(bonusToken: bonusToken),
            traces: traces,
            consumedUniforms: consumedUniforms)
    }

    private static func terminalErrorForRejectedStep(
        _ terminalDraw: SampledMTPBlockTerminalDraw?) -> SampledMTPBlockAcceptanceError
    {
        guard let terminalDraw else {
            return .missingTerminalDraw(expected: .residual)
        }
        return .wrongTerminalDraw(expected: .residual, actual: terminalDraw.purpose)
    }

    private static func blockConsumedUniform(
        _ consumedUniform: SampledMTPConsumedUniform) -> SampledMTPBlockConsumedUniform
    {
        switch consumedUniform.purpose {
        case .acceptance:
            return .init(purpose: .acceptance, value: consumedUniform.value)
        case .residual:
            return .init(purpose: .residual, value: consumedUniform.value)
        }
    }

    private static func acceptedDraftEndIndex(count: Int) -> Int? {
        count > 0 ? count - 1 : nil
    }

    private static func validateBonusDistribution(_ distribution: [Double]) throws {
        guard !distribution.isEmpty else {
            throw SampledMTPBlockAcceptanceError.emptyBonusDistribution
        }
        for (index, value) in distribution.enumerated() {
            guard value.isFinite, value >= 0.0 else {
                throw SampledMTPBlockAcceptanceError.invalidBonusProbability(index: index)
            }
        }

        let sum = distribution.reduce(0.0, +)
        guard abs(sum - 1.0) <= normalizationTolerance else {
            throw SampledMTPBlockAcceptanceError.nonNormalizedBonusDistribution(sum: sum)
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
