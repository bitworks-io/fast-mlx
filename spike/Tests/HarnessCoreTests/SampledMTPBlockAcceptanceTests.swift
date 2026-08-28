import XCTest
@testable import HarnessCore

final class SampledMTPBlockAcceptanceTests: XCTestCase {
    func testFullAcceptsExactlyTwoProposalsAndSamplesBonusFromTargetDistribution() throws {
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: [
                .init(
                    targetDistribution: [0.6, 0.3, 0.1],
                    draftDistribution: [0.3, 0.4, 0.3],
                    proposedToken: 0),
                .init(
                    targetDistribution: [0.2, 0.5, 0.3],
                    draftDistribution: [0.4, 0.4, 0.2],
                    proposedToken: 1),
            ],
            acceptanceUniforms: [0.99, 0.25],
            terminalDraws: [.bonus(0.25)],
            bonusTargetDistribution: [0.1, 0.2, 0.7])

        XCTAssertEqual(decision.tokens, [0, 1, 1])
        XCTAssertEqual(decision.acceptedDraftCount, 2)
        XCTAssertEqual(decision.acceptedDraftEndIndex, 1)
        XCTAssertEqual(decision.outcome, .acceptedAll(bonusToken: 1))
        XCTAssertEqual(decision.traces.map(\.stepIndex), [0, 1])
        XCTAssertEqual(decision.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.99),
            .init(purpose: .acceptance, value: 0.25),
            .init(purpose: .bonus, value: 0.25),
        ])
    }

    func testFirstRejectionReturnsCorrectionAndConsumesOnlyPrefixAcceptanceAndResidual() throws {
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: [
                .init(
                    targetDistribution: [0.5, 0.25, 0.25],
                    draftDistribution: [0.25, 0.5, 0.25],
                    proposedToken: 1),
                .init(
                    targetDistribution: [0.2, 0.5, 0.3],
                    draftDistribution: [0.4, 0.4, 0.2],
                    proposedToken: 1),
            ],
            acceptanceUniforms: [0.75],
            terminalDraws: [.residual(0.4)],
            bonusTargetDistribution: [0.1, 0.2, 0.7])

        XCTAssertEqual(decision.tokens, [0])
        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertNil(decision.acceptedDraftEndIndex)
        XCTAssertEqual(decision.outcome, .rejected(stepIndex: 0, correctionToken: 0))
        XCTAssertEqual(decision.traces.count, 1)
        XCTAssertEqual(decision.traces[0].stepIndex, 0)
        XCTAssertEqual(decision.traces[0].residualTrace.decision, .rejected(correctionToken: 0))
        XCTAssertEqual(decision.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.75),
            .init(purpose: .residual, value: 0.4),
        ])
    }

    func testSecondRejectionReturnsAcceptedPrefixAndCorrection() throws {
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: partialRejectionSteps,
            acceptanceUniforms: [0.25, 0.9],
            terminalDraws: [.residual(0.9)],
            bonusTargetDistribution: [0.1, 0.2, 0.7])

        XCTAssertEqual(decision.tokens, [1, 2])
        XCTAssertEqual(decision.acceptedDraftCount, 1)
        XCTAssertEqual(decision.acceptedDraftEndIndex, 0)
        XCTAssertEqual(decision.outcome, .rejected(stepIndex: 1, correctionToken: 2))
        XCTAssertEqual(decision.traces.map(\.residualTrace.decision), [
            .accepted(token: 1),
            .rejected(correctionToken: 2),
        ])
    }

    func testDeterministicRepeatabilityAndDrawTrace() throws {
        let first = try SampledMTPBlockAcceptance.decide(
            steps: partialRejectionSteps,
            acceptanceUniforms: [0.25, 0.9],
            terminalDraws: [.residual(0.9)],
            bonusTargetDistribution: [0.1, 0.2, 0.7])
        let second = try SampledMTPBlockAcceptance.decide(
            steps: partialRejectionSteps,
            acceptanceUniforms: [0.25, 0.9],
            terminalDraws: [.residual(0.9)],
            bonusTargetDistribution: [0.1, 0.2, 0.7])

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.traces[0].residualTrace.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.25),
        ])
        XCTAssertEqual(first.traces[1].residualTrace.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.9),
            .init(purpose: .residual, value: 0.9),
        ])
        XCTAssertEqual(first.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.25),
            .init(purpose: .acceptance, value: 0.9),
            .init(purpose: .residual, value: 0.9),
        ])
    }

    func testFailsClosedForEmptySteps() {
        assertBlockThrows(.emptySteps) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [],
                acceptanceUniforms: [],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
    }

    func testFailsClosedForMissingAndExtraAcceptanceUniforms() {
        assertBlockThrows(.missingAcceptanceUniform(index: 0)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.extraAcceptanceUniforms(expected: 1, actual: 2)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0, 0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.extraAcceptanceUniforms(expected: 1, actual: 2)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                    .alwaysAccepting(proposedToken: 0),
                ],
                acceptanceUniforms: [0.75, 0.0],
                terminalDraws: [.residual(0.4)],
                bonusTargetDistribution: [1.0])
        }
    }

    func testFailsClosedForMissingExtraAndWrongTerminalDraws() {
        assertBlockThrows(.missingTerminalDraw(expected: .bonus)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.missingTerminalDraw(expected: .residual)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                ],
                acceptanceUniforms: [0.75],
                terminalDraws: [],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.extraTerminalDraws(expected: 1, actual: 2)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0), .bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.wrongTerminalDraw(expected: .bonus, actual: .residual)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.residual(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertBlockThrows(.wrongTerminalDraw(expected: .residual, actual: .bonus)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                ],
                acceptanceUniforms: [0.75],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
    }

    func testFailsClosedForInvalidBonusDistributionAndUniform() {
        assertBlockThrows(.emptyBonusDistribution) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [])
        }
        assertBlockThrows(.invalidBonusProbability(index: 0)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [.nan, 1.0])
        }
        assertBlockThrows(.nonNormalizedBonusDistribution(sum: 0.9)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [0.4, 0.5])
        }
        assertBlockThrows(.invalidBonusUniform) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(1.0)],
                bonusTargetDistribution: [1.0])
        }
    }

    func testPropagatesMalformedStepDistributionAndDrawErrorsFromResidualCorrection() {
        assertResidualThrows(.nonNormalizedTarget(sum: 0.4)) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.2, 0.2],
                        draftDistribution: [0.5, 0.5],
                        proposedToken: 0),
                ],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertResidualThrows(.invalidAcceptanceUniform) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [.alwaysAccepting(proposedToken: 0)],
                acceptanceUniforms: [1.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        }
        assertResidualThrows(.invalidResidualUniform) {
            _ = try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                ],
                acceptanceUniforms: [0.75],
                terminalDraws: [.residual(.nan)],
                bonusTargetDistribution: [1.0])
        }
    }

    private var partialRejectionSteps: [SampledMTPBlockStep] {
        [
            .init(
                targetDistribution: [0.2, 0.5, 0.3],
                draftDistribution: [0.4, 0.4, 0.2],
                proposedToken: 1),
            .init(
                targetDistribution: [0.4, 0.2, 0.4],
                draftDistribution: [0.1, 0.6, 0.3],
                proposedToken: 1),
        ]
    }

    private func assertBlockThrows(
        _ expected: SampledMTPBlockAcceptanceError,
        _ expression: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line)
    {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            XCTAssertEqual(error as? SampledMTPBlockAcceptanceError, expected, file: file, line: line)
        }
    }

    private func assertResidualThrows(
        _ expected: SampledMTPResidualCorrectionError,
        _ expression: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line)
    {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            XCTAssertEqual(error as? SampledMTPResidualCorrectionError, expected, file: file, line: line)
        }
    }
}

private extension SampledMTPBlockStep {
    static func alwaysAccepting(proposedToken: Int) -> Self {
        .init(
            targetDistribution: [1.0],
            draftDistribution: [1.0],
            proposedToken: proposedToken)
    }
}
