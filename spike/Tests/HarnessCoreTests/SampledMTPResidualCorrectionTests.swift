import XCTest
@testable import HarnessCore

final class SampledMTPResidualCorrectionTests: XCTestCase {
    func testAcceptanceProbabilityUsesMinOneTargetOverDraft() throws {
        let trace = try SampledMTPResidualCorrection.decide(
            target: [0.2, 0.5, 0.3],
            draft: [0.4, 0.4, 0.2],
            proposedToken: 1,
            acceptanceUniform: 0.99)

        XCTAssertEqual(trace.proposedToken, 1)
        XCTAssertEqual(trace.acceptanceProbability, 1.0, accuracy: 1e-15)
        XCTAssertEqual(trace.acceptanceUniform, 0.99)
        XCTAssertNil(trace.residualUniform)
        XCTAssertNil(trace.residualDistribution)
        XCTAssertEqual(trace.decision, .accepted(token: 1))
        XCTAssertEqual(trace.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.99),
        ])
    }

    func testRejectedProposalSamplesNormalizedPositiveResidual() throws {
        let trace = try SampledMTPResidualCorrection.decide(
            target: [0.5, 0.25, 0.25],
            draft: [0.25, 0.5, 0.25],
            proposedToken: 1,
            acceptanceUniform: 0.75,
            residualUniform: 0.4)

        XCTAssertEqual(trace.acceptanceProbability, 0.5, accuracy: 1e-15)
        XCTAssertEqual(trace.residualUniform, 0.4)
        assertEqualDoubles(trace.residualDistribution ?? [], [1.0, 0.0, 0.0], accuracy: 1e-15)
        XCTAssertEqual(trace.decision, .rejected(correctionToken: 0))
        XCTAssertEqual(trace.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.75),
            .init(purpose: .residual, value: 0.4),
        ])
    }

    func testAnalyticallyReconstructsTargetDistributionAcrossProposalAcceptanceAndCorrection() throws {
        let target = [0.5, 0.25, 0.25]
        let draft = [0.25, 0.5, 0.25]

        let acceptance = try (0..<3).map { token in
            try SampledMTPResidualCorrection.acceptanceProbability(
                target: target,
                draft: draft,
                proposedToken: token)
        }
        assertEqualDoubles(acceptance, [1.0, 0.5, 1.0], accuracy: 1e-15)

        let residual = try SampledMTPResidualCorrection.residualDistribution(
            target: target,
            draft: draft)
        assertEqualDoubles(residual, [1.0, 0.0, 0.0], accuracy: 1e-15)

        let totalRejectedProposalMass = zip(draft, acceptance).reduce(0.0) { partial, pair in
            partial + pair.0 * (1.0 - pair.1)
        }
        XCTAssertEqual(totalRejectedProposalMass, 0.25, accuracy: 1e-15)

        let reconstructed = (0..<3).map { token in
            draft[token] * acceptance[token] + totalRejectedProposalMass * residual[token]
        }
        assertEqualDoubles(reconstructed, target, accuracy: 1e-15)
        assertEqualDoubles(
            try SampledMTPResidualCorrection.reconstructTargetLaw(target: target, draft: draft),
            target,
            accuracy: 1e-15)
    }

    func testProposalSelectionUsesDraftCumulativeTokenOrder() throws {
        XCTAssertEqual(
            try SampledMTPResidualCorrection.proposalToken(
                draft: [0.2, 0.5, 0.3],
                proposalUniform: 0.0),
            0)
        XCTAssertEqual(
            try SampledMTPResidualCorrection.proposalToken(
                draft: [0.2, 0.5, 0.3],
                proposalUniform: 0.69),
            1)
        XCTAssertEqual(
            try SampledMTPResidualCorrection.proposalToken(
                draft: [0.2, 0.5, 0.3],
                proposalUniform: 0.99),
            2)
    }

    func testDistributionLawAcrossFixedProbabilityTable() throws {
        let cases = [
            (target: [0.6, 0.3, 0.1], draft: [0.2, 0.5, 0.3]),
            (target: [0.1, 0.2, 0.3, 0.4], draft: [0.4, 0.3, 0.2, 0.1]),
            (target: [0.25, 0.25, 0.5], draft: [0.0, 0.5, 0.5]),
            (target: [0.2, 0.3, 0.5], draft: [0.2, 0.3, 0.5]),
        ]

        for distributionCase in cases {
            var acceptance = Array(repeating: 0.0, count: distributionCase.target.count)
            for token in distributionCase.draft.indices where distributionCase.draft[token] > 0 {
                acceptance[token] = try SampledMTPResidualCorrection.acceptanceProbability(
                    target: distributionCase.target,
                    draft: distributionCase.draft,
                    proposedToken: token)
            }

            let rejectedMass = zip(distributionCase.draft, acceptance).reduce(0.0) { partial, pair in
                partial + pair.0 * (1.0 - pair.1)
            }
            let residual = rejectedMass > 0
                ? try SampledMTPResidualCorrection.residualDistribution(
                    target: distributionCase.target,
                    draft: distributionCase.draft)
                : Array(repeating: 0.0, count: distributionCase.target.count)
            let reconstructed = distributionCase.target.indices.map { token in
                distributionCase.draft[token] * acceptance[token] + rejectedMass * residual[token]
            }

            assertEqualDoubles(reconstructed, distributionCase.target, accuracy: 1e-12)
        }
    }

    func testDeterministicAcceptedTraceConsumesOnlyAcceptanceUniform() throws {
        let first = try SampledMTPResidualCorrection.decide(
            target: [0.2, 0.5, 0.3],
            draft: [0.4, 0.4, 0.2],
            proposedToken: 1,
            acceptanceUniform: 0.25,
            residualUniform: 0.75)
        let second = try SampledMTPResidualCorrection.decide(
            target: [0.2, 0.5, 0.3],
            draft: [0.4, 0.4, 0.2],
            proposedToken: 1,
            acceptanceUniform: 0.25,
            residualUniform: 0.75)

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.decision, .accepted(token: 1))
        XCTAssertEqual(first.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.25),
        ])
        XCTAssertNil(first.residualUniform)
        XCTAssertNil(first.residualDistribution)
    }

    func testDeterministicRejectedTraceConsumesAcceptanceThenResidual() throws {
        let first = try SampledMTPResidualCorrection.decide(
            target: [0.4, 0.2, 0.4],
            draft: [0.1, 0.6, 0.3],
            proposedToken: 1,
            acceptanceUniform: 0.9,
            residualUniform: 0.6)
        let second = try SampledMTPResidualCorrection.decide(
            target: [0.4, 0.2, 0.4],
            draft: [0.1, 0.6, 0.3],
            proposedToken: 1,
            acceptanceUniform: 0.9,
            residualUniform: 0.6)

        XCTAssertEqual(first, second)
        assertEqualDoubles(first.residualDistribution ?? [], [0.75, 0.0, 0.25], accuracy: 1e-15)
        XCTAssertEqual(first.decision, .rejected(correctionToken: 0))
        XCTAssertEqual(first.consumedUniforms, [
            .init(purpose: .acceptance, value: 0.9),
            .init(purpose: .residual, value: 0.6),
        ])
    }

    func testMissingTargetSupportIsValidAlwaysRejectWhenResidualMassExists() throws {
        let trace = try SampledMTPResidualCorrection.decide(
            target: [0.0, 0.75, 0.25],
            draft: [0.5, 0.25, 0.25],
            proposedToken: 0,
            acceptanceUniform: 0.0,
            residualUniform: 0.7)

        XCTAssertEqual(trace.acceptanceProbability, 0.0, accuracy: 1e-15)
        assertEqualDoubles(trace.residualDistribution ?? [], [0.0, 1.0, 0.0], accuracy: 1e-15)
        XCTAssertEqual(trace.decision, .rejected(correctionToken: 1))
    }

    func testValidationFailsClosedWithTypedErrors() {
        assertThrows(
            .emptyDistribution,
            target: [],
            draft: [],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .mismatchedDistributionSizes(targetCount: 2, draftCount: 1),
            target: [0.5, 0.5],
            draft: [1.0],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .invalidTargetProbability(index: 0),
            target: [.infinity, 0.0],
            draft: [0.5, 0.5],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .invalidDraftProbability(index: 0),
            target: [0.5, 0.5],
            draft: [-0.1, 1.1],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .nonNormalizedTarget(sum: 0.9),
            target: [0.4, 0.5],
            draft: [0.5, 0.5],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .nonNormalizedDraft(sum: 0.9),
            target: [0.5, 0.5],
            draft: [0.4, 0.5],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .invalidProposalToken(token: 2, vocabularyCount: 2),
            target: [0.5, 0.5],
            draft: [0.5, 0.5],
            proposedToken: 2,
            acceptanceUniform: 0.0)
        assertThrows(
            .zeroDraftMass(token: 0),
            target: [0.5, 0.5],
            draft: [0.0, 1.0],
            proposedToken: 0,
            acceptanceUniform: 0.0)
        assertThrows(
            .invalidAcceptanceUniform,
            target: [0.5, 0.5],
            draft: [0.5, 0.5],
            proposedToken: 0,
            acceptanceUniform: 1.0)
        assertThrows(
            .invalidResidualUniform,
            target: [0.4, 0.2, 0.4],
            draft: [0.1, 0.6, 0.3],
            proposedToken: 1,
            acceptanceUniform: 0.9,
            residualUniform: .nan)
        assertThrows(
            .missingCorrectionUniform,
            target: [0.4, 0.2, 0.4],
            draft: [0.1, 0.6, 0.3],
            proposedToken: 1,
            acceptanceUniform: 0.9)
        XCTAssertThrowsError(
            try SampledMTPResidualCorrection.residualDistribution(
                target: [0.5, 0.5],
                draft: [0.5, 0.5])
        ) { error in
            XCTAssertEqual(error as? SampledMTPResidualCorrectionError, .zeroResidualMass)
        }
    }

    private func assertThrows(
        _ expected: SampledMTPResidualCorrectionError,
        target: [Double],
        draft: [Double],
        proposedToken: Int,
        acceptanceUniform: Double,
        residualUniform: Double? = nil,
        file: StaticString = #filePath,
        line: UInt = #line)
    {
        XCTAssertThrowsError(
            try SampledMTPResidualCorrection.decide(
                target: target,
                draft: draft,
                proposedToken: proposedToken,
                acceptanceUniform: acceptanceUniform,
                residualUniform: residualUniform),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(error as? SampledMTPResidualCorrectionError, expected, file: file, line: line)
        }
    }
}

private extension XCTestCase {
    func assertEqualDoubles(
        _ actual: [Double],
        _ expected: [Double],
        accuracy: Double,
        file: StaticString = #filePath,
        line: UInt = #line)
    {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (actualValue, expectedValue) in zip(actual, expected) {
            XCTAssertEqual(actualValue, expectedValue, accuracy: accuracy, file: file, line: line)
        }
    }
}
