import XCTest
@testable import HarnessCore

final class SpeculativeEconomicsTests: XCTestCase {
    func testAcceptanceSummaryKeepsProposalAndRoundDenominatorsDistinct() throws {
        let summary = SpeculativeAcceptanceSummary(
            proposedDraftTokens: 9,
            acceptedDraftTokens: 4,
            verifyRounds: 3)

        XCTAssertEqual(try XCTUnwrap(summary.proposalAcceptanceRate), 4.0 / 9.0, accuracy: 1e-12)
        XCTAssertEqual(
            try XCTUnwrap(summary.acceptedDraftTokensPerRound), 4.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(
            try XCTUnwrap(summary.inclusiveAcceptanceLength), 1.0 + 4.0 / 3.0,
            accuracy: 1e-12)
    }

    func testInclusiveModelCardAcceptanceLengthConvertsToAcceptedDrafts() {
        XCTAssertEqual(
            SpeculativeAcceptanceSummary.acceptedDraftTokensPerRound(
                inclusiveAcceptanceLength: 2.49),
            1.49,
            accuracy: 1e-12)
    }

    func testAcceptanceSummaryLeavesUndefinedDenominatorsNil() {
        let summary = SpeculativeAcceptanceSummary(
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            verifyRounds: 0)

        XCTAssertNil(summary.proposalAcceptanceRate)
        XCTAssertNil(summary.acceptedDraftTokensPerRound)
        XCTAssertNil(summary.inclusiveAcceptanceLength)
    }

    func testPairingEconomicsDerivesItsOwnBreakEvenAndProjectedSpeedup() {
        // Preserved Qwen3-8B/DSpark serialized phase timings: a 16 ms base token,
        // 12 ms proposal, 3.7 ms Markov correction, 36.9 ms target verify, and
        // 1 ms accept/rollback. The old ~2.35 break-even belongs to this pairing.
        let economics = SpeculativeEconomics(
            baselineTokenSeconds: 0.016,
            acceptedDraftTokensPerRound: 1.46,
            timing: SpeculativePhaseTiming(
                draftSeconds: 0.0157,
                verifySeconds: 0.0369,
                commitSeconds: 0.0010))

        XCTAssertEqual(economics.speculativeRoundSeconds, 0.0536, accuracy: 1e-12)
        XCTAssertEqual(economics.roundCostRatio, 3.35, accuracy: 1e-12)
        XCTAssertEqual(economics.breakEvenAcceptedDraftTokensPerRound, 2.35, accuracy: 1e-12)
        XCTAssertEqual(economics.emittedTokensPerRound, 2.46, accuracy: 1e-12)
        XCTAssertEqual(economics.projectedSpeedup, 2.46 / 3.35, accuracy: 1e-12)
    }

    func testObservedThroughputComparisonNormalizesToItsOwnBaseline() {
        let comparison = SpeculativeThroughputComparison(
            baselineTokensPerSecond: 28.0,
            speculativeTokensPerSecond: 35.0)

        XCTAssertEqual(comparison.speedup, 1.25, accuracy: 1e-12)
        XCTAssertEqual(comparison.deltaPercent, 25.0, accuracy: 1e-12)
    }
}
