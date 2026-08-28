import XCTest
@testable import HarnessCore

final class SampledMTPBlockDistributionCacheGateTests: XCTestCase {
    func testDefaultFixtureFreezesStatisticalInputsBeforeEvaluation() {
        let configuration = SampledMTPBlockDistributionCacheGate.Configuration.defaultFixture

        XCTAssertEqual(configuration.blockSeed, 0x5eed_5eed_2026_0827)
        XCTAssertEqual(configuration.targetSamplerSeed, 0x7461_7267_6574_0827)
        XCTAssertEqual(configuration.sampleCount, 200_000)
        XCTAssertEqual(configuration.distributionTolerance, 0.012)
        XCTAssertEqual(configuration.analyticTolerance, 1e-12)
    }

    func testDefaultFixturePassesAnalyticAndStatisticalDistributionChecks() throws {
        let verdict = try SampledMTPBlockDistributionCacheGate.runDefaultFixture()

        XCTAssertEqual(verdict.blockSeed, 0x5eed_5eed_2026_0827)
        XCTAssertEqual(verdict.targetSamplerSeed, 0x7461_7267_6574_0827)
        XCTAssertEqual(verdict.sampleCount, 200_000)
        XCTAssertTrue(verdict.passed)
        XCTAssertEqual(
            Set(verdict.analyticChecks.map(\.outcome)),
            [.rejectFirst, .rejectSecond, .acceptAll])
        XCTAssertEqual(verdict.analyticChecks.filter(\.passed).count, 3)
        XCTAssertEqual(verdict.distributionChecks.count, 13)
        XCTAssertTrue(verdict.distributionChecks.allSatisfy(\.passed))
        XCTAssertTrue(verdict.distributionChecks.allSatisfy { $0.mtpObservationCount > 0 })
        XCTAssertTrue(verdict.distributionChecks.allSatisfy { $0.targetSamplerObservationCount > 0 })
    }

    func testCacheCompositionUsesAcceptedDraftCountWithoutCommittingTerminalOutput() throws {
        let verdict = try SampledMTPBlockDistributionCacheGate.runDefaultFixture()

        XCTAssertEqual(verdict.cacheChecks.count, 6)
        for layout in [SampledMTPBlockDistributionCacheGate.CacheLayoutEvidence.dense,
                       .qwenStyleHybrid]
        {
            for acceptedDraftCount in 0...2 {
                let check = try XCTUnwrap(verdict.cacheChecks.first {
                    $0.layout == layout && $0.acceptedDraftCount == acceptedDraftCount
                })
                XCTAssertTrue(check.passed)
                XCTAssertEqual(check.proposedDraftTokens, [0, 1])
                XCTAssertEqual(check.committedInputTokens, [900] + Array([0, 1].prefix(acceptedDraftCount)))
                XCTAssertEqual(check.expectedCommittedInputTokens, check.committedInputTokens)
                XCTAssertFalse(check.terminalOutputWasCommitted)
                XCTAssertEqual(check.emittedTerminalToken, 8)
                XCTAssertEqual(check.rejectedDraftCount, 2 - acceptedDraftCount)
                XCTAssertEqual(check.expectedRejectedDraftCount, check.rejectedDraftCount)
                XCTAssertEqual(check.finalSnapshot, check.scalarEquivalentSnapshot)
            }
        }
    }

    func testCacheCompositionEvidenceRejectsContradictoryOutcomeFields() throws {
        let verdict = try SampledMTPBlockDistributionCacheGate.runDefaultFixture()
        let rejectFirst = try XCTUnwrap(verdict.cacheChecks.first {
            $0.layout == .dense && $0.outcome == .rejectFirst
        })

        let contradictory = SampledMTPBlockDistributionCacheGate.CacheCompositionCheck(
            layout: rejectFirst.layout,
            outcome: .acceptAll,
            acceptedDraftCount: rejectFirst.acceptedDraftCount,
            proposedDraftTokens: rejectFirst.proposedDraftTokens,
            expectedCommittedInputTokens: rejectFirst.expectedCommittedInputTokens,
            committedInputTokens: rejectFirst.committedInputTokens,
            expectedRejectedDraftCount: rejectFirst.expectedRejectedDraftCount,
            rejectedDraftCount: rejectFirst.rejectedDraftCount,
            emittedTerminalToken: rejectFirst.emittedTerminalToken,
            terminalOutputWasCommitted: rejectFirst.terminalOutputWasCommitted,
            finalSnapshot: rejectFirst.finalSnapshot,
            scalarEquivalentSnapshot: rejectFirst.scalarEquivalentSnapshot)

        XCTAssertFalse(contradictory.passed)

        let acceptAll = try XCTUnwrap(verdict.cacheChecks.first {
            $0.layout == .dense && $0.outcome == .acceptAll
        })
        let wrongWidth = SampledMTPBlockDistributionCacheGate.CacheCompositionCheck(
            layout: acceptAll.layout,
            outcome: acceptAll.outcome,
            acceptedDraftCount: acceptAll.acceptedDraftCount,
            proposedDraftTokens: acceptAll.proposedDraftTokens + [7],
            expectedCommittedInputTokens: acceptAll.expectedCommittedInputTokens,
            committedInputTokens: acceptAll.committedInputTokens,
            expectedRejectedDraftCount: 1,
            rejectedDraftCount: 1,
            emittedTerminalToken: acceptAll.emittedTerminalToken,
            terminalOutputWasCommitted: false,
            finalSnapshot: acceptAll.finalSnapshot,
            scalarEquivalentSnapshot: acceptAll.scalarEquivalentSnapshot)
        XCTAssertFalse(wrongWidth.passed)

        let hybridRejectFirst = try XCTUnwrap(verdict.cacheChecks.first {
            $0.layout == .qwenStyleHybrid && $0.outcome == .rejectFirst
        })
        let recurrentLayer = try XCTUnwrap(
            hybridRejectFirst.finalSnapshot.kinds.firstIndex(of: .recurrentState))
        var terminalSnapshot = hybridRejectFirst.finalSnapshot
        terminalSnapshot.recurrentStateByLayer[recurrentLayer].append(
            hybridRejectFirst.emittedTerminalToken + recurrentLayer)
        let hiddenTerminal = SampledMTPBlockDistributionCacheGate.CacheCompositionCheck(
            layout: hybridRejectFirst.layout,
            outcome: hybridRejectFirst.outcome,
            acceptedDraftCount: hybridRejectFirst.acceptedDraftCount,
            proposedDraftTokens: hybridRejectFirst.proposedDraftTokens,
            expectedCommittedInputTokens: hybridRejectFirst.expectedCommittedInputTokens,
            committedInputTokens: hybridRejectFirst.committedInputTokens,
            expectedRejectedDraftCount: hybridRejectFirst.expectedRejectedDraftCount,
            rejectedDraftCount: hybridRejectFirst.rejectedDraftCount,
            emittedTerminalToken: hybridRejectFirst.emittedTerminalToken,
            terminalOutputWasCommitted: false,
            finalSnapshot: terminalSnapshot,
            scalarEquivalentSnapshot: terminalSnapshot)
        XCTAssertFalse(hiddenTerminal.passed)
    }

    func testCacheTransactionFailuresAreWrappedAsTypedGateFailures() {
        XCTAssertThrowsError(
            try SampledMTPBlockDistributionCacheGate.runCacheBeginFailureFixtureForTesting()
        ) { error in
            guard case .cacheTransactionFailed(.unsupportedLayout) =
                error as? SampledMTPBlockDistributionCacheGate.Error
            else {
                return XCTFail("expected cacheTransactionFailed unsupportedLayout, got \(error)")
            }
        }

        XCTAssertThrowsError(
            try SampledMTPBlockDistributionCacheGate.runCacheFinalizeFailureFixtureForTesting()
        ) { error in
            guard case .cacheTransactionFailed(.scalarLengthMismatch) =
                error as? SampledMTPBlockDistributionCacheGate.Error
            else {
                return XCTFail("expected cacheTransactionFailed scalarLengthMismatch, got \(error)")
            }
        }
    }

    func testDistributionGateFailsClosedWhenFrozenToleranceCannotPass() {
        var configuration = SampledMTPBlockDistributionCacheGate.Configuration.defaultFixture
        configuration.distributionTolerance = 0.0

        XCTAssertThrowsError(
            try SampledMTPBlockDistributionCacheGate.run(configuration: configuration)
        ) { error in
            guard case .distributionMismatch(let check) = error as? SampledMTPBlockDistributionCacheGate.Error else {
                return XCTFail("expected distributionMismatch, got \(error)")
            }
            XCTAssertGreaterThan(check.maxAbsoluteDelta, 0.0)
        }
    }

    func testAdjacentFailClosedContractsRemainVisible() {
        XCTAssertThrowsError(
            try SampledMTPBlockAcceptance.decide(
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
                acceptanceUniforms: [0.75, 0.0],
                terminalDraws: [.residual(0.4)],
                bonusTargetDistribution: [0.1, 0.2, 0.7])
        ) { error in
            XCTAssertEqual(
                error as? SampledMTPBlockAcceptanceError,
                .extraAcceptanceUniforms(expected: 1, actual: 2))
        }

        XCTAssertThrowsError(
            try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                ],
                acceptanceUniforms: [0.75],
                terminalDraws: [],
                bonusTargetDistribution: [1.0])
        ) { error in
            XCTAssertEqual(
                error as? SampledMTPBlockAcceptanceError,
                .missingTerminalDraw(expected: .residual))
        }

        XCTAssertThrowsError(
            try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.5, 0.25, 0.25],
                        draftDistribution: [0.25, 0.5, 0.25],
                        proposedToken: 1),
                ],
                acceptanceUniforms: [0.75],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        ) { error in
            XCTAssertEqual(
                error as? SampledMTPBlockAcceptanceError,
                .wrongTerminalDraw(expected: .residual, actual: .bonus))
        }

        XCTAssertThrowsError(
            try SampledMTPBlockAcceptance.decide(
                steps: [
                    .init(
                        targetDistribution: [0.2, 0.2],
                        draftDistribution: [0.5, 0.5],
                        proposedToken: 0),
                ],
                acceptanceUniforms: [0.0],
                terminalDraws: [.bonus(0.0)],
                bonusTargetDistribution: [1.0])
        ) { error in
            XCTAssertEqual(
                error as? SampledMTPResidualCorrectionError,
                .nonNormalizedTarget(sum: 0.4))
        }
    }
}
