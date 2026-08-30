import XCTest

import ServingCore

@testable import fastmlx_harness

final class Qwen38ScorecardContinuousRouteTests: XCTestCase {
    func testC2FixtureTraversesBackendContinuousRouteAndProvesSharedBatch()
        async throws
    {
        let result = try await Qwen38ScorecardContinuousRouteRunner
            .runFixture(concurrency: 2)

        XCTAssertEqual(result.concurrency, 2)
        XCTAssertEqual(result.evidenceKind, .syntheticPathProof)
        XCTAssertEqual(result.coordinatorRequestIDs, [1, 2])
        XCTAssertEqual(result.sharedBatchDecodeRequestIDs, [1, 2])
        XCTAssertTrue(result.coordinatorPlanObservations.contains {
            $0.decodeKind == .batch
                && $0.decodeRequestIDs == [1, 2]
                && !$0.speculationAllowed
        })
        XCTAssertEqual(result.peakBatchOccupancy, 2)
        XCTAssertEqual(result.peakActiveSlots, 2)
        XCTAssertEqual(result.finalActiveRequests, 0)
        XCTAssertEqual(result.finalCoordinatorSlots, 0)
        XCTAssertEqual(result.finalReservedKVBytes, 0)
        XCTAssertEqual(result.requests.map(\.outputTokenIDs), [[2_000], [2_001]])
        XCTAssertEqual(result.requests.map(\.usage.completionTokens), [1, 1])
        XCTAssertEqual(result.requests.map(\.finishReason), [.stop, .stop])
        XCTAssertTrue(result.requests.allSatisfy {
            $0.route == .continuousBatchNoSpec
        })
        XCTAssertNoThrow(try result.validate())
    }

    func testC4FixtureTraversesBackendContinuousRouteAndProvesSharedBatch()
        async throws
    {
        let result = try await Qwen38ScorecardContinuousRouteRunner
            .runFixture(concurrency: 4)

        XCTAssertEqual(result.concurrency, 4)
        XCTAssertEqual(result.evidenceKind, .syntheticPathProof)
        XCTAssertEqual(result.coordinatorRequestIDs, [1, 2, 3, 4])
        XCTAssertEqual(result.sharedBatchDecodeRequestIDs, [1, 2, 3, 4])
        XCTAssertTrue(result.coordinatorPlanObservations.contains {
            $0.decodeKind == .batch
                && $0.decodeRequestIDs == [1, 2, 3, 4]
                && !$0.speculationAllowed
        })
        XCTAssertEqual(result.peakBatchOccupancy, 4)
        XCTAssertEqual(result.peakActiveSlots, 4)
        XCTAssertEqual(result.requests.map(\.outputTokenIDs), [
            [2_000], [2_001], [2_002], [2_003],
        ])
        XCTAssertNoThrow(try result.validate())
    }

    func testFixtureRejectsUnsupportedConcurrencyBeforeAdmission()
        async throws
    {
        do {
            _ = try await Qwen38ScorecardContinuousRouteRunner
                .runFixture(concurrency: 3)
            XCTFail("expected exact C2/C4 concurrency rejection")
        } catch let error as Qwen38ScorecardContinuousRouteError {
            XCTAssertEqual(error, .invalidConcurrency(3))
        }
    }

    func testResultValidationRejectsSyntheticIncompleteEvidence() throws {
        let valid = makeValidResult()

        XCTAssertThrowsError(try makeValidResult(
            sharedBatchDecodeRequestIDs: []
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .missingSharedBatchDecode)
        }

        XCTAssertThrowsError(try makeValidResult(
            finalActiveRequests: 1
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .incompleteCleanup(
                    activeRequests: 1,
                    coordinatorSlots: 0,
                    reservedKVBytes: 0))
        }

        XCTAssertThrowsError(try makeValidResult(
            peakActiveSlots: 3
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .occupancyExceeded(limit: 2, observed: 3))
        }

        XCTAssertThrowsError(try makeValidResult(
            peakActiveSlots: 1
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .inconsistentPeakSummary)
        }

        XCTAssertThrowsError(try makeValidResult(
            peakBatchOccupancy: 1
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .inconsistentPeakSummary)
        }

        var noOverlapRequests = valid.requests
        noOverlapRequests[0] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 0,
            coordinatorRequestID: 1,
            route: .continuousBatchNoSpec,
            outputTokenIDs: [2_000],
            finishReason: .stop,
            usage: noOverlapRequests[0].usage,
            admittedAtUptime: 4,
            completedAtUptime: 5)
        noOverlapRequests[1] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 1,
            coordinatorRequestID: 2,
            route: .continuousBatchNoSpec,
            outputTokenIDs: [2_001],
            finishReason: .stop,
            usage: noOverlapRequests[1].usage,
            admittedAtUptime: 6,
            completedAtUptime: 7)

        XCTAssertThrowsError(try makeValidResult(
            requests: noOverlapRequests
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .noRequestOverlap)
        }

        var wrongRouteRequests = valid.requests
        wrongRouteRequests[0] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 0,
            coordinatorRequestID: 1,
            route: .soloPLD,
            outputTokenIDs: [2_000],
            finishReason: .stop,
            usage: wrongRouteRequests[0].usage,
            admittedAtUptime: 1,
            completedAtUptime: 4)

        XCTAssertThrowsError(try makeValidResult(
            requests: wrongRouteRequests
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .unexpectedRoute(index: 0))
        }

        XCTAssertThrowsError(try makeValidResult(
            coordinatorPlanObservations: [
                Qwen38ScorecardContinuousRoutePlanObservation(
                    planSequence: 2,
                    stateRevisionAfterApply: 3,
                    admissions: [1, 2],
                    decodeKind: .batch,
                    decodeRequestIDs: [1, 2],
                    speculationAllowed: true,
                    prefillRequestIDs: [1, 2],
                    activeSlotCount: 2,
                    queuedSlotCount: 0),
            ]
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .speculationEnabled)
        }
    }

    private func makeValidResult(
        coordinatorPlanObservations:
            [Qwen38ScorecardContinuousRoutePlanObservation]? = nil,
        sharedBatchDecodeRequestIDs: [UInt64] = [1, 2],
        peakActiveSlots: Int = 2,
        peakBatchOccupancy: Int? = nil,
        finalActiveRequests: Int = 0,
        requests: [Qwen38ScorecardContinuousRouteRequestResult]? = nil
    ) -> Qwen38ScorecardContinuousRouteResult {
        Qwen38ScorecardContinuousRouteResult(
            evidenceKind: .syntheticPathProof,
            concurrency: 2,
            coordinatorRequestIDs: [1, 2],
            coordinatorPlanObservations: coordinatorPlanObservations ?? [
                Qwen38ScorecardContinuousRoutePlanObservation(
                    planSequence: 2,
                    stateRevisionAfterApply: 3,
                    admissions: [1, 2],
                    decodeKind: .none,
                    decodeRequestIDs: [],
                    speculationAllowed: false,
                    prefillRequestIDs: [1, 2],
                    activeSlotCount: 2,
                    queuedSlotCount: 0),
                Qwen38ScorecardContinuousRoutePlanObservation(
                    planSequence: 3,
                    stateRevisionAfterApply: 4,
                    admissions: [],
                    decodeKind: .batch,
                    decodeRequestIDs: [1, 2],
                    speculationAllowed: false,
                    prefillRequestIDs: [],
                    activeSlotCount: 0,
                    queuedSlotCount: 0),
            ],
            planRevisions: [
                Qwen38ScorecardContinuousRouteRevision(
                    planSequence: 2,
                    stateRevisionAfterApply: 3),
                Qwen38ScorecardContinuousRouteRevision(
                    planSequence: 3,
                    stateRevisionAfterApply: 4),
            ],
            sharedBatchDecodeRequestIDs: sharedBatchDecodeRequestIDs,
            peakActiveSlots: peakActiveSlots,
            peakBatchOccupancy:
                peakBatchOccupancy ?? sharedBatchDecodeRequestIDs.count,
            finalActiveRequests: finalActiveRequests,
            finalCoordinatorSlots: 0,
            finalReservedKVBytes: 0,
            requests: requests ?? [
                Qwen38ScorecardContinuousRouteRequestResult(
                    requestIndex: 0,
                    coordinatorRequestID: 1,
                    route: .continuousBatchNoSpec,
                    outputTokenIDs: [2_000],
                    finishReason: .stop,
                    usage: OpenAIChatUsage(
                        promptTokens: 1,
                        completionTokens: 1),
                    admittedAtUptime: 1,
                    completedAtUptime: 4),
                Qwen38ScorecardContinuousRouteRequestResult(
                    requestIndex: 1,
                    coordinatorRequestID: 2,
                    route: .continuousBatchNoSpec,
                    outputTokenIDs: [2_001],
                    finishReason: .stop,
                    usage: OpenAIChatUsage(
                        promptTokens: 1,
                        completionTokens: 1),
                    admittedAtUptime: 2,
                    completedAtUptime: 3),
            ])
    }
}
