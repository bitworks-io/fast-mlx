import XCTest

@testable import HarnessCore

final class ContinuousBatchPlanObservationTests: XCTestCase {
    func testPlanObservationsExposeMonotonicRevisionsAndBoundedMembership()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 2,
                maxPrefillSlots: 2,
                prefillChunkSize: 8,
                maxQueuedRequests: 2),
            runtime: ObservationRuntime(),
            automaticDrive: false,
            traceLimit: 16)
        let handles = try await coordinator.submitBatch([
            ContinuousBatchSubmission(
                promptTokens: [10],
                maxOutputTokens: 1,
                stopTokenIDs: [99],
                architecture: .denseAttention),
            ContinuousBatchSubmission(
                promptTokens: [20],
                maxOutputTokens: 1,
                stopTokenIDs: [99],
                architecture: .denseAttention),
        ])
        defer { _ = handles }

        while try await coordinator.runOneTick() {}

        let observations = await coordinator.takePlanObservations()
        guard observations.count == 2 else {
            XCTFail("expected 2 committed plan observations, got \(observations.count): \(observations)")
            return
        }
        XCTAssertEqual(observations.map(\.planSequence), [2, 3])
        XCTAssertEqual(observations.map(\.stateRevisionAfterApply), [3, 4])
        XCTAssertEqual(observations[0].admissions, [
            BatchRequestID(1), BatchRequestID(2),
        ])
        XCTAssertEqual(observations[0].prefillIDs, [
            BatchRequestID(1), BatchRequestID(2),
        ])
        XCTAssertEqual(observations[0].activeSlotCount, 2)
        XCTAssertEqual(observations[1].activeSlotCount, 0)
        XCTAssertEqual(
            observations[1].decode,
            .batch(
                requestIDs: [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))
        let emptiedObservations = await coordinator.takePlanObservations()
        XCTAssertEqual(emptiedObservations, [])
    }
}

private final class ObservationRuntime: ContinuousBatchRuntime {
    private var slots: [BatchRequestID: Bool] = [:]

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        for admission in admissions {
            slots[admission.id] = false
        }
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        slots[work.id] = true
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        guard case .batch(let ids, false) = action else {
            return []
        }
        return ids.map {
            ContinuousBatchRuntimeDecodeResult(
                id: $0,
                tokens: [1],
                finished: false,
                soloPipelineState: .canonical)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
    }
}
