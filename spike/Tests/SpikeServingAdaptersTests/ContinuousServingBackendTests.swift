import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ContinuousServingBackendTests: XCTestCase {
    func testDynamicAdmissionSingleHeldExpiryStartsSoloPLDWithSpeculation()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        let start = Task {
            try await backend.start(request(text: "solo", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let handle = try await start.value

        XCTAssertEqual(handle.route, .soloPLD)
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: true))

        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    /// The continuous-batch route decodes greedily (the compiled step folds argmax). A sampled
    /// request (temperature > 0) must be REJECTED here, never silently downgraded to greedy — the
    /// honesty invariant that replaces the removed global `temperature must be 0` guard. Native
    /// sampling is served on the scalar route used by hybrid models (e.g. Qwen3.8).
    func testContinuousBatchRouteRejectsSampledRequestInsteadOfSilentGreedy()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["solo": [10]],
            pieces: [1: "s"],
            stopTokenIDs: [99])

        var sampled = request(text: "solo", maxTokens: 2)
        sampled.temperature = 1.0

        do {
            _ = try await backend.start(sampled)
            XCTFail("continuous route must reject a sampled request, not serve it greedily")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                XCTFail("expected invalidRequest, got \(error)")
                await backend.shutdown()
                return
            }
            XCTAssertEqual(param, "temperature")
        }

        // A penalized request (greedy — temp 0 — but presence_penalty > 0) is ALSO rejected: the
        // compiled continuous path has no logit processor, so accepting it would be a silent no-op.
        var penalized = request(text: "solo", maxTokens: 2)
        penalized.presencePenalty = 1.5
        do {
            _ = try await backend.start(penalized)
            XCTFail("continuous route must reject a penalized request")
        } catch is OpenAIServingError {
            // expected
        }
        await backend.shutdown()
    }

    func testDynamicAdmissionTwoHeldRequestsExpireAsAtomicBatchNoSpec()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99])

        let oneStart = Task {
            try await backend.start(request(text: "one", maxTokens: 2))
        }
        let twoStart = Task {
            try await backend.start(request(text: "two", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 2
        }

        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let one = try await oneStart.value
        let two = try await twoStart.value

        XCTAssertEqual(one.route, .continuousBatchNoSpec)
        XCTAssertEqual(two.route, .continuousBatchNoSpec)
        let batchIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            batchIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .batch(
                [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionThirdArrivalWaitsForInFlightCoordinatorAdmissionWithoutResubmittingCohort()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 3, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                    30: [3, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
                "three": [30],
            ],
            pieces: [1: "a", 2: "b", 3: "c"],
            stopTokenIDs: [99],
            maximumBatchRequests: 3)

        let oneStart = Task {
            try await backend.start(request(text: "one", maxTokens: 2))
        }
        let twoStart = Task {
            try await backend.start(request(text: "two", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 2
        }

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 2
        }

        let threeStart = Task {
            try await backend.start(request(text: "three", maxTokens: 2))
        }
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let executingWhileCoordinatorBlocked =
            await backend.diagnosticExecutingAdmissionRequestCount()
        XCTAssertEqual(
            executingWhileCoordinatorBlocked,
            2,
            "the third request must wait at the coordinator preflight instead of reentering the in-flight cohort")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        let one = try await oneStart.value
        let two = try await twoStart.value
        let three = try await threeStart.value

        XCTAssertEqual(recorder.admissionBatchSizes, [2, 1])
        let coordinatorIDs =
            await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1), BatchRequestID(2), BatchRequestID(3)])

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        _ = await three.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationWhileCoordinatorAdmissionIsInFlight()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["in-flight": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "in-flight", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        start.cancel()
        do {
            _ = try await start.value
            XCTFail("Expected in-flight start cancellation")
        } catch is CancellationError {
        }
        let executingBeforePhysicalRemoval =
            await backend.diagnosticExecutingAdmissionRequestCount()
        XCTAssertEqual(
            executingBeforePhysicalRemoval,
            1,
            "the reducer reservation must remain held until the coordinator can physically remove the cancelled admission")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 0
        }
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        let cancellationSlots = await coordinator.snapshots()
        XCTAssertTrue(cancellationSlots.isEmpty)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationInFlightKeepsReplacementQueuedUntilPhysicalRemoval()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "cancelled": [10],
                "replacement": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 1)

        let cancelledStart = Task {
            try await backend.start(
                request(text: "cancelled", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let replacementStart = Task {
            try await backend.start(
                request(text: "replacement", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let queued =
                await backend.diagnosticQueuedAdmissionRequestCount()
            return held == 1 && queued == 1
        }

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let firstExpiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        cancelledStart.cancel()
        do {
            _ = try await cancelledStart.value
            XCTFail("Expected in-flight start cancellation")
        } catch is CancellationError {
        }

        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 0
        }
        let queuedBeforePhysicalRemoval =
            await backend.diagnosticQueuedAdmissionRequestCount()
        XCTAssertEqual(
            queuedBeforePhysicalRemoval,
            1,
            "replacement must not enter coordinator admission before the cancelled slot is physically removed")

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await firstExpiry.value
        await waitUntil {
            let queued =
                await backend.diagnosticQueuedAdmissionRequestCount()
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            return queued == 0 && held == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let replacement = try await replacementStart.value
        XCTAssertEqual(replacement.route, .soloPLD)
        XCTAssertEqual(recorder.admissionBatchSizes, [1, 1])

        _ = await replacement.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionShutdownCancelsCoordinatorAdmissionInFlight()
        async throws
    {
        let admissionGate = BlockingRuntimeResourceSnapshotGate()
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true,
                resourceSnapshotGate: admissionGate),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["in-flight": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "in-flight", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { admissionGate.hasEntered }
        let expiry = Task {
            await backend.diagnosticExpireAdmissionCoalescingWindow()
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 1
        }

        let shutdown = Task {
            await backend.shutdown()
        }
        do {
            _ = try await start.value
            XCTFail("Expected shutdown to release the in-flight start")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }

        admissionGate.release()
        _ = await coordinatorBlocker.value
        await expiry.value
        await shutdown.value
        XCTAssertEqual(recorder.admissionBatchSizes, [1])
        let shutdownSlots = await coordinator.snapshots()
        XCTAssertTrue(shutdownSlots.isEmpty)
    }

    func testDynamicAdmissionShutdownDuringValidationDoesNotEnqueueLatePendingStart()
        async throws
    {
        let validationBlocker = BlockingRuntimeResourceSnapshotGate()
        let renderGate = BlockingRuntimeResourceSnapshotGate()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                resourceSnapshotGate: validationBlocker),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["validating": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            renderGate: renderGate)

        let coordinatorBlocker = Task {
            await coordinator.runtimeResourceSnapshot()
        }
        await waitUntil { validationBlocker.hasEntered }

        let start = Task {
            try await backend.start(
                request(text: "validating", maxTokens: 2))
        }
        await waitUntil { renderGate.hasEntered }
        renderGate.release()
        let pendingWhileValidationBlocked =
            await backend.diagnosticPendingAdmissionRequestCount()
        XCTAssertEqual(pendingWhileValidationBlocked, 0)

        let shutdown = Task {
            await backend.shutdown()
        }
        await waitUntil {
            await !backend.diagnosticAcceptingRequests()
        }
        do {
            _ = try await backend.start(
                request(text: "validating", maxTokens: 2))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }

        validationBlocker.release()
        _ = await coordinatorBlocker.value
        await shutdown.value
        for _ in 0 ..< 100 {
            await Task.yield()
        }

        let pendingAfterShutdown =
            await backend.diagnosticPendingAdmissionRequestCount()
        if pendingAfterShutdown != 0 {
            start.cancel()
        }
        do {
            _ = try await start.value
            XCTFail("Expected validation-time shutdown rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        } catch is CancellationError {
            XCTFail("shutdown left a late pending start that required cancellation")
        }
        XCTAssertEqual(
            pendingAfterShutdown,
            0,
            "validation must not enqueue a continuation after shutdown snapshots pending admissions")
    }

    func testDynamicAdmissionArrivalAfterSoloPLDJoinsWithoutResubmittingExistingRequest()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 99],
                    20: [3, 99],
                ],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "solo": [10],
                "late": [20],
            ],
            pieces: [1: "a", 2: "b", 3: "c"],
            stopTokenIDs: [99])

        let soloStart = Task {
            try await backend.start(request(text: "solo", maxTokens: 3))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let solo = try await soloStart.value
        XCTAssertEqual(solo.route, .soloPLD)
        let soloConsumer = Task { try? await collect(solo.mailbox) }

        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: true))

        let lateStart = Task {
            try await backend.start(request(text: "late", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticExecutingAdmissionRequestCount() == 2
        }
        let late = try await lateStart.value
        XCTAssertEqual(late.route, .continuousBatchNoSpec)
        let lateConsumer = Task { try? await collect(late.mailbox) }

        let joinedCoordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            joinedCoordinatorIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        XCTAssertEqual(
            recorder.admissionBatchSizes,
            [1, 1],
            "late join should physically submit only the new request, not resubmit the solo request")

        do {
            while recorder.decodeActions.count < 3 {
                _ = try await coordinator.runOneTick()
            }
        } catch {
            let trace = await coordinator.executionTrace()
            let snapshots = await coordinator.snapshots()
            let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
            XCTFail(
                "adaptive tick failed: \(error); trace=\(trace); snapshots=\(snapshots); coordinatorIDs=\(coordinatorIDs)")
            _ = await solo.lease.cancel(.clientDisconnected)
            _ = await late.lease.cancel(.clientDisconnected)
            await backend.shutdown()
            _ = await soloConsumer.value
            _ = await lateConsumer.value
            return
        }
        XCTAssertEqual(
            Array(recorder.decodeActions.prefix(3)),
            [
                .solo(BatchRequestID(1), speculationAllowed: true),
                .drainSoloPipeline(BatchRequestID(1)),
                .batch(
                    [BatchRequestID(1), BatchRequestID(2)],
                    speculationAllowed: false),
            ])

        _ = await solo.lease.cancel(.clientDisconnected)
        _ = await late.lease.cancel(.clientDisconnected)
        await backend.shutdown()
        _ = await soloConsumer.value
        _ = await lateConsumer.value
    }

    func testDynamicAdmissionFailedLateJoinReleasesReservationAndPreservesSolo()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 2),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .aggregateKVByteLimitExceeded(
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "solo": [10],
                "late": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])

        let soloStart = Task {
            try await backend.start(request(text: "solo", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let solo = try await soloStart.value
        XCTAssertEqual(solo.route, .soloPLD)

        do {
            _ = try await backend.start(request(text: "late", maxTokens: 2))
            XCTFail("Expected the late coordinator admission to fail")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .capacityExceeded(retryAfterSeconds: 2))
        }

        let executingAfterFailure =
            await backend.diagnosticExecutingAdmissionRequestCount()
        let queuedAfterFailure =
            await backend.diagnosticQueuedAdmissionRequestCount()
        let coordinatorIDsAfterFailure =
            await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(executingAfterFailure, 1)
        XCTAssertEqual(queuedAfterFailure, 0)
        XCTAssertEqual(coordinatorIDsAfterFailure, [BatchRequestID(1)])

        _ = await solo.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellationWhileHeldReleasesPendingStart()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["held": [10]],
            pieces: [1: "h"],
            stopTokenIDs: [99])

        let start = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        start.cancel()

        do {
            _ = try await start.value
            XCTFail("Expected held start cancellation")
        } catch is CancellationError {
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 0
        }
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(coordinatorIDs, [])
        XCTAssertEqual(snapshots, [])
        await backend.shutdown()
    }

    func testDynamicAdmissionAutomaticWindowStartsWithoutDiagnosticIntervention()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["automatic": [10]],
            pieces: [1: "a"],
            stopTokenIDs: [99],
            coalescing: .automatic(.milliseconds(1)))

        let handle = try await backend.start(
            request(text: "automatic", maxTokens: 2))

        XCTAssertEqual(handle.route, .soloPLD)
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1)])
        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionCancellingLastHeldDisarmsAutomaticWindow()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "cancelled": [10],
                "replacement": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            coalescing: .automatic(.seconds(30)))

        let cancelledStart = Task {
            try await backend.start(
                request(text: "cancelled", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 1 && armed
        }
        let firstWindowArmed =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertTrue(firstWindowArmed)

        cancelledStart.cancel()
        _ = try? await cancelledStart.value
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 0 && !armed
        }
        let windowArmedAfterCancellation =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertFalse(
            windowArmedAfterCancellation,
            "the cancelled cohort must not leave a stale timer for the next request")

        let replacementStart = Task {
            try await backend.start(
                request(text: "replacement", maxTokens: 2))
        }
        await waitUntil {
            let held =
                await backend.diagnosticPendingAdmissionRequestCount()
            let armed =
                await backend.diagnosticAdmissionCoalescingWindowArmed()
            return held == 1 && armed
        }
        let replacementWindowArmed =
            await backend.diagnosticAdmissionCoalescingWindowArmed()
        XCTAssertTrue(replacementWindowArmed)
        await backend.diagnosticExpireAdmissionCoalescingWindow()
        let replacement = try await replacementStart.value

        _ = await replacement.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testDynamicAdmissionShutdownReleasesEveryHeldContinuation()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["held": [10]],
            pieces: [1: "h"],
            stopTokenIDs: [99])
        let start = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        await backend.shutdown()

        do {
            _ = try await start.value
            XCTFail("Expected shutdown to release the held start")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
        let pendingCount =
            await backend.diagnosticPendingAdmissionRequestCount()
        let snapshots = await coordinator.snapshots()
        XCTAssertEqual(
            pendingCount,
            0)
        XCTAssertTrue(snapshots.isEmpty)
    }

    func testDynamicAdmissionRejectsQueueOverflowBeforeCoordinatorAdmission()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder(),
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: [
                "held": [10],
                "overflow": [20],
            ],
            pieces: [1: "h", 2: "o"],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 0)
        let held = Task {
            try await backend.start(request(text: "held", maxTokens: 2))
        }
        await waitUntil {
            await backend.diagnosticPendingAdmissionRequestCount() == 1
        }

        do {
            _ = try await backend.start(
                request(text: "overflow", maxTokens: 2))
            XCTFail("Expected the bounded admission queue to reject overflow")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .queueFull(retryAfterSeconds: 2))
        }
        let snapshots = await coordinator.snapshots()
        XCTAssertTrue(snapshots.isEmpty)

        held.cancel()
        _ = try? await held.value
        await backend.shutdown()
    }

    func testDefaultConfigurationStillStartsImmediateBatchNoSpec()
        async throws
    {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 99]],
                recorder: recorder,
                allowsSpeculation: true),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["default": [10]],
            pieces: [1: "d"],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(text: "default", maxTokens: 2))

        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        let coordinatorIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            coordinatorIDs,
            [BatchRequestID(1)])
        try await runUntilDecode(coordinator, recorder: recorder)
        XCTAssertEqual(
            recorder.decodeActions.first,
            .solo(BatchRequestID(1), speculationAllowed: false))

        _ = await handle.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testContinuousRoutePublishesExactTextUsageAndResolvedStopSet() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 99, 3]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["hello": [10, 11]],
            pieces: [1: "hel", 2: "lo", 3: "hidden"],
            stopTokenIDs: [2_048, 99])

        let handle = try await backend.start(
            request(text: "hello", maxTokens: 8))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(
            events,
            [
                .text("hel"),
                .text("lo"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 2,
                            completionTokens: 2))),
            ])
        XCTAssertTrue(recorder.decodeActions.allSatisfy(\.speculationDisabled))
        await waitUntil {
            await backend.snapshot().activeRequests == 0
        }
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)
    }

    func testSnapshotReportsActiveRequestResourcesMLXMemoryAndPromptFreeJSON()
        async throws
    {
        let prompt = "fixture-secret-prompt"
        let output = "fixture-secret-output"
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 99]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [prompt: [10]],
            pieces: [1: output, 2: "!", 3: "?"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 64))

        let handle = try await backend.start(
            request(text: prompt, maxTokens: 8))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let activeSnapshot = await backend.snapshot()
        XCTAssertEqual(activeSnapshot.activeRequests, 1)
        XCTAssertEqual(activeSnapshot.coordinatorSlots, 1)
        XCTAssertEqual(activeSnapshot.reservedKVBytes, 1_024)
        XCTAssertEqual(activeSnapshot.maxReservedKVBytes, 4_096)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxActiveBytes, 0)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxCacheBytes, 0)
        XCTAssertGreaterThanOrEqual(activeSnapshot.mlxPeakBytes, 0)

        let encoded = try JSONEncoder().encode(activeSnapshot)
        let json = String(decoding: encoded, as: UTF8.self)
        XCTAssertFalse(json.contains(prompt))
        XCTAssertFalse(json.contains(output))

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .clientDisconnected)
        await backend.shutdown()

        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
    }

    func testRequestStopSplitAcrossTokenChunksCancelsCoordinatorNormally() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop": [10]],
            pieces: [
                1: "hello<",
                2: "stop",
                3: ">hidden",
                4: "tail",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop",
                maxTokens: 8,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("hello"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
        let finalSlots = await coordinator.snapshots()
        XCTAssertTrue(finalSlots.isEmpty)
    }

    func testRequestStopAtOutputBudgetReportsStopInsteadOfLength() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop-at-budget": [10]],
            pieces: [
                1: "a",
                2: "b<",
                3: "stop>",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop-at-budget",
                maxTokens: 3,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("a"),
                .text("b"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
    }

    func testDetectedStopReleasesCoordinatorBeforeBlockedPrefixPublication() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["blocked-stop": [10]],
            pieces: [
                1: "buffer",
                2: "prefix<stop>",
                3: "hidden",
            ],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 64))

        let handle = try await backend.start(
            request(
                text: "blocked-stop",
                maxTokens: 4,
                stop: ["<stop>"]))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        XCTAssertTrue(slots.isEmpty)
        XCTAssertEqual(resources?.reservedKVBytes, 0)

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .clientDisconnected)
    }

    func testQueueExhaustionDisconnectAndReplacementRecoverOneSharedCoordinator() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 99],
                    20: [3, 4, 99],
                    30: [5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
                "replacement": [30],
            ],
            pieces: [
                1: "a",
                2: "b",
                3: "c",
                4: "d",
                5: "r",
            ],
            stopTokenIDs: [99])

        let active = try await backend.start(
            request(text: "active", maxTokens: 2))
        _ = try await coordinator.runOneTick()
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))

        do {
            _ = try await backend.start(
                request(text: "replacement", maxTokens: 1))
            XCTFail("Expected queue-full rejection")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .queueFull(retryAfterSeconds: 2))
        }

        let queuedCancelled = await queued.lease.cancel(.clientDisconnected)
        XCTAssertTrue(queuedCancelled)
        await assertMailboxCancelled(
            queued.mailbox,
            reason: .clientDisconnected)
        let activeCancelled = await active.lease.cancel(.clientDisconnected)
        XCTAssertTrue(activeCancelled)
        await assertMailboxCancelled(
            active.mailbox,
            reason: .clientDisconnected)
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }

        let replacement = try await backend.start(
            request(text: "replacement", maxTokens: 1))
        let collector = Task { try await collect(replacement.mailbox) }
        while try await coordinator.runOneTick() {}
        let replacementEvents = try await collector.value
        XCTAssertEqual(
            replacementEvents,
            [
                .text("r"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 1))),
            ])
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)
    }

    func testRuntimeCapacityFailuresAreTypedBeforeHTTPGenerationStarts()
        async throws
    {
        let singleCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 1,
                error: .contextLimitExceeded(
                    BatchRequestID(1),
                    requested: 4_097,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let singleBackend = makeBackend(
            coordinator: singleCoordinator,
            promptByText: ["oversized": [10]],
            pieces: [:],
            stopTokenIDs: [99])

        do {
            _ = try await singleBackend.start(
                request(text: "oversized", maxTokens: 1))
            XCTFail("Expected a typed request-size admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }

        let sharedCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .aggregateKVByteLimitExceeded(
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let sharedBackend = makeBackend(
            coordinator: sharedCoordinator,
            promptByText: [
                "active": [10],
                "blocked": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let active = try await sharedBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await sharedCoordinator.runOneTick()

        do {
            _ = try await sharedBackend.start(
                request(text: "blocked", maxTokens: 1))
            XCTFail("Expected a typed aggregate-capacity admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .capacityExceeded(retryAfterSeconds: 2))
        }
        _ = await active.lease.cancel(.clientDisconnected)
        await sharedBackend.shutdown()

        let permanentCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .requestReservedKVByteLimitExceeded(
                    BatchRequestID(2),
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let permanentBackend = makeBackend(
            coordinator: permanentCoordinator,
            promptByText: [
                "active": [10],
                "permanent": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let permanentActive = try await permanentBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await permanentCoordinator.runOneTick()

        do {
            _ = try await permanentBackend.start(
                request(text: "permanent", maxTokens: 1))
            XCTFail("Expected permanent request oversize to remain non-retryable")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }
        _ = await permanentActive.lease.cancel(.clientDisconnected)
        await permanentBackend.shutdown()
    }

    func testDynamicAdmissionRejectsPermanentOversizeBeforeItOccupiesHoldOrQueue()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: AdmissionValidationFailureContinuousRuntime(
                error: .contextLimitExceeded(
                    BatchRequestID(1),
                    requested: 4_097,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let backend = makeDynamicBackend(
            coordinator: coordinator,
            promptByText: ["oversized": [10]],
            pieces: [:],
            stopTokenIDs: [99],
            maximumBatchRequests: 1,
            maximumQueuedRequests: 1)

        let start = Task {
            try await backend.start(
                request(text: "oversized", maxTokens: 1))
        }
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let pending =
            await backend.diagnosticPendingAdmissionRequestCount()
        XCTAssertEqual(
            pending,
            0,
            "permanently oversized work must fail before dynamic hold/queue admission")
        if pending != 0 {
            start.cancel()
        }
        do {
            _ = try await start.value
            XCTFail("Expected a typed request-size admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        } catch is CancellationError {
            XCTFail("oversized request reached the dynamic hold instead of failing preflight")
        }
        await backend.shutdown()
    }

    func testBoundedDiagnosticsExposeCoordinatorMembershipAndTraceOnly()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 8)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99])

        let one = try await backend.start(
            request(text: "one", maxTokens: 1))
        let two = try await backend.start(
            request(text: "two", maxTokens: 1))

        let requestIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            requestIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        let queuedSnapshots = await backend
            .diagnosticCoordinatorSnapshots()
        XCTAssertEqual(queuedSnapshots.map(\.request.id), [
            BatchRequestID(1), BatchRequestID(2),
        ])
        _ = try await coordinator.runOneTick()
        let trace = await backend.diagnosticCoordinatorExecutionTrace()
        XCTAssertTrue(trace.contains {
            if case .operation(.prefill(let slice)) = $0 {
                return slice.id == BatchRequestID(1)
                    || slice.id == BatchRequestID(2)
            }
            return false
        })

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testServingMailboxBackpressureStopsFurtherCoordinatorDecode() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["slow": [10]],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let handle = try await backend.start(
            request(text: "slow", maxTokens: 4))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
                && recorder.decodeActions.count >= 3
        }
        let blockedDecodeCount = recorder.decodeActions.count
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        XCTAssertEqual(recorder.decodeActions.count, blockedDecodeCount)

        let cancelled = await handle.lease.cancel(.backpressureTimeout)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .backpressureTimeout)
        await waitUntil {
            let snapshot = await backend.snapshot()
            return snapshot.activeRequests == 0
                && snapshot.coordinatorSlots == 0
                && snapshot.reservedKVBytes == 0
        }
    }

    func testShutdownCancelsEveryAcceptedRequestAndRejectsNewWork() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 3, 99],
                    20: [4, 5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
            ],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d", 5: "e"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let active = try await backend.start(
            request(text: "active", maxTokens: 3))
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        await backend.shutdown()
        await assertMailboxCancelled(active.mailbox, reason: .shutdown)
        await assertMailboxCancelled(queued.mailbox, reason: .shutdown)
        let activeState = await active.lease.state
        let queuedState = await queued.lease.state
        XCTAssertEqual(activeState, .cancelled(.shutdown))
        XCTAssertEqual(queuedState, .cancelled(.shutdown))
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.coordinatorSlots, 0)
        XCTAssertEqual(finalSnapshot.reservedKVBytes, 0)
        XCTAssertEqual(finalSnapshot.maxReservedKVBytes, 4_096)

        do {
            _ = try await backend.start(
                request(text: "active", maxTokens: 1))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }
}

private extension BatchDecodeAction {
    var speculationDisabled: Bool {
        switch self {
        case .drainSoloPipeline:
            true
        case .solo(_, let speculationAllowed),
            .batch(_, let speculationAllowed):
            !speculationAllowed
        }
    }
}

private final class ContinuousRuntimeRecorder: Sendable {
    private struct State: Sendable {
        var decodeActions: [BatchDecodeAction] = []
        var admissionBatchSizes: [Int] = []
        var activeRequests = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var decodeActions: [BatchDecodeAction] {
        state.withLock { $0.decodeActions }
    }

    var admissionBatchSizes: [Int] {
        state.withLock { $0.admissionBatchSizes }
    }

    func record(_ action: BatchDecodeAction) {
        state.withLock { $0.decodeActions.append(action) }
    }

    func admitted(_ count: Int) {
        state.withLock {
            $0.admissionBatchSizes.append(count)
            $0.activeRequests += count
        }
    }

    func removed() {
        state.withLock { state in
            state.activeRequests = max(0, state.activeRequests - 1)
        }
    }

    var resources: ContinuousBatchRuntimeResourceSnapshot {
        state.withLock { state in
            ContinuousBatchRuntimeResourceSnapshot(
                kvBytesPerToken: 64,
                reservedKVBytes: state.activeRequests * 1_024,
                maxReservedKVBytes: 4_096)
        }
    }
}

private final class BlockingRuntimeResourceSnapshotGate: Sendable {
    private struct State: Sendable {
        var hasEntered = false
        var isReleased = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var hasEntered: Bool {
        state.withLock(\.hasEntered)
    }

    func wait() {
        state.withLock { $0.hasEntered = true }
        while !state.withLock(\.isReleased) {
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    func release() {
        state.withLock { $0.isReleased = true }
    }
}

private final class FixtureContinuousRuntime: ContinuousBatchRuntime {
    private struct Slot {
        var processedTokens = 0
        var ready = false
        var soloPipelineState: BatchSoloPipelineState = .canonical
        var script: [Int]
        var cursor = 0
    }

    private let scriptsByPromptHead: [Int: [Int]]
    private let recorder: ContinuousRuntimeRecorder
    private let allowsSpeculation: Bool
    private let speculativeSoloState: BatchSoloPipelineState
    private let resourceSnapshotGate: BlockingRuntimeResourceSnapshotGate?
    private var admittedIDs: Set<BatchRequestID> = []
    private var promptHeadByID: [BatchRequestID: Int] = [:]
    private var slots: [BatchRequestID: Slot] = [:]

    init(
        scriptsByPromptHead: [Int: [Int]],
        recorder: ContinuousRuntimeRecorder,
        allowsSpeculation: Bool = false,
        speculativeSoloState: BatchSoloPipelineState = .speculative,
        resourceSnapshotGate: BlockingRuntimeResourceSnapshotGate? = nil
    ) {
        self.scriptsByPromptHead = scriptsByPromptHead
        self.recorder = recorder
        self.allowsSpeculation = allowsSpeculation
        self.speculativeSoloState = speculativeSoloState
        self.resourceSnapshotGate = resourceSnapshotGate
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admittedIDs.formUnion(admissions.map(\.id))
        recorder.admitted(admissions.count)
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        resourceSnapshotGate?.wait()
        return recorder.resources
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        var slot: Slot
        if let existing = slots[work.id] {
            slot = existing
        } else {
            guard work.startToken == 0,
                let head = work.tokens.first,
                let script = scriptsByPromptHead[head]
            else {
                throw FixtureContinuousRuntimeError.invalidPrefill
            }
            slot = Slot(script: script)
            promptHeadByID[work.id] = head
        }
        guard work.startToken == slot.processedTokens else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        recorder.record(action)
        let ids: [BatchRequestID]
        let stateAfter: BatchSoloPipelineState
        let outputlessSpeculativeDrain: Bool
        switch action {
        case .solo(let id, let speculationAllowed):
            guard !speculationAllowed || allowsSpeculation else {
                throw FixtureContinuousRuntimeError.speculation
            }
            ids = [id]
            stateAfter = speculationAllowed
                ? speculativeSoloState
                : .pipelinedLookahead
            outputlessSpeculativeDrain = false
        case .drainSoloPipeline(let id):
            guard slots[id]?.soloPipelineState.requiresDrain == true else {
                throw FixtureContinuousRuntimeError.invalidDrain
            }
            ids = [id]
            stateAfter = .canonical
            outputlessSpeculativeDrain =
                slots[id]?.soloPipelineState == .speculative
        case .batch(let batchIDs, let speculationAllowed):
            guard !speculationAllowed || allowsSpeculation else {
                throw FixtureContinuousRuntimeError.speculation
            }
            guard batchIDs.allSatisfy({
                slots[$0]?.soloPipelineState == .canonical
            }) else {
                throw FixtureContinuousRuntimeError.invalidBatch
            }
            ids = batchIDs
            stateAfter = .canonical
            outputlessSpeculativeDrain = false
        }

        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw FixtureContinuousRuntimeError.decodeBeforeReady
            }
            if outputlessSpeculativeDrain {
                slot.soloPipelineState = .canonical
                slots[id] = slot
                return ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: [],
                    finished: false,
                    soloPipelineState: .canonical)
            }
            let tokens: [Int]
            let finished: Bool
            if slot.cursor < slot.script.count {
                tokens = [slot.script[slot.cursor]]
                slot.cursor += 1
                finished = false
            } else {
                tokens = []
                finished = true
            }
            slot.soloPipelineState = stateAfter
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: tokens,
                finished: finished,
                soloPipelineState: stateAfter)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
        if admittedIDs.remove(id) != nil {
            recorder.removed()
        }
        promptHeadByID[id] = nil
    }
}

private final class AdmissionFailureContinuousRuntime:
    ContinuousBatchRuntime
{
    private let failureOnAdmission: Int
    private let error: DenseContinuousBatchRuntimeError
    private var admissionCount = 0

    init(
        failureOnAdmission: Int,
        error: DenseContinuousBatchRuntimeError
    ) {
        self.failureOnAdmission = failureOnAdmission
        self.error = error
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admissionCount += 1
        if admissionCount == failureOnAdmission {
            throw error
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private final class AdmissionValidationFailureContinuousRuntime:
    ContinuousBatchRuntime
{
    private let error: DenseContinuousBatchRuntimeError

    init(error: DenseContinuousBatchRuntimeError) {
        self.error = error
    }

    func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        throw error
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private enum FixtureContinuousRuntimeError: Error {
    case invalidPrefill
    case decodeBeforeReady
    case invalidDrain
    case invalidBatch
    case speculation
}

private struct FixtureContinuousTextCodec: ScalarServingTextCodec {
    let promptByText: [String: [Int]]
    let pieces: [Int: String]
    let renderGate: BlockingRuntimeResourceSnapshotGate?

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        renderGate?.wait()
        guard let text = messages.last?.text,
            let prompt = promptByText[text]
        else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        return prompt
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureContinuousDetokenizer(pieces: pieces)
    }
}

private struct FixtureContinuousDetokenizer: ScalarServingDetokenizer {
    let pieces: [Int: String]
    private var pending: String?

    init(pieces: [Int: String]) {
        self.pieces = pieces
    }

    mutating func append(token: Int) {
        pending = pieces[token]
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

private func configuration(
    active: Int,
    queued: Int
) throws -> ContinuousBatchConfiguration {
    try ContinuousBatchConfiguration(
        maxActiveSlots: active,
        maxPrefillSlots: active,
        prefillChunkSize: 8,
        maxQueuedRequests: queued)
}

private func makeBackend(
    coordinator: ContinuousBatchCoordinator,
    promptByText: [String: [Int]],
    pieces: [Int: String],
    stopTokenIDs: Set<Int>,
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 2,
        maxBytes: 4_096)
) -> ContinuousServingBackend {
    ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: FixtureContinuousTextCodec(
            promptByText: promptByText,
            pieces: pieces,
            renderGate: nil),
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity))
}

private func makeDynamicBackend(
    coordinator: ContinuousBatchCoordinator,
    promptByText: [String: [Int]],
    pieces: [Int: String],
    stopTokenIDs: Set<Int>,
    maximumBatchRequests: Int = 2,
    maximumQueuedRequests: Int = 4,
    coalescing: ContinuousServingAdmissionCoalescing = .manualDiagnostic,
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 2,
        maxBytes: 4_096),
    renderGate: BlockingRuntimeResourceSnapshotGate? = nil
) -> ContinuousServingBackend {
    ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: FixtureContinuousTextCodec(
            promptByText: promptByText,
            pieces: pieces,
            renderGate: renderGate),
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity,
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: true,
                    maximumBatchRequests: maximumBatchRequests,
                    maximumQueuedRequests: maximumQueuedRequests),
                coalescing: coalescing)))
}

private func request(
    text: String,
    maxTokens: Int,
    stop: [String] = []
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture",
        messages: [
            OpenAIChatMessage(role: .user, text: text),
        ],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop)
}

private func collect(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private func assertMailboxCancelled(
    _ mailbox: BoundedDeltaMailbox,
    reason: ServingCancellationReason,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await mailbox.next()
        XCTFail("Expected mailbox cancellation", file: file, line: line)
    } catch let error as ServingMailboxError {
        XCTAssertEqual(
            error,
            .cancelled(reason),
            file: file,
            line: line)
    } catch {
        XCTFail(
            "Unexpected mailbox error: \(error)",
            file: file,
            line: line)
    }
}

private func runUntilDecode(
    _ coordinator: ContinuousBatchCoordinator,
    recorder: ContinuousRuntimeRecorder,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws {
    for _ in 0 ..< 100 {
        if !recorder.decodeActions.isEmpty {
            return
        }
        let advanced = try await coordinator.runOneTick()
        if !advanced {
            break
        }
    }
    XCTFail("Coordinator did not reach decode", file: file, line: line)
}

private func waitUntil(
    attempts: Int = 10_000,
    _ predicate: () async -> Bool
) async {
    for _ in 0 ..< attempts {
        if await predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}
