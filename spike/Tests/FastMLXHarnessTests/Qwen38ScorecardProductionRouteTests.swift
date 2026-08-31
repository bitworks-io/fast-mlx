import XCTest

import HarnessCore
import ServingCore
@_spi(ProductionRouteEvidence) import SpikeServingAdapters

@testable import fastmlx_harness

final class Qwen38ScorecardProductionRouteTests: XCTestCase {
    func testLoadedProductionRouteRunsC2AndC4ThroughSharedBackendBatch()
        async throws
    {
        for concurrency in [2, 4] {
            let loaded = makeLoadedProductionRouteFixture(
                concurrency: concurrency,
                maxCompletionTokens: 4)

            let result = try await Qwen38ScorecardProductionRouteRunner
                .runLoaded(
                    loaded,
                    concurrency: concurrency,
                    tokenTrace: .outputTokenIDs(
                        maxCompletedRequests: concurrency,
                        maxTokensPerRequest: 4),
                    request: productionRouteRequest(index:model:))

            XCTAssertEqual(result.evidenceKind, .liveProductionRoute)
            XCTAssertEqual(result.concurrency, concurrency)
            XCTAssertEqual(
                result.coordinatorRequestIDs,
                Array(1 ... UInt64(concurrency)))
            XCTAssertEqual(
                result.sharedBatchDecodeRequestIDs,
                Array(1 ... UInt64(concurrency)))
            XCTAssertEqual(Set(result.coordinatorRequestIDs).count, concurrency)
            XCTAssertTrue(result.coordinatorPlanObservations.contains {
                $0.decodeKind == .batch
                    && $0.decodeRequestIDs == Array(1 ... UInt64(concurrency))
                    && !$0.speculationAllowed
            })
            XCTAssertTrue(result.coordinatorPlanObservations.allSatisfy {
                !$0.speculationAllowed
            })
            XCTAssertEqual(result.peakBatchOccupancy, concurrency)
            XCTAssertLessThanOrEqual(result.peakActiveSlots, concurrency)
            XCTAssertEqual(result.finalActiveRequests, 0)
            XCTAssertEqual(result.finalCoordinatorSlots, 0)
            XCTAssertEqual(result.finalReservedKVBytes, 0)
            XCTAssertEqual(
                result.requests.map(\.outputTokenIDs),
                (0 ..< concurrency).map { [2_000 + $0, 3_000 + $0] })
            XCTAssertEqual(
                result.requests.map(\.usage.completionTokens),
                Array(repeating: 2, count: concurrency))
            XCTAssertTrue(result.requests.allSatisfy {
                $0.usage.completionTokens == $0.outputTokenIDs.count
                    && $0.route == .continuousBatchNoSpec
            })
            let latestAdmission = result.requests.map(\.admittedAtUptime).max()!
            let earliestCompletion =
                result.requests.map(\.completedAtUptime).min()!
            XCTAssertLessThan(latestAdmission, earliestCompletion)
            XCTAssertNoThrow(try result.validate())
        }
    }

    func testProductionRouteObservationDigestIsStableForSameTypedValues()
        throws
    {
        let c2 = makeValidProductionRouteResult()
        let c4 = makeValidProductionRouteResult(concurrency: 4)

        XCTAssertEqual(
            Qwen38ScorecardProductionRouteRunner.observationDigest(
                c2: c2,
                c4: c4),
            Qwen38ScorecardProductionRouteRunner.observationDigest(
                c2: c2,
            c4: c4))
    }

    func testRunLoadedClearsPriorPlanObservationsBeforeCurrentCohort()
        async throws
    {
        for concurrency in [2, 4] {
            let loaded = makeLoadedProductionRouteFixture(
                concurrency: concurrency,
                maxCompletionTokens: 4)
            try await seedUnrelatedPlanActivity(loaded)

            let result = try await Qwen38ScorecardProductionRouteRunner
                .runLoaded(
                    loaded,
                    concurrency: concurrency,
                    tokenTrace: .outputTokenIDs(
                        maxCompletedRequests: concurrency,
                        maxTokensPerRequest: 4),
                    request: productionRouteRequest(index:model:))

            let expectedIDs = Array(
                UInt64(3) ... UInt64(concurrency + 2))
            XCTAssertEqual(result.evidenceKind, .liveProductionRoute)
            XCTAssertEqual(result.coordinatorRequestIDs, expectedIDs)
            XCTAssertEqual(result.sharedBatchDecodeRequestIDs, expectedIDs)
            assertPlansOnlyReference(
                result.coordinatorPlanObservations,
                allowedIDs: Set(expectedIDs))
            XCTAssertNoThrow(try result.validate())
            XCTAssertEqual(
                Qwen38ScorecardProductionRouteRunner.observationDigest(
                    c2: concurrency == 2 ? result : makeValidProductionRouteResult(),
                    c4: concurrency == 4
                        ? result
                        : makeValidProductionRouteResult(concurrency: 4)),
                Qwen38ScorecardProductionRouteRunner.observationDigest(
                    c2: concurrency == 2 ? result : makeValidProductionRouteResult(),
                    c4: concurrency == 4
                        ? result
                        : makeValidProductionRouteResult(concurrency: 4)))
        }
    }

    func testRunLoadedUsesCompletedTokenTracesWhenRequestsFinishBeforeAdmissionSnapshot()
        async throws
    {
        let loaded = makeLoadedProductionRouteFixture(
            concurrency: 2,
            maxCompletionTokens: 4)
        let probe = ActiveSnapshotBeforeStartReturnProbe()
        let start = { @Sendable (
            index: Int,
            loaded: LoadedContinuousServingModel,
            authorization:
                ContinuousServingProductionRouteEvidenceAuthorization,
            tokenTrace: ContinuousServingOutputTokenTraceConfiguration,
            request: @Sendable (
                Int,
                String
            ) -> OpenAIChatCompletionRequest
        ) async throws -> Qwen38ScorecardProductionRouteStartResult in
            let handle = try await loaded.backend
                .startProductionRouteEvidence(
                    request(index, loaded.startupReport.launchedModel),
                    authorization: authorization,
                    tokenTrace: tokenTrace)
            return Qwen38ScorecardProductionRouteStartResult(
                handle: handle,
                afterRetained: {
                    while await loaded.backend.snapshot().activeRequests != 0 {
                        await Task.yield()
                    }
                    await probe.recordInactiveObservation()
                })
        }

        let result = try await Qwen38ScorecardProductionRouteRunner.runLoaded(
            loaded,
            concurrency: 2,
            tokenTrace: .outputTokenIDs(
                maxCompletedRequests: 2,
                maxTokensPerRequest: 4),
            request: productionRouteRequest(index:model:),
            start: start)

        XCTAssertEqual(result.evidenceKind, .liveProductionRoute)
        XCTAssertEqual(result.coordinatorRequestIDs, [1, 2])
        XCTAssertEqual(result.sharedBatchDecodeRequestIDs, [1, 2])
        XCTAssertEqual(result.finalActiveRequests, 0)
        XCTAssertEqual(result.finalCoordinatorSlots, 0)
        XCTAssertEqual(result.finalReservedKVBytes, 0)
        let inactiveObservationCount =
            await probe.inactiveObservationCount()
        XCTAssertEqual(inactiveObservationCount, 2)
        XCTAssertTrue(result.requests.allSatisfy {
            $0.usage.completionTokens == $0.outputTokenIDs.count
        })
        XCTAssertNoThrow(try result.validate())
    }

    func testRunLoadedRejectsInsufficientPlanTraceCapacityBeforeRequests()
        async throws
    {
        let loaded = makeLoadedProductionRouteFixture(
            concurrency: 2,
            maxCompletionTokens: 4,
            traceLimit: 0)

        await XCTAssertThrowsErrorAsync(
            try await Qwen38ScorecardProductionRouteRunner.runLoaded(
                loaded,
                concurrency: 2,
                tokenTrace: .outputTokenIDs(
                    maxCompletedRequests: 2,
                    maxTokensPerRequest: 4),
                request: productionRouteRequest(index:model:)))
        { error in
            XCTAssertEqual(
                error as? ContinuousServingProductionRouteEvidenceError,
                .insufficientScorecardTraceCapacity(
                    required: 6,
                    observed: 0))
        }
        let snapshot = await loaded.backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.coordinatorSlots, 0)
        XCTAssertEqual(snapshot.reservedKVBytes, 0)
    }

    func testRunLoadedCleansTracesAfterEvidenceFailureBeforeNextRun()
        async throws
    {
        let loaded = makeLoadedProductionRouteFixture(
            concurrency: 2,
            maxCompletionTokens: 4)

        await XCTAssertThrowsErrorAsync(
            try await Qwen38ScorecardProductionRouteRunner.runLoaded(
                loaded,
                concurrency: 2,
                tokenTrace: .outputTokenIDs(
                    maxCompletedRequests: 2,
                    maxTokensPerRequest: 1),
                request: productionRouteRequest(index:model:)))
        { error in
            guard case .outputTokenTraceTruncated = error as?
                ContinuousServingProductionRouteEvidenceError
            else {
                XCTFail("expected truncated output-token trace, got \(error)")
                return
            }
        }

        let result = try await Qwen38ScorecardProductionRouteRunner.runLoaded(
            loaded,
            concurrency: 2,
            tokenTrace: .outputTokenIDs(
                maxCompletedRequests: 2,
                maxTokensPerRequest: 4),
            request: productionRouteRequest(index:model:))

        XCTAssertEqual(result.coordinatorRequestIDs, [3, 4])
        assertPlansOnlyReference(
            result.coordinatorPlanObservations,
            allowedIDs: [3, 4])
        XCTAssertNoThrow(try result.validate())
    }

    func testRunLoadedCancelsPartialAdmissionAfterSiblingStartFailure()
        async throws
    {
        let loaded = makeLoadedProductionRouteFixture(
            concurrency: 2,
            maxCompletionTokens: 1_024,
            traceLimit: 2_048,
            allowsSoloDecode: true,
            outputTokensPerRequest: 1_024)
        let gate = PartialAdmissionStartGate()
        let start = { @Sendable (
            index: Int,
            loaded: LoadedContinuousServingModel,
            authorization:
                ContinuousServingProductionRouteEvidenceAuthorization,
            tokenTrace: ContinuousServingOutputTokenTraceConfiguration,
            request: @Sendable (
                Int,
                String
            ) -> OpenAIChatCompletionRequest
        ) async throws -> Qwen38ScorecardProductionRouteStartResult in
            if index == 1 {
                await gate.waitForIndex0Retained()
                throw PartialAdmissionFailureError
                    .startFailedAfterPeerAdmission
            }
            let handle = try await loaded.backend
                .startProductionRouteEvidence(
                    request(index, loaded.startupReport.launchedModel),
                    authorization: authorization,
                    tokenTrace: tokenTrace)
            let afterRetained: (@Sendable () async -> Void)? =
                if index == 0 {
                    { await gate.markIndex0Retained(handle.lease) }
                } else {
                    nil
                }
            return Qwen38ScorecardProductionRouteStartResult(
                handle: handle,
                afterRetained: afterRetained)
        }

        await XCTAssertThrowsErrorAsync(
            try await Qwen38ScorecardProductionRouteRunner.runLoaded(
                loaded,
                concurrency: 2,
                tokenTrace: .outputTokenIDs(
                    maxCompletedRequests: 2,
                    maxTokensPerRequest: 1_024),
                request: { index, model in
                    productionRouteRequest(
                        index: index,
                        model: model,
                        maxCompletionTokens: 1_024)
                },
                start: start))
        { error in
            XCTAssertEqual(
                error as? PartialAdmissionFailureError,
                .startFailedAfterPeerAdmission,
                "unexpected error: \(error)")
        }

        var snapshot = await loaded.backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.coordinatorSlots, 0)
        XCTAssertEqual(snapshot.reservedKVBytes, 0)
        let retainedLeaseState = await gate.index0LeaseState()
        XCTAssertEqual(
            retainedLeaseState,
            .cancelled(.clientDisconnected))

        let retry = try await Qwen38ScorecardProductionRouteRunner.runLoaded(
            loaded,
            concurrency: 2,
            tokenTrace: .outputTokenIDs(
                maxCompletedRequests: 2,
                maxTokensPerRequest: 4),
            request: productionRouteRequest(index:model:))

        XCTAssertEqual(retry.evidenceKind, .liveProductionRoute)
        XCTAssertNoThrow(try retry.validate())
        snapshot = await loaded.backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.coordinatorSlots, 0)
        XCTAssertEqual(snapshot.reservedKVBytes, 0)
    }

    func testDuplicateProductionTokenTraceFailsExplicitly() throws {
        let trace = ContinuousServingCompletedRequestTokenTrace(
            responseID: "duplicate",
            coordinatorRequestID: BatchRequestID(1),
            outputTokenIDs: [1],
            completionTokenCount: 1,
            truncated: false)

        XCTAssertThrowsError(
            try Qwen38ScorecardProductionRouteRunner
                .validateUniqueProductionTokenTraces([trace, trace])
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingProductionRouteEvidenceError,
                .duplicateOutputTokenTrace(responseID: "duplicate"))
        }
    }

    func testLiveProductionRouteAcceptsBoundedLengthCompletion()
        throws
    {
        var requests = makeValidProductionRouteResult().requests
        requests[0] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 0,
            coordinatorRequestID: 1,
            route: .continuousBatchNoSpec,
            outputTokenIDs: [2_000, 3_000],
            finishReason: .length,
            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 2),
            admittedAtUptime: 1,
            completedAtUptime: 4)

        XCTAssertNoThrow(try makeValidProductionRouteResult(
            requests: requests
        ).validate())
    }

    func testSyntheticFixtureStillRejectsLengthCompletion() throws {
        var synthetic = makeValidProductionRouteResult(
            evidenceKind: .syntheticPathProof)
        var requests = synthetic.requests
        requests[0] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 0,
            coordinatorRequestID: 1,
            route: .continuousBatchNoSpec,
            outputTokenIDs: [2_000],
            finishReason: .length,
            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1),
            admittedAtUptime: 1,
            completedAtUptime: 4)
        synthetic = makeValidProductionRouteResult(
            evidenceKind: .syntheticPathProof,
            requests: requests)

        XCTAssertThrowsError(try synthetic.validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .incompleteRequest(index: 0))
        }
    }

    func testManualLoadedWrapperCannotMintLiveProductionRouteEvidence()
        async throws
    {
        let loaded = LoadedContinuousServingModel(
            backend: makeProductionRouteFixtureBackend(),
            startupReport: makeStartupReport())

        await XCTAssertThrowsErrorAsync(
            try await Qwen38ScorecardProductionRouteRunner.runLoaded(
                loaded,
                concurrency: 2,
                tokenTrace: .outputTokenIDs(
                    maxCompletedRequests: 2,
                    maxTokensPerRequest: 4),
                request: productionRouteRequest(index:model:)))
        { error in
            XCTAssertEqual(
                error as? ContinuousServingProductionRouteEvidenceError,
                .missingLoadedModelProvenance)
        }
    }

    func testProductionRouteRejectsC1AndC3BeforeAuthorization()
        async throws
    {
        let loaded = LoadedContinuousServingModel(
            backend: makeProductionRouteFixtureBackend(),
            startupReport: makeStartupReport())

        for concurrency in [1, 3] {
            await XCTAssertThrowsErrorAsync(
                try await Qwen38ScorecardProductionRouteRunner.runLoaded(
                    loaded,
                    concurrency: concurrency,
                    tokenTrace: .outputTokenIDs(
                        maxCompletedRequests: concurrency,
                        maxTokensPerRequest: 4),
                    request: productionRouteRequest(index:model:)))
            { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardContinuousRouteError,
                    .invalidConcurrency(concurrency))
            }
        }
    }

    func testProductionRouteResultValidationRejectsOutputAccountingMismatch()
        throws
    {
        var requests = makeValidProductionRouteResult().requests
        requests[0] = Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: 0,
            coordinatorRequestID: 1,
            route: .continuousBatchNoSpec,
            outputTokenIDs: [2_000, 2_001],
            finishReason: .stop,
            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1),
            admittedAtUptime: 1,
            completedAtUptime: 4)

        XCTAssertThrowsError(try makeValidProductionRouteResult(
            requests: requests
        ).validate()) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardContinuousRouteError,
                .outputAccountingMismatch(
                    index: 0,
                    usageCompletionTokens: 1,
                    outputTokenCount: 2))
        }
    }

    func testProcessIsolationCollectsRealProcessFactsForSelf() throws {
        // XNU's proc_pidpath rejects buffer sizes LARGER than
        // PROC_PIDPATHINFO_MAXSIZE (4096) with EOVERFLOW, so a 16 KiB buffer
        // meant this collector could never succeed on any host. The failure
        // stayed latent because every earlier dedicated-host observation died
        // before the post-cleanup step that first exercises it live.
        // The collector also resolves qualification provenance, which
        // correctly refuses a dirty development tree. Reaching that provenance
        // step already proves the process-facts primitives work (the
        // historical defect threw workerError before ever getting there), so
        // a dirty-tree provenance refusal is tolerated; every other error
        // still fails the test.
        do {
            let evidence = try Qwen38MTPScorecardProcessFacts.processIsolation(
                mode: .gdnOn,
                observedEnv: Qwen38MTPScorecardProcessFacts.observedGDNEnv(
                    mode: .gdnOn))
            XCTAssertEqual(evidence.processID, Int(getpid()))
            XCTAssertEqual(evidence.executableSHA256.count, 64)
            XCTAssertTrue(evidence.executableSHA256.allSatisfy { character in
                character.isNumber || ("a" ... "f").contains(character)
            })
            XCTAssertNotEqual(
                evidence.executableSHA256,
                String(repeating: "0", count: 64))
            XCTAssertGreaterThan(evidence.processStartUptimeNanoseconds, 0)
        } catch let error as KVTunerQualificationCLIError {
            guard case .invalidHarnessGitSHA(let value) = error,
                value.hasSuffix("-dirty")
            else {
                throw error
            }
        }
    }
}

private actor ActiveSnapshotBeforeStartReturnProbe {
    private var count = 0

    func recordInactiveObservation() {
        count += 1
    }

    func inactiveObservationCount() -> Int {
        count
    }
}

private func productionRouteRequest(
    index: Int,
    model: String
) -> OpenAIChatCompletionRequest {
    productionRouteRequest(
        index: index,
        model: model,
        maxCompletionTokens: 4)
}

private func productionRouteRequest(
    index: Int,
    model: String,
    maxCompletionTokens: Int
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: model,
        messages: [
            OpenAIChatMessage(role: .user, text: "fixture-\(index)"),
        ],
        maxCompletionTokens: maxCompletionTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: [])
}

private func makeValidProductionRouteResult(
    evidenceKind: Qwen38ScorecardContinuousRouteEvidenceKind =
        .liveProductionRoute,
    concurrency: Int = 2,
    requests: [Qwen38ScorecardContinuousRouteRequestResult]? = nil
) -> Qwen38ScorecardContinuousRouteResult {
    let ids = Array(1 ... UInt64(concurrency))
    return Qwen38ScorecardContinuousRouteResult(
        evidenceKind: evidenceKind,
        concurrency: concurrency,
        coordinatorRequestIDs: ids,
        coordinatorPlanObservations: [
            Qwen38ScorecardContinuousRoutePlanObservation(
                planSequence: 1,
                stateRevisionAfterApply: 1,
                admissions: ids,
                decodeKind: .none,
                decodeRequestIDs: [],
                speculationAllowed: false,
                prefillRequestIDs: ids,
                activeSlotCount: concurrency,
                queuedSlotCount: 0),
            Qwen38ScorecardContinuousRoutePlanObservation(
                planSequence: 2,
                stateRevisionAfterApply: 2,
                admissions: [],
                decodeKind: .batch,
                decodeRequestIDs: ids,
                speculationAllowed: false,
                prefillRequestIDs: [],
                activeSlotCount: 0,
                queuedSlotCount: 0),
        ],
        planRevisions: [
            Qwen38ScorecardContinuousRouteRevision(
                planSequence: 1,
                stateRevisionAfterApply: 1),
            Qwen38ScorecardContinuousRouteRevision(
                planSequence: 2,
                stateRevisionAfterApply: 2),
        ],
        sharedBatchDecodeRequestIDs: ids,
        peakActiveSlots: concurrency,
        peakBatchOccupancy: concurrency,
        finalActiveRequests: 0,
        finalCoordinatorSlots: 0,
        finalReservedKVBytes: 0,
        requests: requests ?? (0 ..< concurrency).map { index in
            Qwen38ScorecardContinuousRouteRequestResult(
                requestIndex: index,
                coordinatorRequestID: UInt64(index + 1),
                route: .continuousBatchNoSpec,
                outputTokenIDs: [2_000 + index],
                finishReason: .stop,
                usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1),
                admittedAtUptime: Double(index + 1),
                completedAtUptime: Double(concurrency + 3 - index))
        })
}

private func makeStartupReport(
    concurrency: Int = 2
) -> ContinuousServingModelStartupReport {
    ContinuousServingModelStartupReport(
        launchedModel: "fixture",
        route: .continuousBatchNoSpec,
        memoryLimitBytes: 1,
        cacheLimitBytes: 1,
        maxReservedKVBytes: 1,
        maxContextTokens: 16,
        maxReservedContextTokens: 16,
        modelFamily: .qwen3,
        modelConfigurationSHA256: String(repeating: "a", count: 64),
        layerCount: 1,
        keyValueHeadCount: 1,
        headDimension: 1,
        stopTokenCount: 1,
        stopStringCount: 0,
        nativeCacheKinds: [.denseAttention],
        startupPromptTokenCount: 1,
        startupGeneratedTokenCount: 1,
        maxActiveSlots: concurrency,
        maxPrefillSlots: concurrency,
        prefillChunkSize: 8,
        maxQueuedRequests: concurrency,
        publicationCapacity: 4,
        soloPLDPolicy: nil,
        modelProofVerified: true)
}

private func makeProductionRouteFixtureBackend() -> ContinuousServingBackend {
    let coordinator = ContinuousBatchCoordinator(
        configuration: try! ContinuousBatchConfiguration(
            maxActiveSlots: 2,
            maxPrefillSlots: 2,
            prefillChunkSize: 8,
            maxQueuedRequests: 2),
        runtime: ProductionRouteFixtureRuntime(),
        automaticDrive: false,
        publicationCapacity: 4,
        traceLimit: 16)
    return ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: ProductionRouteFixtureCodec(),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 4,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 4_096)))
}

private func makeLoadedProductionRouteFixture(
    concurrency: Int,
    maxCompletionTokens: Int,
    traceLimit: Int = 128,
    allowsSoloDecode: Bool = false,
    outputTokensPerRequest: Int = 2
) -> LoadedContinuousServingModel {
    let coordinator = ContinuousBatchCoordinator(
        configuration: try! ContinuousBatchConfiguration(
            maxActiveSlots: concurrency,
            maxPrefillSlots: concurrency,
            prefillChunkSize: 8,
            maxQueuedRequests: concurrency),
        runtime: LoadedProductionRouteFixtureRuntime(
            concurrency: concurrency,
            allowsSoloDecode: allowsSoloDecode,
            outputTokensPerRequest: outputTokensPerRequest),
        automaticDrive: true,
        publicationCapacity: 8,
        traceLimit: traceLimit)
    let backend = ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: LoadedProductionRouteFixtureCodec(concurrency: concurrency),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: maxCompletionTokens,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
            scorecardTraceCapacity: traceLimit,
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: false,
                    maximumBatchRequests: concurrency,
                    maximumQueuedRequests: concurrency),
                coalescing: .automatic(.milliseconds(2)))))
    return LoadedContinuousServingModel
        .testingLoadedContinuousServingModelWithLoaderProvenance(
            backend: backend,
            startupReport: makeStartupReport(concurrency: concurrency))
}

private func seedUnrelatedPlanActivity(
    _ loaded: LoadedContinuousServingModel
) async throws {
    let handles = try await withThrowingTaskGroup(
        of: ServingGenerationHandle.self
    ) { group in
        for index in 0 ..< 2 {
            group.addTask {
                try await loaded.backend.start(
                    productionRouteRequest(
                        index: index,
                        model: loaded.startupReport.launchedModel))
            }
        }
        var values: [ServingGenerationHandle] = []
        values.reserveCapacity(2)
        for try await handle in group {
            values.append(handle)
        }
        return values
    }
    try await withThrowingTaskGroup(of: Void.self) { group in
        for handle in handles {
            group.addTask {
                while try await handle.mailbox.next() != nil {}
            }
        }
        try await group.waitForAll()
    }
    for _ in 0 ..< 100 where await loaded.backend.snapshot().activeRequests != 0 {
        await Task.yield()
    }
}

private func assertPlansOnlyReference(
    _ plans: [Qwen38ScorecardContinuousRoutePlanObservation],
    allowedIDs: Set<UInt64>,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    for plan in plans {
        let observed = Set(
            plan.admissions
                + plan.decodeRequestIDs
                + plan.prefillRequestIDs)
        XCTAssertTrue(
            observed.isSubset(of: allowedIDs),
            "unexpected plan IDs \(observed.subtracting(allowedIDs)) in \(plan)",
            file: file,
            line: line)
    }
}

private final class LoadedProductionRouteFixtureRuntime:
    ContinuousBatchRuntime
{
    private struct Slot {
        var processedTokens = 0
        var ready = false
        var cursor = 0
        let outputTokens: [Int]
    }

    private var slots: [BatchRequestID: Slot] = [:]
    private let outputTokensByPromptHead: [Int: [Int]]
    private let allowsSoloDecode: Bool

    init(
        concurrency: Int,
        allowsSoloDecode: Bool,
        outputTokensPerRequest: Int
    ) {
        var values: [Int: [Int]] = [:]
        for index in 0 ..< concurrency {
            let outputTokens: [Int]
            if outputTokensPerRequest == 2 {
                outputTokens = [2_000 + index, 3_000 + index]
            } else {
                outputTokens = (0 ..< outputTokensPerRequest).map {
                    10_000 + (index * outputTokensPerRequest) + $0
                }
            }
            values[1_000 + index] = outputTokens + [99]
        }
        self.outputTokensByPromptHead = values
        self.allowsSoloDecode = allowsSoloDecode
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        for admission in admissions {
            guard let head = admission.submission.promptTokens.first,
                let outputTokens = outputTokensByPromptHead[head]
            else {
                throw Qwen38ScorecardContinuousRouteError
                    .incompleteRequest(index: 0)
            }
            slots[admission.id] = Slot(outputTokens: outputTokens)
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 1,
            reservedKVBytes: slots.count,
            maxReservedKVBytes: 32)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        guard var slot = slots[work.id],
            work.startToken == slot.processedTokens
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(
                index: 0)
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        let ids: [BatchRequestID]
        switch action {
        case .batch(let batchIDs, speculationAllowed: false):
            ids = batchIDs
        case .solo(let id, speculationAllowed: false) where allowsSoloDecode:
            ids = [id]
        default:
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(
                    index: 0)
            }
            guard slot.cursor < slot.outputTokens.count else {
                return ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: [],
                    finished: true,
                    soloPipelineState: .canonical)
            }
            let token = slot.outputTokens[slot.cursor]
            slot.cursor += 1
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                soloPipelineState: .canonical)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
    }
}

private struct LoadedProductionRouteFixtureCodec: ScalarServingTextCodec {
    let promptByText: [String: [Int]]

    init(concurrency: Int) {
        var values: [String: [Int]] = [:]
        for index in 0 ..< concurrency {
            values["fixture-\(index)"] = [1_000 + index]
        }
        self.promptByText = values
    }

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        guard let text = messages.last?.text,
            let prompt = promptByText[text]
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(
                index: 0)
        }
        return prompt
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        LoadedProductionRouteFixtureDetokenizer()
    }
}

private struct LoadedProductionRouteFixtureDetokenizer:
    ScalarServingDetokenizer
{
    private var pending: Int?

    mutating func append(token: Int) {
        pending = token
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending.map(String.init)
    }
}

private enum PartialAdmissionFailureError: Error, Equatable {
    case startFailedAfterPeerAdmission
}

private actor PartialAdmissionStartGate {
    private var index0Retained = false
    private var lease: ServingRequestLease?

    func markIndex0Retained(_ lease: ServingRequestLease) {
        self.lease = lease
        index0Retained = true
    }

    func waitForIndex0Retained() async {
        while !index0Retained {
            await Task.yield()
        }
    }

    func index0LeaseState() async -> ServingRequestLeaseState? {
        await lease?.state
    }
}

private final class ProductionRouteFixtureRuntime: ContinuousBatchRuntime {
    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 1,
            reservedKVBytes: 0,
            maxReservedKVBytes: 8)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private struct ProductionRouteFixtureCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        [1]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        ProductionRouteFixtureDetokenizer()
    }
}

private struct ProductionRouteFixtureDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) {}
    mutating func next() -> String? { nil }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ verify: (any Error) -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("expected error", file: file, line: line)
    } catch {
        verify(error)
    }
}
