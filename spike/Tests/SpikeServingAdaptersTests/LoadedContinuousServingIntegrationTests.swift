import Foundation
import MLX
import NIOCore
import NIOPosix
import os
import XCTest

import HarnessCore
import ServingCore
import ServingNIO
@testable import SpikeServingAdapters

final class LoadedContinuousServingIntegrationTests: XCTestCase {
    func testLoadedPrefillAndHostileDecodeDisconnectsPreserveSurvivorsAndReuseSlot()
        async throws
    {
        let environment = ProcessInfo.processInfo.environment
        guard
            let modelPath = environment["FASTMLX_CONTINUOUS_TEST_MODEL_PATH"],
            let launchedModel = environment["FASTMLX_CONTINUOUS_TEST_MODEL"],
            let memoryLimit =
                environment["FASTMLX_CONTINUOUS_TEST_MEMORY_LIMIT_BYTES"]
                .flatMap(Int.init),
            let cacheLimit =
                environment["FASTMLX_CONTINUOUS_TEST_CACHE_LIMIT_BYTES"]
                .flatMap(Int.init),
            let maxReservedKV =
                environment["FASTMLX_CONTINUOUS_TEST_MAX_RESERVED_KV_BYTES"]
                .flatMap(Int.init)
        else {
            throw XCTSkip(
                "Set the FASTMLX_CONTINUOUS_TEST_* variables for the loaded-model proof")
        }

        let loaded = try await loadContinuousServingModel(
            configuration: ContinuousServingModelLoadConfiguration(
                launchedModel: launchedModel,
                modelDirectory: URL(
                    fileURLWithPath: modelPath,
                    isDirectory: true),
                memoryLimitBytes: memoryLimit,
                cacheLimitBytes: cacheLimit,
                maxReservedKVBytes: maxReservedKV,
                coordinatorConfiguration: try ContinuousBatchConfiguration(
                    maxActiveSlots: 4,
                    maxPrefillSlots: 1,
                    prefillChunkSize: 512,
                    maxQueuedRequests: 4),
                publicationCapacity: 1,
                traceLimit: 2_048,
                backendConfiguration: ContinuousServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 128,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 8,
                        maxBytes: 32 * 1_024))))
        XCTAssertEqual(loaded.startupReport.route, .continuousBatchNoSpec)
        XCTAssertEqual(loaded.startupReport.modelFamily, .qwen3)
        XCTAssertTrue(loaded.startupReport.modelProofVerified)
        XCTAssertEqual(
            Set(loaded.startupReport.nativeCacheKinds),
            [.denseAttention])
        XCTAssertEqual(
            loaded.startupReport.maxReservedKVBytes,
            maxReservedKV)
        XCTAssertEqual(
            loaded.startupReport.startupGeneratedTokenCount,
            1)

        let server = try await ServingHTTPServer.start(
            configuration: ServingHTTPConfiguration(
                launchedModel: launchedModel,
                requestLimits: .productionDefault,
                requiredBearerToken: nil,
                maximumNonStreamingResponseBytes: 1_048_576,
                backpressureStallTimeout: .seconds(5)),
            backend: loaded.backend)
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)

        do {
            let clock = ContinuousClock()
            let prefillRecorder = ContinuousSocketResponseRecorder()
            let prefillClient = try await ClientBootstrap(group: clientGroup)
                .channelInitializer { channel in
                    channel.pipeline.addHandler(prefillRecorder)
                }
                .connect(to: server.localAddress)
                .get()
            try await sendContinuousStreamingRequest(
                on: prefillClient,
                launchedModel: launchedModel,
                messages: continuousLongPrefillMessages(),
                maximumCompletionTokens: 512)
            try await waitUntilLoadedContinuous(timeout: .seconds(30)) {
                let slots = await loaded.backend
                    .diagnosticCoordinatorSnapshots()
                return slots.count == 1
                    && slots.contains {
                        if case .prefilling = $0.phase { return true }
                        return false
                    }
            }
            let prefillIDs = await loaded.backend
                .diagnosticCoordinatorRequestIDs()
            let prefillID = try XCTUnwrap(prefillIDs.first)
            let prefillCancelledAt = clock.now
            try await prefillClient.close().get()
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                let snapshot = await loaded.backend.snapshot()
                return snapshot.activeRequests == 0
                    && snapshot.coordinatorSlots == 0
                    && snapshot.reservedKVBytes == 0
            }
            let prefillCancellationLatency =
                prefillCancelledAt.duration(to: clock.now)
            XCTAssertLessThan(
                prefillCancellationLatency,
                .seconds(5))
            let prefillTrace = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()
            XCTAssertTrue(
                traceContainsCancellation(
                    prefillTrace,
                    id: prefillID,
                    phase: .prefilling))

            let survivorAMessages = continuousSurvivorAMessages()
            let survivorBMessages = continuousSurvivorBMessages()
            let replacementMessages = continuousReplacementMessages()
            let outputBudget = 128
            let survivorAControl = try await collectContinuousControl(
                try await loaded.backend.start(
                    continuousControlRequest(
                        launchedModel: launchedModel,
                        messages: survivorAMessages,
                        maximumCompletionTokens: outputBudget)))
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                await loaded.backend.snapshot().activeRequests == 0
            }
            let survivorBControl = try await collectContinuousControl(
                try await loaded.backend.start(
                    continuousControlRequest(
                        launchedModel: launchedModel,
                        messages: survivorBMessages,
                        maximumCompletionTokens: outputBudget)))
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                await loaded.backend.snapshot().activeRequests == 0
            }
            let replacementControl = try await collectContinuousControl(
                try await loaded.backend.start(
                    continuousControlRequest(
                        launchedModel: launchedModel,
                        messages: replacementMessages,
                        maximumCompletionTokens: outputBudget)))
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                await loaded.backend.snapshot().activeRequests == 0
            }
            let crossCapacityControl = try await collectContinuousControl(
                try await loaded.backend.start(
                    continuousControlRequest(
                        launchedModel: launchedModel,
                        messages: continuousCrossCapacityMessages(),
                        maximumCompletionTokens: 16)))
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                await loaded.backend.snapshot().activeRequests == 0
            }
            XCTAssertEqual(
                survivorAControl.completion.finishReason,
                .length)
            XCTAssertEqual(
                survivorBControl.completion.finishReason,
                .length)
            XCTAssertEqual(
                replacementControl.completion.finishReason,
                .length)
            XCTAssertEqual(
                crossCapacityControl.completion.finishReason,
                .length)
            _ = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()

            let sharedAHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: survivorAMessages,
                    maximumCompletionTokens: outputBudget))
            let sharedBHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: survivorBMessages,
                    maximumCompletionTokens: outputBudget))
            let sharedIDs = Set(
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs())
            async let sharedA = collectContinuousControl(sharedAHandle)
            async let sharedB = collectContinuousControl(sharedBHandle)
            let (sharedAResult, sharedBResult) =
                try await (sharedA, sharedB)
            assertContinuousControlResult(
                sharedAResult,
                equals: survivorAControl)
            assertContinuousControlResult(
                sharedBResult,
                equals: survivorBControl)
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                let snapshot = await loaded.backend.snapshot()
                return snapshot.activeRequests == 0
                    && snapshot.coordinatorSlots == 0
                    && snapshot.reservedKVBytes == 0
            }
            let sharedTrace = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()
            XCTAssertTrue(
                traceContainsSharedBatch(
                    sharedTrace,
                    ids: sharedIDs))

            let isolatedAHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: survivorAMessages,
                    maximumCompletionTokens: outputBudget))
            let isolatedBHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: survivorBMessages,
                    maximumCompletionTokens: outputBudget))
            let isolatedSurvivorIDs = Set(
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs())
            XCTAssertEqual(isolatedSurvivorIDs.count, 2)
            let incompatibleAHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: continuousCrossCapacityMessages(),
                    maximumCompletionTokens: 16))
            let incompatibleBHandle = try await loaded.backend.start(
                continuousControlRequest(
                    launchedModel: launchedModel,
                    messages: continuousCrossCapacityMessages(),
                    maximumCompletionTokens: 16))
            try await waitUntilLoadedContinuous(timeout: .seconds(30)) {
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs().count == 4
            }
            let isolatedSnapshots = await loaded.backend
                .diagnosticCoordinatorSnapshots()
            let allIsolatedIDs = Set(
                isolatedSnapshots.map(\.request.id))
            let incompatibleIDs = allIsolatedIDs
                .subtracting(isolatedSurvivorIDs)
            XCTAssertEqual(incompatibleIDs.count, 2)
            let survivorCohorts = Set(
                isolatedSnapshots
                    .filter {
                        isolatedSurvivorIDs.contains($0.request.id)
                    }
                    .map(\.request.decodeCohort))
            let incompatibleCohorts = Set(
                isolatedSnapshots
                    .filter {
                        incompatibleIDs.contains($0.request.id)
                    }
                    .map(\.request.decodeCohort))
            XCTAssertEqual(survivorCohorts.count, 1)
            XCTAssertEqual(incompatibleCohorts.count, 1)
            XCTAssertTrue(
                survivorCohorts.isDisjoint(with: incompatibleCohorts))

            async let isolatedA = collectContinuousControl(
                isolatedAHandle)
            async let isolatedB = collectContinuousControl(
                isolatedBHandle)
            async let incompatibleA = collectContinuousControl(
                incompatibleAHandle)
            async let incompatibleB = collectContinuousControl(
                incompatibleBHandle)
            let (
                isolatedAResult,
                isolatedBResult,
                incompatibleAResult,
                incompatibleBResult
            ) = try await (
                isolatedA,
                isolatedB,
                incompatibleA,
                incompatibleB)
            assertContinuousControlResult(
                isolatedAResult,
                equals: survivorAControl)
            assertContinuousControlResult(
                isolatedBResult,
                equals: survivorBControl)
            assertContinuousControlResult(
                incompatibleAResult,
                equals: crossCapacityControl)
            assertContinuousControlResult(
                incompatibleBResult,
                equals: crossCapacityControl)
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                let snapshot = await loaded.backend.snapshot()
                return snapshot.activeRequests == 0
                    && snapshot.coordinatorSlots == 0
                    && snapshot.reservedKVBytes == 0
            }
            let isolatedTrace = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()
            XCTAssertTrue(
                traceContainsSharedBatch(
                    isolatedTrace,
                    ids: isolatedSurvivorIDs))
            XCTAssertTrue(
                traceContainsSharedBatch(
                    isolatedTrace,
                    ids: incompatibleIDs))
            XCTAssertTrue(
                traceBatchesStayWithin(
                    isolatedTrace,
                    permittedGroups: [
                        isolatedSurvivorIDs,
                        incompatibleIDs,
                    ]))
            XCTAssertTrue(
                isolatedTrace.allSatisfy(traceDisablesSpeculation))

            guard let port = server.localAddress.port else {
                throw LoadedContinuousIntegrationError.missingTCPPort
            }
            let survivorATask = Task {
                try await performContinuousRequest(
                    port: port,
                    launchedModel: launchedModel,
                    messages: survivorAMessages,
                    maximumCompletionTokens: outputBudget)
            }
            let survivorBTask = Task {
                try await performContinuousRequest(
                    port: port,
                    launchedModel: launchedModel,
                    messages: survivorBMessages,
                    maximumCompletionTokens: outputBudget)
            }
            try await waitUntilLoadedContinuous(timeout: .seconds(30)) {
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs().count == 2
            }
            let survivorIDs = Set(
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs())

            let middleRecorder = ContinuousSocketResponseRecorder()
            let middleClient = try await ClientBootstrap(group: clientGroup)
                .channelInitializer { channel in
                    channel.pipeline.addHandler(middleRecorder)
                }
                .connect(to: server.localAddress)
                .get()
            try await sendContinuousStreamingRequest(
                on: middleClient,
                launchedModel: launchedModel,
                messages: continuousLongestMiddleMessages(),
                maximumCompletionTokens: outputBudget)
            try await waitUntilLoadedContinuous(timeout: .seconds(30)) {
                let currentIDs = Set(
                    await loaded.backend
                        .diagnosticCoordinatorRequestIDs())
                guard currentIDs.count == 3,
                    middleRecorder.receivedBytes > 0
                else {
                    return false
                }
                let trace = await loaded.backend
                    .diagnosticCoordinatorExecutionTrace()
                return traceContainsSharedBatch(
                    trace,
                    ids: currentIDs)
            }
            let allInitialIDs = Set(
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs())
            let middleID = try XCTUnwrap(
                allInitialIDs.subtracting(survivorIDs).first)
            let allInitialSnapshots = await loaded.backend
                .diagnosticCoordinatorSnapshots()
            let middleSnapshot = try XCTUnwrap(
                allInitialSnapshots.first {
                    $0.request.id == middleID
                })
            let survivorPromptMaximum = try XCTUnwrap(
                allInitialSnapshots
                    .filter { survivorIDs.contains($0.request.id) }
                    .map(\.request.promptTokenCount)
                    .max())
            XCTAssertGreaterThan(
                middleSnapshot.request.promptTokenCount,
                survivorPromptMaximum)
            XCTAssertEqual(
                Set(allInitialSnapshots.map(\.request.decodeCohort)).count,
                1,
                "the hostile compaction case must remain one exact fixed-capacity cohort")
            let resourcesBeforeCancellation =
                await loaded.backend.snapshot()
            XCTAssertGreaterThan(
                resourcesBeforeCancellation.reservedKVBytes,
                0)

            let middleCancelledAt = clock.now
            try await middleClient.close().get()
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                Set(
                    await loaded.backend
                        .diagnosticCoordinatorRequestIDs()
                ) == survivorIDs
            }
            let middleCancellationLatency =
                middleCancelledAt.duration(to: clock.now)
            XCTAssertLessThan(
                middleCancellationLatency,
                .seconds(5))
            let resourcesAfterCancellation =
                await loaded.backend.snapshot()
            XCTAssertEqual(resourcesAfterCancellation.activeRequests, 2)
            XCTAssertGreaterThan(
                resourcesAfterCancellation.reservedKVBytes,
                0)
            XCTAssertLessThanOrEqual(
                resourcesAfterCancellation.reservedKVBytes,
                maxReservedKV)
            let traceAfterCancellation = await loaded.backend
                .diagnosticCoordinatorExecutionTrace()
            XCTAssertTrue(
                traceContainsCancellation(
                    traceAfterCancellation,
                    id: middleID,
                    phase: .decoding))

            let replacementTask = Task {
                try await performContinuousRequest(
                    port: port,
                    launchedModel: launchedModel,
                    messages: replacementMessages,
                    maximumCompletionTokens: outputBudget)
            }
            try await waitUntilLoadedContinuous(timeout: .seconds(30)) {
                let currentIDs = Set(
                    await loaded.backend
                        .diagnosticCoordinatorRequestIDs())
                guard currentIDs.count == 3,
                    survivorIDs.isSubset(of: currentIDs)
                else {
                    return false
                }
                let replacementIDs = currentIDs.subtracting(survivorIDs)
                guard replacementIDs.count == 1 else { return false }
                let trace = await loaded.backend
                    .diagnosticCoordinatorExecutionTrace()
                return traceContainsSharedBatch(
                    trace,
                    ids: currentIDs)
            }
            let currentReplacementIDs = Set(
                await loaded.backend
                    .diagnosticCoordinatorRequestIDs()
            ).subtracting(survivorIDs)
            let replacementID = try XCTUnwrap(
                currentReplacementIDs.first)

            let survivorA = try await survivorATask.value
            let survivorB = try await survivorBTask.value
            let replacement = try await replacementTask.value
            assertContinuousHTTPResult(
                survivorA,
                equals: survivorAControl)
            assertContinuousHTTPResult(
                survivorB,
                equals: survivorBControl)
            assertContinuousHTTPResult(
                replacement,
                equals: replacementControl)

            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                let snapshot = await loaded.backend.snapshot()
                return snapshot.activeRequests == 0
                    && snapshot.coordinatorSlots == 0
                    && snapshot.reservedKVBytes == 0
            }
            let finalTrace = await loaded.backend
                .diagnosticCoordinatorExecutionTrace()
            XCTAssertTrue(
                traceContainsSharedBatch(
                    finalTrace,
                    ids: survivorIDs.union([middleID])))
            XCTAssertTrue(
                traceContainsSharedBatch(
                    finalTrace,
                    ids: survivorIDs.union([replacementID])))
            XCTAssertTrue(
                finalTrace.allSatisfy(traceDisablesSpeculation))
            try await server.shutdown(gracePeriod: .seconds(5))
            try await clientGroup.shutdownGracefully()
            let activeConnectionCount = await server.activeConnectionCount
            XCTAssertEqual(activeConnectionCount, 0)
        } catch {
            try? await server.shutdown(gracePeriod: .seconds(5))
            try? await clientGroup.shutdownGracefully()
            throw error
        }
    }

    func testLoadedDynamicSoloPLDIsExactAndSharedBatchNeverSpeculates()
        async throws
    {
        let environment = ProcessInfo.processInfo.environment
        guard
            let modelPath = environment["FASTMLX_CONTINUOUS_TEST_MODEL_PATH"],
            let launchedModel = environment["FASTMLX_CONTINUOUS_TEST_MODEL"],
            let memoryLimit =
                environment["FASTMLX_CONTINUOUS_TEST_MEMORY_LIMIT_BYTES"]
                .flatMap(Int.init),
            let cacheLimit =
                environment["FASTMLX_CONTINUOUS_TEST_CACHE_LIMIT_BYTES"]
                .flatMap(Int.init),
            let maxReservedKV =
                environment["FASTMLX_CONTINUOUS_TEST_MAX_RESERVED_KV_BYTES"]
                .flatMap(Int.init)
        else {
            throw XCTSkip(
                "Set the FASTMLX_CONTINUOUS_TEST_* variables for the loaded-model proof")
        }
        let policy = ContinuousServingSoloPLDPolicy.qwen3WidthOne
        let messages = continuousPromptLookupMessages()
        let request = continuousControlRequest(
            launchedModel: launchedModel,
            messages: messages,
            maximumCompletionTokens: 128)
        let scalarControl = try await {
            let control = try await loadContinuousServingModel(
                configuration: ContinuousServingModelLoadConfiguration(
                    launchedModel: launchedModel,
                    modelDirectory: URL(
                        fileURLWithPath: modelPath,
                        isDirectory: true),
                    memoryLimitBytes: memoryLimit,
                    cacheLimitBytes: cacheLimit,
                    maxReservedKVBytes: maxReservedKV,
                    coordinatorConfiguration: try ContinuousBatchConfiguration(
                        maxActiveSlots: 4,
                        maxPrefillSlots: 2,
                        prefillChunkSize: 512,
                        maxQueuedRequests: 8),
                    publicationCapacity: 1,
                    traceLimit: 2_048,
                    backendConfiguration: ContinuousServingBackendConfiguration(
                        defaultMaximumCompletionTokens: 128,
                        queueRetryAfterSeconds: 1,
                        mailboxCapacity: .init(
                            maxDeltas: 8,
                            maxBytes: 32 * 1_024))))
            let handle = try await control.backend.start(request)
            XCTAssertEqual(handle.route, .continuousBatchNoSpec)
            let result = try await collectContinuousControl(handle)
            await control.backend.shutdown()
            return result
        }()
        Memory.clearCache()

        let loaded = try await loadContinuousServingModel(
            configuration: ContinuousServingModelLoadConfiguration(
                launchedModel: launchedModel,
                modelDirectory: URL(
                    fileURLWithPath: modelPath,
                    isDirectory: true),
                memoryLimitBytes: memoryLimit,
                cacheLimitBytes: cacheLimit,
                maxReservedKVBytes: maxReservedKV,
                coordinatorConfiguration: try ContinuousBatchConfiguration(
                    maxActiveSlots: 4,
                    maxPrefillSlots: 2,
                    prefillChunkSize: 512,
                    maxQueuedRequests: 8),
                publicationCapacity: 1,
                traceLimit: 2_048,
                backendConfiguration: ContinuousServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 128,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 8,
                        maxBytes: 32 * 1_024),
                    admission: .dynamic(
                        configuration: ServingAdmissionConfiguration(
                            soloPLDQualified: true,
                            maximumBatchRequests: 4,
                            maximumQueuedRequests: 8),
                        coalescing: .automatic(.milliseconds(5)))),
                soloPLDPolicy: policy))
        XCTAssertEqual(loaded.startupReport.soloPLDPolicy, policy)

        do {
            let soloHandle = try await loaded.backend.start(request)
            XCTAssertEqual(soloHandle.route, .soloPLD)
            let solo = try await collectContinuousControl(soloHandle)
            assertContinuousControlResult(solo, equals: scalarControl)
            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                await loaded.backend.snapshot().activeRequests == 0
            }
            let soloRuntimeResources =
                await loaded.backend.diagnosticRuntimeResourceSnapshot()
            let soloResources = try XCTUnwrap(soloRuntimeResources)
            let soloSpeculation = try XCTUnwrap(soloResources.speculation)
            XCTAssertEqual(soloSpeculation.requestedRequests, 1)
            XCTAssertEqual(soloSpeculation.activeSessions, 0)
            XCTAssertTrue(soloSpeculation.engaged)
            XCTAssertGreaterThan(soloSpeculation.acceptedDraftTokens, 0)
            XCTAssertGreaterThan(soloSpeculation.verificationRounds, 0)
            let soloTrace = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()
            XCTAssertTrue(traceContainsSpeculativeSolo(soloTrace))

            async let firstStart = loaded.backend.start(request)
            async let secondStart = loaded.backend.start(request)
            let (firstHandle, secondHandle) =
                try await (firstStart, secondStart)
            XCTAssertEqual(firstHandle.route, .continuousBatchNoSpec)
            XCTAssertEqual(secondHandle.route, .continuousBatchNoSpec)
            async let first = collectContinuousControl(firstHandle)
            async let second = collectContinuousControl(secondHandle)
            let (firstResult, secondResult) = try await (first, second)
            assertContinuousControlResult(firstResult, equals: scalarControl)
            assertContinuousControlResult(secondResult, equals: scalarControl)

            try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
                let snapshot = await loaded.backend.snapshot()
                return snapshot.activeRequests == 0
                    && snapshot.coordinatorSlots == 0
                    && snapshot.reservedKVBytes == 0
            }
            let batchTrace = await loaded.backend
                .diagnosticTakeCoordinatorExecutionTrace()
            XCTAssertTrue(
                batchTrace.contains {
                    if case .operation(
                        .decode(.batch(_, speculationAllowed: false))
                    ) = $0 {
                        return true
                    }
                    return false
                })
            XCTAssertTrue(batchTrace.allSatisfy(traceDisablesSpeculation))
            await loaded.backend.shutdown()
        } catch {
            await loaded.backend.shutdown()
            throw error
        }
    }

    func testLoadedHTTPPhase4DiagnosticDynamicPLDFrontierAgainstBatchNoSpec()
        async throws
    {
        let environment = ProcessInfo.processInfo.environment
        guard
            let modelPath = environment["FASTMLX_CONTINUOUS_TEST_MODEL_PATH"],
            let launchedModel = environment["FASTMLX_CONTINUOUS_TEST_MODEL"],
            let memoryLimit =
                environment["FASTMLX_CONTINUOUS_TEST_MEMORY_LIMIT_BYTES"]
                .flatMap(Int.init),
            let cacheLimit =
                environment["FASTMLX_CONTINUOUS_TEST_CACHE_LIMIT_BYTES"]
                .flatMap(Int.init),
            let maxReservedKV =
                environment["FASTMLX_CONTINUOUS_TEST_MAX_RESERVED_KV_BYTES"]
                .flatMap(Int.init)
        else {
            throw XCTSkip(
                "Set the FASTMLX_CONTINUOUS_TEST_* variables for the loaded-model proof")
        }
        let concurrencies = try loadedHTTPQualificationConcurrencies(
            environment["FASTMLX_CONTINUOUS_PHASE4_CONCURRENCIES"])
        let measuredBurstCount = 2
        let outputBudget = 128
        let messages = continuousPromptLookupMessages()
        let modeOrder: [LoadedHTTPQualificationMode]
        switch environment["FASTMLX_CONTINUOUS_PHASE4_MODE_ORDER"]
            ?? "no-spec-first"
        {
        case "no-spec-first":
            modeOrder = [.explicitBatchNoSpec, .dynamicPLD]
        case "dynamic-first":
            modeOrder = [.dynamicPLD, .explicitBatchNoSpec]
        default:
            throw LoadedContinuousIntegrationError.invalidQualificationModeOrder
        }
        var modeResults: [LoadedHTTPQualificationModeResult] = []
        var exactReference: LoadedContinuousHTTPResult?
        for mode in modeOrder {
            let result = try await runLoadedHTTPQualificationMode(
                mode: mode,
                modelPath: modelPath,
                launchedModel: launchedModel,
                memoryLimit: memoryLimit,
                cacheLimit: cacheLimit,
                maxReservedKV: maxReservedKV,
                messages: messages,
                maximumCompletionTokens: outputBudget,
                concurrencies: concurrencies,
                measuredBurstCount: measuredBurstCount,
                expectedControl: exactReference)
            exactReference = result.reference
            modeResults.append(result)
            Memory.clearCache()
        }
        let noSpec = try XCTUnwrap(
            modeResults.first {
                $0.summary.mode
                    == LoadedHTTPQualificationMode.explicitBatchNoSpec.rawValue
            })
        let dynamic = try XCTUnwrap(
            modeResults.first {
                $0.summary.mode
                    == LoadedHTTPQualificationMode.dynamicPLD.rawValue
            })

        let c1NoSpec = try XCTUnwrap(
            noSpec.summary.concurrency.first { $0.concurrency == 1 })
        let c1Dynamic = try XCTUnwrap(
            dynamic.summary.concurrency.first { $0.concurrency == 1 })
        let comparison = LoadedHTTPThroughputComparison(
            noSpecMeanCompletionTokensPerSecond:
                c1NoSpec.meanCompletionTokensPerSecond,
            dynamicMeanCompletionTokensPerSecond:
                c1Dynamic.meanCompletionTokensPerSecond,
            noSpecMedianCompletionTokensPerSecond:
                c1NoSpec.medianCompletionTokensPerSecond,
            dynamicMedianCompletionTokensPerSecond:
                c1Dynamic.medianCompletionTokensPerSecond,
            requiredDynamicSpeedup: 1.05)
        let summary = LoadedHTTPPhase4Summary(
            schemaVersion: 1,
            evidenceRole: "diagnostic-only",
            promotable: false,
            speedAggregation: "forbidden",
            model: launchedModel,
            maximumCompletionTokens: outputBudget,
            measuredBurstCount: measuredBurstCount,
            concurrencies: concurrencies,
            modeOrder: modeOrder.map(\.rawValue),
            modes: modeResults.map(\.summary),
            c1ThroughputComparison: comparison)
        print(try encodeLoadedHTTPPhase4Summary(summary))
        XCTAssertGreaterThanOrEqual(
            comparison.dynamicMeanCompletionTokensPerSecond,
            comparison.noSpecMeanCompletionTokensPerSecond
                * comparison.requiredDynamicSpeedup)
        XCTAssertGreaterThanOrEqual(
            comparison.dynamicMedianCompletionTokensPerSecond,
            comparison.noSpecMedianCompletionTokensPerSecond
                * comparison.requiredDynamicSpeedup)
    }
}

private final class ContinuousSocketResponseRecorder:
    ChannelInboundHandler, Sendable
{
    typealias InboundIn = ByteBuffer

    private let byteCount = OSAllocatedUnfairLock(initialState: 0)

    var receivedBytes: Int {
        byteCount.withLock { $0 }
    }

    func channelRead(
        context: ChannelHandlerContext,
        data: NIOAny
    ) {
        let buffer = unwrapInboundIn(data)
        byteCount.withLock { $0 += buffer.readableBytes }
    }
}

private struct LoadedContinuousControlResult {
    let text: String
    let completion: ServingGenerationCompletion
}

private struct LoadedContinuousHTTPResult {
    let statusCode: Int
    let text: String
    let finishReason: OpenAIChatFinishReason
    let usage: OpenAIChatUsage
}

private enum LoadedHTTPQualificationMode: String {
    case explicitBatchNoSpec = "explicit-batch-no-spec"
    case dynamicPLD = "dynamic-pld"
}

private struct LoadedHTTPQualificationModeResult {
    let summary: LoadedHTTPModeSummary
    let reference: LoadedContinuousHTTPResult
}

private struct LoadedHTTPBurstObservation {
    let summary: LoadedHTTPBurstSummary
    let results: [LoadedContinuousHTTPResult]
}

private struct LoadedHTTPPhase4Summary: Codable {
    let schemaVersion: Int
    let evidenceRole: String
    let promotable: Bool
    let speedAggregation: String
    let model: String
    let maximumCompletionTokens: Int
    let measuredBurstCount: Int
    let concurrencies: [Int]
    let modeOrder: [String]
    let modes: [LoadedHTTPModeSummary]
    let c1ThroughputComparison: LoadedHTTPThroughputComparison
}

private struct LoadedHTTPModeSummary: Codable {
    let mode: String
    let startupRoute: String
    let soloPLDPolicy: String?
    let concurrency: [LoadedHTTPConcurrencySummary]
}

private struct LoadedHTTPConcurrencySummary: Codable {
    let concurrency: Int
    let warmup: LoadedHTTPBurstSummary
    let measured: [LoadedHTTPBurstSummary]
    let measuredWallMilliseconds: [Double]
    let aggregateWallMilliseconds: Double
    let aggregateCompletionTokensPerSecond: Double
    let meanCompletionTokensPerSecond: Double
    let medianCompletionTokensPerSecond: Double
    let outputSHA256s: [String]
    let routeFacts: LoadedHTTPRouteFacts
}

private struct LoadedHTTPBurstSummary: Codable {
    let index: Int
    let measured: Bool
    let wallMilliseconds: Double
    let completionTokens: Int
    let completionTokensPerSecond: Double
    let outputSHA256s: [String]
    let routeFacts: LoadedHTTPRouteFacts
}

private struct LoadedHTTPRouteFacts: Codable {
    let decodeOperationCount: Int
    let sawSolo: Bool
    let sawSoloSpeculationAllowed: Bool
    let sawSharedBatch: Bool
    let maxDecodeBatchSize: Int
    let speculationAllowed: Bool
    let allDecodeSpeculationDisabled: Bool
    let speculationEngagedDuringInterval: Bool
    let speculationRequestedRequestsDelta: Int
    let draftedTokensDelta: Int
    let acceptedDraftTokensDelta: Int
    let verificationRoundsDelta: Int
}

private struct LoadedHTTPThroughputComparison: Codable {
    let noSpecMeanCompletionTokensPerSecond: Double
    let dynamicMeanCompletionTokensPerSecond: Double
    let noSpecMedianCompletionTokensPerSecond: Double
    let dynamicMedianCompletionTokensPerSecond: Double
    let requiredDynamicSpeedup: Double
}

private func continuousSurvivorAMessages() -> [OpenAIChatMessage] {
    [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                "Output the integers from 1 through 500 as one comma-separated sequence. "
                + "Do not explain and do not stop early."),
    ]
}

private func continuousSurvivorBMessages() -> [OpenAIChatMessage] {
    [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                "Repeat the lowercase alphabet twenty times as one comma-separated sequence. "
                + "Do not explain and do not stop early."),
    ]
}

private func continuousPromptLookupMessages() -> [OpenAIChatMessage] {
    let payload = Array(
        repeating:
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        count: 32
    ).joined(separator: " ")
    return [
        OpenAIChatMessage(
            role: .system,
            text: "Copy the requested payload exactly without commentary."),
        OpenAIChatMessage(
            role: .user,
            text:
                "Output exactly the text between BEGIN and END, without the markers.\n"
                + "BEGIN\n\(payload)\nEND"),
    ]
}

private func continuousReplacementMessages() -> [OpenAIChatMessage] {
    [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                "Output the even integers from 2 through 1000 as one comma-separated sequence. "
                + "Do not explain and do not stop early."),
    ]
}

private func continuousLongestMiddleMessages() -> [OpenAIChatMessage] {
    let retainedPrefix = Array(
        repeating:
            "This prefix exists only to make this request the longest active row.",
        count: 4
    ).joined(separator: " ")
    return [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                retainedPrefix
                + " Output the integers from 1000 downward as one comma-separated sequence. "
                + "Do not explain and do not stop early."),
    ]
}

private func continuousCrossCapacityMessages() -> [OpenAIChatMessage] {
    let retainedPrefix = Array(
        repeating:
            "This deterministic prefix deliberately belongs to a wider KV capacity cohort.",
        count: 120
    ).joined(separator: " ")
    return [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                retainedPrefix
                + " Output the integers from 1000 downward as one comma-separated sequence. "
                + "Do not explain and do not stop early."),
    ]
}

private func continuousLongPrefillMessages() -> [OpenAIChatMessage] {
    let retainedPrefix = Array(
        repeating:
            "Observe this deterministic prefix and retain it while preparing the answer.",
        count: 600
    ).joined(separator: " ")
    return [
        OpenAIChatMessage(role: .system, text: "Follow the output format exactly."),
        OpenAIChatMessage(
            role: .user,
            text:
                retainedPrefix
                + " Then output the integers from 1 through 500 as a comma-separated sequence."),
    ]
}

private func continuousControlRequest(
    launchedModel: String,
    messages: [OpenAIChatMessage],
    maximumCompletionTokens: Int
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: launchedModel,
        messages: messages,
        maxCompletionTokens: maximumCompletionTokens,
        temperature: 0,
        choiceCount: 1,
        stream: false,
        stop: [])
}

private func sendContinuousStreamingRequest(
    on channel: any Channel,
    launchedModel: String,
    messages: [OpenAIChatMessage],
    maximumCompletionTokens: Int
) async throws {
    let body = try JSONSerialization.data(
        withJSONObject: [
            "model": launchedModel,
            "messages": messages.map {
                ["role": $0.role.rawValue, "content": $0.text]
            },
            "max_completion_tokens": maximumCompletionTokens,
            "temperature": 0,
            "n": 1,
            "stream": true,
        ],
        options: [.sortedKeys])
    let request = """
        POST /v1/chat/completions HTTP/1.1\r
        Host: 127.0.0.1\r
        Content-Type: application/json\r
        Content-Length: \(body.count)\r
        \r
        \(String(decoding: body, as: UTF8.self))
        """
    try await channel.writeAndFlush(ByteBuffer(string: request)).get()
}

private func collectContinuousControl(
    _ handle: ServingGenerationHandle
) async throws -> LoadedContinuousControlResult {
    var text = ""
    var completion: ServingGenerationCompletion?
    while let delta = try await handle.mailbox.next() {
        switch delta {
        case .text(let value):
            text += value
        case .completion(let value):
            completion = value
        }
    }
    guard let completion else {
        throw LoadedContinuousIntegrationError.missingCompletion
    }
    return LoadedContinuousControlResult(
        text: text,
        completion: completion)
}

private func performContinuousRequest(
    port: Int,
    launchedModel: String,
    messages: [OpenAIChatMessage],
    maximumCompletionTokens: Int
) async throws -> LoadedContinuousHTTPResult {
    var request = URLRequest(
        url: URL(
            string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(
        withJSONObject: [
            "model": launchedModel,
            "messages": messages.map {
                ["role": $0.role.rawValue, "content": $0.text]
            },
            "max_completion_tokens": maximumCompletionTokens,
            "temperature": 0,
            "n": 1,
            "stream": false,
        ],
        options: [.sortedKeys])
    let (data, response) = try await URLSession.shared.data(for: request)
    guard
        let response = response as? HTTPURLResponse,
        let root = try JSONSerialization.jsonObject(with: data)
            as? [String: Any],
        let choices = root["choices"] as? [[String: Any]],
        let first = choices.first,
        let message = first["message"] as? [String: Any],
        let text = message["content"] as? String,
        let rawFinishReason = first["finish_reason"] as? String,
        let finishReason = OpenAIChatFinishReason(rawValue: rawFinishReason),
        let usage = root["usage"] as? [String: Any],
        let promptTokens = usage["prompt_tokens"] as? Int,
        let completionTokens = usage["completion_tokens"] as? Int
    else {
        throw LoadedContinuousIntegrationError.invalidHTTPResponse
    }
    return LoadedContinuousHTTPResult(
        statusCode: response.statusCode,
        text: text,
        finishReason: finishReason,
        usage: OpenAIChatUsage(
            promptTokens: promptTokens,
            completionTokens: completionTokens))
}

private func assertContinuousHTTPResult(
    _ result: LoadedContinuousHTTPResult,
    equals control: LoadedContinuousControlResult,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(result.statusCode, 200, file: file, line: line)
    XCTAssertEqual(result.text, control.text, file: file, line: line)
    XCTAssertEqual(
        result.finishReason,
        control.completion.finishReason,
        file: file,
        line: line)
    XCTAssertEqual(
        result.usage,
        control.completion.usage,
        file: file,
        line: line)
}

private func assertContinuousHTTPResult(
    _ result: LoadedContinuousHTTPResult,
    equals expected: LoadedContinuousHTTPResult,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(result.statusCode, expected.statusCode, file: file, line: line)
    XCTAssertEqual(result.text, expected.text, file: file, line: line)
    XCTAssertEqual(result.finishReason, expected.finishReason, file: file, line: line)
    XCTAssertEqual(result.usage, expected.usage, file: file, line: line)
}

private func assertContinuousControlResult(
    _ result: LoadedContinuousControlResult,
    equals control: LoadedContinuousControlResult,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(result.text, control.text, file: file, line: line)
    XCTAssertEqual(
        result.completion,
        control.completion,
        file: file,
        line: line)
}

private func runLoadedHTTPQualificationMode(
    mode: LoadedHTTPQualificationMode,
    modelPath: String,
    launchedModel: String,
    memoryLimit: Int,
    cacheLimit: Int,
    maxReservedKV: Int,
    messages: [OpenAIChatMessage],
    maximumCompletionTokens: Int,
    concurrencies: [Int],
    measuredBurstCount: Int,
    expectedControl: LoadedContinuousHTTPResult?,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> LoadedHTTPQualificationModeResult {
    let loaded = try await loadContinuousServingModel(
        configuration: try loadedHTTPQualificationConfiguration(
            mode: mode,
            modelPath: modelPath,
            launchedModel: launchedModel,
            memoryLimit: memoryLimit,
            cacheLimit: cacheLimit,
            maxReservedKV: maxReservedKV,
            maximumCompletionTokens: maximumCompletionTokens))
    let server = try await ServingHTTPServer.start(
        configuration: ServingHTTPConfiguration(
            launchedModel: launchedModel,
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(5)),
        backend: loaded.backend)
    do {
        guard let port = server.localAddress.port else {
            throw LoadedContinuousIntegrationError.missingTCPPort
        }
        var reference = expectedControl
        var concurrencySummaries: [LoadedHTTPConcurrencySummary] = []
        for concurrency in concurrencies {
            let warmup = try await runLoadedHTTPBurst(
                backend: loaded.backend,
                port: port,
                launchedModel: launchedModel,
                messages: messages,
                maximumCompletionTokens: maximumCompletionTokens,
                concurrency: concurrency,
                index: 0,
                measured: false)
            if reference == nil {
                reference = warmup.results.first
            }
            let control = try XCTUnwrap(reference, file: file, line: line)
            XCTAssertEqual(control.statusCode, 200, file: file, line: line)
            XCTAssertEqual(
                control.finishReason,
                .length,
                file: file,
                line: line)
            XCTAssertEqual(
                control.usage.completionTokens,
                maximumCompletionTokens,
                file: file,
                line: line)
            assertLoadedHTTPBurst(
                warmup,
                mode: mode,
                concurrency: concurrency,
                expectedControl: control,
                file: file,
                line: line)

            var measured: [LoadedHTTPBurstSummary] = []
            measured.reserveCapacity(measuredBurstCount)
            for index in 0..<measuredBurstCount {
                let burst = try await runLoadedHTTPBurst(
                    backend: loaded.backend,
                    port: port,
                    launchedModel: launchedModel,
                    messages: messages,
                    maximumCompletionTokens: maximumCompletionTokens,
                    concurrency: concurrency,
                    index: index,
                    measured: true)
                assertLoadedHTTPBurst(
                    burst,
                    mode: mode,
                    concurrency: concurrency,
                    expectedControl: control,
                    file: file,
                    line: line)
                measured.append(burst.summary)
            }
            let concurrencySummary = summarizeLoadedHTTPConcurrency(
                concurrency: concurrency,
                warmup: warmup.summary,
                measured: measured)
            assertLoadedHTTPConcurrencyFacts(
                concurrencySummary,
                mode: mode,
                file: file,
                line: line)
            concurrencySummaries.append(concurrencySummary)
        }
        let summary = LoadedHTTPModeSummary(
            mode: mode.rawValue,
            startupRoute: loaded.startupReport.route.rawValue,
            soloPLDPolicy: loaded.startupReport.soloPLDPolicy == nil
                ? nil : "qwen3-width-one",
            concurrency: concurrencySummaries)
        try await server.shutdown(gracePeriod: .seconds(5))
        await loaded.backend.shutdown()
        return LoadedHTTPQualificationModeResult(
            summary: summary,
            reference: try XCTUnwrap(reference, file: file, line: line))
    } catch {
        try? await server.shutdown(gracePeriod: .seconds(5))
        await loaded.backend.shutdown()
        throw error
    }
}

private func loadedHTTPQualificationConfiguration(
    mode: LoadedHTTPQualificationMode,
    modelPath: String,
    launchedModel: String,
    memoryLimit: Int,
    cacheLimit: Int,
    maxReservedKV: Int,
    maximumCompletionTokens: Int
) throws -> ContinuousServingModelLoadConfiguration {
    // This 8-slot stress fixture diagnoses the internal router. It is deliberately not a
    // shipping CLI configuration and its output is marked non-promotable.
    let admission: ContinuousServingAdmissionMode
    let soloPLDPolicy: ContinuousServingSoloPLDPolicy?
    switch mode {
    case .explicitBatchNoSpec:
        admission = .immediateBatchNoSpec
        soloPLDPolicy = nil
    case .dynamicPLD:
        admission = .dynamic(
            configuration: ServingAdmissionConfiguration(
                soloPLDQualified: true,
                maximumBatchRequests: 8,
                maximumQueuedRequests: 16),
            coalescing: .automatic(.milliseconds(5)))
        soloPLDPolicy = .qwen3WidthOne
    }
    return ContinuousServingModelLoadConfiguration(
        launchedModel: launchedModel,
        modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
        memoryLimitBytes: memoryLimit,
        cacheLimitBytes: cacheLimit,
        maxReservedKVBytes: maxReservedKV,
        coordinatorConfiguration: try ContinuousBatchConfiguration(
            maxActiveSlots: 8,
            maxPrefillSlots: 8,
            prefillChunkSize: 512,
            maxQueuedRequests: 16),
        publicationCapacity: 1,
        traceLimit: 4_096,
        backendConfiguration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: maximumCompletionTokens,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(
                maxDeltas: 8,
                maxBytes: 32 * 1_024),
            admission: admission),
        soloPLDPolicy: soloPLDPolicy)
}

private func loadedHTTPQualificationConcurrencies(
    _ rawValue: String?
) throws -> [Int] {
    guard let rawValue else {
        return [1, 2, 4, 8]
    }
    let values = rawValue.split(separator: ",").compactMap {
        Int($0.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    guard !values.isEmpty,
        values.count == rawValue.split(separator: ",").count,
        values.allSatisfy({ [1, 2, 4, 8].contains($0) }),
        Set(values).count == values.count,
        values.contains(1)
    else {
        throw LoadedContinuousIntegrationError
            .invalidQualificationConcurrencies
    }
    return values
}

private func runLoadedHTTPBurst(
    backend: ContinuousServingBackend,
    port: Int,
    launchedModel: String,
    messages: [OpenAIChatMessage],
    maximumCompletionTokens: Int,
    concurrency: Int,
    index: Int,
    measured: Bool
) async throws -> LoadedHTTPBurstObservation {
    let startSpeculation = await backend
        .diagnosticRuntimeResourceSnapshot()?.speculation
    _ = await backend.diagnosticTakeCoordinatorExecutionTrace()
    let clock = ContinuousClock()
    let startedAt = clock.now
    let results = try await withThrowingTaskGroup(
        of: LoadedContinuousHTTPResult.self
    ) { group in
        for _ in 0..<concurrency {
            group.addTask {
                try await performContinuousRequest(
                    port: port,
                    launchedModel: launchedModel,
                    messages: messages,
                    maximumCompletionTokens: maximumCompletionTokens)
            }
        }
        var collected: [LoadedContinuousHTTPResult] = []
        collected.reserveCapacity(concurrency)
        for try await result in group {
            collected.append(result)
        }
        return collected
    }
    let wallSeconds = max(
        durationSeconds(startedAt.duration(to: clock.now)),
        0.000_001)
    try await waitUntilLoadedContinuous(timeout: .seconds(10)) {
        let snapshot = await backend.snapshot()
        return snapshot.activeRequests == 0
            && snapshot.coordinatorSlots == 0
            && snapshot.reservedKVBytes == 0
    }
    let endSpeculation = await backend
        .diagnosticRuntimeResourceSnapshot()?.speculation
    let trace = await backend.diagnosticTakeCoordinatorExecutionTrace()
    let completionTokens = results.map(\.usage.completionTokens).reduce(0, +)
    let routeFacts = loadedHTTPRouteFacts(
        trace: trace,
        startSpeculation: startSpeculation,
        endSpeculation: endSpeculation)
    return LoadedHTTPBurstObservation(
        summary: LoadedHTTPBurstSummary(
            index: index,
            measured: measured,
            wallMilliseconds: wallSeconds * 1_000,
            completionTokens: completionTokens,
            completionTokensPerSecond:
                Double(completionTokens) / wallSeconds,
            outputSHA256s: loadedHTTPOutputSHA256s(results),
            routeFacts: routeFacts),
        results: results)
}

private func assertLoadedHTTPBurst(
    _ burst: LoadedHTTPBurstObservation,
    mode: LoadedHTTPQualificationMode,
    concurrency: Int,
    expectedControl: LoadedContinuousHTTPResult,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    for result in burst.results {
        assertContinuousHTTPResult(
            result,
            equals: expectedControl,
            file: file,
            line: line)
    }
    let facts = burst.summary.routeFacts
    switch (mode, concurrency) {
    case (.dynamicPLD, 1):
        guard burst.summary.measured else { return }
        XCTAssertTrue(facts.sawSoloSpeculationAllowed, file: file, line: line)
        XCTAssertTrue(
            facts.speculationEngagedDuringInterval,
            file: file,
            line: line)
        XCTAssertGreaterThan(
            facts.acceptedDraftTokensDelta,
            0,
            file: file,
            line: line)
        XCTAssertGreaterThan(
            facts.verificationRoundsDelta,
            0,
            file: file,
            line: line)
    case (.dynamicPLD, _):
        guard burst.summary.measured else { return }
        XCTAssertTrue(facts.sawSharedBatch, file: file, line: line)
        XCTAssertEqual(
            facts.maxDecodeBatchSize,
            concurrency,
            file: file,
            line: line)
        XCTAssertTrue(
            facts.allDecodeSpeculationDisabled,
            file: file,
            line: line)
        XCTAssertFalse(facts.speculationAllowed, file: file, line: line)
        XCTAssertFalse(
            facts.speculationEngagedDuringInterval,
            file: file,
            line: line)
    case (.explicitBatchNoSpec, _):
        guard burst.summary.measured else { return }
        if concurrency >= 2 {
            XCTAssertTrue(facts.sawSharedBatch, file: file, line: line)
            XCTAssertEqual(
                facts.maxDecodeBatchSize,
                concurrency,
                file: file,
                line: line)
        }
        XCTAssertTrue(
            facts.allDecodeSpeculationDisabled,
            file: file,
            line: line)
        XCTAssertFalse(facts.speculationAllowed, file: file, line: line)
        XCTAssertFalse(
            facts.speculationEngagedDuringInterval,
            file: file,
            line: line)
    }
}

private func assertLoadedHTTPConcurrencyFacts(
    _ summary: LoadedHTTPConcurrencySummary,
    mode: LoadedHTTPQualificationMode,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    switch (mode, summary.concurrency) {
    case (.dynamicPLD, 1):
        XCTAssertTrue(
            summary.routeFacts.sawSoloSpeculationAllowed,
            file: file,
            line: line)
        XCTAssertTrue(
            summary.routeFacts.speculationEngagedDuringInterval,
            file: file,
            line: line)
    case (.dynamicPLD, _):
        XCTAssertTrue(summary.routeFacts.sawSharedBatch, file: file, line: line)
        XCTAssertEqual(
            summary.routeFacts.maxDecodeBatchSize,
            summary.concurrency,
            file: file,
            line: line)
        XCTAssertTrue(
            summary.routeFacts.allDecodeSpeculationDisabled,
            file: file,
            line: line)
        XCTAssertFalse(
            summary.routeFacts.speculationAllowed,
            file: file,
            line: line)
        XCTAssertFalse(
            summary.routeFacts.speculationEngagedDuringInterval,
            file: file,
            line: line)
    case (.explicitBatchNoSpec, _):
        if summary.concurrency >= 2 {
            XCTAssertTrue(
                summary.routeFacts.sawSharedBatch,
                file: file,
                line: line)
            XCTAssertEqual(
                summary.routeFacts.maxDecodeBatchSize,
                summary.concurrency,
                file: file,
                line: line)
        }
        XCTAssertTrue(
            summary.routeFacts.allDecodeSpeculationDisabled,
            file: file,
            line: line)
        XCTAssertFalse(
            summary.routeFacts.speculationAllowed,
            file: file,
            line: line)
    }
}

private func summarizeLoadedHTTPConcurrency(
    concurrency: Int,
    warmup: LoadedHTTPBurstSummary,
    measured: [LoadedHTTPBurstSummary]
) -> LoadedHTTPConcurrencySummary {
    let wallMilliseconds = measured.map(\.wallMilliseconds)
    let aggregateWallMilliseconds = wallMilliseconds.reduce(0, +)
    let completionTokens = measured.map(\.completionTokens).reduce(0, +)
    let aggregateSeconds = max(aggregateWallMilliseconds / 1_000, 0.000_001)
    let throughputs = measured.map(\.completionTokensPerSecond)
    return LoadedHTTPConcurrencySummary(
        concurrency: concurrency,
        warmup: warmup,
        measured: measured,
        measuredWallMilliseconds: wallMilliseconds,
        aggregateWallMilliseconds: aggregateWallMilliseconds,
        aggregateCompletionTokensPerSecond:
            Double(completionTokens) / aggregateSeconds,
        meanCompletionTokensPerSecond: mean(throughputs),
        medianCompletionTokensPerSecond: median(throughputs),
        outputSHA256s: Array(
            Set(measured.flatMap(\.outputSHA256s)).sorted()),
        routeFacts: combineLoadedHTTPRouteFacts(measured.map(\.routeFacts)))
}

private func loadedHTTPRouteFacts(
    trace: [ContinuousBatchCoordinatorEvent],
    startSpeculation: ContinuousBatchRuntimeSpeculationSnapshot?,
    endSpeculation: ContinuousBatchRuntimeSpeculationSnapshot?
) -> LoadedHTTPRouteFacts {
    var decodeOperationCount = 0
    var sawSolo = false
    var sawSoloSpeculationAllowed = false
    var sawSharedBatch = false
    var maxDecodeBatchSize = 0
    var speculationAllowed = false
    for event in trace {
        guard case .operation(.decode(let action)) = event else {
            continue
        }
        switch action {
        case .drainSoloPipeline:
            continue
        case .solo(_, let allowed):
            decodeOperationCount += 1
            sawSolo = true
            sawSoloSpeculationAllowed = sawSoloSpeculationAllowed || allowed
            maxDecodeBatchSize = max(maxDecodeBatchSize, 1)
            speculationAllowed = speculationAllowed || allowed
        case .batch(let ids, let allowed):
            decodeOperationCount += 1
            sawSharedBatch = sawSharedBatch || ids.count > 1
            maxDecodeBatchSize = max(maxDecodeBatchSize, ids.count)
            speculationAllowed = speculationAllowed || allowed
        }
    }
    return LoadedHTTPRouteFacts(
        decodeOperationCount: decodeOperationCount,
        sawSolo: sawSolo,
        sawSoloSpeculationAllowed: sawSoloSpeculationAllowed,
        sawSharedBatch: sawSharedBatch,
        maxDecodeBatchSize: maxDecodeBatchSize,
        speculationAllowed: speculationAllowed,
        allDecodeSpeculationDisabled:
            decodeOperationCount > 0 && trace.allSatisfy(traceDisablesSpeculation),
        speculationEngagedDuringInterval: speculationEngagedDuringInterval(
            from: startSpeculation,
            to: endSpeculation),
        speculationRequestedRequestsDelta: speculationDelta(
            startSpeculation,
            endSpeculation,
            \.requestedRequests),
        draftedTokensDelta: speculationDelta(
            startSpeculation,
            endSpeculation,
            \.draftedTokens),
        acceptedDraftTokensDelta: speculationDelta(
            startSpeculation,
            endSpeculation,
            \.acceptedDraftTokens),
        verificationRoundsDelta: speculationDelta(
            startSpeculation,
            endSpeculation,
            \.verificationRounds))
}

private func combineLoadedHTTPRouteFacts(
    _ facts: [LoadedHTTPRouteFacts]
) -> LoadedHTTPRouteFacts {
    LoadedHTTPRouteFacts(
        decodeOperationCount: facts.map(\.decodeOperationCount).reduce(0, +),
        sawSolo: facts.contains { $0.sawSolo },
        sawSoloSpeculationAllowed: facts.contains {
            $0.sawSoloSpeculationAllowed
        },
        sawSharedBatch: facts.contains { $0.sawSharedBatch },
        maxDecodeBatchSize: facts.map(\.maxDecodeBatchSize).max() ?? 0,
        speculationAllowed: facts.contains { $0.speculationAllowed },
        allDecodeSpeculationDisabled: !facts.isEmpty
            && facts.allSatisfy(\.allDecodeSpeculationDisabled),
        speculationEngagedDuringInterval: facts.contains {
            $0.speculationEngagedDuringInterval
        },
        speculationRequestedRequestsDelta:
            facts.map(\.speculationRequestedRequestsDelta).reduce(0, +),
        draftedTokensDelta: facts.map(\.draftedTokensDelta).reduce(0, +),
        acceptedDraftTokensDelta:
            facts.map(\.acceptedDraftTokensDelta).reduce(0, +),
        verificationRoundsDelta:
            facts.map(\.verificationRoundsDelta).reduce(0, +))
}

private func speculationDelta(
    _ start: ContinuousBatchRuntimeSpeculationSnapshot?,
    _ end: ContinuousBatchRuntimeSpeculationSnapshot?,
    _ keyPath: KeyPath<ContinuousBatchRuntimeSpeculationSnapshot, Int>
) -> Int {
    guard let start, let end else { return 0 }
    return max(0, end[keyPath: keyPath] - start[keyPath: keyPath])
}

private func loadedHTTPOutputSHA256s(
    _ results: [LoadedContinuousHTTPResult]
) -> [String] {
    Array(
        Set(
            results.map {
                sha256Hex(Data($0.text.utf8))
            }
        ).sorted())
}

private func mean(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    return values.reduce(0, +) / Double(values.count)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
    return sorted[middle]
}

private func durationSeconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds)
        + Double(components.attoseconds) / 1e18
}

private func encodeLoadedHTTPPhase4Summary(
    _ summary: LoadedHTTPPhase4Summary
) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(summary)
    return String(decoding: data, as: UTF8.self)
}

private enum ExpectedCancellationPhase {
    case prefilling
    case decoding
}

private func traceContainsCancellation(
    _ trace: [ContinuousBatchCoordinatorEvent],
    id: BatchRequestID,
    phase expectedPhase: ExpectedCancellationPhase
) -> Bool {
    trace.contains { event in
        guard case .cancelled(
            .cancelled(let cancelledID, let previousPhase)
        ) = event, cancelledID == id else {
            return false
        }
        switch (expectedPhase, previousPhase) {
        case (.prefilling, .prefilling), (.decoding, .decoding):
            return true
        default:
            return false
        }
    }
}

private func traceContainsSharedBatch(
    _ trace: [ContinuousBatchCoordinatorEvent],
    ids expectedIDs: Set<BatchRequestID>
) -> Bool {
    trace.contains { event in
        guard case .operation(
            .decode(.batch(let ids, speculationAllowed: false))
        ) = event else {
            return false
        }
        return expectedIDs.isSubset(of: Set(ids))
    }
}

private func traceContainsSpeculativeSolo(
    _ trace: [ContinuousBatchCoordinatorEvent]
) -> Bool {
    trace.contains { event in
        if case .operation(
            .decode(.solo(_, speculationAllowed: true))
        ) = event {
            return true
        }
        return false
    }
}

private func traceBatchesStayWithin(
    _ trace: [ContinuousBatchCoordinatorEvent],
    permittedGroups: [Set<BatchRequestID>]
) -> Bool {
    trace.allSatisfy { event in
        guard case .operation(
            .decode(.batch(let ids, speculationAllowed: _))
        ) = event else {
            return true
        }
        let batchIDs = Set(ids)
        return permittedGroups.contains {
            batchIDs.isSubset(of: $0)
        }
    }
}

private func traceDisablesSpeculation(
    _ event: ContinuousBatchCoordinatorEvent
) -> Bool {
    guard case .operation(.decode(let action)) = event else {
        return true
    }
    switch action {
    case .drainSoloPipeline:
        return true
    case .solo(_, let speculationAllowed),
        .batch(_, let speculationAllowed):
        return !speculationAllowed
    }
}

private func waitUntilLoadedContinuous(
    timeout: Duration,
    _ predicate: () async -> Bool
) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if await predicate() {
            return
        }
        try await Task.sleep(for: .milliseconds(1))
    }
    throw LoadedContinuousIntegrationError.timeout
}

private enum LoadedContinuousIntegrationError: Error {
    case missingTCPPort
    case missingCompletion
    case invalidHTTPResponse
    case invalidQualificationConcurrencies
    case invalidQualificationModeOrder
    case timeout
}
