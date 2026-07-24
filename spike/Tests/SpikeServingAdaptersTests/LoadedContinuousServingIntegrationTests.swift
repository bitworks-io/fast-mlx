import Foundation
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
    case timeout
}
