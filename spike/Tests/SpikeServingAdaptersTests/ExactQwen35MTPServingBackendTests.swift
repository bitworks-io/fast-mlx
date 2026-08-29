import XCTest
import os

@testable import HarnessCore
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ExactQwen35MTPServingBackendTests: XCTestCase {
    func testEligibleGreedyRequestUsesMTPRouteAndPublishesTextUsageAndLength() async throws {
        let runner = ScriptedMTPRunner(script: .completed(text: ["hel", "lo"], promptTokens: 2, completionTokens: 2, stopReason: .length))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 2))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .exactQwen35MTP)
        XCTAssertEqual(events, [
            .text("hel"),
            .text("lo"),
            .completion(
                ServingGenerationCompletion(
                    finishReason: .length,
                    usage: OpenAIChatUsage(promptTokens: 2, completionTokens: 2))),
        ])
        let runnerSnapshot = runner.snapshot()
        XCTAssertEqual(runnerSnapshot.startCount, 1)
        XCTAssertEqual(runnerSnapshot.lastMaximumCompletionTokens, 2)
        XCTAssertEqual(scalar.snapshot().startCount, 0)
        await waitUntil {
            await backend.snapshot().activeMTPReservations == 0
        }
    }

    func testIneligibleRequestsFallBackWithoutCallingMTPRunner() async throws {
        for request in [
            request(maxTokens: 2, temperature: 0.7),
            request(maxTokens: 2, presencePenalty: 0.1),
            request(maxTokens: 2, tools: [weatherTool]),
        ] {
            let runner = ScriptedMTPRunner(script: .completed(text: ["mtp"], promptTokens: 1, completionTokens: 1, stopReason: .stop))
            let scalar = ScriptedScalarFallback()
            let backend = try makeBackend(runner: runner, scalarFallback: scalar)

            let handle = try await backend.start(request)
            _ = try await collect(handle.mailbox)

            XCTAssertEqual(handle.route, .scalarGreedy)
            XCTAssertEqual(runner.snapshot().startCount, 0)
            XCTAssertEqual(scalar.snapshot().startCount, 1)
        }
    }

    func testMissingBindingFallsBackWithoutCallingMTPRunner() async throws {
        let runner = ScriptedMTPRunner(
            binding: nil,
            script: .completed(
                text: ["mtp"], promptTokens: 1, completionTokens: 1, stopReason: .stop))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 2))
        _ = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .scalarGreedy)
        XCTAssertEqual(runner.snapshot().startCount, 0)
        XCTAssertEqual(scalar.snapshot().startCount, 1)
    }

    func testRunnerLoadFailureReleasesReservationAndFallsBackScalar() async throws {
        let runner = ScriptedMTPRunner(script: .failToStart)
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 2))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .scalarGreedy)
        XCTAssertEqual(events.last, .completion(
            ServingGenerationCompletion(
                finishReason: .stop,
                usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1))))
        XCTAssertEqual(runner.snapshot().startCount, 1)
        XCTAssertEqual(scalar.snapshot().startCount, 1)
        let snapshot = await backend.snapshot()
        XCTAssertEqual(snapshot.activeMTPReservations, 0)
    }

    func testOverlappingMTPRequestFallsBackScalarWithoutCallingRunnerTwice() async throws {
        let gate = MTPGate()
        let runner = ScriptedMTPRunner(script: .held(gate))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let active = try await backend.start(request(maxTokens: 4))
        await waitUntil {
            await backend.snapshot().activeMTPReservations == 1
        }

        let overlapping = try await backend.start(request(maxTokens: 2))
        _ = try await collect(overlapping.mailbox)

        XCTAssertEqual(active.route, .exactQwen35MTP)
        XCTAssertEqual(overlapping.route, .scalarGreedy)
        XCTAssertEqual(runner.snapshot().startCount, 1)
        XCTAssertEqual(scalar.snapshot().startCount, 1)

        await gate.finish()
        _ = try await collect(active.mailbox)
        await waitUntil {
            await backend.snapshot().activeMTPReservations == 0
        }
    }

    func testRequestStopSplitAcrossMTPChunksIsNotPublished() async throws {
        let runner = ScriptedMTPRunner(
            script: .completed(
                text: ["hello<", "stop", ">hidden", "tail"],
                promptTokens: 2,
                completionTokens: 4,
                stopReason: .length))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 8, stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .exactQwen35MTP)
        XCTAssertEqual(events, [
            .text("hello"),
            .completion(
                ServingGenerationCompletion(
                    finishReason: .stop,
                    usage: OpenAIChatUsage(promptTokens: 2, completionTokens: 3))),
        ])
        XCTAssertEqual(runner.snapshot().lastStopStrings, ["<stop>"])
    }

    func testCancellationFinalizesUpstreamTaskAndReleasesReservation() async throws {
        let gate = MTPGate()
        let runner = ScriptedMTPRunner(script: .heldAfterChunk(gate, text: "ready"))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 4))
        await waitUntil {
            await backend.snapshot().activeMTPReservations == 1
        }
        let firstDelta = try await handle.mailbox.next()
        XCTAssertEqual(firstDelta, .text("ready"))

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(handle.mailbox, reason: .clientDisconnected)
        await waitUntil {
            let runnerSnapshot = runner.snapshot()
            let backendSnapshot = await backend.snapshot()
            return runnerSnapshot.cancelledTaskCount == 1
                && runnerSnapshot.finalizedTaskCount == 1
                && backendSnapshot.activeMTPReservations == 0
        }
        XCTAssertEqual(scalar.snapshot().startCount, 0)
    }

    func testShutdownWaitsForSuspendedStartupThenRejectsTheRequestAndFutureAdmission() async throws {
        let startGate = MTPStartGate()
        let runner = ScriptedMTPRunner(script: .suspendedStart(startGate))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)
        let startTask = Task { () -> ExactQwen35MTPServingBackendError? in
            do {
                _ = try await backend.start(request(maxTokens: 4))
                return nil
            } catch {
                return error as? ExactQwen35MTPServingBackendError
            }
        }
        await startGate.waitUntilEntered()

        let shutdownFinished = OSAllocatedUnfairLock(initialState: false)
        let shutdownTask = Task {
            await backend.shutdown()
            shutdownFinished.withLock { $0 = true }
        }
        await waitUntil {
            let snapshot = await backend.snapshot()
            return !snapshot.acceptingRequests
        }

        let suspendedSnapshot = await backend.snapshot()
        XCTAssertEqual(suspendedSnapshot.pendingMTPStartups, 1)
        XCTAssertEqual(suspendedSnapshot.activeMTPReservations, 1)
        XCTAssertFalse(shutdownFinished.withLock { $0 })

        await startGate.finish()
        await shutdownTask.value
        let startupError = await startTask.value
        XCTAssertEqual(startupError, .shuttingDown)
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(finalSnapshot.pendingMTPStartups, 0)
        XCTAssertEqual(finalSnapshot.activeMTPReservations, 0)
        XCTAssertFalse(finalSnapshot.acceptingRequests)
        XCTAssertEqual(scalar.snapshot().shutdownCount, 1)

        do {
            _ = try await backend.start(request(maxTokens: 1))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ExactQwen35MTPServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }

    func testUnexpectedToolEventFailsClosedWithoutPublishingToolCalls() async throws {
        let runner = ScriptedMTPRunner(script: .unexpectedToolCall)
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(runner: runner, scalarFallback: scalar)

        let handle = try await backend.start(request(maxTokens: 4))
        XCTAssertEqual(handle.route, .exactQwen35MTP)
        await assertMailboxBackendFailed(handle.mailbox)
        await waitUntil {
            await backend.snapshot().activeMTPReservations == 0
        }
        XCTAssertEqual(scalar.snapshot().startCount, 0)
    }

    func testConstructionRejectsScalarFallbackThatMayShareRawTarget() {
        XCTAssertThrowsError(
            try makeBackend(
                runner: ScriptedMTPRunner(script: .completed(text: [], promptTokens: 1, completionTokens: 0, stopReason: .stop)),
                scalarFallback: ScriptedScalarFallback(),
                scalarFallbackIsolation: .sharedRawTarget)
        ) { error in
            XCTAssertEqual(
                error as? ExactQwen35MTPServingBackendConstructionError,
                .scalarFallbackMustUseSeparateRawTarget)
        }
    }

    func testModelAwareBudgetUsesExactPromptAndReachesRunnerAndHandle() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 6,
            requestedDefaultCompletionTokens: 4,
            maximumNonStreamingCompletionTokens: 5,
            completionLimitPolicy: .clamp)
        let runner = ScriptedMTPRunner(
            script: .completed(
                text: ["a", "b", "c", "d"],
                promptTokens: 2,
                completionTokens: 4,
                stopReason: .length))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(
            runner: runner,
            scalarFallback: scalar,
            modelCapabilities: capabilities)

        let handle = try await backend.start(request(maxTokens: 8))
        let resolution = try XCTUnwrap(handle.completionBudgetResolution)
        XCTAssertEqual(resolution.renderedPromptTokens, 2)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 4)
        XCTAssertEqual(resolution.appliedCompletionTokens, 4)
        XCTAssertTrue(resolution.wasClamped)
        XCTAssertEqual(runner.snapshot().lastMaximumCompletionTokens, 4)
        _ = try await collect(handle.mailbox)
    }

    func testModelAwareRejectionMutatesNoStartupReservationOrRunnerState() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 4,
            requestedDefaultCompletionTokens: 2,
            maximumNonStreamingCompletionTokens: 3,
            completionLimitPolicy: .reject)
        let runner = ScriptedMTPRunner(
            script: .completed(
                text: ["unused"],
                promptTokens: 2,
                completionTokens: 1,
                stopReason: .stop))
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(
            runner: runner,
            scalarFallback: scalar,
            modelCapabilities: capabilities)

        do {
            _ = try await backend.start(request(maxTokens: 4))
            XCTFail("Expected model-aware budget rejection")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "completion_limit_exceeded")
        }

        XCTAssertEqual(runner.snapshot().startCount, 0)
        XCTAssertEqual(scalar.snapshot().startCount, 0)
        let snapshot = await backend.snapshot()
        XCTAssertEqual(snapshot.pendingMTPStartups, 0)
        XCTAssertEqual(snapshot.activeMTPReservations, 0)
    }

    func testRunnerFailurePreservesResolvedBudgetAcrossScalarFallback() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 6,
            requestedDefaultCompletionTokens: 4,
            maximumNonStreamingCompletionTokens: 5,
            completionLimitPolicy: .clamp)
        let runner = ScriptedMTPRunner(script: .failToStart)
        let scalar = ScriptedScalarFallback()
        let backend = try makeBackend(
            runner: runner,
            scalarFallback: scalar,
            modelCapabilities: capabilities)

        let handle = try await backend.start(request(maxTokens: 8))
        _ = try await collect(handle.mailbox)

        let preserved = try XCTUnwrap(scalar.snapshot().lastCompletionBudgetResolution)
        XCTAssertEqual(preserved.appliedCompletionTokens, 4)
        XCTAssertEqual(preserved.renderedPromptTokens, 2)
        XCTAssertEqual(handle.completionBudgetResolution, preserved)
    }
}

private func makeBackend(
    enabled: Bool = true,
    runner: ScriptedMTPRunner,
    scalarFallback: ScriptedScalarFallback,
    scalarFallbackIsolation: ExactQwen35MTPScalarFallbackIsolation = .strictlySeparateRawTarget,
    modelCapabilities: ServingModelCapabilities? = nil
) throws -> ExactQwen35MTPServingBackend {
    try ExactQwen35MTPServingBackend(
        launchedModel: "fixture-model",
        enabled: enabled,
        runner: runner,
        scalarFallback: scalarFallback,
        scalarFallbackIsolation: scalarFallbackIsolation,
        codec: FixtureScalarTextCodec(promptTokens: [10, 11]),
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            modelCapabilities: modelCapabilities))
}

private let lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1

private func exactBinding() -> QwenMTPArtifactBinding {
    QwenMTPArtifactBinding(
        targetModelID: lock.targetIdentity.modelID,
        drafterModelID: lock.drafterIdentity.modelID,
        targetRevision: lock.targetIdentity.revision,
        drafterRevision: lock.drafterIdentity.revision,
        sourceRevision: lock.sourceRevision,
        architecture: lock.architecture,
        runtimeBlockSize: 3,
        maximumAcceptedDraftTokens: 2)
}

private func request(
    maxTokens: Int,
    temperature: Double? = 0,
    presencePenalty: Double? = nil,
    stop: [String] = [],
    tools: [OpenAIToolSpec] = []
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [OpenAIChatMessage(role: .user, text: "private prompt")],
        maxCompletionTokens: maxTokens,
        temperature: temperature,
        choiceCount: 1,
        stream: true,
        stop: stop,
        tools: tools,
        toolChoice: tools.isEmpty ? .none : .auto,
        presencePenalty: presencePenalty)
}

private func collect(_ mailbox: BoundedDeltaMailbox) async throws -> [ServingResponseDelta] {
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
        XCTAssertEqual(error, .cancelled(reason), file: file, line: line)
    } catch {
        XCTFail("Unexpected mailbox error: \(error)", file: file, line: line)
    }
}

private func assertMailboxBackendFailed(
    _ mailbox: BoundedDeltaMailbox,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await mailbox.next()
        XCTFail("Expected mailbox backend failure", file: file, line: line)
    } catch let error as ServingMailboxError {
        guard case .backend = error else {
            XCTFail("Unexpected mailbox error: \(error)", file: file, line: line)
            return
        }
    } catch {
        XCTFail("Unexpected mailbox error: \(error)", file: file, line: line)
    }
}

private func waitUntil(
    attempts: Int = 10_000,
    _ predicate: () async -> Bool
) async {
    for _ in 0..<attempts {
        if await predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}

private struct FixtureScalarTextCodec: ScalarServingTextCodec {
    let promptTokens: [Int]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        promptTokens
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureScalarDetokenizer()
    }
}

private struct FixtureScalarDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) {}
    mutating func next() -> String? { nil }
}

private let weatherTool = OpenAIToolSpec(
    name: "get_weather",
    description: "Look up the weather for a city",
    parameters: .object(["type": .string("object")]),
    raw: .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("get_weather"),
            "description": .string("Look up the weather for a city"),
            "parameters": .object(["type": .string("object")]),
        ]),
    ]))

private final class ScriptedScalarFallback: ServingGenerationBackend, Sendable {
    struct Snapshot: Sendable {
        let startCount: Int
        let shutdownCount: Int
        let lastCompletionBudgetResolution: ServingCompletionBudgetResolution?
    }

    private struct State: Sendable {
        var startCount = 0
        var shutdownCount = 0
        var lastCompletionBudgetResolution: ServingCompletionBudgetResolution?
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        try await start(request, completionBudgetResolution: nil)
    }

    func start(
        _ request: OpenAIChatCompletionRequest,
        resolvedCompletionBudget: ServingCompletionBudgetResolution
    ) async throws -> ServingGenerationHandle {
        try await start(
            request,
            completionBudgetResolution: resolvedCompletionBudget)
    }

    private func start(
        _ request: OpenAIChatCompletionRequest,
        completionBudgetResolution: ServingCompletionBudgetResolution?
    ) async throws -> ServingGenerationHandle {
        let sequence = state.withLock { state in
            state.startCount += 1
            state.lastCompletionBudgetResolution = completionBudgetResolution
            return state.startCount
        }
        let mailbox = BoundedDeltaMailbox(capacity: .init(maxDeltas: 4, maxBytes: 1_024))
        let lease = ServingRequestLease(id: ServingRequestID("scalar-\(sequence)"))
        Task {
            do {
                try await mailbox.send(.text("scalar"))
                try await mailbox.send(
                    .completion(
                        ServingGenerationCompletion(
                            finishReason: .stop,
                            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1))))
                await mailbox.finish()
            } catch {}
        }
        return ServingGenerationHandle(
            responseID: "chatcmpl-scalar-\(sequence)",
            created: 1,
            model: request.model,
            route: .scalarGreedy,
            mailbox: mailbox,
            lease: lease,
            completionBudgetResolution: completionBudgetResolution)
    }

    func shutdown() async {
        state.withLock { $0.shutdownCount += 1 }
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                startCount: $0.startCount,
                shutdownCount: $0.shutdownCount,
                lastCompletionBudgetResolution: $0.lastCompletionBudgetResolution)
        }
    }
}

private final class ScriptedMTPRunner: ExactQwen35MTPServingRunner, Sendable {
    let binding: QwenMTPArtifactBinding?

    enum Script: Sendable {
        case completed(
            text: [String],
            promptTokens: Int,
            completionTokens: Int,
            stopReason: GenerateStopReason)
        case held(MTPGate)
        case heldAfterChunk(MTPGate, text: String)
        case suspendedStart(MTPStartGate)
        case unexpectedToolCall
        case failToStart
    }

    struct Snapshot: Sendable {
        let startCount: Int
        let cancelledTaskCount: Int
        let finalizedTaskCount: Int
        let lastStopStrings: Set<String>
        let lastMaximumCompletionTokens: Int?
    }

    private struct State: Sendable {
        var startCount = 0
        var cancelledTaskCount = 0
        var finalizedTaskCount = 0
        var lastStopStrings: Set<String> = []
        var lastMaximumCompletionTokens: Int?
    }

    private let script: Script
    private let state = OSAllocatedUnfairLock(initialState: State())

    init(binding: QwenMTPArtifactBinding? = exactBinding(), script: Script) {
        self.binding = binding
        self.script = script
    }

    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        state.withLock {
            $0.startCount += 1
            $0.lastStopStrings = request.stopStrings
            $0.lastMaximumCompletionTokens = request.maximumCompletionTokens
        }
        switch script {
        case .failToStart:
            throw FixtureMTPError.loadFailed
        case .suspendedStart(let gate):
            await gate.enterAndWait()
            let (stream, continuation) = AsyncStream<Generation>.makeStream()
            let task = Task {
                continuation.finish()
                state.withLock { $0.finalizedTaskCount += 1 }
            }
            return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
        case .unexpectedToolCall:
            let (stream, continuation) = AsyncStream<Generation>.makeStream()
            let task = Task {
                continuation.yield(
                    .toolCall(
                        ToolCall(
                            function: .init(
                                name: "unexpected",
                                arguments: [String: JSONValue]()))))
                continuation.yield(
                    .info(
                        GenerateCompletionInfo(
                            promptTokenCount: request.promptTokens.count,
                            generationTokenCount: 1,
                            promptTime: 0,
                            generationTime: 0,
                            stopReason: .stop)))
                continuation.finish()
                state.withLock { $0.finalizedTaskCount += 1 }
            }
            return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
        case .completed(let text, let promptTokens, let completionTokens, let stopReason):
            let (stream, continuation) = AsyncStream<Generation>.makeStream()
            let task = Task {
                var stopFilter = ServingStopStringFilter(stopStrings: request.stopStrings)
                var emittedChunks = 0
                var stopped = false
                for chunk in text {
                    guard !stopped else {
                        break
                    }
                    emittedChunks += 1
                    let output = stopFilter.process(chunk)
                    if let text = output.text {
                        continuation.yield(.chunk(text))
                    }
                    stopped = output.stopped
                }
                if !stopped, let tail = stopFilter.finish() {
                    continuation.yield(.chunk(tail))
                }
                continuation.yield(
                    .info(
                        GenerateCompletionInfo(
                            promptTokenCount: promptTokens,
                            generationTokenCount: stopped ? emittedChunks : completionTokens,
                            promptTime: 0,
                            generationTime: 0,
                            stopReason: stopped ? .stop : stopReason)))
                continuation.finish()
                state.withLock { $0.finalizedTaskCount += 1 }
            }
            return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
        case .held(let gate):
            let (stream, continuation) = AsyncStream<Generation>.makeStream()
            let task = Task {
                await withTaskCancellationHandler {
                    await gate.wait()
                    if !Task.isCancelled {
                        continuation.yield(
                            .info(
                                GenerateCompletionInfo(
                                    promptTokenCount: request.promptTokens.count,
                                    generationTokenCount: 0,
                                    promptTime: 0,
                                    generationTime: 0,
                                    stopReason: .stop)))
                    }
                    continuation.finish()
                } onCancel: {
                    continuation.finish()
                    state.withLock { $0.cancelledTaskCount += 1 }
                    Task { await gate.finish() }
                }
                state.withLock { $0.finalizedTaskCount += 1 }
            }
            return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
        case .heldAfterChunk(let gate, let text):
            let (stream, continuation) = AsyncStream<Generation>.makeStream()
            let task = Task {
                continuation.yield(.chunk(text))
                await withTaskCancellationHandler {
                    await gate.wait()
                    if !Task.isCancelled {
                        continuation.yield(
                            .info(
                                GenerateCompletionInfo(
                                    promptTokenCount: request.promptTokens.count,
                                    generationTokenCount: 1,
                                    promptTime: 0,
                                    generationTime: 0,
                                    stopReason: .stop)))
                    }
                    continuation.finish()
                } onCancel: {
                    continuation.finish()
                    state.withLock { $0.cancelledTaskCount += 1 }
                    Task { await gate.finish() }
                }
                state.withLock { $0.finalizedTaskCount += 1 }
            }
            return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
        }
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                startCount: $0.startCount,
                cancelledTaskCount: $0.cancelledTaskCount,
                finalizedTaskCount: $0.finalizedTaskCount,
                lastStopStrings: $0.lastStopStrings,
                lastMaximumCompletionTokens: $0.lastMaximumCompletionTokens)
        }
    }
}

private enum FixtureMTPError: Error {
    case loadFailed
}

private actor MTPGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var finished = false

    func wait() async {
        if finished { return }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func finish() {
        finished = true
        continuation?.resume()
        continuation = nil
    }
}

private actor MTPStartGate {
    private var entered = false
    private var released = false
    private var releaseContinuation: CheckedContinuation<Void, Never>?
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []

    func enterAndWait() async {
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll(keepingCapacity: false)
        for waiter in waiters {
            waiter.resume()
        }
        guard !released else { return }
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func finish() {
        released = true
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}
