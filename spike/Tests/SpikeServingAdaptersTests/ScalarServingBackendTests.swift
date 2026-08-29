import XCTest
import os

import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ScalarServingBackendTests: XCTestCase {
    // The scalar backend must build its ToolCallProcessor with the model's configured wire format,
    // not a hardcoded `.json`. A qwen3_5 model emits xmlFunction syntax
    // (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`) which the
    // JSON parser cannot decode. With `toolCallFormat: .xmlFunction` the call must parse.
    func testScalarRouteHonorsXMLFunctionToolCallFormat() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [
                1: "<tool_call><function=get_weather>",
                2: "<parameter=city>Paris</parameter>",
                3: "</function></tool_call>",
            ],
            promptTokens: [10],
            toolCallFormat: .xmlFunction)
        let handle = try await backend.start(request(maxTokens: 4, tools: [weatherTool]))

        let events = try await collect(handle.mailbox)

        let toolCalls = events.compactMap { event -> [OpenAIToolCall]? in
            if case .toolCalls(let calls) = event { return calls }
            return nil
        }.first
        let call = try XCTUnwrap(toolCalls?.first, "xmlFunction tool call must be parsed: \(events)")
        XCTAssertEqual(call.function.name, "get_weather")
        XCTAssertTrue(call.function.arguments.contains("Paris"), "args: \(call.function.arguments)")

        let finish = events.compactMap { event -> OpenAIChatFinishReason? in
            if case .completion(let completion) = event { return completion.finishReason }
            return nil
        }.first
        XCTAssertEqual(finish, .toolCalls)
    }

    // Negative control: the SAME xmlFunction stream under the default `.json` format must NOT
    // parse as a tool call — proving the format field is actually consumed, not incidental.
    func testScalarRouteJSONFormatDoesNotParseXMLFunctionStream() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [
                1: "<tool_call><function=get_weather>",
                2: "<parameter=city>Paris</parameter>",
                3: "</function></tool_call>",
            ],
            promptTokens: [10],
            toolCallFormat: .json)
        let handle = try await backend.start(request(maxTokens: 4, tools: [weatherTool]))

        let events = try await collect(handle.mailbox)

        let hasToolCalls = events.contains { event in
            if case .toolCalls = event { return true }
            return false
        }
        XCTAssertFalse(hasToolCalls, "JSON parser must not decode xmlFunction syntax: \(events)")
    }
    // Handle plumbing for streaming reasoning separation. The backend must derive
    // `handle.separatesReasoning` from the family (`thinksByDefault`) folded with the SAME resolved
    // thinking value it renders the prompt from — so the SSE handler can route a thinks-by-default
    // stream through the splitter without re-deriving anything.
    func testHandleSeparatesReasoningWhenFamilyThinksAndThinkingNotOff() async throws {
        // qwen3_5-class family + client omits enable_thinking (template default = thinking) → separate.
        let backend = makeBackend(
            script: [1, 99], pieces: [1: "hi"], promptTokens: [10],
            thinksByDefault: true)
        let handle = try await backend.start(request(maxTokens: 2))
        XCTAssertTrue(handle.separatesReasoning)
        _ = try await collect(handle.mailbox)
    }

    func testHandleDoesNotSeparateWhenThinkingExplicitlyOff() async throws {
        // Same family, but the request turns thinking OFF → a closed empty <think></think> is injected,
        // nothing is generated to split → passthrough.
        let backend = makeBackend(
            script: [1, 99], pieces: [1: "hi"], promptTokens: [10],
            thinksByDefault: true)
        let handle = try await backend.start(request(maxTokens: 2, enableThinking: false))
        XCTAssertFalse(handle.separatesReasoning)
        _ = try await collect(handle.mailbox)
    }

    func testHandleDoesNotSeparateWhenFamilyDoesNotThinkByDefault() async throws {
        // Dense/compiled family (thinksByDefault=false) → never separate, even with thinking on. This is
        // the conservative default that keeps today's streams byte-identical.
        let backend = makeBackend(
            script: [1, 99], pieces: [1: "hi"], promptTokens: [10],
            thinksByDefault: false)
        let handle = try await backend.start(request(maxTokens: 2, enableThinking: true))
        XCTAssertFalse(handle.separatesReasoning)
        _ = try await collect(handle.mailbox)
    }

    func testHandleHonorsDisableThinkingWhenToolsActiveConsistencyWithRender() async throws {
        // The legacy tool workaround forces resolvedEnableThinking=false (tools attached, no explicit
        // flag). The gate must consume that SAME resolved value the codec renders from, so it does NOT
        // separate — proving render/gate consistency, not a desync.
        let backend = makeBackend(
            script: [1, 99], pieces: [1: "hi"], promptTokens: [10],
            thinksByDefault: true,
            disableThinkingWhenToolsActive: true)
        let handle = try await backend.start(request(maxTokens: 2, tools: [weatherTool]))
        XCTAssertFalse(handle.separatesReasoning)
        _ = try await collect(handle.mailbox)
    }

    func testScalarRoutePublishesExactTextUsageAndLength() async throws {
        let backend = makeBackend(
            script: [1, 2, 99],
            pieces: [1: "hel", 2: "lo"],
            promptTokens: [10, 11])
        let handle = try await backend.start(
            request(maxTokens: 2))

        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .scalarGreedy)
        XCTAssertEqual(
            events,
            [
                .text("hel"),
                .text("lo"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(
                            promptTokens: 2,
                            completionTokens: 2))),
            ])
        await waitUntil {
            await backend.snapshot().activeRequests == 0
        }
        let snapshot = await backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)
    }

    func testRequestStopSplitAcrossTokenChunksIsNotPublished() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 4, 99],
            pieces: [
                1: "hello<",
                2: "stop",
                3: ">hidden",
                4: "tail",
            ],
            promptTokens: [10])
        let handle = try await backend.start(
            request(maxTokens: 8, stop: ["<stop>"]))

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
    }

    func testQueueExhaustionQueuedCancellationAndActiveCancellationRecover() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1)

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            let backend = await backend.snapshot()
            return backend.activeRequests == 1
                && mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let queued = try await backend.start(request(maxTokens: 3))
        let queueSnapshot = await backend.snapshot()
        XCTAssertEqual(queueSnapshot.activeRequests, 1)
        XCTAssertEqual(queueSnapshot.queuedRequests, 1)

        do {
            _ = try await backend.start(request(maxTokens: 3))
            XCTFail("Expected queue-full rejection")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .queueFull(retryAfterSeconds: 2))
        }

        let queuedCancellation = await queued.lease.cancel(.clientDisconnected)
        XCTAssertTrue(queuedCancellation)
        await assertMailboxCancelled(queued.mailbox, reason: .clientDisconnected)

        let activeCancellation = await active.lease.cancel(.clientDisconnected)
        XCTAssertTrue(activeCancellation)
        await assertMailboxCancelled(active.mailbox, reason: .clientDisconnected)
        await waitUntil {
            let snapshot = await backend.snapshot()
            return snapshot.activeRequests == 0 && snapshot.queuedRequests == 0
        }

        let recovered = try await backend.start(request(maxTokens: 1))
        let recoveredEvents = try await collect(recovered.mailbox)
        XCTAssertEqual(
            recoveredEvents,
            [
                .text("a"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 1))),
            ])
    }

    func testShutdownCancelsActiveAndQueuedRequestsAndRejectsNewWork() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1)

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }
        let queued = try await backend.start(request(maxTokens: 3))

        await backend.shutdown()

        await assertMailboxCancelled(active.mailbox, reason: .shutdown)
        await assertMailboxCancelled(queued.mailbox, reason: .shutdown)
        let activeLeaseState = await active.lease.state
        let queuedLeaseState = await queued.lease.state
        XCTAssertEqual(activeLeaseState, .cancelled(.shutdown))
        XCTAssertEqual(queuedLeaseState, .cancelled(.shutdown))
        let snapshot = await backend.snapshot()
        XCTAssertEqual(
            snapshot,
            ScalarServingBackendSnapshot(
                activeRequests: 0,
                queuedRequests: 0))

        do {
            _ = try await backend.start(request(maxTokens: 1))
            XCTFail("Expected shutdown rejection")
        } catch let error as ScalarServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }

    func testShutdownMarksFastActiveRequestBeforeDrainingQueuedRequests() async throws {
        let backend = makeBackend(
            script: [1, 2, 99],
            pieces: [1: "a", 2: "b"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 128)

        let active = try await backend.start(request(maxTokens: 2))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }
        var queued: [ServingGenerationHandle] = []
        for _ in 0..<128 {
            queued.append(
                try await backend.start(request(maxTokens: 2)))
        }

        let shutdown = Task {
            await backend.shutdown()
        }
        await waitUntil {
            await queued[0].lease.state == .cancelled(.shutdown)
        }

        var observedCompletion = false
        do {
            while let delta = try await active.mailbox.next() {
                if case .completion = delta {
                    observedCompletion = true
                }
            }
        } catch let error as ServingMailboxError {
            XCTAssertEqual(error, .cancelled(.shutdown))
        }
        await shutdown.value

        XCTAssertFalse(
            observedCompletion,
            "An active request may not complete successfully after shutdown begins")
        let activeLeaseState = await active.lease.state
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(
            activeLeaseState,
            .cancelled(.shutdown))
        XCTAssertEqual(
            finalSnapshot,
            ScalarServingBackendSnapshot(
                activeRequests: 0,
                queuedRequests: 0))
    }

    func testActiveShutdownCancellationStopsAdmissionAndNeverLaunchesQueuedWork() async throws {
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1)

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }
        let queued = try await backend.start(request(maxTokens: 3))

        let cancelled = await active.lease.cancel(.shutdown)

        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(active.mailbox, reason: .shutdown)
        await assertMailboxCancelled(queued.mailbox, reason: .shutdown)
        let queuedLeaseState = await queued.lease.state
        XCTAssertEqual(queuedLeaseState, .cancelled(.shutdown))
        let snapshot = await backend.snapshot()
        XCTAssertEqual(
            snapshot,
            ScalarServingBackendSnapshot(
                activeRequests: 0,
                queuedRequests: 0))
        do {
            _ = try await backend.start(request(maxTokens: 1))
            XCTFail("Expected shutdown rejection")
        } catch let error as ScalarServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }

    func testFullQueueRejectsBeforeRenderingPrompt() async throws {
        let renderCounter = RenderCounter()
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1,
            renderCounter: renderCounter)

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }
        _ = try await backend.start(request(maxTokens: 3))

        do {
            _ = try await backend.start(request(maxTokens: 3))
            XCTFail("Expected queue-full rejection")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .queueFull(retryAfterSeconds: 2))
        }

        XCTAssertEqual(renderCounter.value, 2)
        await backend.shutdown()
    }

    func testModelAwareBudgetUsesExactRenderedPromptAndIsCarriedByHandle() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 6,
            requestedDefaultCompletionTokens: 4,
            maximumNonStreamingCompletionTokens: 4,
            completionLimitPolicy: .clamp)
        let backend = makeBackend(
            script: [1, 2, 3, 4, 99],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d"],
            promptTokens: [10, 11],
            modelCapabilities: capabilities)

        let handle = try await backend.start(request(maxTokens: 8))
        let resolution = try XCTUnwrap(handle.completionBudgetResolution)

        XCTAssertEqual(resolution.requestedCompletionTokens, 8)
        XCTAssertEqual(resolution.renderedPromptTokens, 2)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 4)
        XCTAssertEqual(resolution.appliedCompletionTokens, 4)
        XCTAssertTrue(resolution.wasClamped)
        let events = try await collect(handle.mailbox)
        XCTAssertEqual(
            events.last,
            .completion(
                ServingGenerationCompletion(
                    finishReason: .length,
                    usage: OpenAIChatUsage(promptTokens: 2, completionTokens: 4))))
    }

    func testModelAwareInvalidBudgetPrecedesQueueFullAndDoesNotEnqueue() async throws {
        let renderCounter = RenderCounter()
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 4,
            requestedDefaultCompletionTokens: 2,
            maximumNonStreamingCompletionTokens: 3,
            completionLimitPolicy: .reject)
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1,
            renderCounter: renderCounter,
            modelCapabilities: capabilities)

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1 && mailbox.waitingProducers == 1
        }
        _ = try await backend.start(request(maxTokens: 3))

        do {
            _ = try await backend.start(request(maxTokens: 4))
            XCTFail("Expected model-aware budget rejection")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "completion_limit_exceeded")
        }

        XCTAssertEqual(renderCounter.value, 3)
        let snapshot = await backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        await backend.shutdown()
    }
}

private func makeBackend(
    script: [Int],
    pieces: [Int: String],
    promptTokens: [Int],
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 4,
        maxBytes: 1_024),
    maximumQueuedRequests: Int = 2,
    renderCounter: RenderCounter? = nil,
    toolCallFormat: ToolCallFormat = .json,
    thinksByDefault: Bool = false,
    disableThinkingWhenToolsActive: Bool = false,
    modelCapabilities: ServingModelCapabilities? = nil
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(
            decoder: ScriptedDecoder(script: script, eos: 99)),
        codec: FixtureScalarTextCodec(
            promptTokens: promptTokens,
            pieces: pieces,
            renderCounter: renderCounter),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: maximumQueuedRequests,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity,
            toolCallFormat: toolCallFormat,
            disableThinkingWhenToolsActive: disableThinkingWhenToolsActive,
            thinksByDefault: thinksByDefault,
            modelCapabilities: modelCapabilities))
}

private let weatherTool = OpenAIToolSpec(
    name: "get_weather",
    description: "Look up the weather for a city",
    parameters: .object([
        "type": .string("object"),
        "properties": .object([
            "city": .object(["type": .string("string")])
        ]),
    ]),
    raw: .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("get_weather"),
            "description": .string("Look up the weather for a city"),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "city": .object(["type": .string("string")])
                ]),
            ]),
        ]),
    ]))

private func request(
    maxTokens: Int,
    stop: [String] = [],
    tools: [OpenAIToolSpec] = [],
    enableThinking: Bool? = nil
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [
            OpenAIChatMessage(role: .user, text: "private prompt")
        ],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop,
        tools: tools,
        toolChoice: tools.isEmpty ? .none : .auto,
        enableThinking: enableThinking)
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
        XCTAssertEqual(error, .cancelled(reason), file: file, line: line)
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
    let pieces: [Int: String]
    let renderCounter: RenderCounter?

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        renderCounter?.increment()
        return promptTokens
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureScalarDetokenizer(pieces: pieces)
    }
}

private final class RenderCounter: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: 0)

    var value: Int {
        state.withLock { $0 }
    }

    func increment() {
        state.withLock { $0 += 1 }
    }
}

private struct FixtureScalarDetokenizer: ScalarServingDetokenizer {
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
