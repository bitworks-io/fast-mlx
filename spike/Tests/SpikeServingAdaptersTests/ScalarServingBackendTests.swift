import XCTest
import os

import MLX
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

    // MARK: - Legacy /v1/completions (promptInput: .rawText)

    // A `.rawText` request must go through `codec.encode(rawText:)`, never `codec.render(...)` — the
    // chat-template path must not be exercised for a request that carries no messages to template.
    func testRawTextPromptCallsEncodeRawTextAndSkipsRender() async throws {
        let renderCounter = RenderCounter()
        let encodeRawTextCounter = RenderCounter()
        let codec = RawTextFixtureScalarTextCodec(
            chatPromptTokens: [10],
            rawTextTokens: [20, 21],
            pieces: [1: "hi"],
            renderCounter: renderCounter,
            encodeRawTextCounter: encodeRawTextCounter)
        let backend = makeBackendWithCodec(script: [1, 99], codec: codec)

        let handle = try await backend.start(rawTextRequest(prompt: "Once upon a time", maxTokens: 2))
        _ = try await collect(handle.mailbox)

        XCTAssertEqual(renderCounter.value, 0)
        XCTAssertEqual(encodeRawTextCounter.value, 1)
    }

    // Negative control: an ordinary `.chat` request on the SAME codec must still go through
    // `render(...)` and must NOT call `encode(rawText:)` — proving the branch is real, not incidental.
    func testChatPromptStillCallsRenderAndSkipsEncodeRawText() async throws {
        let renderCounter = RenderCounter()
        let encodeRawTextCounter = RenderCounter()
        let codec = RawTextFixtureScalarTextCodec(
            chatPromptTokens: [10],
            rawTextTokens: [20, 21],
            pieces: [1: "hi"],
            renderCounter: renderCounter,
            encodeRawTextCounter: encodeRawTextCounter)
        let backend = makeBackendWithCodec(script: [1, 99], codec: codec)

        let handle = try await backend.start(request(maxTokens: 2))
        _ = try await collect(handle.mailbox)

        XCTAssertEqual(renderCounter.value, 1)
        XCTAssertEqual(encodeRawTextCounter.value, 0)
    }

    // `ScalarServingTextCodec`'s default `encode(rawText:)` (the protocol extension) fails closed
    // with `completions_unsupported` for any codec that has not opted in — `FixtureScalarTextCodec`
    // below deliberately does NOT override it.
    func testDefaultEncodeRawTextThrowsCompletionsUnsupported() async throws {
        let backend = makeBackend(script: [1, 99], pieces: [1: "hi"], promptTokens: [10])

        do {
            _ = try await backend.start(rawTextRequest(prompt: "hi", maxTokens: 2))
            XCTFail("Expected completions_unsupported rejection")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "completions_unsupported")
            XCTAssertEqual(error.openAIError.param, "prompt")
        }
    }

    // `makeBackend` binds a `ScriptedDecoder`, which does not conform to `SpikeCore.
    // LogprobDecoding` — this is the "decoder capability" refusal path (mirrors the
    // compiled-fp16/MTP-speculative production routes, neither of which conforms either), distinct
    // from `testScalarRouteEmitsRealTokenLogprobsWhenDecoderSupportsThem` below, which proves the
    // ADMITTED path against a decoder that DOES conform.
    func testLogprobsRequestThrowsLogprobsUnsupported() async throws {
        let backend = makeBackend(script: [1, 99], pieces: [1: "hi"], promptTokens: [10])
        var logprobsRequest = request(maxTokens: 2)
        logprobsRequest.logprobsRequest = .chat(topLogprobs: 0)

        do {
            _ = try await backend.start(logprobsRequest)
            XCTFail("Expected logprobs_unsupported rejection")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "logprobs_unsupported")
            XCTAssertEqual(error.openAIError.param, "logprobs")
        }
    }

    // The scalar backend now computes REAL per-token logprobs when its bound decoder conforms to
    // `SpikeCore.LogprobDecoding` (in production: a plain `MLXDecoder`, never `CompiledMLXDecoder`
    // or `MTPSpeculativeDecoder` — see `SpikeServingAdapters.swift`'s admission-guard comment).
    // `FixtureLogprobDecoder` stands in for that real decoder here so this stays MLX-free: it
    // proves the WIRING (mapping `DecodedTokenLogprob` -> `ServingTokenLogprob`, resolving token
    // text via `codec.decodeSingleToken`, and emitting `.tokenLogprobs` in generation order) without
    // needing a real model — `MLXDecoderLogprobComputationTests` (`SpikeCoreTests`) already covers
    // the underlying `logSoftmax`/top-N computation itself.
    func testScalarRouteEmitsRealTokenLogprobsWhenDecoderSupportsThem() async throws {
        let decoder = FixtureLogprobDecoder(
            script: [1, 2, 99], eos: 99, candidateTokenIDs: [50, 60])
        let codec = LogprobFixtureScalarTextCodec(
            promptTokens: [10], pieces: [1: "a", 2: "b"])
        let backend = makeLogprobBackend(decoder: decoder, codec: codec)

        var logprobsRequest = request(maxTokens: 4)
        logprobsRequest.logprobsRequest = .chat(topLogprobs: 1)

        let handle = try await backend.start(logprobsRequest)
        let events = try await collect(handle.mailbox)

        let logprobBatches = events.compactMap { event -> [ServingTokenLogprob]? in
            if case .tokenLogprobs(let tokens) = event { return tokens }
            return nil
        }
        let tokens = logprobBatches.flatMap { $0 }

        XCTAssertEqual(tokens.map(\.tokenText), ["tok1", "tok2"], "events: \(events)")
        XCTAssertEqual(tokens[0].topCandidates.map(\.tokenText), ["tok50"])
        XCTAssertEqual(tokens[1].topCandidates.map(\.tokenText), ["tok60"])

        // `.tokenLogprobs` for a token must appear no later than that same token's own `.text`
        // delta (the streaming carry-once ordering `onLogprob`'s doc comment documents).
        let textIndex = events.firstIndex { if case .text = $0 { return true }; return false }
        let firstLogprobIndex = events.firstIndex { if case .tokenLogprobs = $0 { return true }; return false }
        XCTAssertNotNil(textIndex)
        XCTAssertNotNil(firstLogprobIndex)
        XCTAssertLessThan(firstLogprobIndex!, textIndex!)

        let finish = events.compactMap { event -> OpenAIChatFinishReason? in
            if case .completion(let completion) = event { return completion.finishReason }
            return nil
        }.first
        XCTAssertEqual(finish, .stop)
    }

    // Negative control for the ordering assertion above: `logprobsRequest == nil` on the SAME
    // logprob-capable decoder/codec must never emit `.tokenLogprobs` — proves the feature is
    // opt-in, not "on whenever the decoder happens to support it".
    func testScalarRouteOmitsTokenLogprobsWhenNotRequestedEvenOnASupportingDecoder() async throws {
        let decoder = FixtureLogprobDecoder(
            script: [1, 2, 99], eos: 99, candidateTokenIDs: [50, 60])
        let codec = LogprobFixtureScalarTextCodec(
            promptTokens: [10], pieces: [1: "a", 2: "b"])
        let backend = makeLogprobBackend(decoder: decoder, codec: codec)

        let handle = try await backend.start(request(maxTokens: 4))
        let events = try await collect(handle.mailbox)

        let hasLogprobs = events.contains { if case .tokenLogprobs = $0 { return true }; return false }
        XCTAssertFalse(hasLogprobs, "events: \(events)")
    }

    // Raw-text completions never separate reasoning, even for a family that thinks by default —
    // there is no chat template pre-filling a `<think>` block for a raw-text prompt to begin inside.
    func testHandleDoesNotSeparateReasoningForRawTextEvenWhenFamilyThinksByDefault() async throws {
        let codec = RawTextFixtureScalarTextCodec(
            chatPromptTokens: [10],
            rawTextTokens: [20],
            pieces: [1: "hi"],
            renderCounter: RenderCounter(),
            encodeRawTextCounter: RenderCounter())
        let backend = makeBackendWithCodec(script: [1, 99], codec: codec, thinksByDefault: true)

        let handle = try await backend.start(rawTextRequest(prompt: "hi", maxTokens: 2))
        XCTAssertFalse(handle.separatesReasoning)
        _ = try await collect(handle.mailbox)
    }

    // The empty-rendered-prompt screen applies to raw text exactly as it does to chat.
    func testRawTextEmptyRenderedPromptIsRejected() async throws {
        let codec = RawTextFixtureScalarTextCodec(
            chatPromptTokens: [10],
            rawTextTokens: [],
            pieces: [:],
            renderCounter: RenderCounter(),
            encodeRawTextCounter: RenderCounter())
        let backend = makeBackendWithCodec(script: [99], codec: codec)

        do {
            _ = try await backend.start(rawTextRequest(prompt: "", maxTokens: 2))
            XCTFail("Expected emptyRenderedPrompt rejection")
        } catch let error as ScalarServingBackendError {
            XCTAssertEqual(error, .emptyRenderedPrompt)
        }
    }

    // The rejected-prompt-token screen applies to raw text exactly as it does to chat.
    func testRawTextRejectedPromptTokenIsCaught() async throws {
        let codec = RawTextFixtureScalarTextCodec(
            chatPromptTokens: [10],
            rawTextTokens: [7],
            pieces: [:],
            renderCounter: RenderCounter(),
            encodeRawTextCounter: RenderCounter())
        let backend = makeBackendWithCodec(
            script: [99], codec: codec, rejectedPromptTokenIDs: [7])

        do {
            _ = try await backend.start(rawTextRequest(prompt: "hi", maxTokens: 2))
            XCTFail("Expected unsupported_prompt_token rejection")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "unsupported_prompt_token")
        }
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

    // Work item A: `ScalarServingBackendSnapshot` must report the live MLX allocator bytes,
    // wired from `Memory.snapshot()` (mirroring `ContinuousServingBackendSnapshot.current`), not a
    // hardcoded value. Force a real allocation first so `mlxActiveBytes`/`mlxPeakBytes` are
    // provably nonzero — a stub hardcoded to 0 would fail the `XCTAssertGreaterThan` below.
    func testSnapshotReportsAllocatorBytesWiredFromMemorySnapshot() async throws {
        let backend = makeBackend(
            script: [1, 99], pieces: [1: "hi"], promptTokens: [10])

        let allocation = MLXArray.zeros([1_024, 1_024], type: Float32.self)
        allocation.eval()
        withExtendedLifetime(allocation) {}

        let snapshot = await backend.snapshot()

        XCTAssertGreaterThan(snapshot.mlxActiveBytes, 0, "expected a real allocation to be visible")
        XCTAssertGreaterThanOrEqual(snapshot.mlxCacheBytes, 0)
        // Peak active memory is a process-wide high-water mark of active memory alone (never a
        // sum with cache — see MLX's Memory.Snapshot doc), so it must be >= the current active
        // figure at this point in the same process.
        XCTAssertGreaterThanOrEqual(snapshot.mlxPeakBytes, snapshot.mlxActiveBytes)
    }

    // The exact-MTP route falls back to `ScalarServingBackendSnapshot.processAllocatorSample()`
    // when the concrete backend it is serving does not downcast to `ScalarServingBackend` (e.g. an
    // exact-success `ExactQwen35MTPServingBackend`). `Memory.snapshot()` is process-global, so this
    // fallback must report the REAL resident allocator bytes, not a fabricated zero — a zero there
    // would also poison the fit-check's measured-vs-modeled drift comparison. Force a real
    // allocation first, mirroring `testSnapshotReportsAllocatorBytesWiredFromMemorySnapshot`.
    func testProcessAllocatorSampleReportsRealBytesWithNoRequestCounts() {
        let allocation = MLXArray.zeros([1_024, 1_024], type: Float32.self)
        allocation.eval()
        withExtendedLifetime(allocation) {}

        let snapshot = ScalarServingBackendSnapshot.processAllocatorSample()

        XCTAssertGreaterThan(snapshot.mlxActiveBytes, 0, "expected a real allocation to be visible")
        XCTAssertGreaterThanOrEqual(snapshot.mlxPeakBytes, snapshot.mlxActiveBytes)
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
        // Assert only the two fields this test is actually about — `mlxCacheBytes`/`mlxPeakBytes`
        // reflect the whole process's live MLX allocator state (shared across every test in this
        // executable, see `testSnapshotReportsAllocatorBytesWiredFromMemorySnapshot`), so a
        // full-struct equality against a hardcoded-zero allocator snapshot would be a false
        // failure once those fields are wired to `Memory.snapshot()` instead of unmeasured.
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)

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
        // See the comment in `testShutdownCancelsActiveAndQueuedRequestsAndRejectsNewWork` for why
        // this asserts scalars rather than full-struct equality against a hardcoded-zero snapshot.
        XCTAssertEqual(finalSnapshot.activeRequests, 0)
        XCTAssertEqual(finalSnapshot.queuedRequests, 0)
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
        // See the comment in `testShutdownCancelsActiveAndQueuedRequestsAndRejectsNewWork` for why
        // this asserts scalars rather than full-struct equality against a hardcoded-zero snapshot.
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.queuedRequests, 0)
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

    // A model family's OWN forward-path input validation reaches specific token IDs via a
    // `preconditionFailure` (a process abort). Generated tokens are already masked away from these
    // IDs, but prompt tokens are not, so an admitted request that merely tokenizes to one of them
    // would be an HTTP-reachable way to kill the server. Configuring `rejectedPromptTokenIDs` must
    // reject such a request at admission with a machine-readable code instead.
    // Model capabilities are supplied so this exercises the EARLY screen inside the
    // model-capabilities branch (before budget resolution), independent of queue state.
    func testRejectsPromptContainingRejectedTokenID() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 6,
            requestedDefaultCompletionTokens: 4,
            maximumNonStreamingCompletionTokens: 4,
            completionLimitPolicy: .clamp)
        let backend = makeBackend(
            script: [1, 99],
            pieces: [1: "hi"],
            promptTokens: [10, 7, 11],
            modelCapabilities: capabilities,
            rejectedPromptTokenIDs: [7])

        do {
            _ = try await backend.start(request(maxTokens: 4))
            XCTFail("Expected rejection for a prompt containing a rejected token ID")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "unsupported_prompt_token")
            XCTAssertEqual(error.openAIError.type, .invalidRequest)
        }
    }

    // The default `rejectedPromptTokenIDs` is empty — proving today's behavior (before this field
    // existed) is completely unchanged for every model family that reports nothing. This is the
    // critical regression guard: it must remain green even though the SAME token ID (7) appears in
    // the rendered prompt as in the rejection test above.
    func testDefaultRejectedPromptTokenIDsIsInertAndAdmitsNormally() async throws {
        let backend = makeBackend(
            script: [1, 99],
            pieces: [1: "hi"],
            promptTokens: [10, 7, 11])

        let handle = try await backend.start(request(maxTokens: 4))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("hi"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1))),
            ])
    }

    // Only the token IDs actually configured are rejected; a prompt made up of other IDs is admitted
    // normally even though the configuration is non-empty.
    func testOnlyConfiguredRejectedTokenIDsAreRejected() async throws {
        let backend = makeBackend(
            script: [1, 99],
            pieces: [1: "hi"],
            promptTokens: [10, 11, 12],
            rejectedPromptTokenIDs: [7])

        let handle = try await backend.start(request(maxTokens: 4))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("hi"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1))),
            ])
    }

    // Mirrors `testModelAwareInvalidBudgetPrecedesQueueFullAndDoesNotEnqueue`: a permanently-invalid
    // request (a rejected prompt token) must win over a retryable queue-full rejection. Otherwise a
    // client would retry-after a request that can never succeed. Model capabilities are supplied so
    // this exercises the EARLY screen inside the capabilities branch, before the queue-full check.
    func testRejectedPromptTokenPrecedesQueueFullAndDoesNotEnqueue() async throws {
        let renderCounter = RenderCounter()
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 8,
            effectiveMaxContextTokens: 8,
            requestedDefaultCompletionTokens: 2,
            maximumNonStreamingCompletionTokens: 8,
            completionLimitPolicy: .reject)
        // The fixture codec renders a fixed prompt regardless of message text, so the active and
        // queued requests below (no tools) render the plain, clean prompt; only the third request
        // (with a tool attached) renders the extra rejected token — proving the rejection is a
        // property of THAT request, not a permanent property of the backend, and that it wins over
        // the queue-full check for that one request without disturbing the two already admitted.
        let backend = makeBackend(
            script: [1, 2, 3, 99],
            pieces: [1: "a", 2: "b", 3: "c"],
            promptTokens: [10],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8),
            maximumQueuedRequests: 1,
            renderCounter: renderCounter,
            modelCapabilities: capabilities,
            rejectedPromptTokenIDs: [7],
            extraPromptTokensWhenToolsPresent: [7])

        let active = try await backend.start(request(maxTokens: 3))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1 && mailbox.waitingProducers == 1
        }
        _ = try await backend.start(request(maxTokens: 3))

        do {
            _ = try await backend.start(request(maxTokens: 3, tools: [weatherTool]))
            XCTFail("Expected unsupported_prompt_token rejection, not queueFull")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "unsupported_prompt_token")
        }

        let snapshot = await backend.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 1)
        XCTAssertEqual(snapshot.queuedRequests, 1)
        await backend.shutdown()
    }

    // The fallback branch (no `modelCapabilities`) defers rendering until AFTER the queue-full check,
    // so it can only be covered by the second (post-render) screen. This proves that site independently
    // catches the rejection when `modelCapabilities` is nil.
    func testRejectedPromptTokenIsCaughtInFallbackBranchWithoutModelCapabilities() async throws {
        let backend = makeBackend(
            script: [1, 99],
            pieces: [1: "hi"],
            promptTokens: [10, 7, 11],
            modelCapabilities: nil,
            rejectedPromptTokenIDs: [7])

        do {
            _ = try await backend.start(request(maxTokens: 4))
            XCTFail("Expected rejection for a prompt containing a rejected token ID")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.code, "unsupported_prompt_token")
        }
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
    modelCapabilities: ServingModelCapabilities? = nil,
    rejectedPromptTokenIDs: Set<Int> = [],
    extraPromptTokensWhenToolsPresent: [Int] = []
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(
            decoder: ScriptedDecoder(script: script, eos: 99)),
        codec: FixtureScalarTextCodec(
            promptTokens: promptTokens,
            pieces: pieces,
            renderCounter: renderCounter,
            extraPromptTokensWhenToolsPresent: extraPromptTokensWhenToolsPresent),
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
            modelCapabilities: modelCapabilities,
            rejectedPromptTokenIDs: rejectedPromptTokenIDs))
}

/// Minimal harness for the legacy-completions tests: only requires `script` + `codec`, since the
/// `.rawText` tests need a codec that overrides `encode(rawText:)` (unlike `FixtureScalarTextCodec`,
/// which deliberately does not, so it exercises the protocol-default `completions_unsupported` path).
private func makeBackendWithCodec(
    script: [Int],
    codec: any ScalarServingTextCodec,
    thinksByDefault: Bool = false,
    rejectedPromptTokenIDs: Set<Int> = []
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(
            decoder: ScriptedDecoder(script: script, eos: 99)),
        codec: codec,
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            thinksByDefault: thinksByDefault,
            rejectedPromptTokenIDs: rejectedPromptTokenIDs))
}

/// Builds a backend around a CALLER-SUPPLIED decoder/codec pair (unlike `makeBackend`/
/// `makeBackendWithCodec`, which always bind `ScriptedDecoder`) — used by the logprobs tests to
/// bind a `SpikeCore.LogprobDecoding`-conforming decoder so `ScalarServingBackend` actually admits
/// and serves a logprobs request instead of refusing at `InferenceActor.supportsLogprobs()`.
private func makeLogprobBackend(
    decoder: sending any Decoder,
    codec: any ScalarServingTextCodec
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: decoder),
        codec: codec,
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096)))
}

private func rawTextRequest(
    prompt: String,
    maxTokens: Int
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: [],
        promptInput: .rawText(prompt))
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

/// Test double conforming to `SpikeCore.LogprobDecoding`: replays a fixed script like
/// `ScriptedDecoder` (never a real MLX computation), and reports a synthetic `DecodedTokenLogprob`
/// per generated token — one candidate per token, sourced from `candidateTokenIDs` by generation
/// order — so `ScalarServingBackendTests`'s logprobs tests can assert the ADAPTER'S wiring
/// (`DecodedTokenLogprob` -> `ServingTokenLogprob`, token-text resolution, delta ordering) without
/// needing a real model; the underlying `logSoftmax`/top-N computation itself is covered by
/// `MLXDecoderLogprobComputationTests` (`SpikeCoreTests`).
private struct FixtureLogprobDecoder: LogprobDecoding {
    let script: [Int]
    let eos: Int
    let candidateTokenIDs: [Int]
    var i = 0

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func step(last: Int) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func reset() { i = 0 }

    mutating func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let index = i
        let token = prefill(promptTokens)
        return (token, makeLogprob(token: token, generationIndex: index, topN: topN))
    }

    mutating func stepWithLogprob(
        last: Int, topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let index = i
        let token = step(last: last)
        return (token, makeLogprob(token: token, generationIndex: index, topN: topN))
    }

    private func makeLogprob(
        token: Int, generationIndex: Int, topN: Int
    ) -> DecodedTokenLogprob {
        var top: [DecodedTokenLogprobCandidate] = []
        if topN > 0, generationIndex < candidateTokenIDs.count {
            top = [
                DecodedTokenLogprobCandidate(
                    tokenID: candidateTokenIDs[generationIndex], logprob: -1.0)
            ]
        }
        return DecodedTokenLogprob(
            tokenID: token, logprob: Float(-(generationIndex + 1)), top: top)
    }
}

/// Codec fixture for the logprobs tests: fixed prompt/pieces like `FixtureScalarTextCodec`, but ALSO
/// overrides `decodeSingleToken` (unlike `FixtureScalarTextCodec`, which relies on the protocol's
/// empty-string default) so a test can assert on real per-token text.
private struct LogprobFixtureScalarTextCodec: ScalarServingTextCodec {
    let promptTokens: [Int]
    let pieces: [Int: String]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        promptTokens
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureScalarDetokenizer(pieces: pieces)
    }

    func decodeSingleToken(_ tokenID: Int) -> String {
        "tok\(tokenID)"
    }
}

private struct FixtureScalarTextCodec: ScalarServingTextCodec {
    let promptTokens: [Int]
    let pieces: [Int: String]
    let renderCounter: RenderCounter?
    // Lets a single backend instance render two distinct prompts across a sequence of `start(...)`
    // calls (e.g. to prove a per-request rejection without disturbing earlier admitted requests that
    // share the same backend/codec) by appending extra tokens only when the request carries tools.
    var extraPromptTokensWhenToolsPresent: [Int] = []

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        renderCounter?.increment()
        return tools.isEmpty ? promptTokens : promptTokens + extraPromptTokensWhenToolsPresent
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureScalarDetokenizer(pieces: pieces)
    }
}

/// A codec that DOES override `encode(rawText:)` (unlike `FixtureScalarTextCodec` above, which
/// exercises the protocol-default fail-closed path) — lets tests independently observe which of
/// `render(...)` / `encode(rawText:)` a given request shape actually invoked.
private struct RawTextFixtureScalarTextCodec: ScalarServingTextCodec {
    let chatPromptTokens: [Int]
    let rawTextTokens: [Int]
    let pieces: [Int: String]
    let renderCounter: RenderCounter
    let encodeRawTextCounter: RenderCounter

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        renderCounter.increment()
        return chatPromptTokens
    }

    func encode(rawText: String) throws -> [Int] {
        encodeRawTextCounter.increment()
        return rawTextTokens
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
