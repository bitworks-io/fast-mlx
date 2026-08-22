import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import os
import XCTest

@testable import ServingCore
@testable import ServingNIO

final class OpenAIChatCompletionsHTTPHandlerTests: XCTestCase {
    func testSuccessfulRequestRecordsExactlyOnePromptFreeOnWireEvidence() async throws {
        let promptSentinel = "PROMPT-SENTINEL-handler"
        let apiKeySentinel = "sk-API-KEY-SENTINEL-handler"
        let generatedSentinel = "GENERATED-SENTINEL-handler"
        let backend = ScriptedBackend(scripts: [
            .completed(
                text: [generatedSentinel],
                promptTokens: 1,
                completionTokens: 1)
        ])
        let recorder = ServingEvidenceRecorder()
        let snapshots = ServingSnapshotSequence()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: { try await snapshots.next() },
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { message in
                    Task { await recorder.recordFailure(message) }
                }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"\(promptSentinel)"}],"max_completion_tokens":8,"temperature":0,"stream":false}
        """

        try await writeRequest(
            channel,
            body: body,
            authorization: "Bearer \(apiKeySentinel)")
        let response = try await collectResponse(from: channel)
        await waitUntil { await recorder.evidence.count == 1 }

        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        let canonical = try evidence.canonicalJSONData()
        let json = try XCTUnwrap(String(data: canonical, encoding: .utf8))
        XCTAssertEqual(evidence.response.status, 200)
        XCTAssertTrue(evidence.response.completed)
        XCTAssertEqual(evidence.response.chunkCount, 1)
        XCTAssertEqual(evidence.response.bodyBytes, response.body.utf8.count)
        XCTAssertEqual(
            evidence.response.bodySHA256,
            ServingEvidence.SHA256.hexDigest(of: Data(response.body.utf8)))
        XCTAssertEqual(evidence.route?.kind, .continuousBatchNoSpec)
        XCTAssertEqual(evidence.cancellation?.cancelled, false)
        XCTAssertEqual(evidence.resources?.admission, .accepted)
        XCTAssertEqual(evidence.resources?.before?.activeRequests, 0)
        XCTAssertEqual(evidence.resources?.active?.activeRequests, 1)
        XCTAssertEqual(evidence.resources?.terminal?.activeRequests, 0)
        XCTAssertFalse(json.contains(promptSentinel))
        XCTAssertFalse(json.contains(apiKeySentinel))
        XCTAssertFalse(json.contains(generatedSentinel))
        XCTAssertTrue(recorded.failures.isEmpty)

        _ = try await channel.finish()
    }

    func testAdmissionFailureRecordsCompletedTypedEvidenceWithoutRoute() async throws {
        let backend = ScriptedBackend(scripts: [
            .admissionRejected(.queueFull(retryAfterSeconds: 2))
        ])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: nil,
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { _ in }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        _ = try await collectResponse(from: channel)
        await waitUntil { await recorder.evidence.count == 1 }

        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertEqual(evidence.response.status, 429)
        XCTAssertTrue(evidence.response.completed)
        XCTAssertNil(evidence.route)
        XCTAssertEqual(evidence.resources?.admission, .queueFull)
        XCTAssertEqual(evidence.cancellation?.cancelled, false)

        _ = try await channel.finish()
    }

    func testAdmittedStreamingHeadPrecedesFirstDeltaAndDisconnectReleasesResources()
        async throws
    {
        let backend = ScriptedBackend(scripts: [.held])
        let recorder = ServingEvidenceRecorder()
        let snapshots = ServingSnapshotSequence()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: { try await snapshots.next() },
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { _ in }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().startCount == 1 }
        var responseHead: HTTPResponseHead?
        for _ in 0..<10_000 {
            if let part = try await channel.readOutbound(
                as: HTTPServerResponsePart.self)
            {
                if case .head(let head) = part {
                    responseHead = head
                    break
                }
                XCTFail("Admitted streaming response must begin with an HTTP head")
                break
            }
            await Task.yield()
        }
        let admittedHead = try XCTUnwrap(responseHead)
        XCTAssertEqual(admittedHead.status, .ok)
        XCTAssertEqual(
            admittedHead.headers.first(name: "content-type"),
            "text/event-stream")
        let bodyBeforeFirstDelta = try await channel.readOutbound(
            as: HTTPServerResponsePart.self)
        XCTAssertNil(bodyBeforeFirstDelta)

        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        }
        await waitUntil { await recorder.evidence.count == 1 }

        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertEqual(evidence.response.status, 200)
        XCTAssertFalse(evidence.response.completed)
        XCTAssertEqual(evidence.response.bodyBytes, 0)
        XCTAssertEqual(evidence.response.chunkCount, 0)
        XCTAssertEqual(
            evidence.response.bodySHA256,
            ServingEvidence.SHA256.hexDigest(of: Data()))
        XCTAssertEqual(evidence.route?.kind, .continuousBatchNoSpec)
        XCTAssertEqual(evidence.resources?.admission, .accepted)
        XCTAssertEqual(evidence.resources?.active?.activeRequests, 1)
        XCTAssertEqual(evidence.cancellation?.reason, .clientDisconnected)
        XCTAssertEqual(evidence.resources?.terminal?.activeRequests, 0)
        XCTAssertEqual(backend.snapshot().cancelCount, 1)

        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testQuiesceRecordsShutdownBeforeClosingConnection() async throws {
        let backend = ScriptedBackend(scripts: [.held])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: nil,
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { _ in }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().startCount == 1 }
        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireUserInboundEventTriggered(
                ChannelShouldQuiesceEvent())
        }
        await waitUntil { await recorder.evidence.count == 1 }

        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertFalse(evidence.response.completed)
        XCTAssertEqual(evidence.cancellation?.reason, .shutdown)
        XCTAssertEqual(backend.snapshot().cancelCount, 1)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testTerminalSnapshotWaitsForShutdownResourceRelease() async throws {
        let cancellationGate = DelayedCancellationGate()
        let backend = ScriptedBackend(scripts: [
            .heldWithDelayedCancel(cancellationGate)
        ])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: {
                    let active = backend.snapshot().cancelCount == 0 ? 1 : 0
                    return try ServingEvidence.ResourceSnapshot(
                        activeRequests: active,
                        coordinatorSlots: active,
                        reservedKVBytes: active * 4_096,
                        maxReservedKVBytes: 16_384,
                        mlxActiveBytes: 8_192,
                        mlxCacheBytes: 1_024,
                        mlxPeakBytes: 16_384)
                },
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { _ in }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().startCount == 1 }
        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireUserInboundEventTriggered(
                ChannelShouldQuiesceEvent())
        }
        await waitUntil { await cancellationGate.isWaiting }
        try await Task.sleep(for: .milliseconds(20))
        let evidenceBeforeRelease = await recorder.evidence
        XCTAssertTrue(evidenceBeforeRelease.isEmpty)

        await cancellationGate.release()
        await waitUntil { await recorder.evidence.count == 1 }
        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertEqual(evidence.cancellation?.reason, .shutdown)
        XCTAssertEqual(evidence.resources?.terminal?.activeRequests, 0)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testRecorderFailureIsReportedAndConnectionFailsClosed() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["done"], promptTokens: 1, completionTokens: 1)
        ])
        let reporter = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: nil,
                record: { _ in
                    throw ServingEvidenceRecorder.RecorderError.rejected
                },
                reportFailure: { message in
                    Task { await reporter.recordFailure(message) }
                }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)
        await waitUntil { await reporter.failures.count == 1 }
        await waitUntil { !channel.isActive }

        XCTAssertEqual(response.head.status, .ok)
        let reported = await reporter.snapshot()
        XCTAssertEqual(
            reported.failures,
            ["serving evidence terminal persistence failed"])
        XCTAssertFalse(configuration.evidence?.tracker.begin() ?? true)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testThrowingResourceSnapshotsRemainTypedInTerminalEvidence() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["done"], promptTokens: 1, completionTokens: 1)
        ])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: {
                    throw ServingSnapshotError.unavailable
                },
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { message in
                    Task { await recorder.recordFailure(message) }
                }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)
        await waitUntil { await recorder.evidence.count == 1 }

        XCTAssertEqual(response.head.status, .ok)
        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertEqual(
            evidence.resources?.failedSnapshots,
            [.before, .active, .terminal])
        XCTAssertEqual(recorded.failures.count, 3)
        XCTAssertTrue(evidence.response.completed)

        _ = try await channel.finish()
    }

    func testAdmissionFailureWriteFailureRecordsClientDisconnect() async throws {
        let backend = ScriptedBackend(scripts: [
            .admissionRejected(.queueFull(retryAfterSeconds: 1))
        ])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: nil,
                record: { evidence in try await recorder.record(evidence) },
                reportFailure: { _ in }))
        let channel = try await NIOAsyncTestingChannel { channel in
            try channel.pipeline.syncOperations.addHandlers(
                FailingOutboundHandler(),
                OpenAIChatCompletionsHTTPHandler(
                    configuration: configuration,
                    backend: backend))
        }

        try await writeRequest(channel, body: requestBody(stream: false))
        await waitUntil { await recorder.evidence.count == 1 }

        let recorded = await recorder.snapshot()
        let evidence = try XCTUnwrap(recorded.evidence.first)
        XCTAssertFalse(evidence.response.completed)
        XCTAssertEqual(evidence.cancellation?.reason, .clientDisconnected)
        XCTAssertEqual(evidence.resources?.admission, .queueFull)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testNonStreamingRequestReturnsOpenAIJSONAndAllowsSequentialKeepAlive() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2),
            .completed(text: ["again"], promptTokens: 4, completionTokens: 1),
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: false))
        let first = try await collectResponse(from: channel)
        XCTAssertEqual(first.head.status, .ok)
        XCTAssertEqual(first.head.headers.first(name: "content-type"), "application/json")
        let firstObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(first.body.utf8)) as? [String: Any])
        let firstChoice = try XCTUnwrap((firstObject["choices"] as? [[String: Any]])?.first)
        let firstMessage = try XCTUnwrap(firstChoice["message"] as? [String: Any])
        XCTAssertEqual(firstObject["object"] as? String, "chat.completion")
        XCTAssertEqual(firstMessage["content"] as? String, "hello")

        try await writeRequest(channel, body: requestBody(stream: false))
        let second = try await collectResponse(from: channel)
        XCTAssertEqual(second.head.status, .ok)
        XCTAssertTrue(second.body.contains(#""content":"again""#))
        XCTAssertEqual(backend.snapshot().startCount, 2)

        _ = try await channel.finish()
    }

    func testStreamingRequestReturnsOrderedSSEAndExactlyOneDone() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(
                text: ["hel", "lo"],
                finishReason: .length,
                promptTokens: 3,
                completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "text/event-stream")
        let roleRange = try XCTUnwrap(response.body.range(of: #""role":"assistant""#))
        let firstRange = try XCTUnwrap(response.body.range(of: #""content":"hel""#))
        let secondRange = try XCTUnwrap(response.body.range(of: #""content":"lo""#))
        let finishRange = try XCTUnwrap(response.body.range(of: #""finish_reason":"length""#))
        let doneRange = try XCTUnwrap(response.body.range(of: "data: [DONE]\n\n"))
        XCTAssertLessThan(roleRange.lowerBound, firstRange.lowerBound)
        XCTAssertLessThan(firstRange.lowerBound, secondRange.lowerBound)
        XCTAssertLessThan(secondRange.lowerBound, finishRange.lowerBound)
        XCTAssertLessThan(finishRange.lowerBound, doneRange.lowerBound)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)
        let events = try sseJSONEvents(from: response.body)
        let terminal = try XCTUnwrap(
            events.last { object in
                let choices = object["choices"] as? [[String: Any]]
                return choices?.first?["finish_reason"] as? String == "length"
            })
        let usage = try XCTUnwrap(terminal["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)

        _ = try await channel.finish()
    }

    // Streaming reasoning separation (happy path): a thinks-by-default handle
    // (separatesReasoning=true) routes `.text` deltas through StreamingReasoningSplitter, so the
    // `<think>` block arrives as delta.reasoning_content and the answer as delta.content — the joined
    // fields matching the non-streaming ReasoningContentSplitter on the same concatenated output,
    // across an arbitrary chunking that tears the closing tag.
    func testStreamingSeparatesReasoningWhenHandleThinksByDefault() async throws {
        let chunks = ["rea", "soning</th", "ink>ans", "wer"]
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: chunks,
                    finishReason: .stop,
                    promptTokens: 3,
                    completionTokens: 4)
            ],
            separatesReasoning: true)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let events = try sseJSONEvents(from: response.body)
        var reasoning = "", content = ""
        for object in events {
            guard let choices = object["choices"] as? [[String: Any]],
                let delta = choices.first?["delta"] as? [String: Any] else { continue }
            if let r = delta["reasoning_content"] as? String { reasoning += r }
            if let c = delta["content"] as? String { content += c }
        }
        // The concatenated fields equal the non-streaming split of the whole output — the parity contract.
        let expected = ReasoningContentSplitter.split(chunks.joined())
        XCTAssertEqual(reasoning, expected.reasoning)
        XCTAssertEqual(content, expected.content)
        XCTAssertEqual(reasoning, "reasoning")
        XCTAssertEqual(content, "answer")
        // The reasoning must never leak into a plain content delta.
        XCTAssertFalse(response.body.contains(#""content":"rea""#), response.body)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // Streaming reasoning separation (documented divergence): a thinking stream truncated before it
    // emits `</think>` yields ALL text as reasoning_content and NO content — streaming cannot
    // retro-label already-sent reasoning, and by construction those tokens genuinely were reasoning.
    func testStreamingSeparatedThinkingWithoutCloseIsAllReasoning() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: ["still think", "ing, no close"],
                    finishReason: .length,
                    promptTokens: 2,
                    completionTokens: 4)
            ],
            separatesReasoning: true)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        let events = try sseJSONEvents(from: response.body)
        var reasoning = "", content = ""
        for object in events {
            guard let choices = object["choices"] as? [[String: Any]],
                let delta = choices.first?["delta"] as? [String: Any] else { continue }
            if let r = delta["reasoning_content"] as? String { reasoning += r }
            if let c = delta["content"] as? String { content += c }
        }
        XCTAssertEqual(reasoning, "still thinking, no close")
        XCTAssertEqual(content, "")

        _ = try await channel.finish()
    }

    // DECISION PIN (fable ruling 2026-08-20,
    // docs/task-inbox/2026-08-20-streaming-reasoning-truncation-answer-loss.md): a thinking stream that
    // truncates before `</think>` (finish_reason=length) emits reasoning_content ONLY and an EMPTY
    // content — BY DESIGN. This matches the OpenAI o-series / DeepSeek reasoning-model contract (reasoning
    // that consumes the whole max_completion_tokens budget → empty content + finish "length"); it is
    // LIVE-CONFIRMED on qwen3_5 (its reasoning routinely exceeds a small budget). It is NOT the 7f71f5c
    // answer-loss class: no answer was generated to lose, and within the thinking family a no-`</think>`
    // stream is 100% reasoning by construction. Do NOT "fix" this to mirror reasoning into content —
    // that would fabricate an assistant answer into multi-turn history for every SDK that accumulates
    // delta.content. This test locks the behavior against a future well-meaning regression.
    func testTruncatedThinkingStreamEmitsReasoningOnlyWithEmptyContentByDesign() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: ["thinking, ", "still no close tag"],
                    finishReason: .length,
                    promptTokens: 2,
                    completionTokens: 5)
            ],
            separatesReasoning: true)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        let events = try sseJSONEvents(from: response.body)
        var reasoningDeltas = 0
        for object in events {
            guard let choices = object["choices"] as? [[String: Any]],
                let delta = choices.first?["delta"] as? [String: Any] else { continue }
            if delta["reasoning_content"] is String { reasoningDeltas += 1 }
            // No delta may carry a non-null content value — the answer budget was spent on reasoning.
            if let content = delta["content"], !(content is NSNull) {
                XCTFail("truncated thinking must not emit content by design; got \(content)")
            }
        }
        XCTAssertGreaterThanOrEqual(reasoningDeltas, 1, response.body)
        XCTAssertNotNil(response.body.range(of: #""finish_reason":"length""#), response.body)

        _ = try await channel.finish()
    }

    // Frozen contract + the 7f71f5c answer-loss trap pin: a non-separating stream (separatesReasoning
    // =false, the default for every family not live-attested as thinks-by-default) is byte-identical to
    // today — including a stream whose CONTENT begins with the literal text `<think>` and never closes
    // it. That answer must pass through as delta.content verbatim and NEVER be relabeled as reasoning.
    func testStreamingNonSeparatingPassesLiteralThinkContentThroughUnchanged() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: ["<think>", "not really reasoning"],
                    finishReason: .stop,
                    promptTokens: 2,
                    completionTokens: 2)
            ],
            separatesReasoning: false)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        // The literal-<think> answer survives intact as content; no reasoning_content is ever emitted.
        XCTAssertNotNil(response.body.range(of: #""content":"<think>""#), response.body)
        XCTAssertNotNil(response.body.range(of: #""content":"not really reasoning""#), response.body)
        XCTAssertFalse(response.body.contains("reasoning_content"), response.body)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // Non-streaming parity with the by-design streaming truncation contract above: a separating request
    // (separatesReasoning=true) whose thinking truncates before `</think>` (finish_reason=length) must
    // return the whole output as `reasoning_content` with EMPTY content — matching streaming's Option A,
    // not the old behavior of retro-labeling raw chain-of-thought as the assistant's answer.
    func testNonStreamingTruncatedThinkingReturnsReasoningContentAndEmptyContent() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: ["thinking, ", "still no close tag"],
                    finishReason: .length,
                    promptTokens: 2,
                    completionTokens: 5)
            ],
            separatesReasoning: true)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertEqual(message["reasoning_content"] as? String, "thinking, still no close tag")
        // No answer exists (budget spent mid-reasoning): content must be absent/empty, never raw CoT.
        if let content = message["content"], !(content is NSNull) {
            XCTAssertEqual(content as? String, "", response.body)
        }
        XCTAssertEqual(choice["finish_reason"] as? String, "length", response.body)

        _ = try await channel.finish()
    }

    // The 7f71f5c answer-loss trap pin, non-streaming side: a NON-separating request
    // (separatesReasoning=false, the default for every family not live-attested thinks-by-default) is
    // byte-identical to today — a plain answer with no `</think>` stays verbatim `content`, even one that
    // literally begins with `<think>` and never closes it. It must NEVER be relabeled reasoning_content.
    func testNonStreamingNonSeparatingKeepsLiteralThinkAsContent() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completed(
                    text: ["<think>", "not really reasoning"],
                    finishReason: .stop,
                    promptTokens: 2,
                    completionTokens: 2)
            ],
            separatesReasoning: false)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertEqual(message["content"] as? String, "<think>not really reasoning", response.body)
        XCTAssertNil(message["reasoning_content"], response.body)

        _ = try await channel.finish()
    }

    // Interleave characterization: a separating stream (separatesReasoning=true) that reasons, closes
    // </think>, then emits a tool call must yield reasoning_content deltas BEFORE the tool-call deltas,
    // with finish_reason tool_calls — the natural qwen3_5 shape (it closes </think> before tools). The
    // tool-call delta path itself is untouched by the splitter; only .text deltas route through it.
    func testStreamingSeparatesReasoningThenEmitsToolCallInOrder() async throws {
        let backend = ScriptedBackend(
            scripts: [
                .completedWithToolCalls(
                    text: ["let me check</think>"],
                    toolCalls: [
                        OpenAIToolCall(
                            id: "call_0",
                            function: .init(name: "get_weather", arguments: #"{"city":"Paris"}"#))
                    ],
                    finishReason: .toolCalls,
                    promptTokens: 6,
                    completionTokens: 4)
            ],
            separatesReasoning: true)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        let reasoningRange = try XCTUnwrap(
            response.body.range(of: #""reasoning_content":"let me check""#), response.body)
        let toolNameRange = try XCTUnwrap(
            response.body.range(of: #""name":"get_weather""#), response.body)
        // Reasoning is separated and precedes the tool call; the answer (there is none) never appears.
        XCTAssertLessThan(reasoningRange.lowerBound, toolNameRange.lowerBound)
        XCTAssertNotNil(response.body.range(of: #""finish_reason":"tool_calls""#), response.body)

        _ = try await channel.finish()
    }

    // AC6: a streaming response with tool calls emits OpenAI tool-call deltas (head + arguments)
    // and a terminal finish_reason "tool_calls".
    func testStreamingToolCallsEmitOpenAIDeltasAndToolCallsFinishReason() async throws {
        let backend = ScriptedBackend(scripts: [
            .completedWithToolCalls(
                text: [],
                toolCalls: [
                    OpenAIToolCall(
                        id: "call_0",
                        function: .init(name: "get_product", arguments: #"{"query":"RTX 6000 Ada"}"#))
                ],
                finishReason: .toolCalls,
                promptTokens: 12,
                completionTokens: 8)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "text/event-stream")

        // Ordering: the tool name arrives before its arguments before the terminal finish.
        let nameRange = try XCTUnwrap(response.body.range(of: #""name":"get_product""#))
        let argsRange = try XCTUnwrap(response.body.range(of: "RTX 6000 Ada"))
        let finishRange = try XCTUnwrap(response.body.range(of: #""finish_reason":"tool_calls""#))
        let doneRange = try XCTUnwrap(response.body.range(of: "data: [DONE]\n\n"))
        XCTAssertLessThan(nameRange.lowerBound, argsRange.lowerBound)
        XCTAssertLessThan(argsRange.lowerBound, finishRange.lowerBound)
        XCTAssertLessThan(finishRange.lowerBound, doneRange.lowerBound)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        // The head tool-call delta carries the OpenAI streaming shape.
        let events = try sseJSONEvents(from: response.body)
        let toolDeltas = events.compactMap { object -> [String: Any]? in
            let choices = object["choices"] as? [[String: Any]]
            let delta = choices?.first?["delta"] as? [String: Any]
            return (delta?["tool_calls"] != nil) ? delta : nil
        }
        XCTAssertFalse(toolDeltas.isEmpty, "expected at least one tool_calls delta")
        let headCall = try XCTUnwrap((toolDeltas.first?["tool_calls"] as? [[String: Any]])?.first)
        XCTAssertEqual(headCall["index"] as? Int, 0)
        XCTAssertEqual(headCall["id"] as? String, "call_0")
        XCTAssertEqual(headCall["type"] as? String, "function")
        XCTAssertEqual((headCall["function"] as? [String: Any])?["name"] as? String, "get_product")

        // Terminal chunk carries tool_calls finish reason + usage.
        let terminal = try XCTUnwrap(
            events.last { object in
                let choices = object["choices"] as? [[String: Any]]
                return choices?.first?["finish_reason"] as? String == "tool_calls"
            })
        XCTAssertEqual((terminal["usage"] as? [String: Any])?["total_tokens"] as? Int, 20)

        _ = try await channel.finish()
    }

    func testAuthenticationAndBodyBoundsRejectBeforeBackendAdmission() async throws {
        let backend = ScriptedBackend(scripts: [])
        let authLimits = OpenAIChatRequestLimits(
            maximumBodyBytes: 1_024,
            maximumCompletionTokens: 8)
        let authChannel = try await makeChannel(
            backend: backend,
            configuration: .init(
                launchedModel: "qwen3-32b",
                requestLimits: authLimits,
                requiredBearerToken: "secret",
                maximumNonStreamingResponseBytes: 1_024,
                backpressureStallTimeout: .seconds(1)))

        try await writeRequest(authChannel, body: requestBody(stream: false))
        let unauthorized = try await collectResponse(from: authChannel)
        XCTAssertEqual(unauthorized.head.status, .unauthorized)
        XCTAssertTrue(unauthorized.body.contains(#""type":"invalid_request_error""#))
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await authChannel.finish()

        let bodyLimits = OpenAIChatRequestLimits(
            maximumBodyBytes: 32,
            maximumCompletionTokens: 8)
        let oversizedChannel = try await makeChannel(
            backend: backend,
            configuration: .init(
                launchedModel: "qwen3-32b",
                requestLimits: bodyLimits,
                requiredBearerToken: nil,
                maximumNonStreamingResponseBytes: 1_024,
                backpressureStallTimeout: .seconds(1)))
        let head = validHead(contentLength: 128)
        _ = try await oversizedChannel.writeInbound(HTTPServerRequestPart.head(head))
        let tooLarge = try await collectResponse(from: oversizedChannel)
        XCTAssertEqual(tooLarge.head.status, .payloadTooLarge)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await oversizedChannel.finish()
    }

    func testConfiguredCompletionLimitAcceptsBoundaryAndRejectsLargerRequests()
        async throws
    {
        let limits = OpenAIChatRequestLimits(
            maximumBodyBytes: 1_024,
            maximumCompletionTokens: 8_192)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: limits,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_024,
            backpressureStallTimeout: .seconds(1))

        for field in ["max_completion_tokens", "max_tokens"] {
            let acceptedBackend = ScriptedBackend(scripts: [
                .completed(text: ["ok"], promptTokens: 1, completionTokens: 1)
            ])
            let acceptedChannel = try await makeChannel(
                backend: acceptedBackend,
                configuration: configuration)
            try await writeRequest(
                acceptedChannel,
                body: """
                {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"\(field)":8192,"temperature":0,"stream":false}
                """)
            let acceptedResponse = try await collectResponse(
                from: acceptedChannel)
            XCTAssertEqual(acceptedResponse.head.status, .ok)
            XCTAssertEqual(acceptedBackend.snapshot().startCount, 1)
            _ = try await acceptedChannel.finish()

            let rejectedBackend = ScriptedBackend(scripts: [])
            let rejectedChannel = try await makeChannel(
                backend: rejectedBackend,
                configuration: configuration)
            try await writeRequest(
                rejectedChannel,
                body: """
                {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"\(field)":8193,"temperature":0,"stream":false}
                """)
            let rejectedResponse = try await collectResponse(
                from: rejectedChannel)
            XCTAssertEqual(rejectedResponse.head.status, .badRequest)
            XCTAssertTrue(
                rejectedResponse.body.contains(
                    "max_completion_tokens exceeds the configured limit"))
            XCTAssertEqual(rejectedBackend.snapshot().startCount, 0)
            _ = try await rejectedChannel.finish()
        }

        let defaultBackend = ScriptedBackend(scripts: [])
        let defaultChannel = try await makeChannel(backend: defaultBackend)
        try await writeRequest(
            defaultChannel,
            body: """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":4097,"temperature":0,"stream":false}
            """)
        let defaultResponse = try await collectResponse(from: defaultChannel)
        XCTAssertEqual(defaultResponse.head.status, .badRequest)
        XCTAssertEqual(defaultBackend.snapshot().startCount, 0)
        _ = try await defaultChannel.finish()
    }

    func testQueueExhaustionReturnsTyped429WithBoundedRetrySignal() async throws {
        let backend = ScriptedBackend(scripts: [
            .admissionRejected(.queueFull(retryAfterSeconds: 2))
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .tooManyRequests)
        XCTAssertEqual(response.head.headers.first(name: "retry-after"), "2")
        XCTAssertTrue(response.body.contains(#""type":"rate_limit_error""#))
        XCTAssertTrue(response.body.contains(#""code":"queue_full""#))
        XCTAssertEqual(backend.snapshot().startCount, 1)

        _ = try await channel.finish()
    }

    func testRuntimeCapacityAndRequestSizeAdmissionFailuresStayTyped()
        async throws
    {
        let capacityBackend = ScriptedBackend(scripts: [
            .admissionRejected(
                .capacityExceeded(retryAfterSeconds: 3))
        ])
        let capacityChannel = try await makeChannel(
            backend: capacityBackend)
        try await writeRequest(
            capacityChannel,
            body: requestBody(stream: false))
        let capacityResponse = try await collectResponse(
            from: capacityChannel)

        XCTAssertEqual(capacityResponse.head.status, .tooManyRequests)
        XCTAssertEqual(
            capacityResponse.head.headers.first(name: "retry-after"),
            "3")
        XCTAssertTrue(
            capacityResponse.body.contains(#""code":"capacity_exhausted""#))
        _ = try await capacityChannel.finish()

        let oversizedBackend = ScriptedBackend(scripts: [
            .admissionRejected(.requestTooLarge())
        ])
        let oversizedChannel = try await makeChannel(
            backend: oversizedBackend)
        try await writeRequest(
            oversizedChannel,
            body: requestBody(stream: false))
        let oversizedResponse = try await collectResponse(
            from: oversizedChannel)

        XCTAssertEqual(oversizedResponse.head.status, .badRequest)
        XCTAssertNil(
            oversizedResponse.head.headers.first(name: "retry-after"))
        XCTAssertTrue(
            oversizedResponse.body.contains(
                #""type":"invalid_request_error""#))
        XCTAssertTrue(
            oversizedResponse.body.contains(
                #"Request exceeds the loaded model or KV limit"#))
        _ = try await oversizedChannel.finish()
    }

    func testInputCloseCancelsActiveLeaseExactlyOnce() async throws {
        let backend = ScriptedBackend(scripts: [.held])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().startCount == 1 }

        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
            channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        }
        await waitUntil { backend.snapshot().cancelCount == 1 }

        XCTAssertEqual(backend.snapshot().startCount, 1)
        XCTAssertEqual(backend.snapshot().cancelCount, 1)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testOverlappingSecondRequestIsRejectedBeforeAdmission() async throws {
        let backend = ScriptedBackend(scripts: [.held])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().startCount == 1 }
        _ = try await channel.writeInbound(
            HTTPServerRequestPart.head(validHead(contentLength: 1)))
        await waitUntil { backend.snapshot().cancelCount == 1 }

        XCTAssertEqual(backend.snapshot().startCount, 1)
        XCTAssertEqual(backend.snapshot().cancelCount, 1)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testNonWritableChannelLeavesProducerBoundedUntilWritabilityReturns() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["a", "b"], promptTokens: 1, completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)
        channel.isWritable = false
        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireChannelWritabilityChanged()
        }

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil {
            guard let mailbox = backend.snapshot().lastMailbox else { return false }
            let snapshot = await mailbox.snapshot()
            return snapshot.bufferedDeltas == 1 && snapshot.waitingProducers == 1
        }
        let blockedOutbound = try await channel.readOutbound(
            as: HTTPServerResponsePart.self)
        XCTAssertNil(blockedOutbound)

        channel.isWritable = true
        try await channel.testingEventLoop.executeInContext {
            channel.pipeline.fireChannelWritabilityChanged()
        }
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains(#""content":"a""#))
        XCTAssertTrue(response.body.contains(#""content":"b""#))

        _ = try await channel.finish()
    }

    func testWritabilityGateAppliesRapidTransitionsInEventLoopOrder() async throws {
        let gate = ServingChannelWritabilityGate(initiallyWritable: true)

        gate.update(isWritable: false)
        gate.update(isWritable: true)

        try await gate.waitUntilWritable(timeout: .milliseconds(10))
    }

    func testOutboundWriteFailureCancelsActiveLease() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hello"], promptTokens: 1, completionTokens: 1)
        ])
        let configuration = defaultConfiguration()
        let channel = try await NIOAsyncTestingChannel { channel in
            try channel.pipeline.syncOperations.addHandlers(
                FailingOutboundHandler(),
                OpenAIChatCompletionsHTTPHandler(
                    configuration: configuration,
                    backend: backend))
        }

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().cancelCount == 1 }

        XCTAssertEqual(backend.snapshot().startCount, 1)
        XCTAssertEqual(backend.snapshot().cancelCount, 1)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testBackendMailboxCancellationClosesClientInsteadOfLeavingItHanging() async throws {
        let backend = ScriptedBackend(scripts: [.cancelled(.shutdown)])
        let channel = try await makeChannel(backend: backend)
        let connectPromise = channel.eventLoop.makePromise(of: Void.self)
        channel.connect(
            to: try SocketAddress(ipAddress: "127.0.0.1", port: 9_999),
            promise: connectPromise)
        try await connectPromise.futureResult.get()
        XCTAssertTrue(channel.isActive)

        try await writeRequest(channel, body: requestBody(stream: true))
        await waitUntil { backend.snapshot().cancelCount == 1 }
        await waitUntil { !channel.isActive }

        let lease = try XCTUnwrap(backend.snapshot().lastLease)
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .cancelled(.shutdown))
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testRawHTTPPipelineParsesRequestAndEncodesResponse() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["raw"], promptTokens: 1, completionTokens: 1)
        ])
        let configuration = defaultConfiguration()
        let channel = try await NIOAsyncTestingChannel { channel in
            try channel.pipeline.syncOperations.configureHTTPServerPipeline(
                withPipeliningAssistance: false,
                withErrorHandling: true)
            try channel.pipeline.syncOperations.addHandler(
                OpenAIChatCompletionsHTTPHandler(
                    configuration: configuration,
                    backend: backend))
        }

        let body = requestBody(stream: false)
        let rawRequest = """
        POST /v1/chat/completions HTTP/1.1\r
        Host: localhost\r
        Content-Type: application/json\r
        Content-Length: \(body.utf8.count)\r
        \r
        \(body)
        """
        _ = try await channel.writeInbound(ByteBuffer(string: rawRequest))

        var rawResponse = ""
        while !rawResponse.contains(#""content":"raw""#) {
            var buffer: ByteBuffer = try await channel.waitForOutboundWrite()
            rawResponse += buffer.readString(length: buffer.readableBytes) ?? ""
        }
        XCTAssertTrue(rawResponse.hasPrefix("HTTP/1.1 200"))
        XCTAssertTrue(rawResponse.contains(#""object":"chat.completion""#))

        _ = try await channel.finish()
    }

    func testConnectionCloseCompletesLeaseBeforeClosingChannel() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["bye"], promptTokens: 1, completionTokens: 1)
        ])
        let channel = try await makeChannel(backend: backend)
        let connectPromise = channel.eventLoop.makePromise(of: Void.self)
        channel.connect(
            to: try SocketAddress(ipAddress: "127.0.0.1", port: 9_998),
            promise: connectPromise)
        try await connectPromise.futureResult.get()
        XCTAssertTrue(channel.isActive)
        let body = requestBody(stream: false)
        var head = validHead(contentLength: body.utf8.count)
        head.headers.replaceOrAdd(name: "connection", value: "close")

        _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
        _ = try await channel.writeInbound(
            HTTPServerRequestPart.body(ByteBuffer(string: body)))
        _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        let response = try await collectResponse(from: channel)
        await waitUntil { !channel.isActive }

        XCTAssertEqual(response.head.status, .ok)
        let lease = try XCTUnwrap(backend.snapshot().lastLease)
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .completed)
        XCTAssertEqual(backend.snapshot().cancelCount, 0)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }
}

private struct CollectedResponse {
    let head: HTTPResponseHead
    let body: String
}

private func makeChannel(
    backend: ScriptedBackend,
    configuration: ServingHTTPConfiguration = defaultConfiguration()
) async throws -> NIOAsyncTestingChannel {
    try await NIOAsyncTestingChannel { channel in
        try channel.pipeline.syncOperations.addHandler(
            OpenAIChatCompletionsHTTPHandler(
                configuration: configuration,
                backend: backend))
    }
}

private func defaultConfiguration() -> ServingHTTPConfiguration {
    defaultConfiguration(evidence: nil)
}

private func defaultConfiguration(
    evidence: ServingHTTPEvidenceConfiguration?
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        evidence: evidence)
}

private func requestBody(stream: Bool) -> String {
    """
    {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":\(stream)}
    """
}

private func validHead(contentLength: Int) -> HTTPRequestHead {
    HTTPRequestHead(
        version: .http1_1,
        method: .POST,
        uri: "/v1/chat/completions",
        headers: [
            "host": "localhost",
            "content-type": "application/json",
            "content-length": "\(contentLength)",
        ])
}

private func writeRequest(
    _ channel: NIOAsyncTestingChannel,
    body: String,
    authorization: String? = nil
) async throws {
    var head = validHead(contentLength: body.utf8.count)
    if let authorization {
        head.headers.add(name: "authorization", value: authorization)
    }
    _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
    _ = try await channel.writeInbound(
        HTTPServerRequestPart.body(ByteBuffer(string: body)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
}

private actor ServingEvidenceRecorder {
    enum RecorderError: Error {
        case rejected
    }

    private(set) var evidence: [ServingEvidence] = []
    private(set) var failures: [String] = []

    func record(_ value: ServingEvidence) throws {
        evidence.append(value)
    }

    func recordFailure(_ message: String) {
        failures.append(message)
    }

    func snapshot() -> (evidence: [ServingEvidence], failures: [String]) {
        (evidence, failures)
    }
}

private actor ServingSnapshotSequence {
    private var index = 0

    func next() throws -> ServingEvidence.ResourceSnapshot {
        defer { index += 1 }
        switch index {
        case 0:
            return try snapshot(activeRequests: 0)
        case 1:
            return try snapshot(activeRequests: 1)
        default:
            return try snapshot(activeRequests: 0)
        }
    }

    private func snapshot(
        activeRequests: Int
    ) throws -> ServingEvidence.ResourceSnapshot {
        try ServingEvidence.ResourceSnapshot(
            activeRequests: activeRequests,
            coordinatorSlots: activeRequests,
            reservedKVBytes: activeRequests * 4_096,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 8_192 + activeRequests * 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 16_384)
    }
}

private func collectResponse(
    from channel: NIOAsyncTestingChannel
) async throws -> CollectedResponse {
    var head: HTTPResponseHead?
    var body = ""
    while true {
        let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
        switch part {
        case .head(let value):
            head = value
        case .body(.byteBuffer(var buffer)):
            body += buffer.readString(length: buffer.readableBytes) ?? ""
        case .body(.fileRegion):
            XCTFail("Serving responses must not emit file regions")
        case .end:
            return CollectedResponse(head: try XCTUnwrap(head), body: body)
        }
    }
}

private func sseJSONEvents(from body: String) throws -> [[String: Any]] {
    try body
        .components(separatedBy: "\n\n")
        .compactMap { rawEvent -> [String: Any]? in
            guard rawEvent.hasPrefix("data: "), rawEvent != "data: [DONE]" else {
                return nil
            }
            let payload = String(rawEvent.dropFirst("data: ".count))
            return try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
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

private final class ScriptedBackend: ServingGenerationBackend, Sendable {
    struct Snapshot: Sendable {
        let startCount: Int
        let cancelCount: Int
        let lastMailbox: BoundedDeltaMailbox?
        let lastLease: ServingRequestLease?
    }

    enum Script: Sendable {
        case completed(
            text: [String],
            finishReason: OpenAIChatFinishReason = .stop,
            promptTokens: Int,
            completionTokens: Int)
        case completedWithToolCalls(
            text: [String],
            toolCalls: [OpenAIToolCall],
            finishReason: OpenAIChatFinishReason,
            promptTokens: Int,
            completionTokens: Int)
        case cancelled(ServingCancellationReason)
        case admissionRejected(ServingBackendAdmissionError)
        case held
        case heldWithDelayedCancel(DelayedCancellationGate)
    }

    private struct State: Sendable {
        var scripts: [Script]
        var startCount = 0
        var cancelCount = 0
        var lastMailbox: BoundedDeltaMailbox?
        var lastLease: ServingRequestLease?
    }

    private let state: OSAllocatedUnfairLock<State>
    private let separatesReasoning: Bool

    init(scripts: [Script], separatesReasoning: Bool = false) {
        state = OSAllocatedUnfairLock(initialState: State(scripts: scripts))
        self.separatesReasoning = separatesReasoning
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let (script, sequence) = state.withLock { state -> (Script, Int) in
            state.startCount += 1
            let script = state.scripts.isEmpty ? .held : state.scripts.removeFirst()
            return (script, state.startCount)
        }
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 64))
        let lease: ServingRequestLease
        if case .heldWithDelayedCancel(let cancellationGate) = script {
            lease = ServingRequestLease(
                id: ServingRequestID("request-\(sequence)"),
                onCancel: { [self] in
                    await cancellationGate.wait()
                    state.withLock { $0.cancelCount += 1 }
                })
        } else {
            lease = ServingRequestLease(
                id: ServingRequestID("request-\(sequence)"),
                onCancel: { [self] in
                    state.withLock { $0.cancelCount += 1 }
                })
        }
        state.withLock {
            $0.lastMailbox = mailbox
            $0.lastLease = lease
        }
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-\(sequence)",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease,
            separatesReasoning: separatesReasoning)

        if case .admissionRejected(let error) = script {
            throw error
        } else if case .completed(let text, let finishReason, let promptTokens, let completionTokens) = script {
            Task {
                do {
                    for delta in text {
                        try await mailbox.send(.text(delta))
                    }
                    try await mailbox.send(
                        .completion(
                            ServingGenerationCompletion(
                                finishReason: finishReason,
                                usage: OpenAIChatUsage(
                                    promptTokens: promptTokens,
                                    completionTokens: completionTokens))))
                    await mailbox.finish()
                } catch {
                    // Cancellation is observed through the lease counter.
                }
            }
        } else if case .completedWithToolCalls(let text, let toolCalls, let finishReason, let promptTokens, let completionTokens) = script {
            Task {
                do {
                    for delta in text {
                        try await mailbox.send(.text(delta))
                    }
                    if !toolCalls.isEmpty {
                        try await mailbox.send(.toolCalls(toolCalls))
                    }
                    try await mailbox.send(
                        .completion(
                            ServingGenerationCompletion(
                                finishReason: finishReason,
                                usage: OpenAIChatUsage(
                                    promptTokens: promptTokens,
                                    completionTokens: completionTokens))))
                    await mailbox.finish()
                } catch {
                    // Cancellation is observed through the lease counter.
                }
            }
        } else if case .cancelled(let reason) = script {
            Task {
                await mailbox.cancel(reason)
            }
        }
        return handle
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                startCount: $0.startCount,
                cancelCount: $0.cancelCount,
                lastMailbox: $0.lastMailbox,
                lastLease: $0.lastLease)
        }
    }
}

private actor DelayedCancellationGate {
    private(set) var isWaiting = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        isWaiting = true
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}

private enum ServingSnapshotError: Error {
    case unavailable
}

private struct SyntheticWriteFailure: Error {}

private final class FailingOutboundHandler: ChannelOutboundHandler, Sendable {
    typealias OutboundIn = HTTPServerResponsePart
    typealias OutboundOut = HTTPServerResponsePart

    func write(
        context: ChannelHandlerContext,
        data: NIOAny,
        promise: EventLoopPromise<Void>?
    ) {
        promise?.fail(SyntheticWriteFailure())
    }
}
