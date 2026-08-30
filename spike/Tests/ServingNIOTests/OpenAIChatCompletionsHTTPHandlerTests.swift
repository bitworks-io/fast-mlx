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

    func testStreamingToolCallAboveLegacyMailboxLimitUsesConfiguredByteBudget() async throws {
        let arguments = #"{"payload":""# + String(repeating: "x", count: 40 * 1_024) + #""}"#
        let toolCall = OpenAIToolCall(
            id: "call-large",
            type: "function",
            function: .init(name: "store_payload", arguments: arguments))
        let backend = ScriptedBackend(
            scripts: [
                .completedWithToolCalls(
                    text: [],
                    toolCalls: [toolCall],
                    finishReason: .toolCalls,
                    promptTokens: 8,
                    completionTokens: 256)
            ],
            mailboxMaximumBytes: 1_048_576)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains(#""id":"call-large""#), response.body)
        XCTAssertTrue(response.body.contains(#""name":"store_payload""#), response.body)
        XCTAssertTrue(response.body.contains(#""finish_reason":"tool_calls""#), response.body)
        XCTAssertEqual(
            response.body.components(separatedBy: "data: [DONE]\n\n").count - 1,
            1)

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

    func testStructuralOnlyHTTPDecoderAcceptsLargeCompletionBudgetForModelAwareAdmission()
        async throws
    {
        let limits = OpenAIChatRequestLimits(
            maximumBodyBytes: 1_024,
            maximumCompletionTokens: 4_096,
            enforceMaximumCompletionTokensDuringDecoding: false)
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(
                """
                {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":32768,"temperature":0,"stream":true}
                """.utf8),
            limits: limits)

        XCTAssertEqual(request.maxCompletionTokens, 32_768)
    }

    func testModelsEndpointRequiresAuthAndReturnsAllowlistedCapabilityMetadata()
        async throws
    {
        let capabilities = try modelCapabilities(
            effectiveContext: 131_072,
            defaultCompletion: 8_192,
            maximumCompletion: 65_536,
            maximumNonStreaming: 16_384,
            requestBodyMaximum: 8 * 1_048_576,
            nonStreamingResponseMaximum: 16 * 1_048_576,
            policy: .clamp)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: OpenAIChatRequestLimits(
                maximumBodyBytes: capabilities.maximumRequestBodyBytes,
                maximumCompletionTokens: capabilities.maximumCompletionTokens),
            requiredBearerToken: "secret",
            maximumNonStreamingResponseBytes:
                capabilities.maximumNonStreamingResponseBytes,
            backpressureStallTimeout: .seconds(1),
            modelCapabilities: capabilities)

        let unauthorizedBackend = ScriptedBackend(scripts: [])
        let unauthorizedChannel = try await makeChannel(
            backend: unauthorizedBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            unauthorizedChannel,
            method: .GET,
            uri: "/v1/models")
        let unauthorized = try await collectResponse(from: unauthorizedChannel)
        XCTAssertEqual(unauthorized.head.status, .unauthorized)
        XCTAssertEqual(unauthorizedBackend.snapshot().startCount, 0)
        _ = try await unauthorizedChannel.finish()

        let authorizedBackend = ScriptedBackend(scripts: [])
        let authorizedChannel = try await makeChannel(
            backend: authorizedBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            authorizedChannel,
            method: .GET,
            uri: "/v1/models",
            authorization: "Bearer secret")
        let response = try await collectResponse(from: authorizedChannel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        XCTAssertEqual(root["object"] as? String, "list")
        let model = try XCTUnwrap((root["data"] as? [[String: Any]])?.first)
        XCTAssertEqual(model["id"] as? String, "qwen3-32b")
        XCTAssertEqual(model["object"] as? String, "model")
        XCTAssertNotNil(model["created"])
        XCTAssertEqual(model["owned_by"] as? String, "fast-mlx")
        XCTAssertEqual(model["max_model_len"] as? Int, 131_072)
        let extensionKeys = try XCTUnwrap(
            (model["fast_mlx_capabilities"] as? [String: Any])?.keys.sorted())
        XCTAssertEqual(
            extensionKeys,
            [
                "completion_limit_policy",
                "default_completion_tokens",
                "effective_max_context_tokens",
                "maximum_completion_tokens",
                "maximum_non_streaming_completion_tokens",
                "maximum_non_streaming_response_bytes",
                "maximum_request_body_bytes",
                "native_max_context_tokens",
                "reasoning_tokens_count_toward_completion",
            ])
        let extensionObject = try XCTUnwrap(
            model["fast_mlx_capabilities"] as? [String: Any])
        XCTAssertEqual(extensionObject["native_max_context_tokens"] as? Int, 262_144)
        XCTAssertEqual(extensionObject["effective_max_context_tokens"] as? Int, 131_072)
        XCTAssertEqual(extensionObject["default_completion_tokens"] as? Int, 8_192)
        XCTAssertEqual(extensionObject["maximum_completion_tokens"] as? Int, 65_536)
        XCTAssertEqual(
            extensionObject["maximum_non_streaming_completion_tokens"] as? Int,
            16_384)
        XCTAssertEqual(
            extensionObject["maximum_request_body_bytes"] as? Int,
            8 * 1_048_576)
        XCTAssertEqual(
            extensionObject["maximum_non_streaming_response_bytes"] as? Int,
            16 * 1_048_576)
        XCTAssertEqual(extensionObject["completion_limit_policy"] as? String, "clamp")
        XCTAssertEqual(
            extensionObject["reasoning_tokens_count_toward_completion"] as? Bool,
            true)
        XCTAssertEqual(authorizedBackend.snapshot().startCount, 0)
        _ = try await authorizedChannel.finish()
    }

    func testMetricsEndpointRequiresAuthAndReturnsSnapshotPrometheusText()
        async throws
    {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 2,
            coordinatorSlots: 4,
            reservedKVBytes: 65_536,
            maxReservedKVBytes: 131_072,
            mlxActiveBytes: 262_144,
            mlxCacheBytes: 32_768,
            mlxPeakBytes: 524_288,
            fitModeledPeakBytes: 700_000,
            fitMeasuredPeakBytes: 710_000,
            fitModeledWeightsBytes: 400_000,
            fitModeledKVBytes: 200_000,
            fitModeledTransientBytes: 50_000,
            fitModeledHeadroomBytes: 50_000)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: "secret",
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: { snapshot },
                record: { _ in },
                reportFailure: { _ in }))

        let unauthorizedBackend = ScriptedBackend(scripts: [])
        let unauthorizedChannel = try await makeChannel(
            backend: unauthorizedBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            unauthorizedChannel,
            method: .GET,
            uri: "/metrics")
        let unauthorized = try await collectResponse(from: unauthorizedChannel)
        XCTAssertEqual(unauthorized.head.status, .unauthorized)
        XCTAssertEqual(unauthorizedBackend.snapshot().startCount, 0)
        _ = try await unauthorizedChannel.finish()

        let authorizedBackend = ScriptedBackend(scripts: [])
        let authorizedChannel = try await makeChannel(
            backend: authorizedBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            authorizedChannel,
            method: .GET,
            uri: "/metrics",
            authorization: "Bearer secret")
        let response = try await collectResponse(from: authorizedChannel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(
            response.head.headers.first(name: "content-type"),
            "text/plain; version=0.0.4; charset=utf-8")
        XCTAssertEqual(
            response.body,
            """
            # HELP fastmlx_up fast-mlx serving metrics endpoint availability.
            # TYPE fastmlx_up gauge
            fastmlx_up 1
            # HELP fastmlx_active_requests Active generation requests.
            # TYPE fastmlx_active_requests gauge
            fastmlx_active_requests 2
            # HELP fastmlx_coordinator_slots Coordinator slots currently reserved by serving.
            # TYPE fastmlx_coordinator_slots gauge
            fastmlx_coordinator_slots 4
            # HELP fastmlx_reserved_kv_bytes Reserved KV-cache bytes.
            # TYPE fastmlx_reserved_kv_bytes gauge
            fastmlx_reserved_kv_bytes 65536
            # HELP fastmlx_max_reserved_kv_bytes Peak reserved KV-cache bytes.
            # TYPE fastmlx_max_reserved_kv_bytes gauge
            fastmlx_max_reserved_kv_bytes 131072
            # HELP fastmlx_mlx_active_bytes MLX active allocator bytes.
            # TYPE fastmlx_mlx_active_bytes gauge
            fastmlx_mlx_active_bytes 262144
            # HELP fastmlx_mlx_cache_bytes MLX cache allocator bytes.
            # TYPE fastmlx_mlx_cache_bytes gauge
            fastmlx_mlx_cache_bytes 32768
            # HELP fastmlx_mlx_peak_bytes MLX peak allocator bytes.
            # TYPE fastmlx_mlx_peak_bytes gauge
            fastmlx_mlx_peak_bytes 524288
            # HELP fastmlx_fit_modeled_peak_bytes Fit-check modeled peak bytes.
            # TYPE fastmlx_fit_modeled_peak_bytes gauge
            fastmlx_fit_modeled_peak_bytes 700000
            # HELP fastmlx_fit_measured_peak_bytes Fit-check measured peak bytes.
            # TYPE fastmlx_fit_measured_peak_bytes gauge
            fastmlx_fit_measured_peak_bytes 710000
            # HELP fastmlx_fit_modeled_weights_bytes Fit-check modeled weights bytes.
            # TYPE fastmlx_fit_modeled_weights_bytes gauge
            fastmlx_fit_modeled_weights_bytes 400000
            # HELP fastmlx_fit_modeled_kv_bytes Fit-check modeled KV-cache bytes.
            # TYPE fastmlx_fit_modeled_kv_bytes gauge
            fastmlx_fit_modeled_kv_bytes 200000
            # HELP fastmlx_fit_modeled_transient_bytes Fit-check modeled transient bytes.
            # TYPE fastmlx_fit_modeled_transient_bytes gauge
            fastmlx_fit_modeled_transient_bytes 50000
            # HELP fastmlx_fit_modeled_headroom_bytes Fit-check modeled headroom bytes.
            # TYPE fastmlx_fit_modeled_headroom_bytes gauge
            fastmlx_fit_modeled_headroom_bytes 50000

            """)
        XCTAssertEqual(
            response.head.headers.first(name: "content-length"),
            "\(response.body.utf8.count)")
        XCTAssertFalse(response.body.contains("qwen3-32b"))
        XCTAssertFalse(response.body.contains("secret"))
        XCTAssertEqual(authorizedBackend.snapshot().startCount, 0)
        _ = try await authorizedChannel.finish()
    }

    func testMetricsEndpointRejectsWrongMethodAndRequestBodyBeforeBackendWork()
        async throws
    {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: { snapshot },
                record: { _ in },
                reportFailure: { _ in }))

        let postBackend = ScriptedBackend(scripts: [])
        let postChannel = try await makeChannel(
            backend: postBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            postChannel,
            method: .POST,
            uri: "/metrics")
        let postResponse = try await collectResponse(from: postChannel)
        XCTAssertEqual(postResponse.head.status, .methodNotAllowed)
        XCTAssertEqual(postBackend.snapshot().startCount, 0)
        _ = try await postChannel.finish()

        let bodyBackend = ScriptedBackend(scripts: [])
        let bodyChannel = try await makeChannel(
            backend: bodyBackend,
            configuration: configuration)
        try await writeHeadWithBodyRequest(
            bodyChannel,
            method: .GET,
            uri: "/metrics",
            body: "{}")
        let bodyResponse = try await collectResponse(from: bodyChannel)
        XCTAssertEqual(bodyResponse.head.status, .badRequest)
        XCTAssertTrue(
            bodyResponse.body.contains("GET \\/metrics does not accept a request body"),
            bodyResponse.body)
        XCTAssertEqual(bodyBackend.snapshot().startCount, 0)
        _ = try await bodyChannel.finish()
    }

    func testMetricsEndpointWithoutSnapshotProviderReturnsDeterministic500WithoutBackendWork()
        async throws
    {
        let backend = ScriptedBackend(scripts: [])
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeHeadOnlyRequest(
            channel,
            method: .GET,
            uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .internalServerError)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        XCTAssertEqual(
            response.head.headers.first(name: "content-length"),
            "\(response.body.utf8.count)")
        XCTAssertNil(response.head.headers.first(name: "connection"))
        XCTAssertEqual(
            response.body,
            """
            {"error":{"code":"metrics_unavailable","message":"Metrics snapshot is not configured","param":null,"type":"server_error"}}
            """)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    func testMetricsEndpointSnapshotFailureReturnsDeterministic500AndReportsFailure()
        async throws
    {
        let backend = ScriptedBackend(scripts: [])
        let recorder = ServingEvidenceRecorder()
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            evidence: ServingHTTPEvidenceConfiguration(
                snapshot: { throw MetricsSnapshotTestError.rejected },
                record: { _ in },
                reportFailure: { message in
                    Task {
                        await recorder.recordFailure(message)
                    }
                }))
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)

        try await writeHeadOnlyRequest(
            channel,
            method: .GET,
            uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .internalServerError)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        XCTAssertEqual(
            response.head.headers.first(name: "content-length"),
            "\(response.body.utf8.count)")
        XCTAssertNil(response.head.headers.first(name: "connection"))
        XCTAssertEqual(
            response.body,
            """
            {"error":{"code":"metrics_snapshot_failed","message":"Metrics snapshot failed","param":null,"type":"server_error"}}
            """)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        await waitUntil {
            await recorder.snapshot().failures == ["serving metrics snapshot failed"]
        }
        let recorderSnapshot = await recorder.snapshot()
        XCTAssertEqual(
            recorderSnapshot.failures,
            ["serving metrics snapshot failed"])
        XCTAssertEqual(recorderSnapshot.evidence.count, 0)
        _ = try await channel.finish()
    }

    func testBackendInvalidRequestWithCodeReturnsHTTP400InsteadOfInternalError()
        async throws
    {
        let backend = ScriptedBackend(scripts: [
            .servingError(
                .invalidRequestWithCode(
                    "Requested completion exceeds this model's remaining context",
                    param: "max_completion_tokens",
                    code: "completion_limit_exceeded"))
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .badRequest)
        XCTAssertTrue(response.body.contains(#""type":"invalid_request_error""#), response.body)
        XCTAssertTrue(response.body.contains(#""param":"max_completion_tokens""#), response.body)
        XCTAssertTrue(response.body.contains(#""code":"completion_limit_exceeded""#), response.body)
        XCTAssertFalse(response.body.contains(#""code":"internal_error""#), response.body)

        _ = try await channel.finish()
    }

    func testNonStreamingToolCallBytesAreBoundedAndCancelGeneration() async throws {
        let backend = ScriptedBackend(scripts: [
            .completedWithToolCalls(
                text: [],
                toolCalls: [
                    OpenAIToolCall(
                        id: "call_0",
                        function: .init(
                            name: "lookup",
                            arguments: String(repeating: "x", count: 256)))
                ],
                finishReason: .toolCalls,
                promptTokens: 4,
                completionTokens: 4)
        ])
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 128,
            backpressureStallTimeout: .seconds(1))
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .payloadTooLarge)
        XCTAssertTrue(response.body.contains(#""code":"response_too_large""#), response.body)
        XCTAssertFalse(response.body.contains(String(repeating: "x", count: 128)))
        await waitUntil { backend.snapshot().cancelCount == 1 }
        XCTAssertEqual(backend.snapshot().cancelCount, 1)
        _ = try await channel.finish()
    }

    func testSuccessfulResponsesIncludeModelAwareCompletionBudgetHeaders()
        async throws
    {
        let nonStreamingResolution = budgetResolution(
            requested: 65_536,
            applied: 16_384,
            maximumAllowed: 16_384,
            prompt: 114_688,
            wasClamped: true,
            limitingFactor: .contextWindow)
        let streamResolution = budgetResolution(
            requested: 65_536,
            applied: 65_536,
            maximumAllowed: 65_536,
            prompt: 65_536,
            wasClamped: false,
            limitingFactor: .operatorMaximumAndContextWindow)
        let backend = ScriptedBackend(scripts: [
            .completed(
                text: ["json"],
                promptTokens: 114_688,
                completionTokens: 16_384,
                budgetResolution: nonStreamingResolution),
            .completed(
                text: ["sse"],
                promptTokens: 65_536,
                completionTokens: 65_536,
                budgetResolution: streamResolution),
        ])
        let modelAwareTransportConfiguration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: OpenAIChatRequestLimits(
                maximumBodyBytes: 1_048_576,
                maximumCompletionTokens: 4_096,
                enforceMaximumCompletionTokensDuringDecoding: false),
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))

        let nonStreamingChannel = try await makeChannel(
            backend: backend,
            configuration: modelAwareTransportConfiguration)
        try await writeRequest(
            nonStreamingChannel,
            body: """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":65536,"temperature":0,"stream":false}
            """)
        let nonStreaming = try await collectResponse(from: nonStreamingChannel)
        XCTAssertEqual(nonStreaming.head.status, .ok)
        XCTAssertEqual(
            nonStreaming.head.headers.first(name: "x-fastmlx-requested-completion-tokens"),
            "65536")
        XCTAssertEqual(
            nonStreaming.head.headers.first(name: "x-fastmlx-applied-completion-tokens"),
            "16384")
        XCTAssertEqual(
            nonStreaming.head.headers.first(name: "x-fastmlx-max-completion-tokens"),
            "16384")
        XCTAssertEqual(
            nonStreaming.head.headers.first(name: "x-fastmlx-completion-tokens-clamped"),
            "true")
        XCTAssertEqual(
            nonStreaming.head.headers.first(name: "x-fastmlx-completion-limit-policy"),
            "clamp")
        _ = try await nonStreamingChannel.finish()

        let streamingChannel = try await makeChannel(
            backend: backend,
            configuration: modelAwareTransportConfiguration)
        try await writeRequest(
            streamingChannel,
            body: """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":65536,"temperature":0,"stream":true}
            """)
        let streaming = try await collectResponse(from: streamingChannel)
        XCTAssertEqual(streaming.head.status, .ok)
        XCTAssertEqual(
            streaming.head.headers.first(name: "x-fastmlx-requested-completion-tokens"),
            "65536")
        XCTAssertEqual(
            streaming.head.headers.first(name: "x-fastmlx-applied-completion-tokens"),
            "65536")
        XCTAssertEqual(
            streaming.head.headers.first(name: "x-fastmlx-max-completion-tokens"),
            "65536")
        XCTAssertEqual(
            streaming.head.headers.first(name: "x-fastmlx-completion-tokens-clamped"),
            "false")
        XCTAssertEqual(
            streaming.head.headers.first(name: "x-fastmlx-completion-limit-policy"),
            "reject")

        _ = try await streamingChannel.finish()
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

private func modelCapabilities(
    effectiveContext: Int,
    defaultCompletion: Int,
    maximumCompletion: Int,
    maximumNonStreaming: Int,
    requestBodyMaximum: Int? = nil,
    nonStreamingResponseMaximum: Int = 16 * 1_048_576,
    policy: ServingCompletionLimitPolicy
) throws -> ServingModelCapabilities {
    try ServingModelCapabilities(
        model: "qwen3-32b",
        nativeMaxContextTokens: 262_144,
        effectiveMaxContextTokens: effectiveContext,
        requestedDefaultCompletionTokens: defaultCompletion,
        defaultCompletionTokensWasExplicit: true,
        maximumCompletionTokens: maximumCompletion,
        maximumNonStreamingCompletionTokens: maximumNonStreaming,
        maximumRequestBodyBytes: requestBodyMaximum,
        maximumNonStreamingResponseBytes: nonStreamingResponseMaximum,
        completionLimitPolicy: policy)
}

private func budgetResolution(
    requested: Int?,
    applied: Int,
    maximumAllowed: Int,
    prompt: Int,
    wasClamped: Bool,
    limitingFactor: ServingCompletionLimitingFactor
) -> ServingCompletionBudgetResolution {
    ServingCompletionBudgetResolution(
        requestedCompletionTokens: requested,
        appliedCompletionTokens: applied,
        maximumAllowedCompletionTokens: maximumAllowed,
        renderedPromptTokens: prompt,
        wasClamped: wasClamped,
        limitingFactor: limitingFactor)
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

private func validHead(
    method: HTTPMethod,
    uri: String,
    authorization: String? = nil
) -> HTTPRequestHead {
    var head = HTTPRequestHead(
        version: .http1_1,
        method: method,
        uri: uri,
        headers: [
            "host": "localhost"
        ])
    if let authorization {
        head.headers.add(name: "authorization", value: authorization)
    }
    return head
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

private func writeHeadOnlyRequest(
    _ channel: NIOAsyncTestingChannel,
    method: HTTPMethod,
    uri: String,
    authorization: String? = nil
) async throws {
    _ = try await channel.writeInbound(
        HTTPServerRequestPart.head(
            validHead(method: method, uri: uri, authorization: authorization)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
}

private func writeHeadWithBodyRequest(
    _ channel: NIOAsyncTestingChannel,
    method: HTTPMethod,
    uri: String,
    body: String,
    authorization: String? = nil
) async throws {
    var head = validHead(method: method, uri: uri, authorization: authorization)
    head.headers.add(name: "content-length", value: "\(body.utf8.count)")
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

private enum MetricsSnapshotTestError: Error {
    case rejected
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
            completionTokens: Int,
            budgetResolution: ServingCompletionBudgetResolution? = nil)
        case completedWithToolCalls(
            text: [String],
            toolCalls: [OpenAIToolCall],
            finishReason: OpenAIChatFinishReason,
            promptTokens: Int,
            completionTokens: Int)
        case servingError(OpenAIServingError)
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

    private let mailboxMaximumBytes: Int

    init(
        scripts: [Script],
        separatesReasoning: Bool = false,
        mailboxMaximumBytes: Int = 1_024
    ) {
        state = OSAllocatedUnfairLock(initialState: State(scripts: scripts))
        self.separatesReasoning = separatesReasoning
        self.mailboxMaximumBytes = mailboxMaximumBytes
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let (script, sequence) = state.withLock { state -> (Script, Int) in
            state.startCount += 1
            let script = state.scripts.isEmpty ? .held : state.scripts.removeFirst()
            return (script, state.startCount)
        }
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(
                maxDeltas: 1,
                maxBytes: mailboxMaximumBytes))
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
            completionBudgetResolution: budgetResolution(for: script),
            separatesReasoning: separatesReasoning)

        if case .admissionRejected(let error) = script {
            throw error
        } else if case .servingError(let error) = script {
            throw error
        } else if case .completed(let text, let finishReason, let promptTokens, let completionTokens, _) = script {
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

    private func budgetResolution(
        for script: Script
    ) -> ServingCompletionBudgetResolution? {
        guard case .completed(_, _, _, _, let budgetResolution) = script else {
            return nil
        }
        return budgetResolution
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
