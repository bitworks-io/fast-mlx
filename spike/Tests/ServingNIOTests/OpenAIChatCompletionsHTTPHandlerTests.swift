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

    func testRequestFailureDiagnosticLineFormatsReasonWithSpacesAsUnderscores() {
        struct SpacedError: Error, CustomStringConvertible {
            var description: String { "backend exploded during render" }
        }
        let line = OpenAIChatCompletionsHTTPHandler.requestFailureDiagnosticLine(
            requestID: "chatcmpl-42",
            error: SpacedError())
        XCTAssertEqual(
            line,
            "request_failure=true request_id=chatcmpl-42 "
                + "request_failure_reason=backend_exploded_during_render")
    }

    func testGenerationFailureReportsSwallowedCauseToRequestFailureReporter() async throws {
        let backend = ScriptedBackend(scripts: [.untypedFailure])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            requestFailureReporter: { line in
                Task { await recorder.recordFailure(line) }
            })
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        _ = try await collectResponse(from: channel)
        await waitUntil { await recorder.failures.count == 1 }

        let reported = await recorder.snapshot()
        let line = try XCTUnwrap(reported.failures.first)
        // Anti-vacuity: assert on the distinctive error's own text, not merely that some line
        // arrived — a reporter that fired with an unrelated or empty message must fail this.
        XCTAssertTrue(
            line.contains("DISTINCTIVE-UNTYPED-GENERATION-FAILURE-7f2c9a41"),
            "expected the swallowed error's text in the reported line, got: \(line)")
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testSuccessfulRequestReportsNoRequestFailure() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["done"], promptTokens: 1, completionTokens: 1)
        ])
        let recorder = ServingEvidenceRecorder()
        let configuration = defaultConfiguration(
            requestFailureReporter: { line in
                Task { await recorder.recordFailure(line) }
            })
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .ok)
        // Give any spuriously-scheduled reporter Task a chance to land before asserting absence.
        try await Task.sleep(for: .milliseconds(20))

        let reported = await recorder.snapshot()
        XCTAssertTrue(reported.failures.isEmpty)
        _ = try await channel.finish()
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

    /// Deterministic reproduction of
    /// `docs/task-inbox/2026-09-16-keepalive-next-request-race-closes-connection.md`: the first
    /// request's final response bytes are fully flushed (`collectResponse` below returns) before the
    /// second request's head is delivered, but the handler's own request-finishing bookkeeping is
    /// parked at `responseCompletionTestHook` -- exactly the window between "client can see the
    /// full response" and "this connection is ready for a new request" -- until this test releases
    /// it. Before the fix, `receiveHead` still finds a non-terminal `activeControl` for the first
    /// request in that window and treats the second head as a concurrent request, closing the
    /// connection out from under it.
    func testSecondKeepAliveRequestArrivingWhileFirstResponseFinishesIsNotDropped() async throws {
        let raceGate = ResponseCompletionRaceGate()
        // The handler installs ONE hook shared by every generation it runs on this connection --
        // only the FIRST request (the one this test races against) should park on `raceGate`. The
        // second request's own hook call must be a no-op, or it would block forever on a gate this
        // test only ever releases once.
        let hookCallCount = OSAllocatedUnfairLock(initialState: 0)
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2),
            .completed(text: ["again"], promptTokens: 4, completionTokens: 1),
        ])
        let channel = try await makeChannel(
            backend: backend,
            responseCompletionTestHook: {
                let isFirstCall = hookCallCount.withLock { count -> Bool in
                    count += 1
                    return count == 1
                }
                if isFirstCall {
                    await raceGate.wait()
                }
            })

        try await writeRequest(channel, body: requestBody(stream: false))
        let first = try await collectResponse(from: channel)
        XCTAssertEqual(first.head.status, .ok)
        XCTAssertTrue(first.body.contains(#""content":"hello""#))

        // The client has now received the complete first response. The first request's own
        // generation task is parked exactly at the race window (see the hook's own doc comment).
        await waitUntil { await raceGate.isWaiting }

        // Deliver the second keep-alive request's head+body+end while that window is still open.
        // Before the fix, this throws `ChannelError.ioOnClosedChannel` (from the closed-channel
        // guard `receiveHead` takes because `activeControl` is not yet terminal) instead of
        // reaching the assertions below.
        try await writeRequest(channel, body: requestBody(stream: false))
        await raceGate.release()

        let second = try await collectResponse(from: channel)
        XCTAssertEqual(second.head.status, .ok)
        XCTAssertTrue(second.body.contains(#""content":"again""#))
        XCTAssertEqual(backend.snapshot().startCount, 2)

        _ = try await channel.finish()
    }

    /// Deterministic reproduction of the SECOND, narrower keep-alive race window described in
    /// `docs/task-inbox/2026-09-16-keepalive-next-request-race-closes-connection.md`: even with
    /// `control.markTerminal()` moved before `started.lease.complete()` (verified by
    /// `testSecondKeepAliveRequestArrivingWhileFirstResponseFinishesIsNotDropped` above), the
    /// generation `Task`'s resumption after the final response write's own internal `await` (on
    /// the NIO write promise inside `writePart`) is scheduled independently of "the write's bytes
    /// reached the client" -- see `finalWriteRaceTestHook`'s doc comment on
    /// `OpenAIChatCompletionsHTTPHandler` for exactly where that gap comes from. This test parks
    /// the FIRST request's generation task inside `writePart`'s own write-completion callback --
    /// strictly earlier than where the test above parks -- so `collectResponse` can observe the
    /// complete first response while `control.markTerminal()` (called from `writeFinalPart`, ahead
    /// of the write in the fixed version, but not reached at all yet in this parked state) has not
    /// yet run. Before the `writeFinalPart` fix, this reliably throws
    /// `ChannelError.ioOnClosedChannel` when the second request's head is delivered; after the fix,
    /// `control.isTerminal` is already `true` before this hook ever fires (the write has not even
    /// started), so the second request succeeds regardless of how long this hook parks.
    func testSecondKeepAliveRequestDuringFinalWriteCompletionIsNotDropped() async throws {
        let raceGate = ResponseCompletionRaceGate()
        // Same one-hook-per-connection, only-the-first-call-parks shape as
        // `testSecondKeepAliveRequestArrivingWhileFirstResponseFinishesIsNotDropped` above -- the
        // second request's own final-write completion must not block forever on a gate this test
        // only ever releases once.
        let hookCallCount = OSAllocatedUnfairLock(initialState: 0)
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2),
            .completed(text: ["again"], promptTokens: 4, completionTokens: 1),
        ])
        let channel = try await makeChannel(
            backend: backend,
            finalWriteRaceTestHook: {
                let isFirstCall = hookCallCount.withLock { count -> Bool in
                    count += 1
                    return count == 1
                }
                if isFirstCall {
                    await raceGate.wait()
                }
            })

        try await writeRequest(channel, body: requestBody(stream: false))
        let first = try await collectResponse(from: channel)
        XCTAssertEqual(first.head.status, .ok)
        XCTAssertTrue(first.body.contains(#""content":"hello""#))

        // The client has now received the complete first response, but the first request's
        // generation task is parked inside `writePart`'s write-completion callback, strictly
        // before `writeFinalPart`'s `control.markTerminal()` call can be reached (see the fix:
        // `control.markTerminal()` now runs BEFORE this write is even initiated, so by the time
        // this hook fires, `control.isTerminal` is already `true`).
        await waitUntil { await raceGate.isWaiting }

        // Deliver the second keep-alive request's head+body+end while that window is still open.
        // Before the fix, this throws `ChannelError.ioOnClosedChannel`.
        try await writeRequest(channel, body: requestBody(stream: false))
        await raceGate.release()

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

    // `stream_options.include_usage:true` opt-in: one extra empty-choices usage chunk lands
    // immediately before `data: [DONE]`, carrying the SAME prompt/completion numbers the
    // finish-reason chunk (and the non-streaming path) would report.
    func testStreamOptionsIncludeUsageEmitsTerminalUsageChunkBeforeDone() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(
            channel,
            body: """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}
            """)
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let doneRange = try XCTUnwrap(response.body.range(of: "data: [DONE]\n\n"))
        let events = try sseJSONEvents(from: response.body)
        let usageOnlyEvents = events.filter { object in
            (object["choices"] as? [[String: Any]])?.isEmpty == true
        }
        XCTAssertEqual(usageOnlyEvents.count, 1, response.body)
        let usage = try XCTUnwrap(usageOnlyEvents.first?["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)

        // Positioned immediately before [DONE]: nothing else appears between the usage chunk and it.
        let usageChunkRange = try XCTUnwrap(response.body.range(of: #""choices":[]"#))
        XCTAssertLessThan(usageChunkRange.lowerBound, doneRange.lowerBound)
        let between = response.body[usageChunkRange.upperBound..<doneRange.lowerBound]
        XCTAssertFalse(
            between.contains("data: "),
            "expected no other chunk between the usage chunk and [DONE], got: \(between)")
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // Absence of `stream_options` must leave today's streamed bytes unchanged: no empty-choices
    // usage chunk, and the same four events (role, two content deltas, finish) as before this
    // increment.
    func testStreamingWithoutStreamOptionsEmitsNoUsageChunkByteShapeUnchanged() async throws {
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
        XCTAssertFalse(response.body.contains(#""choices":[]"#), response.body)
        let events = try sseJSONEvents(from: response.body)
        XCTAssertEqual(events.count, 4, response.body)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // MARK: - Legacy POST /v1/completions

    func testCompletionsNonStreamingReturnsTextCompletionObjectShape() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "text_completion")
        // Legacy completions ids use `cmpl-`, not chat's `chatcmpl-` — the backend generates
        // `chatcmpl-1` for the first request; the HTTP encoding layer rewrites it for this route.
        XCTAssertEqual(object["id"] as? String, "cmpl-1")
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["text"] as? String, "hello")
        XCTAssertEqual(choice["index"] as? Int, 0)
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
        XCTAssertTrue(choice.keys.contains("logprobs"))
        XCTAssertTrue(choice["logprobs"] is NSNull)
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)

        _ = try await channel.finish()
    }

    func testCompletionsStreamingChunkShapeAndDone() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(
                text: ["hel", "lo"],
                finishReason: .length,
                promptTokens: 3,
                completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "text/event-stream")
        // No role announcement — the legacy shape has no `delta` wrapper at all.
        XCTAssertFalse(response.body.contains(#""role""#), response.body)
        let events = try sseJSONEvents(from: response.body)
        XCTAssertTrue(events.allSatisfy { $0["object"] as? String == "text_completion" })
        // Every streamed chunk (including the terminal one) carries the `cmpl-` prefixed id.
        XCTAssertTrue(events.allSatisfy { $0["id"] as? String == "cmpl-1" }, response.body)
        let texts = events.compactMap { event -> String? in
            (event["choices"] as? [[String: Any]])?.first?["text"] as? String
        }
        XCTAssertEqual(texts, ["hel", "lo", ""])
        let finishReasons = events.compactMap { event -> String? in
            (event["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String
        }
        XCTAssertEqual(finishReasons, ["length"])
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // The legacy `text_completion` shape has no field to carry a tool call. A backend emitting one
    // on this route must be treated as an invalid backend handle — never silently dropped, and
    // never surfaced as a fabricated `finish_reason:"tool_calls"` on a shape that cannot represent it.
    func testCompletionsNonStreamingBackendToolCallEmissionIsInvalidBackendHandle() async throws {
        let backend = ScriptedBackend(scripts: [
            .completedWithToolCalls(
                text: ["partial"],
                toolCalls: [
                    OpenAIToolCall(id: "call_1", function: .init(name: "get_weather", arguments: "{}"))
                ],
                finishReason: .toolCalls,
                promptTokens: 3,
                completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .internalServerError)
        XCTAssertTrue(response.body.contains(#""code":"generation_failed""#), response.body)
        XCTAssertFalse(response.body.contains(#""tool_calls""#), response.body)
        XCTAssertFalse(response.body.contains(#""finish_reason":"tool_calls""#), response.body)

        _ = try await channel.finish()
    }

    // Same contract on the streaming path: a tool-call event mid-stream on `/v1/completions` must
    // not be downgraded into chat-shaped delta chunks — it fails the generation instead.
    func testCompletionsStreamingBackendToolCallEmissionIsInvalidBackendHandle() async throws {
        let backend = ScriptedBackend(scripts: [
            .completedWithToolCalls(
                text: ["partial"],
                toolCalls: [
                    OpenAIToolCall(id: "call_1", function: .init(name: "get_weather", arguments: "{}"))
                ],
                finishReason: .toolCalls,
                promptTokens: 3,
                completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: true))
        let response = try await collectResponse(from: channel)

        // The SSE head is already sent by the time the tool-call event arrives, so the failure is
        // reported as a terminal SSE error event rather than a fresh HTTP status.
        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains(#""code":"generation_failed""#), response.body)
        XCTAssertFalse(response.body.contains(#""tool_calls""#), response.body)

        _ = try await channel.finish()
    }

    func testCompletionsStreamOptionsIncludeUsageEmitsTerminalUsageChunkBeforeDone() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], promptTokens: 3, completionTokens: 2)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(
            channel,
            body: """
            {"model":"qwen3-32b","prompt":"Hello","max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}
            """)
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let events = try sseJSONEvents(from: response.body)
        let usageOnlyEvents = events.filter { object in
            (object["choices"] as? [[String: Any]])?.isEmpty == true
        }
        XCTAssertEqual(usageOnlyEvents.count, 1, response.body)
        let usage = try XCTUnwrap(usageOnlyEvents.first?["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)

        _ = try await channel.finish()
    }

    // Same auth contract as chat: a configured bearer token is required on /v1/completions too.
    func testCompletionsRequiresBearerTokenExactlyLikeChat() async throws {
        let backend = ScriptedBackend(scripts: [])
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: "secret",
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .unauthorized)
        XCTAssertEqual(backend.snapshot().startCount, 0)

        _ = try await channel.finish()
    }

    // `validateHead` checks method against route membership with the exact same structure for
    // `/v1/completions` as for `/v1/chat/completions` (`isCompletions && head.method == .POST`
    // alongside `isChat && head.method == .POST`, in the same combined guard) — no separate method
    // check was added for the legacy route. This test pins that a `GET` on either route is refused
    // identically (405, same error body shape), proving the shared behavior rather than assuming it.
    func testCompletionsRejectsWrongMethodExactlyLikeChat() async throws {
        let completionsBackend = ScriptedBackend(scripts: [])
        let completionsChannel = try await makeChannel(backend: completionsBackend)
        try await writeHeadOnlyRequest(completionsChannel, method: .GET, uri: "/v1/completions")
        let completionsResponse = try await collectResponse(from: completionsChannel)
        XCTAssertEqual(completionsResponse.head.status, .methodNotAllowed)
        XCTAssertEqual(completionsBackend.snapshot().startCount, 0)
        _ = try await completionsChannel.finish()

        let chatBackend = ScriptedBackend(scripts: [])
        let chatChannel = try await makeChannel(backend: chatBackend)
        try await writeHeadOnlyRequest(chatChannel, method: .GET, uri: "/v1/chat/completions")
        let chatResponse = try await collectResponse(from: chatChannel)
        XCTAssertEqual(chatResponse.head.status, .methodNotAllowed)
        XCTAssertEqual(chatBackend.snapshot().startCount, 0)
        _ = try await chatChannel.finish()

        XCTAssertEqual(completionsResponse.body, chatResponse.body)
    }

    // `ServingEvidence.validateCanonicalHTTPRequest` only recognizes `/v1/chat/completions` as a
    // canonical path — building `ServingEvidence.Request` for `/v1/completions` while evidence is
    // configured previously threw inside that constructor with no response ever written, dropping
    // the connection. With evidence enabled, `/v1/completions` must instead get a clean 400
    // `completions_unsupported` BEFORE the backend is ever started, while `/v1/chat/completions` on
    // the SAME evidence-enabled configuration still succeeds exactly as before.
    func testCompletionsRefusesRequestWithoutDroppingConnectionWhenEvidenceIsEnabled() async throws {
        let recorder = ServingEvidenceRecorder()
        let evidenceConfiguration = ServingHTTPEvidenceConfiguration(
            snapshot: nil,
            record: { evidence in try await recorder.record(evidence) },
            reportFailure: { message in
                Task { await recorder.recordFailure(message) }
            })

        let completionsBackend = ScriptedBackend(scripts: [
            .completed(text: ["should-not-run"], promptTokens: 1, completionTokens: 1)
        ])
        let completionsChannel = try await makeChannel(
            backend: completionsBackend,
            configuration: defaultConfiguration(evidence: evidenceConfiguration))
        try await writeCompletionsRequest(completionsChannel, body: completionsRequestBody(stream: false))
        let completionsResponse = try await collectResponse(from: completionsChannel)

        XCTAssertEqual(completionsResponse.head.status, .badRequest)
        XCTAssertTrue(
            completionsResponse.body.contains(#""code":"completions_unsupported""#),
            completionsResponse.body)
        XCTAssertTrue(
            completionsResponse.body.contains(#""param":"prompt""#), completionsResponse.body)
        XCTAssertEqual(completionsBackend.snapshot().startCount, 0)
        _ = try await completionsChannel.finish()

        // The SAME evidence-enabled configuration still serves chat normally.
        let chatBackend = ScriptedBackend(scripts: [
            .completed(text: ["hi"], promptTokens: 1, completionTokens: 1)
        ])
        let chatChannel = try await makeChannel(
            backend: chatBackend,
            configuration: defaultConfiguration(evidence: evidenceConfiguration))
        try await writeRequest(chatChannel, body: requestBody(stream: false))
        let chatResponse = try await collectResponse(from: chatChannel)

        XCTAssertEqual(chatResponse.head.status, .ok)
        XCTAssertEqual(chatBackend.snapshot().startCount, 1)
        await waitUntil { await recorder.evidence.count == 1 }
        let recorded = await recorder.snapshot()
        XCTAssertEqual(recorded.evidence.first?.response.status, 200)
        XCTAssertTrue(recorded.failures.isEmpty)
        _ = try await chatChannel.finish()
    }

    // The backend must receive a request whose `promptInput == .rawText(<prompt>)` — proving the
    // route decodes through `OpenAICompletionRequest` and converts, rather than reusing the chat
    // decoder directly.
    func testCompletionsBackendReceivesRawTextPromptInput() async throws {
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["ok"], promptTokens: 1, completionTokens: 1)
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(
            channel, body: completionsRequestBody(prompt: "Once upon a time", stream: false))
        _ = try await collectResponse(from: channel)

        XCTAssertEqual(
            backend.snapshot().lastRequest?.promptInput, .rawText("Once upon a time"))

        _ = try await channel.finish()
    }

    // A backend that cannot serve this route (e.g. a chat-template-only backend) throws
    // `completions_unsupported`, which must map to HTTP 400 exactly like any other
    // `invalidRequestWithCode` — the shared error-mapping path, not a special case.
    func testCompletionsBackendCompletionsUnsupportedMapsToHTTP400() async throws {
        let backend = ScriptedBackend(scripts: [
            .servingError(
                .invalidRequestWithCode(
                    "This server does not support the legacy text-completions route for the loaded backend",
                    param: "prompt",
                    code: "completions_unsupported"))
        ])
        let channel = try await makeChannel(backend: backend)

        try await writeCompletionsRequest(channel, body: completionsRequestBody(stream: false))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .badRequest)
        XCTAssertTrue(response.body.contains(#""code":"completions_unsupported""#), response.body)
        XCTAssertTrue(response.body.contains(#""param":"prompt""#), response.body)

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

    // MARK: - GET /v1/models/{id}

    func testModelDetailEndpointReturnsModelObjectForServedId() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/v1/models/qwen3-32b")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        XCTAssertEqual(object["id"] as? String, "qwen3-32b")
        XCTAssertEqual(object["object"] as? String, "model")
        XCTAssertNotNil(object["created"])
        XCTAssertEqual(object["owned_by"] as? String, "fast-mlx")
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    // Model ids may contain `/` (e.g. `org/name`); a client may percent-encode the path segment.
    func testModelDetailEndpointDecodesPercentEncodedSlashInModelId() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "org/name",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/v1/models/org%2Fname")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        XCTAssertEqual(object["id"] as? String, "org/name")
        _ = try await channel.finish()
    }

    func testModelDetailEndpointIgnoresQueryStringInModelId() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "org/name",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/v1/models/org%2Fname?probe=1")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        XCTAssertEqual(object["id"] as? String, "org/name")
        _ = try await channel.finish()
    }

    func testModelDetailEndpointReturns404ForUnknownModelId() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/v1/models/does-not-exist")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .notFound)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let errorObject = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error")
        XCTAssertEqual(errorObject["code"] as? String, "model_not_found")
        XCTAssertEqual(
            errorObject["message"] as? String,
            "The model 'does-not-exist' does not exist")
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    func testModelDetailEndpointRequiresAuthWhenConfigured() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: "secret",
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))

        let unauthorizedBackend = ScriptedBackend(scripts: [])
        let unauthorizedChannel = try await makeChannel(
            backend: unauthorizedBackend, configuration: configuration)
        try await writeHeadOnlyRequest(unauthorizedChannel, method: .GET, uri: "/v1/models/qwen3-32b")
        let unauthorized = try await collectResponse(from: unauthorizedChannel)
        XCTAssertEqual(unauthorized.head.status, .unauthorized)
        _ = try await unauthorizedChannel.finish()

        let authorizedBackend = ScriptedBackend(scripts: [])
        let authorizedChannel = try await makeChannel(
            backend: authorizedBackend, configuration: configuration)
        try await writeHeadOnlyRequest(
            authorizedChannel,
            method: .GET,
            uri: "/v1/models/qwen3-32b",
            authorization: "Bearer secret")
        let authorized = try await collectResponse(from: authorizedChannel)
        XCTAssertEqual(authorized.head.status, .ok)
        _ = try await authorizedChannel.finish()
    }

    // Same method handling as `/v1/models`: a non-GET is rejected before any backend admission.
    func testModelDetailEndpointRejectsWrongMethod() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend)
        try await writeHeadOnlyRequest(channel, method: .POST, uri: "/v1/models/qwen3-32b")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .methodNotAllowed)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    func testModelDetailEndpointRejectsRequestBody() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend)
        try await writeHeadWithBodyRequest(
            channel, method: .GET, uri: "/v1/models/qwen3-32b", body: "{}")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .badRequest)
        _ = try await channel.finish()
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
        // The evidence-derived gauge block is still fully deterministic -- asserted as an exact
        // prefix. It is no longer the WHOLE body: the always-on `/metrics` HTTP dependability
        // series (`ServingHTTPMetricsRecorder`, fed by every finished request on this shared
        // `configuration`, including the 401 rejection above) is appended after it, and that tail
        // embeds a real wall-clock request duration -- asserting the full body via `==` would pin
        // a nondeterministic float and make this test flaky. See the dedicated
        // `ServingHTTPMetricsTests` unit suite and the `/metrics`-specific end-to-end tests below
        // for exact, deterministic coverage of that series' shape.
        let expectedEvidenceGaugeBlock = """
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

            """
        XCTAssertTrue(
            response.body.hasPrefix(expectedEvidenceGaugeBlock),
            "evidence gauge block prefix changed:\n\(response.body)")
        // The 401-rejected scrape above (same shared `configuration`) is the only prior request,
        // so it is the sole observation backing these always-on HTTP series.
        XCTAssertTrue(
            response.body.contains(
                "# HELP fastmlx_http_requests_total Cumulative count of finished HTTP requests, "
                    + "labeled by templated route, response status class, and outcome.\n"
                    + "# TYPE fastmlx_http_requests_total counter\n"))
        XCTAssertTrue(
            response.body.contains(
                #"fastmlx_http_requests_total{route="/metrics",status_class="4xx",outcome="error"} 1"#))
        XCTAssertTrue(
            response.body.contains(
                #"fastmlx_http_request_duration_seconds_bucket{route="/metrics",le="+Inf"} 1"#))
        XCTAssertTrue(
            response.body.contains(
                #"fastmlx_http_request_duration_seconds_count{route="/metrics"} 1"#))
        XCTAssertTrue(
            response.body.contains(#"fastmlx_http_request_duration_seconds_sum{route="/metrics"} "#),
            "a _sum line must exist for the route, even though its real wall-clock value isn't pinned")
        XCTAssertTrue(
            response.body.hasSuffix(
                "# HELP fastmlx_http_time_to_first_token_seconds Time to first streamed token in "
                    + "seconds, for streaming chat/completions successes only, labeled by templated "
                    + "route.\n# TYPE fastmlx_http_time_to_first_token_seconds histogram\n"),
            "the 401 rejection never produced a TTFT, so that family stays header-only")
        XCTAssertEqual(
            response.head.headers.first(name: "content-length"),
            "\(response.body.utf8.count)")
        XCTAssertFalse(response.body.contains("qwen3-32b"))
        XCTAssertFalse(response.body.contains("secret"))
        XCTAssertEqual(authorizedBackend.snapshot().startCount, 0)
        _ = try await authorizedChannel.finish()
    }

    /// A present `speculativeDecoding` block — even all-zero — must render as `0`, not be dropped.
    /// This is the specific bug `appendOptionalMetric`'s early-return-on-nil would reintroduce (see
    /// the sibling absent-case test below; together the two prove `0` and "absent" are
    /// distinguishable on the wire, which is the entire point of this increment).
    func testMetricsEndpointRendersPresentZeroSpeculativeDecodingCountersNotOmitted() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                verifyRounds: 0))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(
            response.body.contains("fastmlx_mtp_proposed_draft_tokens_total 0"),
            "an all-zero present block must still render the metric line, not be omitted")
        XCTAssertTrue(response.body.contains("fastmlx_mtp_accepted_draft_tokens_total 0"))
        XCTAssertTrue(response.body.contains("fastmlx_mtp_verify_rounds_total 0"))
        XCTAssertTrue(response.body.contains("# TYPE fastmlx_mtp_proposed_draft_tokens_total counter"))
        _ = try await channel.finish()
    }

    /// A `nil` `speculativeDecoding` (no drafter bound) must render NO `fastmlx_mtp_` substring at
    /// all. Alone this test is inert (it would also pass if the feature were never implemented);
    /// paired with the present-zero test above it proves the renderer distinguishes "absent" from
    /// "present with 0", which a `appendOptionalMetric`-style early-return-on-nil route cannot do
    /// once the block itself starts routing through it (see the mutation check in the increment
    /// report).
    func testMetricsEndpointOmitsSpeculativeDecodingMetricsEntirelyWhenNoDrafterBound() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0)
        XCTAssertNil(snapshot.speculativeDecoding)
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertFalse(response.body.contains("fastmlx_mtp_"))
        _ = try await channel.finish()
    }

    /// Nonzero counters render their real values and the `# TYPE` line says `counter`, matching the
    /// Prometheus convention for a monotonic total (never `gauge`, which `appendMetric` emits).
    func testMetricsEndpointRendersNonzeroSpeculativeDecodingCountersAsCounterType() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 17,
                acceptedDraftTokens: 11,
                verifyRounds: 5))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains("fastmlx_mtp_proposed_draft_tokens_total 17"))
        XCTAssertTrue(response.body.contains("fastmlx_mtp_accepted_draft_tokens_total 11"))
        XCTAssertTrue(response.body.contains("fastmlx_mtp_verify_rounds_total 5"))
        XCTAssertTrue(response.body.contains("# TYPE fastmlx_mtp_accepted_draft_tokens_total counter"))
        XCTAssertFalse(response.body.contains("# TYPE fastmlx_mtp_accepted_draft_tokens_total gauge"))
        _ = try await channel.finish()
    }

    /// `passthroughActive: true` renders `fastmlx_mtp_passthrough_active 1` as a `gauge` (not a
    /// `counter` — it is a 0/1 state, not a monotonic total). This is the case an operator needs:
    /// MTP bound but sticky passthrough means it will never propose another draft token this serve.
    func testMetricsEndpointRendersPassthroughActiveAsGaugeOne() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                verifyRounds: 0,
                passthroughActive: true))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        // Exact LINE match, never `contains`: the rendered `# HELP <name> <help>` line begins with
        // the metric name, so a substring check for "<name> 1" can be satisfied by the help text
        // alone and would pass even if the sampled value were 0.
        let lines = response.body.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        XCTAssertTrue(
            lines.contains("fastmlx_mtp_passthrough_active 1"),
            "expected an exact `fastmlx_mtp_passthrough_active 1` line; body was:\n\(response.body)")
        XCTAssertFalse(lines.contains("fastmlx_mtp_passthrough_active 0"))
        XCTAssertTrue(lines.contains("# TYPE fastmlx_mtp_passthrough_active gauge"))
        _ = try await channel.finish()
    }

    /// `passthroughActive: false` must still render `fastmlx_mtp_passthrough_active 0` — proving
    /// the falsy value is NOT omitted, which is the entire point of routing this field through
    /// `appendMetric` directly rather than `appendOptionalMetric`.
    func testMetricsEndpointRendersPassthroughActiveAsGaugeZeroNotOmitted() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                verifyRounds: 0,
                passthroughActive: false))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        // Same exact-line discipline as the `1` case above.
        let lines = response.body.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        XCTAssertTrue(
            lines.contains("fastmlx_mtp_passthrough_active 0"),
            "expected an exact `fastmlx_mtp_passthrough_active 0` line — a falsy reading must "
                + "render, not be omitted; body was:\n\(response.body)")
        XCTAssertFalse(lines.contains("fastmlx_mtp_passthrough_active 1"))
        XCTAssertTrue(lines.contains("# TYPE fastmlx_mtp_passthrough_active gauge"))
        _ = try await channel.finish()
    }

    /// `fastmlx_mtp_speculative_requests_total`/`fastmlx_mtp_passthrough_requests_total` are the
    /// per-request, non-latching counterpart to the sticky `fastmlx_mtp_passthrough_active` gauge
    /// above — an operator computes a genuine passthrough RATE from these two, which the gauge
    /// alone cannot provide once it has latched to `1`. Exact-line assertions, same discipline as
    /// the gauge tests: a `contains` check on the metric name alone would also match its own HELP
    /// line.
    func testMetricsEndpointRendersPerRequestSpeculativeAndPassthroughCountersAsExactCounterLines()
        async throws
    {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                verifyRounds: 0,
                passthroughActive: true,
                speculativeRequestCount: 4,
                passthroughRequestCount: 1))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let lines = response.body.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        XCTAssertTrue(
            lines.contains("fastmlx_mtp_speculative_requests_total 4"),
            "expected an exact `fastmlx_mtp_speculative_requests_total 4` line; body was:\n\(response.body)")
        XCTAssertTrue(
            lines.contains("fastmlx_mtp_passthrough_requests_total 1"),
            "expected an exact `fastmlx_mtp_passthrough_requests_total 1` line; body was:\n\(response.body)")
        XCTAssertTrue(lines.contains("# TYPE fastmlx_mtp_speculative_requests_total counter"))
        XCTAssertTrue(lines.contains("# TYPE fastmlx_mtp_passthrough_requests_total counter"))
        XCTAssertFalse(lines.contains("# TYPE fastmlx_mtp_speculative_requests_total gauge"))
        XCTAssertFalse(lines.contains("# TYPE fastmlx_mtp_passthrough_requests_total gauge"))
        // The gauge above is latched (`passthroughActive: true`) even though this fixture reports
        // 4 requests that genuinely speculated — proving the two rendered metric families are
        // independent, exactly the point of this increment.
        XCTAssertTrue(lines.contains("fastmlx_mtp_passthrough_active 1"))
        _ = try await channel.finish()
    }

    /// The `fastmlx_mtp_passthrough_active` HELP text previously asserted a FALSE invariant — that
    /// the gauge going to `1` meant the decoder was "no longer proposing draft tokens". It is
    /// sticky/latching, not a live state: a later request on the same decoder can, and does,
    /// speculate again. This test pins the corrected HELP line's absence of that false phrasing so
    /// a future edit cannot silently reintroduce it, without weakening any existing exact-line
    /// gauge-VALUE assertion above (those assert on `fastmlx_mtp_passthrough_active <n>`, not on
    /// the HELP text, and are left untouched).
    func testMetricsEndpointPassthroughActiveHelpTextDoesNotClaimSpeculationHasStopped() async throws {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0,
            speculativeDecoding: try ServingEvidence.SpeculativeDecodingCounters(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                verifyRounds: 0,
                passthroughActive: true))
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

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)
        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let helpLine = try XCTUnwrap(
            response.body.split(separator: "\n", omittingEmptySubsequences: false)
                .map(String.init)
                .first { $0.hasPrefix("# HELP fastmlx_mtp_passthrough_active ") },
            "expected a HELP line for fastmlx_mtp_passthrough_active; body was:\n\(response.body)")
        XCTAssertFalse(
            helpLine.contains("no longer proposing"),
            "HELP text must not claim the decoder has stopped proposing draft tokens — it is a "
                + "sticky latch, not a live state; line was: \(helpLine)")
        XCTAssertTrue(
            helpLine.contains("fastmlx_mtp_speculative_requests_total")
                || helpLine.contains("fastmlx_mtp_passthrough_requests_total"),
            "HELP text should point an operator at the per-request counters for a current rate; "
                + "line was: \(helpLine)")
        _ = try await channel.finish()
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

    // Proves the gate-2 fix: a scalar (or exact-MTP) serve has NO `--evidence` sink, so
    // `configuration.evidence` is `nil` — but `configuration.metricsSnapshot` (populated
    // unconditionally by FastMLXServe from `PreparedServingBackend.evidenceSnapshot`) must still
    // let `/metrics` render a real Prometheus body instead of `metrics_unavailable`.
    func testMetricsEndpointWithMetricsSnapshotButNoEvidenceRendersPrometheusText()
        async throws
    {
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 1,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 8_192)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            metricsSnapshot: { snapshot })

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            channel,
            method: .GET,
            uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(
            response.head.headers.first(name: "content-type"),
            "text/plain; version=0.0.4; charset=utf-8")
        XCTAssertTrue(response.body.contains("fastmlx_up 1"), response.body)
        XCTAssertTrue(response.body.contains("fastmlx_active_requests 1"), response.body)
        XCTAssertFalse(response.body.contains("metrics_unavailable"), response.body)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    // Regression guard: when BOTH `evidence` and `metricsSnapshot` are absent, `/metrics` must
    // still fail closed with `metrics_unavailable` — the fix above must not make the guard
    // unconditionally true.
    func testMetricsEndpointWithoutEvidenceOrMetricsSnapshotStillReturnsUnavailable()
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
        XCTAssertEqual(
            response.body,
            """
            {"error":{"code":"metrics_unavailable","message":"Metrics snapshot is not configured","param":null,"type":"server_error"}}
            """)
        _ = try await channel.finish()
    }

    // Regression guard: when `evidence.snapshot` IS present it still takes precedence over
    // `metricsSnapshot` (the continuous route only ever supplies `evidence`, never
    // `metricsSnapshot`, but this locks the precedence order even if both were ever set).
    func testMetricsEndpointPrefersEvidenceSnapshotOverMetricsSnapshot()
        async throws
    {
        let evidenceSnapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 9,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0)
        let metricsOnlySnapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 3,
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
                snapshot: { evidenceSnapshot },
                record: { _ in },
                reportFailure: { _ in }),
            metricsSnapshot: { metricsOnlySnapshot })

        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: configuration)
        try await writeHeadOnlyRequest(
            channel,
            method: .GET,
            uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains("fastmlx_active_requests 9"), response.body)
        XCTAssertFalse(response.body.contains("fastmlx_active_requests 3"), response.body)
        _ = try await channel.finish()
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

    // MARK: - Always-on `/metrics` HTTP dependability series (`ServingHTTPMetricsRecorder`)

    // A scrape taken before any other traffic must still declare all three families' shape (`#
    // HELP`/`# TYPE` headers), with zero sample lines -- Prometheus' own guidance that a metric
    // family's shape shouldn't depend on whether it has been populated yet.
    func testMetricsEndpointRendersHTTPRequestMetricsHeadersWithNoSamplesWhenRecorderIsEmpty()
        async throws
    {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            metricsSnapshot: { try emptyResourceSnapshot() })
        let channel = try await makeChannel(backend: ScriptedBackend(scripts: []), configuration: configuration)

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/metrics")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertTrue(response.body.contains("# TYPE fastmlx_http_requests_total counter"))
        XCTAssertTrue(response.body.contains("# TYPE fastmlx_http_request_duration_seconds histogram"))
        XCTAssertTrue(response.body.contains("# TYPE fastmlx_http_time_to_first_token_seconds histogram"))
        XCTAssertFalse(response.body.contains("fastmlx_http_requests_total{"), "no traffic yet: no sample")
        XCTAssertFalse(response.body.contains("fastmlx_http_request_duration_seconds_bucket{"))
        XCTAssertFalse(response.body.contains("fastmlx_http_time_to_first_token_seconds_bucket{"))
        _ = try await channel.finish()
    }

    // The core Lane 2 acceptance criterion: an operator scraping `/metrics` sees the HTTP
    // dependability counters even though `--request-log` was never turned on (the default,
    // `configuration.requestLog == nil`) -- these series are a baseline signal, not an opt-in
    // diagnostic.
    func testMetricsEndpointRecordsHTTPRequestCounterIndependentlyOfRequestLogSetting() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            metricsSnapshot: { try emptyResourceSnapshot() })
        XCTAssertNil(configuration.requestLog, "must stay off to prove metrics are independent of it")

        let requestChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(requestChannel, method: .GET, uri: "/no/such/route")
        let notFound = try await collectResponse(from: requestChannel)
        XCTAssertEqual(notFound.head.status, .notFound)
        _ = try await requestChannel.finish()

        let scrapeChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(scrapeChannel, method: .GET, uri: "/metrics")
        let scrape = try await collectResponse(from: scrapeChannel)

        XCTAssertEqual(scrape.head.status, .ok)
        XCTAssertTrue(
            scrape.body.contains(
                #"fastmlx_http_requests_total{route="other",status_class="4xx",outcome="error"} 1"#))
        _ = try await scrapeChannel.finish()
    }

    // Deterministic, non-timing-dependent slice of the histogram render shape for one series: the
    // `+Inf` bucket and `_count` always equal the observation count by construction regardless of
    // wall-clock duration, so both are asserted exactly; `_sum` embeds a real duration and is only
    // checked for presence (see the dedicated `ServingHTTPMetricsTests` unit suite for exact,
    // synthetic-value histogram/bucket-boundary coverage).
    func testMetricsEndpointRendersExactHTTPRequestDurationHistogramLinesForOneRoute() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            metricsSnapshot: { try emptyResourceSnapshot() })

        let requestChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(requestChannel, method: .GET, uri: "/v1/models/qwen3-32b")
        let modelResponse = try await collectResponse(from: requestChannel)
        XCTAssertEqual(modelResponse.head.status, .ok)
        _ = try await requestChannel.finish()

        let scrapeChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(scrapeChannel, method: .GET, uri: "/metrics")
        let scrape = try await collectResponse(from: scrapeChannel)
        let lines = scrape.body.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

        XCTAssertTrue(
            lines.contains(
                #"fastmlx_http_requests_total{route="/v1/models/{id}",status_class="2xx",outcome="completed"} 1"#))
        XCTAssertTrue(
            lines.contains(
                #"fastmlx_http_request_duration_seconds_bucket{route="/v1/models/{id}",le="+Inf"} 1"#))
        XCTAssertTrue(
            lines.contains(#"fastmlx_http_request_duration_seconds_count{route="/v1/models/{id}"} 1"#))
        XCTAssertTrue(
            lines.contains {
                $0.hasPrefix(#"fastmlx_http_request_duration_seconds_sum{route="/v1/models/{id}"} "#)
            },
            "a _sum line must exist; the value itself is real wall-clock duration and isn't pinned")
        _ = try await scrapeChannel.finish()
    }

    // Both always-on paths fire for the same finished request when `--request-log` IS enabled:
    // the JSON access-log line (opt-in) and the `/metrics` counter (always-on) are independent
    // outputs of the same `servingEmitRequestLog` call, not alternatives.
    func testMetricsAndRequestLogBothRecordTheSameRequestWhenRequestLoggingIsEnabled() async throws {
        let recorder = RequestLogRecorder()
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            metricsSnapshot: { try emptyResourceSnapshot() },
            requestLog: recorder.sink())

        let requestChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(requestChannel, method: .GET, uri: "/no/such/route")
        let notFound = try await collectResponse(from: requestChannel)
        XCTAssertEqual(notFound.head.status, .notFound)
        await waitUntil { recorder.lines().count == 1 }
        _ = try await requestChannel.finish()

        let scrapeChannel = try await makeChannel(
            backend: ScriptedBackend(scripts: []), configuration: configuration)
        try await writeHeadOnlyRequest(scrapeChannel, method: .GET, uri: "/metrics")
        let scrape = try await collectResponse(from: scrapeChannel)
        XCTAssertTrue(
            scrape.body.contains(
                #"fastmlx_http_requests_total{route="other",status_class="4xx",outcome="error"} 1"#))

        let object = try requestLogJSONObject(try XCTUnwrap(recorder.lines().first))
        XCTAssertEqual(object["status"] as? Int, 404)
        XCTAssertEqual(object["outcome"] as? String, "error")
        _ = try await scrapeChannel.finish()
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

    // MARK: - Health / readiness probes

    func testHealthzReturnsOkWithExactBodyAndNoBearerTokenConfigured() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration())

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/healthz")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        XCTAssertEqual(response.body, #"{"status":"ok"}"#)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    func testHealthzBypassesBearerTokenWhileModelsStillRequiresIt() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: "secret",
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))

        // No Authorization header at all — an orchestrator health probe carries no API key.
        let healthzBackend = ScriptedBackend(scripts: [])
        let healthzChannel = try await makeChannel(
            backend: healthzBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(healthzChannel, method: .GET, uri: "/healthz")
        let healthzResponse = try await collectResponse(from: healthzChannel)
        XCTAssertEqual(healthzResponse.head.status, .ok)
        XCTAssertEqual(healthzResponse.body, #"{"status":"ok"}"#)
        _ = try await healthzChannel.finish()

        // Control: the same configuration still enforces the bearer token on every other route.
        let modelsBackend = ScriptedBackend(scripts: [])
        let modelsChannel = try await makeChannel(
            backend: modelsBackend,
            configuration: configuration)
        try await writeHeadOnlyRequest(modelsChannel, method: .GET, uri: "/v1/models")
        let modelsResponse = try await collectResponse(from: modelsChannel)
        XCTAssertEqual(modelsResponse.head.status, .unauthorized)
        _ = try await modelsChannel.finish()
    }

    func testReadyzReturnsReadyByDefault() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration())

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/readyz")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        XCTAssertEqual(response.body, #"{"status":"ready"}"#)
        XCTAssertEqual(backend.snapshot().startCount, 0)
        _ = try await channel.finish()
    }

    func testReadyzReturns503WhenReadinessHookReportsNotReady() async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            readiness: { false })
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/readyz")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .serviceUnavailable)
        XCTAssertEqual(response.body, #"{"status":"not_ready"}"#)
        _ = try await channel.finish()
    }

    func testPostHealthzReturns405() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration())

        try await writeHeadOnlyRequest(channel, method: .POST, uri: "/healthz")
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .methodNotAllowed)
        _ = try await channel.finish()
    }

    func testHealthzTrailingSlashIsAnUnknownRouteLikeOtherUnmatchedURIs() async throws {
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration())

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/healthz/")
        let response = try await collectResponse(from: channel)

        // `validateHead` matches routes by exact `head.uri ==` comparison (see `/v1/models` and
        // `/metrics` above), so a trailing-slash variant is an unmatched URI and 404s exactly like
        // any other unknown route — pinning that behavior here rather than inventing new
        // normalization for probes only.
        XCTAssertEqual(response.head.status, .notFound)
        _ = try await channel.finish()
    }

    // MARK: - Logprobs

    func testChatNonStreamingResponseIncludesLogprobsForEveryGeneratedTokenInOrder() async throws {
        let tokenA = ServingTokenLogprob(
            tokenText: "Hi",
            logprob: -0.05,
            topCandidates: [
                ServingTokenLogprobCandidate(tokenText: "Hi", logprob: -0.05),
                ServingTokenLogprobCandidate(tokenText: "Hey", logprob: -3.0),
            ])
        let tokenB = ServingTokenLogprob(tokenText: " there", logprob: -0.2)
        let backend = LogprobsScriptedBackend(
            deltas: [.tokenLogprobs([tokenA]), .text("Hi"), .tokenLogprobs([tokenB]), .text(" there")])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        let body = """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":false,"logprobs":true,"top_logprobs":2}
            """
        try await writeRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let logprobs = try XCTUnwrap(choices.first?["logprobs"] as? [String: Any])
        XCTAssertTrue(logprobs["refusal"] is NSNull)
        let content = try XCTUnwrap(logprobs["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 2)
        XCTAssertEqual(content[0]["token"] as? String, "Hi")
        XCTAssertEqual(content[0]["logprob"] as? Double, -0.05)
        XCTAssertEqual((content[0]["top_logprobs"] as? [[String: Any]])?.count, 2)
        XCTAssertEqual(content[1]["token"] as? String, " there")
        _ = try await channel.finish()
    }

    // Contract: `logprobs:false`/absent must leave every existing response BYTE-IDENTICAL to
    // before this feature existed — no `logprobs` key on the chat choice at all.
    func testChatResponseWithoutLogprobsRequestOmitsLogprobsKeyEntirely() async throws {
        let backend = ScriptedBackend(
            scripts: [.completed(text: ["Hello"], promptTokens: 3, completionTokens: 1)])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertFalse(choices.first?.keys.contains("logprobs") ?? true)
        _ = try await channel.finish()
    }

    // The streaming "carry to the next emitted chunk" contract: a `.tokenLogprobs` delta with no
    // following `.text` before completion must not be lost — it is flushed on the finish chunk.
    // Concatenating `logprobs.content` across every chunk this stream emits must cover exactly the
    // three generated tokens, in order, each exactly once.
    func testChatStreamingLogprobsCarryToNextChunkAndFinalChunkFlushesRemainder() async throws {
        let tokenA = ServingTokenLogprob(tokenText: "A", logprob: -0.1)
        let tokenB = ServingTokenLogprob(tokenText: "B", logprob: -0.2)
        let tokenC = ServingTokenLogprob(tokenText: "C", logprob: -0.3)
        let backend = LogprobsScriptedBackend(
            deltas: [
                .tokenLogprobs([tokenA]), .tokenLogprobs([tokenB]), .text("AB"),
                .tokenLogprobs([tokenC]),
            ])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        let body = """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":true,"logprobs":true,"top_logprobs":0}
            """
        try await writeRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        let events = try sseJSONEvents(from: response.body)

        // Role-announcement chunk carries no logprobs at all.
        let roleChoice = try XCTUnwrap((events.first?["choices"] as? [[String: Any]])?.first)
        XCTAssertFalse(roleChoice.keys.contains("logprobs"))

        var seenTokens: [String] = []
        for event in events.dropFirst() {
            guard let choice = (event["choices"] as? [[String: Any]])?.first,
                let logprobs = choice["logprobs"] as? [String: Any],
                let content = logprobs["content"] as? [[String: Any]]
            else { continue }
            seenTokens.append(contentsOf: content.compactMap { $0["token"] as? String })
        }
        XCTAssertEqual(seenTokens, ["A", "B", "C"])
        _ = try await channel.finish()
    }

    func testCompletionsNonStreamingResponseIncludesLogprobsWithTextOffsets() async throws {
        let tokenA = ServingTokenLogprob(tokenText: "a", logprob: -0.1)
        let tokenB = ServingTokenLogprob(tokenText: "b", logprob: -0.2)
        let backend = LogprobsScriptedBackend(deltas: [.tokenLogprobs([tokenA, tokenB]), .text("ab")])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        let body = """
            {"model":"qwen3-32b","prompt":"Hello","max_tokens":8,"temperature":0,"stream":false,"logprobs":2}
            """
        try await writeCompletionsRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let logprobs = try XCTUnwrap(choices.first?["logprobs"] as? [String: Any])
        XCTAssertEqual(logprobs["tokens"] as? [String], ["a", "b"])
        XCTAssertEqual(logprobs["token_logprobs"] as? [Double], [-0.1, -0.2])
        XCTAssertEqual(logprobs["text_offset"] as? [Int], [0, 1])
        _ = try await channel.finish()
    }

    // `logprobs:0` (legacy completions) means "sampled token's logprob only" — a MEANINGFUL,
    // distinct request, never "off" — but `top_logprobs[i]` must still be `null` for every token.
    func testCompletionsLogprobsZeroForcesNullTopLogprobsEntries() async throws {
        let token = ServingTokenLogprob(
            tokenText: "a",
            logprob: -0.1,
            topCandidates: [ServingTokenLogprobCandidate(tokenText: "a", logprob: -0.1)])
        let backend = LogprobsScriptedBackend(deltas: [.tokenLogprobs([token]), .text("a")])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        let body = """
            {"model":"qwen3-32b","prompt":"Hello","max_tokens":8,"temperature":0,"stream":false,"logprobs":0}
            """
        try await writeCompletionsRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let logprobs = try XCTUnwrap(choices.first?["logprobs"] as? [String: Any])
        let topLogprobs = try XCTUnwrap(logprobs["top_logprobs"] as? [Any])
        XCTAssertEqual(topLogprobs.count, 1)
        XCTAssertTrue(topLogprobs[0] is NSNull)
        _ = try await channel.finish()
    }

    // The running `text_offset` must continue across streaming chunks, not reset to 0 each time.
    func testCompletionsStreamingLogprobsTextOffsetContinuesAcrossChunks() async throws {
        let tokenA = ServingTokenLogprob(tokenText: "Hel", logprob: -0.1)
        let tokenB = ServingTokenLogprob(tokenText: "lo", logprob: -0.2)
        let backend = LogprobsScriptedBackend(
            deltas: [.tokenLogprobs([tokenA]), .text("Hel"), .tokenLogprobs([tokenB]), .text("lo")])
        let channel = try await makeChannel(backend: backend, configuration: defaultConfiguration())

        let body = """
            {"model":"qwen3-32b","prompt":"Hello","max_tokens":8,"temperature":0,"stream":true,"logprobs":1}
            """
        try await writeCompletionsRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        let events = try sseJSONEvents(from: response.body)

        var offsets: [Int] = []
        for event in events {
            guard let choice = (event["choices"] as? [[String: Any]])?.first,
                let logprobs = choice["logprobs"] as? [String: Any],
                let textOffset = logprobs["text_offset"] as? [Int],
                !textOffset.isEmpty
            else { continue }
            offsets.append(contentsOf: textOffset)
        }
        XCTAssertEqual(offsets, [0, 3])
        _ = try await channel.finish()
    }

    // Serving evidence recording has no way to serialize per-token logprobs — a logprobs request
    // must be rejected with a normal 400 (never a dropped connection) while evidence is configured.
    func testLogprobsRequestIsRejectedWhenEvidenceRecordingIsConfigured() async throws {
        let recorder = ServingEvidenceRecorder()
        let evidenceConfiguration = ServingHTTPEvidenceConfiguration(
            snapshot: nil,
            record: { evidence in try await recorder.record(evidence) },
            reportFailure: { message in
                Task { await recorder.recordFailure(message) }
            })
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration(evidence: evidenceConfiguration))

        let body = """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":false,"logprobs":true}
            """
        try await writeRequest(channel, body: body)
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .badRequest)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let error = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? String, "logprobs_unsupported")
        _ = try await channel.finish()
    }

    // MARK: - Request log (`--request-log json` / `ServingHTTPConfiguration.requestLog`)

    func testRequestLogChatNonStreamingSuccessEmitsExactlyOneLineWithNoContent() async throws {
        let promptSentinel = "PROMPT-SENTINEL-requestlog"
        let generatedSentinel = "GENERATED-SENTINEL-requestlog"
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [
            .completed(text: [generatedSentinel], promptTokens: 3, completionTokens: 2)
        ])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))
        let body = """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"\(promptSentinel)"}],"max_completion_tokens":8,"temperature":0,"stream":false}
            """

        try await writeRequest(channel, body: body)
        _ = try await collectResponse(from: channel)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["method"] as? String, "POST")
        XCTAssertEqual(object["route"] as? String, "/v1/chat/completions")
        XCTAssertEqual(object["status"] as? Int, 200)
        XCTAssertEqual(object["outcome"] as? String, "completed")
        XCTAssertEqual(object["stream"] as? Bool, false)
        XCTAssertEqual(object["model"] as? String, "qwen3-32b")
        XCTAssertEqual(object["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(object["completion_tokens"] as? Int, 2)
        XCTAssertEqual(object["finish_reason"] as? String, "stop")
        XCTAssertNotNil(object["request_id"] as? String)
        XCTAssertNotNil(object["duration_ms"] as? Double)
        XCTAssertNotNil(object["ts"] as? String)
        // Never message/prompt content or the model's generated text.
        XCTAssertFalse(line.contains(promptSentinel))
        XCTAssertFalse(line.contains(generatedSentinel))
        _ = try await channel.finish()
    }

    func testRequestLogStreamingSuccessEmitsExactlyOneLineWithTTFT() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["hel", "lo"], finishReason: .length, promptTokens: 3, completionTokens: 2)
        ])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))

        try await writeRequest(channel, body: requestBody(stream: true))
        _ = try await collectResponse(from: channel)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["route"] as? String, "/v1/chat/completions")
        XCTAssertEqual(object["status"] as? Int, 200)
        XCTAssertEqual(object["outcome"] as? String, "completed")
        XCTAssertEqual(object["stream"] as? Bool, true)
        XCTAssertEqual(object["finish_reason"] as? String, "length")
        let ttft = try XCTUnwrap(object["ttft_ms"] as? Double)
        XCTAssertGreaterThanOrEqual(ttft, 0)
        _ = try await channel.finish()
    }

    func testRequestLog400InvalidRequestEmitsExactlyOneErrorLine() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))

        try await writeRequest(channel, body: "not json")
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .badRequest)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["status"] as? Int, 400)
        XCTAssertEqual(object["outcome"] as? String, "error")
        XCTAssertNotNil(object["error_code"] as? String)
        _ = try await channel.finish()
    }

    // A wrong/missing bearer token must log status 401 with no key material anywhere in the line.
    func testRequestLog401AuthFailureEmitsLineWithNoKeyMaterial() async throws {
        let apiKeySentinel = "sk-API-KEY-SENTINEL-requestlog"
        let wrongKeySentinel = "sk-WRONG-KEY-SENTINEL-requestlog"
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [])
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: apiKeySentinel,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1),
            requestLog: recorder.sink())
        let channel = try await makeChannel(backend: backend, configuration: configuration)

        try await writeRequest(
            channel,
            body: requestBody(stream: false),
            authorization: "Bearer \(wrongKeySentinel)")
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .unauthorized)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["status"] as? Int, 401)
        XCTAssertEqual(object["outcome"] as? String, "error")
        XCTAssertFalse(line.contains(apiKeySentinel))
        XCTAssertFalse(line.contains(wrongKeySentinel))
        XCTAssertFalse(line.lowercased().contains("bearer"))
        XCTAssertFalse(line.lowercased().contains("authorization"))
        _ = try await channel.finish()
    }

    func testRequestLog404UnknownRouteEmitsExactlyOneLine() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/no/such/route-zq7secret")
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .notFound)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["method"] as? String, "GET")
        XCTAssertEqual(object["route"] as? String, "<unmatched>")
        XCTAssertFalse(line.contains("zq7secret"))
        XCTAssertEqual(object["status"] as? Int, 404)
        XCTAssertEqual(object["outcome"] as? String, "error")
        _ = try await channel.finish()
    }

    // `GET /v1/models/{id}` must log the TEMPLATED route, never the raw requested id.
    func testRequestLogModelDetailRouteIsTemplated() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))

        try await writeHeadOnlyRequest(channel, method: .GET, uri: "/v1/models/qwen3-32b")
        let response = try await collectResponse(from: channel)
        XCTAssertEqual(response.head.status, .ok)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertEqual(object["route"] as? String, "/v1/models/{id}")
        XCTAssertFalse(line.contains("/v1/models/qwen3-32b"))
        XCTAssertEqual(object["status"] as? Int, 200)
        XCTAssertEqual(object["outcome"] as? String, "completed")
        _ = try await channel.finish()
    }

    // An unrecognized top-level request field is accepted-but-ignored (see
    // `OpenAIChatCompletionRequest.ignoredFields`); the request-log line must surface its
    // sanitized `"unknown:<key>"` name.
    func testRequestLogIgnoredFieldsPresentForUnknownTopLevelKey() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["ok"], promptTokens: 1, completionTokens: 1)
        ])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))
        let body = """
            {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":false,"a_totally_unknown_field":true}
            """

        try await writeRequest(channel, body: body)
        _ = try await collectResponse(from: channel)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        let ignoredFields = try XCTUnwrap(object["ignored_fields"] as? [String])
        XCTAssertEqual(ignoredFields, ["unknown:a_totally_unknown_field"])
        _ = try await channel.finish()
    }

    // A successful request with NO ignored fields must omit the key entirely (never an empty
    // array), matching the serve flag's "omitted when empty" contract.
    func testRequestLogOmitsIgnoredFieldsKeyWhenEmpty() async throws {
        let recorder = RequestLogRecorder()
        let backend = ScriptedBackend(scripts: [
            .completed(text: ["ok"], promptTokens: 1, completionTokens: 1)
        ])
        let channel = try await makeChannel(
            backend: backend,
            configuration: requestLogConfiguration(sink: recorder.sink()))

        try await writeRequest(channel, body: requestBody(stream: false))
        _ = try await collectResponse(from: channel)
        await waitUntil { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try requestLogJSONObject(line)
        XCTAssertNil(object["ignored_fields"])
        _ = try await channel.finish()
    }
}

private struct CollectedResponse {
    let head: HTTPResponseHead
    let body: String
}

private func makeChannel(
    backend: any ServingGenerationBackend,
    configuration: ServingHTTPConfiguration = defaultConfiguration(),
    responseCompletionTestHook: (@Sendable () async -> Void)? = nil,
    finalWriteRaceTestHook: (@Sendable () async -> Void)? = nil
) async throws -> NIOAsyncTestingChannel {
    try await NIOAsyncTestingChannel { channel in
        try channel.pipeline.syncOperations.addHandler(
            OpenAIChatCompletionsHTTPHandler(
                configuration: configuration,
                backend: backend,
                responseCompletionTestHook: responseCompletionTestHook,
                finalWriteRaceTestHook: finalWriteRaceTestHook))
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

private func defaultConfiguration(
    requestFailureReporter: @escaping ServingHTTPEvidenceConfiguration.FailureReporter
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        requestFailureReporter: requestFailureReporter)
}

private func requestLogConfiguration(
    sink: @escaping @Sendable (String) -> Void
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        requestLog: sink)
}

private func requestLogJSONObject(_ line: String) throws -> [String: Any] {
    try XCTUnwrap(
        JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
}

private func requestBody(stream: Bool) -> String {
    """
    {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":\(stream)}
    """
}

/// A minimal all-zero snapshot for `/metrics` tests that only care about the always-on HTTP
/// dependability series (`ServingHTTPMetricsRecorder`), not the evidence-derived gauges -- avoids
/// each such test repeating the same seven-field literal.
private func emptyResourceSnapshot() throws -> ServingEvidence.ResourceSnapshot {
    try ServingEvidence.ResourceSnapshot(
        activeRequests: 0,
        coordinatorSlots: 0,
        reservedKVBytes: 0,
        maxReservedKVBytes: 0,
        mlxActiveBytes: 0,
        mlxCacheBytes: 0,
        mlxPeakBytes: 0)
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

private func completionsRequestBody(prompt: String = "Hello", stream: Bool) -> String {
    """
    {"model":"qwen3-32b","prompt":"\(prompt)","max_tokens":8,"temperature":0,"stream":\(stream)}
    """
}

private func completionsHead(contentLength: Int) -> HTTPRequestHead {
    HTTPRequestHead(
        version: .http1_1,
        method: .POST,
        uri: "/v1/completions",
        headers: [
            "host": "localhost",
            "content-type": "application/json",
            "content-length": "\(contentLength)",
        ])
}

private func writeCompletionsRequest(
    _ channel: NIOAsyncTestingChannel,
    body: String,
    authorization: String? = nil
) async throws {
    var head = completionsHead(contentLength: body.utf8.count)
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

/// Captures `--request-log json` lines through a synchronous, thread-safe sink: unlike
/// `ServingEvidenceRecorder` (an actor), `ServingHTTPConfiguration.requestLog` is a plain
/// `@Sendable (String) -> Void` closure invoked from both synchronous NIO event-loop callbacks and
/// detached generation `Task`s, so the capture point itself must be synchronous -- an
/// `OSAllocatedUnfairLock`-backed array, mirroring `ServingHTTPEvidenceTracker`'s own lock usage.
private final class RequestLogRecorder: Sendable {
    private let state = OSAllocatedUnfairLock<[String]>(initialState: [])

    func sink() -> @Sendable (String) -> Void {
        { [state] line in
            state.withLock { $0.append(line) }
        }
    }

    func lines() -> [String] {
        state.withLock { $0 }
    }
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

/// A distinctively-named error, deliberately typed as neither `OpenAIServingError` nor any of
/// `runGeneration`'s other typed catch clauses, so it can only be observed by the untyped
/// catch-all. `description` embeds a marker string with no internal spaces so tests can assert on
/// its presence without also re-exercising `requestFailureDiagnosticLine`'s space-to-underscore
/// substitution (covered separately by the formatter unit test).
private struct DistinctiveUntypedGenerationFailure: Error, Sendable, CustomStringConvertible {
    var description: String { "DISTINCTIVE-UNTYPED-GENERATION-FAILURE-7f2c9a41" }
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
        let lastRequest: OpenAIChatCompletionRequest?
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
        /// Throws an error typed as neither `OpenAIServingError`, `ServingBackendAdmissionError`,
        /// `CancellationError`, `ServingChannelWritabilityGate.GateError`, `ServingMailboxError`,
        /// nor `RunError` — so it lands in `runGeneration`'s untyped catch-all rather than any of
        /// its typed `catch` clauses. This is the only script variant that exercises that path.
        case untypedFailure
        case held
        case heldWithDelayedCancel(DelayedCancellationGate)
    }

    private struct State: Sendable {
        var scripts: [Script]
        var startCount = 0
        var cancelCount = 0
        var lastMailbox: BoundedDeltaMailbox?
        var lastLease: ServingRequestLease?
        var lastRequest: OpenAIChatCompletionRequest?
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
            state.lastRequest = request
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
        } else if case .untypedFailure = script {
            throw DistinctiveUntypedGenerationFailure()
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
                lastLease: $0.lastLease,
                lastRequest: $0.lastRequest)
        }
    }
}

/// A minimal fake backend that streams an exact, caller-specified sequence of `ServingResponseDelta`
/// values (including `.tokenLogprobs`) into the mailbox, then completes. Unlike `ScriptedBackend`'s
/// enum-driven scripts, this exists specifically to test the logprobs-carrying contract at the HTTP
/// layer: how `.tokenLogprobs` deltas interleaved with `.text`/`.toolCalls` deltas turn into
/// `logprobs.content`/`OpenAICompletionLogprobs` on real chunks and the final non-streaming response.
private final class LogprobsScriptedBackend: ServingGenerationBackend, Sendable {
    private let deltas: [ServingResponseDelta]
    private let finishReason: OpenAIChatFinishReason
    private let promptTokens: Int
    private let completionTokens: Int
    private let mailboxMaximumBytes: Int

    init(
        deltas: [ServingResponseDelta],
        finishReason: OpenAIChatFinishReason = .stop,
        promptTokens: Int = 3,
        completionTokens: Int = 3,
        mailboxMaximumBytes: Int = 4_096
    ) {
        self.deltas = deltas
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.mailboxMaximumBytes = mailboxMaximumBytes
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 64, maxBytes: mailboxMaximumBytes))
        let lease = ServingRequestLease(id: ServingRequestID("logprobs-request"), onCancel: {})
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-logprobs",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
        let deltas = self.deltas
        let finishReason = self.finishReason
        let promptTokens = self.promptTokens
        let completionTokens = self.completionTokens
        Task {
            for delta in deltas {
                try? await mailbox.send(delta)
            }
            try? await mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: finishReason,
                        usage: OpenAIChatUsage(
                            promptTokens: promptTokens, completionTokens: completionTokens))))
            await mailbox.finish()
        }
        return handle
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

/// Deterministically parks `OpenAIChatCompletionsHTTPHandler`'s `responseCompletionTestHook` --
/// exactly the point after a keep-alive response's final bytes have been flushed to the client --
/// until a test explicitly releases it. Same wait/isWaiting/release shape as
/// `DelayedCancellationGate` (which parks a lease's cancellation callback instead), kept as its own
/// type so its name doesn't imply cancellation semantics at this different call site.
private actor ResponseCompletionRaceGate {
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
