import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import os
import XCTest

@testable import ServingCore
@testable import ServingNIO

/// Cycle 291d: a reply the client received in full is never logged or counted as a client
/// disconnect. A Content-Length client that reads exactly the body and closes (the front-port
/// proxy does) used to be able to close between the last body write and the final `.end` write:
/// `channelInactive` found the request not yet terminal, recorded `.clientDisconnected`, the `.end`
/// write then failed, and the request was logged `client_disconnected` / 500 (streaming:
/// `client_disconnected` / 200) with the lease cancelled instead of completed. See
/// `docs/task-inbox/2026-10-23-PREDECLARATION-a-delivered-reply-is-never-logged-as-a-disconnect.md`.
///
/// The race is reproduced deterministically with `lastBodyWriteRaceTestHook`, which parks inside
/// the write-completion callback of the LAST body write: the test reads the complete body off the
/// channel, closes it, and only then releases the writer.
final class DeliveredReplyIsNotADisconnectTests: XCTestCase {
    // T1: non-streaming, Content-Length reply read in full and closed before the writer moves on.
    func testNonStreamingReplyClosedAfterFullBodyIsLoggedCompleted() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let backend = DeliveredBackend()
        let gate = DeliveredGate()
        let channel = try await makeDeliveredChannel(
            backend: backend,
            configuration: configuration,
            lastBodyWriteRaceTestHook: { await gate.wait() })

        try await writeDeliveredRequest(channel, stream: false)
        try await waitUntilDelivered { await gate.isWaiting }
        let reply = try await readDeliveredReply(from: channel) { reply in
            reply.expectedBodyBytes.map { reply.body.utf8.count >= $0 } ?? false
        }
        XCTAssertEqual(reply.head?.status, .ok)
        XCTAssertEqual(reply.body.utf8.count, reply.expectedBodyBytes, "the full body was read")
        XCTAssertTrue(reply.body.contains(#""content":"hello""#), reply.body)

        // The client has the whole reply and closes before the writer resumes.
        try await channel.close()
        await gate.release()

        let line = try await assertSingleLogLine(
            recorder, outcome: "completed", status: 200, stream: false)
        XCTAssertFalse(line.contains("client_disconnected"), line)
        assertOneCompleted200Metric(configuration)
        let lease = try XCTUnwrap(backend.lastLease())
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .completed, "the lease completes, it is not cancelled")
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    // T2: the same for `stream: true`: the client reads through `data: [DONE]` and closes.
    func testStreamingReplyClosedAfterDoneIsLoggedCompleted() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let backend = DeliveredBackend()
        let gate = DeliveredGate()
        let channel = try await makeDeliveredChannel(
            backend: backend,
            configuration: configuration,
            lastBodyWriteRaceTestHook: { await gate.wait() })

        try await writeDeliveredRequest(channel, stream: true)
        try await waitUntilDelivered { await gate.isWaiting }
        let reply = try await readDeliveredReply(from: channel) { reply in
            reply.body.contains("data: [DONE]")
        }
        XCTAssertEqual(reply.head?.status, .ok)
        XCTAssertTrue(reply.body.contains(#""content":"hel""#), reply.body)
        XCTAssertTrue(reply.body.contains(#""content":"lo""#), reply.body)
        XCTAssertTrue(reply.body.hasSuffix("data: [DONE]\n\n"), reply.body)

        try await channel.close()
        await gate.release()

        let line = try await assertSingleLogLine(
            recorder, outcome: "completed", status: 200, stream: true)
        XCTAssertFalse(line.contains("client_disconnected"), line)
        assertOneCompleted200Metric(configuration)
        let lease = try XCTUnwrap(backend.lastLease())
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .completed)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    // T3 (D4): a close BEFORE any body byte is written is still a disconnect. The writer is parked
    // just before the last body write; the channel is really closed.
    func testCloseBeforeAnyBodyByteIsStillLoggedClientDisconnected() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let gate = DeliveredGate()
        let channel = try await makeDeliveredChannel(
            backend: DeliveredBackend(),
            configuration: configuration,
            beforeLastBodyWriteTestHook: { await gate.wait() })

        try await writeDeliveredRequest(channel, stream: false)
        try await waitUntilDelivered { await gate.isWaiting }
        try await channel.close()
        await gate.release()

        let line = try await assertSingleLogLine(
            recorder, outcome: "client_disconnected", status: nil, stream: false)
        XCTAssertFalse(line.contains(#""outcome":"completed""#), line)
        let counters = configuration.httpMetrics.snapshot().counters
        XCTAssertEqual(counters.count, 1, "\(counters)")
        XCTAssertEqual(counters.first?.outcome, "client_disconnected", "\(counters)")
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    // T3 (D4), second form (the discriminating one for mutation M4): the peer-gone notification
    // arrives before the body write but the transport would still accept the bytes
    // (`fireChannelInactive` without closing). Only the handler's own bookkeeping can record the
    // disconnect here, so a reply is never reported as delivered when the close was seen before the
    // last body write was issued. (The pre-fix ordering also logs `completed` here: it never looked
    // at the recorded cancellation before writing.)
    func testPeerGoneNotificationBeforeBodyWriteIsStillLoggedClientDisconnected() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let gate = DeliveredGate()
        let channel = try await makeDeliveredChannel(
            backend: DeliveredBackend(),
            configuration: configuration,
            beforeLastBodyWriteTestHook: { await gate.wait() })

        try await writeDeliveredRequest(channel, stream: false)
        try await waitUntilDelivered { await gate.isWaiting }
        channel.pipeline.fireChannelInactive()
        // `fireChannelInactive` hops onto the channel's event loop; let it run before releasing.
        try await channel.eventLoop.submit {}.get()
        await gate.release()

        let line = try await assertSingleLogLine(
            recorder, outcome: "client_disconnected", status: nil, stream: false)
        XCTAssertFalse(line.contains(#""outcome":"completed""#), line)
        let counters = configuration.httpMetrics.snapshot().counters
        XCTAssertEqual(counters.count, 1, "\(counters)")
        XCTAssertEqual(counters.first?.outcome, "client_disconnected", "\(counters)")
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    // D3 (safety net): the last body write succeeded and only the `.end` write failed -- the reply
    // is recorded as delivered.
    func testNonStreamingEndWriteFailureAfterBodyIsLoggedCompleted() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let backend = DeliveredBackend()
        let channel = try await makeDeliveredChannel(
            backend: backend,
            configuration: configuration,
            failEndWrites: true)

        try await writeDeliveredRequest(channel, stream: false)
        let line = try await assertSingleLogLine(
            recorder, outcome: "completed", status: 200, stream: false)
        XCTAssertFalse(line.contains("client_disconnected"), line)
        assertOneCompleted200Metric(configuration)
        let lease = try XCTUnwrap(backend.lastLease())
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .completed)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    func testStreamingEndWriteFailureAfterDoneIsLoggedCompleted() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let backend = DeliveredBackend()
        let channel = try await makeDeliveredChannel(
            backend: backend,
            configuration: configuration,
            failEndWrites: true)

        try await writeDeliveredRequest(channel, stream: true)
        let line = try await assertSingleLogLine(
            recorder, outcome: "completed", status: 200, stream: true)
        XCTAssertFalse(line.contains("client_disconnected"), line)
        assertOneCompleted200Metric(configuration)
        let lease = try XCTUnwrap(backend.lastLease())
        let leaseState = await lease.state
        XCTAssertEqual(leaseState, .completed)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }

    // D3 on a not-yet-started error reply (admission failure): the 429 body was written, only the
    // `.end` failed -- the request is logged with the error outcome and its real status, never as a
    // client disconnect.
    func testAdmissionFailureEndWriteFailureAfterBodyIsLoggedAsErrorNotDisconnect() async throws {
        let recorder = DeliveredLogRecorder()
        let configuration = deliveredConfiguration(sink: recorder.sink())
        let channel = try await makeDeliveredChannel(
            backend: DeliveredBackend(rejectsAdmission: true),
            configuration: configuration,
            failEndWrites: true)

        try await writeDeliveredRequest(channel, stream: false)
        let line = try await assertSingleLogLine(
            recorder, outcome: "error", status: 429, stream: false)
        XCTAssertFalse(line.contains("client_disconnected"), line)
        _ = try await channel.finish(acceptAlreadyClosed: true)
    }
}

// MARK: - Assertions

private func assertSingleLogLine(
    _ recorder: DeliveredLogRecorder,
    outcome: String,
    status: Int?,
    stream: Bool,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> String {
    try await waitUntilDelivered(file: file, line: line) { recorder.lines().count >= 1 }
    // A second line would be a double count; give any straggler a moment to appear.
    try await Task.sleep(for: .milliseconds(100))
    let lines = recorder.lines()
    XCTAssertEqual(lines.count, 1, "exactly one request-log line: \(lines)", file: file, line: line)
    let text = try XCTUnwrap(lines.first, file: file, line: line)
    let object = try XCTUnwrap(
        JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
        file: file, line: line)
    XCTAssertEqual(object["outcome"] as? String, outcome, text, file: file, line: line)
    if let status {
        XCTAssertEqual(object["status"] as? Int, status, text, file: file, line: line)
    }
    XCTAssertEqual(object["stream"] as? Bool, stream, text, file: file, line: line)
    XCTAssertEqual(
        object["route"] as? String, "/v1/chat/completions", text, file: file, line: line)
    return text
}

private func assertOneCompleted200Metric(
    _ configuration: ServingHTTPConfiguration,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let counters = configuration.httpMetrics.snapshot().counters
    XCTAssertEqual(counters.count, 1, "exactly one counter series: \(counters)", file: file, line: line)
    XCTAssertEqual(counters.first?.route, "/v1/chat/completions", "\(counters)", file: file, line: line)
    XCTAssertEqual(counters.first?.statusClass, "2xx", "\(counters)", file: file, line: line)
    XCTAssertEqual(counters.first?.outcome, "completed", "\(counters)", file: file, line: line)
    XCTAssertEqual(counters.first?.count, 1, "\(counters)", file: file, line: line)
}

// MARK: - Helpers

private struct DeliveredReply {
    var head: HTTPResponseHead?
    var body = ""
    var sawEnd = false

    var expectedBodyBytes: Int? {
        head?.headers.first(name: "content-length").flatMap { Int($0) }
    }
}

/// Reads outbound response parts until `isComplete` is satisfied. It deliberately does not wait for
/// `.end`: on the pre-fix ordering the `.end` is not written until the parked writer is released.
private func readDeliveredReply(
    from channel: NIOAsyncTestingChannel,
    until isComplete: (DeliveredReply) -> Bool
) async throws -> DeliveredReply {
    var reply = DeliveredReply()
    while !isComplete(reply) {
        let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
        switch part {
        case .head(let value):
            reply.head = value
        case .body(.byteBuffer(var buffer)):
            reply.body += buffer.readString(length: buffer.readableBytes) ?? ""
        case .body(.fileRegion):
            XCTFail("Serving responses must not emit file regions")
        case .end:
            reply.sawEnd = true
            return reply
        }
    }
    return reply
}

private func makeDeliveredChannel(
    backend: any ServingGenerationBackend,
    configuration: ServingHTTPConfiguration,
    lastBodyWriteRaceTestHook: (@Sendable () async -> Void)? = nil,
    beforeLastBodyWriteTestHook: (@Sendable () async -> Void)? = nil,
    failEndWrites: Bool = false
) async throws -> NIOAsyncTestingChannel {
    try await NIOAsyncTestingChannel { channel in
        let handler = OpenAIChatCompletionsHTTPHandler(
            configuration: configuration,
            backend: backend,
            lastBodyWriteRaceTestHook: lastBodyWriteRaceTestHook,
            beforeLastBodyWriteTestHook: beforeLastBodyWriteTestHook)
        if failEndWrites {
            // Added first, so it sits between the handler and the head of the pipeline: every
            // write the handler issues passes through it.
            try channel.pipeline.syncOperations.addHandlers(
                FailingEndOutboundHandler(), handler)
        } else {
            try channel.pipeline.syncOperations.addHandler(handler)
        }
    }
}

private func deliveredConfiguration(
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

private func writeDeliveredRequest(
    _ channel: NIOAsyncTestingChannel,
    stream: Bool
) async throws {
    let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":\(stream)}
        """
    let head = HTTPRequestHead(
        version: .http1_1,
        method: .POST,
        uri: "/v1/chat/completions",
        headers: [
            "host": "localhost",
            "content-type": "application/json",
            "content-length": "\(body.utf8.count)",
        ])
    _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
    _ = try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: body)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
}

private func waitUntilDelivered(
    timeout: Duration = .seconds(5),
    file: StaticString = #filePath,
    line: UInt = #line,
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
    XCTFail("Condition was not reached before the deadline", file: file, line: line)
    throw DeliveredTestError.timedOut
}

private enum DeliveredTestError: Error {
    case timedOut
}

private final class DeliveredLogRecorder: Sendable {
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

/// Parks one awaiting hook until the test releases it. Safe if released before it is awaited.
private actor DeliveredGate {
    private(set) var isWaiting = false
    private var released = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        if released {
            return
        }
        isWaiting = true
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

/// Fails every `.end` write and passes everything else through: models a peer that has received
/// the complete body but is gone by the time the final part is written.
private final class FailingEndOutboundHandler: ChannelOutboundHandler, Sendable {
    typealias OutboundIn = HTTPServerResponsePart
    typealias OutboundOut = HTTPServerResponsePart

    func write(
        context: ChannelHandlerContext,
        data: NIOAny,
        promise: EventLoopPromise<Void>?
    ) {
        if case .end = unwrapOutboundIn(data) {
            promise?.fail(DeliveredEndWriteFailure())
            return
        }
        context.write(data, promise: promise)
    }
}

private struct DeliveredEndWriteFailure: Error {}

/// Completes a short reply ("hello") immediately; optionally rejects admission instead.
private final class DeliveredBackend: ServingGenerationBackend, Sendable {
    private let rejectsAdmission: Bool
    private let lease = OSAllocatedUnfairLock<ServingRequestLease?>(initialState: nil)

    init(rejectsAdmission: Bool = false) {
        self.rejectsAdmission = rejectsAdmission
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        if rejectsAdmission {
            throw ServingBackendAdmissionError.queueFull(retryAfterSeconds: 1)
        }
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 1_024))
        let lease = ServingRequestLease(id: ServingRequestID("delivered-request"))
        self.lease.withLock { $0 = lease }
        Task {
            do {
                try await mailbox.send(.text("hel"))
                try await mailbox.send(.text("lo"))
                try await mailbox.send(
                    .completion(
                        ServingGenerationCompletion(
                            finishReason: .stop,
                            usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 2))))
                await mailbox.finish()
            } catch {
                // Cancellation is observed through the lease state.
            }
        }
        return ServingGenerationHandle(
            responseID: "chatcmpl-delivered",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }

    func shutdown() async {}

    func lastLease() -> ServingRequestLease? {
        lease.withLock { $0 }
    }
}
