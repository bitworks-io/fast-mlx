import Foundation
import NIOCore
import NIOPosix
import ServingCore
import os
import XCTest

@testable import ServingNIO

/// Cycle 291: a malformed-HTTP (parser-error) 400 carries exactly one `x-request-id`, writes one
/// request-log line with the same id, counts once in `httpMetrics`, and is never written when a
/// response is already in flight. Every test drives a REAL socket server so the parser error
/// reaches the same pipeline production uses.
final class ProtocolErrorRequestIDTests: XCTestCase {
    // T1: a malformed request line gets a 400 with exactly one x-request-id, Connection: close,
    // Content-Length: 0, and the connection closes after it.
    func testMalformedRequestLineGets400WithOneRequestIDAndCloses() async throws {
        let exchange = try await rawExchange(
            sending: ["\u{1}\u{2}BAD REQUEST LINE\r\n\r\n"],
            configuration: protocolErrorConfiguration())
        try assertProtocolError400(exchange.text, label: "malformed request line")
        XCTAssertTrue(exchange.closed, "the connection must close after the 400")
    }

    // T1: a malformed header (space inside a header name) on a fresh connection.
    func testMalformedHeaderGets400WithOneRequestIDAndCloses() async throws {
        let exchange = try await rawExchange(
            sending: ["GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nBad Header: x\r\n\r\n"],
            configuration: protocolErrorConfiguration())
        try assertProtocolError400(exchange.text, label: "malformed header")
        XCTAssertTrue(exchange.closed, "the connection must close after the 400")
    }

    // T2: with the request log on, exactly one line: status 400, request_id equal to the header,
    // no method/route, no raw request bytes; httpMetrics counts one 400.
    func testProtocolErrorWritesOneRequestLogLineWithHeaderIDAndCountsOne400() async throws {
        let recorder = ProtocolErrorLogRecorder()
        let configuration = protocolErrorConfiguration(sink: recorder.sink())
        let marker = "SECRETRAWBYTES"
        let exchange = try await rawExchange(
            sending: ["\u{1}\(marker) BAD\r\n\r\n"],
            configuration: configuration)
        let id = try assertProtocolError400(exchange.text, label: "logged protocol error")

        try await waitUntilProtocolErrorTest { recorder.lines().count >= 1 }
        // A second line would be a double count; give any straggler a moment to appear.
        try await Task.sleep(for: .milliseconds(50))
        let lines = recorder.lines()
        XCTAssertEqual(lines.count, 1, "exactly one request-log line: \(lines)")
        let line = try XCTUnwrap(lines.first)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        XCTAssertEqual(object["status"] as? Int, 400)
        XCTAssertEqual(object["request_id"] as? String, id, "logged id equals the header id")
        XCTAssertTrue(
            object["method"] == nil || object["method"] is NSNull, "method absent or null")
        XCTAssertTrue(
            object["route"] == nil || object["route"] is NSNull, "route absent or null")
        XCTAssertFalse(line.contains(marker), "no raw request bytes in the log: \(line)")

        let snapshot = configuration.httpMetrics.snapshot()
        let counted = snapshot.counters.filter { $0.statusClass == "4xx" }
        XCTAssertEqual(counted.map(\.count).reduce(0, +), 1, "one 4xx counted: \(snapshot.counters)")
        XCTAssertEqual(snapshot.counters.map(\.count).reduce(0, +), 1, "nothing else counted")
    }

    // T2 (edge): with the request log OFF the 400 still carries the id and still counts once.
    func testProtocolErrorWithRequestLogOffStillCarriesIDAndCountsOne400() async throws {
        let configuration = protocolErrorConfiguration()
        let exchange = try await rawExchange(
            sending: ["\u{1}\u{2}BAD\r\n\r\n"],
            configuration: configuration)
        try assertProtocolError400(exchange.text, label: "log off")
        let counters = configuration.httpMetrics.snapshot().counters
        XCTAssertEqual(counters.map(\.count).reduce(0, +), 1, "\(counters)")
        XCTAssertEqual(counters.first?.statusClass, "4xx")
    }

    // T3: a valid request whose response is in flight (SSE head written, never terminated),
    // followed by garbage on the same connection: exactly one response head reaches the client.
    func testGarbageAfterInFlightResponseWritesNoSecondResponse() async throws {
        let recorder = ProtocolErrorLogRecorder()
        let configuration = protocolErrorConfiguration(sink: recorder.sink())
        let server = try await ServingHTTPServer.start(
            configuration: configuration,
            backend: HoldingProtocolErrorBackend())
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let collector = ProtocolErrorByteCollector()
        do {
            let client = try await ClientBootstrap(group: clientGroup)
                .channelInitializer { $0.pipeline.addHandler(collector) }
                .connect(to: server.localAddress)
                .get()
            try await client.writeAndFlush(ByteBuffer(string: streamingRequest())).get()
            try await waitUntilProtocolErrorTest { collector.text().contains("started") }
            XCTAssertFalse(collector.isClosed(), "the response must still be in flight")

            try await client.writeAndFlush(
                ByteBuffer(string: "\u{1}\u{2}GARBAGE\r\n\r\n")).get()
            try await waitUntilProtocolErrorTest { collector.isClosed() }
        } catch {
            try? await server.shutdown(gracePeriod: .seconds(1))
            try? await clientGroup.shutdownGracefully()
            throw error
        }
        try await server.shutdown(gracePeriod: .seconds(1))
        try await clientGroup.shutdownGracefully()

        let text = collector.text()
        XCTAssertEqual(
            text.components(separatedBy: "HTTP/1.1 ").count - 1, 1,
            "exactly one response head: \(text)")
        XCTAssertTrue(text.hasPrefix("HTTP/1.1 200 OK\r\n"), text)
        XCTAssertFalse(text.contains("400 Bad Request"), "no second (400) response: \(text)")
    }

    // T4 (edge): a non-parser path is unchanged -- a valid request on the same server still works
    // and carries one x-request-id.
    func testValidRequestStillAnswersWithOneRequestID() async throws {
        let exchange = try await rawExchange(
            sending: ["GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"],
            configuration: protocolErrorConfiguration())
        XCTAssertTrue(exchange.text.hasPrefix("HTTP/1.1 200 OK\r\n"), exchange.text)
        XCTAssertEqual(requestIDs(in: exchange.text).count, 1)
        XCTAssertTrue(exchange.closed)
    }
}

// MARK: - Helpers

/// The 400 response's header block, validated; returns the single `x-request-id`.
@discardableResult
private func assertProtocolError400(_ text: String, label: String) throws -> String {
    XCTAssertTrue(text.hasPrefix("HTTP/1.1 400 Bad Request\r\n"), "\(label): \(text)")
    XCTAssertEqual(
        text.components(separatedBy: "HTTP/1.1 ").count - 1, 1, "\(label): one response: \(text)")
    let lowered = text.lowercased()
    XCTAssertTrue(lowered.contains("connection: close\r\n"), "\(label): \(text)")
    XCTAssertTrue(lowered.contains("content-length: 0\r\n"), "\(label): \(text)")
    let ids = requestIDs(in: text)
    XCTAssertEqual(ids.count, 1, "\(label): exactly one x-request-id: \(text)")
    let id = try XCTUnwrap(ids.first, label)
    // The handler mints `UUID().uuidString` for every request; the protocol-error id is the same
    // format.
    XCTAssertNotNil(UUID(uuidString: id), "\(label): id must be in the handler's format: \(id)")
    return id
}

private func requestIDs(in text: String) -> [String] {
    let headerBlock = text.components(separatedBy: "\r\n\r\n").first ?? text
    return headerBlock.components(separatedBy: "\r\n").compactMap { line -> String? in
        guard line.lowercased().hasPrefix("x-request-id:") else {
            return nil
        }
        return line.dropFirst("x-request-id:".count).trimmingCharacters(in: .whitespaces)
    }
}

private func protocolErrorConfiguration(
    sink: (@Sendable (String) -> Void)? = nil
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        requestLog: sink)
}

private func streamingRequest() -> String {
    let body = """
    {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":true}
    """
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        + "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
}

/// Starts a server, sends each chunk in order on one raw connection, waits for the server to
/// close it, and returns every byte received.
private func rawExchange(
    sending chunks: [String],
    configuration: ServingHTTPConfiguration
) async throws -> (text: String, closed: Bool) {
    let server = try await ServingHTTPServer.start(
        configuration: configuration,
        backend: HoldingProtocolErrorBackend())
    let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let collector = ProtocolErrorByteCollector()
    do {
        let client = try await ClientBootstrap(group: clientGroup)
            .channelInitializer { $0.pipeline.addHandler(collector) }
            .connect(to: server.localAddress)
            .get()
        for chunk in chunks {
            try await client.writeAndFlush(ByteBuffer(string: chunk)).get()
        }
        try await waitUntilProtocolErrorTest { collector.isClosed() }
    } catch {
        try? await server.shutdown(gracePeriod: .seconds(1))
        try? await clientGroup.shutdownGracefully()
        throw error
    }
    try await server.shutdown(gracePeriod: .seconds(1))
    try await clientGroup.shutdownGracefully()
    return (collector.text(), collector.isClosed())
}

private func waitUntilProtocolErrorTest(
    timeout: Duration = .seconds(3),
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
    XCTFail("Condition was not reached before the protocol-error test deadline")
}

private final class ProtocolErrorLogRecorder: Sendable {
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

private final class ProtocolErrorByteCollector: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    private struct State {
        var bytes: [UInt8] = []
        var closed = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        var buffer = unwrapInboundIn(data)
        let incoming = buffer.readBytes(length: buffer.readableBytes) ?? []
        state.withLock { $0.bytes.append(contentsOf: incoming) }
    }

    func channelInactive(context: ChannelHandlerContext) {
        state.withLock { $0.closed = true }
        context.fireChannelInactive()
    }

    func text() -> String {
        state.withLock { String(decoding: $0.bytes, as: UTF8.self) }
    }

    func isClosed() -> Bool {
        state.withLock { $0.closed }
    }
}

/// Publishes one delta ("started") and then holds forever, so a streaming response is in flight
/// (head written, never terminated) until the connection closes.
private final class HoldingProtocolErrorBackend: ServingGenerationBackend, Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(capacity: .init(maxDeltas: 1, maxBytes: 65_536))
        let lease = ServingRequestLease(
            id: ServingRequestID("protocol-error-request"),
            onCancel: { [mailbox] in
                await mailbox.cancel(.clientDisconnected)
            })
        Task { [mailbox] in
            try? await mailbox.send(.text("started"))
        }
        return ServingGenerationHandle(
            responseID: "chatcmpl-protocol-error",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }

    func shutdown() async {}
}
