import Foundation
import NIOCore
import NIOPosix
import ServingCore
import os
import XCTest

@testable import ServingNIO

/// Cycle 291: the DNS-rebinding `Host` guard on a server with NO bearer token. Socket tests drive a
/// REAL server so the refusal travels the production pipeline (parser -> handler -> writer); the
/// pure `ServingHostGuard.evaluate` cases pin the strict parse independently of the transport.
final class HostGuardTests: XCTestCase {
    // MARK: - Pure parser

    private let lookalikeHosts = [
        "127.0.0.1.evil.com", "localhost.evil.com", "[::1].evil", "0x7f.1", "2130706433",
        "127.1", "127.000.000.001", "a.localhost", "127.0.0.1.", "localhost..",
        "evil.com", "127.0.0.256", "127.0.0.01", "+127.0.0.1", "[::2]", "128.0.0.1",
    ]
    private let acceptedHosts = [
        "localhost:9999", "LOCALHOST.", "127.8.9.10", "[::1]:1", "127.0.0.1", "localhost",
        "127.0.0.1:8080", "[::1]", "  127.0.0.1  ", "127.255.255.255",
    ]

    func testPureLookalikesAreNotAllowed() {
        let guardian = ServingHostGuard()
        for host in lookalikeHosts where host != "127.0.0.1." {
            XCTAssertEqual(
                guardian.evaluate(hostHeaders: [host], requestTarget: "/healthz"),
                .rejectNotAllowed, host)
        }
        // One trailing dot is stripped (FQDN form), so this one is loopback.
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["127.0.0.1."], requestTarget: "/healthz"), .allow)
    }

    func testPureAcceptedHostsAreAllowedAndPortIsIgnored() {
        let guardian = ServingHostGuard()
        for host in acceptedHosts {
            XCTAssertEqual(
                guardian.evaluate(hostHeaders: [host], requestTarget: "/v1/models"), .allow, host)
        }
    }

    func testPureMissingDuplicateAndMalformedHostAreInvalid() {
        let guardian = ServingHostGuard()
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: [], requestTarget: "/healthz"), .rejectInvalidHost)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["localhost", "localhost"], requestTarget: "/healthz"),
            .rejectInvalidHost)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["localhost", "evil.com"], requestTarget: "/healthz"),
            .rejectInvalidHost)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["evil.com", "localhost"], requestTarget: "/healthz"),
            .rejectInvalidHost)
        for malformed in [
            "", "   ", "::1", "localhost:abc", "localhost:123456", "localhost:80:80", "[::1",
            "user@localhost", "local host", "localhost/x", ":80", ".", "[]",
        ] {
            XCTAssertEqual(
                guardian.evaluate(hostHeaders: [malformed], requestTarget: "/healthz"),
                .rejectInvalidHost, "\(malformed)")
        }
    }

    func testPureNonOriginFormTargetIsNotAllowedEvenWithLoopbackHost() {
        let guardian = ServingHostGuard()
        for target in ["http://localhost/healthz", "*", "", "healthz"] {
            XCTAssertEqual(
                guardian.evaluate(hostHeaders: ["localhost"], requestTarget: target),
                .rejectNotAllowed, target)
        }
    }

    func testPureAllowListAddsExactNamesCaseInsensitivelyOnly() {
        let guardian = ServingHostGuard(allowedHosts: ["My-Box.example", "203.0.113.5"])
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["MY-BOX.example:8080"], requestTarget: "/"), .allow)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["203.0.113.5"], requestTarget: "/"), .allow)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["evil.my-box.example"], requestTarget: "/"),
            .rejectNotAllowed)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["my-box.example.evil.com"], requestTarget: "/"),
            .rejectNotAllowed)
        XCTAssertEqual(
            guardian.evaluate(hostHeaders: ["203.0.113.50"], requestTarget: "/"),
            .rejectNotAllowed)
    }

    func testConfigurationBuildsGuardOnlyWithoutToken() {
        XCTAssertNotNil(hostGuardConfiguration(token: nil).hostGuard)
        XCTAssertNil(hostGuardConfiguration(token: "secret").hostGuard)
        XCTAssertEqual(
            hostGuardConfiguration(token: nil, allowedHosts: ["Box.Example"]).hostGuard?
                .allowedHosts,
            ["box.example"])
    }

    // MARK: - Socket: E1 / E2 / E3

    func testE1LookalikeHostsGet421AndNeverStartTheBackend() async throws {
        let backend = HostGuardCountingBackend()
        let hosts = lookalikeHosts.filter { $0 != "127.0.0.1." }
        try await withHostGuardServer(backend: backend) { server in
            for host in hosts {
                let chat = try await hostGuardExchange(
                    server: server, request: chatRequest(host: host))
                let response = try HostGuardResponse(chat.text)
                XCTAssertEqual(response.status, 421, "POST host=\(host): \(chat.text)")
                XCTAssertTrue(response.body.contains("host_not_allowed"), "\(host): \(chat.text)")
                XCTAssertFalse(
                    response.body.contains(host) && !host.isEmpty && host.count > 3,
                    "the Host value must not be echoed: \(chat.text)")
                XCTAssertTrue(chat.closed, "\(host): connection must close")

                let models = try await hostGuardExchange(
                    server: server, request: getRequest(path: "/v1/models", host: host))
                XCTAssertEqual(try HostGuardResponse(models.text).status, 421, "GET host=\(host)")
            }
        }
        XCTAssertEqual(backend.startCount(), 0, "a refused Host must never reach the backend")
    }

    func testE2LoopbackHostsPassIncludingPortAndCase() async throws {
        let backend = HostGuardCountingBackend()
        try await withHostGuardServer(backend: backend) { server in
            for host in acceptedHosts {
                let health = try await hostGuardExchange(
                    server: server, request: getRequest(path: "/healthz", host: host))
                XCTAssertEqual(
                    try HostGuardResponse(health.text).status, 200, "host=\(host): \(health.text)")
                let models = try await hostGuardExchange(
                    server: server, request: getRequest(path: "/v1/models", host: host))
                XCTAssertEqual(try HostGuardResponse(models.text).status, 200, "host=\(host)")
            }
            let chat = try await hostGuardExchange(
                server: server, request: chatRequest(host: "localhost:9999"))
            XCTAssertFalse(chat.text.contains("host_not_allowed"), chat.text)
        }
        XCTAssertEqual(backend.startCount(), 1, "an allowed Host reaches the backend")
    }

    func testE3GuardAppliesToProbesBeforeTheProbeBypass() async throws {
        try await withHostGuardServer(backend: HostGuardCountingBackend()) { server in
            for path in ["/healthz", "/readyz", "/metrics", "/v1/models", "/unknown-route"] {
                let exchange = try await hostGuardExchange(
                    server: server, request: getRequest(path: path, host: "evil.example"))
                XCTAssertEqual(
                    try HostGuardResponse(exchange.text).status, 421, "\(path): \(exchange.text)")
            }
            // A probe with a wrong METHOD and an evil Host is still the Host refusal, not a 405.
            let wrongMethod = try await hostGuardExchange(
                server: server,
                request: "DELETE /healthz HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
            )
            XCTAssertEqual(try HostGuardResponse(wrongMethod.text).status, 421)
        }
    }

    // MARK: - Socket: E4 / E5 / E6

    func testE4MissingAndDuplicateHostGet400() async throws {
        let backend = HostGuardCountingBackend()
        try await withHostGuardServer(backend: backend) { server in
            let requests: [(String, String)] = [
                ("http/1.0 no host", "GET /healthz HTTP/1.0\r\n\r\n"),
                ("http/1.1 no host", "GET /healthz HTTP/1.1\r\nConnection: close\r\n\r\n"),
                (
                    "dup good,evil",
                    "GET /healthz HTTP/1.1\r\nHost: localhost\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
                ),
                (
                    "dup evil,good",
                    "GET /healthz HTTP/1.1\r\nHost: evil.example\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                ),
                (
                    "dup good,good",
                    "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                ),
                (
                    "bare ipv6",
                    "GET /healthz HTTP/1.1\r\nHost: ::1\r\nConnection: close\r\n\r\n"
                ),
                (
                    "bad port",
                    "GET /healthz HTTP/1.1\r\nHost: localhost:abc\r\nConnection: close\r\n\r\n"
                ),
            ]
            for (label, raw) in requests {
                let exchange = try await hostGuardExchange(server: server, request: raw)
                let response = try HostGuardResponse(exchange.text)
                XCTAssertEqual(response.status, 400, "\(label): \(exchange.text)")
                XCTAssertTrue(response.body.contains("invalid_host_header"), "\(label)")
                XCTAssertTrue(exchange.closed, "\(label): connection must close")
            }
        }
        XCTAssertEqual(backend.startCount(), 0)
    }

    func testE5HeadRefusalHasNoBody() async throws {
        try await withHostGuardServer(backend: HostGuardCountingBackend()) { server in
            for (host, expected) in [("evil.example", 421), ("", 400)] {
                let raw =
                    host.isEmpty
                    ? "HEAD /healthz HTTP/1.0\r\n\r\n"
                    : "HEAD /healthz HTTP/1.1\r\nHost: \(host)\r\nConnection: close\r\n\r\n"
                let exchange = try await hostGuardExchange(server: server, request: raw)
                let response = try HostGuardResponse(exchange.text)
                XCTAssertEqual(response.status, expected, exchange.text)
                XCTAssertEqual(response.body, "", "HEAD refusal must carry no body: \(exchange.text)")
                XCTAssertNotNil(response.headers["x-request-id"])
            }
        }
    }

    func testE6RefusalIsOneLogLineOneCountOneRequestIDAndCloses() async throws {
        let recorder = HostGuardLogRecorder()
        let configuration = hostGuardConfiguration(token: nil, sink: recorder.sink())
        try await withHostGuardServer(
            backend: HostGuardCountingBackend(), configuration: configuration
        ) { server in
            let exchange = try await hostGuardExchange(
                server: server,
                request: chatRequest(host: "evil.example", keepAlive: true))
            let response = try HostGuardResponse(exchange.text)
            XCTAssertEqual(response.status, 421, exchange.text)
            XCTAssertEqual(response.headerCounts["x-request-id"], 1, "one x-request-id")
            XCTAssertEqual(response.headers["connection"]?.lowercased(), "close")
            XCTAssertTrue(exchange.closed, "the connection must close even when keep-alive asked")

            try await waitUntilHostGuardTest { recorder.lines().count >= 1 }
            try await Task.sleep(for: .milliseconds(50))
            let lines = recorder.lines()
            XCTAssertEqual(lines.count, 1, "exactly one request-log line: \(lines)")
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(try XCTUnwrap(lines.first).utf8))
                    as? [String: Any])
            XCTAssertEqual(object["status"] as? Int, 421)
            XCTAssertEqual(object["request_id"] as? String, response.headers["x-request-id"])
            XCTAssertFalse(
                (lines.first ?? "").contains("evil.example"), "the Host must not be logged")

            let counters = configuration.httpMetrics.snapshot().counters
            XCTAssertEqual(counters.map(\.count).reduce(0, +), 1, "one request counted: \(counters)")
            XCTAssertEqual(counters.first?.statusClass, "4xx")
        }
    }

    // MARK: - Socket: E7 / E8 / E9

    func testE7TokenedServerIgnoresHostAndKeepsItsAuthContract() async throws {
        let configuration = hostGuardConfiguration(token: "secret-token")
        let backend = HostGuardCountingBackend()
        try await withHostGuardServer(backend: backend, configuration: configuration) { server in
            let authorized = try await hostGuardExchange(
                server: server,
                request:
                    "GET /v1/models HTTP/1.1\r\nHost: evil.example\r\nAuthorization: Bearer secret-token\r\nConnection: close\r\n\r\n"
            )
            XCTAssertEqual(try HostGuardResponse(authorized.text).status, 200, authorized.text)
            let unauthorized = try await hostGuardExchange(
                server: server, request: getRequest(path: "/v1/models", host: "evil.example"))
            XCTAssertEqual(try HostGuardResponse(unauthorized.text).status, 401, unauthorized.text)
            let probe = try await hostGuardExchange(
                server: server, request: getRequest(path: "/healthz", host: "evil.example"))
            XCTAssertEqual(try HostGuardResponse(probe.text).status, 200, probe.text)
            // A tokened server does not even require a Host header (unchanged behavior).
            let noHost = try await hostGuardExchange(
                server: server, request: "GET /healthz HTTP/1.0\r\n\r\n")
            XCTAssertEqual(try HostGuardResponse(noHost.text).status, 200, noHost.text)
        }
    }

    func testE8AbsoluteFormAndAsteriskTargetsGet421() async throws {
        try await withHostGuardServer(backend: HostGuardCountingBackend()) { server in
            for target in ["http://localhost/healthz", "http://evil.example/healthz", "*"] {
                let method = target == "*" ? "OPTIONS" : "GET"
                let exchange = try await hostGuardExchange(
                    server: server,
                    request:
                        "\(method) \(target) HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                )
                let response = try HostGuardResponse(exchange.text)
                XCTAssertEqual(response.status, 421, "\(target): \(exchange.text)")
                XCTAssertTrue(response.body.contains("host_not_allowed"))
            }
        }
    }

    func testE9AllowedNameIsAcceptedOthersStillRefused() async throws {
        let configuration = hostGuardConfiguration(
            token: nil, allowedHosts: ["my-box.example", "203.0.113.5"])
        try await withHostGuardServer(
            backend: HostGuardCountingBackend(), configuration: configuration
        ) { server in
            for host in ["my-box.example", "MY-BOX.EXAMPLE:8080", "203.0.113.5:9000", "localhost"] {
                let exchange = try await hostGuardExchange(
                    server: server, request: getRequest(path: "/healthz", host: host))
                XCTAssertEqual(try HostGuardResponse(exchange.text).status, 200, host)
            }
            for host in ["evil.example", "my-box.example.evil.com", "203.0.113.50"] {
                let exchange = try await hostGuardExchange(
                    server: server, request: getRequest(path: "/healthz", host: host))
                XCTAssertEqual(try HostGuardResponse(exchange.text).status, 421, host)
            }
        }
    }
}

// MARK: - Helpers

private func hostGuardConfiguration(
    token: String?,
    allowedHosts: [String] = [],
    sink: (@Sendable (String) -> Void)? = nil
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: token,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        requestLog: sink,
        allowedHosts: allowedHosts)
}

private func getRequest(path: String, host: String) -> String {
    "GET \(path) HTTP/1.1\r\nHost: \(host)\r\nConnection: close\r\n\r\n"
}

private func chatRequest(host: String, keepAlive: Bool = false) -> String {
    let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0}
        """
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: \(host)\r\n"
        + (keepAlive ? "" : "Connection: close\r\n")
        + "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
}

private func withHostGuardServer(
    backend: HostGuardCountingBackend,
    configuration: ServingHTTPConfiguration? = nil,
    _ body: (ServingHTTPServer) async throws -> Void
) async throws {
    let server = try await ServingHTTPServer.start(
        configuration: configuration ?? hostGuardConfiguration(token: nil),
        backend: backend)
    do {
        try await body(server)
    } catch {
        try? await server.shutdown(gracePeriod: .seconds(1))
        throw error
    }
    try await server.shutdown(gracePeriod: .seconds(1))
}

/// One raw connection: send `request`, wait for the server to close it, return every byte.
private func hostGuardExchange(
    server: ServingHTTPServer,
    request: String
) async throws -> (text: String, closed: Bool) {
    let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let collector = HostGuardByteCollector()
    do {
        let client = try await ClientBootstrap(group: clientGroup)
            .channelInitializer { $0.pipeline.addHandler(collector) }
            .connect(to: server.localAddress)
            .get()
        try await client.writeAndFlush(ByteBuffer(string: request)).get()
        try await waitUntilHostGuardTest { collector.isClosed() }
    } catch {
        try? await clientGroup.shutdownGracefully()
        throw error
    }
    try await clientGroup.shutdownGracefully()
    return (collector.text(), collector.isClosed())
}

private struct HostGuardResponse {
    let status: Int
    let headers: [String: String]
    let headerCounts: [String: Int]
    let body: String

    init(_ text: String) throws {
        let parts = text.components(separatedBy: "\r\n\r\n")
        let headBlock = try XCTUnwrap(parts.first, "no response: \(text)")
        let lines = headBlock.components(separatedBy: "\r\n")
        let statusLine = try XCTUnwrap(lines.first)
        let pieces = statusLine.split(separator: " ")
        guard pieces.count >= 2, let code = Int(pieces[1]) else {
            throw NSError(
                domain: "HostGuardTests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "bad status line: \(text)"])
        }
        status = code
        var headers: [String: String] = [:]
        var counts: [String: Int] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].lowercased()
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            counts[name, default: 0] += 1
        }
        self.headers = headers
        headerCounts = counts
        body = parts.dropFirst().joined(separator: "\r\n\r\n")
    }
}

private func waitUntilHostGuardTest(
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
    XCTFail("Condition was not reached before the host-guard test deadline")
}

private final class HostGuardLogRecorder: Sendable {
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

private final class HostGuardByteCollector: ChannelInboundHandler, @unchecked Sendable {
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

/// Counts `start` calls and refuses each one, so a request that reaches it finishes quickly with
/// an error while a refused Host provably never does.
private final class HostGuardCountingBackend: ServingGenerationBackend, Sendable {
    private let starts = OSAllocatedUnfairLock(initialState: 0)

    func startCount() -> Int {
        starts.withLock { $0 }
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        starts.withLock { $0 += 1 }
        throw OpenAIServingError.server("host guard test backend", code: "host_guard_test_backend")
    }

    func shutdown() async {}
}
