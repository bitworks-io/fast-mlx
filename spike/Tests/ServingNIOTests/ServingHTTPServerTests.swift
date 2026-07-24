import Foundation
import NIOCore
import NIOPosix
import os
import XCTest

@testable import ServingCore
@testable import ServingNIO

final class ServingHTTPServerTests: XCTestCase {
    func testRemoteBindRequiresAuthenticationBeforeOpeningListener() async throws {
        let backend = SocketBackend(script: .held)
        let configuration = socketConfiguration(requiredBearerToken: nil)

        do {
            _ = try await ServingHTTPServer.start(
                bind: .init(host: "0.0.0.0", port: 0),
                configuration: configuration,
                backend: backend)
            XCTFail("An unauthenticated non-loopback listener must fail closed")
        } catch let error as ServingHTTPServerError {
            XCTAssertEqual(error, .remoteBindRequiresAuthentication)
        }

        do {
            _ = try await ServingHTTPServer.start(
                bind: .init(host: "127.example.invalid", port: 0),
                configuration: configuration,
                backend: backend)
            XCTFail("A hostname with a loopback-looking prefix must still require authentication")
        } catch let error as ServingHTTPServerError {
            XCTAssertEqual(error, .remoteBindRequiresAuthentication)
        }

        XCTAssertEqual(backend.snapshot().startCount, 0)
    }

    func testRealSocketCloseCancelsActiveSSELeaseExactlyOnce() async throws {
        let backend = SocketBackend(script: .oneDeltaThenHold)
        let server = try await ServingHTTPServer.start(
            configuration: socketConfiguration(requiredBearerToken: nil),
            backend: backend)
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)

        do {
            let client = try await ClientBootstrap(group: clientGroup)
                .connect(to: server.localAddress)
                .get()
            try await sendRawStreamingRequest(on: client)
            try await waitUntilSocketTest {
                backend.snapshot().publishedDeltaCount == 1
            }

            try await client.close().get()
            try await waitUntilSocketTest {
                backend.snapshot().cancelCount == 1
            }

            XCTAssertEqual(backend.snapshot().startCount, 1)
            XCTAssertEqual(backend.snapshot().cancelCount, 1)
            let leaseState = await backend.lastLeaseState()
            XCTAssertEqual(leaseState, .cancelled(.clientDisconnected))
        } catch {
            try? await server.shutdown(gracePeriod: .seconds(1))
            try? await clientGroup.shutdownGracefully()
            throw error
        }

        try await server.shutdown(gracePeriod: .seconds(1))
        try await clientGroup.shutdownGracefully()
        let activeConnectionCount = await server.activeConnectionCount
        XCTAssertEqual(activeConnectionCount, 0)
    }

    func testRealSlowReaderTimesOutAndMailboxNeverExceedsCapacity() async throws {
        let backend = SocketBackend(script: .unboundedLargeDeltas)
        let server = try await ServingHTTPServer.start(
            configuration: socketConfiguration(
                requiredBearerToken: nil,
                backpressureStallTimeout: .milliseconds(100)),
            backend: backend,
            tuning: .init(
                writeBufferLowWaterMarkBytes: 512,
                writeBufferHighWaterMarkBytes: 1_024,
                socketSendBufferBytes: 1_024))
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)

        do {
            let client = try await ClientBootstrap(group: clientGroup)
                .channelOption(.autoRead, value: false)
                .connect(to: server.localAddress)
                .get()
            try await sendRawStreamingRequest(on: client)

            try await waitUntilSocketTest(timeout: .seconds(10)) {
                backend.snapshot().cancelCount == 1
            }

            let mailbox = try XCTUnwrap(backend.snapshot().lastMailbox)
            let snapshot = await mailbox.snapshot()
            XCTAssertLessThanOrEqual(snapshot.bufferedDeltas, 1)
            XCTAssertLessThanOrEqual(snapshot.bufferedBytes, 65_536)
            XCTAssertEqual(backend.snapshot().cancelCount, 1)
            let leaseState = await backend.lastLeaseState()
            XCTAssertEqual(leaseState, .cancelled(.backpressureTimeout))
            try? await client.close().get()
        } catch {
            try? await server.shutdown(gracePeriod: .seconds(1))
            try? await clientGroup.shutdownGracefully()
            throw error
        }

        try await server.shutdown(gracePeriod: .seconds(1))
        try await clientGroup.shutdownGracefully()
    }

    func testStructuredShutdownStopsListenerAndCancelsActiveRequest() async throws {
        let backend = SocketBackend(script: .oneDeltaThenHold)
        let server = try await ServingHTTPServer.start(
            configuration: socketConfiguration(requiredBearerToken: nil),
            backend: backend)
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let client = try await ClientBootstrap(group: clientGroup)
            .connect(to: server.localAddress)
            .get()
        try await sendRawStreamingRequest(on: client)
        try await waitUntilSocketTest {
            backend.snapshot().publishedDeltaCount == 1
        }

        try await server.shutdown(gracePeriod: .seconds(1))
        try await waitUntilSocketTest {
            backend.snapshot().cancelCount == 1
        }

        let leaseState = await backend.lastLeaseState()
        let activeConnectionCount = await server.activeConnectionCount
        XCTAssertEqual(leaseState, .cancelled(.shutdown))
        XCTAssertEqual(backend.snapshot().shutdownCount, 1)
        XCTAssertEqual(activeConnectionCount, 0)
        do {
            _ = try await ClientBootstrap(group: clientGroup)
                .connect(to: server.localAddress)
                .get()
            XCTFail("A shut down server must not accept a replacement connection")
        } catch {
            // Refused connection is the observable listener-release proof.
        }

        try? await client.close().get()
        try await clientGroup.shutdownGracefully()
    }

    func testShutdownGraceBoundsSuspendedBackendShutdownAndClosesSockets() async throws {
        let backend = SuspendedShutdownBackend()
        let server = try await ServingHTTPServer.start(
            configuration: socketConfiguration(requiredBearerToken: nil),
            backend: backend)
        let clientGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let client = try await ClientBootstrap(group: clientGroup)
            .connect(to: server.localAddress)
            .get()
        try await sendRawStreamingRequest(on: client)
        try await waitUntilSocketTest {
            await backend.snapshot().startCount == 1
        }

        let clock = ContinuousClock()
        let started = clock.now
        do {
            try await server.shutdown(gracePeriod: .milliseconds(100))
            XCTFail("A suspended backend shutdown must time out instead of exceeding grace")
        } catch let error as ServingHTTPServerError {
            XCTAssertEqual(error, .backendShutdownTimedOut)
        }
        let elapsed = started.duration(to: clock.now)
        XCTAssertLessThan(elapsed, .milliseconds(500))

        let activeConnectionCount = await server.activeConnectionCount
        let backendSnapshot = await backend.snapshot()
        XCTAssertEqual(activeConnectionCount, 0)
        XCTAssertTrue(backendSnapshot.shutdownStarted)
        XCTAssertFalse(backendSnapshot.shutdownFinished)
        do {
            _ = try await ClientBootstrap(group: clientGroup)
                .connect(to: server.localAddress)
                .get()
            XCTFail("The listener must be closed even when backend shutdown times out")
        } catch {
            // Refused connection proves listener release before the typed timeout returns.
        }

        await backend.releaseShutdown()
        try? await client.close().get()
        try await clientGroup.shutdownGracefully()
    }
}

private func socketConfiguration(
    requiredBearerToken: String?,
    backpressureStallTimeout: Duration = .seconds(1)
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: requiredBearerToken,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: backpressureStallTimeout)
}

private func sendRawStreamingRequest(on channel: any Channel) async throws {
    let body = """
    {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":true}
    """
    let request = """
    POST /v1/chat/completions HTTP/1.1\r
    Host: 127.0.0.1\r
    Content-Type: application/json\r
    Content-Length: \(body.utf8.count)\r
    \r
    \(body)
    """
    try await channel.writeAndFlush(ByteBuffer(string: request)).get()
}

private func waitUntilSocketTest(
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
    XCTFail("Condition was not reached before the socket-test deadline")
}

private final class SocketBackend: ServingGenerationBackend, Sendable {
    enum Script: Sendable {
        case held
        case oneDeltaThenHold
        case unboundedLargeDeltas
    }

    struct Snapshot: Sendable {
        let startCount: Int
        let cancelCount: Int
        let shutdownCount: Int
        let publishedDeltaCount: Int
        let lastMailbox: BoundedDeltaMailbox?
        let lastLease: ServingRequestLease?
    }

    private struct State: Sendable {
        var startCount = 0
        var cancelCount = 0
        var shutdownCount = 0
        var publishedDeltaCount = 0
        var lastMailbox: BoundedDeltaMailbox?
        var lastLease: ServingRequestLease?
    }

    private let script: Script
    private let state = OSAllocatedUnfairLock(initialState: State())

    init(script: Script) {
        self.script = script
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: .init(maxDeltas: 1, maxBytes: 65_536))
        let lease = ServingRequestLease(
            id: ServingRequestID("socket-request"),
            onCancel: { [self, mailbox] in
                state.withLock { $0.cancelCount += 1 }
                await mailbox.cancel(.clientDisconnected)
            })
        state.withLock {
            $0.startCount += 1
            $0.lastMailbox = mailbox
            $0.lastLease = lease
        }

        switch script {
        case .held:
            break
        case .oneDeltaThenHold:
            Task { [self, mailbox] in
                do {
                    try await mailbox.send(.text("started"))
                    state.withLock { $0.publishedDeltaCount += 1 }
                } catch {
                    // The lease state is the cancellation authority.
                }
            }
        case .unboundedLargeDeltas:
            Task { [self, mailbox] in
                let delta = String(repeating: "x", count: 65_536)
                do {
                    while true {
                        try await mailbox.send(.text(delta))
                        state.withLock { $0.publishedDeltaCount += 1 }
                    }
                } catch {
                    // The lease state is the cancellation authority.
                }
            }
        }

        return ServingGenerationHandle(
            responseID: "chatcmpl-socket",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }

    func shutdown() async {
        state.withLock { $0.shutdownCount += 1 }
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                startCount: $0.startCount,
                cancelCount: $0.cancelCount,
                shutdownCount: $0.shutdownCount,
                publishedDeltaCount: $0.publishedDeltaCount,
                lastMailbox: $0.lastMailbox,
                lastLease: $0.lastLease)
        }
    }

    func lastLeaseState() async -> ServingRequestLeaseState? {
        guard let lease = snapshot().lastLease else {
            return nil
        }
        return await lease.state
    }
}

private actor SuspendedShutdownBackend: ServingGenerationBackend {
    struct Snapshot: Sendable {
        let startCount: Int
        let shutdownStarted: Bool
        let shutdownFinished: Bool
    }

    private var startCount = 0
    private var shutdownStarted = false
    private var shutdownFinished = false
    private var shutdownContinuation: CheckedContinuation<Void, Never>?

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: .init(maxDeltas: 1, maxBytes: 65_536))
        let lease = ServingRequestLease(
            id: ServingRequestID("suspended-shutdown-request"),
            onCancelWithReason: { reason in
                await mailbox.cancel(reason)
            })
        startCount += 1
        return ServingGenerationHandle(
            responseID: "chatcmpl-suspended-shutdown",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }

    func shutdown() async {
        shutdownStarted = true
        await withCheckedContinuation { continuation in
            shutdownContinuation = continuation
        }
        shutdownFinished = true
    }

    func releaseShutdown() {
        shutdownContinuation?.resume()
        shutdownContinuation = nil
    }

    func snapshot() -> Snapshot {
        Snapshot(
            startCount: startCount,
            shutdownStarted: shutdownStarted,
            shutdownFinished: shutdownFinished)
    }
}
