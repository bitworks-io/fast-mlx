import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import ServingCore
import os

public enum ServingHTTPServerError: Error, Equatable, Sendable {
    case invalidBindHost
    case invalidBindPort
    case remoteBindRequiresAuthentication
    case listenerAddressUnavailable
    case serverStopping
    case backendShutdownTimedOut
}

public struct ServingHTTPBind: Equatable, Sendable {
    public static let loopbackEphemeral = ServingHTTPBind(host: "127.0.0.1", port: 0)

    public let host: String
    public let port: Int

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }

    fileprivate func validate(requiredBearerToken: String?) throws {
        let normalizedHost = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalizedHost.isEmpty else {
            throw ServingHTTPServerError.invalidBindHost
        }
        guard (0...65_535).contains(port) else {
            throw ServingHTTPServerError.invalidBindPort
        }
        guard Self.isLoopback(normalizedHost) || requiredBearerToken != nil else {
            throw ServingHTTPServerError.remoteBindRequiresAuthentication
        }
    }

    private static func isLoopback(_ host: String) -> Bool {
        if host == "localhost"
            || host == "localhost."
            || host == "::1"
            || host == "[::1]"
        {
            return true
        }

        let octets = host.split(separator: ".", omittingEmptySubsequences: false)
        guard octets.count == 4,
            let first = Int(octets[0]),
            first == 127
        else {
            return false
        }
        return octets.dropFirst().allSatisfy {
            guard let octet = Int($0) else {
                return false
            }
            return (0...255).contains(octet)
        }
    }
}

public struct ServingHTTPTransportTuning: Equatable, Sendable {
    public static let productionDefault = ServingHTTPTransportTuning(
        writeBufferLowWaterMarkBytes: 32 * 1_024,
        writeBufferHighWaterMarkBytes: 64 * 1_024,
        socketSendBufferBytes: nil)

    public let writeBufferLowWaterMarkBytes: Int
    public let writeBufferHighWaterMarkBytes: Int
    public let socketSendBufferBytes: Int?

    public init(
        writeBufferLowWaterMarkBytes: Int,
        writeBufferHighWaterMarkBytes: Int,
        socketSendBufferBytes: Int?
    ) {
        precondition(
            writeBufferLowWaterMarkBytes >= 1,
            "writeBufferLowWaterMarkBytes must be positive")
        precondition(
            writeBufferHighWaterMarkBytes >= writeBufferLowWaterMarkBytes,
            "writeBufferHighWaterMarkBytes must be at least the low watermark")
        if let socketSendBufferBytes {
            precondition(socketSendBufferBytes > 0, "socketSendBufferBytes must be positive")
            precondition(
                socketSendBufferBytes <= Int(Int32.max),
                "socketSendBufferBytes exceeds the platform socket-option range")
        }

        self.writeBufferLowWaterMarkBytes = writeBufferLowWaterMarkBytes
        self.writeBufferHighWaterMarkBytes = writeBufferHighWaterMarkBytes
        self.socketSendBufferBytes = socketSendBufferBytes
    }
}

public actor ServingHTTPServer {
    public nonisolated let localAddress: SocketAddress

    private let listener: any Channel
    private let eventLoopGroup: MultiThreadedEventLoopGroup
    private let connections: ServingHTTPConnectionRegistry
    private let backend: any ServingGenerationBackend
    private var isShutDown = false

    private init(
        listener: any Channel,
        eventLoopGroup: MultiThreadedEventLoopGroup,
        connections: ServingHTTPConnectionRegistry,
        backend: any ServingGenerationBackend,
        localAddress: SocketAddress
    ) {
        self.listener = listener
        self.eventLoopGroup = eventLoopGroup
        self.connections = connections
        self.backend = backend
        self.localAddress = localAddress
    }

    public static func start(
        bind: ServingHTTPBind = .loopbackEphemeral,
        configuration: ServingHTTPConfiguration,
        backend: any ServingGenerationBackend,
        eventLoopThreads: Int = 1,
        tuning: ServingHTTPTransportTuning = .productionDefault
    ) async throws -> ServingHTTPServer {
        try bind.validate(requiredBearerToken: configuration.requiredBearerToken)
        precondition(eventLoopThreads > 0, "eventLoopThreads must be positive")

        let group = MultiThreadedEventLoopGroup(numberOfThreads: eventLoopThreads)
        let connections = ServingHTTPConnectionRegistry()
        var bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(.socketOption(.tcp_nodelay), value: 1)
            .childChannelOption(
                .writeBufferWaterMark,
                value: .init(
                    low: tuning.writeBufferLowWaterMarkBytes,
                    high: tuning.writeBufferHighWaterMarkBytes))
            .childChannelInitializer { channel in
                channel.eventLoop.makeCompletedFuture {
                    let pipeline = channel.pipeline.syncOperations
                    try pipeline.configureHTTPServerPipeline(
                        withPipeliningAssistance: false,
                        withErrorHandling: true)
                    try pipeline.addHandler(
                        OpenAIChatCompletionsHTTPHandler(
                            configuration: configuration,
                            backend: backend))
                    guard connections.register(channel) else {
                        throw ServingHTTPServerError.serverStopping
                    }
                    channel.closeFuture.whenComplete { _ in
                        connections.remove(channel)
                    }
                }
            }
        if let socketSendBufferBytes = tuning.socketSendBufferBytes {
            bootstrap = bootstrap.childChannelOption(
                .socketOption(.so_sndbuf),
                value: Int32(socketSendBufferBytes))
        }

        do {
            let listener = try await bootstrap
                .bind(host: bind.host, port: bind.port)
                .get()
            guard let localAddress = listener.localAddress else {
                try? await listener.close().get()
                throw ServingHTTPServerError.listenerAddressUnavailable
            }
            return ServingHTTPServer(
                listener: listener,
                eventLoopGroup: group,
                connections: connections,
                backend: backend,
                localAddress: localAddress)
        } catch {
            try? await group.shutdownGracefully()
            throw error
        }
    }

    public var activeConnectionCount: Int {
        connections.activeCount
    }

    public func shutdown(gracePeriod: Duration) async throws {
        precondition(gracePeriod > .zero, "gracePeriod must be positive")
        guard !isShutDown else {
            return
        }
        isShutDown = true

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: gracePeriod)
        let activeChannels = connections.stopAcceptingAndSnapshot()
        let backendShutdownCompletion = ServingHTTPBackendShutdownCompletion()
        let backend = self.backend
        let backendShutdownTask = Task { [backend, backendShutdownCompletion] in
            await backend.shutdown()
            backendShutdownCompletion.markComplete()
        }

        try? await listener.close().get()
        for channel in activeChannels {
            channel.pipeline.fireUserInboundEventTriggered(ChannelShouldQuiesceEvent())
        }

        for channel in activeChannels {
            let remaining = clock.now.duration(to: deadline)
            if remaining <= .zero {
                try? await channel.close().get()
                continue
            }
            let forcedClose = channel.eventLoop.scheduleTask(
                in: servingNIOTimeAmount(for: remaining)
            ) {
                channel.close(promise: nil)
            }
            try? await channel.closeFuture.get()
            forcedClose.cancel()
        }

        let backendShutdownCompleted = await backendShutdownCompletion.waitUntilComplete(
            clock: clock,
            deadline: deadline)
        if !backendShutdownCompleted {
            backendShutdownTask.cancel()
        }

        try await eventLoopGroup.shutdownGracefully()

        if !backendShutdownCompleted {
            throw ServingHTTPServerError.backendShutdownTimedOut
        }
    }
}

private final class ServingHTTPBackendShutdownCompletion: Sendable {
    private let completed = OSAllocatedUnfairLock(initialState: false)

    func markComplete() {
        completed.withLock { $0 = true }
    }

    func waitUntilComplete(
        clock: ContinuousClock,
        deadline: ContinuousClock.Instant
    ) async -> Bool {
        while clock.now < deadline {
            if completed.withLock({ $0 }) {
                return true
            }
            let remaining = clock.now.duration(to: deadline)
            guard remaining > .zero else {
                break
            }
            try? await Task.sleep(
                for: remaining < .milliseconds(1) ? remaining : .milliseconds(1))
        }
        return completed.withLock { $0 }
    }
}

private final class ServingHTTPConnectionRegistry: Sendable {
    private struct State: Sendable {
        var accepting = true
        var channels: [ObjectIdentifier: any Channel] = [:]
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var activeCount: Int {
        state.withLock { $0.channels.count }
    }

    func register(_ channel: any Channel) -> Bool {
        state.withLock { state in
            guard state.accepting else {
                return false
            }
            state.channels[ObjectIdentifier(channel)] = channel
            return true
        }
    }

    func remove(_ channel: any Channel) {
        _ = state.withLock { state in
            state.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func stopAcceptingAndSnapshot() -> [any Channel] {
        state.withLock { state in
            state.accepting = false
            return Array(state.channels.values)
        }
    }
}

func servingNIOTimeAmount(for duration: Duration) -> TimeAmount {
    let components = duration.components
    let seconds = max(components.seconds, 0)
    let subsecondNanoseconds = max(components.attoseconds / 1_000_000_000, 0)
    let (secondNanoseconds, secondsOverflow) = seconds.multipliedReportingOverflow(
        by: 1_000_000_000)
    guard !secondsOverflow else {
        return .nanoseconds(.max)
    }
    let (totalNanoseconds, totalOverflow) = secondNanoseconds.addingReportingOverflow(
        subsecondNanoseconds)
    return .nanoseconds(totalOverflow ? .max : totalNanoseconds)
}
