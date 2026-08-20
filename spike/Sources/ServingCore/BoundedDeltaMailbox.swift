import Foundation
import os

private final class PendingContinuationGate: Sendable {
    private enum State: Sendable {
        case pending
        case claimed
        case cancelled
    }

    private let state = OSAllocatedUnfairLock(initialState: State.pending)

    func claim() -> Bool {
        state.withLock { state in
            guard state == .pending else {
                return false
            }
            state = .claimed
            return true
        }
    }

    func cancel() -> Bool {
        state.withLock { state in
            guard state == .pending else {
                return false
            }
            state = .cancelled
            return true
        }
    }
}

public enum ServingResponseDelta: Equatable, Sendable {
    case text(String)
    case toolCalls([OpenAIToolCall])
    case completion(ServingGenerationCompletion)

    public var utf8ByteCount: Int {
        switch self {
        case .text(let text):
            return text.utf8.count
        case .toolCalls:
            return 0
        case .completion:
            return 0
        }
    }
}

public enum ServingCancellationReason: String, Codable, Equatable, Sendable {
    case backpressureTimeout
    case clientDisconnected
    case responseLimitExceeded
    case shutdown
}

public enum ServingMailboxError: Error, Equatable, Sendable {
    case backend(String)
    case cancelled(ServingCancellationReason)
}

public enum ServingMailboxTerminal: Equatable, Sendable {
    case finished
    case failed(String)
    case cancelled(ServingCancellationReason)
}

public actor BoundedDeltaMailbox {
    public struct Capacity: Equatable, Sendable {
        public let maxDeltas: Int
        public let maxBytes: Int

        public init(maxDeltas: Int, maxBytes: Int) {
            precondition(maxDeltas > 0, "maxDeltas must be positive")
            precondition(maxBytes > 0, "maxBytes must be positive")
            self.maxDeltas = maxDeltas
            self.maxBytes = maxBytes
        }
    }

    public struct Snapshot: Equatable, Sendable {
        public let bufferedDeltas: Int
        public let bufferedBytes: Int
        public let waitingProducers: Int
        public let waitingConsumers: Int
        public let terminal: ServingMailboxTerminal?
    }

    private struct PendingSend {
        let id: UUID
        let gate: PendingContinuationGate
        let delta: ServingResponseDelta
        let continuation: CheckedContinuation<Void, Error>
    }

    private struct PendingReceive {
        let id: UUID
        let gate: PendingContinuationGate
        let continuation: CheckedContinuation<ServingResponseDelta?, Error>
    }

    private let capacity: Capacity
    private var buffered: [ServingResponseDelta] = []
    private var bufferedBytes = 0
    private var pendingSends: [PendingSend] = []
    private var pendingReceives: [PendingReceive] = []
    private var terminal: ServingMailboxTerminal?

    public init(capacity: Capacity) {
        self.capacity = capacity
    }

    public func send(_ delta: ServingResponseDelta) async throws {
        try Task.checkCancellation()
        if let terminal {
            throw error(for: terminal)
        }
        guard delta.utf8ByteCount <= capacity.maxBytes else {
            throw ServingMailboxError.backend("delta exceeds mailbox byte capacity")
        }

        while pendingSends.isEmpty, buffered.isEmpty, !pendingReceives.isEmpty {
            let receiver = pendingReceives.removeFirst()
            guard receiver.gate.claim() else {
                receiver.continuation.resume(throwing: CancellationError())
                continue
            }
            receiver.continuation.resume(returning: delta)
            return
        }

        if pendingSends.isEmpty, canBuffer(delta) {
            append(delta)
            return
        }

        let id = UUID()
        let gate = PendingContinuationGate()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pendingSends.append(
                    PendingSend(
                        id: id,
                        gate: gate,
                        delta: delta,
                        continuation: continuation))
            }
        } onCancel: {
            if gate.cancel() {
                Task {
                    await self.cancelPendingSend(id: id)
                }
            }
        }
    }

    public func next() async throws -> ServingResponseDelta? {
        try Task.checkCancellation()
        if !buffered.isEmpty {
            let delta = removeFirstBuffered()
            flushPendingSends()
            return delta
        }

        if let terminal {
            switch terminal {
            case .finished:
                return nil
            case .failed, .cancelled:
                throw error(for: terminal)
            }
        }

        let id = UUID()
        let gate = PendingContinuationGate()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pendingReceives.append(
                    PendingReceive(id: id, gate: gate, continuation: continuation))
            }
        } onCancel: {
            if gate.cancel() {
                Task {
                    await self.cancelPendingReceive(id: id)
                }
            }
        }
    }

    public func finish() {
        guard terminal == nil else { return }
        terminal = .finished
        resumeReceiversForTerminal(.finished)
        resumeSendersForTerminal(.finished)
    }

    public func fail(_ error: ServingMailboxError) {
        guard terminal == nil else { return }
        let newTerminal: ServingMailboxTerminal
        switch error {
        case .backend(let message):
            newTerminal = .failed(message)
        case .cancelled(let reason):
            newTerminal = .cancelled(reason)
        }
        terminal = newTerminal
        buffered.removeAll()
        bufferedBytes = 0
        resumeReceiversForTerminal(newTerminal)
        resumeSendersForTerminal(newTerminal)
    }

    public func cancel(_ reason: ServingCancellationReason) {
        fail(.cancelled(reason))
    }

    public func snapshot() -> Snapshot {
        Snapshot(
            bufferedDeltas: buffered.count,
            bufferedBytes: bufferedBytes,
            waitingProducers: pendingSends.count,
            waitingConsumers: pendingReceives.count,
            terminal: terminal)
    }

    private func canBuffer(_ delta: ServingResponseDelta) -> Bool {
        buffered.count < capacity.maxDeltas
            && bufferedBytes + delta.utf8ByteCount <= capacity.maxBytes
    }

    private func append(_ delta: ServingResponseDelta) {
        buffered.append(delta)
        bufferedBytes += delta.utf8ByteCount
    }

    private func removeFirstBuffered() -> ServingResponseDelta {
        let delta = buffered.removeFirst()
        bufferedBytes -= delta.utf8ByteCount
        return delta
    }

    private func flushPendingSends() {
        while terminal == nil, let pending = pendingSends.first, canBuffer(pending.delta) {
            pendingSends.removeFirst()
            guard pending.gate.claim() else {
                pending.continuation.resume(throwing: CancellationError())
                continue
            }
            append(pending.delta)
            pending.continuation.resume()
        }
    }

    private func resumeReceiversForTerminal(_ terminal: ServingMailboxTerminal) {
        let receivers = pendingReceives
        pendingReceives.removeAll()
        for receiver in receivers {
            guard receiver.gate.claim() else {
                receiver.continuation.resume(throwing: CancellationError())
                continue
            }
            switch terminal {
            case .finished:
                receiver.continuation.resume(returning: nil)
            case .failed, .cancelled:
                receiver.continuation.resume(throwing: error(for: terminal))
            }
        }
    }

    private func resumeSendersForTerminal(_ terminal: ServingMailboxTerminal) {
        let senders = pendingSends
        pendingSends.removeAll()
        for sender in senders {
            guard sender.gate.claim() else {
                sender.continuation.resume(throwing: CancellationError())
                continue
            }
            sender.continuation.resume(throwing: error(for: terminal))
        }
    }

    private func cancelPendingSend(id: UUID) {
        guard let index = pendingSends.firstIndex(where: { $0.id == id }) else {
            return
        }
        let pending = pendingSends.remove(at: index)
        pending.continuation.resume(throwing: CancellationError())
    }

    private func cancelPendingReceive(id: UUID) {
        guard let index = pendingReceives.firstIndex(where: { $0.id == id }) else {
            return
        }
        let pending = pendingReceives.remove(at: index)
        pending.continuation.resume(throwing: CancellationError())
    }

    private func error(for terminal: ServingMailboxTerminal) -> ServingMailboxError {
        switch terminal {
        case .finished:
            return .backend("mailbox is finished")
        case .failed(let message):
            return .backend(message)
        case .cancelled(let reason):
            return .cancelled(reason)
        }
    }
}

public struct BackpressureStallState: Equatable, Sendable {
    public private(set) var deadlineTick: Int?
    private let maxStallTicks: Int
    private var expiredDeadlineTick: Int?

    public init(maxStallTicks: Int) {
        precondition(maxStallTicks >= 0, "maxStallTicks must be non-negative")
        self.maxStallTicks = maxStallTicks
    }

    public mutating func observe(isFull: Bool, tick: Int) -> BackpressureStallDecision {
        if !isFull {
            let hadDeadline = deadlineTick != nil || expiredDeadlineTick != nil
            deadlineTick = nil
            expiredDeadlineTick = nil
            return hadDeadline ? .cleared : .none
        }

        if let expiredDeadlineTick {
            return .expired(deadlineTick: expiredDeadlineTick)
        }

        if let deadlineTick {
            if tick >= deadlineTick {
                expiredDeadlineTick = deadlineTick
                return .expired(deadlineTick: deadlineTick)
            }
            return .waiting(deadlineTick: deadlineTick)
        }

        let newDeadline = tick + maxStallTicks
        deadlineTick = newDeadline
        if tick >= newDeadline {
            expiredDeadlineTick = newDeadline
            return .expired(deadlineTick: newDeadline)
        }
        return .started(deadlineTick: newDeadline)
    }
}

public enum BackpressureStallDecision: Equatable, Sendable {
    case none
    case started(deadlineTick: Int)
    case waiting(deadlineTick: Int)
    case expired(deadlineTick: Int)
    case cleared
}
