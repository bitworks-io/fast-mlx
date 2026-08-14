import Foundation
import os

private final class ContinuousBatchContinuationGate: Sendable {
    private enum State: Sendable {
        case pending
        case claimed
        case cancelled
    }

    private let state = OSAllocatedUnfairLock(initialState: State.pending)

    func claim() -> Bool {
        state.withLock { state in
            guard state == .pending else { return false }
            state = .claimed
            return true
        }
    }

    func cancel() -> Bool {
        state.withLock { state in
            guard state == .pending else { return false }
            state = .cancelled
            return true
        }
    }
}

final class ContinuousBatchCancellationHook: Sendable {
    private enum Phase: Sendable {
        case active
        case fired
        case completed
    }

    private struct State: Sendable {
        var phase = Phase.active
        var activeIterators = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let action: @Sendable () -> Void

    init(action: @escaping @Sendable () -> Void) {
        self.action = action
    }

    func fire() {
        let shouldFire = state.withLock { state in
            guard state.phase == .active else { return false }
            state.phase = .fired
            return true
        }
        performAction(if: shouldFire)
    }

    func complete() {
        state.withLock { state in
            guard state.phase == .active else { return }
            state.phase = .completed
        }
    }

    func iteratorStarted() {
        state.withLock { state in
            guard state.phase == .active else { return }
            state.activeIterators += 1
        }
    }

    func iteratorReleased() {
        let shouldFire = state.withLock { state in
            guard state.phase == .active else { return false }
            if state.activeIterators > 0 {
                state.activeIterators -= 1
            }
            guard state.activeIterators == 0 else { return false }
            state.phase = .fired
            return true
        }
        performAction(if: shouldFire)
    }

    func streamReleased() {
        let shouldFire = state.withLock { state in
            guard state.phase == .active, state.activeIterators == 0 else {
                return false
            }
            state.phase = .fired
            return true
        }
        performAction(if: shouldFire)
    }

    private func performAction(if shouldFire: Bool) {
        if shouldFire {
            action()
        }
    }
}

fileprivate final class ContinuousBatchStreamLease: Sendable {
    private let cancellationHook: ContinuousBatchCancellationHook

    init(cancellationHook: ContinuousBatchCancellationHook) {
        self.cancellationHook = cancellationHook
    }

    deinit {
        cancellationHook.streamReleased()
    }
}

fileprivate final class ContinuousBatchIteratorLease: Sendable {
    private let cancellationHook: ContinuousBatchCancellationHook

    init(cancellationHook: ContinuousBatchCancellationHook) {
        self.cancellationHook = cancellationHook
        cancellationHook.iteratorStarted()
    }

    deinit {
        cancellationHook.iteratorReleased()
    }
}

public enum ContinuousBatchTokenTerminal: Sendable, Equatable {
    case finished
    case failed(String)
    case cancelled
}

public enum ContinuousBatchTokenStreamError: Error, Sendable, Equatable {
    case closed(ContinuousBatchTokenTerminal)
    case invalidReservation
}

public struct ContinuousBatchTokenStreamSnapshot: Sendable, Equatable {
    public let bufferedTokens: Int
    public let reservedTokens: Int
    public let waitingProducers: Int
    public let waitingConsumers: Int
    public let terminal: ContinuousBatchTokenTerminal?
}

struct ContinuousBatchPublicationReservation: Sendable, Hashable {
    fileprivate let id: UUID
}

actor ContinuousBatchTokenMailbox {
    private struct PendingReservation {
        let id: UUID
        let gate: ContinuousBatchContinuationGate
        let continuation: CheckedContinuation<ContinuousBatchPublicationReservation, Error>
    }

    private struct PendingReceive {
        let id: UUID
        let gate: ContinuousBatchContinuationGate
        let continuation: CheckedContinuation<Int?, Error>
    }

    private let capacity: Int
    private var buffered: [Int] = []
    private var reservations: Set<ContinuousBatchPublicationReservation> = []
    private var pendingReservations: [PendingReservation] = []
    private var pendingReceives: [PendingReceive] = []
    private var terminal: ContinuousBatchTokenTerminal?
    private var failure: (any Error)?
    private let consumerCancellation: ContinuousBatchCancellationHook

    init(
        capacity: Int,
        consumerCancellation: ContinuousBatchCancellationHook
    ) {
        precondition(capacity > 0, "continuous batch publication capacity must be positive")
        self.capacity = capacity
        self.consumerCancellation = consumerCancellation
    }

    func tryReserve() throws -> ContinuousBatchPublicationReservation? {
        if let terminal { throw error(for: terminal) }
        guard pendingReservations.isEmpty, usedCapacity < capacity else {
            return nil
        }
        return makeReservation()
    }

    func reserve() async throws -> ContinuousBatchPublicationReservation {
        try Task.checkCancellation()
        if let terminal { throw error(for: terminal) }

        if pendingReservations.isEmpty, usedCapacity < capacity {
            return makeReservation()
        }

        let id = UUID()
        let gate = ContinuousBatchContinuationGate()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pendingReservations.append(
                    PendingReservation(
                        id: id,
                        gate: gate,
                        continuation: continuation))
            }
        } onCancel: {
            if gate.cancel() {
                Task { await self.cancelPendingReservation(id: id) }
            }
        }
    }

    /// Completes publication that the scheduler/runtime transaction has already committed.
    ///
    /// Task cancellation cannot discard committed visible tokens. Request cancellation or
    /// shutdown still terminates the mailbox and resumes this waiter with the terminal error.
    func reserveCommitted() async throws -> ContinuousBatchPublicationReservation {
        if let terminal { throw error(for: terminal) }

        if pendingReservations.isEmpty, usedCapacity < capacity {
            return makeReservation()
        }

        let id = UUID()
        let gate = ContinuousBatchContinuationGate()
        return try await withCheckedThrowingContinuation { continuation in
            pendingReservations.append(
                PendingReservation(
                    id: id,
                    gate: gate,
                    continuation: continuation))
        }
    }

    func commit(
        _ reservation: ContinuousBatchPublicationReservation,
        token: Int
    ) throws {
        if let terminal { throw error(for: terminal) }
        guard reservations.remove(reservation) != nil else {
            throw ContinuousBatchTokenStreamError.invalidReservation
        }

        while !pendingReceives.isEmpty {
            let receiver = pendingReceives.removeFirst()
            guard receiver.gate.claim() else {
                receiver.continuation.resume(throwing: CancellationError())
                continue
            }
            receiver.continuation.resume(returning: token)
            flushPendingReservations()
            return
        }

        buffered.append(token)
    }

    func release(_ reservation: ContinuousBatchPublicationReservation) {
        guard reservations.remove(reservation) != nil else { return }
        flushPendingReservations()
    }

    func send(_ token: Int) async throws {
        let reservation = try await reserve()
        do {
            try Task.checkCancellation()
            try commit(reservation, token: token)
        } catch {
            release(reservation)
            throw error
        }
    }

    func next() async throws -> Int? {
        try Task.checkCancellation()
        if !buffered.isEmpty {
            let token = buffered.removeFirst()
            flushPendingReservations()
            return token
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
        let gate = ContinuousBatchContinuationGate()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pendingReceives.append(
                    PendingReceive(
                        id: id,
                        gate: gate,
                        continuation: continuation))
            }
        } onCancel: {
            if gate.cancel() {
                Task { await self.cancelPendingReceive(id: id) }
            }
        }
    }

    func finish() {
        guard terminal == nil else { return }
        terminal = .finished
        consumerCancellation.complete()
        discardReservations(for: .finished)
        resumeReceivers(for: .finished)
    }

    func fail(_ error: any Error) {
        guard terminal == nil else { return }
        failure = error
        terminate(.failed(String(describing: error)), discardingBufferedTokens: true)
    }

    func cancel() {
        terminate(.cancelled, discardingBufferedTokens: true)
    }

    func snapshot() -> ContinuousBatchTokenStreamSnapshot {
        ContinuousBatchTokenStreamSnapshot(
            bufferedTokens: buffered.count,
            reservedTokens: reservations.count,
            waitingProducers: pendingReservations.count,
            waitingConsumers: pendingReceives.count,
            terminal: terminal)
    }

    private var usedCapacity: Int {
        buffered.count + reservations.count
    }

    private func makeReservation() -> ContinuousBatchPublicationReservation {
        let reservation = ContinuousBatchPublicationReservation(id: UUID())
        reservations.insert(reservation)
        return reservation
    }

    private func flushPendingReservations() {
        while terminal == nil, usedCapacity < capacity, !pendingReservations.isEmpty {
            let pending = pendingReservations.removeFirst()
            guard pending.gate.claim() else {
                pending.continuation.resume(throwing: CancellationError())
                continue
            }
            pending.continuation.resume(returning: makeReservation())
        }
    }

    private func terminate(
        _ newTerminal: ContinuousBatchTokenTerminal,
        discardingBufferedTokens: Bool
    ) {
        guard terminal == nil else { return }
        terminal = newTerminal
        consumerCancellation.complete()
        if discardingBufferedTokens {
            buffered.removeAll(keepingCapacity: false)
        }
        discardReservations(for: newTerminal)
        resumeReceivers(for: newTerminal)
    }

    private func discardReservations(for terminal: ContinuousBatchTokenTerminal) {
        reservations.removeAll()
        let pending = pendingReservations
        pendingReservations.removeAll()
        for waiter in pending {
            guard waiter.gate.claim() else {
                waiter.continuation.resume(throwing: CancellationError())
                continue
            }
            waiter.continuation.resume(throwing: error(for: terminal))
        }
    }

    private func resumeReceivers(for terminal: ContinuousBatchTokenTerminal) {
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

    private func cancelPendingReservation(id: UUID) {
        guard let index = pendingReservations.firstIndex(where: { $0.id == id }) else {
            return
        }
        let pending = pendingReservations.remove(at: index)
        pending.continuation.resume(throwing: CancellationError())
    }

    private func cancelPendingReceive(id: UUID) {
        guard let index = pendingReceives.firstIndex(where: { $0.id == id }) else {
            return
        }
        let pending = pendingReceives.remove(at: index)
        pending.continuation.resume(throwing: CancellationError())
    }

    private func error(
        for terminal: ContinuousBatchTokenTerminal
    ) -> any Error {
        switch terminal {
        case .cancelled:
            CancellationError()
        case .failed:
            failure ?? ContinuousBatchTokenStreamError.closed(terminal)
        case .finished:
            ContinuousBatchTokenStreamError.closed(terminal)
        }
    }
}

public struct ContinuousBatchTokenStream: AsyncSequence, Sendable {
    public typealias Element = Int

    public struct AsyncIterator: AsyncIteratorProtocol, Sendable {
        fileprivate let mailbox: ContinuousBatchTokenMailbox
        fileprivate let cancellationHook: ContinuousBatchCancellationHook
        private let iteratorLease: ContinuousBatchIteratorLease

        fileprivate init(
            mailbox: ContinuousBatchTokenMailbox,
            cancellationHook: ContinuousBatchCancellationHook,
            iteratorLease: ContinuousBatchIteratorLease
        ) {
            self.mailbox = mailbox
            self.cancellationHook = cancellationHook
            self.iteratorLease = iteratorLease
        }

        public mutating func next() async throws -> Int? {
            let mailbox = mailbox
            let cancellationHook = cancellationHook
            return try await withTaskCancellationHandler {
                try await mailbox.next()
            } onCancel: {
                cancellationHook.fire()
            }
        }
    }

    private let mailbox: ContinuousBatchTokenMailbox
    private let cancellationHook: ContinuousBatchCancellationHook
    private let streamLease: ContinuousBatchStreamLease

    init(
        mailbox: ContinuousBatchTokenMailbox,
        cancellationHook: ContinuousBatchCancellationHook
    ) {
        self.mailbox = mailbox
        self.cancellationHook = cancellationHook
        self.streamLease = ContinuousBatchStreamLease(
            cancellationHook: cancellationHook)
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(
            mailbox: mailbox,
            cancellationHook: cancellationHook,
            iteratorLease: ContinuousBatchIteratorLease(
                cancellationHook: cancellationHook))
    }

    public func snapshot() async -> ContinuousBatchTokenStreamSnapshot {
        await mailbox.snapshot()
    }
}
