import Foundation
import os

public enum ServingRequestLeaseState: Equatable, Sendable {
    case pending
    case active
    case completed
    case failed(String)
    case cancelled(ServingCancellationReason)

    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled:
            return true
        case .pending, .active:
            return false
        }
    }
}

public actor ServingRequestLease {
    public let id: ServingRequestID

    private var leaseState: ServingRequestLeaseState = .pending
    private var cancelAction: (@Sendable (ServingCancellationReason) async -> Void)?
    private nonisolated let terminalCancellation = OSAllocatedUnfairLock(
        initialState: Optional<ServingCancellationReason>.none)

    public init(
        id: ServingRequestID,
        onCancel: (@Sendable () async -> Void)? = nil
    ) {
        self.id = id
        if let onCancel {
            self.cancelAction = { _ in
                await onCancel()
            }
        } else {
            self.cancelAction = nil
        }
    }

    public init(
        id: ServingRequestID,
        onCancelWithReason: @escaping @Sendable (ServingCancellationReason) async -> Void
    ) {
        self.id = id
        self.cancelAction = onCancelWithReason
    }

    public var state: ServingRequestLeaseState {
        leaseState
    }

    public nonisolated var terminalCancellationReason: ServingCancellationReason? {
        terminalCancellation.withLock { $0 }
    }

    @discardableResult
    public func activate() -> Bool {
        guard leaseState == .pending else {
            return false
        }
        leaseState = .active
        return true
    }

    @discardableResult
    public func complete() -> Bool {
        transitionToTerminal(.completed)
    }

    @discardableResult
    public func fail(_ message: String) -> Bool {
        transitionToTerminal(.failed(message))
    }

    @discardableResult
    public func cancel(_ reason: ServingCancellationReason) async -> Bool {
        guard !leaseState.isTerminal else {
            return false
        }
        leaseState = .cancelled(reason)
        terminalCancellation.withLock { $0 = reason }
        let action = cancelAction
        cancelAction = nil
        await action?(reason)
        return true
    }

    /// Records cancellation initiated by the backend without re-entering its
    /// lease callback. Backend owners use this while atomically draining work.
    @discardableResult
    public func cancelFromBackend(_ reason: ServingCancellationReason) -> Bool {
        transitionToTerminal(.cancelled(reason))
    }

    @discardableResult
    private func transitionToTerminal(_ terminalState: ServingRequestLeaseState) -> Bool {
        guard !leaseState.isTerminal else {
            return false
        }
        leaseState = terminalState
        if case .cancelled(let reason) = terminalState {
            terminalCancellation.withLock { $0 = reason }
        }
        cancelAction = nil
        return true
    }
}
