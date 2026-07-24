import Foundation

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
    private var cancelAction: (@Sendable () async -> Void)?

    public init(
        id: ServingRequestID,
        onCancel: (@Sendable () async -> Void)? = nil
    ) {
        self.id = id
        self.cancelAction = onCancel
    }

    public var state: ServingRequestLeaseState {
        leaseState
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
        let action = cancelAction
        cancelAction = nil
        await action?()
        return true
    }

    @discardableResult
    private func transitionToTerminal(_ terminalState: ServingRequestLeaseState) -> Bool {
        guard !leaseState.isTerminal else {
            return false
        }
        leaseState = terminalState
        cancelAction = nil
        return true
    }
}
