import Foundation

public struct ServingRequestID: RawRepresentable, Hashable, Codable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }
}

public enum ServingExecutionRoute: String, Codable, Equatable, Sendable {
    case continuousBatchNoSpec = "continuous-batch-no-spec"
    case scalarGreedy = "scalar-greedy"
    case scriptedTransport = "scripted-transport"
    case soloPLD = "solo-pld"
}

public struct ServingAdmissionConfiguration: Equatable, Sendable {
    public let soloPLDQualified: Bool
    public let maximumBatchRequests: Int
    public let maximumQueuedRequests: Int

    public init(
        soloPLDQualified: Bool,
        maximumBatchRequests: Int = 8,
        maximumQueuedRequests: Int = 128
    ) {
        precondition(maximumBatchRequests > 0, "maximumBatchRequests must be positive")
        precondition(maximumQueuedRequests >= 0, "maximumQueuedRequests must be non-negative")
        self.soloPLDQualified = soloPLDQualified
        self.maximumBatchRequests = maximumBatchRequests
        self.maximumQueuedRequests = maximumQueuedRequests
    }
}

public enum ServingAdmissionRejection: String, Codable, Equatable, Sendable {
    case duplicateRequest
    case queueFull
}

public enum ServingAdmissionDecision: Equatable, Sendable {
    case held([ServingRequestID])
    case queued([ServingRequestID])
    case removedFromHold([ServingRequestID])
    case removedFromExecution(
        remaining: [ServingRequestID],
        replacements: [ServingRequestID],
        nextHeld: [ServingRequestID])
    case joinedContinuousBatch(requests: [ServingRequestID])
    case continuedExecution(
        route: ServingExecutionRoute,
        remaining: [ServingRequestID],
        replacements: [ServingRequestID])
    case rejected(request: ServingRequestID, reason: ServingAdmissionRejection)
    case start(route: ServingExecutionRoute, requests: [ServingRequestID])
    case noRouteChange(route: ServingExecutionRoute)
    case idle
}

public struct ServingAdmissionReducer: Equatable, Sendable {
    private let configuration: ServingAdmissionConfiguration
    private var held: [ServingRequestID] = []
    private var queued: [ServingRequestID] = []
    private var executing: [ServingRequestID] = []

    public private(set) var currentExecutionRoute: ServingExecutionRoute?

    public var heldRequestIDs: [ServingRequestID] { held }
    public var queuedRequestIDs: [ServingRequestID] { queued }
    public var executingRequestIDs: [ServingRequestID] { executing }

    public init(configuration: ServingAdmissionConfiguration) {
        self.configuration = configuration
    }

    public mutating func submit(_ id: ServingRequestID) -> ServingAdmissionDecision {
        guard !contains(id) else {
            return .rejected(request: id, reason: .duplicateRequest)
        }

        if canJoinRunningCohort,
            queued.isEmpty,
            executing.count < configuration.maximumBatchRequests
        {
            executing.append(id)
            return .joinedContinuousBatch(requests: executing)
        }

        if currentExecutionRoute == nil, held.count < configuration.maximumBatchRequests {
            held.append(id)
            return .held(held)
        }

        return enqueue(id)
    }

    public mutating func cancel(_ id: ServingRequestID) -> ServingAdmissionDecision {
        if let index = held.firstIndex(of: id) {
            held.remove(at: index)
            refillHeldFromQueue()
            return .removedFromHold(held)
        }

        if let index = queued.firstIndex(of: id) {
            queued.remove(at: index)
            return .queued(queued)
        }

        if let index = executing.firstIndex(of: id) {
            executing.remove(at: index)
            let replacements: [ServingRequestID]
            if executing.isEmpty {
                currentExecutionRoute = nil
                refillHeldFromQueue()
                replacements = []
            } else {
                replacements = refillExecutingFromQueue()
            }
            return .removedFromExecution(
                remaining: executing,
                replacements: replacements,
                nextHeld: held)
        }

        if let route = currentExecutionRoute {
            return .noRouteChange(route: route)
        }

        return .idle
    }

    public mutating func coalescingExpired() -> ServingAdmissionDecision {
        if let route = currentExecutionRoute {
            return .noRouteChange(route: route)
        }

        guard !held.isEmpty else {
            return .idle
        }

        let route: ServingExecutionRoute
        if held.count >= 2 {
            route = .continuousBatchNoSpec
        } else if configuration.soloPLDQualified {
            route = .soloPLD
        } else {
            route = .continuousBatchNoSpec
        }

        executing = held
        held.removeAll()
        currentExecutionRoute = route
        return .start(route: route, requests: executing)
    }

    public mutating func executionFinished(requests completed: [ServingRequestID]) -> ServingAdmissionDecision {
        if !completed.isEmpty {
            executing.removeAll { completed.contains($0) }
        }
        if executing.isEmpty {
            currentExecutionRoute = nil
            refillHeldFromQueue()
        } else if currentExecutionRoute != nil {
            let replacements = refillExecutingFromQueue()
            if !replacements.isEmpty {
                return .continuedExecution(
                    route: .continuousBatchNoSpec,
                    remaining: executing,
                    replacements: replacements)
            }
        }

        if let route = currentExecutionRoute {
            return .noRouteChange(route: route)
        }

        if !held.isEmpty {
            return .held(held)
        }

        return .idle
    }

    private func contains(_ id: ServingRequestID) -> Bool {
        held.contains(id) || queued.contains(id) || executing.contains(id)
    }

    private mutating func enqueue(_ id: ServingRequestID) -> ServingAdmissionDecision {
        guard queued.count < configuration.maximumQueuedRequests else {
            return .rejected(request: id, reason: .queueFull)
        }
        queued.append(id)
        return .queued(queued)
    }

    private mutating func refillHeldFromQueue() {
        guard held.count < configuration.maximumBatchRequests, !queued.isEmpty else {
            return
        }
        let count = min(
            configuration.maximumBatchRequests - held.count,
            queued.count)
        held.append(contentsOf: queued.prefix(count))
        queued.removeFirst(count)
    }

    private mutating func refillExecutingFromQueue() -> [ServingRequestID] {
        guard canJoinRunningCohort,
            !executing.isEmpty,
            executing.count < configuration.maximumBatchRequests,
            !queued.isEmpty
        else {
            return []
        }

        let count = min(
            configuration.maximumBatchRequests - executing.count,
            queued.count)
        let replacements = Array(queued.prefix(count))
        executing.append(contentsOf: replacements)
        queued.removeFirst(count)
        return replacements
    }

    private var canJoinRunningCohort: Bool {
        currentExecutionRoute == .continuousBatchNoSpec
            || (currentExecutionRoute == .soloPLD
                && configuration.soloPLDQualified)
    }
}
