import Foundation

public protocol ServingGenerationBackend: Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle
    func start(
        _ request: OpenAIChatCompletionRequest,
        resolvedCompletionBudget: ServingCompletionBudgetResolution
    ) async throws -> ServingGenerationHandle
    func shutdown() async
}

extension ServingGenerationBackend {
    public func start(
        _ request: OpenAIChatCompletionRequest,
        resolvedCompletionBudget: ServingCompletionBudgetResolution
    ) async throws -> ServingGenerationHandle {
        throw OpenAIServingError.server(
            "The selected fallback route cannot preserve the resolved completion budget",
            code: "resolved_budget_fallback_unsupported")
    }

    public func shutdown() async {}
}

public struct ServingBackendAdmissionError: Error, Equatable, Sendable {
    public enum Reason: Equatable, Sendable {
        case queueFull
        case capacityExceeded
        case requestTooLarge
    }

    public let reason: Reason
    public let retryAfterSeconds: Int?

    private init(reason: Reason, retryAfterSeconds: Int?) {
        if let retryAfterSeconds {
            precondition(
                (1...3_600).contains(retryAfterSeconds),
                "retryAfterSeconds must be between 1 and 3600")
        }
        self.reason = reason
        self.retryAfterSeconds = retryAfterSeconds
    }

    public static func queueFull(retryAfterSeconds: Int) -> ServingBackendAdmissionError {
        ServingBackendAdmissionError(
            reason: .queueFull,
            retryAfterSeconds: retryAfterSeconds)
    }

    public static func capacityExceeded(
        retryAfterSeconds: Int
    ) -> ServingBackendAdmissionError {
        ServingBackendAdmissionError(
            reason: .capacityExceeded,
            retryAfterSeconds: retryAfterSeconds)
    }

    public static func requestTooLarge() -> ServingBackendAdmissionError {
        ServingBackendAdmissionError(
            reason: .requestTooLarge,
            retryAfterSeconds: nil)
    }
}

public struct ServingGenerationHandle: Sendable {
    public let responseID: String
    public let created: Int
    public let model: String
    public let route: ServingExecutionRoute
    public let mailbox: BoundedDeltaMailbox
    public let lease: ServingRequestLease
    /// Exact post-template admission result. Nil is retained only for source-compatible fixture and
    /// third-party backends that have not opted into model-aware production admission.
    public let completionBudgetResolution: ServingCompletionBudgetResolution?
    /// Whether this stream separates reasoning from the visible answer: the streaming SSE handler routes
    /// its `.text` deltas through `StreamingReasoningSplitter` (reasoning until `</think>`, then content)
    /// when true, and passes them through as raw `delta.content` (byte-identical to before) when false.
    /// Derived at admission from `servingSeparatesReasoning(thinksByDefault:resolvedEnableThinking:)`.
    /// Defaults false so backends/tests that do not separate reasoning compile and behave unchanged.
    public let separatesReasoning: Bool

    public init(
        responseID: String,
        created: Int,
        model: String,
        route: ServingExecutionRoute,
        mailbox: BoundedDeltaMailbox,
        lease: ServingRequestLease,
        completionBudgetResolution: ServingCompletionBudgetResolution? = nil,
        separatesReasoning: Bool = false
    ) {
        self.responseID = responseID
        self.created = created
        self.model = model
        self.route = route
        self.mailbox = mailbox
        self.lease = lease
        self.completionBudgetResolution = completionBudgetResolution
        self.separatesReasoning = separatesReasoning
    }
}

public struct ServingGenerationCompletion: Equatable, Sendable {
    public let finishReason: OpenAIChatFinishReason
    public let usage: OpenAIChatUsage

    public init(
        finishReason: OpenAIChatFinishReason,
        usage: OpenAIChatUsage
    ) {
        self.finishReason = finishReason
        self.usage = usage
    }
}
