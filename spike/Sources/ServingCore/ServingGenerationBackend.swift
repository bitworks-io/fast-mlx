import Foundation

public protocol ServingGenerationBackend: Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle
    func start(
        _ request: OpenAIChatCompletionRequest,
        resolvedCompletionBudget: ServingCompletionBudgetResolution
    ) async throws -> ServingGenerationHandle
    func shutdown() async
    /// Whether this backend's decoding route can apply the `response_format: {"type":"json_object"}`
    /// token constraint (the byte-level JSON pushdown automaton in slice 1a/1b of the response-format
    /// design). Defaults to `false` in the protocol extension below, so every existing and
    /// not-yet-updated backend fails closed: `validateResponseFormatCapability` (this file) must be
    /// called at the single dispatch point before `start(_:)` so a request carrying
    /// `.responseFormat == .jsonObject` is refused with a 400 rather than silently served as free
    /// text by a backend that never applies the mask. No backend sets this `true` yet (slice 1c).
    var supportsJSONObjectResponseFormat: Bool { get }
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

    public var supportsJSONObjectResponseFormat: Bool { false }
}

/// The single ServingCore-level admission check for `response_format: {"type":"json_object"}`
/// against a specific backend's declared capability. Callers must invoke this at the one dispatch
/// point that calls `backend.start(_:)` for a chat/completions request, before construction of any
/// evidence/admission state for that request, so a capability-false backend never receives a
/// request it would silently serve as free text. Throws `OpenAIServingError.invalidRequest` (param
/// `response_format`) when `request.responseFormat == .jsonObject` and
/// `backendSupportsJSONObjectResponseFormat` is `false`; otherwise returns normally (including when
/// `request.responseFormat` is `nil`, i.e. every request before this feature existed).
public func validateResponseFormatCapability(
    request: OpenAIChatCompletionRequest,
    backendSupportsJSONObjectResponseFormat: Bool
) throws {
    guard request.responseFormat == .jsonObject, !backendSupportsJSONObjectResponseFormat else {
        return
    }
    throw OpenAIServingError.invalidRequest(
        "response_format json_object is not supported by the loaded model's decoding route",
        param: "response_format")
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
