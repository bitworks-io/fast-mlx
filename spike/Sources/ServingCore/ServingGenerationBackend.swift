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
    /// Whether this backend's decoding route can apply a compiled `response_format:
    /// {"type":"json_schema", ...}` constraint (response-format slice 2). Defaults to `false` in the
    /// protocol extension below, matching `supportsJSONObjectResponseFormat`'s fail-closed default —
    /// no backend sets this `true` yet (stage 2a compiles/validates the schema but wires no
    /// automaton).
    var supportsJSONSchemaResponseFormat: Bool { get }
    /// Whether this backend's decoding route can apply a non-empty OpenAI `logit_bias` map to the
    /// logits before sampling. Defaults to `false` (fail closed): only the plain scalar route, whose
    /// decoder owns a logit-processor seam, declares `true`. Every other route (continuous batching,
    /// compiled/speculative decoders, test doubles) must be refused with a 400 by
    /// `validateLogitBiasCapability` at the single dispatch point rather than serve an unbiased
    /// response to a caller who asked for a bias.
    var supportsLogitBias: Bool { get }
    /// Whether this backend's decoding route can ENFORCE a forcing `tool_choice` (a named function
    /// or `"required"`), i.e. guarantee the reply is a tool call rather than prose. Defaults to
    /// `false` (fail closed): only the scalar route that can run the json_schema mask declares
    /// `true`, and a route-switching wrapper reports whatever the route that serves forcing choices
    /// reports. `validateToolChoiceCapability` must be called at the single dispatch point before
    /// `start(_:)` so a request demanding a tool call is refused with a 400 rather than answered in
    /// prose by a route that cannot force one. Deliberately separate from
    /// `supportsJSONSchemaResponseFormat`, which gates `response_format`.
    var enforcesForcingToolChoice: Bool { get }
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
    public var supportsJSONSchemaResponseFormat: Bool { false }
    public var supportsLogitBias: Bool { false }
    public var enforcesForcingToolChoice: Bool { false }
}

/// The single ServingCore-level admission check for `logit_bias` against a backend's declared
/// capability. Called at the one dispatch point before `backend.start(_:)` (alongside
/// `validateResponseFormatCapability`), so a backend that cannot apply a bias never receives a
/// request carrying one and no generation is started for it. The neutral empty map needs no
/// capability and passes on every backend. Throws `OpenAIServingError.invalidRequest` with
/// `param: "logit_bias"`.
public func validateLogitBiasCapability(
    request: OpenAIChatCompletionRequest,
    backendSupportsLogitBias: Bool
) throws {
    guard !request.logitBias.isEmpty else { return }
    guard backendSupportsLogitBias else {
        throw OpenAIServingError.invalidRequest(
            "logit_bias is not supported by the loaded model's decoding route",
            param: "logit_bias")
    }
}

/// The single ServingCore-level admission check for a forcing `tool_choice` (a named function or
/// `"required"`) against whether the dispatched backend can ENFORCE it (today: the scalar route
/// that can run the json_schema mask; see `ServingGenerationBackend.enforcesForcingToolChoice`).
/// Enforcement also requires a chat-templated prompt (`promptInput == .chat`). Called at the one
/// dispatch point before `backend.start(_:)` (alongside `validateLogitBiasCapability`) and again
/// inside each backend's `start` as defense in depth, so a route that cannot force a tool call
/// never answers a caller who demanded one in prose with a 200. `auto`, `none` and an absent choice
/// need no enforcement and always pass. Uses an EXHAUSTIVE `switch` over `request.toolChoice` (no
/// `default:`) so a future `OpenAIToolChoice` case cannot silently skip this gate and fail OPEN.
/// The message never echoes the client's function name. Throws `OpenAIServingError.invalidRequest`
/// with `param: "tool_choice"`.
public func validateToolChoiceCapability(
    request: OpenAIChatCompletionRequest,
    backendEnforcesToolChoice: Bool
) throws {
    // `/v1/completions` always decodes `toolChoice` as `.none`, so HTTP never reaches this with a
    // raw-text prompt; the extra term closes an in-process fail-open (the backend's enforcement
    // blocks only run for a chat prompt).
    let enforced = backendEnforcesToolChoice && request.promptInput == .chat
    switch request.toolChoice {
    case .none, .auto:
        return
    case .function:
        guard enforced else {
            throw OpenAIServingError.invalidRequest(
                "a named tool_choice cannot be enforced by the loaded model's decoding route"
                    + "; it is enforced only on the non-speculative single-stream route of a model "
                    + "whose tokenizer supports json_schema constraints",
                param: "tool_choice")
        }
    case .required:
        guard enforced else {
            throw OpenAIServingError.invalidRequest(
                "tool_choice \"required\" cannot be enforced by the loaded model's decoding route"
                    + "; it is enforced only on the non-speculative single-stream route of a model "
                    + "whose tokenizer supports json_schema constraints",
                param: "tool_choice")
        }
    }
}

/// The single ServingCore-level admission check for `response_format` against a specific backend's
/// declared capabilities. Callers must invoke this at the one dispatch point that calls
/// `backend.start(_:)` for a chat/completions request, before construction of any evidence/admission
/// state for that request, so a capability-false backend never receives a request it would silently
/// serve as free text. Uses an EXHAUSTIVE `switch` over `request.responseFormat` (not `==` against a
/// single case) so a future `ServingResponseFormat` case cannot silently skip this gate and fail
/// OPEN — see the response-format design doc's "fail-open guard" finding. Throws
/// `OpenAIServingError.invalidRequest` (param `response_format`) naming the actual format when the
/// matching capability flag is `false`; otherwise returns normally (including when
/// `request.responseFormat` is `nil`, i.e. every request before this feature existed).
public func validateResponseFormatCapability(
    request: OpenAIChatCompletionRequest,
    backendSupportsJSONObjectResponseFormat: Bool,
    backendSupportsJSONSchemaResponseFormat: Bool
) throws {
    switch request.responseFormat {
    case nil:
        return
    case .jsonObject:
        guard backendSupportsJSONObjectResponseFormat else {
            throw OpenAIServingError.invalidRequest(
                "response_format json_object is not supported by the loaded model's decoding route",
                param: "response_format")
        }
    case .jsonSchema:
        guard backendSupportsJSONSchemaResponseFormat else {
            throw OpenAIServingError.invalidRequest(
                "response_format json_schema is not supported by the loaded model's decoding route",
                param: "response_format")
        }
    }
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
