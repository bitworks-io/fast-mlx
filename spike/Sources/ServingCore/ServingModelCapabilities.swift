import Foundation

public enum ServingCompletionLimitPolicy: String, Codable, Sendable, Equatable {
    case reject
    case clamp
}

public enum ServingCompletionLimitingFactor: String, Codable, Sendable, Equatable {
    case operatorMaximum = "operator_maximum"
    case contextWindow = "context_window"
    case operatorMaximumAndContextWindow = "operator_maximum_and_context_window"
}

public enum ServingModelCapabilitiesError:
    Error, Sendable, Equatable, CustomStringConvertible
{
    case emptyModel
    case invalidNativeContext
    case invalidEffectiveContext
    case effectiveContextExceedsNative
    case invalidDefaultCompletionTokens
    case invalidMaximumCompletionTokens
    case explicitDefaultExceedsMaximum
    case invalidMaximumNonStreamingCompletionTokens
    case invalidMaximumRequestBodyBytes
    case invalidMaximumNonStreamingResponseBytes

    public var description: String {
        switch self {
        case .emptyModel:
            "the launched model identifier is empty"
        case .invalidNativeContext:
            "the authenticated model context must be at least two tokens"
        case .invalidEffectiveContext:
            "the admitted served context must be at least two tokens"
        case .effectiveContextExceedsNative:
            "the admitted served context exceeds the model's authenticated native context"
        case .invalidDefaultCompletionTokens:
            "the default completion budget must be positive"
        case .invalidMaximumCompletionTokens:
            "the operator completion ceiling must be positive and leave room for at least one prompt token"
        case .explicitDefaultExceedsMaximum:
            "the explicit default completion budget exceeds the model/host/operator maximum"
        case .invalidMaximumNonStreamingCompletionTokens:
            "the non-streaming completion ceiling must be positive"
        case .invalidMaximumRequestBodyBytes:
            "the HTTP request-body byte ceiling must be positive"
        case .invalidMaximumNonStreamingResponseBytes:
            "the non-streaming response byte ceiling must be positive"
        }
    }
}

public struct ServingCompletionBudgetResolution: Sendable, Equatable {
    public let requestedCompletionTokens: Int?
    public let appliedCompletionTokens: Int
    public let maximumAllowedCompletionTokens: Int
    public let renderedPromptTokens: Int
    public let wasClamped: Bool
    public let limitingFactor: ServingCompletionLimitingFactor
    public let completionLimitPolicy: ServingCompletionLimitPolicy

    public init(
        requestedCompletionTokens: Int?,
        appliedCompletionTokens: Int,
        maximumAllowedCompletionTokens: Int,
        renderedPromptTokens: Int,
        wasClamped: Bool,
        limitingFactor: ServingCompletionLimitingFactor,
        completionLimitPolicy: ServingCompletionLimitPolicy? = nil
    ) {
        precondition(
            requestedCompletionTokens == nil || requestedCompletionTokens! > 0,
            "requestedCompletionTokens must be nil or positive")
        precondition(appliedCompletionTokens > 0, "appliedCompletionTokens must be positive")
        precondition(
            maximumAllowedCompletionTokens >= appliedCompletionTokens,
            "maximumAllowedCompletionTokens must cover the applied budget")
        precondition(renderedPromptTokens > 0, "renderedPromptTokens must be positive")
        self.requestedCompletionTokens = requestedCompletionTokens
        self.appliedCompletionTokens = appliedCompletionTokens
        self.maximumAllowedCompletionTokens = maximumAllowedCompletionTokens
        self.renderedPromptTokens = renderedPromptTokens
        self.wasClamped = wasClamped
        self.limitingFactor = limitingFactor
        self.completionLimitPolicy =
            completionLimitPolicy ?? (wasClamped ? .clamp : .reject)
    }
}

/// Immutable model and host-fit capability used by request decoding, discovery, every generation
/// backend, and startup reporting. The only widening inputs are authenticated model metadata and the
/// fit planner's admitted context; operator values may only narrow them.
public struct ServingModelCapabilities: Sendable, Equatable {
    public static let minimumAutomaticRequestBodyBytes = 1_048_576
    public static let automaticRequestBodyBytesPerContextToken = 64
    public static let maximumAutomaticRequestBodyBytes = 64 * 1_048_576

    public let model: String
    public let nativeMaxContextTokens: Int
    public let effectiveMaxContextTokens: Int
    public let defaultCompletionTokens: Int
    public let maximumCompletionTokens: Int
    public let maximumNonStreamingCompletionTokens: Int
    public let maximumRequestBodyBytes: Int
    public let maximumNonStreamingResponseBytes: Int
    public let completionLimitPolicy: ServingCompletionLimitPolicy
    public let reasoningTokensCountTowardCompletion: Bool

    public init(
        model: String,
        nativeMaxContextTokens: Int,
        effectiveMaxContextTokens: Int,
        requestedDefaultCompletionTokens: Int = 4_096,
        defaultCompletionTokensWasExplicit: Bool = false,
        maximumCompletionTokens: Int? = nil,
        maximumNonStreamingCompletionTokens: Int = 16_384,
        maximumRequestBodyBytes: Int? = nil,
        maximumNonStreamingResponseBytes: Int = 16 * 1_048_576,
        completionLimitPolicy: ServingCompletionLimitPolicy = .reject
    ) throws {
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ServingModelCapabilitiesError.emptyModel
        }
        guard nativeMaxContextTokens >= 2 else {
            throw ServingModelCapabilitiesError.invalidNativeContext
        }
        guard effectiveMaxContextTokens >= 2 else {
            throw ServingModelCapabilitiesError.invalidEffectiveContext
        }
        guard effectiveMaxContextTokens <= nativeMaxContextTokens else {
            throw ServingModelCapabilitiesError.effectiveContextExceedsNative
        }
        guard requestedDefaultCompletionTokens > 0 else {
            throw ServingModelCapabilitiesError.invalidDefaultCompletionTokens
        }
        guard maximumNonStreamingCompletionTokens > 0 else {
            throw ServingModelCapabilitiesError.invalidMaximumNonStreamingCompletionTokens
        }
        if let maximumRequestBodyBytes, maximumRequestBodyBytes <= 0 {
            throw ServingModelCapabilitiesError.invalidMaximumRequestBodyBytes
        }
        guard maximumNonStreamingResponseBytes > 0 else {
            throw ServingModelCapabilitiesError.invalidMaximumNonStreamingResponseBytes
        }

        let contextMaximum = effectiveMaxContextTokens - 1
        let resolvedMaximum: Int
        if let maximumCompletionTokens {
            guard maximumCompletionTokens > 0,
                maximumCompletionTokens <= contextMaximum
            else {
                throw ServingModelCapabilitiesError.invalidMaximumCompletionTokens
            }
            resolvedMaximum = maximumCompletionTokens
        } else {
            resolvedMaximum = contextMaximum
        }
        if defaultCompletionTokensWasExplicit,
            requestedDefaultCompletionTokens > resolvedMaximum
        {
            throw ServingModelCapabilitiesError.explicitDefaultExceedsMaximum
        }

        self.model = model
        self.nativeMaxContextTokens = nativeMaxContextTokens
        self.effectiveMaxContextTokens = effectiveMaxContextTokens
        self.defaultCompletionTokens = min(
            requestedDefaultCompletionTokens,
            resolvedMaximum)
        self.maximumCompletionTokens = resolvedMaximum
        self.maximumNonStreamingCompletionTokens = min(
            maximumNonStreamingCompletionTokens,
            resolvedMaximum)
        self.maximumRequestBodyBytes = maximumRequestBodyBytes
            ?? Self.automaticRequestBodyBytes(
                effectiveMaxContextTokens: effectiveMaxContextTokens)
        self.maximumNonStreamingResponseBytes = maximumNonStreamingResponseBytes
        self.completionLimitPolicy = completionLimitPolicy
        self.reasoningTokensCountTowardCompletion = true
    }

    /// A bounded transport default, separate from token admission. Sixty-four bytes per admitted
    /// context token leaves generous room for UTF-8, JSON escaping, chat envelopes, and tool schemas
    /// on today's long-context models. Operators can raise or lower it explicitly when a workload's
    /// wire representation differs; discovery always reports the effective byte ceiling.
    public static func automaticRequestBodyBytes(
        effectiveMaxContextTokens: Int
    ) -> Int {
        precondition(effectiveMaxContextTokens > 0)
        let (scaled, overflow) = effectiveMaxContextTokens.multipliedReportingOverflow(
            by: automaticRequestBodyBytesPerContextToken)
        let bounded = overflow
            ? maximumAutomaticRequestBodyBytes
            : min(scaled, maximumAutomaticRequestBodyBytes)
        return max(minimumAutomaticRequestBodyBytes, bounded)
    }

    /// Use the same advertised byte boundary for the response mailbox's total buffered payload.
    /// This keeps a complete structured tool-call delta from hitting a smaller hidden production
    /// ceiling while retaining bounded backpressure across text and tool output.
    public func responseMailboxCapacity(
        maxDeltas: Int
    ) -> BoundedDeltaMailbox.Capacity {
        BoundedDeltaMailbox.Capacity(
            maxDeltas: maxDeltas,
            maxBytes: maximumNonStreamingResponseBytes)
    }

    public func resolveCompletionBudget(
        requestedCompletionTokens: Int?,
        renderedPromptTokens: Int,
        stream: Bool
    ) throws -> ServingCompletionBudgetResolution {
        guard renderedPromptTokens > 0 else {
            throw OpenAIServingError.invalidRequestWithCode(
                "The rendered prompt must contain at least one token",
                param: "messages",
                code: "invalid_rendered_prompt")
        }
        guard renderedPromptTokens < effectiveMaxContextTokens else {
            throw OpenAIServingError.invalidRequestWithCode(
                "The rendered prompt uses \(renderedPromptTokens) tokens and exceeds the effective context limit \(effectiveMaxContextTokens)",
                param: "messages",
                code: "context_length_exceeded")
        }
        if let requestedCompletionTokens, requestedCompletionTokens <= 0 {
            throw OpenAIServingError.invalidRequest(
                "max_completion_tokens must be greater than zero",
                param: "max_completion_tokens")
        }

        // Both values are positive and effectiveMaxContextTokens is already validated, so this
        // subtraction cannot underflow.
        let contextRemaining = effectiveMaxContextTokens - renderedPromptTokens
        let maximumAllowed = min(maximumCompletionTokens, contextRemaining)
        let limitingFactor: ServingCompletionLimitingFactor
        if contextRemaining < maximumCompletionTokens {
            limitingFactor = .contextWindow
        } else if contextRemaining > maximumCompletionTokens {
            limitingFactor = .operatorMaximum
        } else {
            limitingFactor = .operatorMaximumAndContextWindow
        }

        let desired = requestedCompletionTokens ?? defaultCompletionTokens
        let applied: Int
        let wasClamped: Bool
        if desired <= maximumAllowed {
            applied = desired
            wasClamped = false
        } else if requestedCompletionTokens == nil {
            applied = maximumAllowed
            wasClamped = false
        } else {
            switch completionLimitPolicy {
            case .reject:
                throw OpenAIServingError.invalidRequestWithCode(
                    "max_completion_tokens \(desired) exceeds the maximum allowed \(maximumAllowed) for this rendered prompt",
                    param: "max_completion_tokens",
                    code: "completion_limit_exceeded")
            case .clamp:
                applied = maximumAllowed
                wasClamped = true
            }
        }

        guard stream || applied <= maximumNonStreamingCompletionTokens else {
            throw OpenAIServingError.invalidRequestWithCode(
                "Non-streaming responses are limited to \(maximumNonStreamingCompletionTokens) completion tokens; set stream=true for \(applied) tokens",
                param: "stream",
                code: "stream_required")
        }

        return ServingCompletionBudgetResolution(
            requestedCompletionTokens: requestedCompletionTokens,
            appliedCompletionTokens: applied,
            maximumAllowedCompletionTokens: maximumAllowed,
            renderedPromptTokens: renderedPromptTokens,
            wasClamped: wasClamped,
            limitingFactor: limitingFactor,
            completionLimitPolicy: completionLimitPolicy)
    }
}
