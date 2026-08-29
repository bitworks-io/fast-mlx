import Foundation

public struct OpenAIModelCapabilities: Codable, Sendable, Equatable {
    public let nativeMaxContextTokens: Int
    public let effectiveMaxContextTokens: Int
    public let defaultCompletionTokens: Int
    public let maximumCompletionTokens: Int
    public let maximumNonStreamingCompletionTokens: Int
    public let maximumRequestBodyBytes: Int
    public let maximumNonStreamingResponseBytes: Int
    public let completionLimitPolicy: ServingCompletionLimitPolicy
    public let reasoningTokensCountTowardCompletion: Bool

    public init(_ capabilities: ServingModelCapabilities) {
        nativeMaxContextTokens = capabilities.nativeMaxContextTokens
        effectiveMaxContextTokens = capabilities.effectiveMaxContextTokens
        defaultCompletionTokens = capabilities.defaultCompletionTokens
        maximumCompletionTokens = capabilities.maximumCompletionTokens
        maximumNonStreamingCompletionTokens = capabilities.maximumNonStreamingCompletionTokens
        maximumRequestBodyBytes = capabilities.maximumRequestBodyBytes
        maximumNonStreamingResponseBytes = capabilities.maximumNonStreamingResponseBytes
        completionLimitPolicy = capabilities.completionLimitPolicy
        reasoningTokensCountTowardCompletion = capabilities.reasoningTokensCountTowardCompletion
    }

    private enum CodingKeys: String, CodingKey {
        case nativeMaxContextTokens = "native_max_context_tokens"
        case effectiveMaxContextTokens = "effective_max_context_tokens"
        case defaultCompletionTokens = "default_completion_tokens"
        case maximumCompletionTokens = "maximum_completion_tokens"
        case maximumNonStreamingCompletionTokens = "maximum_non_streaming_completion_tokens"
        case maximumRequestBodyBytes = "maximum_request_body_bytes"
        case maximumNonStreamingResponseBytes = "maximum_non_streaming_response_bytes"
        case completionLimitPolicy = "completion_limit_policy"
        case reasoningTokensCountTowardCompletion = "reasoning_tokens_count_toward_completion"
    }
}

public struct OpenAIModelObject: Codable, Sendable, Equatable {
    public let id: String
    public let object: String
    public let created: Int
    public let ownedBy: String
    public let maxModelLen: Int?
    public let fastMLXCapabilities: OpenAIModelCapabilities?

    public init(model: String, capabilities: ServingModelCapabilities?) {
        id = model
        object = "model"
        created = 0
        ownedBy = "fast-mlx"
        maxModelLen = capabilities?.effectiveMaxContextTokens
        fastMLXCapabilities = capabilities.map(OpenAIModelCapabilities.init)
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
        case maxModelLen = "max_model_len"
        case fastMLXCapabilities = "fast_mlx_capabilities"
    }
}

public struct OpenAIModelListResponse: Codable, Sendable, Equatable {
    public let object: String
    public let data: [OpenAIModelObject]

    public init(model: String, capabilities: ServingModelCapabilities?) {
        object = "list"
        data = [OpenAIModelObject(model: model, capabilities: capabilities)]
    }
}
