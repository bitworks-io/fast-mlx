import CoreFoundation
import Foundation

public enum OpenAIErrorType: String, Codable, Sendable, Equatable {
    case invalidRequest = "invalid_request_error"
    case serverError = "server_error"
}

public struct OpenAIErrorPayload: Codable, Sendable, Equatable {
    public var type: OpenAIErrorType
    public var message: String
    public var param: String?
    public var code: String?

    public init(type: OpenAIErrorType, message: String, param: String?, code: String?) {
        self.type = type
        self.message = message
        self.param = param
        self.code = code
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case message
        case param
        case code
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encode(message, forKey: .message)
        if let param {
            try container.encode(param, forKey: .param)
        } else {
            try container.encodeNil(forKey: .param)
        }
        if let code {
            try container.encode(code, forKey: .code)
        } else {
            try container.encodeNil(forKey: .code)
        }
    }
}

public struct OpenAIErrorEnvelope: Codable, Sendable, Equatable {
    public var error: OpenAIErrorPayload

    public init(error: OpenAIErrorPayload) {
        self.error = error
    }
}

public enum OpenAIServingError: Error, Sendable, Equatable {
    case invalidRequest(String, param: String?)
    case server(String, code: String?)

    public var openAIError: OpenAIErrorPayload {
        switch self {
        case .invalidRequest(let message, let param):
            OpenAIErrorPayload(type: .invalidRequest, message: message, param: param, code: nil)
        case .server(let message, let code):
            OpenAIErrorPayload(type: .serverError, message: message, param: nil, code: code)
        }
    }
}

public enum OpenAIMessageRole: String, Codable, Sendable, Equatable {
    case developer
    case system
    case user
    case assistant
}

public struct OpenAIChatMessage: Sendable, Equatable {
    public var role: OpenAIMessageRole
    public var text: String

    public init(role: OpenAIMessageRole, text: String) {
        self.role = role
        self.text = text
    }
}

public struct OpenAIChatRequestLimits: Sendable, Equatable {
    public static let productionDefault = OpenAIChatRequestLimits(
        maximumBodyBytes: 1_048_576,
        maximumCompletionTokens: 4_096)

    public let maximumBodyBytes: Int
    public let maximumCompletionTokens: Int

    public init(maximumBodyBytes: Int, maximumCompletionTokens: Int) {
        precondition(maximumBodyBytes > 0, "maximumBodyBytes must be positive")
        precondition(maximumCompletionTokens > 0, "maximumCompletionTokens must be positive")
        self.maximumBodyBytes = maximumBodyBytes
        self.maximumCompletionTokens = maximumCompletionTokens
    }
}

public struct OpenAIChatCompletionRequest: Sendable, Equatable {
    public var model: String
    public var messages: [OpenAIChatMessage]
    public var maxCompletionTokens: Int?
    public var temperature: Double?
    public var choiceCount: Int
    public var stream: Bool
    public var stop: [String]

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxCompletionTokens: Int?,
        temperature: Double?,
        choiceCount: Int,
        stream: Bool,
        stop: [String]
    ) {
        self.model = model
        self.messages = messages
        self.maxCompletionTokens = maxCompletionTokens
        self.temperature = temperature
        self.choiceCount = choiceCount
        self.stream = stream
        self.stop = stop
    }

    public static func decodeStrict(
        from data: Data,
        limits: OpenAIChatRequestLimits = .productionDefault
    ) throws -> OpenAIChatCompletionRequest {
        guard data.count <= limits.maximumBodyBytes else {
            throw OpenAIServingError.invalidRequest(
                "Request body exceeds the configured byte limit",
                param: nil)
        }

        let root = try decodeJSONObject(data, param: nil)
        let allowedKeys: Set<String> = [
            "model",
            "messages",
            "max_completion_tokens",
            "max_tokens",
            "temperature",
            "n",
            "stream",
            "stop",
        ]
        try rejectUnknownKeys(in: root, allowed: allowedKeys, paramPrefix: nil)

        let model = try requiredString(root["model"], param: "model")
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAIServingError.invalidRequest("model must be a non-empty string", param: "model")
        }

        let rawMessages = try requiredArray(root["messages"], param: "messages")
        guard !rawMessages.isEmpty else {
            throw OpenAIServingError.invalidRequest("messages must contain at least one item", param: "messages")
        }
        let messages = try rawMessages.map { try decodeMessage($0) }

        let primaryBudget = try optionalPositiveInt(root["max_completion_tokens"], param: "max_completion_tokens")
        let deprecatedBudget = try optionalPositiveInt(root["max_tokens"], param: "max_tokens")
        if let primaryBudget, let deprecatedBudget, primaryBudget != deprecatedBudget {
            throw OpenAIServingError.invalidRequest(
                "max_completion_tokens conflicts with deprecated max_tokens",
                param: "max_completion_tokens")
        }

        let completionBudget = primaryBudget ?? deprecatedBudget
        if let completionBudget, completionBudget > limits.maximumCompletionTokens {
            throw OpenAIServingError.invalidRequest(
                "max_completion_tokens exceeds the configured limit",
                param: "max_completion_tokens")
        }

        let temperature = try optionalDouble(root["temperature"], param: "temperature")
        if let temperature, temperature != 0 {
            throw OpenAIServingError.invalidRequest("temperature must be 0 for this route", param: "temperature")
        }

        let choiceCount = try optionalInt(root["n"], param: "n") ?? 1
        guard choiceCount == 1 else {
            throw OpenAIServingError.invalidRequest("n must be 1 for this route", param: "n")
        }

        let stream = try optionalBool(root["stream"], param: "stream") ?? false
        let stop = try decodeStop(root["stop"])

        return OpenAIChatCompletionRequest(
            model: model,
            messages: messages,
            maxCompletionTokens: completionBudget,
            temperature: temperature,
            choiceCount: choiceCount,
            stream: stream,
            stop: stop)
    }

    public func requireLaunchedModel(_ launchedModel: String) throws {
        guard model == launchedModel else {
            throw OpenAIServingError.invalidRequest(
                "The requested model is not loaded by this server",
                param: "model")
        }
    }
}

public enum OpenAIChatFinishReason: String, Codable, Sendable, Equatable {
    case stop
    case length
}

public struct OpenAIChatUsage: Encodable, Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(promptTokens, forKey: .promptTokens)
        try container.encode(completionTokens, forKey: .completionTokens)
        try container.encode(totalTokens, forKey: .totalTokens)
    }
}

public struct OpenAIChatCompletionResponse: Encodable, Sendable, Equatable {
    public var id: String
    public var object = "chat.completion"
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: OpenAIChatUsage

    public init(
        id: String,
        created: Int,
        model: String,
        content: String,
        finishReason: OpenAIChatFinishReason,
        usage: OpenAIChatUsage
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: Message(role: "assistant", content: content),
                finishReason: finishReason)
        ]
        self.usage = usage
    }

    public struct Choice: Encodable, Sendable, Equatable {
        public var index: Int
        public var message: Message
        public var finishReason: OpenAIChatFinishReason

        private enum CodingKeys: String, CodingKey {
            case index
            case message
            case finishReason = "finish_reason"
        }
    }

    public struct Message: Encodable, Sendable, Equatable {
        public var role: String
        public var content: String
    }
}

public struct OpenAIChatCompletionChunk: Encodable, Sendable, Equatable {
    public var id: String
    public var object = "chat.completion.chunk"
    public var created: Int
    public var model: String
    public var choices: [Choice]

    public init(
        id: String,
        created: Int,
        model: String,
        index: Int,
        delta: Delta,
        finishReason: OpenAIChatFinishReason?
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [Choice(index: index, delta: delta, finishReason: finishReason)]
    }

    public static let doneSSEEvent = "data: [DONE]\n\n"

    public func sseEvent() throws -> String {
        let data = try JSONEncoder.openAI.encode(self)
        return "data: \(String(decoding: data, as: UTF8.self))\n\n"
    }

    public struct Choice: Encodable, Sendable, Equatable {
        public var index: Int
        public var delta: Delta
        public var finishReason: OpenAIChatFinishReason?

        private enum CodingKeys: String, CodingKey {
            case index
            case delta
            case finishReason = "finish_reason"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(index, forKey: .index)
            try container.encode(delta, forKey: .delta)
            if let finishReason {
                try container.encode(finishReason, forKey: .finishReason)
            } else {
                try container.encodeNil(forKey: .finishReason)
            }
        }
    }

    public struct Delta: Encodable, Sendable, Equatable {
        public var role: String?
        public var content: String?

        public init(role: String?, content: String?) {
            self.role = role
            self.content = content
        }
    }
}

public extension JSONEncoder {
    static var openAI: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

private func decodeJSONObject(_ data: Data, param: String?) throws -> [String: Any] {
    do {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OpenAIServingError.invalidRequest("Request body must be a JSON object", param: param)
        }
        return object
    } catch let error as OpenAIServingError {
        throw error
    } catch {
        throw OpenAIServingError.invalidRequest("Request body is not valid JSON", param: param)
    }
}

private func rejectUnknownKeys(in object: [String: Any], allowed: Set<String>, paramPrefix: String?) throws {
    for key in object.keys.sorted() where !allowed.contains(key) {
        let param = paramPrefix.map { "\($0).\(key)" } ?? key
        throw OpenAIServingError.invalidRequest("Unsupported field: \(param)", param: param)
    }
}

private func decodeMessage(_ raw: Any) throws -> OpenAIChatMessage {
    guard let object = raw as? [String: Any] else {
        throw OpenAIServingError.invalidRequest("messages entries must be objects", param: "messages")
    }
    try rejectUnknownKeys(in: object, allowed: ["role", "content"], paramPrefix: "messages")
    let roleValue = try requiredString(object["role"], param: "messages.role")
    guard let role = OpenAIMessageRole(rawValue: roleValue) else {
        throw OpenAIServingError.invalidRequest("Unsupported message role: \(roleValue)", param: "messages.role")
    }
    let text = try decodeContent(object["content"])
    return OpenAIChatMessage(role: role, text: text)
}

private func decodeContent(_ raw: Any?) throws -> String {
    if let string = raw as? String {
        return string
    }
    guard let parts = raw as? [Any], !parts.isEmpty else {
        throw OpenAIServingError.invalidRequest(
            "messages.content must be a string or non-empty text part array",
            param: "messages.content")
    }

    var text = ""
    for part in parts {
        guard let object = part as? [String: Any] else {
            throw OpenAIServingError.invalidRequest("content parts must be objects", param: "messages.content")
        }
        let type = try requiredString(object["type"], param: "messages.content")
        guard type == "text" else {
            throw OpenAIServingError.invalidRequest("Only text content parts are supported", param: "messages.content")
        }
        try rejectUnknownKeys(in: object, allowed: ["type", "text"], paramPrefix: "messages.content")
        text += try requiredString(object["text"], param: "messages.content")
    }
    return text
}

private func decodeStop(_ raw: Any?) throws -> [String] {
    guard let raw else { return [] }
    if let stop = raw as? String {
        guard !stop.isEmpty else {
            throw OpenAIServingError.invalidRequest("stop entries must be non-empty", param: "stop")
        }
        return [stop]
    }
    guard let values = raw as? [Any], values.count <= 4 else {
        throw OpenAIServingError.invalidRequest("stop must be a string or an array of up to four strings", param: "stop")
    }
    return try values.map { value in
        let stop = try requiredString(value, param: "stop")
        guard !stop.isEmpty else {
            throw OpenAIServingError.invalidRequest("stop entries must be non-empty", param: "stop")
        }
        return stop
    }
}

private func requiredString(_ raw: Any?, param: String) throws -> String {
    guard let string = raw as? String else {
        throw OpenAIServingError.invalidRequest("\(param) must be a string", param: param)
    }
    return string
}

private func requiredArray(_ raw: Any?, param: String) throws -> [Any] {
    guard let array = raw as? [Any] else {
        throw OpenAIServingError.invalidRequest("\(param) must be an array", param: param)
    }
    return array
}

private func optionalBool(_ raw: Any?, param: String) throws -> Bool? {
    guard let raw else { return nil }
    guard let bool = raw as? Bool else {
        throw OpenAIServingError.invalidRequest("\(param) must be a boolean", param: param)
    }
    return bool
}

private func optionalDouble(_ raw: Any?, param: String) throws -> Double? {
    guard let raw else { return nil }
    guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
        throw OpenAIServingError.invalidRequest("\(param) must be numeric", param: param)
    }
    let double = number.doubleValue
    guard double.isFinite else {
        throw OpenAIServingError.invalidRequest("\(param) must be finite", param: param)
    }
    return double
}

private func optionalInt(_ raw: Any?, param: String) throws -> Int? {
    guard let raw else { return nil }
    guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }
    let double = number.doubleValue
    guard double.rounded() == double else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }
    return number.intValue
}

private func optionalPositiveInt(_ raw: Any?, param: String) throws -> Int? {
    guard let value = try optionalInt(raw, param: param) else { return nil }
    guard value > 0 else {
        throw OpenAIServingError.invalidRequest("\(param) must be greater than zero", param: param)
    }
    return value
}
