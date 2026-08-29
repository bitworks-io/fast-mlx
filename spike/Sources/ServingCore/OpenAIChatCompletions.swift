import CoreFoundation
import Foundation

public enum OpenAIErrorType: String, Codable, Sendable, Equatable {
    case invalidRequest = "invalid_request_error"
    case rateLimit = "rate_limit_error"
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
    case invalidRequestWithCode(String, param: String?, code: String)
    case rateLimited(String, code: String?)
    case server(String, code: String?)

    public var openAIError: OpenAIErrorPayload {
        switch self {
        case .invalidRequest(let message, let param):
            OpenAIErrorPayload(type: .invalidRequest, message: message, param: param, code: nil)
        case .invalidRequestWithCode(let message, let param, let code):
            OpenAIErrorPayload(type: .invalidRequest, message: message, param: param, code: code)
        case .rateLimited(let message, let code):
            OpenAIErrorPayload(type: .rateLimit, message: message, param: nil, code: code)
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
    case tool
}

public struct OpenAIChatMessage: Sendable, Equatable {
    public var role: OpenAIMessageRole
    public var text: String
    /// Tool calls carried by an assistant history turn (empty otherwise).
    public var toolCalls: [OpenAIToolCall]
    /// The `tool_call_id` a `tool` result answers (nil otherwise).
    public var toolCallId: String?
    /// Optional author name (e.g. the tool name on a `tool` message).
    public var name: String?

    public init(
        role: OpenAIMessageRole,
        text: String,
        toolCalls: [OpenAIToolCall] = [],
        toolCallId: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.text = text
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
    }
}

public struct OpenAIChatRequestLimits: Sendable, Equatable {
    public static let productionDefault = OpenAIChatRequestLimits(
        maximumBodyBytes: 1_048_576,
        maximumCompletionTokens: 4_096,
        enforceMaximumCompletionTokensDuringDecoding: true)

    public let maximumBodyBytes: Int
    public let maximumCompletionTokens: Int
    public let enforceMaximumCompletionTokensDuringDecoding: Bool

    public init(
        maximumBodyBytes: Int,
        maximumCompletionTokens: Int,
        enforceMaximumCompletionTokensDuringDecoding: Bool = true
    ) {
        precondition(maximumBodyBytes > 0, "maximumBodyBytes must be positive")
        precondition(maximumCompletionTokens > 0, "maximumCompletionTokens must be positive")
        self.maximumBodyBytes = maximumBodyBytes
        self.maximumCompletionTokens = maximumCompletionTokens
        self.enforceMaximumCompletionTokensDuringDecoding = enforceMaximumCompletionTokensDuringDecoding
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
    /// Function tools the model may call (empty when none requested).
    public var tools: [OpenAIToolSpec]
    /// How the model may use tools. Defaults to `.auto` when tools are present, else `.none`.
    public var toolChoice: OpenAIToolChoice
    /// Client hint for parallel tool calls (nil = unspecified).
    public var parallelToolCalls: Bool?
    /// Qwen3 thinking-mode control (nil = server default).
    public var enableThinking: Bool?
    /// Nucleus sampling threshold (nil = server default; validated by ServingSamplingPolicy).
    public var topP: Double?
    /// Top-k sampling cutoff (nil = unset; validated by ServingSamplingPolicy).
    public var topK: Int?
    /// Minimum-probability sampling floor (nil = unset; validated by ServingSamplingPolicy).
    public var minP: Double?
    /// Caller-supplied sampling seed (nil = server-assigned).
    public var seed: Int64?
    /// Qwen3 reasoning-effort hint (nil = server default; validated against xhigh/medium/low).
    public var reasoningEffort: String?
    /// OpenAI presence penalty in [-2, 2] (nil = none). Applied via the decoder's logit processor.
    public var presencePenalty: Double?
    /// OpenAI frequency penalty in [-2, 2] (nil = none).
    public var frequencyPenalty: Double?
    /// HF-style repetition penalty > 0 (nil = none; 1.0 = no penalty).
    public var repetitionPenalty: Double?

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxCompletionTokens: Int?,
        temperature: Double?,
        choiceCount: Int,
        stream: Bool,
        stop: [String],
        tools: [OpenAIToolSpec] = [],
        toolChoice: OpenAIToolChoice = .none,
        parallelToolCalls: Bool? = nil,
        enableThinking: Bool? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        seed: Int64? = nil,
        reasoningEffort: String? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxCompletionTokens = maxCompletionTokens
        self.temperature = temperature
        self.choiceCount = choiceCount
        self.stream = stream
        self.stop = stop
        self.tools = tools
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.enableThinking = enableThinking
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.reasoningEffort = reasoningEffort
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
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
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "enable_thinking",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "reasoning_effort",
            "chat_template_kwargs",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
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
        if limits.enforceMaximumCompletionTokensDuringDecoding,
           let completionBudget,
           completionBudget > limits.maximumCompletionTokens {
            throw OpenAIServingError.invalidRequest(
                "max_completion_tokens exceeds the configured limit",
                param: "max_completion_tokens")
        }

        let temperature = try optionalDouble(root["temperature"], param: "temperature")

        let choiceCount = try optionalInt(root["n"], param: "n") ?? 1
        guard choiceCount == 1 else {
            throw OpenAIServingError.invalidRequest("n must be 1 for this route", param: "n")
        }

        let stream = try optionalBool(root["stream"], param: "stream") ?? false
        let stop = try decodeStop(root["stop"])

        let tools = try OpenAIToolDecoding.decodeTools(root["tools"])
        let toolChoice = try OpenAIToolDecoding.decodeToolChoice(root["tool_choice"], hasTools: !tools.isEmpty)
        let parallelToolCalls = try optionalBool(root["parallel_tool_calls"], param: "parallel_tool_calls")
        let enableThinking = try optionalBool(root["enable_thinking"], param: "enable_thinking")

        let topP = try optionalDouble(root["top_p"], param: "top_p")
        let topK = try optionalInt(root["top_k"], param: "top_k")
        if let topK, topK <= 0 {
            throw OpenAIServingError.invalidRequest("top_k must be greater than zero", param: "top_k")
        }
        let minP = try optionalDouble(root["min_p"], param: "min_p")
        if let minP, !(minP >= 0 && minP <= 1) {
            throw OpenAIServingError.invalidRequest("min_p must be between 0 and 1", param: "min_p")
        }
        let seed = try optionalInt64(root["seed"], param: "seed")

        let topLevelReasoningEffort = try optionalReasoningEffort(root["reasoning_effort"], param: "reasoning_effort")
        let (kwargsEnableThinking, kwargsReasoningEffort) = try decodeChatTemplateKwargs(root["chat_template_kwargs"])
        let resolvedEnableThinking = enableThinking ?? kwargsEnableThinking
        let resolvedReasoningEffort = topLevelReasoningEffort ?? kwargsReasoningEffort

        let presencePenalty = try optionalDouble(root["presence_penalty"], param: "presence_penalty")
        if let presencePenalty, !(presencePenalty >= -2 && presencePenalty <= 2) {
            throw OpenAIServingError.invalidRequest(
                "presence_penalty must be between -2 and 2", param: "presence_penalty")
        }
        let frequencyPenalty = try optionalDouble(root["frequency_penalty"], param: "frequency_penalty")
        if let frequencyPenalty, !(frequencyPenalty >= -2 && frequencyPenalty <= 2) {
            throw OpenAIServingError.invalidRequest(
                "frequency_penalty must be between -2 and 2", param: "frequency_penalty")
        }
        let repetitionPenalty = try optionalDouble(root["repetition_penalty"], param: "repetition_penalty")
        if let repetitionPenalty, !(repetitionPenalty > 0) {
            throw OpenAIServingError.invalidRequest(
                "repetition_penalty must be greater than zero", param: "repetition_penalty")
        }

        return OpenAIChatCompletionRequest(
            model: model,
            messages: messages,
            maxCompletionTokens: completionBudget,
            temperature: temperature,
            choiceCount: choiceCount,
            stream: stream,
            stop: stop,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            enableThinking: resolvedEnableThinking,
            topP: topP,
            topK: topK,
            minP: minP,
            seed: seed,
            reasoningEffort: resolvedReasoningEffort,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            repetitionPenalty: repetitionPenalty)
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
    case toolCalls = "tool_calls"
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
        content: String?,
        finishReason: OpenAIChatFinishReason,
        usage: OpenAIChatUsage,
        toolCalls: [OpenAIToolCall] = [],
        reasoningContent: String? = nil
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: Message(
                    role: "assistant",
                    content: content,
                    toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                    reasoningContent: reasoningContent),
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
        public var content: String?
        public var toolCalls: [OpenAIToolCall]?
        /// Qwen `<think>` reasoning, separated from `content` (nil when there was none).
        public var reasoningContent: String?

        public init(
            role: String, content: String?, toolCalls: [OpenAIToolCall]? = nil,
            reasoningContent: String? = nil
        ) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.reasoningContent = reasoningContent
        }

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case toolCalls = "tool_calls"
            case reasoningContent = "reasoning_content"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            // `content` is always emitted (null when the turn is tool-calls-only).
            if let content {
                try container.encode(content, forKey: .content)
            } else {
                try container.encodeNil(forKey: .content)
            }
            if let reasoningContent {
                try container.encode(reasoningContent, forKey: .reasoningContent)
            }
            if let toolCalls, !toolCalls.isEmpty {
                try container.encode(toolCalls, forKey: .toolCalls)
            }
        }
    }
}

public struct OpenAIChatCompletionChunk: Encodable, Sendable, Equatable {
    public var id: String
    public var object = "chat.completion.chunk"
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: OpenAIChatUsage?

    public init(
        id: String,
        created: Int,
        model: String,
        index: Int,
        delta: Delta,
        finishReason: OpenAIChatFinishReason?,
        usage: OpenAIChatUsage? = nil
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = [Choice(index: index, delta: delta, finishReason: finishReason)]
        self.usage = usage
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case model
        case choices
        case usage
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(object, forKey: .object)
        try container.encode(created, forKey: .created)
        try container.encode(model, forKey: .model)
        try container.encode(choices, forKey: .choices)
        if let usage {
            try container.encode(usage, forKey: .usage)
        } else {
            try container.encodeNil(forKey: .usage)
        }
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
        public var toolCalls: [OpenAIToolCallDelta]?
        /// Streaming `<think>` reasoning delta (nil unless the server separates streamed reasoning;
        /// today only the non-streaming path splits reasoning — streaming separation is a follow-up).
        public var reasoningContent: String?

        public init(
            role: String?, content: String?, toolCalls: [OpenAIToolCallDelta]? = nil,
            reasoningContent: String? = nil
        ) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.reasoningContent = reasoningContent
        }

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case toolCalls = "tool_calls"
            case reasoningContent = "reasoning_content"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            if let role { try container.encode(role, forKey: .role) }
            if let content { try container.encode(content, forKey: .content) }
            if let reasoningContent { try container.encode(reasoningContent, forKey: .reasoningContent) }
            if let toolCalls, !toolCalls.isEmpty { try container.encode(toolCalls, forKey: .toolCalls) }
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

/// Splits Qwen model output into reasoning (the `<think>…</think>` block) and the final content.
/// The chat template pre-fills the opening `<think>` into the PROMPT, so the model output is typically
/// `<reasoning>…</think>…<answer>` (a leading `<think>` is stripped if the model emits one anyway).
/// When there is no `</think>` — thinking off, or reasoning truncated — everything is content and
/// reasoning is nil (content is returned verbatim so a normal answer is byte-identical to before).
public enum ReasoningContentSplitter {
    public static func split(
        _ text: String,
        separationActive: Bool = false
    ) -> (reasoning: String?, content: String?) {
        guard let marker = text.range(of: "</think>") else {
            // Gate on marker ABSENCE (not `reasoning == nil`): a closed empty `<think></think>answer`
            // takes the tag path below and keeps its real answer. With no `</think>`, thinking was
            // truncated by the token budget mid-reasoning. When separation is active (thinking-ON, a
            // thinks-by-default family), the whole output IS reasoning — mirror the streaming Option A
            // contract (all reasoning_content, empty content) and the streaming splitter's flush
            // normalization (strip a leading `<think>`, trim boundary whitespace) byte-for-byte, rather
            // than retro-labeling raw chain-of-thought as the assistant's answer. Separation off
            // (thinking-OFF / non-thinks-by-default family) stays byte-identical: a plain answer with
            // no think block is returned verbatim as content.
            guard separationActive else {
                return (nil, text.isEmpty ? nil : text)
            }
            var reasoning = text
            if reasoning.hasPrefix("<think>") {
                reasoning.removeFirst("<think>".count)
            }
            reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, nil)
        }
        var reasoning = String(text[text.startIndex..<marker.lowerBound])
        if reasoning.hasPrefix("<think>") {
            reasoning.removeFirst("<think>".count)
        }
        reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let content = String(text[marker.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
    }
}

/// Incremental, streaming counterpart to `ReasoningContentSplitter`. Fed the raw text deltas in
/// order, it partitions them at the FIRST `</think>` into `reasoning` (before) and `content` (after),
/// matching `ReasoningContentSplitter.split` byte-for-byte on any output that CONTAINS `</think>` —
/// including the leading-`<think>` strip and whitespace trimming at both boundaries — regardless of
/// how the byte stream happens to be chunked. The closing tag is never torn across a chunk boundary:
/// a suffix that could still be a tag prefix is held back (at most `"</think>".count - 1` = 7 bytes of
/// non-whitespace), and trailing whitespace that might be trimmed at a boundary is held until resolved.
///
/// Divergence from non-streaming (unavoidable, by design): callers MUST route through this only when
/// thinking is active (the chat template pre-fills `<think>` into the prompt, so generation begins
/// INSIDE the reasoning block). A stream that never emits `</think>` (thinking truncated by the token
/// budget) therefore yields ALL text as `reasoning` with nil `content` — streaming cannot retro-label
/// already-sent reasoning as content, and by construction those tokens genuinely were reasoning.
/// Non-streaming, seeing no `</think>`, instead treats the whole (untrimmed) text as content. This is
/// the one intentional streaming/non-streaming difference and it is documented rather than papered over.
public struct StreamingReasoningSplitter {
    private static let closeTag = "</think>"
    private static let openTag = "<think>"

    private enum Phase { case reasoning, content }
    private var phase: Phase = .reasoning
    /// Unemitted tail: held-back tag-prefix and/or trailing whitespace waiting to be resolved.
    private var buffer: String = ""
    private var checkedLeadingThink = false
    private var reasoningLeadingWhitespaceDone = false
    private var contentLeadingWhitespaceDone = false

    public init() {}

    /// Consume one text delta and return the reasoning / content bytes that became emittable now
    /// (nil for a side that produced nothing on this call).
    public mutating func consume(_ delta: String) -> (reasoning: String?, content: String?) {
        buffer += delta
        return drain(final: false)
    }

    /// Signal end-of-stream: emit whatever remains, dropping held trailing whitespace.
    public mutating func flush() -> (reasoning: String?, content: String?) {
        return drain(final: true)
    }

    private static func isWhitespace(_ ch: Character) -> Bool {
        ch.unicodeScalars.count == 1
            && CharacterSet.whitespacesAndNewlines.contains(ch.unicodeScalars.first!)
    }

    private static func trimmingTrailingWhitespace(_ s: Substring) -> String {
        var end = s.endIndex
        while end > s.startIndex {
            let prev = s.index(before: end)
            if isWhitespace(s[prev]) { end = prev } else { break }
        }
        return String(s[s.startIndex..<end])
    }

    /// Length of the suffix of `s` that must be held back in the reasoning phase: a trailing partial
    /// match of `</think>` (so the tag is never torn), plus any whitespace immediately preceding it
    /// (which becomes trailing-and-trimmed if the tag completes, or interior if it does not).
    private static func reasoningHeldSuffixLength(_ s: Substring) -> Int {
        var tagLen = 0
        let maxK = min(s.count, closeTag.count - 1)
        var k = maxK
        while k >= 1 {
            if closeTag.hasPrefix(s.suffix(k)) { tagLen = k; break }
            k -= 1
        }
        let beforeTag = s.dropLast(tagLen)
        var wsLen = 0
        for ch in beforeTag.reversed() {
            if isWhitespace(ch) { wsLen += 1 } else { break }
        }
        return tagLen + wsLen
    }

    private mutating func drain(final: Bool) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""

        loop: while true {
            switch phase {
            case .reasoning:
                // 1. Optional leading `<think>` strip (matches non-streaming's hasPrefix check on the
                //    UNtrimmed reasoning: a leading whitespace suppresses the strip).
                if !checkedLeadingThink {
                    guard let first = buffer.first else { break loop }
                    if Self.isWhitespace(first) {
                        checkedLeadingThink = true
                    } else if buffer.hasPrefix(Self.openTag) {
                        buffer.removeFirst(Self.openTag.count)
                        checkedLeadingThink = true
                        continue loop
                    } else if !final && Self.openTag.hasPrefix(buffer) {
                        break loop  // ambiguous partial `<think>` — wait for more
                    } else {
                        checkedLeadingThink = true
                    }
                }

                // 2. Drop leading whitespace of the reasoning block.
                if !reasoningLeadingWhitespaceDone {
                    while let f = buffer.first, Self.isWhitespace(f) { buffer.removeFirst() }
                    if buffer.isEmpty { break loop }
                    reasoningLeadingWhitespaceDone = true
                }

                // 3. First `</think>` ends the reasoning block.
                if let range = buffer.range(of: Self.closeTag) {
                    let seg = buffer[buffer.startIndex..<range.lowerBound]
                    reasoning += Self.trimmingTrailingWhitespace(seg)
                    buffer = String(buffer[range.upperBound...])
                    phase = .content
                    continue loop
                }

                if final {
                    // No `</think>` ever arrived (documented divergence): the held tag-prefix is
                    // literal reasoning; drop only trailing whitespace.
                    reasoning += Self.trimmingTrailingWhitespace(buffer[...])
                    buffer = ""
                    break loop
                }

                // Emit everything except the held-back suffix (tag prefix + adjacent trailing ws).
                let held = Self.reasoningHeldSuffixLength(buffer[...])
                let emitEnd = buffer.index(buffer.endIndex, offsetBy: -held)
                reasoning += String(buffer[buffer.startIndex..<emitEnd])
                buffer = String(buffer[emitEnd...])
                break loop

            case .content:
                // Drop leading whitespace of the answer.
                if !contentLeadingWhitespaceDone {
                    while let f = buffer.first, Self.isWhitespace(f) { buffer.removeFirst() }
                    if buffer.isEmpty { break loop }
                    contentLeadingWhitespaceDone = true
                }

                if final {
                    content += Self.trimmingTrailingWhitespace(buffer[...])
                    buffer = ""
                    break loop
                }

                // Answer bytes pass through verbatim; only a trailing whitespace run is held (it may
                // be the trimmed end of the stream). No tag scanning — a later `</think>` is literal.
                var wsLen = 0
                for ch in buffer.reversed() {
                    if Self.isWhitespace(ch) { wsLen += 1 } else { break }
                }
                let emitEnd = buffer.index(buffer.endIndex, offsetBy: -wsLen)
                content += String(buffer[buffer.startIndex..<emitEnd])
                buffer = String(buffer[emitEnd...])
                break loop
            }
        }

        return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
    }
}

/// Gates the streaming reasoning split on a GENERATION-SIDE signal: whether the model's output stream
/// itself begins with a `<think>` block. The self-emitting thinking families we serve (Qwen3, Qwen3.5)
/// do NOT pre-open `<think>` in the prompt — the chat template injects a CLOSED empty `<think></think>`
/// when thinking is OFF and injects nothing when it is ON, so the model emits the opening `<think>` as
/// its first generated tokens. The honest, prompt-independent gate is therefore: if the first
/// non-whitespace output is `<think>`, this is a thinking stream → route through
/// `StreamingReasoningSplitter` (reasoning until `</think>`, then content); otherwise pass the stream
/// through as raw content, byte-for-byte identical to a non-thinking response (no trimming, no relabeling).
///
/// The one shape this does NOT cover is an R1-style checkpoint whose PROMPT pre-opens `<think>` (so the
/// generation contains only the closing `</think>`); we serve none today. If one is ever added, a
/// prompt-side "prompt ends in an open think block" flag can be threaded then and live-verified — this
/// gate stays correct for it by simply never firing (safe: today's raw-content behavior).
///
/// ⚠️ NOT SAFE STANDALONE — the gate is FAMILY-BLIND. It cannot distinguish a genuine reasoning opener
/// from *content that happens to begin with the literal text `<think>`*. A stream whose ANSWER starts
/// with `<think>` and never emits `</think>` (e.g. echoed markup, a non-thinking model, or the no-opener
/// Qwen3.5 family where a leading `<think>` is spurious) is routed ENTIRELY into `reasoning`, so a client
/// rendering only `content` loses the answer — strictly MORE corrupting than both the raw-content and
/// non-streaming paths (see `testLeadingThinkAsContentWithoutCloseIsMislabeled`). This is why the SSE
/// wiring was reverted (commit `3a806f6`): the gate must be composed with a real thinking-active signal
/// (the deferred `thinksByDefault` family classifier + `resolvedEnableThinking != false`, see
/// `docs/task-inbox/2026-08-19-streaming-reasoning-sse-wiring.md`) that only routes a stream through it
/// when the family is KNOWN to be reasoning. As a bare component it is only the `</think>` partitioner,
/// not a safe gate.
///
/// SHIPPED RESOLUTION (streaming separation): the real gate that composes that family signal is
/// `servingSeparatesReasoning(thinksByDefault:resolvedEnableThinking:)` (StreamingReasoningPolicy.swift),
/// which the loader/backend derive onto `ServingGenerationHandle.separatesReasoning`; the streaming SSE
/// handler then feeds `.text` deltas DIRECTLY through `StreamingReasoningSplitter` (which begins in the
/// reasoning phase, correct for the no-opener qwen3_5 family). This `StreamingReasoningGate` is NOT wired
/// into that path — for the no-opener family it would passthrough exactly when it must split (wrong
/// polarity), and the splitter already strips an optional leading `<think>`, so the gate adds nothing. It
/// remains a shelved component for a hypothetical opener-REQUIRED checkpoint, kept for its tested
/// leading-`<think>` decision logic.
public struct StreamingReasoningGate {
    private static let openTag = "<think>"

    private enum Mode { case deciding, thinking, passthrough }
    private var mode: Mode = .deciding
    /// Head bytes held back only while deciding whether the stream opens with `<think>`. Bounded: leading
    /// whitespace plus at most `openTag.count - 1` tag-prefix bytes before the decision resolves.
    private var head: String = ""
    private var splitter = StreamingReasoningSplitter()

    public init() {}

    /// Consume one raw text delta; return the reasoning / content bytes emittable now (nil per side that
    /// produced nothing — e.g. while still buffering the head to decide).
    public mutating func consume(_ delta: String) -> (reasoning: String?, content: String?) {
        switch mode {
        case .thinking:
            return splitter.consume(delta)
        case .passthrough:
            return (nil, delta.isEmpty ? nil : delta)
        case .deciding:
            head += delta
            return decide(final: false)
        }
    }

    /// Signal end-of-stream: resolve any still-buffered head and flush the splitter if engaged.
    public mutating func flush() -> (reasoning: String?, content: String?) {
        switch mode {
        case .thinking:
            return splitter.flush()
        case .passthrough:
            return (nil, nil)
        case .deciding:
            return decide(final: true)
        }
    }

    private static func isWhitespace(_ ch: Character) -> Bool {
        ch.unicodeScalars.count == 1
            && CharacterSet.whitespacesAndNewlines.contains(ch.unicodeScalars.first!)
    }

    private mutating func decide(final: Bool) -> (reasoning: String?, content: String?) {
        // Leading whitespace is not meaningful for the tag decision (and the splitter trims reasoning
        // leading whitespace anyway); measure the tag against the whitespace-stripped head.
        let trimmed = head.drop(while: Self.isWhitespace)

        if trimmed.isEmpty {
            // Nothing but whitespace so far. Wait for more unless the stream ended, in which case this is
            // a degenerate all-whitespace answer → passthrough it verbatim.
            guard final else { return (nil, nil) }
            mode = .passthrough
            let out = head
            head = ""
            return (nil, out.isEmpty ? nil : out)
        }

        if String(trimmed).hasPrefix(Self.openTag) {
            // Confirmed leading `<think>` → thinking stream. Feed the whitespace-stripped head (starting at
            // the tag) to the splitter, which performs its own `<think>` strip and `</think>` partition.
            mode = .thinking
            head = ""
            return splitter.consume(String(trimmed))
        }

        if !final && Self.openTag.hasPrefix(String(trimmed)) {
            // Still an ambiguous partial `<think` prefix — wait for more before deciding.
            return (nil, nil)
        }

        // Definitively not a leading `<think>` → passthrough. Emit the ORIGINAL head verbatim (including
        // any leading whitespace) so a non-thinking stream is byte-for-byte identical to today.
        mode = .passthrough
        let out = head
        head = ""
        return (nil, out.isEmpty ? nil : out)
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
    try rejectUnknownKeys(
        in: object,
        allowed: ["role", "content", "tool_calls", "tool_call_id", "name"],
        paramPrefix: "messages")
    let roleValue = try requiredString(object["role"], param: "messages.role")
    guard let role = OpenAIMessageRole(rawValue: roleValue) else {
        throw OpenAIServingError.invalidRequest("Unsupported message role: \(roleValue)", param: "messages.role")
    }

    let toolCalls = try OpenAIToolDecoding.decodeToolCalls(object["tool_calls"], param: "messages.tool_calls")
    let name = try optionalName(object["name"])
    let toolCallId = try optionalToolCallId(object["tool_call_id"])

    switch role {
    case .tool:
        guard let toolCallId else {
            throw OpenAIServingError.invalidRequest(
                "tool messages require a tool_call_id", param: "messages.tool_call_id")
        }
        guard toolCalls.isEmpty else {
            throw OpenAIServingError.invalidRequest(
                "tool messages must not carry tool_calls", param: "messages.tool_calls")
        }
        let text = try decodeContent(object["content"])
        return OpenAIChatMessage(role: role, text: text, toolCallId: toolCallId, name: name)
    case .assistant:
        let text = try decodeAssistantContent(object["content"], hasToolCalls: !toolCalls.isEmpty)
        return OpenAIChatMessage(role: role, text: text, toolCalls: toolCalls, name: name)
    default:
        guard toolCalls.isEmpty else {
            throw OpenAIServingError.invalidRequest(
                "Only assistant messages may carry tool_calls", param: "messages.tool_calls")
        }
        let text = try decodeContent(object["content"])
        return OpenAIChatMessage(role: role, text: text, name: name)
    }
}

/// Assistant turns may omit/null `content` when they carry `tool_calls`.
private func decodeAssistantContent(_ raw: Any?, hasToolCalls: Bool) throws -> String {
    if raw == nil || raw is NSNull {
        if hasToolCalls { return "" }
        throw OpenAIServingError.invalidRequest(
            "messages.content must be a string or non-empty text part array",
            param: "messages.content")
    }
    return try decodeContent(raw)
}

private func optionalName(_ raw: Any?) throws -> String? {
    guard let raw, !(raw is NSNull) else { return nil }
    guard let name = raw as? String else {
        throw OpenAIServingError.invalidRequest("messages.name must be a string", param: "messages.name")
    }
    return name
}

private func optionalToolCallId(_ raw: Any?) throws -> String? {
    guard let raw, !(raw is NSNull) else { return nil }
    guard let id = raw as? String, !id.isEmpty else {
        throw OpenAIServingError.invalidRequest(
            "messages.tool_call_id must be a non-empty string", param: "messages.tool_call_id")
    }
    return id
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
    return try exactInteger(raw, param: param, lowerBound: Decimal(Int.min), upperBound: Decimal(Int.max), convert: Int.init)
}

private func optionalInt64(_ raw: Any?, param: String) throws -> Int64? {
    guard let raw else { return nil }
    return try exactInteger(
        raw,
        param: param,
        lowerBound: Decimal(Int64.min),
        upperBound: Decimal(Int64.max),
        convert: Int64.init)
}

private func optionalPositiveInt(_ raw: Any?, param: String) throws -> Int? {
    guard let value = try optionalInt(raw, param: param) else { return nil }
    guard value > 0 else {
        throw OpenAIServingError.invalidRequest("\(param) must be greater than zero", param: param)
    }
    return value
}

private func exactInteger<T>(
    _ raw: Any,
    param: String,
    lowerBound: Decimal,
    upperBound: Decimal,
    convert: (String) -> T?
) throws -> T {
    guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }

    let decimal = number.decimalValue
    let exact = NSDecimalNumber(decimal: decimal)
    guard exact != NSDecimalNumber.notANumber,
          exact.doubleValue.isFinite else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }

    var roundedDecimal = Decimal()
    var workingDecimal = decimal
    NSDecimalRound(&roundedDecimal, &workingDecimal, 0, .plain)
    guard roundedDecimal == decimal,
          decimal >= lowerBound,
          decimal <= upperBound else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }

    let integerString = NSDecimalNumber(decimal: decimal).stringValue
    guard let value = convert(integerString) else {
        throw OpenAIServingError.invalidRequest("\(param) must be an integer", param: param)
    }
    return value
}

private func optionalString(_ raw: Any?, param: String) throws -> String? {
    guard let raw, !(raw is NSNull) else { return nil }
    guard let string = raw as? String else {
        throw OpenAIServingError.invalidRequest("\(param) must be a string", param: param)
    }
    return string
}

private let supportedReasoningEfforts: Set<String> = ["xhigh", "medium", "low"]

private func optionalReasoningEffort(_ raw: Any?, param: String) throws -> String? {
    guard let value = try optionalString(raw, param: param) else { return nil }
    guard supportedReasoningEfforts.contains(value) else {
        throw OpenAIServingError.invalidRequest(
            "\(param) must be one of xhigh, medium, low", param: param)
    }
    return value
}

/// Decodes the Qwen `chat_template_kwargs` passthrough dict, extracting only the
/// `enable_thinking` / `reasoning_effort` fields we understand. Other keys are ignored.
private func decodeChatTemplateKwargs(_ raw: Any?) throws -> (enableThinking: Bool?, reasoningEffort: String?) {
    guard let raw, !(raw is NSNull) else { return (nil, nil) }
    guard let object = raw as? [String: Any] else {
        throw OpenAIServingError.invalidRequest(
            "chat_template_kwargs must be an object", param: "chat_template_kwargs")
    }
    let enableThinking = try optionalBool(object["enable_thinking"], param: "chat_template_kwargs.enable_thinking")
    let reasoningEffort = try optionalReasoningEffort(
        object["reasoning_effort"], param: "chat_template_kwargs.reasoning_effort")
    return (enableThinking, reasoningEffort)
}
