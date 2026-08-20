import CoreFoundation
import Foundation

// OpenAI-compatible tool-calling contract types for the /v1/chat/completions route.
//
// This file is PURE Swift (no MLX): it defines the wire types and their strict decode/encode so
// the whole contract is unit-testable off-GPU. The serving adapter maps the vendored
// MLXLMCommon.ToolCall into `OpenAIToolCall` and converts `ServingJSONValue` tool specs into the
// `[String: any Sendable]` shape the tokenizer chat template consumes.
//
// Verified contract facts (from the tool-calling spec research):
// - OpenAI `function.arguments` is a JSON *string* (not an object).
// - `tool_choice`: "none" | "auto" | "required" | {"type":"function","function":{"name":…}}.
// - assistant tool-call turns carry `content:null` + `tool_calls`; `tool` results carry
//   `tool_call_id`. `finish_reason` becomes "tool_calls".

// MARK: - ServingJSONValue

/// A Sendable/Equatable JSON value used to carry arbitrary tool JSON-Schema through the pure
/// ServingCore layer without depending on MLXLMCommon's `JSONValue`.
public enum ServingJSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([ServingJSONValue])
    case object([String: ServingJSONValue])

    /// Build from a `JSONSerialization` value tree.
    public init(foundation value: Any) {
        switch value {
        case is NSNull:
            self = .null
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else {
                let d = number.doubleValue
                if d.rounded() == d, abs(d) < 9.007_199_254_740_992e15 {
                    self = .int(number.intValue)
                } else {
                    self = .double(d)
                }
            }
        case let string as String:
            self = .string(string)
        case let array as [Any]:
            self = .array(array.map { ServingJSONValue(foundation: $0) })
        case let object as [String: Any]:
            self = .object(object.mapValues { ServingJSONValue(foundation: $0) })
        default:
            self = .string(String(describing: value))
        }
    }

    /// The object entries as a Sendable dictionary (nil unless this is an `.object`).
    /// Used to hand tool specs / parsed arguments to the chat template.
    public var asObjectSendable: [String: any Sendable]? {
        guard case .object(let object) = self else { return nil }
        return object.mapValues { $0.asSendable }
    }

    /// Convert back to a Sendable Foundation tree suitable for the chat-template `tools` argument.
    public var asSendable: any Sendable {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { $0.asSendable }
        case .object(let values): return values.mapValues { $0.asSendable }
        }
    }

    /// Serialize to a compact JSON string (sorted keys) — used to produce OpenAI `arguments`.
    public func jsonString() -> String {
        let object = asFoundation
        guard JSONSerialization.isValidJSONObject(object)
            || !(object is [String: Any]) && !(object is [Any])
        else {
            return "{}"
        }
        // Wrap scalars are not valid top-level JSON for JSONSerialization; only objects/arrays are.
        if let data = try? JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]),
            let string = String(data: data, encoding: .utf8) {
            return string
        }
        return "{}"
    }

    private var asFoundation: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { $0.asFoundation }
        case .object(let values): return values.mapValues { $0.asFoundation }
        }
    }
}

// MARK: - Tool specs (request `tools`)

/// One entry of the request `tools` array: `{"type":"function","function":{name,description,parameters}}`.
public struct OpenAIToolSpec: Sendable, Equatable {
    public var name: String
    public var description: String?
    /// The JSON-Schema `parameters` object (may be absent → treated as empty object).
    public var parameters: ServingJSONValue
    /// The full, unmodified `{"type":"function","function":{…}}` object, forwarded verbatim to the
    /// chat template's `tools` argument.
    public var raw: ServingJSONValue

    public init(name: String, description: String?, parameters: ServingJSONValue, raw: ServingJSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.raw = raw
    }
}

// MARK: - tool_choice

public enum OpenAIToolChoice: Sendable, Equatable {
    case none
    case auto
    case required
    case function(String)
}

// MARK: - Tool calls (response + request history)

/// OpenAI tool call: `{"id","type":"function","function":{"name","arguments":<JSON string>}}`.
/// `arguments` is always a JSON string on the wire (both directions).
public struct OpenAIToolCall: Sendable, Equatable, Encodable {
    public struct Function: Sendable, Equatable, Encodable {
        public var name: String
        public var arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    public var id: String
    public var type: String
    public var function: Function

    public init(id: String, type: String = "function", function: Function) {
        self.id = id
        self.type = type
        self.function = function
    }
}

/// Streaming tool-call delta: `{"index","id?","type?","function":{"name?","arguments?"}}`.
public struct OpenAIToolCallDelta: Sendable, Equatable, Encodable {
    public struct Function: Sendable, Equatable, Encodable {
        public var name: String?
        public var arguments: String?

        public init(name: String?, arguments: String?) {
            self.name = name
            self.arguments = arguments
        }

        var isEmpty: Bool { name == nil && arguments == nil }
    }

    public var index: Int
    public var id: String?
    public var type: String?
    public var function: Function?

    public init(index: Int, id: String?, type: String?, function: Function?) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }

    private enum CodingKeys: String, CodingKey {
        case index, id, type, function
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(index, forKey: .index)
        if let id { try container.encode(id, forKey: .id) }
        if let type { try container.encode(type, forKey: .type) }
        if let function, !function.isEmpty { try container.encode(function, forKey: .function) }
    }
}

// MARK: - tool_choice resolution

public extension OpenAIChatCompletionRequest {
    /// The tools the model may actually call for this request, honoring `tool_choice`:
    /// - `.none` → no tools (the model cannot call anything);
    /// - `.function(name)` → only that named function (reliably restricting the callable set);
    /// - `.auto` / `.required` → all declared tools.
    var activeTools: [OpenAIToolSpec] {
        switch toolChoice {
        case .none:
            return []
        case .function(let name):
            return tools.filter { $0.name == name }
        case .auto, .required:
            return tools
        }
    }

    /// The thinking mode to render with, under the serving model's tool-thinking policy. Always
    /// honors an explicit client `enable_thinking`. When the client leaves it unset, `nil` means
    /// "use the model's own template default" (thinking on, for Qwen).
    ///
    /// `disableThinkingWhenToolsActive` forces `false` when tools are attached — a LEGACY workaround
    /// for very old dense Qwen3, where thinking-ON measurably degraded tool-call reliability (the
    /// model reasons about calling a tool then answers as if it already had; QwenLM/Qwen3 #1817,
    /// ~40% failure). It is set ONLY for that legacy family. The agentic qwen3_5 family
    /// (Qwen3.5/3.6/3.8) is trained + evaluated to think AND call tools together, so it passes the
    /// flag off and respects the template default — matching how other runtimes serve it.
    func resolvedEnableThinking(disableThinkingWhenToolsActive: Bool) -> Bool? {
        if let enableThinking {
            return enableThinking
        }
        return (disableThinkingWhenToolsActive && !activeTools.isEmpty) ? false : nil
    }
}

// MARK: - Decode helpers (strict, matching OpenAIChatCompletions.swift style)

enum OpenAIToolDecoding {
    /// Decode the request `tools` array. Empty array is allowed (behaves as no tools).
    static func decodeTools(_ raw: Any?) throws -> [OpenAIToolSpec] {
        guard let raw else { return [] }
        guard let array = raw as? [Any] else {
            throw OpenAIServingError.invalidRequest("tools must be an array", param: "tools")
        }
        return try array.enumerated().map { index, element in
            guard let object = element as? [String: Any] else {
                throw OpenAIServingError.invalidRequest("tools entries must be objects", param: "tools")
            }
            let type = (object["type"] as? String) ?? "function"
            guard type == "function" else {
                throw OpenAIServingError.invalidRequest(
                    "Only function tools are supported", param: "tools[\(index)].type")
            }
            guard let function = object["function"] as? [String: Any] else {
                throw OpenAIServingError.invalidRequest(
                    "tools entries require a function object", param: "tools[\(index)].function")
            }
            guard let name = function["name"] as? String, !name.isEmpty else {
                throw OpenAIServingError.invalidRequest(
                    "tools function requires a non-empty name", param: "tools[\(index)].function.name")
            }
            let description = function["description"] as? String
            let parametersValue: ServingJSONValue
            if let parameters = function["parameters"] {
                guard parameters is [String: Any] else {
                    throw OpenAIServingError.invalidRequest(
                        "tools function parameters must be a JSON object",
                        param: "tools[\(index)].function.parameters")
                }
                parametersValue = ServingJSONValue(foundation: parameters)
            } else {
                parametersValue = .object([:])
            }
            return OpenAIToolSpec(
                name: name,
                description: description,
                parameters: parametersValue,
                raw: ServingJSONValue(foundation: object))
        }
    }

    /// Decode `tool_choice`. Default is `.auto` when tools are present, otherwise `.none`.
    static func decodeToolChoice(_ raw: Any?, hasTools: Bool) throws -> OpenAIToolChoice {
        guard let raw else { return hasTools ? .auto : .none }
        if let string = raw as? String {
            switch string {
            case "none": return .none
            case "auto": return .auto
            case "required": return .required
            default:
                throw OpenAIServingError.invalidRequest(
                    "tool_choice must be none, auto, required, or a function object",
                    param: "tool_choice")
            }
        }
        if let object = raw as? [String: Any] {
            let type = object["type"] as? String
            guard type == "function", let function = object["function"] as? [String: Any],
                let name = function["name"] as? String, !name.isEmpty
            else {
                throw OpenAIServingError.invalidRequest(
                    "tool_choice function must be {\"type\":\"function\",\"function\":{\"name\":…}}",
                    param: "tool_choice")
            }
            return .function(name)
        }
        throw OpenAIServingError.invalidRequest(
            "tool_choice must be a string or a function object", param: "tool_choice")
    }

    /// Decode assistant-history `tool_calls`.
    static func decodeToolCalls(_ raw: Any?, param: String) throws -> [OpenAIToolCall] {
        guard let raw else { return [] }
        guard let array = raw as? [Any] else {
            throw OpenAIServingError.invalidRequest("\(param) must be an array", param: param)
        }
        return try array.map { element in
            guard let object = element as? [String: Any] else {
                throw OpenAIServingError.invalidRequest("\(param) entries must be objects", param: param)
            }
            guard let id = object["id"] as? String, !id.isEmpty else {
                throw OpenAIServingError.invalidRequest("\(param) entries require an id", param: "\(param).id")
            }
            guard let function = object["function"] as? [String: Any] else {
                throw OpenAIServingError.invalidRequest(
                    "\(param) entries require a function object", param: "\(param).function")
            }
            guard let name = function["name"] as? String, !name.isEmpty else {
                throw OpenAIServingError.invalidRequest(
                    "\(param) function requires a name", param: "\(param).function.name")
            }
            // arguments is a JSON string on the wire; tolerate an absent value as "{}".
            let arguments: String
            if let string = function["arguments"] as? String {
                arguments = string
            } else if function["arguments"] == nil {
                arguments = "{}"
            } else {
                throw OpenAIServingError.invalidRequest(
                    "\(param) function arguments must be a JSON string",
                    param: "\(param).function.arguments")
            }
            return OpenAIToolCall(id: id, function: .init(name: name, arguments: arguments))
        }
    }
}
