import Foundation

import Jinja
import ServingCore

/// Template-safe conversion of `ServingJSONValue` for the chat-template `messages`/`tools` inputs.
///
/// `ServingJSONValue.asSendable` (ServingCore) maps `.null` to Foundation's `NSNull()`, which
/// swift-jinja's `Value(any:)` has no case for: it throws `Cannot convert value of type NSNull to
/// Jinja Value`, so a history tool call with a `null` argument (or a tool schema with
/// `"default": null`) was rejected. Here `.null` becomes an explicit
/// `Jinja.Value.null` (Python `None`, rendered by `tojson` as `null`, as in transformers).
/// (Before this fix the failure surfaced as an HTTP 500 `generation_failed` on the default route,
/// not a `chat_template_rejected` 400: the `JinjaError` is not a `TemplateException`.)
/// `Value(any:)` passes a `Value` through unchanged. Every other case is identical to
/// `asSendable`, so no other value changes rendering. ServingCore keeps its Foundation
/// `asSendable` for its other consumers.
extension ServingJSONValue {
    var templateSendable: any Sendable {
        switch self {
        case .null: return Jinja.Value.null
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { $0.templateSendable }
        case .object(let values): return values.mapValues { $0.templateSendable }
        }
    }

    /// The object entries as a template-safe Sendable dictionary (nil unless this is an `.object`).
    var templateObjectSendable: [String: any Sendable]? {
        guard case .object(let object) = self else { return nil }
        return object.mapValues { $0.templateSendable }
    }
}

/// Order-preserving conversion for the chat-template `tools` / history `arguments` inputs.
///
/// `Value(any:)` sorts every Swift dictionary's keys, so a `[String: Any]` tool renders via
/// `tojson` in alphabetical order. Python transformers renders the client's JSON order. A
/// `Jinja.Value` passes through `Value(any:)` unchanged and `.object` holds an
/// `OrderedDictionary`, so building the `Value` tree directly from the order-preserving parse keeps
/// the client's key order at every nesting level. Numbers follow Python `json.loads`: a literal
/// matching the JSON integer grammar is an int (a double if it overflows `Int`); anything with a
/// fraction or exponent is a float.
extension JSONSchemaSubsetCompiler.RawValue {
    var templateJinjaValue: Jinja.Value {
        switch self {
        case .null: return .null
        case .bool(let value): return .boolean(value)
        case .string(let value): return .string(value)
        case .number(let text): return Self.jinjaNumber(text)
        case .array(let values): return .array(values.map { $0.templateJinjaValue })
        case .object(let pairs):
            var storage = OrderedDictionary<ObjectKey, Jinja.Value>(minimumCapacity: pairs.count)
            for (key, value) in pairs { storage[.string(key)] = value.templateJinjaValue }
            return .object(storage)
        }
    }

    private static func jinjaNumber(_ text: String) -> Jinja.Value {
        if isJSONIntegerLiteral(text), let int = Int(text) { return .int(int) }
        return .double(Double(text) ?? 0)
    }

    /// `-?(0|[1-9][0-9]*)`, nothing else.
    private static func isJSONIntegerLiteral(_ text: String) -> Bool {
        var digits = Substring(text)
        if digits.first == "-" { digits = digits.dropFirst() }
        guard let first = digits.first else { return false }
        guard digits.allSatisfy({ $0 >= "0" && $0 <= "9" }) else { return false }
        return first != "0" || digits.count == 1
    }
}
