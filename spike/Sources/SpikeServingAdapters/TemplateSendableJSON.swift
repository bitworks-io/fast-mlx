import Foundation

import Jinja
import ServingCore

/// Template-safe conversion of `ServingJSONValue` for the chat-template `messages`/`tools` inputs.
///
/// `ServingJSONValue.asSendable` (ServingCore) maps `.null` to Foundation's `NSNull()`, which
/// swift-jinja's `Value(any:)` has no case for: it throws `Cannot convert value of type NSNull to
/// Jinja Value`, so a history tool call with a `null` argument (or a tool schema with
/// `"default": null`) was rejected as `chat_template_rejected`. Here `.null` becomes an explicit
/// `Jinja.Value.null` (Python `None`, rendered by `tojson` as `null`, as in transformers).
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
