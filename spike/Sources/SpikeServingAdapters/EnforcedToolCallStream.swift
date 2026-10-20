import Foundation

import MLXLMCommon
import ServingCore

/// Incremental form of an enforced call: the post-reasoning generated text is fed in as it is
/// decoded and comes out as the call's name (once, when the call is known) plus the bytes that
/// belong to its `arguments`. Named: the generated text IS the arguments, so the head goes out
/// with the first non-empty text. Required: the union document's frame is stripped and the
/// wrapper's closing brace held back (`ToolUnionArgumentStream`).
///
/// Shared by the scalar and the continuous-batch serving backends, so both release an enforced
/// `tool_choice` as the same `.toolCallStart` + `.toolCallArguments` sequence.
enum EnforcedToolCallStream: Sendable {
    case named(name: String, headSent: Bool)
    case union(ToolUnionArgumentStream)

    mutating func consume(_ text: String) -> (name: String?, arguments: String) {
        switch self {
        case .named(let name, let headSent):
            guard !text.isEmpty else { return (nil, "") }
            self = .named(name: name, headSent: true)
            return (headSent ? nil : name, text)
        case .union(var stream):
            let output = stream.consume(text)
            self = .union(stream)
            return (output.name, output.arguments)
        }
    }
}

/// An id for an enforced call's head, in the same shape the vendored `ToolCallFormat
/// .generateToolCallID()` (internal to that module) gives parsed calls: `call_<32 hex>`, or 9 hex
/// characters for the Mistral format.
func makeEnforcedToolCallID(format: ToolCallFormat) -> String {
    let uuid = UUID().uuidString.replacingOccurrences(of: "-", with: "")
    switch format {
    case .mistral: return String(uuid.prefix(9))
    default: return "call_" + uuid.lowercased()
    }
}
