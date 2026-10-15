import Foundation

/// Routes detokenized chunks so the tool-call parser sees exactly the bytes the client is shown as
/// content (the HTTP reasoning splitters label everything before the first `</think>` as
/// `reasoning_content`).
///
/// The gate starts in the reasoning phase. Everything up to and INCLUDING the first `</think>` is
/// `passthrough` (the tag itself must still reach the client so the splitter sees it); only the bytes
/// after it, and every later chunk, are `toParser`. A suffix of the buffered reasoning text that is a
/// proper prefix of `</think>` (at most 7 bytes) is held back until resolved, so the tag is never torn
/// across chunks; `flush()` releases it as reasoning at the end of the stream.
///
/// Bytes are conserved: `passthrough` then `toParser` for every `route` call in order, followed by
/// `flush()`, equals the concatenated input.
public struct ReasoningPhaseToolCallGate: Sendable {
    private static let closingTag = Array("</think>".utf8)

    private var inReasoning = true
    private var held: [UInt8] = []

    public init() {}

    public mutating func route(_ text: String) -> (passthrough: String, toParser: String) {
        guard inReasoning else {
            return ("", text)
        }
        let buffer = held + Array(text.utf8)
        held = []
        let tag = Self.closingTag
        if let start = Self.firstIndex(of: tag, in: buffer) {
            inReasoning = false
            let end = start + tag.count
            return (
                String(decoding: buffer[..<end], as: UTF8.self),
                String(decoding: buffer[end...], as: UTF8.self)
            )
        }
        // Hold back the longest proper prefix of the tag that the buffer ends with.
        var holdCount = 0
        for length in stride(from: min(tag.count - 1, buffer.count), through: 1, by: -1)
        where buffer.suffix(length).elementsEqual(tag.prefix(length)) {
            holdCount = length
            break
        }
        held = Array(buffer.suffix(holdCount))
        return (String(decoding: buffer.dropLast(holdCount), as: UTF8.self), "")
    }

    /// Bytes held back for a possible `</think>`; they are reasoning when the stream ends unresolved.
    /// Idempotent: a second call returns the empty string.
    public mutating func flush() -> String {
        defer { held = [] }
        return String(decoding: held, as: UTF8.self)
    }

    /// End of stream: route the stop filter's held `tail` like a chunk (the held bytes are prepended, so
    /// stream order holds), then release whatever is still held as reasoning. A tail that completes a
    /// split `</think>` is therefore recognised, and in the parser phase the tail is parser input.
    /// `finish(tail: nil)` is `flush()` returned as passthrough. Idempotent: a second call is empty.
    public mutating func finish(tail: String?) -> (passthrough: String, toParser: String) {
        let routed = tail.map { route($0) } ?? (passthrough: "", toParser: "")
        return (routed.passthrough + flush(), routed.toParser)
    }

    private static func firstIndex(of needle: [UInt8], in haystack: [UInt8]) -> Int? {
        guard haystack.count >= needle.count else {
            return nil
        }
        for start in 0...(haystack.count - needle.count)
        where haystack[start] == needle[0] && haystack[start..<(start + needle.count)].elementsEqual(needle) {
            return start
        }
        return nil
    }
}
