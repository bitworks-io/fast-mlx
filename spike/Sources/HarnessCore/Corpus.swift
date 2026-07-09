import Foundation

/// A hermetic (weight-free) input→expected pair. Real entries are harvested by capturing actual
/// model output (raw, pre-chat-layer text containing think-tags/tool-call markup) and pasting it
/// here alongside the expected post-processing result — no model weights required to run the suite.
public struct CorpusEntry: Sendable {
    public let name: String
    public let raw: String
    public let expectedVisible: String?
    public let expectedTool: String?
    public init(name: String, raw: String, expectedVisible: String? = nil, expectedTool: String? = nil) {
        self.name = name; self.raw = raw; self.expectedVisible = expectedVisible; self.expectedTool = expectedTool
    }
}

public struct CorpusProcessed: Sendable {
    public let visibleText: String
    public let toolArgsJSON: String?
    public init(visibleText: String, toolArgsJSON: String? = nil) {
        self.visibleText = visibleText; self.toolArgsJSON = toolArgsJSON
    }
}

/// Universal invariants + hermetic corpus. `process` is a minimal pure implementation (strip
/// think-tags, lift a trailing tool-call JSON block) that satisfies the invariants today; it gets
/// wired to the engine's real chat/tool-parse layer once that layer exists (see plan notes).
public enum HarnessCorpus {
    /// Strip `<think>...</think>` spans and any other `<|...|>` control tags from visible text;
    /// lift a `<tool_call>{...}</tool_call>` block (if present) into `toolArgsJSON`.
    public static func process(_ raw: String) -> CorpusProcessed {
        var text = raw

        // Strip <think>...</think> blocks (including their content). An UNCLOSED trailing <think>
        // (truncated generation — e.g. hit max-tokens mid-reasoning) is stripped from the tag to the
        // end of the text so reasoning never leaks into visible output.
        while let openRange = text.range(of: "<think>") {
            if let closeRange = text.range(of: "</think>", range: openRange.upperBound..<text.endIndex) {
                text.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
            } else {
                text.removeSubrange(openRange.lowerBound..<text.endIndex)
                break
            }
        }

        // Lift a <tool_call>...</tool_call> block into its own field; remove it from visible text.
        var toolArgs: String?
        if let openRange = text.range(of: "<tool_call>"), let closeRange = text.range(of: "</tool_call>") {
            let contentStart = openRange.upperBound
            let contentEnd = closeRange.lowerBound
            if contentStart <= contentEnd {
                toolArgs = String(text[contentStart..<contentEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            text.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
        }

        // Strip any remaining control tags of the form <|...|>.
        while let start = text.range(of: "<|"), let end = text.range(of: "|>", range: start.upperBound..<text.endIndex) {
            text.removeSubrange(start.lowerBound..<end.upperBound)
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return CorpusProcessed(visibleText: text, toolArgsJSON: toolArgs)
    }

    public static let entries: [CorpusEntry] = [
        CorpusEntry(
            name: "plain_response",
            raw: "The capital of France is Paris.",
            expectedVisible: "The capital of France is Paris."
        ),
        CorpusEntry(
            name: "think_tag_stripped",
            raw: "<think>reasoning about the answer here</think>The answer is 42.",
            expectedVisible: "The answer is 42."
        ),
        CorpusEntry(
            name: "tool_call_lifted",
            raw: "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
            expectedVisible: "",
            expectedTool: "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
        ),
        CorpusEntry(
            name: "control_tag_leak_guard",
            raw: "<|im_start|>assistant\nHello there.<|im_end|>",
            expectedVisible: "assistant\nHello there."
        ),
        // Deliberately hostile bytes: emoji, combining marks, null-adjacent control chars, RTL override.
        CorpusEntry(
            name: "hostile_bytes",
            raw: "<think>\u{0000}\u{202E}garbage</think>Résult\u{0301}: \u{1F600} done.",
            expectedVisible: "Résult\u{0301}: \u{1F600} done."
        ),
        // Truncated generation: max-tokens hit mid-reasoning, so </think> never arrives. Must not leak.
        CorpusEntry(
            name: "unclosed_think_truncated",
            raw: "<think>reasoning that got cut off mid-generation",
            expectedVisible: ""
        ),
    ]
}
