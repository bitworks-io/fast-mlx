import XCTest

import MLXLMCommon

/// Chunk-boundary invariance for the XML-function tool-call streaming parser
/// (`MLXLMCommon.ToolCallProcessor(format: .xmlFunction, ...)`), the wire format `ToolCallFormat
/// .infer(from:)` selects for the `qwen4_exp` family (Qwen3.8-Flash-Next) — see
/// `ToolCallFormat.swift`'s `type.hasPrefix("qwen4_exp")` case.
///
/// This matters because speculative decoding (MTP) can change how generated text is segmented
/// into chunks without changing the generated text itself. The parser must produce the SAME parsed
/// tool call (or the same absence of one) regardless of how the underlying text is split.
///
/// Deliberately UNCONDITIONAL (no `FASTMLX_QWEN3_TOKENIZER_DIR` env gate, unlike
/// `Qwen3TemplateRenderTests.swift`) — this suite uses no model weights and no tokenizer, only the
/// pure-Swift `ToolCallProcessor`/`XMLFunctionParser`, so it is the CI-visible control for the
/// gated template suite: if this ever regresses, it fails on every run, not only on machines with a
/// local tokenizer fixture.
///
/// Deviation from the original task brief (documented per AGENTS.md "if something in the spec turns
/// out to be wrong against the real code... implement the nearest correct thing"): the brief's
/// example payload was `<tool_call>\n{"name": "get_weather", "arguments": {...}}\n</tool_call>`,
/// which is the WIRE FORMAT FOR `.json`, not `.xmlFunction`. `ToolCallFormat.infer` routes
/// `qwen4_exp` to `.xmlFunction`, whose real wire format —
/// `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` — is
/// confirmed by `XMLFunctionParser.swift`'s doc comment and by
/// `ScalarServingBackendTests.testScalarRouteHonorsXMLFunctionToolCallFormat`. This suite targets
/// that real format. Correspondingly, "a split that lands inside the JSON" (arm 2 of the brief) has
/// no literal JSON body in this format; the nearest correct equivalent implemented below is a split
/// that lands inside the `<parameter=...>` argument VALUE instead (see `Segmentation` table below).
final class ToolCallChunkBoundaryTests: XCTestCase {
    /// A realistic complete `xmlFunction` tool-call output, matching the production
    /// `chat_template.jinja`'s documented tool-call reply format exactly (see
    /// `deploy/qwen3.8-27b/chat_template.jinja`'s "<tool_call>\n<function=example_function_name>\n
    /// <parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>" example).
    private static let completeToolCallText =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"

    /// Plain prose with no tool call at all. Anti-vacuity control (a): the processor must not
    /// manufacture a tool call from ordinary text, and must not swallow it either.
    private static let plainProseWithNoToolCall =
        "The weather in Paris is sunny today, so there is no reason to check anything further right now."

    /// Prose containing a `<` character AND the word "tool", but NOT a real `<tool_call>` tag.
    /// Anti-vacuity control (b): a naive "contains '<tool'" heuristic would false-fire on this; the
    /// real parser must not, because `<check>` does not match the `<tool_call>` start tag.
    private static let proseWithAngleBracketAndToolWord =
        "Please <check> the tool status page for details, but do not call anything yet today."

    private static let noToolCallProseFixtures: [String] = [
        plainProseWithNoToolCall,
        proseWithAngleBracketAndToolWord,
    ]

    /// One entry per chunk segmentation strategy. Adding a new segmentation is one line here.
    private struct Segmentation: Sendable {
        let name: String
        let chunker: @Sendable (String) -> [String]
    }

    private static let segmentations: [Segmentation] = [
        Segmentation(name: "whole string at once") { text in [text] },
        Segmentation(name: "one character per chunk") { text in text.map { String($0) } },
        Segmentation(name: "two characters per chunk") { text in
            let characters = Array(text)
            var result: [String] = []
            var index = 0
            while index < characters.count {
                let end = min(index + 2, characters.count)
                result.append(String(characters[index..<end]))
                index = end
            }
            return result
        },
        // For `completeToolCallText` (95 chars) this lands at index 4 — inside the opening
        // "<tool_call>" tag (span [0, 11)), between "tool" and "_call>". For other fixture texts
        // it is simply an early two-way split; the property under test (same result regardless of
        // split point) still applies.
        Segmentation(name: "split inside the opening <tool_call> tag (for the fixture above)") { text in
            let characters = Array(text)
            let splitIndex = min(Int(Double(characters.count) * 0.05), characters.count)
            return [String(characters[0..<splitIndex]), String(characters[splitIndex...])]
        },
        // For `completeToolCallText` this lands at index 55 — inside the "Paris" parameter value
        // (span [52, 57)), between "Pari" and "s". This is the nearest correct equivalent of the
        // task brief's "split lands inside the JSON" for a format with no JSON body (see the
        // deviation note in the file header).
        Segmentation(name: "split inside the parameter value (for the fixture above)") { text in
            let characters = Array(text)
            let splitIndex = min(Int(Double(characters.count) * 0.58), characters.count)
            return [String(characters[0..<splitIndex]), String(characters[splitIndex...])]
        },
    ]

    /// Feeds `chunks` through a FRESH `.xmlFunction` processor and returns the parsed tool calls
    /// plus the concatenated visible/display text (the non-tool-call text a client would see).
    private func run(chunks: [String]) -> (calls: [ToolCall], displayed: String) {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: nil)
        var displayed = ""
        for chunk in chunks {
            if let visible = processor.processChunk(chunk) {
                displayed += visible
            }
        }
        if let residual = processor.processEOS(returnBufferedText: true) {
            displayed += residual
        }
        return (processor.toolCalls, displayed)
    }

    /// (1) A complete, realistic `xmlFunction` tool call must parse to the SAME single tool call —
    /// same function name, same parsed arguments — no matter how the generated text is segmented
    /// into chunks. Also (3): the visible/display text must never contain the raw `<tool_call>`
    /// markup.
    func testXMLFunctionToolCallParsesIdenticallyAcrossChunkSegmentations() throws {
        for segmentation in Self.segmentations {
            let chunks = segmentation.chunker(Self.completeToolCallText)
            let result = run(chunks: chunks)

            XCTAssertEqual(
                result.calls.count, 1,
                "segmentation '\(segmentation.name)' expected exactly one tool call, got \(result.calls.count): \(result.calls)"
            )
            guard let call = result.calls.first else { continue }
            XCTAssertEqual(
                call.function.name, "get_weather",
                "segmentation '\(segmentation.name)' parsed the wrong function name")
            XCTAssertEqual(
                call.function.arguments["city"], .string("Paris"),
                "segmentation '\(segmentation.name)' parsed the wrong argument value")
            XCTAssertFalse(
                result.displayed.contains("<tool_call>"),
                "segmentation '\(segmentation.name)' leaked raw <tool_call> markup into display text: \(result.displayed)"
            )
        }
    }

    /// (2) Anti-vacuity / false-fire control: prose with no tool call — including prose that
    /// contains a `<` character and the word "tool" without a real `<tool_call>` tag — must yield
    /// ZERO tool calls under every one of the same segmentations, and must not have its visible
    /// text swallowed (non-empty display text).
    func testNonToolCallProseYieldsZeroToolCallsAcrossChunkSegmentations() throws {
        for prose in Self.noToolCallProseFixtures {
            for segmentation in Self.segmentations {
                let chunks = segmentation.chunker(prose)
                let result = run(chunks: chunks)

                XCTAssertEqual(
                    result.calls.count, 0,
                    "segmentation '\(segmentation.name)' on prose '\(prose)' must not manufacture a tool call, got \(result.calls)"
                )
                XCTAssertFalse(
                    result.displayed.isEmpty,
                    "segmentation '\(segmentation.name)' on prose '\(prose)' swallowed ordinary text (display text is empty)"
                )
            }
        }
    }
}
