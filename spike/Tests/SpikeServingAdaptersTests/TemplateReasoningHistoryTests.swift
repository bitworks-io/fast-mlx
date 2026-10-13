import Foundation
import XCTest

import HuggingFace
import Jinja
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// An assistant history turn's `reasoning_content` reaches the chat template as `message.reasoning_content`
/// when (and only when) the client sent it, so a template's `is defined` test sees the same thing it sees
/// from Python transformers. Cycle 238.
final class TemplateReasoningHistoryTests: XCTestCase {
    /// A small hand-written excerpt in the shape of the served templates' assistant branch.
    private static let excerpt = try! Template(
        """
        {%- for message in messages %}
        {%- if message.role == "assistant" %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if message.reasoning_content is defined %}
        {{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}
        {%- endif %}
        {{- message.content + '<|im_end|>\\n' }}
        {%- else %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
        {%- endfor %}
        """)

    private final class Box: @unchecked Sendable {
        var messages: [[String: any Sendable]] = []
        var rendered: String?
    }

    private struct CapturingTokenizer: MLXLMCommon.Tokenizer {
        let box: Box

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            box.messages = messages
            box.rendered = try TemplateReasoningHistoryTests.excerpt.render([
                "messages": try Value(any: messages)
            ])
            return [box.rendered?.count ?? 0]
        }
    }

    private func renderHistory(_ messages: [OpenAIChatMessage]) throws -> Box {
        let box = Box()
        let codec = MLXScalarTextCodec(tokenizer: CapturingTokenizer(box: box))
        _ = try codec.render(
            messages: messages, tools: [], enableThinking: nil, reasoningEffort: nil,
            addGenerationPrompt: nil)
        return box
    }

    // T5
    func testReasoningContentKeyPresentAndEqualWhenSet() throws {
        let box = try renderHistory([
            OpenAIChatMessage(role: .user, text: "q"),
            OpenAIChatMessage(role: .assistant, text: "a", reasoningContent: "because"),
        ])
        XCTAssertEqual(box.messages[1]["reasoning_content"] as? String, "because")
        XCTAssertNil(box.messages[0]["reasoning_content"])
        XCTAssertTrue(
            try XCTUnwrap(box.rendered).contains("<think>\nbecause\n</think>\n\na<|im_end|>"),
            "reasoning not rendered:\n\(box.rendered ?? "nil")")
    }

    func testReasoningContentKeyAbsentWhenNil() throws {
        let box = try renderHistory([
            OpenAIChatMessage(role: .user, text: "q"),
            OpenAIChatMessage(role: .assistant, text: "a"),
        ])
        XCTAssertFalse(box.messages[1].keys.contains("reasoning_content"))
        // Positive control: the same template, same path, renders no think block without the key.
        let rendered = try XCTUnwrap(box.rendered)
        XCTAssertFalse(rendered.contains("<think>"), "unexpected think block:\n\(rendered)")
        XCTAssertTrue(rendered.contains("<|im_start|>assistant\na<|im_end|>"))
    }

    func testEmptyReasoningContentStringIsStillPassedThrough() throws {
        let box = try renderHistory([OpenAIChatMessage(role: .assistant, text: "a", reasoningContent: "")])
        XCTAssertEqual(box.messages[0]["reasoning_content"] as? String, "")
        XCTAssertTrue(try XCTUnwrap(box.rendered).contains("<think>\n\n</think>"))
    }

    // Cycle 240: the SDK-dump echo and the `reasoning`-alias echo reach the template exactly like the minimal echo.
    private static let minimalEcho = #"{"role":"assistant","content":"a","reasoning_content":"because"}"#
    private static let sdkDumpEcho =
        #"{"role":"assistant","content":"a","reasoning_content":"because","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":null}"#
    private static let aliasEcho = #"{"role":"assistant","content":"a","reasoning":"because"}"#

    private func decodedHistory(echo: String) throws -> [OpenAIChatMessage] {
        let body = #"{"model":"m","messages":[{"role":"user","content":"q"},\#(echo)]}"#
        return try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)).messages
    }

    func testSDKDumpAndAliasEchoesGiveTheSameTemplateMessagesAsMinimalEcho() throws {
        let minimal = try renderHistory(decodedHistory(echo: Self.minimalEcho))
        for echo in [Self.sdkDumpEcho, Self.aliasEcho] {
            let box = try renderHistory(decodedHistory(echo: echo))
            XCTAssertEqual(box.messages.count, minimal.messages.count)
            for (lhs, rhs) in zip(box.messages, minimal.messages) {
                XCTAssertEqual(Set(lhs.keys), Set(rhs.keys), "key set differs for \(echo)")
                XCTAssertEqual(lhs["role"] as? String, rhs["role"] as? String)
                XCTAssertEqual(lhs["content"] as? String, rhs["content"] as? String)
                XCTAssertEqual(lhs["reasoning_content"] as? String, rhs["reasoning_content"] as? String)
            }
            XCTAssertEqual(box.messages[1]["reasoning_content"] as? String, "because")
            XCTAssertNil(box.messages[1]["reasoning"], "the alias must not leak to the template")
            XCTAssertEqual(box.rendered, minimal.rendered)
        }
        XCTAssertTrue(
            try XCTUnwrap(minimal.rendered).contains("<think>\nbecause\n</think>\n\na<|im_end|>"))
    }

    // MARK: end to end (env-gated tokenizer)

    private func loadQwen3Tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        guard let dirPath = ProcessInfo.processInfo.environment["FASTMLX_QWEN3_TOKENIZER_DIR"],
            !dirPath.isEmpty
        else {
            throw XCTSkip("FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping reasoning-history render proof")
        }
        let directoryURL = URL(fileURLWithPath: dirPath, isDirectory: true)
        for requiredFile in ["tokenizer_config.json", "tokenizer.json"] {
            guard FileManager.default.fileExists(
                atPath: directoryURL.appendingPathComponent(requiredFile).path)
            else { throw XCTSkip("Qwen3 tokenizer file '\(requiredFile)' is missing; skipping") }
        }
        do {
            let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directoryURL)
            return #adaptHuggingFaceTokenizer(upstream)
        } catch {
            throw XCTSkip("Qwen3 tokenizer not available: \(error)")
        }
    }

    // T6: the real template renders reasoning on the assistant turns after the last user query.
    func testRealTemplateRendersEchoedReasoning() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        let body = """
            {"model":"qwen3","messages":[
              {"role":"user","content":"go"},
              {"role":"assistant","content":null,"reasoning_content":"need the tool","tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{}"}}]},
              {"role":"tool","tool_call_id":"c1","content":"{}"}
            ],"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{}}}}]}
            """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        let ids = try codec.render(
            messages: request.messages, tools: request.tools, enableThinking: nil,
            reasoningEffort: nil, addGenerationPrompt: nil)
        let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        XCTAssertTrue(text.contains("<think>\nneed the tool\n</think>"), "reasoning not rendered:\n\(text)")

        let withoutReasoning = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(body.replacingOccurrences(of: #""reasoning_content":"need the tool","#, with: "").utf8))
        let plainIds = try codec.render(
            messages: withoutReasoning.messages, tools: withoutReasoning.tools, enableThinking: nil,
            reasoningEffort: nil, addGenerationPrompt: nil)
        XCTAssertFalse(
            tokenizer.decode(tokenIds: plainIds, skipSpecialTokens: false).contains("need the tool"))
    }

    // Cycle 240: the minimal, SDK-dump and `reasoning`-alias echoes render byte-identical prompts.
    func testRealTemplateRendersSDKDumpAndAliasEchoesIdentically() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        func prompt(_ echo: String) throws -> [Int] {
            // The assistant turn follows the last user query, so the real template renders its reasoning.
            try codec.render(
                messages: decodedHistory(echo: echo), tools: [], enableThinking: nil, reasoningEffort: nil,
                addGenerationPrompt: nil)
        }
        let minimal = try prompt(Self.minimalEcho)
        XCTAssertTrue(
            tokenizer.decode(tokenIds: minimal, skipSpecialTokens: false).contains("<think>\nbecause\n</think>"))
        XCTAssertEqual(try prompt(Self.sdkDumpEcho), minimal)
        XCTAssertEqual(try prompt(Self.aliasEcho), minimal)
    }
}
