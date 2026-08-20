import Foundation
import XCTest

import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// Empirically proves the multi-turn tool re-render mapping in
/// `MLXScalarTextCodec.render(messages:tools:enableThinking:)` against a real Qwen3
/// tokenizer/chat-template on CPU (no GPU, no model weights — tokenizer files only).
final class Qwen3TemplateRenderTests: XCTestCase {
    private static let concierteRequestJSON = """
        {"model":"qwen3","messages":[
          {"role":"system","content":"You are a concierge."},
          {"role":"user","content":"do you have the RTX 6000 Ada in stock?"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"call_1_0","type":"function","function":{"name":"get_product","arguments":"{\\"query\\":\\"RTX 6000 Ada\\"}"}}]},
          {"role":"tool","tool_call_id":"call_1_0","content":"{\\"products\\":[]}"}
        ],"tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"max_tokens":384}
        """

    private func loadQwen3Tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        let environment = ProcessInfo.processInfo.environment
        guard let dirPath = environment["FASTMLX_QWEN3_TOKENIZER_DIR"], !dirPath.isEmpty else {
            throw XCTSkip("FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping Qwen3 template render proof")
        }

        let directoryURL = URL(fileURLWithPath: dirPath, isDirectory: true)
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directoryURL.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw XCTSkip("Qwen3 tokenizer directory is not available: \(directoryURL.path)")
        }
        for requiredFile in ["tokenizer_config.json", "tokenizer.json"] {
            guard fileManager.fileExists(atPath: directoryURL.appendingPathComponent(requiredFile).path) else {
                throw XCTSkip("Qwen3 tokenizer file '\(requiredFile)' is missing; skipping")
            }
        }

        do {
            let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directoryURL)
            return #adaptHuggingFaceTokenizer(upstream)
        } catch {
            throw XCTSkip("Qwen3 tokenizer not available: \(error)")
        }
    }

    private func makeConciergeRequest() throws -> OpenAIChatCompletionRequest {
        let data = Self.concierteRequestJSON.data(using: .utf8)!
        return try OpenAIChatCompletionRequest.decodeStrict(from: data)
    }

    func testMultiTurnToolRerenderMatchesQwen3ChatTemplate() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        let request = try makeConciergeRequest()

        // enableThinking: false — the codec must inject the forced-empty think block on
        // the generation prompt.
        let disabledThinkingIDs = try codec.render(
            messages: request.messages,
            tools: request.tools,
            enableThinking: false,
            reasoningEffort: nil)
        let disabledThinkingPrompt = tokenizer.decode(
            tokenIds: disabledThinkingIDs, skipSpecialTokens: false)

        // 1. The tools block rendered, carrying the declared function name.
        XCTAssertTrue(
            disabledThinkingPrompt.contains("<tools>"),
            "Expected a <tools> block in:\n\(disabledThinkingPrompt)")
        XCTAssertTrue(
            disabledThinkingPrompt.contains("get_product"),
            "Expected the tool name 'get_product' in:\n\(disabledThinkingPrompt)")

        // 2. The assistant history tool call rendered with real (not double-encoded) JSON
        // arguments.
        XCTAssertTrue(
            disabledThinkingPrompt.contains("<tool_call>"),
            "Expected a <tool_call> block in:\n\(disabledThinkingPrompt)")
        XCTAssertTrue(
            disabledThinkingPrompt.contains("\"name\": \"get_product\""),
            "Expected the rendered tool call name field in:\n\(disabledThinkingPrompt)")
        XCTAssertTrue(
            disabledThinkingPrompt.contains("RTX 6000 Ada"),
            "Expected the tool call argument value in:\n\(disabledThinkingPrompt)")
        XCTAssertFalse(
            disabledThinkingPrompt.contains("\\\"query\\\""),
            "Tool call arguments must render as real JSON, not a backslash-escaped string, in:\n\(disabledThinkingPrompt)")

        // 3. The tool result rendered via the tool_response wrapper.
        XCTAssertTrue(
            disabledThinkingPrompt.contains("<tool_response>"),
            "Expected a <tool_response> block in:\n\(disabledThinkingPrompt)")
        XCTAssertTrue(
            disabledThinkingPrompt.contains("\"products\""),
            "Expected the tool result payload in:\n\(disabledThinkingPrompt)")

        // 4. The prompt opens a fresh assistant turn for generation.
        XCTAssertTrue(
            disabledThinkingPrompt.contains("<|im_start|>assistant"),
            "Expected the assistant generation turn to be opened in:\n\(disabledThinkingPrompt)")

        // 5. enableThinking: false forces an empty think block immediately before generation.
        let emptyThinkPrefill = "<think>\n\n</think>"
        XCTAssertTrue(
            disabledThinkingPrompt.contains(emptyThinkPrefill),
            "Expected the forced-empty think prefill '\(emptyThinkPrefill)' in:\n\(disabledThinkingPrompt)")

        // Sanity: the empty think prefill must appear after the final assistant generation
        // turn was opened (it belongs to the generation prompt, not an earlier historical turn).
        guard
            let finalAssistantRange = disabledThinkingPrompt.range(
                of: "<|im_start|>assistant", options: .backwards),
            let thinkRange = disabledThinkingPrompt.range(
                of: emptyThinkPrefill, options: .backwards)
        else {
            XCTFail("Expected both markers to be present in:\n\(disabledThinkingPrompt)")
            return
        }
        XCTAssertTrue(
            thinkRange.lowerBound >= finalAssistantRange.lowerBound,
            "Expected the forced-empty think prefill to follow the generation-prompt assistant turn in:\n\(disabledThinkingPrompt)")

        // Second block: with enableThinking left enabled, the same forced-empty think
        // prefill must NOT be injected onto the generation prompt.
        let enabledThinkingIDs = try codec.render(
            messages: request.messages,
            tools: request.tools,
            enableThinking: true,
            reasoningEffort: nil)
        let enabledThinkingPrompt = tokenizer.decode(
            tokenIds: enabledThinkingIDs, skipSpecialTokens: false)

        let generationSuffix: String
        if let range = enabledThinkingPrompt.range(
            of: "<|im_start|>assistant", options: .backwards)
        {
            generationSuffix = String(enabledThinkingPrompt[range.lowerBound...])
        } else {
            generationSuffix = enabledThinkingPrompt
        }
        XCTAssertFalse(
            generationSuffix.contains(emptyThinkPrefill),
            "Did not expect the forced-empty think prefill when enableThinking is true, in generation suffix:\n\(generationSuffix)")
    }
}
