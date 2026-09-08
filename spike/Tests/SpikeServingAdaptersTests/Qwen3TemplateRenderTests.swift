import Foundation
import XCTest

import HuggingFace
import Jinja
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

    // MARK: - `--chat-template` override: decisive end-to-end proof through the real production
    // rendering seam (`scalarServingTokenizerWithChatTemplateOverride` -> `MLXScalarTextCodec.render`
    // -> a genuine Jinja render), on a REAL Qwen3 tokenizer/vocabulary (gated the same way as the
    // suite above — `FASTMLX_QWEN3_TOKENIZER_DIR`).

    /// A faithful STOCK-shaped excerpt: a non-leading `system` message raises, matching
    /// `ServingChatTemplateRefusalTests.swift`'s `nonLeadingSystemGuardTemplate` anchor text
    /// exactly (so `scalarServingChatTemplateRefusesNonLeadingSystemMessage`'s own anchor-string
    /// probe attests it). Also raises on any role it does not recognize, so arm (d) below can
    /// prove the override does not disable that OTHER refusal.
    private static let stockShapedTemplateText = """
        {%- for message in messages %}
        {%- if message.role == "system" %}
        {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
        {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}
        {%- elif message.role == "user" %}
        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
        {%- else %}
        {{- raise_exception('Unexpected message role.') }}
        {%- endif %}
        {%- endfor %}
        """

    /// A faithful PATCHED-shaped excerpt: renders a non-leading `system` turn in place instead of
    /// raising, mirroring `ServingDeveloperRoleMappingTests.swift`'s `patchedTemplateExcerpt`'s
    /// loop branch. Still raises on an unrecognized role — the override must not disable that
    /// OTHER refusal.
    private static let patchedShapedTemplateText = """
        {%- for message in messages %}
        {%- if message.role == "system" %}
        {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}
        {%- elif message.role == "user" %}
        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
        {%- else %}
        {{- raise_exception('Unexpected message role.') }}
        {%- endif %}
        {%- endfor %}
        """

    private static let systemTurnMarkerContent = "SYSTEM_OVERRIDE_MARKER_CONTENT_7f3c2a"

    /// Copies a REAL tokenizer's `tokenizer.json`/`tokenizer_config.json` into a fresh temp
    /// directory and writes `chat_template.jinja` = `stockShapedTemplateText` alongside them — the
    /// STOCK-shaped fixture directory both arms below load from. Skips (never fails) when the
    /// gating env var is unset/unavailable, exactly like `loadQwen3Tokenizer()` above.
    private func writeStockShapedModelDirectory() async throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        guard let dirPath = environment["FASTMLX_QWEN3_TOKENIZER_DIR"], !dirPath.isEmpty else {
            throw XCTSkip(
                "FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping --chat-template override proof")
        }
        let sourceDirectory = URL(fileURLWithPath: dirPath, isDirectory: true)
        let fileManager = FileManager.default
        for requiredFile in ["tokenizer_config.json", "tokenizer.json"] {
            guard fileManager.fileExists(atPath: sourceDirectory.appendingPathComponent(requiredFile).path)
            else {
                throw XCTSkip("Qwen3 tokenizer file '\(requiredFile)' is missing; skipping")
            }
        }

        let directory = fileManager.temporaryDirectory
            .appendingPathComponent("chat-template-override-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        for file in ["tokenizer_config.json", "tokenizer.json"] {
            try fileManager.copyItem(
                at: sourceDirectory.appendingPathComponent(file),
                to: directory.appendingPathComponent(file))
        }
        try Data(Self.stockShapedTemplateText.utf8).write(
            to: directory.appendingPathComponent("chat_template.jinja"))
        return directory
    }

    private func nonLeadingSystemMessages() -> [OpenAIChatMessage] {
        [
            OpenAIChatMessage(role: .user, text: "opening user turn"),
            OpenAIChatMessage(role: .system, text: Self.systemTurnMarkerContent),
            OpenAIChatMessage(role: .user, text: "closing user turn"),
        ]
    }

    private func occurrenceCount(of substring: String, in text: String) -> Int {
        var count = 0
        var searchRange = text.startIndex..<text.endIndex
        while let range = text.range(of: substring, range: searchRange) {
            count += 1
            searchRange = range.upperBound..<text.endIndex
        }
        return count
    }

    /// (a) No override: the codec renders through the checkpoint's own STOCK-shaped
    /// `chat_template.jinja`, and a non-leading `system` message raises — translated to a client
    /// `OpenAIServingError`, matching `ServingChatTemplateRefusalTests.swift`'s existing proof of
    /// that translation. The probe agrees: `true` (still refuses).
    func testNoOverrideStockTemplateRaisesOnNonLeadingSystemMessage() async throws {
        let directory = try await writeStockShapedModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        let tokenizer = #adaptHuggingFaceTokenizer(upstream)
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)

        var caught: (any Error)?
        do {
            _ = try codec.render(
                messages: nonLeadingSystemMessages(), tools: [], enableThinking: nil,
                reasoningEffort: nil)
            XCTFail("expected the stock template's non-leading system guard to raise")
        } catch {
            caught = error
        }

        let servingError = try XCTUnwrap(caught as? OpenAIServingError)
        XCTAssertEqual(servingError.openAIError.code, "chat_template_rejected")
        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory))
    }

    /// (b) With `--chat-template` override: the SAME message list, through the SAME on-disk
    /// checkpoint, now renders — via `scalarServingTokenizerWithChatTemplateOverride` building a
    /// real replacement tokenizer from the patched excerpt text, then `MLXScalarTextCodec.render`
    /// completely UNCHANGED. Anti-vacuity: asserts the EXACT occurrence count of the system marker
    /// (not bare `.contains`) and the message's OWN content landing strictly inside that one system
    /// block — the trap this suite guards against is a template excerpt that emits
    /// `<|im_start|>system` unconditionally regardless of whether anything worked (see
    /// `ServingDeveloperRoleMappingTests.swift`'s identical warning); `patchedShapedTemplateText`
    /// above has no such unconditional prologue, and arm (c) below proves that directly. The probe
    /// agrees: `false` with the override text, `true` for the SAME directory read again with no
    /// override — proving the override changed what renders without mutating the on-disk file.
    func testOverrideRendersNonLeadingSystemMessageInPlace() async throws {
        let directory = try await writeStockShapedModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let overriddenTokenizer = try await scalarServingTokenizerWithChatTemplateOverride(
            modelDirectory: directory, overrideTemplateText: Self.patchedShapedTemplateText)
        let codec = MLXScalarTextCodec(tokenizer: overriddenTokenizer)

        let tokens = try codec.render(
            messages: nonLeadingSystemMessages(), tools: [], enableThinking: nil,
            reasoningEffort: nil)
        let rendered = overriddenTokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        XCTAssertEqual(
            occurrenceCount(of: "<|im_start|>system", in: rendered), 1,
            "expected exactly one system turn, in:\n\(rendered)")
        let systemRange = try XCTUnwrap(rendered.range(of: "<|im_start|>system"))
        let closeRange = try XCTUnwrap(
            rendered.range(of: "<|im_end|>", range: systemRange.upperBound..<rendered.endIndex))
        let systemBlock = rendered[systemRange.upperBound..<closeRange.lowerBound]
        XCTAssertTrue(
            systemBlock.contains(Self.systemTurnMarkerContent),
            "expected the system message's own content strictly inside the system block, in:\n\(rendered)")

        XCTAssertFalse(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(
                modelDirectory: directory, overrideText: Self.patchedShapedTemplateText))
        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory),
            "the on-disk chat_template.jinja must be unaffected by the override")
    }

    /// (c) Anti-vacuity: `patchedShapedTemplateText` does NOT emit `<|im_start|>system`
    /// unconditionally — a user-only request (no system message at all) renders zero occurrences.
    /// Without this arm, arm (b)'s occurrence-count assertion could pass vacuously against an
    /// excerpt that always emits a system preamble regardless of the input.
    func testOverrideDoesNotEmitSystemMarkerForAUserOnlyRequest() async throws {
        let directory = try await writeStockShapedModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let overriddenTokenizer = try await scalarServingTokenizerWithChatTemplateOverride(
            modelDirectory: directory, overrideTemplateText: Self.patchedShapedTemplateText)
        let codec = MLXScalarTextCodec(tokenizer: overriddenTokenizer)

        let tokens = try codec.render(
            messages: [OpenAIChatMessage(role: .user, text: "just a user turn")],
            tools: [], enableThinking: nil, reasoningEffort: nil)
        let rendered = overriddenTokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        XCTAssertEqual(occurrenceCount(of: "<|im_start|>system", in: rendered), 0)
    }

    /// (d) The override does NOT disable the template's OTHER refusals: an unrecognized role still
    /// raises a genuine `Jinja.TemplateException` under the override, exactly as it does on the
    /// stock template. Drives the overridden tokenizer's `applyChatTemplate` directly (not through
    /// the codec) because `OpenAIChatMessage.role` is a closed enum that cannot express a role no
    /// mapping produces — the same reason `ServingDeveloperRoleMappingTests.swift`'s equivalent arm
    /// does the same thing.
    func testOverrideStillRejectsARoleNoMappingProduces() async throws {
        let directory = try await writeStockShapedModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let overriddenTokenizer = try await scalarServingTokenizerWithChatTemplateOverride(
            modelDirectory: directory, overrideTemplateText: Self.patchedShapedTemplateText)

        var caught: (any Error)?
        do {
            _ = try overriddenTokenizer.applyChatTemplate(
                messages: [["role": "critic", "content": "nope"]],
                tools: nil, additionalContext: nil)
            XCTFail("expected an unmapped role to raise under the override too")
        } catch {
            caught = error
        }

        XCTAssertTrue(
            caught is TemplateException,
            "expected a genuine Jinja.TemplateException, got \(String(describing: caught))")
    }
}
