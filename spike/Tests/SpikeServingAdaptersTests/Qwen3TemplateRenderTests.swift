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

    // MARK: - Tools x non-leading system message (the intersection of the two axes above)

    private static let toolPolicyMarkerContent =
        "TOOL_POLICY_MARKER_CONTENT_9e1b4d: only call get_product for a named product."

    private static let toolsWithNonLeadingSystemRequestJSON = """
        {"model":"qwen3","messages":[
          {"role":"user","content":"do you have the RTX 6000 Ada in stock?"},
          {"role":"assistant","content":"Let me check that for you."},
          {"role":"system","content":"\(toolPolicyMarkerContent)"},
          {"role":"user","content":"and what about the price?"}
        ],"tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"max_tokens":384}
        """

    /// Loads the REAL production override text straight off disk — `deploy/qwen3.8-27b/chat_template.jinja`
    /// — rather than a toy excerpt, so the derivation comment on the test below can cite exact line
    /// numbers in the file this test actually renders through.
    private func productionChatTemplateText() throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Qwen3TemplateRenderTests.swift
            .deletingLastPathComponent()  // SpikeServingAdaptersTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // spike
        let templateURL = repoRoot.appendingPathComponent("deploy/qwen3.8-27b/chat_template.jinja")
        // `deploy/` is an internal deployment asset and is deliberately not part of the public
        // distribution, while this test file IS published. Skip rather than fail there: without
        // this guard, setting the tokenizer variable on a source-only checkout would produce a
        // confusing read error for a file that checkout is not supposed to contain.
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw XCTSkip(
                "deployment chat template is not present in this checkout; skipping the "
                    + "tools-with-non-leading-system render proof")
        }
        return try String(contentsOf: templateURL, encoding: .utf8)
    }

    /// (e) Tools x non-leading system message — the exact INTERSECTION of the two axes each of the
    /// suites above cover individually: `testMultiTurnToolRerenderMatchesQwen3ChatTemplate` (tools +
    /// LEADING system only) and `testOverrideRendersNonLeadingSystemMessageInPlace` /
    /// `testNoOverrideStockTemplateRaisesOnNonLeadingSystemMessage` (non-leading system, NO tools).
    /// The production agent/tool-calling shape is exactly this combination, and it was untested
    /// (project lesson: "the intersection of two individually-covered axes is uncovered").
    ///
    /// Renders through the REAL production override — the full, on-disk
    /// `deploy/qwen3.8-27b/chat_template.jinja` text (loaded above, not a toy excerpt) — applied via
    /// `scalarServingTokenizerWithChatTemplateOverride` onto a real Qwen3 tokenizer/vocabulary built
    /// from a STOCK-shaped fixture directory (`writeStockShapedModelDirectory()`; its own on-disk
    /// `chat_template.jinja` is irrelevant here because the override function replaces the
    /// tokenizer_config's `chat_template` dictionary entry outright, never reading the on-disk
    /// `chat_template.jinja` file — see `scalarServingTokenizerWithChatTemplateOverride`'s doc
    /// comment). That production file's own PATCH comment (search "PATCH (bitworks/fast-mlx" in the
    /// template) is what renders a mid-conversation system turn in place instead of raising, exactly
    /// as this test proves under the tools-attached shape.
    ///
    /// `<|im_start|>system` occurrence-count derivation (read directly from
    /// `deploy/qwen3.8-27b/chat_template.jinja`, not guessed): with `tools` present, the template's
    /// `{%- if tools and tools is iterable and tools is not mapping %}` branch (line 57) always
    /// emits exactly ONE `<|im_start|>system` opening the tools/leading-system block, regardless of
    /// `messages[0].role` (line 69 only controls whether the ORIGINAL leading system message's OWN
    /// content is appended inside that SAME already-open block — here `messages[0]` is `user`, so it
    /// is not, but the opening tag at line 58 has already been emitted unconditionally). The main
    /// per-message loop (line 102) then emits a SECOND `<|im_start|>system` for the mid-conversation
    /// `system` message at index 2, because it is not `loop.first` (line 105's
    /// `{%- if not loop.first %}` guard — the PATCH branch that renders in place instead of raising).
    /// No other branch in the template can emit `<|im_start|>system` for this message shape (no
    /// bare `reasoning_instructions` preamble is possible once the `tools` branch is taken; see line
    /// 76's `{%- else %}`). Expected count: 2.
    func testOverrideRendersToolsWithNonLeadingSystemMessage() async throws {
        let directory = try await writeStockShapedModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let productionTemplateText = try productionChatTemplateText()
        let overriddenTokenizer = try await scalarServingTokenizerWithChatTemplateOverride(
            modelDirectory: directory, overrideTemplateText: productionTemplateText)
        let codec = MLXScalarTextCodec(tokenizer: overriddenTokenizer)

        let data = Self.toolsWithNonLeadingSystemRequestJSON.data(using: .utf8)!
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: data)

        // Does not throw — the stock (pre-patch) guard would raise here; see
        // `testNoOverrideStockTemplateRaisesOnNonLeadingSystemMessage` for that refusal proved
        // directly against a toy excerpt of the same guard.
        let tokens = try codec.render(
            messages: request.messages, tools: request.tools, enableThinking: false,
            reasoningEffort: nil)
        let rendered = overriddenTokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        // The tool schema block is present exactly once.
        XCTAssertEqual(
            occurrenceCount(of: "<tools>", in: rendered), 1,
            "expected exactly one <tools> block, in:\n\(rendered)")
        XCTAssertTrue(
            rendered.contains("get_product"),
            "expected the tool name 'get_product' in:\n\(rendered)")

        // Exactly two <|im_start|>system turns — see the derivation comment above this test.
        XCTAssertEqual(
            occurrenceCount(of: "<|im_start|>system", in: rendered), 2,
            "expected exactly two system turns (tools block + mid-conversation system), in:\n\(rendered)"
        )

        // The mid-conversation system content was not dropped.
        XCTAssertTrue(
            rendered.contains(Self.toolPolicyMarkerContent),
            "expected the tool policy content to survive rendering, in:\n\(rendered)")
    }

    // MARK: - Hermetic proof: NO environment variable required (project lesson: a suite gated
    // entirely on `FASTMLX_QWEN3_TOKENIZER_DIR` stays green under `swift test` in CI, which never
    // sets that variable, even if a refactor deletes the override mechanism entirely). These two
    // arms build a minimal synthetic tokenizer from string/dictionary literals in a temp directory,
    // so they execute unconditionally, driving the SAME production seam as the arms above
    // (`scalarServingTokenizerWithChatTemplateOverride` -> `MLXScalarTextCodec.render` -> a genuine
    // Jinja render, then a genuine encode/decode round trip) without re-implementing any of it.

    /// Builds a `stockShapedTemplateText`-based fixture directory like
    /// `writeStockShapedModelDirectory()` above, except `tokenizer.json`/`tokenizer_config.json` are
    /// constructed from literals here instead of copied from `FASTMLX_QWEN3_TOKENIZER_DIR`.
    /// `PreTrainedTokenizer` (vendored `swift-transformers`) requires a `tokenizer_class` registered
    /// in `TokenizerModel.knownTokenizers` — `Qwen2Tokenizer` resolves to its `BPETokenizer` model —
    /// and that model requires a `model.vocab` plus a `model.merges` array (empty is valid: with no
    /// merge ranks, `BPETokenizer.bpe(token:)` performs no merges at all, so the codec's rendered
    /// text tokenizes one Unicode scalar at a time). With no `preTokenizer`/`normalizer`/`decoder`
    /// configured (all three become `nil` when their config key is absent — see
    /// `PreTokenizerFactory`/`NormalizerFactory`/`DecoderFactory`.`fromConfig`), pre-tokenization,
    /// normalization, and decoding are all the identity function, so a vocab covering every
    /// printable ASCII character plus `\n`, alongside `<|im_start|>`/`<|im_end|>` as `special` added
    /// tokens (matched whole, ahead of the character-level fallback, by `PreTrainedTokenizer`'s own
    /// `addedTokensRegex`), is sufficient to round-trip any of this suite's fixture message text
    /// losslessly through `encode` then `decode`.
    private func writeHermeticStockShapedModelDirectory(
        chatTemplateText: String = Qwen3TemplateRenderTests.stockShapedTemplateText
    ) throws -> URL {
        var vocab: [String: Int] = [:]
        var nextID = 2
        for scalarValue in UInt32(0x20)...UInt32(0x7E) {
            vocab[String(UnicodeScalar(scalarValue)!)] = nextID
            nextID += 1
        }
        vocab["\n"] = nextID

        let tokenizerData: [String: Any] = [
            "added_tokens": [
                [
                    "id": 0, "content": "<|im_start|>", "special": true, "lstrip": false,
                    "rstrip": false,
                ],
                [
                    "id": 1, "content": "<|im_end|>", "special": true, "lstrip": false,
                    "rstrip": false,
                ],
            ],
            "model": [
                "type": "BPE",
                "vocab": vocab,
                "merges": [String](),
            ],
        ]
        let tokenizerConfig: [String: Any] = [
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "clean_up_tokenization_spaces": false,
        ]

        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory
            .appendingPathComponent(
                "chat-template-hermetic-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: tokenizerData)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: tokenizerConfig)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try Data(chatTemplateText.utf8).write(
            to: directory.appendingPathComponent("chat_template.jinja"))
        return directory
    }

    /// (hermetic-a) No override: mirrors `testNoOverrideStockTemplateRaisesOnNonLeadingSystemMessage`
    /// above exactly, against the synthetic fixture instead of a real checkpoint, so it runs with no
    /// environment variable set.
    func testHermeticNoOverrideStockTemplateRaisesOnNonLeadingSystemMessage() async throws {
        let directory = try writeHermeticStockShapedModelDirectory()
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

    /// (hermetic-b) With `--chat-template` override: mirrors
    /// `testOverrideRendersNonLeadingSystemMessageInPlace` above exactly, against the synthetic
    /// fixture, so it runs with no environment variable set. This is the decisive regression guard
    /// for the production mitigation: deleting
    /// `dictionary["chat_template"] = .init(overrideTemplateText)` in `MLXScalarServing.swift` makes
    /// this test fail (see the mutation check accompanying this change), whereas every
    /// env-var-gated test above it in this file would silently skip instead of catching that.
    func testHermeticOverrideRendersNonLeadingSystemMessageInPlace() async throws {
        let directory = try writeHermeticStockShapedModelDirectory()
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
            "expected the system message's own content strictly inside the system block, in:\n\(rendered)"
        )

        // The system turn must render IN PLACE, not hoisted to the front. Both renderings produce
        // exactly one system block containing the marker, so the occurrence count above cannot tell
        // them apart -- only position can. This matters beyond tidiness: the in-place form keeps the
        // prompt prefix stable for KV-cache reuse, which is why the patch merges only the LEADING
        // system block and emits later system turns where they occur.
        let openingRange = try XCTUnwrap(rendered.range(of: "opening user turn"))
        let closingRange = try XCTUnwrap(rendered.range(of: "closing user turn"))
        XCTAssertLessThan(
            openingRange.upperBound, systemRange.lowerBound,
            "the first user turn must precede the system turn, in:\n\(rendered)")
        XCTAssertLessThan(
            closeRange.upperBound, closingRange.lowerBound,
            "the system turn must precede the closing user turn, in:\n\(rendered)")

        XCTAssertFalse(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(
                modelDirectory: directory, overrideText: Self.patchedShapedTemplateText))
        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory),
            "the on-disk chat_template.jinja must be unaffected by the override")
    }
}
