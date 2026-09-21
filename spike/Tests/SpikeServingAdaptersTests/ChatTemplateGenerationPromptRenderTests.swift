import Foundation
import XCTest

import MLXHuggingFace
import MLXLMCommon
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// Empirically proves `MLXScalarTextCodec.render`'s `addGenerationPrompt` plumbing against a real
/// Qwen3 tokenizer/chat-template on CPU (no GPU, no model weights — tokenizer files only) — the
/// render-level counterpart to `OpenAIChatCompletionsTests.swift`'s decode-level coverage of
/// `OpenAIChatCompletionRequest.addGenerationPrompt`. Copies `Qwen3TemplateRenderTests.swift`'s
/// tokenizer-loading and `FASTMLX_QWEN3_TOKENIZER_DIR` env-var idiom exactly.
///
/// Every assertion here is RELATIONAL (token-array length/prefix/equality), never a hardcoded token
/// ID — the concrete vocabulary is real but incidental to what this proves: the SHAPE of the effect
/// `addGenerationPrompt` has on the rendered prompt, which must hold for any served chat template.
final class ChatTemplateGenerationPromptRenderTests: XCTestCase {
    private func loadQwen3Tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        let environment = ProcessInfo.processInfo.environment
        guard let dirPath = environment["FASTMLX_QWEN3_TOKENIZER_DIR"], !dirPath.isEmpty else {
            throw XCTSkip(
                "FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping addGenerationPrompt render proof")
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
            guard fileManager.fileExists(atPath: directoryURL.appendingPathComponent(requiredFile).path)
            else {
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

    private func makeMessages() -> [OpenAIChatMessage] {
        [
            OpenAIChatMessage(role: .system, text: "You are a concise assistant."),
            OpenAIChatMessage(role: .user, text: "What is the capital of France?"),
        ]
    }

    /// Proves all four behavior requirements from the design in one pass so the relational
    /// assertions read as one coherent story: `false` and `true` diverge, `false` is a strict
    /// prefix-truncation of `true` (the generation prompt is APPENDED, not inserted mid-stream), and
    /// `nil` is byte-identical to explicit `true` — the no-regression pin for "nil means true" (see
    /// `MLXScalarTextCodec.render`'s doc comment on why that pin lives at the `additionalContext`
    /// boundary rather than a swift-transformers `addGenerationPrompt:` parameter default).
    func testAddGenerationPromptShapesTheRenderedPrompt() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        let messages = makeMessages()

        let withFalse = try codec.render(
            messages: messages, tools: [], enableThinking: nil, reasoningEffort: nil,
            addGenerationPrompt: false)
        let withTrue = try codec.render(
            messages: messages, tools: [], enableThinking: nil, reasoningEffort: nil,
            addGenerationPrompt: true)
        let withNil = try codec.render(
            messages: messages, tools: [], enableThinking: nil, reasoningEffort: nil,
            addGenerationPrompt: nil)

        // 1. false and true produce DIFFERENT token arrays.
        XCTAssertNotEqual(
            withFalse, withTrue,
            "addGenerationPrompt: false must render a different prompt than addGenerationPrompt: true")

        // 2. false is SHORTER than true — the generation prompt adds tokens, it never removes any.
        XCTAssertLessThan(
            withFalse.count, withTrue.count,
            "addGenerationPrompt: false must render fewer tokens than addGenerationPrompt: true")

        // 3. true STARTS WITH false — the generation prompt is APPENDED at the end, not spliced into
        // the middle of the rendered messages.
        XCTAssertEqual(
            Array(withTrue.prefix(withFalse.count)), withFalse,
            "addGenerationPrompt: true must start with the exact addGenerationPrompt: false prompt")

        // 4. nil renders byte-identically to explicit true — the no-regression pin: every existing
        // caller that never sends `add_generation_prompt` keeps the exact prompt it rendered before
        // this parameter existed.
        XCTAssertEqual(
            withNil, withTrue,
            "addGenerationPrompt: nil must render identically to addGenerationPrompt: true")
    }
}
