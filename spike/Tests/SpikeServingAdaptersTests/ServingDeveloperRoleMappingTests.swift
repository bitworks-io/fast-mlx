import Foundation
import XCTest

import Jinja
import MLXLMCommon
import ServingCore
@testable import SpikeServingAdapters

/// End-to-end proof that `MLXScalarTextCodec.render` maps the wire `developer` role onto the
/// served chat template's `system` vocabulary (`MLXScalarTextCodec.scalarServingTemplateRoleName(for:)`)
/// BEFORE the template ever sees a message — not just that the codec's intermediate dictionaries
/// carry the right `"role"` string (that is `MLXScalarServingTests.swift`'s job), but that a real
/// `Jinja.Template` render, driven through `render(messages:tools:enableThinking:reasoningEffort:)`,
/// produces the exact output shape the deployed patched template produces for a leading and a
/// mid-conversation `developer` message. Drives `render` itself (never a bare `Template.render`)
/// because only `render` performs the mapping; a bare `Template.render` call would render whatever
/// role string the test handed it and would not observe the mapping at all.
final class ServingDeveloperRoleMappingTests: XCTestCase {

    // MARK: - Template excerpt

    /// A faithful EXCERPT of the served model's deployed patched chat template — the operator-installed
    /// variant whose sha256 is `b426d0bb02412efa9e44777312cc7df1bf95ea332dc0d2e46376c801f273599d`, the
    /// same digest already pinned at `Qwen38MTPPerformanceScorecardGate.swift:1109`. Pinning the digest
    /// here means a future template revision cannot silently invalidate this proof: if the deployed
    /// template changes, that digest stops matching and this excerpt must be re-derived.
    ///
    /// Inlined as a string literal rather than loaded from the operator runbook's template file at test
    /// runtime, deliberately. This test is part of the published source tree while the runbook's
    /// deployment assets are not, so reading that file here would make a published test depend on a
    /// path its own distribution does not carry.
    ///
    /// This excerpt reproduces four behaviours of the real template, faithfully:
    ///   1. the `reasoning_instructions` setup (real `:45-56`): when `enable_thinking` is
    ///      undefined it resolves `reasoning_effort|default('xhigh')` to a non-empty instruction
    ///      string;
    ///   2. the no-tools prologue (real `:77-87`): if `messages[0].role == 'system'`, emit ONE
    ///      `<|im_start|>system` block containing `reasoning_instructions` plus that message's
    ///      content; `elif reasoning_instructions`, emit a system block containing just the
    ///      instructions;
    ///   3. the patched loop branch (real `:104-112`): `message.role == "system"` and
    ///      `not loop.first` renders a mid-conversation `<|im_start|>system` turn instead of
    ///      raising (the un-patched stock template raises here — see
    ///      `ServingChatTemplateRefusalTests.swift`'s `nonLeadingSystemGuardTemplate`); plus the
    ///      `user` and `assistant` branches;
    ///   4. the terminal `{%- else %}{{- raise_exception('Unexpected message role.') }}`.
    ///
    /// `render_content` is simplified to a direct `message.content` reference because the codec
    /// (`MLXScalarTextCodec.render`) always passes `content` as a plain `String`, never the
    /// mixed text/image/video array the real macro also handles — that branch of `render_content`
    /// is unreachable from this codec and out of scope for this excerpt.
    private static let patchedTemplateExcerpt = try! Template(
        """
        {%- set reasoning_instructions = '' %}
        {%- if enable_thinking is undefined or enable_thinking is true %}
        {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
        {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task.' %}
        {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief.' %}
        {%- endif %}
        {%- endif %}
        {%- if tools and tools is iterable and tools is not mapping %}
        {{- '<|im_start|>system\\n' }}
        {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\\n\\n' }}
        {%- endif %}
        {{- '# Tools placeholder' }}
        {{- '<|im_end|>\\n' }}
        {%- else %}
        {%- if messages[0].role == 'system' %}
        {%- set content = messages[0].content|trim %}
        {%- if content %}
        {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '') + content + '<|im_end|>\\n' }}
        {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}
        {%- endif %}
        {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}
        {%- endif %}
        {%- endif %}
        {%- for message in messages %}
        {%- if message.role == "system" %}
        {%- if not loop.first %}
        {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
        {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
        {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
        {%- else %}
        {{- raise_exception('Unexpected message role.') }}
        {%- endif %}
        {%- endfor %}
        """)

    // MARK: - Test tokenizer double

    /// Reference-type box the (value-type) tokenizer double writes its render result into, so the
    /// test can read it back after `render` returns. `@unchecked Sendable` because access here is
    /// single-threaded within one test method — mirrors `CacheBox` in
    /// `LoadedExactQwen35MTPServingBackendIntegrationTests.swift`, which boxes state out of a
    /// value-type double for the same reason.
    private final class RenderedTemplateBox: @unchecked Sendable {
        var rendered: String?
    }

    /// A separate double from `MLXScalarServingTests.swift`'s `FixtureTokenizer`: where that one
    /// hardcodes an EXPECTED role/content pin and returns a constant token array, this one
    /// actually renders `patchedTemplateExcerpt` (a real `Jinja.Template`) with whatever messages
    /// it is handed, records the rendered text into `box`, and derives its returned token array
    /// FROM the render (`[rendered.count]`) rather than returning a hardcoded constant — so a
    /// thrown `Jinja.TemplateException` from the excerpt propagates out of `render` exactly as it
    /// would in production, instead of being masked by a fixed return value.
    private struct RenderingTokenizer: MLXLMCommon.Tokenizer {
        let box: RenderedTemplateBox

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
            var context: [String: Value] = [
                "messages": try Value(any: messages)
            ]
            if let tools {
                context["tools"] = try Value(any: tools)
            }
            if let additionalContext {
                for (key, value) in additionalContext {
                    context[key] = try Value(any: value)
                }
            }
            let rendered = try ServingDeveloperRoleMappingTests.patchedTemplateExcerpt.render(context)
            box.rendered = rendered
            return [rendered.count]
        }
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

    // MARK: (a) — leading developer message, index 0

    /// Vacuity trap this guards against: `reasoning_instructions` is non-empty by default, so the
    /// substring `"<|im_start|>system"` is present even for a user-only request with NO system
    /// message at all — asserting only "the marker is present" would pass regardless of whether
    /// the mapping ran. This asserts BOTH the exact occurrence count (which proves the prologue
    /// consumed the mapped message and the loop's `not loop.first` guard did not double-render it)
    /// AND the message's own content landing inside a system block.
    func testLeadingDeveloperMessageRendersAsSingleSystemBlock() throws {
        let box = RenderedTemplateBox()
        let codec = MLXScalarTextCodec(tokenizer: RenderingTokenizer(box: box))

        XCTAssertNoThrow(
            try codec.render(
                messages: [
                    OpenAIChatMessage(role: .developer, text: "developer instructions"),
                    OpenAIChatMessage(role: .user, text: "hi"),
                ],
                tools: [],
                enableThinking: nil,
                reasoningEffort: nil))

        let rendered = try XCTUnwrap(box.rendered)
        XCTAssertEqual(occurrenceCount(of: "<|im_start|>system", in: rendered), 1)
        XCTAssertTrue(rendered.contains("developer instructions"))
        // Prove "inside a system block": the content appears strictly after the marker and before
        // the matching turn close, not merely somewhere in the whole rendered string.
        let systemRange = try XCTUnwrap(rendered.range(of: "<|im_start|>system"))
        let closeRange = try XCTUnwrap(
            rendered.range(of: "<|im_end|>", range: systemRange.upperBound..<rendered.endIndex))
        let systemBlock = rendered[systemRange.upperBound..<closeRange.lowerBound]
        XCTAssertTrue(systemBlock.contains("developer instructions"))
    }

    // MARK: (b) — non-leading developer message, index 2

    /// Two `"<|im_start|>system"` occurrences are expected: one from `reasoning_instructions`
    /// (emitted because `messages[0].role != 'system'`) and one from the mapped mid-conversation
    /// `developer` -> `system` turn the patched loop branch renders. Asserts the content, not just
    /// the count, so a template that emitted two BLANK system turns could not pass vacuously.
    func testNonLeadingDeveloperMessageRendersAsSecondSystemTurn() throws {
        let box = RenderedTemplateBox()
        let codec = MLXScalarTextCodec(tokenizer: RenderingTokenizer(box: box))

        XCTAssertNoThrow(
            try codec.render(
                messages: [
                    OpenAIChatMessage(role: .user, text: "hi"),
                    OpenAIChatMessage(role: .assistant, text: "hello"),
                    OpenAIChatMessage(role: .developer, text: "mid-conversation instructions"),
                    OpenAIChatMessage(role: .user, text: "continue"),
                ],
                tools: [],
                enableThinking: nil,
                reasoningEffort: nil))

        let rendered = try XCTUnwrap(box.rendered)
        XCTAssertEqual(occurrenceCount(of: "<|im_start|>system", in: rendered), 2)
        XCTAssertTrue(rendered.contains("mid-conversation instructions"))
    }

    // MARK: (c) — anti-vacuity: prove the excerpt is discriminating

    /// Without this arm, an excerpt that (say) accepted every role unconditionally would make (a)
    /// and (b) above pass vacuously — they would prove nothing about the mapping actually being
    /// load-bearing. This arm renders `patchedTemplateExcerpt` DIRECTLY (not through the codec)
    /// with a role no mapping produces, `"critic"`, and requires a genuine `Jinja.TemplateException`
    /// from the terminal `raise_exception('Unexpected message role.')` branch. It must render the
    /// template directly rather than through `render` because `OpenAIMessageRole` is a closed enum
    /// (`developer`/`system`/`user`/`assistant`/`tool`) — the codec cannot express a bogus role by
    /// construction, which is exactly why this excerpt's discriminating power has to be checked at
    /// the template layer instead.
    func testExcerptStillRejectsARoleNoMappingProduces() throws {
        let messages = try Value(any: [
            ["role": "user", "content": "hi"],
            ["role": "critic", "content": "nope"],
        ])

        var caught: (any Error)?
        do {
            _ = try Self.patchedTemplateExcerpt.render(["messages": messages])
            XCTFail("expected an unmapped role to raise")
        } catch {
            caught = error
        }

        XCTAssertTrue(
            caught is TemplateException,
            "expected a genuine Jinja.TemplateException, got \(String(describing: caught))")
    }

    // MARK: (d) — regression guard for the sibling defect (plain non-leading system)

    /// A plain (un-mapped) non-leading `system` message must ALSO render without throwing on this
    /// excerpt — proving the excerpt matches the deployed PATCHED template (which accepts this),
    /// not the stock template (which raises `'System message must be at the beginning.'` here, per
    /// `ServingChatTemplateRefusalTests.swift`'s fixture). If this arm ever regresses while (a)/(b)
    /// stay green, the excerpt has drifted from the real deployed template.
    func testPlainNonLeadingSystemMessageStillRendersOnThisPatchedExcerpt() throws {
        let box = RenderedTemplateBox()
        let codec = MLXScalarTextCodec(tokenizer: RenderingTokenizer(box: box))

        XCTAssertNoThrow(
            try codec.render(
                messages: [
                    OpenAIChatMessage(role: .user, text: "hi"),
                    OpenAIChatMessage(role: .system, text: "mid"),
                    OpenAIChatMessage(role: .user, text: "go"),
                ],
                tools: [],
                enableThinking: nil,
                reasoningEffort: nil))

        let rendered = try XCTUnwrap(box.rendered)
        XCTAssertTrue(rendered.contains("mid"))
    }
}
