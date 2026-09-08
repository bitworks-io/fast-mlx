import Foundation
import XCTest

@testable import SpikeServingAdapters

/// `scalarServingChatTemplateAttestsThinkMarkers` is the artifact-derived probe that keys
/// `servingThinksByDefault`/`servingDisablesThinkingWhenToolsActive` (`StreamingReasoningPolicy.swift`)
/// to the LOADED checkpoint's own chat template rather than a compile-time assumption about the
/// family. It exists because of a live-measured defect this cycle: on the `.nativeHeterogeneous`
/// route, `qwen4_exp` (Flash Next, checkpoint `flashnext-oq4-mtp`, harness `c771346e`) was absent from
/// the attested family set, so the streaming path never separated reasoning at all — zero
/// `reasoning_content` deltas were observed, the answer arrived as `content` deltas, and the literal
/// `</think>` marker leaked into visible content once, even though non-streaming on the identical
/// prompt/budget split cleanly (`content` = `"7pm"`, `reasoning_content` = the full reasoning). Adding
/// the family string to the attested set alone would NOT be fail-closed for a checkpoint whose
/// template never emits the markers the splitter hardcodes — this probe is the artifact-derived half
/// of that conjunction.
///
/// Fixture note: the Flash Next checkpoint's `tokenizer_config.json` `chat_template` and its sibling
/// `chat_template.jinja` are byte-identical (8952 bytes) and both contain `<think>` x3 and `</think>`
/// x2 — the fixtures below exercise the resolution order (`chat_template.jinja` preferred, matching
/// swift-transformers' `Hub.swift`; falling back to `tokenizer_config.json` only when
/// `chat_template.jinja` is missing or unreadable) and the fail-closed behavior when neither
/// attests the markers or neither is readable.
final class FlashNextChatTemplateAttestationTests: XCTestCase {
    private func writeModelDirectory(
        tokenizerConfigJSON: String? = nil,
        chatTemplateJinja: String? = nil
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flashnext-template-probe-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let tokenizerConfigJSON {
            try Data(tokenizerConfigJSON.utf8).write(
                to: directory.appendingPathComponent("tokenizer_config.json"))
        }
        if let chatTemplateJinja {
            try Data(chatTemplateJinja.utf8).write(
                to: directory.appendingPathComponent("chat_template.jinja"))
        }
        return directory
    }

    private let attestingTemplateText =
        "{% if x %}<think>{% endif %}reasoning</think>answer<think>more</think>"

    private let nonAttestingTemplateText = "{% if x %}plain{% endif %}answer only, no markers here"

    /// Positive: `tokenizer_config.json`'s `chat_template` field contains both markers.
    func testAttestsTrueWhenTokenizerConfigChatTemplateContainsBothMarkers() throws {
        let json = #"{"chat_template": "\#(attestingTemplateText)"}"#
        let directory = try writeModelDirectory(tokenizerConfigJSON: json)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// Negative: a readable, well-formed template that never emits the markers must NOT attest.
    func testAttestsFalseWhenTokenizerConfigChatTemplateLacksMarkers() throws {
        let json = #"{"chat_template": "\#(nonAttestingTemplateText)"}"#
        let directory = try writeModelDirectory(tokenizerConfigJSON: json)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// Negative: neither `tokenizer_config.json` nor `chat_template.jinja` is present — fail-closed.
    func testAttestsFalseWhenNeitherFileIsPresent() throws {
        let directory = try writeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// Negative: an unreadable/absent model directory entirely — fail-closed, not a crash.
    func testAttestsFalseForAnUnreadableModelDirectory() {
        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("flashnext-template-probe-absent-\(UUID().uuidString)")

        XCTAssertFalse(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: absent))
    }

    /// `tokenizer_config.json` has no `chat_template` key and `chat_template.jinja` DOES attest —
    /// resolution succeeds from the preferred `chat_template.jinja` source regardless of what (or
    /// whether) `tokenizer_config.json` carries.
    func testResolvesFromChatTemplateJinjaWhenTokenizerConfigHasNoChatTemplateKey() throws {
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: #"{"tokenizer_class": "PreTrainedTokenizerFast"}"#,
            chatTemplateJinja: attestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// Fallback path: `chat_template.jinja` is absent, so resolution falls back to
    /// `tokenizer_config.json`'s `chat_template` field, proving the fallback source is load-bearing
    /// when the preferred `chat_template.jinja` source does not resolve.
    func testFallsBackToTokenizerConfigWhenChatTemplateJinjaIsAbsent() throws {
        let json = #"{"chat_template": "\#(attestingTemplateText)"}"#
        let directory = try writeModelDirectory(tokenizerConfigJSON: json, chatTemplateJinja: nil)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("chat_template.jinja").path))

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// `chat_template.jinja` is present and attests, and `tokenizer_config.json` is present but
    /// unreadable JSON — resolution still succeeds via the preferred `chat_template.jinja` source,
    /// so the malformed fallback source never even gets read.
    func testResolvesFromChatTemplateJinjaWhenTokenizerConfigIsUnparseableJSON() throws {
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: "{ not valid json",
            chatTemplateJinja: attestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// Decisive precedence test: BOTH sources are present with DIFFERENT content —
    /// `chat_template.jinja` attests the markers, `tokenizer_config.json`'s `chat_template` field
    /// does NOT. Resolution must follow `chat_template.jinja`'s content and attest `true`. This is
    /// the scenario the deployment runbook actually creates when an operator drops a patched
    /// `chat_template.jinja` next to a stale `tokenizer_config.json`: swift-transformers' `Hub.swift`
    /// renders from the `.jinja` file, so this probe must read the same file or it attests the wrong
    /// template entirely.
    func testChatTemplateJinjaContentWinsWhenBothSourcesArePresentAndDiffer() throws {
        let tokenizerConfigJSON = #"{"chat_template": "\#(nonAttestingTemplateText)"}"#
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: tokenizerConfigJSON,
            chatTemplateJinja: attestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// The converse of the decisive test above: `chat_template.jinja` does NOT attest while
    /// `tokenizer_config.json`'s `chat_template` field DOES. Resolution must still follow
    /// `chat_template.jinja` and attest `false` — proving `chat_template.jinja`'s content wins
    /// outright, not merely "wins when it happens to attest".
    func testChatTemplateJinjaContentWinsEvenWhenItDoesNotAttestAndTokenizerConfigDoes() throws {
        let tokenizerConfigJSON = #"{"chat_template": "\#(attestingTemplateText)"}"#
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: tokenizerConfigJSON,
            chatTemplateJinja: nonAttestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    // MARK: - scalarServingChatTemplateRefusesNonLeadingSystemMessage
    //
    // Boot-time visibility for the deployment runbook's manual "drop a patched chat_template.jinja
    // into the model directory" step (cycle 96: a live serve shipped on a checkpoint whose template
    // was never re-patched, undetected). Uses the SAME resolver the tests above exercise, so these
    // fixtures reuse `writeModelDirectory`/`attestingTemplateText`-style helpers rather than
    // re-deriving resolution order.

    private let refusingTemplateText =
        """
        {%- for message in messages %}
        {%- if message.role == "system" %}
        {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
        {%- endif %}
        {%- endfor %}
        """

    private let nonRefusingTemplateText =
        "{% for message in messages %}{{ message.content }}{% endfor %}"

    /// Positive: the resolved template still contains the stock refusal anchor — the un-patched,
    /// unsafe-for-non-leading-system-messages state this probe exists to surface.
    func testRefusesNonLeadingSystemMessageTrueWhenAnchorPresent() throws {
        let directory = try writeModelDirectory(chatTemplateJinja: refusingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory))
    }

    /// Negative: a readable, resolved template that does NOT contain the anchor — the patched,
    /// safe state.
    func testRefusesNonLeadingSystemMessageFalseWhenAnchorAbsent() throws {
        let directory = try writeModelDirectory(chatTemplateJinja: nonRefusingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory))
    }

    /// Fail-closed: no template resolves at all (neither source present). An unresolvable template
    /// is NOT evidence of a successful patch, so this must report `true` (the conservative
    /// "assume still refuses, keep warning" state), never `false` (which would read as "confirmed
    /// patched") — see the probe's doc comment.
    func testRefusesNonLeadingSystemMessageTrueWhenNoTemplateResolves() throws {
        let directory = try writeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory))
    }

    /// Same fail-closed value for a wholly unreadable/absent model directory, not just an empty one.
    func testRefusesNonLeadingSystemMessageTrueForAnUnreadableModelDirectory() {
        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("flashnext-refusal-probe-absent-\(UUID().uuidString)")

        XCTAssertTrue(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: absent))
    }

    /// Resolution precedence applies to this probe too: `chat_template.jinja` (patched, anchor
    /// removed) wins over a stale `tokenizer_config.json` that still carries the anchor — the exact
    /// scenario the deployment runbook creates and this probe exists to attest correctly.
    func testRefusesNonLeadingSystemMessageFollowsChatTemplateJinjaPrecedenceOverStaleTokenizerConfig()
        throws
    {
        let tokenizerConfigJSON = #"{"chat_template": "\#(refusingTemplateText)"}"#
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: tokenizerConfigJSON,
            chatTemplateJinja: nonRefusingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(
            scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: directory))
    }
}
