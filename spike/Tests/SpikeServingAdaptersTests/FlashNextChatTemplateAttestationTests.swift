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
/// x2 — the fixtures below exercise the resolution order (`tokenizer_config.json` preferred, falling
/// back to `chat_template.jinja`) and the fail-closed behavior when neither attests the markers or
/// neither is readable.
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

    /// Fallback path: `tokenizer_config.json` has no `chat_template` key (and no file at all in this
    /// case), so resolution falls back to a sibling `chat_template.jinja` that DOES attest.
    func testFallsBackToChatTemplateJinjaWhenTokenizerConfigHasNoChatTemplateKey() throws {
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: #"{"tokenizer_class": "PreTrainedTokenizerFast"}"#,
            chatTemplateJinja: attestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// `tokenizer_config.json` carries the template and `chat_template.jinja` is absent — resolution
    /// must succeed from `tokenizer_config.json` alone, proving the fallback is not load-bearing when
    /// the preferred source already resolves.
    func testResolvesFromTokenizerConfigAloneWhenChatTemplateJinjaIsAbsent() throws {
        let json = #"{"chat_template": "\#(attestingTemplateText)"}"#
        let directory = try writeModelDirectory(tokenizerConfigJSON: json, chatTemplateJinja: nil)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("chat_template.jinja").path))

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }

    /// `tokenizer_config.json` is present but unreadable JSON — falls back to `chat_template.jinja`.
    func testFallsBackToChatTemplateJinjaWhenTokenizerConfigIsUnparseableJSON() throws {
        let directory = try writeModelDirectory(
            tokenizerConfigJSON: "{ not valid json",
            chatTemplateJinja: attestingTemplateText)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: directory))
    }
}
