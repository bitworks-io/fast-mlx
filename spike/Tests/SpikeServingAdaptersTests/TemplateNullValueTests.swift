import Foundation
import XCTest

import HuggingFace
import Jinja
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// JSON `null` in chat-template inputs (history tool-call arguments, tool schemas) must reach the
/// template as Jinja `none`, not Foundation's `NSNull` (which swift-jinja's `Value(any:)` rejects,
/// surfacing as a `chat_template_rejected` 400). Cycle 236c.
final class TemplateNullValueTests: XCTestCase {
    private static let nullyJSON = #"{"a":null,"b":[1,null,{"c":null}]}"#

    private func nullyValue() throws -> ServingJSONValue {
        let object = try JSONSerialization.jsonObject(with: Data(Self.nullyJSON.utf8))
        return ServingJSONValue(foundation: object)
    }

    // MARK: (a) helper level, hermetic

    func testTemplateSendableConvertsAndTojsonRendersNull() throws {
        let tree = try XCTUnwrap(nullyValue().templateObjectSendable)
        let value = try Value(any: tree)
        let template = try Template("{{ x | tojson }}")
        let rendered = try template.render(["x": value])
        XCTAssertEqual(rendered, #"{"a": null, "b": [1, null, {"c": null}]}"#)
    }

    func testTemplateSendableLeavesNonNullScalarsUnchanged() throws {
        let object = try JSONSerialization.jsonObject(
            with: Data(#"{"s":"x","i":3,"d":1.5,"t":true,"f":false}"#.utf8))
        let tree = try XCTUnwrap(ServingJSONValue(foundation: object).templateObjectSendable)
        let rendered = try Template("{{ x | tojson }}").render(["x": try Value(any: tree)])
        XCTAssertEqual(rendered, #"{"d": 1.5, "f": false, "i": 3, "s": "x", "t": true}"#)
    }

    func testFoundationAsSendableStillRejectedByJinja_documentsCause() throws {
        // Control: the ServingCore Foundation conversion keeps NSNull, which Jinja refuses. This
        // pins WHY the adapter-side conversion exists.
        let tree = try XCTUnwrap(nullyValue().asObjectSendable)
        XCTAssertThrowsError(try Value(any: tree))
    }

    // MARK: (b) end to end through MLXScalarTextCodec.render (env-gated tokenizer)

    private func loadQwen3Tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        guard let dirPath = ProcessInfo.processInfo.environment["FASTMLX_QWEN3_TOKENIZER_DIR"],
            !dirPath.isEmpty
        else {
            throw XCTSkip("FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping null-argument render proof")
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

    private static let requestJSON = """
        {"model":"qwen3","messages":[
          {"role":"user","content":"search"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"call_1_0","type":"function","function":{"name":"search","arguments":"{\\"query\\":\\"x\\",\\"opt\\":null,\\"list\\":[1,null,{\\"deep\\":null}]}"}}]},
          {"role":"tool","tool_call_id":"call_1_0","content":"{}"}
        ],"tools":[{"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"},"opt":{"type":["string","null"],"default":null}},"required":["query"]}}}],"max_tokens":64}
        """

    func testHistoryToolCallWithNullArgumentsAndNullToolDefaultRender() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(Self.requestJSON.utf8))

        let ids = try codec.render(
            messages: request.messages,
            tools: request.tools,
            enableThinking: false,
            reasoningEffort: nil,
            addGenerationPrompt: nil)
        let prompt = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)

        // History arguments: null at top level and nested in array/object, tojson -> null.
        XCTAssertTrue(prompt.contains("\"opt\": null"), "missing top-level null argument in:\n\(prompt)")
        XCTAssertTrue(
            prompt.contains("\"list\": [1, null, {\"deep\": null}]"),
            "missing nested null arguments in:\n\(prompt)")
        // Tool schema: "default": null.
        XCTAssertTrue(prompt.contains("\"default\": null"), "missing tool-schema null default in:\n\(prompt)")
    }
}
