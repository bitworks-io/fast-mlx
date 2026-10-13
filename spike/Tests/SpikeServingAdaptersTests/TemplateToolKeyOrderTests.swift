import Foundation
import XCTest

import HuggingFace
import Jinja
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import Tokenizers
@testable import SpikeServingAdapters

/// Chat-template `tools` and history tool-call `arguments` must reach Jinja in the client's JSON key
/// order (Python transformers parity), with `json.loads` number semantics (`2.0` stays a float,
/// `1` stays an int). Cycle 237.
final class TemplateToolKeyOrderTests: XCTestCase {
    private static let toolJSON =
        #"{"type":"function","function":{"name":"search","description":"d","parameters":{"type":"object","properties":{"query":{"type":"string"},"opt":{"type":"string"}},"required":["query"]}}}"#

    private func render(_ json: String) throws -> String {
        let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(json.utf8))
        let value = raw.templateJinjaValue
        return try Template("{{ x | tojson }}").render(["x": value])
    }

    // MARK: hermetic

    func testOrderedConverterTojsonKeepsNestedOrder() throws {
        XCTAssertEqual(
            try render(Self.toolJSON),
            #"{"type": "function", "function": {"name": "search", "description": "d", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "opt": {"type": "string"}}, "required": ["query"]}}}"#
        )
    }

    func testOrderedConverterKeepsFloatLiteralAndInt() throws {
        XCTAssertEqual(try render(#"{"z":1,"a":2.0}"#), #"{"z": 1, "a": 2.0}"#)
    }

    func testOrderedConverterFollowsPythonNumberSemantics() throws {
        XCTAssertEqual(
            try render(#"{"e":1e5,"f":1E16,"g":-0.0,"n":-0,"big":123456789012345678901,"s":"é","t":true,"u":null}"#),
            #"{"e": 100000.0, "f": 1e+16, "g": -0.0, "n": 0, "big": 1.2345678901234568e+20, "s": "é", "t": true, "u": null}"#)
    }

    // MARK: end to end (env-gated tokenizer)

    private func loadQwen3Tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        guard let dirPath = ProcessInfo.processInfo.environment["FASTMLX_QWEN3_TOKENIZER_DIR"],
            !dirPath.isEmpty
        else {
            throw XCTSkip("FASTMLX_QWEN3_TOKENIZER_DIR is not set; skipping key-order render proof")
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

    private func renderText(
        _ tokenizer: any MLXLMCommon.Tokenizer, body: String
    ) throws -> (text: String, ids: [Int]) {
        let codec = MLXScalarTextCodec(tokenizer: tokenizer)
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        let ids = try codec.render(
            messages: request.messages, tools: request.tools, enableThinking: false,
            reasoningEffort: nil, addGenerationPrompt: nil)
        return (tokenizer.decode(tokenIds: ids, skipSpecialTokens: false), ids)
    }

    func testToolsRenderInClientKeyOrderThroughRender() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let body =
            #"{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"tools":[\#(Self.toolJSON)]}"#
        let (text, _) = try renderText(tokenizer, body: body)
        XCTAssertTrue(
            text.contains(
                #"{"type": "function", "function": {"name": "search", "description": "d", "parameters": {"type": "object", "properties": {"query""#
            ), "tool not in client key order:\n\(text)")
    }

    func testHistoryArgumentsRenderInClientOrderWithFloatLiteral() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let body = """
            {"model":"qwen3","messages":[
              {"role":"user","content":"go"},
              {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{\\"z\\":1,\\"a\\":2.0}"}}]},
              {"role":"tool","tool_call_id":"c1","content":"{}"}
            ],"tools":[\(Self.toolJSON)]}
            """
        let (text, _) = try renderText(tokenizer, body: body)
        XCTAssertTrue(text.contains(#"{"z": 1, "a": 2.0}"#), "history arguments not in client order:\n\(text)")
    }

    func testPlainPromptUnchanged() async throws {
        let tokenizer = try await loadQwen3Tokenizer()
        let body = #"{"model":"qwen3","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]}"#
        let (_, ids) = try renderText(tokenizer, body: body)
        let expected = try tokenizer.applyChatTemplate(
            messages: [["role": "system", "content": "s"], ["role": "user", "content": "hi"]],
            tools: nil,
            additionalContext: ["enable_thinking": false, "add_generation_prompt": true])
        XCTAssertEqual(ids, expected)
    }
}
