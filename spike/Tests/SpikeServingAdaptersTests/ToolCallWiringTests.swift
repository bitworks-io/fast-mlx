// T1 pure-CPU proof for the SERVING-ADAPTER tool-calling wiring: drives the vendored
// MLXLMCommon.ToolCallProcessor with a canned Qwen Hermes stream and proves the
// `openAIToolCalls(from:)` mapping produces the OpenAI wire shape (arguments as a JSON
// *string*). No MLX/model load — safe to run under `swift test` on any host.
import XCTest
import MLXLMCommon

import ServingCore
@testable import SpikeServingAdapters

final class ToolCallWiringTests: XCTestCase {
    func testTwoChunkQwenToolCallMapsToOpenAIWireShape() throws {
        let processor = ToolCallProcessor(format: .json)

        let first = processor.processChunk("<tool_cal")
        let second = processor.processChunk(
            "l>\n{\"name\": \"get_product\", \"arguments\": {\"query\": \"RTX 6000 Ada\"}}\n</tool_call>")
        processor.processEOS()

        XCTAssertNil(first)
        XCTAssertNil(second)
        XCTAssertEqual(processor.toolCalls.count, 1)

        let mapped = openAIToolCalls(from: processor.toolCalls)
        XCTAssertEqual(mapped.count, 1)

        let call = try XCTUnwrap(mapped.first)
        XCTAssertFalse(call.id.isEmpty)
        XCTAssertEqual(call.function.name, "get_product")

        let argumentsData = try XCTUnwrap(call.function.arguments.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: argumentsData) as? [String: Any]
        XCTAssertEqual(parsed?["query"] as? String, "RTX 6000 Ada")
    }

    // The loader maps the model's inferred format (from config.json) into the backend config.
    // A `nil` inference (Llama/Qwen JSON-standard) must fall back to `.json`; a concrete inference
    // (e.g. qwen3_5 → `.xmlFunction`) must be preserved verbatim.
    func testServingToolCallFormatFallsBackToJSONWhenUninferred() {
        XCTAssertEqual(servingToolCallFormat(inferred: nil), .json)
    }

    func testServingToolCallFormatPreservesInferredFormat() {
        XCTAssertEqual(servingToolCallFormat(inferred: .xmlFunction), .xmlFunction)
        XCTAssertEqual(servingToolCallFormat(inferred: .glm4), .glm4)
        XCTAssertEqual(servingToolCallFormat(inferred: .json), .json)
    }

    func testPlainTextWithoutToolCallLeavesDisplayTextIntactAndToolCallsEmpty() throws {
        let processor = ToolCallProcessor(format: .json)

        let displayed = processor.processChunk("The RTX 6000 Ada is in stock.")
        processor.processEOS()

        XCTAssertEqual(displayed, "The RTX 6000 Ada is in stock.")
        XCTAssertTrue(processor.toolCalls.isEmpty)
        XCTAssertTrue(openAIToolCalls(from: processor.toolCalls).isEmpty)
    }

    // AC7: an unclosed/partial tool call at EOS must be recoverable as content, not silently
    // dropped. The backends emit this residual via processEOS(returnBufferedText: true).
    func testUnclosedToolCallBufferIsReturnedAsResidualTextAtEOS() throws {
        let processor = ToolCallProcessor(format: .json)
        // The model starts what looks like a tool call but generation ends before it closes.
        let display = processor.processChunk("<tool_call>\n{\"name\": \"get_p")
        XCTAssertNil(display, "a partial tool call must be withheld from display while buffering")

        let residual = processor.processEOS(returnBufferedText: true)
        XCTAssertTrue(processor.toolCalls.isEmpty, "an unclosed tool call yields no parsed calls")
        let recovered = try XCTUnwrap(residual, "residual buffered text must be returned, not dropped")
        XCTAssertTrue(recovered.contains("get_p"), "buffered content is preserved: \(recovered)")
    }
}
