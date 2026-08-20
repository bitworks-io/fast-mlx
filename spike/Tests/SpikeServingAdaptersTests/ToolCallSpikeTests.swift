// Regression guard: exercising the vendored MLXLMCommon ToolCallProcessor / ToolCallFormat under
// `swift test` must NOT trip the MLX metallib runtime load ("Failed to load the default metallib")
// — tool-call PARSING is pure CPU string work and must stay unit-testable off-GPU. Also pins that
// Qwen3 resolves to the Hermes JSON format (not the Coder xmlFunction parser).
import XCTest
import MLXLMCommon

final class ToolCallSpikeTests: XCTestCase {
    func testProcessorParsesQwenHermesToolCallWithoutMetalRuntime() throws {
        let processor = ToolCallProcessor(format: .json)
        let emitted = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"
        _ = processor.processChunk(emitted)
        processor.processEOS()
        XCTAssertEqual(processor.toolCalls.count, 1, "expected exactly one parsed tool call")
        XCTAssertEqual(processor.toolCalls.first?.function.name, "get_weather")
    }

    func testFormatInferenceForQwen3ResolvesToHermesJSON() throws {
        // Qwen3-8B / Qwen3-32B report model_type "qwen3"; must NOT resolve to xmlFunction.
        let inferred = ToolCallFormat.infer(from: "qwen3")
        XCTAssertNotEqual(inferred, .xmlFunction, "plain qwen3 must not use the Coder xmlFunction parser")
    }
}
