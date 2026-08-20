import XCTest

@testable import ServingCore

final class OpenAIReasoningEffortTests: XCTestCase {
    func testTopLevelReasoningEffortIsDecoded() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"medium"}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.reasoningEffort, "medium")
    }

    func testInvalidTopLevelReasoningEffortIsRejected() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"ultra"}
        """

        XCTAssertThrowsError(try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("Expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "reasoning_effort")
        }
    }

    func testChatTemplateKwargsEnableThinkingIsDecoded() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":false}}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.enableThinking, false)
    }

    func testChatTemplateKwargsReasoningEffortIsDecoded() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_effort":"low"}}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.reasoningEffort, "low")
    }

    func testInvalidChatTemplateKwargsReasoningEffortIsRejected() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_effort":"ultra"}}
        """

        XCTAssertThrowsError(try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("Expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "chat_template_kwargs.reasoning_effort")
        }
    }

    func testTopLevelEnableThinkingWinsOverChatTemplateKwargs() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":false}}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.enableThinking, true)
    }

    func testTopLevelReasoningEffortWinsOverChatTemplateKwargs() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"xhigh","chat_template_kwargs":{"reasoning_effort":"low"}}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.reasoningEffort, "xhigh")
    }

    func testUnknownChatTemplateKwargsKeysAreIgnored() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"unknown_key":1}}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertNil(request.enableThinking)
        XCTAssertNil(request.reasoningEffort)
    }

    func testReasoningFieldsAreNilWhenAbsent() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}]}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertNil(request.reasoningEffort)
        XCTAssertNil(request.enableThinking)
    }

    func testChatTemplateKwargsNotObjectIsRejected() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":"nope"}
        """

        XCTAssertThrowsError(try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("Expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "chat_template_kwargs")
        }
    }
}
