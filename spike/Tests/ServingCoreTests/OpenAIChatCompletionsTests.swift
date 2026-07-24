import XCTest

@testable import ServingCore

final class OpenAIChatCompletionsTests: XCTestCase {
    func testAcceptsSupportedTextOnlyRequest() throws {
        let body = """
        {
          "model": "qwen3-32b",
          "messages": [
            {"role": "developer", "content": "Be terse."},
            {"role": "system", "content": "Use plain text."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": [{"type": "text", "text": "Continue"}]}
          ],
          "max_completion_tokens": 32,
          "temperature": 0,
          "n": 1,
          "stream": true,
          "stop": ["</s>", "<|end|>"]
        }
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.model, "qwen3-32b")
        XCTAssertEqual(request.messages.map(\.role), [.developer, .system, .user, .assistant, .user])
        XCTAssertEqual(request.messages.map(\.text), ["Be terse.", "Use plain text.", "Hello", "Hi", "Continue"])
        XCTAssertEqual(request.maxCompletionTokens, 32)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.choiceCount, 1)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.stop, ["</s>", "<|end|>"])
    }

    func testDeprecatedMaxTokensAliasAndConflictBehavior() throws {
        let alias = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7}
        """

        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(alias.utf8)).maxCompletionTokens, 7)

        let matching = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7,"max_completion_tokens":7}
        """

        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(matching.utf8)).maxCompletionTokens, 7)

        let conflicting = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7,"max_completion_tokens":8}
        """

        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(conflicting.utf8)),
            type: .invalidRequest,
            param: "max_completion_tokens")
    }

    func testRequestLimitsRejectOversizedBodyAndCompletionBudget() throws {
        let limits = OpenAIChatRequestLimits(
            maximumBodyBytes: 128,
            maximumCompletionTokens: 8)

        let oversizedBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"\(String(repeating: "x", count: 128))"}]}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(oversizedBody.utf8),
                limits: limits),
            type: .invalidRequest,
            param: nil)

        let oversizedBudget = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":9}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(oversizedBudget.utf8),
                limits: limits),
            type: .invalidRequest,
            param: "max_completion_tokens")
    }

    func testLaunchedModelIdentityFailsClosedBeforeAdmission() throws {
        let body = """
        {"model":"other-model","messages":[{"role":"user","content":"Hi"}]}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertOpenAIError(
            try request.requireLaunchedModel("qwen3-32b"),
            type: .invalidRequest,
            param: "model")
        XCTAssertNoThrow(try request.requireLaunchedModel("other-model"))
    }

    func testRejectsUnknownAndUnsupportedFieldsBeforeAdmission() throws {
        let cases: [(String, String?)] = [
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"unknown":true}"#, "unknown"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"temperature":0.1}"#, "temperature"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"temperature":true}"#, "temperature"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_p":0.5}"#, "top_p"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"n":2}"#, "n"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"tools":[]}"#, "tools"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_object"}}"#, "response_format"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#, "messages.content"),
            (#"{"model":"","messages":[{"role":"user","content":"Hi"}]}"#, "model"),
            (#"{"model":"qwen3-32b","messages":[]}"#, "messages"),
            (#"{"model":"qwen3-32b","messages":[{"role":"tool","content":"Hi"}]}"#, "messages.role"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stop":["a","b","c","d","e"]}"#, "stop"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stop":""}"#, "stop"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":0}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":true}"#, "max_completion_tokens"),
        ]

        for (body, param) in cases {
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: param,
                file: #filePath,
                line: #line)
        }
    }

    func testErrorEnvelopeAlwaysCarriesOfficialShape() throws {
        let error = OpenAIServingError.invalidRequest("Unsupported field: tools", param: "tools")
        let data = try JSONEncoder.openAI.encode(OpenAIErrorEnvelope(error: error.openAIError))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let errorObject = try XCTUnwrap(object["error"] as? [String: Any])

        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error")
        XCTAssertEqual(errorObject["message"] as? String, "Unsupported field: tools")
        XCTAssertEqual(errorObject["param"] as? String, "tools")
        XCTAssertTrue(errorObject.keys.contains("code"))
        XCTAssertTrue(errorObject["code"] is NSNull)
    }

    func testNonStreamResponseObjectShape() throws {
        let response = OpenAIChatCompletionResponse(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            content: "hello",
            finishReason: .stop,
            usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1)
        )

        let data = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])

        XCTAssertEqual(object["object"] as? String, "chat.completion")
        XCTAssertEqual(message["role"] as? String, "assistant")
        XCTAssertEqual(message["content"] as? String, "hello")
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "stop")
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 1)
        XCTAssertEqual(usage["total_tokens"] as? Int, 4)
    }

    func testSSEChunksAreOrderedAndTerminateWithDone() throws {
        let role = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: "assistant", content: nil),
            finishReason: nil)
        let first = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: "hel"),
            finishReason: nil)
        let second = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: "lo"),
            finishReason: nil)
        let finish = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: nil),
            finishReason: .stop)
        let events = try [
            role.sseEvent(),
            first.sseEvent(),
            second.sseEvent(),
            finish.sseEvent(),
            OpenAIChatCompletionChunk.doneSSEEvent,
        ]

        XCTAssertEqual(events.count, 5)
        XCTAssertTrue(events[0].contains(#""object":"chat.completion.chunk""#))
        XCTAssertTrue(events[0].contains(#""role":"assistant""#))
        XCTAssertTrue(events[1].contains(#""content":"hel""#))
        XCTAssertTrue(events[2].contains(#""content":"lo""#))
        XCTAssertTrue(events[3].contains(#""finish_reason":"stop""#))
        XCTAssertEqual(events[4], "data: [DONE]\n\n")
    }
}

private func XCTAssertOpenAIError<T>(
    _ expression: @autoclosure () throws -> T,
    type: OpenAIErrorType,
    param: String?,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    do {
        _ = try expression()
        XCTFail("Expected OpenAIServingError", file: file, line: line)
    } catch let error as OpenAIServingError {
        XCTAssertEqual(error.openAIError.type, type, file: file, line: line)
        XCTAssertEqual(error.openAIError.param, param, file: file, line: line)
    } catch {
        XCTFail("Expected OpenAIServingError, got \(error)", file: file, line: line)
    }
}
