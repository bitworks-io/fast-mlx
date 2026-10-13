import XCTest

@testable import ServingCore

/// An assistant history turn may echo back the `reasoning_content` our own responses emit (cycle 238).
/// Strict decode keeps it as a string on assistant messages, treats `null`/absent as nil, and refuses
/// any other type or any non-assistant role with a specific 400 (never a 500).
final class AssistantReasoningHistoryDecodeTests: XCTestCase {
    private func body(messages: String) -> Data {
        Data(#"{"model":"m","messages":\#(messages)}"#.utf8)
    }

    private func expectInvalidRequest(
        _ data: Data, message: String, param: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        do {
            _ = try OpenAIChatCompletionRequest.decodeStrict(from: data)
            XCTFail("Expected OpenAIServingError", file: file, line: line)
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.type, .invalidRequest, file: file, line: line)
            XCTAssertEqual(error.openAIError.message, message, file: file, line: line)
            XCTAssertEqual(error.openAIError.param, param, file: file, line: line)
        } catch {
            XCTFail("Expected OpenAIServingError, got \(error)", file: file, line: line)
        }
    }

    // T1
    func testAssistantMessageKeepsReasoningContent() throws {
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: body(
                messages:
                    #"[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"because"}]"#
            ))
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertNil(request.messages[0].reasoningContent)
        XCTAssertEqual(request.messages[1].reasoningContent, "because")
        XCTAssertEqual(request.messages[1].text, "a")
    }

    // T2: the message our response encoder emits is accepted back as history, unchanged.
    func testResponseMessageEchoedBackAsHistoryRoundTrips() throws {
        let response = OpenAIChatCompletionResponse(
            id: "chatcmpl-x", created: 1, model: "m", content: "the answer",
            finishReason: .toolCalls,
            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1),
            toolCalls: [
                OpenAIToolCall(
                    id: "call_1", function: .init(name: "search", arguments: #"{"q":"x"}"#))
            ],
            reasoningContent: "step one\nstep two")
        let encoded = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertEqual(message["reasoning_content"] as? String, "step one\nstep two")

        let history = try JSONSerialization.data(withJSONObject: [
            "model": "m",
            "messages": [["role": "user", "content": "go"], message],
        ])
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: history)
        XCTAssertEqual(request.messages[1].role, .assistant)
        XCTAssertEqual(request.messages[1].reasoningContent, "step one\nstep two")
        XCTAssertEqual(request.messages[1].toolCalls.map(\.id), ["call_1"])
        XCTAssertEqual(request.messages[1].text, "the answer")
    }

    // T3
    func testNullAndAbsentReasoningContentDecodeToNil() throws {
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: body(
                messages:
                    #"[{"role":"user","content":"q"},{"role":"assistant","content":"a"},{"role":"assistant","content":"b","reasoning_content":null}]"#
            ))
        XCTAssertNil(request.messages[1].reasoningContent)
        XCTAssertNil(request.messages[2].reasoningContent)
    }

    func testNullReasoningContentOnNonAssistantRoleIsTreatedAsAbsent() throws {
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: body(messages: #"[{"role":"user","content":"q","reasoning_content":null}]"#))
        XCTAssertNil(request.messages[0].reasoningContent)
    }

    // T4
    func testNonAssistantRolesRefuseReasoningContentString() {
        let cases = [
            #"{"role":"user","content":"q","reasoning_content":"r"}"#,
            #"{"role":"system","content":"q","reasoning_content":"r"}"#,
            #"{"role":"developer","content":"q","reasoning_content":"r"}"#,
            #"{"role":"tool","tool_call_id":"c1","content":"q","reasoning_content":"r"}"#,
        ]
        for message in cases {
            expectInvalidRequest(
                body(messages: "[\(message)]"),
                message: "Only assistant messages may carry reasoning_content",
                param: "messages.reasoning_content")
        }
    }

    func testReasoningContentOfWrongTypeIsRefused() {
        for value in ["123", "{}", "[]", "true", "1.5"] {
            expectInvalidRequest(
                body(messages: #"[{"role":"assistant","content":"a","reasoning_content":\#(value)}]"#),
                message: "messages.reasoning_content must be a string",
                param: "messages.reasoning_content")
        }
        // The type check precedes the role check, so a wrong type on a non-assistant role says so.
        expectInvalidRequest(
            body(messages: #"[{"role":"user","content":"q","reasoning_content":123}]"#),
            message: "messages.reasoning_content must be a string",
            param: "messages.reasoning_content")
    }

    // MARK: cycle 240: the SDK's serialized assistant message and the `reasoning` alias

    private func decodeOne(_ message: String) throws -> OpenAIChatMessage {
        try OpenAIChatCompletionRequest.decodeStrict(from: body(messages: "[\(message)]")).messages[0]
    }

    // The OpenAI Python SDK's `ChatCompletionMessage.model_dump()` shape, None -> null.
    func testSDKDumpEchoDecodes() throws {
        let message = try decodeOne(
            #"{"role":"assistant","content":"a","reasoning_content":"because","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":null}"#
        )
        XCTAssertEqual(message.role, .assistant)
        XCTAssertEqual(message.text, "a")
        XCTAssertEqual(message.reasoningContent, "because")
        XCTAssertTrue(message.toolCalls.isEmpty)
    }

    func testEmptyAnnotationsArrayIsAccepted() throws {
        let message = try decodeOne(#"{"role":"assistant","content":"a","annotations":[]}"#)
        XCTAssertEqual(message.text, "a")
        XCTAssertNil(message.reasoningContent)
    }

    func testNullToolCallsIsTreatedAsAbsent() throws {
        let assistant = try decodeOne(#"{"role":"assistant","content":"a","tool_calls":null}"#)
        XCTAssertEqual(assistant.text, "a")
        XCTAssertTrue(assistant.toolCalls.isEmpty)
        let user = try decodeOne(#"{"role":"user","content":"q","tool_calls":null}"#)
        XCTAssertEqual(user.text, "q")
        XCTAssertTrue(user.toolCalls.isEmpty)
    }

    func testNullSDKFieldsOnUserMessageAreAccepted() throws {
        let message = try decodeOne(
            #"{"role":"user","content":"q","refusal":null,"annotations":null,"audio":null,"function_call":null}"#
        )
        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.text, "q")
    }

    func testReasoningAliasStringOnAssistantFeedsReasoningContent() throws {
        let message = try decodeOne(#"{"role":"assistant","content":"a","reasoning":"because"}"#)
        XCTAssertEqual(message.reasoningContent, "because")
    }

    func testEqualReasoningAndReasoningContentAreAccepted() throws {
        let message = try decodeOne(
            #"{"role":"assistant","content":"a","reasoning":"because","reasoning_content":"because"}"#)
        XCTAssertEqual(message.reasoningContent, "because")
    }

    func testDifferingReasoningAndReasoningContentAreRefused() {
        expectInvalidRequest(
            body(
                messages:
                    #"[{"role":"assistant","content":"a","reasoning":"one","reasoning_content":"two"}]"#),
            message: "messages.reasoning and messages.reasoning_content disagree",
            param: "messages.reasoning")
    }

    func testReasoningAliasOnNonAssistantRoleIsRefused() {
        for role in ["user", "system", "developer"] {
            expectInvalidRequest(
                body(messages: #"[{"role":"\#(role)","content":"q","reasoning":"r"}]"#),
                message: "Only assistant messages may carry reasoning",
                param: "messages.reasoning")
        }
        expectInvalidRequest(
            body(messages: #"[{"role":"tool","tool_call_id":"c1","content":"q","reasoning":"r"}]"#),
            message: "Only assistant messages may carry reasoning",
            param: "messages.reasoning")
    }

    func testReasoningAliasOfWrongTypeIsRefused() {
        for value in ["5", "{}", "[]", "true", "1.5"] {
            expectInvalidRequest(
                body(messages: #"[{"role":"assistant","content":"a","reasoning":\#(value)}]"#),
                message: "messages.reasoning must be a string",
                param: "messages.reasoning")
        }
    }

    func testNullReasoningAliasFallsBackToReasoningContent() throws {
        let message = try decodeOne(
            #"{"role":"assistant","content":"a","reasoning":null,"reasoning_content":"because"}"#)
        XCTAssertEqual(message.reasoningContent, "because")
        let user = try decodeOne(#"{"role":"user","content":"q","reasoning":null}"#)
        XCTAssertNil(user.reasoningContent)
    }

    func testNonNullRefusalIsRefused() {
        for role in ["assistant", "user"] {
            expectInvalidRequest(
                body(messages: #"[{"role":"\#(role)","content":"a","refusal":"no"}]"#),
                message: "messages.refusal is not supported; send null or omit it",
                param: "messages.refusal")
        }
    }

    func testNonEmptyOrNonArrayAnnotationsAreRefused() {
        for value in [#"[{"type":"url_citation"}]"#, #""x""#, "{}", "5"] {
            expectInvalidRequest(
                body(messages: #"[{"role":"assistant","content":"a","annotations":\#(value)}]"#),
                message: "messages.annotations is not supported; send null, [] or omit it",
                param: "messages.annotations")
        }
    }

    func testAudioObjectIsRefused() {
        expectInvalidRequest(
            body(messages: #"[{"role":"assistant","content":"a","audio":{"id":"a"}}]"#),
            message: "messages.audio is not supported; send null or omit it",
            param: "messages.audio")
    }

    func testFunctionCallObjectIsRefused() {
        expectInvalidRequest(
            body(
                messages:
                    #"[{"role":"assistant","content":"a","function_call":{"name":"f","arguments":"{}"}}]"#),
            message: "messages.function_call is not supported; send null or omit it",
            param: "messages.function_call")
    }
}
