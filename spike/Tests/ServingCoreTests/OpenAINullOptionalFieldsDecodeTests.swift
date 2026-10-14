import XCTest

@testable import ServingCore

/// JSON `null` on an optional request field means "absent" (cycle 241). The OpenAI spec allows `null`
/// for these fields and SDK clients serialize it (`stop=None`); the strict decoder used to answer
/// `… must be a boolean` / `must be numeric` / `stop must be a string or an array…` (a 400) for it.
/// Predeclared acceptance AC1/AC2 in
/// `docs/task-inbox/2026-10-13-PREDECLARATION-null-optional-request-fields-mean-absent.md`.
///
/// Not changed, and pinned here as controls: `ignore_eos: null` stays a 400 (also pinned by
/// `OpenAIIgnoreEOSTests`), required fields (`model`, `messages`) still refuse `null`, and a wrong
/// non-null type is still a specific 400 naming its field.
final class OpenAINullOptionalFieldsDecodeTests: XCTestCase {
    private let chatBase = #"{"model":"m","messages":[{"role":"user","content":"hi"}]"#
    private let completionBase = #"{"model":"m","prompt":"hi""#

    private func chat(_ extra: String = "") throws -> OpenAIChatCompletionRequest {
        try OpenAIChatCompletionRequest.decodeStrict(from: Data((chatBase + extra + "}").utf8))
    }

    private func completion(_ extra: String = "") throws -> OpenAICompletionRequest {
        try OpenAICompletionRequest.decodeStrict(from: Data((completionBase + extra + "}").utf8))
    }

    /// (message, param) of the `invalidRequest` a decode throws; `nil` when it decodes.
    private func chatError(_ extra: String) -> (message: String, param: String?)? {
        do {
            _ = try chat(extra)
            return nil
        } catch OpenAIServingError.invalidRequest(let message, let param) {
            return (message, param)
        } catch {
            XCTFail("expected invalidRequest, got \(error)")
            return nil
        }
    }

    private func completionError(_ extra: String) -> (message: String, param: String?)? {
        do {
            _ = try completion(extra)
            return nil
        } catch OpenAIServingError.invalidRequest(let message, let param) {
            return (message, param)
        } catch {
            XCTFail("expected invalidRequest, got \(error)")
            return nil
        }
    }

    private let toolsJSON =
        #"[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}]"#

    private let chatFields = [
        "max_completion_tokens", "max_tokens", "temperature", "n", "stream", "stop", "tools",
        "tool_choice", "parallel_tool_calls", "top_p", "top_k", "min_p", "seed",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "enable_thinking",
        "add_generation_prompt",
    ]

    private let completionFields = [
        "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "n", "stream", "stop",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "best_of",
    ]

    // MARK: AC1: decode equivalence

    func testEachNullableChatFieldDecodesLikeAbsent() throws {
        let absent = try chat()
        for field in chatFields {
            let withNull: OpenAIChatCompletionRequest
            do {
                withNull = try chat(#","\#(field)":null"#)
            } catch {
                XCTFail("chat field \(field): null was refused (\(error)) but must decode like absent")
                continue
            }
            XCTAssertEqual(withNull, absent, "chat field \(field): null must decode like absent")
            XCTAssertEqual(withNull.ignoredFields, [], "chat field \(field): null must not be listed as ignored")
        }
    }

    func testEachNullableCompletionsFieldDecodesLikeAbsent() throws {
        let absent = try completion()
        for field in completionFields {
            let withNull: OpenAICompletionRequest
            do {
                withNull = try completion(#","\#(field)":null"#)
            } catch {
                XCTFail("completions field \(field): null was refused (\(error)) but must decode like absent")
                continue
            }
            XCTAssertEqual(withNull, absent, "completions field \(field): null must decode like absent")
            XCTAssertEqual(
                withNull.ignoredFields, [], "completions field \(field): null must not be listed as ignored")
        }
    }

    func testChatStoreNullDecodesLikeAbsentAndIsNotListedIgnored() throws {
        let withNull = try chat(#","store":null"#)
        XCTAssertEqual(withNull, try chat())
        XCTAssertEqual(withNull.ignoredFields, [])
    }

    func testSDKNullDefaultsBodyDecodesLikeMinimal() throws {
        let nulls = chatFields.map { #","\#($0)":null"# }.joined()
        let request = try chat(nulls)
        XCTAssertEqual(request, try chat())
        XCTAssertEqual(request.ignoredFields, [])

        let completionNulls = completionFields.map { #","\#($0)":null"# }.joined()
        let completionRequest = try completion(completionNulls)
        XCTAssertEqual(completionRequest, try completion())
        XCTAssertEqual(completionRequest.ignoredFields, [])
    }

    func testNullDoesNotMaskARealValue() throws {
        // A null on one field must not disturb a real value on another.
        let request = try chat(#","temperature":0,"max_tokens":null,"stop":null,"top_p":0.5"#)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.topP, 0.5)
        XCTAssertNil(request.maxCompletionTokens, "max_tokens:null gets the server default budget, like absent")
        XCTAssertEqual(request.stop, [])
    }

    // MARK: tool_choice resolution

    func testToolChoiceNullWithToolsResolvesAuto() throws {
        let request = try chat(#","tools":\#(toolsJSON),"tool_choice":null"#)
        XCTAssertEqual(request.toolChoice, .auto)
        XCTAssertEqual(request.tools.count, 1)
        XCTAssertEqual(request, try chat(#","tools":\#(toolsJSON)"#))
    }

    func testToolChoiceNullWithoutToolsResolvesNone() throws {
        XCTAssertEqual(try chat(#","tool_choice":null"#).toolChoice, .none)
        XCTAssertEqual(try chat(#","tools":null,"tool_choice":null"#).toolChoice, .none)
        XCTAssertTrue(try chat(#","tools":null"#).tools.isEmpty)
    }

    // MARK: nested call sites of the shared helpers

    func testNestedNullsDecodeLikeAbsent() throws {
        // stream_options.include_usage
        XCTAssertEqual(
            try chat(#","stream":true,"stream_options":{"include_usage":null}"#),
            try chat(#","stream":true,"stream_options":{}"#),
            "stream_options.include_usage: null must decode like absent (chat)")
        XCTAssertEqual(
            try completion(#","stream":true,"stream_options":{"include_usage":null}"#),
            try completion(#","stream":true,"stream_options":{}"#),
            "stream_options.include_usage: null must decode like absent (completions)")
        // chat_template_kwargs.enable_thinking / add_generation_prompt
        XCTAssertEqual(
            try chat(#","chat_template_kwargs":{"enable_thinking":null,"add_generation_prompt":null}"#),
            try chat(#","chat_template_kwargs":{}"#),
            "chat_template_kwargs.enable_thinking / add_generation_prompt: null must decode like absent")
        // response_format.json_schema.strict
        let schema = #""name":"s","schema":{"type":"object","properties":{"a":{"type":"string"}}}"#
        XCTAssertEqual(
            try chat(#","response_format":{"type":"json_schema","json_schema":{\#(schema),"strict":null}}"#),
            try chat(#","response_format":{"type":"json_schema","json_schema":{\#(schema)}}"#),
            "response_format.json_schema.strict: null must decode like absent")
    }

    // MARK: controls

    func testNullStreamWithStreamOptionsBehavesLikeAbsentStream() {
        // Absent `stream` + `stream_options` is a 400 today ("requires stream to be true"); a null
        // `stream` must be the same answer, not a different one and not a silent acceptance.
        let absent = chatError(#","stream_options":{"include_usage":true}"#)
        let null = chatError(#","stream":null,"stream_options":{"include_usage":true}"#)
        XCTAssertNotNil(absent)
        XCTAssertEqual(null?.message, absent?.message)
        XCTAssertEqual(null?.param, absent?.param)
        XCTAssertEqual(null?.param, "stream_options")

        let absentCompletion = completionError(#","stream_options":{"include_usage":true}"#)
        let nullCompletion = completionError(#","stream":null,"stream_options":{"include_usage":true}"#)
        XCTAssertNotNil(absentCompletion)
        XCTAssertEqual(nullCompletion?.message, absentCompletion?.message)
        XCTAssertEqual(nullCompletion?.param, absentCompletion?.param)

        // And a null stream_options with a null stream is fine (the SDK default shape).
        XCTAssertNoThrow(try chat(#","stream":null,"stream_options":null"#))
    }

    func testRequiredFieldsNullStillRefused() {
        for (body, param) in [
            (#"{"model":null,"messages":[{"role":"user","content":"hi"}]}"#, "model"),
            (#"{"model":"m","messages":null}"#, "messages"),
        ] {
            XCTAssertThrowsError(try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)), body) {
                guard case OpenAIServingError.invalidRequest(_, let actual) = $0 else {
                    return XCTFail("expected invalidRequest, got \($0)")
                }
                XCTAssertEqual(actual, param)
            }
        }
        XCTAssertThrowsError(
            try OpenAICompletionRequest.decodeStrict(from: Data(#"{"model":null,"prompt":"hi"}"#.utf8))
        ) {
            guard case OpenAIServingError.invalidRequest(_, let actual) = $0 else {
                return XCTFail("expected invalidRequest, got \($0)")
            }
            XCTAssertEqual(actual, "model")
        }
    }

    func testWrongTypesStillRefused() {
        for (extra, param) in [
            (#","temperature":"0.5""#, "temperature"),
            (#","stop":5"#, "stop"),
            (#","tools":{}"#, "tools"),
            (#","max_tokens":"8""#, "max_tokens"),
            (#","stream":"yes""#, "stream"),
            (#","tool_choice":5"#, "tool_choice"),
        ] {
            let error = chatError(extra)
            XCTAssertNotNil(error, "\(extra) must be refused")
            XCTAssertEqual(error?.param, param, "\(extra) must name its field")
        }
        let completionError = completionError(#","temperature":"0.5""#)
        XCTAssertEqual(completionError?.param, "temperature")
    }

    func testIgnoreEOSNullStillRefused() {
        XCTAssertEqual(chatError(#","ignore_eos":null"#)?.param, "ignore_eos")
        XCTAssertEqual(completionError(#","ignore_eos":null"#)?.param, "ignore_eos")
        // Even next to other nulls that are now accepted.
        XCTAssertEqual(chatError(#","stop":null,"temperature":null,"ignore_eos":null"#)?.param, "ignore_eos")
    }
}
