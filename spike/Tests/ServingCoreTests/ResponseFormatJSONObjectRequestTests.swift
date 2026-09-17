import XCTest

@testable import ServingCore

/// Slice 1c acceptance tests (see `docs/task-inbox/2026-09-17-DECISION-response-format-json-object-pure-swift.md`):
/// `response_format: {"type":"json_object"}` is parsed and HONORED on `OpenAIChatCompletionRequest`,
/// with its own refusal messages for the combinations slice 1 cannot support yet. The backend
/// capability gate itself (no backend implements the constraint yet) is covered at the HTTP layer in
/// `OpenAIChatCompletionsHTTPHandlerTests`; this file is scoped to `decodeStrict` parsing only.
final class ResponseFormatJSONObjectRequestTests: XCTestCase {
    private func decode(_ body: String) throws -> OpenAIChatCompletionRequest {
        try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
    }

    // MARK: - Honored `json_object`

    func testJSONObjectIsParsedAndHonoredNotIgnored() throws {
        let request = try decode(
            #"{"model":"m","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_object"}}"#)
        XCTAssertEqual(request.responseFormat, .jsonObject)
        XCTAssertFalse(request.ignoredFields.contains("response_format"))
    }

    // MARK: - `text` behavior is unchanged (control)

    func testAbsentResponseFormatLeavesFieldNilAndUnlisted() throws {
        let request = try decode(#"{"model":"m","messages":[{"role":"user","content":"Hi"}]}"#)
        XCTAssertNil(request.responseFormat)
        XCTAssertFalse(request.ignoredFields.contains("response_format"))
    }

    func testExplicitNullResponseFormatLeavesFieldNilAndUnlisted() throws {
        let request = try decode(
            #"{"model":"m","messages":[{"role":"user","content":"Hi"}],"response_format":null}"#)
        XCTAssertNil(request.responseFormat)
        XCTAssertFalse(request.ignoredFields.contains("response_format"))
    }

    func testTextResponseFormatIsIgnoredNotHonored() throws {
        let request = try decode(
            #"{"model":"m","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"text"}}"#)
        XCTAssertNil(request.responseFormat)
        XCTAssertTrue(request.ignoredFields.contains("response_format"))
    }

    // MARK: - `json_schema` gets its own message/param

    func testJSONSchemaIsRefusedWithItsOwnParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_schema","json_schema":{"name":"x"}}}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    // MARK: - Unknown type / extra keys still 400 (control: proves the parser can still fail)

    func testUnknownTypeIsRejected() throws {
        XCTAssertThrowsError(
            try decode(
                #"{"model":"m","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"bogus"}}"#))
    }

    func testJSONObjectWithExtraKeysIsRejected() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object","extra":1}}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    // MARK: - Combination refusals, each with its own param

    func testJSONObjectWithToolsIsRefusedWithToolsParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object"},
                 "tools":[{"type":"function","function":{"name":"f"}}]}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "tools")
        }
    }

    func testJSONObjectWithLogprobsTrueIsRefusedWithLogprobsParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object"},"logprobs":true}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "logprobs")
        }
    }

    func testJSONObjectWithTopLogprobsIsRefusedWithLogprobsParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object"},"logprobs":true,"top_logprobs":3}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "logprobs")
        }
    }

    func testJSONObjectWithStopIsRefusedWithStopParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object"},"stop":["END"]}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "stop")
        }
    }

    /// `n > 1` is already refused unconditionally for every chat request (before `response_format`
    /// is ever inspected) — this is a control proving that pre-existing global refusal still fires
    /// with `param: "n"` even in the presence of `json_object`, not a new code path.
    func testJSONObjectWithNGreaterThanOneIsRefusedWithNParam() throws {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_object"},"n":2}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "n")
        }
    }

    // MARK: - Legacy `/v1/completions` never honors `response_format` (mutation-proof control)

    /// The legacy completions route has no `response_format` key in its allowed set at all, so a
    /// caller sending it there is accepted as an unrecognized top-level field (tolerated and folded
    /// into `ignoredFields`, per the existing unknown-top-level-key policy) and NEVER honored:
    /// `asChatCompletionRequest()` has no way to set `responseFormat`, so it is always `nil` on the
    /// route this slice does not add support to. A mutation that started honoring it here would flip
    /// `responseFormat` to non-nil and fail this test.
    func testLegacyCompletionsNeverHonorsResponseFormat() throws {
        let body = #"{"model":"m","prompt":"Hi","response_format":{"type":"json_object"}}"#
        let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertNil(request.asChatCompletionRequest().responseFormat)
    }
}
