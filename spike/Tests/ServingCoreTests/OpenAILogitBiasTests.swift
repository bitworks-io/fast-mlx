import XCTest

@testable import ServingCore

/// Decode + capability-gate contract for OpenAI `logit_bias` on both OpenAI routes.
///
/// `logit_bias` is an object mapping token-id STRINGS (canonical base-10, non-negative) to finite
/// numbers in [-100, 100], at most 300 entries. A valid non-empty map is HONORED (carried as
/// `[Int: Float]` on the decoded request and never listed in `ignoredFields`); absent / `null` /
/// `{}` stay neutral exactly as before this feature. Any violation is a 400 with
/// `param == "logit_bias"`. A backend that cannot apply a non-empty map must be refused at the
/// capability gate (`validateLogitBiasCapability`), never silently ignored.
final class OpenAILogitBiasTests: XCTestCase {
    private func chatBody(_ biasJSON: String?) -> Data {
        let field = biasJSON.map { #","logit_bias":\#($0)"# } ?? ""
        return Data(
            (#"{"model":"m","messages":[{"role":"user","content":"hi"}]"# + field + "}").utf8)
    }

    private func completionBody(_ biasJSON: String?) -> Data {
        let field = biasJSON.map { #","logit_bias":\#($0)"# } ?? ""
        return Data((#"{"model":"m","prompt":"hi""# + field + "}").utf8)
    }

    private func assertRejected(
        _ decode: () throws -> Any,
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            _ = try decode()
            XCTFail("expected a 400 for \(label)", file: file, line: line)
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.type, .invalidRequest, label, file: file, line: line)
            XCTAssertEqual(error.openAIError.param, "logit_bias", label, file: file, line: line)
        } catch {
            XCTFail("unexpected error \(error) for \(label)", file: file, line: line)
        }
    }

    // MARK: accept table

    func testValidMapsDecodeToIntFloatDictionaryOnChat() throws {
        let cases: [(String, [Int: Float])] = [
            (#"{"50":1}"#, [50: 1]),
            (#"{"0":-100}"#, [0: -100]),
            (#"{"7":100}"#, [7: 100]),
            (#"{"12":0.5,"13":-0.25,"14":0}"#, [12: 0.5, 13: -0.25, 14: 0]),
            (#"{"1":1e1}"#, [1: 10]),
            (#"{"2147483647":5}"#, [2_147_483_647: 5]),
        ]
        for (json, expected) in cases {
            let request = try OpenAIChatCompletionRequest.decodeStrict(from: chatBody(json))
            XCTAssertEqual(request.logitBias, expected, json)
            // HONORED, so a non-empty map is never reported as an ignored field.
            XCTAssertFalse(request.ignoredFields.contains("logit_bias"), json)
        }
    }

    func testValidMapsDecodeToIntFloatDictionaryOnCompletions() throws {
        let request = try OpenAICompletionRequest.decodeStrict(from: completionBody(#"{"50":-100,"51":100}"#))
        XCTAssertEqual(request.logitBias, [50: -100, 51: 100])
        XCTAssertFalse(request.ignoredFields.contains("logit_bias"))
        // The completions->chat conversion every backend consumes must carry the map.
        XCTAssertEqual(request.asChatCompletionRequest().logitBias, [50: -100, 51: 100])
    }

    func testExactly300EntriesAreAccepted() throws {
        let json = "{" + (0..<300).map { #""\#($0)":1"# }.joined(separator: ",") + "}"
        let chat = try OpenAIChatCompletionRequest.decodeStrict(from: chatBody(json))
        XCTAssertEqual(chat.logitBias.count, 300)
        let completion = try OpenAICompletionRequest.decodeStrict(from: completionBody(json))
        XCTAssertEqual(completion.logitBias.count, 300)
    }

    // MARK: neutral forms are unchanged

    func testAbsentNullAndEmptyAreNeutral() throws {
        let absent = try OpenAIChatCompletionRequest.decodeStrict(from: chatBody(nil))
        XCTAssertEqual(absent.logitBias, [:])
        XCTAssertEqual(absent.ignoredFields, [])

        let null = try OpenAIChatCompletionRequest.decodeStrict(from: chatBody("null"))
        XCTAssertEqual(null.logitBias, [:])
        XCTAssertEqual(null.ignoredFields, [])

        // `{}` is accepted and (as before this feature) recorded as an ignored neutral field.
        let empty = try OpenAIChatCompletionRequest.decodeStrict(from: chatBody("{}"))
        XCTAssertEqual(empty.logitBias, [:])
        XCTAssertEqual(empty.ignoredFields, ["logit_bias"])

        let completionEmpty = try OpenAICompletionRequest.decodeStrict(from: completionBody("{}"))
        XCTAssertEqual(completionEmpty.logitBias, [:])
        XCTAssertEqual(completionEmpty.ignoredFields, ["logit_bias"])
        let completionNull = try OpenAICompletionRequest.decodeStrict(from: completionBody("null"))
        XCTAssertEqual(completionNull.logitBias, [:])
        XCTAssertEqual(completionNull.ignoredFields, [])
    }

    // MARK: reject table (each asserts param == "logit_bias")

    private static let rejectedMaps: [(String, String)] = [
        (#"{"-1":1}"#, "negative id"),
        (#"{"1.5":1}"#, "fractional id"),
        (#"{"abc":1}"#, "non-numeric id"),
        (#"{"":1}"#, "empty id"),
        (#"{"+5":1}"#, "leading plus"),
        (#"{" 5":1}"#, "leading space"),
        (#"{"5 ":1}"#, "trailing space"),
        (#"{"1e2":1}"#, "exponent id"),
        (#"{"0x10":1}"#, "hex id"),
        (#"{"050":1}"#, "leading zero id (ambiguous with 50)"),
        (#"{"99999999999999999999":1}"#, "id overflowing Int"),
        (#"{"٣":1}"#, "non-ASCII digit id"),
        (#"{"5":100.5}"#, "value above 100"),
        (#"{"5":101}"#, "value 101"),
        (#"{"5":-100.0001}"#, "value below -100"),
        (#"{"5":1e308}"#, "huge value"),
        (#"{"5":"1"}"#, "string value"),
        (#"{"5":true}"#, "boolean value"),
        (#"{"5":null}"#, "null value"),
        (#"{"5":[1]}"#, "array value"),
        (#"{"5":{}}"#, "object value"),
        (#"[1,2]"#, "array instead of object"),
        (#""50""#, "string instead of object"),
        (#"7"#, "number instead of object"),
        (#"true"#, "boolean instead of object"),
    ]

    func testRejectTableOnChat() {
        for (json, label) in Self.rejectedMaps {
            assertRejected({ try OpenAIChatCompletionRequest.decodeStrict(from: chatBody(json)) }, label)
        }
    }

    func testRejectTableOnCompletions() {
        for (json, label) in Self.rejectedMaps {
            assertRejected({ try OpenAICompletionRequest.decodeStrict(from: completionBody(json)) }, label)
        }
    }

    func testMoreThan300EntriesAreRejected() {
        let json = "{" + (0..<301).map { #""\#($0)":1"# }.joined(separator: ",") + "}"
        assertRejected({ try OpenAIChatCompletionRequest.decodeStrict(from: chatBody(json)) }, "301 chat entries")
        assertRejected(
            { try OpenAICompletionRequest.decodeStrict(from: completionBody(json)) }, "301 completion entries")
    }

    // MARK: capability gate

    func testCapabilityGateRefusesANonEmptyMapOnAnUnsupportedBackend() {
        var request = OpenAIChatCompletionRequest(
            model: "m", messages: [OpenAIChatMessage(role: .user, text: "hi")],
            maxCompletionTokens: 4, temperature: 0, choiceCount: 1, stream: false, stop: [])
        request.logitBias = [5: 1]
        do {
            try validateLogitBiasCapability(request: request, backendSupportsLogitBias: false)
            XCTFail("a backend without logit_bias support must be refused with a 400")
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.type, .invalidRequest)
            XCTAssertEqual(error.openAIError.param, "logit_bias")
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    func testCapabilityGateAdmitsASupportedBackendAndTheNeutralMap() throws {
        var request = OpenAIChatCompletionRequest(
            model: "m", messages: [OpenAIChatMessage(role: .user, text: "hi")],
            maxCompletionTokens: 4, temperature: 0, choiceCount: 1, stream: false, stop: [])
        request.logitBias = [5: 1]
        XCTAssertNoThrow(try validateLogitBiasCapability(request: request, backendSupportsLogitBias: true))
        request.logitBias = [:]
        // The neutral map never needs the capability: `{}` keeps working on every backend.
        XCTAssertNoThrow(try validateLogitBiasCapability(request: request, backendSupportsLogitBias: false))
    }

    /// The protocol default must fail closed: a backend that never heard of logit_bias is unsupported.
    func testBackendProtocolDefaultIsUnsupported() {
        struct Bare: ServingGenerationBackend {
            func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
                throw OpenAIServingError.server("unused", code: "unused")
            }
        }
        XCTAssertFalse(Bare().supportsLogitBias)
        XCTAssertFalse(Bare().enforcesForcingToolChoice)
    }

    // MARK: forcing tool_choice capability gate (`validateToolChoiceCapability`)

    private func toolChoiceRequest(
        _ choice: OpenAIToolChoice, promptInput: ServingPromptInput = .chat
    ) -> OpenAIChatCompletionRequest {
        var request = OpenAIChatCompletionRequest(
            model: "m", messages: [OpenAIChatMessage(role: .user, text: "hi")],
            maxCompletionTokens: 4, temperature: 0, choiceCount: 1, stream: false, stop: [])
        request.toolChoice = choice
        request.promptInput = promptInput
        return request
    }

    private func assertToolChoiceRefused(
        _ request: OpenAIChatCompletionRequest, enforces: Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        do {
            try validateToolChoiceCapability(request: request, backendEnforcesToolChoice: enforces)
            XCTFail("expected a tool_choice refusal", file: file, line: line)
        } catch let error as OpenAIServingError {
            XCTAssertEqual(error.openAIError.type, .invalidRequest, file: file, line: line)
            XCTAssertEqual(error.openAIError.param, "tool_choice", file: file, line: line)
        } catch {
            XCTFail("unexpected error \(error)", file: file, line: line)
        }
    }

    func testToolChoiceGateAutoAndNonePassOnANonEnforcingBackend() {
        for choice: OpenAIToolChoice in [.auto, .none] {
            XCTAssertNoThrow(
                try validateToolChoiceCapability(
                    request: toolChoiceRequest(choice), backendEnforcesToolChoice: false))
        }
    }

    func testToolChoiceGateRefusesForcingChoicesOnANonEnforcingBackend() {
        for choice: OpenAIToolChoice in [.function("get_weather"), .required] {
            assertToolChoiceRefused(toolChoiceRequest(choice), enforces: false)
        }
    }

    func testToolChoiceGateAdmitsForcingChoicesOnAnEnforcingBackendForChat() {
        for choice: OpenAIToolChoice in [.function("get_weather"), .required] {
            XCTAssertNoThrow(
                try validateToolChoiceCapability(
                    request: toolChoiceRequest(choice), backendEnforcesToolChoice: true))
        }
    }

    /// Enforcement only runs for a chat-templated prompt, so an in-process raw-text request carrying
    /// a forcing choice is refused even on an enforcing backend (closes a fail-open).
    func testToolChoiceGateRefusesForcingChoicesForRawTextEvenOnAnEnforcingBackend() {
        for choice: OpenAIToolChoice in [.function("get_weather"), .required] {
            assertToolChoiceRefused(
                toolChoiceRequest(choice, promptInput: .rawText("Once upon a time")), enforces: true)
        }
    }
}
