import XCTest

@testable import ServingCore

/// `tool_choice` that cannot be satisfied is refused at decode time (fail closed) instead of being
/// silently downgraded: `"required"` with no tools used to behave like `"auto"`, and a named function
/// absent from `tools` used to yield an empty callable set. Satisfiable choices decode as before.
final class OpenAIToolChoiceValidationTests: XCTestCase {
    private let chatBase = #"{"model":"m","messages":[{"role":"user","content":"hi"}]"#

    private func tool(_ name: String) -> String {
        #"{"type":"function","function":{"name":"\#(name)","description":"d","parameters":{"type":"object"}}}"#
    }

    private func named(_ name: String) -> String {
        #"{"type":"function","function":{"name":"\#(name)"}}"#
    }

    private func chat(_ extra: String) throws -> OpenAIChatCompletionRequest {
        try OpenAIChatCompletionRequest.decodeStrict(from: Data((chatBase + extra + "}").utf8))
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

    /// The three "no usable tools" shapes: field absent, `null`, empty array.
    private let noToolsShapes = ["", #","tools":null"#, #","tools":[]"#]

    private func assertRefused(
        _ extra: String, message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        guard let error = chatError(extra) else {
            return XCTFail("expected a refusal for \(extra)", file: file, line: line)
        }
        XCTAssertEqual(error.message, message, extra, file: file, line: line)
        XCTAssertEqual(error.param, "tool_choice", extra, file: file, line: line)
    }

    func testRequiredWithoutToolsIsRefused() {
        for shape in noToolsShapes {
            assertRefused(
                shape + #","tool_choice":"required""#,
                message: #"tool_choice "required" requires a non-empty tools array"#)
        }
    }

    func testNamedFunctionWithoutToolsIsRefused() {
        for shape in noToolsShapes {
            assertRefused(
                shape + #","tool_choice":\#(named("f"))"#,
                message: "tool_choice function requires a non-empty tools array")
        }
    }

    func testNamedFunctionNotDeclaredIsRefused() {
        assertRefused(
            #","tools":[\#(tool("f"))],"tool_choice":\#(named("g"))"#,
            message: "tool_choice function name is not declared in tools")
        // Exact, case-sensitive match.
        assertRefused(
            #","tools":[\#(tool("f"))],"tool_choice":\#(named("F"))"#,
            message: "tool_choice function name is not declared in tools")
    }

    func testRefusalDoesNotEchoTheClientName() throws {
        let error = try XCTUnwrap(
            chatError(#","tools":[\#(tool("f"))],"tool_choice":\#(named("secret_name_xyz"))"#))
        XCTAssertFalse(error.message.contains("secret_name_xyz"))
    }

    func testSatisfiableChoicesStillDecode() throws {
        let both = "[\(tool("f")),\(tool("g"))]"
        let namedRequest = try chat(#","tools":\#(both),"tool_choice":\#(named("g"))"#)
        XCTAssertEqual(namedRequest.toolChoice, .function("g"))
        XCTAssertEqual(namedRequest.activeTools.map(\.name), ["g"])

        let required = try chat(#","tools":\#(both),"tool_choice":"required""#)
        XCTAssertEqual(required.toolChoice, .required)
        XCTAssertEqual(required.activeTools.map(\.name), ["f", "g"])

        for shape in noToolsShapes {
            XCTAssertEqual(try chat(shape + #","tool_choice":"auto""#).toolChoice, .auto, shape)
            XCTAssertEqual(
                try chat(shape + #","tool_choice":"none""#).toolChoice, OpenAIToolChoice.none, shape)
            XCTAssertEqual(
                try chat(shape + #","tool_choice":null"#).toolChoice, OpenAIToolChoice.none, shape)
        }
        XCTAssertEqual(try chat("").toolChoice, OpenAIToolChoice.none)
    }

    // MARK: - Named tool_choice + response_format (cycle 257)

    func testResponseFormatWithNamedToolChoiceIsRefusedOnToolChoiceParam() {
        for format in [
            #"{"type":"json_object"}"#,
            #"{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"#,
        ] {
            assertRefused(
                #","tools":[\#(tool("f"))],"tool_choice":\#(named("f")),"response_format":\#(format)"#,
                message: "response_format \(format.contains("json_object") ? "json_object" : "json_schema") "
                    + "is not supported together with a named tool_choice")
        }
    }

    func testResponseFormatWithRequiredToolChoiceIsRefusedOnToolChoiceParam() {
        for format in [
            #"{"type":"json_object"}"#,
            #"{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"#,
        ] {
            assertRefused(
                #","tools":[\#(tool("f"))],"tool_choice":"required","response_format":\#(format)"#,
                message: "response_format \(format.contains("json_object") ? "json_object" : "json_schema") "
                    + "is not supported together with tool_choice \"required\"")
        }
    }

    /// Control: any OTHER tool_choice keeps today's `tools`-param refusal.
    func testResponseFormatWithAutoToolsKeepsTheToolsParamRefusal() {
        let error = chatError(
            #","tools":[\#(tool("f"))],"tool_choice":"auto","response_format":{"type":"json_object"}"#)
        XCTAssertEqual(error?.param, "tools")
    }

    // MARK: - OpenAIToolSpec.forcedArgumentsFormat

    private func spec(_ toolJSON: String, at index: Int = 0) throws -> OpenAIToolSpec {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[\#(toolJSON)]}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        return request.tools[index]
    }

    private func objectPropertyNames(_ format: JSONSchemaResponseFormat) -> [String]? {
        guard case .object(let properties) = format.root else { return nil }
        return properties.map(\.name)
    }

    func testForcedArgumentsFormatKeepsTheClientsPropertyOrder() throws {
        let tool = try spec(
            #"{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}},"required":["zeta"]}}}"#)
        let format = try tool.forcedArgumentsFormat()
        XCTAssertEqual(objectPropertyNames(format), ["zeta", "alpha"])
    }

    func testForcedArgumentsFormatTreatsAbsentAndEmptyParametersAsAnEmptyObject() throws {
        for json in [
            #"{"type":"function","function":{"name":"f"}}"#,
            #"{"type":"function","function":{"name":"f","parameters":{}}}"#,
            #"{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}"#,
        ] {
            let format = try spec(json).forcedArgumentsFormat()
            XCTAssertEqual(objectPropertyNames(format), [], json)
        }
    }

    func testForcedArgumentsFormatRefusesUnsupportedSchemasOnToolChoice() throws {
        for json in [
            // keyword outside the subset
            #"{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string","pattern":"^x"}}}}}"#,
            // not an object root
            #"{"type":"function","function":{"name":"f","parameters":{"type":"string"}}}"#,
            // additionalProperties other than false
            #"{"type":"function","function":{"name":"f","parameters":{"type":"object","additionalProperties":true}}}"#,
        ] {
            let tool = try spec(json)
            XCTAssertThrowsError(try tool.forcedArgumentsFormat(), json) { error in
                guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                    return XCTFail("expected invalidRequest, got \(error)")
                }
                XCTAssertEqual(param, "tool_choice", json)
            }
        }
    }

    func testForcedArgumentsFormatRefusesWhenTheClientKeyOrderIsUnavailable() {
        let tool = OpenAIToolSpec(
            name: "f", description: nil, parameters: .object([:]), raw: .object([:]), orderedRaw: nil)
        XCTAssertThrowsError(try tool.forcedArgumentsFormat()) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "tool_choice")
        }
    }
}
