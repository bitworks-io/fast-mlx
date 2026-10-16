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
}
