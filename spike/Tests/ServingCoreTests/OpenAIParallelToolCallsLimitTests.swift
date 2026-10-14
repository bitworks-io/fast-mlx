import XCTest

@testable import ServingCore

/// `parallel_tool_calls: false` caps a reply at ONE tool call (the first, unchanged); absent, `null`
/// and `true` keep every call. Pure helper tests -- the HTTP call sites are covered in ServingNIOTests.
final class OpenAIParallelToolCallsLimitTests: XCTestCase {
    private func request(parallel: String?) throws -> OpenAIChatCompletionRequest {
        let field = parallel.map { #","parallel_tool_calls":\#($0)"# } ?? ""
        let body =
            #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]"#
            + field + "}"
        return try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
    }

    private let calls = [
        OpenAIToolCall(id: "call_0", function: .init(name: "get_weather", arguments: #"{"city":"Paris"}"#)),
        OpenAIToolCall(id: "call_1", function: .init(name: "get_time", arguments: #"{"city":"Rome"}"#)),
    ]

    func testFalseKeepsOnlyTheFirstCallUnchanged() throws {
        let limited = try request(parallel: "false").limitingToolCalls(calls, alreadyEmitted: 0)
        XCTAssertEqual(limited.count, 1)
        XCTAssertEqual(limited.first?.id, "call_0")
        XCTAssertEqual(limited.first?.function.name, "get_weather")
        XCTAssertEqual(limited.first?.function.arguments, #"{"city":"Paris"}"#)
        XCTAssertEqual(limited, [calls[0]])
    }

    func testFalseIsCumulativeAcrossEvents() throws {
        let falseRequest = try request(parallel: "false")
        XCTAssertEqual(falseRequest.limitingToolCalls(calls, alreadyEmitted: 1), [])
        XCTAssertEqual(falseRequest.limitingToolCalls([calls[1]], alreadyEmitted: 1), [])
        XCTAssertEqual(falseRequest.limitingToolCalls(calls, alreadyEmitted: 5), [])
    }

    func testAbsentNullAndTrueKeepEveryCall() throws {
        for parallel in [nil, "null", "true"] {
            let parsed = try request(parallel: parallel)
            XCTAssertEqual(
                parsed.limitingToolCalls(calls, alreadyEmitted: 0), calls, "parallel=\(parallel ?? "absent")")
            XCTAssertEqual(
                parsed.limitingToolCalls(calls, alreadyEmitted: 3), calls, "parallel=\(parallel ?? "absent")")
        }
    }

    func testFalseLeavesZeroOrOneCallUnchanged() throws {
        let falseRequest = try request(parallel: "false")
        XCTAssertEqual(falseRequest.limitingToolCalls([], alreadyEmitted: 0), [])
        XCTAssertEqual(falseRequest.limitingToolCalls([calls[1]], alreadyEmitted: 0), [calls[1]])
    }

    func testNullDecodesAsNilAndFalseAsFalse() throws {
        XCTAssertNil(try request(parallel: "null").parallelToolCalls)
        XCTAssertNil(try request(parallel: nil).parallelToolCalls)
        XCTAssertEqual(try request(parallel: "false").parallelToolCalls, false)
    }

    func testNonBooleanParallelToolCallsIsStillAnInvalidRequestNamingTheField() {
        XCTAssertThrowsError(try request(parallel: #""no""#)) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "parallel_tool_calls")
        }
    }
}
