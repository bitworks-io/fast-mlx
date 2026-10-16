import XCTest

import MLXLMCommon
@testable import SpikeServingAdapters

/// Cycle 254: an out-of-range / non-finite / hex number in a schema-`number` tool argument must not
/// trap the process (`Int(Double)` traps outside `Int`'s range and on inf/nan), must not make the
/// adapter replace ALL of the call's arguments with "{}" (non-finite Doubles fail
/// `JSONSerialization.isValidJSONObject`), and must not turn a hex lexeme into a number (Python
/// `float()` refuses `0x10`; Swift `Double("0x10")` is 16).
///
/// Drives the real `ToolCallProcessor` (both number-conversion sites: `convertParameterValue` for
/// `.xmlFunction`, `convertValueWithTypes` for `.minimaxM2`) and the adapter's `openAIToolCalls`.
/// Pure CPU; no Metal runtime.
final class ToolCallNumberArgumentTests: XCTestCase {
    private static let tools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "pay",
                "description": "Pay an amount",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "amount": ["type": "number"] as [String: any Sendable],
                        "note": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    private static func xmlText(amount: String) -> String {
        "<tool_call>\n<function=pay>\n<parameter=amount>\n\(amount)\n</parameter>\n"
            + "<parameter=note>\nx\n</parameter>\n</function>\n</tool_call>"
    }

    private static func minimaxText(amount: String) -> String {
        "<minimax:tool_call>\n<invoke name=\"pay\">\n<parameter name=\"amount\">\(amount)</parameter>\n"
            + "<parameter name=\"note\">x</parameter>\n</invoke>\n</minimax:tool_call>"
    }

    /// Feed `text` (in one chunk) to a real processor and return the adapter's arguments JSON.
    private func arguments(format: ToolCallFormat, text: String) throws -> String {
        let processor = ToolCallProcessor(format: format, tools: Self.tools)
        _ = processor.processChunk(text)
        processor.processEOS()
        XCTAssertEqual(processor.toolCalls.count, 1, "expected one parsed call for \(text)")
        let calls = openAIToolCalls(from: processor.toolCalls)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.function.name, "pay")
        return call.function.arguments
    }

    private func check(
        _ amount: String, expectAmount: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let xml = try arguments(format: .xmlFunction, text: Self.xmlText(amount: amount))
        XCTAssertEqual(
            xml, "{\"amount\":\(expectAmount),\"note\":\"x\"}", "xmlFunction \(amount)",
            file: file, line: line)
        let mm = try arguments(format: .minimaxM2, text: Self.minimaxText(amount: amount))
        XCTAssertEqual(
            mm, "{\"amount\":\(expectAmount),\"note\":\"x\"}", "minimaxM2 \(amount)",
            file: file, line: line)
    }

    // N1 / N5
    func testOutOfIntRangePositiveDoesNotTrapAndKeepsDouble() throws {
        try check("1e20", expectAmount: "1e+20")
    }

    // N2 / N6
    func testOutOfIntRangeNegativeDoesNotTrapAndKeepsDouble() throws {
        try check("-1e19", expectAmount: "-1e+19")
    }

    // N3 / N7: non-finite stays the raw string and the sibling argument survives (no "{}").
    func testNonFiniteStaysRawStringAndSiblingArgumentSurvives() throws {
        try check("inf", expectAmount: "\"inf\"")
        try check("nan", expectAmount: "\"nan\"")
        try check("Infinity", expectAmount: "\"Infinity\"")
    }

    // N4 / N8: hex is not a number.
    func testHexLexemeIsNotANumber() throws {
        try check("0x10", expectAmount: "\"0x10\"")
        try check("-0x10", expectAmount: "\"-0x10\"")
        try check("0X1F", expectAmount: "\"0X1F\"")
    }

    // Controls: in-range behaviour is unchanged (green on the parent too).
    func testInRangeControlsAreUnchanged() throws {
        try check("2.5", expectAmount: "2.5")
        try check("1.0", expectAmount: "1")
        try check("42", expectAmount: "42")
        try check("9.2e18", expectAmount: "9200000000000000000")
    }
}
