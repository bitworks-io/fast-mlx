import XCTest

@testable import ServingCore

/// `ReasoningPhaseToolCallGate` routes each detokenized chunk so the tool parser sees exactly the bytes
/// the client is shown as content: everything up to and including the first `</think>` is passthrough,
/// the rest goes to the parser. Pure value-type tests -- the backend call sites are covered in
/// SpikeServingAdaptersTests.
final class ReasoningPhaseToolCallGateTests: XCTestCase {
    private let fixture =
        "<think>a <tool_call>{x}</tool_call></think>\n<tool_call>{y}</tool_call>"

    // G1
    func testReasoningTextPassesThroughAndNothingReachesTheParser() {
        var gate = ReasoningPhaseToolCallGate()
        let first = gate.route("<think>draft <tool_call>{\"a\":1}</tool_call>")
        XCTAssertEqual(first.passthrough, "<think>draft <tool_call>{\"a\":1}</tool_call>")
        XCTAssertEqual(first.toParser, "")
        let second = gate.route(" more thinking")
        XCTAssertEqual(second.passthrough, " more thinking")
        XCTAssertEqual(second.toParser, "")
        XCTAssertEqual(gate.flush(), "")
    }

    // G2
    func testTextAfterTheClosingTagAndEveryLaterChunkGoToTheParser() {
        var gate = ReasoningPhaseToolCallGate()
        _ = gate.route("thinking")
        let closing = gate.route("</think>")
        XCTAssertEqual(closing.passthrough, "</think>")
        XCTAssertEqual(closing.toParser, "")
        let after = gate.route("<tool_call>{p}</tool_call>")
        XCTAssertEqual(after.passthrough, "")
        XCTAssertEqual(after.toParser, "<tool_call>{p}</tool_call>")
        // A second `</think>` is ordinary parser input, not a phase change.
        let again = gate.route("x</think>y")
        XCTAssertEqual(again.passthrough, "")
        XCTAssertEqual(again.toParser, "x</think>y")
    }

    // G3
    func testClosingTagSplitAcrossChunksIsHeldAndNeverTorn() {
        var gate = ReasoningPhaseToolCallGate()
        let first = gate.route("</th")
        XCTAssertEqual(first.passthrough, "")
        XCTAssertEqual(first.toParser, "")
        let second = gate.route("ink><tool_call>")
        XCTAssertEqual(second.passthrough, "</think>")
        XCTAssertEqual(second.toParser, "<tool_call>")
    }

    // G4
    func testSingleChunkIsSplitExactlyAtTheClosingTag() {
        var gate = ReasoningPhaseToolCallGate()
        let routed = gate.route("draft</think><tool_call>{\"n\":1}</tool_call>")
        XCTAssertEqual(routed.passthrough, "draft</think>")
        XCTAssertEqual(routed.toParser, "<tool_call>{\"n\":1}</tool_call>")
        XCTAssertEqual(gate.flush(), "")
    }

    // G5
    func testUnclosedStreamFlushesTheHeldTailAsReasoningAndIsIdempotent() {
        var gate = ReasoningPhaseToolCallGate()
        let routed = gate.route("still thinking</thi")
        XCTAssertEqual(routed.passthrough, "still thinking")
        XCTAssertEqual(routed.toParser, "")
        XCTAssertEqual(gate.flush(), "</thi")
        XCTAssertEqual(gate.flush(), "")
    }

    // G6
    func testEveryTwoWayChunkingConservesBytesAndParserSeesOnlyTextAfterTheFirstTag() {
        let characters = Array(fixture)
        let expectedParser = String(fixture.components(separatedBy: "</think>").dropFirst().joined(separator: "</think>"))
        XCTAssertEqual(expectedParser, "\n<tool_call>{y}</tool_call>")
        for cut in 0...characters.count {
            let chunks = [String(characters[..<cut]), String(characters[cut...])]
            var gate = ReasoningPhaseToolCallGate()
            var passthrough = ""
            var parser = ""
            for chunk in chunks {
                let routed = gate.route(chunk)
                passthrough += routed.passthrough
                parser += routed.toParser
            }
            let tail = gate.flush()
            passthrough += tail
            XCTAssertEqual(passthrough + parser, fixture, "cut \(cut)")
            XCTAssertEqual(parser, expectedParser, "cut \(cut)")
            XCTAssertTrue(passthrough.hasSuffix("</think>"), "cut \(cut)")
        }
    }

    // G7
    func testHeldPrefixThatIsNotTheTagIsReleasedAsPassthrough() {
        var gate = ReasoningPhaseToolCallGate()
        let first = gate.route("</th")
        XCTAssertEqual(first.passthrough, "")
        let second = gate.route("x")
        XCTAssertEqual(second.passthrough, "</thx")
        XCTAssertEqual(second.toParser, "")
        XCTAssertEqual(gate.flush(), "")
    }
}
