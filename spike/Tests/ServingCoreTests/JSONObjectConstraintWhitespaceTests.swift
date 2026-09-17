import XCTest

@testable import ServingCore

/// Defect B (json_object slice 1f): live runs showed small models emitting large indentation runs
/// (`"                {}"`, 16 newlines before the object) because the grammar accepted ANY
/// whitespace at every structural position, including before the top-level `{` and after the
/// top-level object closed, capped only at 16 CONSECUTIVE bytes regardless of composition. This
/// file proves the tightened grammar: (1) no whitespace is ever allowed before the first byte or
/// after the last byte of the document, and (2) a structural whitespace run may still be up to 16
/// bytes (unchanged), but may contain at most ONE line-break EVENT — `\n`, `\r`, or an immediately
/// paired `\r\n` all count as exactly one event; pretty-printed JSON (one newline, then spaces of
/// indentation) must still be accepted, since that is a single event per run.
final class JSONObjectConstraintWhitespaceTests: XCTestCase {
    // MARK: - No whitespace before the top-level `{`

    func testRejectsLeadingSpaceBeforeOpenBrace() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(" {}"))
    }

    func testRejectsLeadingTabBeforeOpenBrace() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("\t{}"))
    }

    func testRejectsLeadingNewlineBeforeOpenBrace() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("\n{}"))
    }

    func testAcceptsOpenBraceAsFirstByte() {
        // `{` alone, right at the initial state, must still be a valid first byte.
        var automaton = JSONObjectAutomaton()
        XCTAssertTrue(automaton.advance(byte: UInt8(ascii: "{")))
    }

    // MARK: - No whitespace after the top-level object closes

    func testRejectsTrailingSpaceAfterCloseBrace() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("{} "))
    }

    func testRejectsTrailingNewlineAfterCloseBrace() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("{}\n"))
    }

    func testEOSAllowedImmediatelyAfterCloseBrace() {
        var automaton = JSONObjectAutomaton()
        for byte in Array("{}".utf8) {
            XCTAssertTrue(automaton.advance(byte: byte))
        }
        XCTAssertTrue(automaton.isComplete, "EOS must remain allowed once the top-level object closes")
    }

    // MARK: - Pretty-printed JSON (one line break per run) is still accepted

    func testAcceptsPrettyPrintedNestedObjectAndArray() {
        let json = "{\n  \"a\": 1,\n  \"b\": [\n    1\n  ]\n}"
        XCTAssertTrue(JSONObjectAutomaton.accepts(json))
    }

    // MARK: - At most one line-break event per structural whitespace run

    func testRejectsTwoConsecutiveNewlinesInOneRun() {
        // The SECOND `\n` is the byte that must be rejected — verified precisely below, not just
        // via whole-string `accepts`.
        var automaton = JSONObjectAutomaton()
        for byte in Array("{\n".utf8) {
            XCTAssertTrue(automaton.advance(byte: byte))
        }
        XCTAssertFalse(automaton.advance(byte: UInt8(ascii: "\n")), "a second line break in the same run must be rejected")
        XCTAssertFalse(JSONObjectAutomaton.accepts("{\n\n\"a\":1}"))
    }

    func testAcceptsCRLFPairAsOneLineBreakEvent() {
        XCTAssertTrue(JSONObjectAutomaton.accepts("{\r\n  \"a\":1}"))
    }

    func testRejectsTwoConsecutiveBareCarriageReturns() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("{\r\r\"a\":1}"))
    }

    func testRejectsCRThenLFThenAnotherLF() {
        // `\r\n` merges into ONE event; a THIRD line-break byte in the same run is a second event.
        XCTAssertFalse(JSONObjectAutomaton.accepts("{\r\n\n\"a\":1}"))
    }

    func testRejectsBareCRFollowedByLF3TimesEquivalent() {
        // Sanity: a lone `\r` not immediately followed by `\n` counts as its own event, so a run
        // with `\r` then unrelated whitespace then `\n` is two events.
        XCTAssertFalse(JSONObjectAutomaton.accepts("{\r \n\"a\":1}"))
    }

    func testAcceptsSingleBareCRAsWhitespace() {
        XCTAssertTrue(JSONObjectAutomaton.accepts("{\r\"a\":1}"))
    }

    // MARK: - The byte cap (16 consecutive whitespace bytes) is unchanged

    func testAcceptsWhitespaceRunAtCapInsideObject() {
        let json = "{" + String(repeating: " ", count: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace)
            + "\"a\":1}"
        XCTAssertTrue(JSONObjectAutomaton.accepts(json))
    }

    func testRejectsWhitespaceRunOverCapInsideObject() {
        let json = "{"
            + String(repeating: " ", count: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace + 1)
            + "\"a\":1}"
        XCTAssertFalse(JSONObjectAutomaton.accepts(json))
    }

    func testAcceptsCapBytesIncludingOneLeadingNewline() {
        // A pretty-printer's indentation run: one newline followed by up to (cap - 1) spaces must
        // still be accepted at the cap.
        let json = "{\n" + String(repeating: " ", count: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace - 1)
            + "\"a\":1}"
        XCTAssertTrue(JSONObjectAutomaton.accepts(json))
    }
}
