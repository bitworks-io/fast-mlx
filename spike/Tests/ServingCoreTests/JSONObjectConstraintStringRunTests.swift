import XCTest

@testable import ServingCore

/// Correctness proof for the json_object slice 1f MISS-PATH optimization: string-run STAY/EXIT
/// classification (`JSONObjectConstraintTrie.classifyStringRun(from:)`,
/// `JSONObjectConstraintTable`'s string-run cache). Compares the OPTIMIZED
/// `JSONObjectConstraintTable.allowedTokenIds(for:)` against an independent, naive per-token
/// reference (mirrors `JSONObjectConstraintDifferentialTests.referenceAllowedIds`, deliberately
/// duplicated here rather than shared — this file's owned write set does not include that test
/// file) at states reached deep inside a JSON string: key and value strings, depths 1...30 mixing
/// objects and arrays, just-after-escape, every `\uXXXX` hex-digit position, mid 2/3/4-byte UTF-8
/// continuation sequences, tokens that close a string then continue with `:`, `,`, `}`, `]`,
/// whitespace (including a line break), tokens that close then reopen another string, and tokens
/// that close the top-level object from inside a string (which must then accept only EOS).
final class JSONObjectConstraintStringRunTests: XCTestCase {
    // MARK: - Reference (naive per-token) decision — see
    // `JSONObjectConstraintDifferentialTests.referenceAllowedIds`'s doc comment for the rationale;
    // this is the same textbook definition, intentionally duplicated (not shared) per this file's
    // owned write set.
    private static func referenceAllowedIds(
        from automaton: JSONObjectAutomaton, classifications: [TokenByteClassification]
    ) -> Set<Int> {
        var allowed = Set<Int>()
        for (id, classification) in classifications.enumerated() {
            switch classification {
            case .eos:
                if automaton.isComplete { allowed.insert(id) }
            case .banned:
                continue
            case .bytes(let bytes):
                var probe = automaton
                var ok = true
                for byte in bytes {
                    guard probe.advance(byte: byte) else {
                        ok = false
                        break
                    }
                }
                if ok { allowed.insert(id) }
            }
        }
        return allowed
    }

    // MARK: - Synthetic vocab

    /// All 256 raw single bytes (guarantees any byte sequence can be exactly tokenized and gives
    /// the reference/optimized comparison full coverage) plus curated multi-byte tokens designed
    /// specifically to exercise the STAY/EXIT split: plain multi-byte string content (STAY through
    /// several `.normal` bytes), simple/`\u` escape sequences at every countdown position (STAY
    /// through `.escape`/`.unicodeEscape`), raw multi-byte UTF-8 sequences and split continuation
    /// tails (STAY through `.continuation`), and EXIT tokens that close a string then continue with
    /// every structural byte class (`:`, `,`, `}`, `]`, space, `\n`), including EXIT tokens that
    /// close a key string and reopen a new one, and EXIT tokens that close the top-level object.
    private enum StringRunVocab {
        static let classifications: [TokenByteClassification] = {
            var out: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }

            let curated: [[UInt8]] = [
                // Plain multi-byte content (STAY, `.normal` -> `.normal`).
                Array("ab".utf8), Array("abc".utf8), Array("hello".utf8), Array("  ".utf8),

                // Simple escapes (STAY, `.normal` -> `.escape` -> `.normal` within one token).
                [0x5C, 0x6E],  // \n
                [0x5C, 0x22],  // \"
                [0x5C, 0x5C],  // \\

                // `\u` unicode-escape progression (STAY through every countdown position).
                [0x5C, 0x75],  // \u            : normal -> unicodeEscape(4)
                Array("\\u0".utf8),  // \u0      : normal -> unicodeEscape(3)
                Array("\\u00".utf8),  // \u00    : normal -> unicodeEscape(2)
                Array("\\u00e".utf8),  // \u00e  : normal -> unicodeEscape(1)
                Array("\\u00e9".utf8),  // é: normal -> normal (full escape, one token)
                Array("00e9".utf8),  // 4 hex digits from unicodeEscape(4) -> normal
                Array("0e9".utf8),  // 3 hex digits from unicodeEscape(3) -> normal
                Array("e9".utf8),  // 2 hex digits from unicodeEscape(2) -> normal

                // Raw multi-byte UTF-8 (STAY through `.continuation`).
                [0xC2, 0xA9],  // © : full 2-byte sequence, normal -> normal
                [0xE2, 0x82, 0xAC],  // € : full 3-byte sequence, normal -> normal
                [0xF0, 0x9F, 0x98, 0x80],  // 😀 : full 4-byte sequence, normal -> normal
                [0xE4, 0xBD],  // split-CJK head: normal -> continuation(1, 0x80, 0xBF)
                [0xA0],  // split-CJK tail: already covered by the raw-byte set above
                [0x82, 0xAC],  // continuation tail pair completing a 3-byte sequence
                [0x9F, 0x98, 0x80],  // continuation tail triple completing a 4-byte sequence
                [0xE0, 0xA0],  // E0 lead + first (min 0xA0) continuation byte -> continuation(1,..)
                [0xED, 0x80],  // ED lead + first (max 0x9F) continuation byte -> continuation(1,..)

                // EXIT: closes the string, then every structural byte class.
                [0x22, 0x3A],  // ": closes a KEY string, then colon
                [0x22, 0x2C],  // ", closes a VALUE string, then comma
                [0x22, 0x7D],  // "} closes a VALUE string, then closes the enclosing object
                [0x22, 0x5D],  // "] closes a VALUE string, then closes the enclosing array
                [0x22, 0x20],  // " (space) closes, then structural whitespace
                [0x22, 0x0A],  // " (\n) closes, then a single line-break event
                [0x22, 0x7D, 0x7D],  // "}} closes a value string then TWO nested objects
                [0x22, 0x3A, 0x22],  // ":" closes a key string, colon, reopens a new string
                [0x22, 0x2C, 0x22],  // ", " closes a value string, comma, reopens a new string (key)
                Array("ab\"".utf8),  // STAY prefix ("ab") then EXIT (the closing quote)
                Array("ab\":".utf8),  // STAY prefix, EXIT, then a further structural byte
            ]
            for bytes in curated {
                out.append(.bytes(bytes))
            }
            out.append(.eos)
            return out
        }()
        static var eosId: Int { classifications.count - 1 }
    }

    private let table = JSONObjectConstraintTable(classifications: StringRunVocab.classifications)

    // MARK: - Prefix builders

    /// Nests `depth` alternating object/array containers (level 0 is always an object — the
    /// top-level value must be one) and lands at a point where a KEY string (if `asKey`) or a
    /// VALUE string (otherwise) has just been opened (the opening `"` byte is included). The
    /// innermost (`depth`-th) container is opened SEPARATELY from the `depth - 1` outer ones: for
    /// `asKey`, it must be an object and the key string is opened directly inside it (no dummy
    /// key/colon — this string IS the key); for a value string, the same dummy-key/colon dance the
    /// outer loop uses gets us to the value position for an object, or the array's first-element
    /// position needs nothing extra.
    private func nestedStringOpenPrefix(depth: Int, asKey: Bool, alternate: Bool) -> [UInt8] {
        precondition(depth >= 1)
        var out: [UInt8] = []
        let outerCount = depth - 1
        for level in 0..<outerCount {
            let useObject = level == 0 ? true : (alternate ? level % 2 == 0 : true)
            if useObject {
                out.append(0x7B)  // {
                out.append(contentsOf: Array(#""k""#.utf8))
                out.append(0x3A)  // :
            } else {
                out.append(0x5B)  // [
            }
        }
        // level 0 (whether it's an outer level above or this innermost one) is always an object.
        let lastIsObject = outerCount == 0 ? true : (alternate ? outerCount % 2 == 0 : true)
        if asKey {
            out.append(0x7B)  // must be an object to host a key
            out.append(0x22)  // opens the key string directly — no dummy key/colon first
        } else if lastIsObject {
            out.append(0x7B)
            out.append(contentsOf: Array(#""k""#.utf8))
            out.append(0x3A)
            out.append(0x22)
        } else {
            out.append(0x5B)
            out.append(0x22)
        }
        return out
    }

    /// Every "interesting" string-internal lexical sub-state, as a byte suffix appended right
    /// after an opening `"`: normal content, just-after-escape, each `\u` countdown position, and
    /// each of the 7 distinct UTF-8 continuation specs (see
    /// `JSONObjectAutomaton.utf8ContinuationSpec` — mirrored here by lead byte).
    private static let stringInternalSuffixes: [(name: String, bytes: [UInt8])] = [
        ("normal-empty", []),
        ("normal-content", Array("ab".utf8)),
        ("escape", [0x5C]),
        ("unicodeEscape4", [0x5C, 0x75]),
        ("unicodeEscape3", [0x5C, 0x75, 0x30]),
        ("unicodeEscape2", [0x5C, 0x75, 0x30, 0x30]),
        ("unicodeEscape1", [0x5C, 0x75, 0x30, 0x30, 0x65]),
        ("continuation-C2", [0xC2]),  // (1, 0x80, 0xBF)
        ("continuation-E0", [0xE0]),  // (2, 0xA0, 0xBF)
        ("continuation-E1", [0xE1]),  // (2, 0x80, 0xBF)
        ("continuation-ED", [0xED]),  // (2, 0x80, 0x9F)
        ("continuation-F0", [0xF0]),  // (3, 0x90, 0xBF)
        ("continuation-F1", [0xF1]),  // (3, 0x80, 0xBF)
        ("continuation-F4", [0xF4]),  // (3, 0x80, 0x8F)
        ("continuation-E1-mid", [0xE1, 0x80]),  // decremented to (1, 0x80, 0xBF)
        ("continuation-F1-mid1", [0xF1, 0x80]),  // decremented to (2, 0x80, 0xBF)
        ("continuation-F1-mid2", [0xF1, 0x80, 0x80]),  // decremented to (1, 0x80, 0xBF)
    ]

    /// Asserts the optimized path matches the reference at `prefix`'s resulting automaton state,
    /// with a descriptive failure message. Returns the automaton reached, for further chaining.
    @discardableResult
    private func assertMatchesReference(_ prefix: [UInt8], _ label: String, file: StaticString = #filePath, line: UInt = #line) throws -> JSONObjectAutomaton {
        var automaton = JSONObjectAutomaton()
        for byte in prefix {
            XCTAssertTrue(automaton.advance(byte: byte), "\(label): generator produced an invalid prefix \(prefix)", file: file, line: line)
        }
        let expected = Self.referenceAllowedIds(from: automaton, classifications: StringRunVocab.classifications)
        let actual = Set(try table.allowedTokenIds(for: automaton))
        XCTAssertEqual(
            actual, expected,
            "\(label): mismatch for prefix \(String(decoding: prefix, as: UTF8.self).debugDescription) "
                + "— missing: \(expected.subtracting(actual)), extra: \(actual.subtracting(expected))",
            file: file, line: line)
        return automaton
    }

    // MARK: - Depths 1...30, key and value strings, every lexical sub-state

    func testStringRunOptimizationMatchesReferenceAcrossDepthsAndSubstates() throws {
        var comparedStates = 0
        for depth in 1...30 {
            for alternate in [false, true] {
                // Value string at this depth.
                let valuePrefix = nestedStringOpenPrefix(depth: depth, asKey: false, alternate: alternate)
                for (name, suffix) in Self.stringInternalSuffixes {
                    try assertMatchesReference(
                        valuePrefix + suffix, "depth=\(depth) alternate=\(alternate) value \(name)")
                    comparedStates += 1
                }
                // Key string at this depth (opens a fresh object as the innermost container).
                let keyPrefix = nestedStringOpenPrefix(depth: depth, asKey: true, alternate: alternate)
                for (name, suffix) in Self.stringInternalSuffixes {
                    try assertMatchesReference(
                        keyPrefix + suffix, "depth=\(depth) alternate=\(alternate) key \(name)")
                    comparedStates += 1
                }
            }
        }
        XCTAssertGreaterThan(comparedStates, 1500, "sanity: depths 1...30 x substates must produce many compared states")
    }

    // MARK: - EXIT tokens that close then continue with every structural byte class

    func testExitTokensAreCheckedAgainstRealContextNotAlwaysValid() throws {
        // Inside a KEY string (depth 1): only ": (close-then-colon) is a valid EXIT token here —
        // ", / "} / "] must be rejected, since a key must be followed by a colon, not a
        // comma/close.
        let keyAutomaton = try assertMatchesReference(Array(#"{""#.utf8), "key-open depth1")
        let keyAllowed = Set(try table.allowedTokenIds(for: keyAutomaton))
        let quoteColonId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x3A]) = $0 { return true }; return false }!
        let quoteCommaId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x2C]) = $0 { return true }; return false }!
        let quoteCloseBraceId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x7D]) = $0 { return true }; return false }!
        XCTAssertTrue(keyAllowed.contains(quoteColonId), "\": must be valid right after a key string opens")
        XCTAssertFalse(keyAllowed.contains(quoteCommaId), "\", must be rejected after a KEY string (needs a colon, not a comma)")
        XCTAssertFalse(keyAllowed.contains(quoteCloseBraceId), "\"} must be rejected after a KEY string")

        // Inside a VALUE string that is the object's only member (depth 1): "} (close-then-close-
        // object) and ", (close-then-comma, if more members could follow — here it's the only
        // member so ", is invalid unless followed validly) are checked against the SAME two-byte
        // token set; only "} is valid at "only member" depth-1 object closing to completion.
        let valueAutomaton = try assertMatchesReference(Array(#"{"a":""#.utf8), "value-open depth1")
        let valueAllowed = Set(try table.allowedTokenIds(for: valueAutomaton))
        XCTAssertTrue(valueAllowed.contains(quoteCloseBraceId), "\"} must be valid right after a value string opens (closes the object)")
        XCTAssertTrue(valueAllowed.contains(quoteCommaId), "\", must be valid (starts another key)")
        XCTAssertFalse(valueAllowed.contains(quoteColonId), "\": must be rejected after a VALUE string (a colon cannot follow a value)")
    }

    // MARK: - EXIT token that closes then reopens another string

    func testExitTokenClosingThenReopeningAnotherStringIsValidAtKeyPosition() throws {
        // `":"` (0x22,0x3A,0x22) closes a key string, consumes the colon, and reopens a value
        // string — valid right after a key string opens (same state as the previous test's key
        // case), invalid after a value string opens (no colon expected there).
        let keyAutomaton = try assertMatchesReference(Array(#"{""#.utf8), "key-open for reopen check")
        let reopenId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x3A, 0x22]) = $0 { return true }; return false }!
        XCTAssertTrue(Set(try table.allowedTokenIds(for: keyAutomaton)).contains(reopenId))

        let valueAutomaton = try assertMatchesReference(Array(#"{"a":""#.utf8), "value-open for reopen check")
        XCTAssertFalse(Set(try table.allowedTokenIds(for: valueAutomaton)).contains(reopenId))

        // `", "` (0x22,0x2C,0x22) closes a value string, consumes the comma, and reopens a new
        // (key) string — valid after a value string opens, invalid after a key string opens.
        let commaReopenId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x2C, 0x22]) = $0 { return true }; return false }!
        XCTAssertTrue(Set(try table.allowedTokenIds(for: valueAutomaton)).contains(commaReopenId))
        XCTAssertFalse(Set(try table.allowedTokenIds(for: keyAutomaton)).contains(commaReopenId))
    }

    // MARK: - EXIT token that closes the top-level object from inside a string

    func testExitTokenClosingTopLevelObjectMustThenAcceptOnlyEOS() throws {
        // `{"a":"v` is inside a value string at depth 1, about to close both the string and the
        // top-level object via the two-byte token "}.
        let prefix = Array(#"{"a":"v"#.utf8)
        let automaton = try assertMatchesReference(prefix, "about to close top-level object")
        let quoteCloseBraceId = StringRunVocab.classifications.firstIndex { if case .bytes([0x22, 0x7D]) = $0 { return true }; return false }!
        XCTAssertTrue(
            Set(try table.allowedTokenIds(for: automaton)).contains(quoteCloseBraceId),
            "\"} must be allowed: it validly closes the string and the top-level object")

        // Replay through "} for real and confirm the resulting state accepts ONLY EOS.
        var closed = automaton
        XCTAssertTrue(closed.advance(byte: 0x22))
        XCTAssertTrue(closed.advance(byte: 0x7D))
        XCTAssertTrue(closed.isComplete)
        let expectedAfterClose = Self.referenceAllowedIds(from: closed, classifications: StringRunVocab.classifications)
        let actualAfterClose = Set(try table.allowedTokenIds(for: closed))
        XCTAssertEqual(actualAfterClose, expectedAfterClose)
        XCTAssertEqual(actualAfterClose, [StringRunVocab.eosId], "after closing the top-level object, only EOS may follow")
    }

    // MARK: - Sanity: the string-run cache is actually bounded (~13 lexical sub-states max)

    func testStringRunCacheStaysBoundedAcrossManyDistinctStates() throws {
        // Drive through every depth/substate combination from the main sweep above, then confirm
        // the lazily-populated string-run cache never grew past the ~13 reachable lexical
        // sub-states, regardless of how many distinct (deep, key-vs-value) MaskCacheKey states were
        // queried.
        for depth in [1, 5, 12, 30] {
            for asKey in [false, true] {
                let prefix = nestedStringOpenPrefix(depth: depth, asKey: asKey, alternate: depth % 2 == 0)
                for (_, suffix) in Self.stringInternalSuffixes {
                    _ = try assertMatchesReference(prefix + suffix, "cache-bound depth=\(depth) asKey=\(asKey)")
                }
            }
        }
        XCTAssertLessThanOrEqual(table.stringRunCacheEntryCount, 13)
        XCTAssertGreaterThan(table.stringRunCacheEntryCount, 0)
        XCTAssertGreaterThan(table.stringRunCacheBytes, 0)
    }
}
