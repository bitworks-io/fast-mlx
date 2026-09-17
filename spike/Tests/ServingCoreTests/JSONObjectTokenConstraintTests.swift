import XCTest

@testable import ServingCore

final class JSONObjectTokenConstraintTests: XCTestCase {
    // MARK: - Automaton accept/reject table

    func testAcceptsEmptyObject() {
        XCTAssertTrue(JSONObjectAutomaton.accepts("{}"))
    }

    func testAcceptsSimpleKeyValue() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":1}"#))
    }

    func testAcceptsNegativeZero() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":-0}"#))
    }

    func testAcceptsExponentWithExplicitPlus() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":1e+5}"#))
    }

    func testAcceptsFractionalExponent() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":1.5e-10}"#))
    }

    func testAcceptsLiterals() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":true,"b":false,"c":null}"#))
    }

    func testAcceptsNestedArrayValue() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":[1,2,3]}"#))
    }

    func testAcceptsNestedObjects() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":{"b":{"c":1}}}"#))
    }

    func testAcceptsUnicodeEscape() {
        // A real `\uXXXX` JSON escape (é = U+00E9), not a raw UTF-8 continuation byte — that case
        // is covered separately by `testAcceptsRawMultibyteUTF8` below.
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":"café"}"#))
    }

    func testAcceptsSurrogatePairEscape() {
        // 😀 (U+1F600) as a JSON surrogate-pair escape: high surrogate \ud83d, low surrogate \ude00.
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":"😀"}"#))
    }

    func testRejectsInvalidEscapeCharacter() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":"\x"}"#))
    }

    func testRejectsInvalidUnicodeEscapeHexDigit() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":"\u12G4"}"#))
    }

    func testRejectsTrailingCommaInObject() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":1,}"#))
    }

    func testRejectsTrailingCommaInArray() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":[1,]}"#))
    }

    func testRejectsMissingColonAfterKey() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a"}"#))
    }

    func testRejectsTruncatedLiteral() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":tru"#))
    }

    func testRejectsTruncatedNumberExponent() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":1e"#))
    }

    func testRejectsTruncatedNegativeNumber() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":-"#))
    }

    func testRejectsContentAfterTopLevelObjectCloses() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("{}{}"))
    }

    func testAcceptsRawMultibyteUTF8() {
        XCTAssertTrue(JSONObjectAutomaton.accepts(#"{"a":"café"}"#))
    }

    func testAcceptsInteriorWhitespace() {
        // Updated for defect B (json_object slice 1f): whitespace is no longer accepted BEFORE the
        // top-level `{` or AFTER the top-level `}` closes — see `JSONObjectConstraintWhitespaceTests`
        // — but interior structural whitespace (around a key, colon, value, and before the close)
        // is unaffected.
        XCTAssertTrue(JSONObjectAutomaton.accepts("{  \"a\" : 1  }"))
    }

    func testAcceptsDepth64Nesting() {
        // Top-level object is depth 1; 63 nested arrays reach exactly maxDepth (64).
        let json = "{\"a\":" + String(repeating: "[", count: 63) + "1"
            + String(repeating: "]", count: 63) + "}"
        XCTAssertTrue(JSONObjectAutomaton.accepts(json))
    }

    func testRejectsDepth65Nesting() {
        // One level past maxDepth must be rejected — the discriminating control for the above.
        let json = "{\"a\":" + String(repeating: "[", count: 64) + "1"
            + String(repeating: "]", count: 64) + "}"
        XCTAssertFalse(JSONObjectAutomaton.accepts(json))
    }

    func testRejectsTopLevelArray() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("[1,2,3]"))
    }

    func testRejectsTrailingGarbage() {
        XCTAssertFalse(JSONObjectAutomaton.accepts("{}x"))
    }

    func testRejectsLeadingZeroFollowedByDigit() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":01}"#))
    }

    func testRejectsTrailingDotWithNoFractionDigit() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":1.}"#))
    }

    func testRejectsNumberStartingWithDot() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":.5}"#))
    }

    func testRejectsNumberWithLeadingPlus() {
        XCTAssertFalse(JSONObjectAutomaton.accepts(#"{"a":+1}"#))
    }

    func testRejectsUnescapedControlCharacterInString() {
        // A literal, unescaped newline byte inside a JSON string is invalid per RFC 8259.
        XCTAssertFalse(JSONObjectAutomaton.accepts("{\"a\":\"line1\nline2\"}"))
    }

    func testRejectsLoneInvalidContinuationByte() {
        let bytes = Array("{\"a\":\"".utf8) + [0xA0] + Array("\"}".utf8)
        XCTAssertFalse(JSONObjectAutomaton.accepts(bytes: bytes))
    }

    func testRejectsOverlongTwoByteLeadByte() {
        // 0xC0 is never a valid UTF-8 lead byte (would encode an overlong sequence).
        let bytes = Array("{\"a\":\"".utf8) + [0xC0, 0x80] + Array("\"}".utf8)
        XCTAssertFalse(JSONObjectAutomaton.accepts(bytes: bytes))
    }

    func testRejectsSurrogateRangeThreeByteSequence() {
        // 0xED followed by a byte in 0xA0...0xBF would encode a UTF-16 surrogate; RFC 3629
        // (and Unicode well-formedness) excludes this even though the raw bytes look 3-byte-shaped.
        let bytes = Array("{\"a\":\"".utf8) + [0xED, 0xA0, 0x80] + Array("\"}".utf8)
        XCTAssertFalse(JSONObjectAutomaton.accepts(bytes: bytes))
    }

    func testAcceptsValidThreeByteSequenceInSurrogateLeadByte() {
        // Discriminating control for the surrogate-exclusion test above: 0xED followed by a byte
        // in the VALID 0x80...0x9F range must still be accepted.
        let bytes = Array("{\"a\":\"".utf8) + [0xED, 0x80, 0x80] + Array("\"}".utf8)
        XCTAssertTrue(JSONObjectAutomaton.accepts(bytes: bytes))
    }

    func testAcceptsWhitespaceRunAtCap() {
        let json = "{" + String(repeating: " ", count: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace)
            + "\"a\":1}"
        XCTAssertTrue(JSONObjectAutomaton.accepts(json))
    }

    func testRejectsWhitespaceRunOverCap() {
        let json = "{"
            + String(repeating: " ", count: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace + 1)
            + "\"a\":1}"
        XCTAssertFalse(JSONObjectAutomaton.accepts(json))
    }

    // MARK: - Split-CJK fixture (byte-recovery mutation test)

    /// A tiny vocab where "你" (UTF-8 E4 BD A0) is split across two byte-level tokens: a 2-byte
    /// head (E4 BD) and a 1-byte tail (A0). Real byte-level BPE vocabs routinely split multi-byte
    /// characters like this across token boundaries.
    private enum SplitCJKVocab {
        static let openBrace = 0
        static let closeBrace = 1
        static let quote = 2
        static let colon = 3
        static let keyChar = 4
        static let head = 5  // bytes [0xE4, 0xBD]
        static let tail = 6  // bytes [0xA0]
        static let eos = 7

        static let classifications: [TokenByteClassification] = [
            .bytes([0x7B]),  // {
            .bytes([0x7D]),  // }
            .bytes([0x22]),  // "
            .bytes([0x3A]),  // :
            .bytes([0x6B]),  // k
            .bytes([0xE4, 0xBD]),
            .bytes([0xA0]),
            .eos,
        ]
    }

    func testSplitCJKTailIsOnlyAllowedAfterHead() throws {
        let table = JSONObjectConstraintTable(classifications: SplitCJKVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)

        // Drive into an open string value: {"k":"
        for token in [
            SplitCJKVocab.openBrace, SplitCJKVocab.quote, SplitCJKVocab.keyChar, SplitCJKVocab.quote,
            SplitCJKVocab.colon, SplitCJKVocab.quote,
        ] {
            try constraint.advance(token: token)
        }

        var allowed = try constraint.allowedTokenIds()
        XCTAssertTrue(allowed.contains(SplitCJKVocab.head), "head should be allowed to start the char")
        XCTAssertFalse(allowed.contains(SplitCJKVocab.tail), "tail alone is an invalid lead byte here")

        try constraint.advance(token: SplitCJKVocab.head)

        allowed = try constraint.allowedTokenIds()
        XCTAssertTrue(allowed.contains(SplitCJKVocab.tail), "tail should be allowed right after the head")
        XCTAssertFalse(allowed.contains(SplitCJKVocab.head), "head is not a valid continuation of itself here")

        try constraint.advance(token: SplitCJKVocab.tail)
        try constraint.advance(token: SplitCJKVocab.quote)
        try constraint.advance(token: SplitCJKVocab.closeBrace)
        XCTAssertTrue(constraint.isComplete)
    }

    func testSplitCJKThroughRealClassifyAndVocabStringInversion() throws {
        // The REAL byte-level vocab strings a tokenizer.json would store for a two-token split of
        // "你" (U+4F60, UTF-8 E4 BD A0): a head token covering [0xE4, 0xBD] and a tail token
        // covering [0xA0], each expressed as the GPT-2 byte-level glyph string (via
        // `ByteLevelTokenBytes.byteToUnicodeScalar`), routed through `classify` and
        // `bytes(forVocabString:)` — not hand-built `.bytes([...])` literals.
        func vocabString(forBytes bytes: [UInt8]) -> String {
            String(String.UnicodeScalarView(bytes.map { ByteLevelTokenBytes.byteToUnicodeScalar[$0]! }))
        }
        let headVocabString = vocabString(forBytes: [0xE4, 0xBD])
        let tailVocabString = vocabString(forBytes: [0xA0])

        enum Ids {
            static let openBrace = 0, closeBrace = 1, quote = 2, colon = 3, keyChar = 4
            static let head = 5, tail = 6, eos = 7
        }
        let vocabStrings: [Int: String] = [
            Ids.openBrace: "{", Ids.closeBrace: "}", Ids.quote: "\"", Ids.colon: ":",
            Ids.keyChar: "k", Ids.head: headVocabString, Ids.tail: tailVocabString,
        ]
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 8,
            vocabString: { vocabStrings[$0] },
            addedTokenIds: [],
            eosTokenIds: [Ids.eos])

        guard case .bytes(let headBytes) = classifications[Ids.head],
            case .bytes(let tailBytes) = classifications[Ids.tail]
        else {
            return XCTFail("expected both split-CJK halves to classify as .bytes")
        }
        XCTAssertEqual(headBytes, [0xE4, 0xBD])
        XCTAssertEqual(tailBytes, [0xA0])

        let table = JSONObjectConstraintTable(classifications: classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        for token in [Ids.openBrace, Ids.quote, Ids.keyChar, Ids.quote, Ids.colon, Ids.quote] {
            try constraint.advance(token: token)
        }
        try constraint.advance(token: Ids.head)
        try constraint.advance(token: Ids.tail)
        try constraint.advance(token: Ids.quote)
        try constraint.advance(token: Ids.closeBrace)
        XCTAssertTrue(constraint.isComplete)
        XCTAssertEqual(String(decoding: headBytes + tailBytes, as: UTF8.self), "你")

        // MUTATION CONTROL: a deliberately wrong recovery that treats the vocab string's own
        // UTF-8 bytes as the token's bytes (instead of inverting the byte-level glyph table)
        // produces a DIFFERENT byte sequence — and decodes to different content, not "你".
        let mutatedHeadBytes = Array(headVocabString.utf8)
        XCTAssertNotEqual(mutatedHeadBytes, headBytes, "the mutation must actually change the bytes")
        let mutatedTailBytes = Array(tailVocabString.utf8)
        XCTAssertNotEqual(
            String(decoding: mutatedHeadBytes + mutatedTailBytes, as: UTF8.self), "你",
            "the wrong recovery must not coincidentally decode to the same character")
    }

    func testSelfCheckMutationMirrorCatchesWrongByteRecovery() {
        // Mirrors the byte-recovery mutation: if the head token's bytes were (incorrectly) derived
        // by re-encoding a naive `decode([id])`-produced U+FFFD instead of the true byte inversion,
        // the self-check must catch it as soon as a decode oracle disagrees. Here the "mutated"
        // classification holds U+FFFD's own UTF-8 bytes (EF BF BD) for a token whose real decode is
        // the correct, complete "你" — proving the self-check discriminates a broken recovery from
        // a correct one (see `testSelfCheckSkipsIdsWhoseBytesAreNotStandaloneValidUTF8` in
        // `ByteLevelTokenBytesTests` for the companion "correctly stays quiet" case).
        let mutatedClassifications: [TokenByteClassification] = [.bytes([0xEF, 0xBF, 0xBD])]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: mutatedClassifications,
            decode: { _ in "你" })
        XCTAssertEqual(mismatches.count, 1)
        XCTAssertEqual(mismatches[0].recoveredUTF8, "\u{FFFD}")
        XCTAssertEqual(mismatches[0].decoded, "你")
    }

    // MARK: - Adversarial decoding simulation

    /// A tiny vocab covering every JSON-relevant single byte plus a few multi-byte tokens
    /// (`{"`, `":`, `"}`), an EOS id, and a banned added token (`<|im_end|>`).
    private enum AdversarialVocab {
        static let openBrace = 0
        static let closeBrace = 1
        static let quote = 2
        static let colon = 3
        static let comma = 4
        static let space = 5
        static let a = 6
        static let b = 7
        static let one = 8
        static let openBraceQuote = 9  // {"
        static let quoteColon = 10  // ":
        static let quoteCloseBrace = 11  // "}
        static let eos = 12
        static let banned = 13

        static let classifications: [TokenByteClassification] = [
            .bytes([0x7B]),
            .bytes([0x7D]),
            .bytes([0x22]),
            .bytes([0x3A]),
            .bytes([0x2C]),
            .bytes([0x20]),
            .bytes([0x61]),
            .bytes([0x62]),
            .bytes([0x31]),
            .bytes([0x7B, 0x22]),
            .bytes([0x22, 0x3A]),
            .bytes([0x22, 0x7D]),
            .eos,
            .banned,
        ]

        /// EOS first, then the banned token, then `}`, then everything else in ascending id
        /// order — an adversarial scorer that always tries to end generation (or cheat past the
        /// grammar) before anything else.
        static let priority: [Int] = [eos, banned, closeBrace, openBrace, quote, colon, comma, space,
                                       a, b, one, openBraceQuote, quoteColon, quoteCloseBrace]
    }

    /// Greedily decodes by walking `priority`, at each step picking the first id that's a member
    /// of `allowed()`, until it picks `eosId` or hits `maxSteps`. Returns the emitted token ids
    /// (never including a rejected pick — `allowed` fully determines what CAN be picked).
    private func greedyDecode(
        priority: [Int], eosId: Int, maxSteps: Int, allowed: () -> [Int]
    ) -> [Int] {
        var emitted: [Int] = []
        for _ in 0..<maxSteps {
            let allowedSet = Set(allowed())
            guard let pick = priority.first(where: { allowedSet.contains($0) }) else {
                break
            }
            emitted.append(pick)
            if pick == eosId { break }
        }
        return emitted
    }

    /// Like `greedyDecode`, but for a real constraint: advances it by each picked (non-EOS) id
    /// as it goes, so `allowedTokenIds()` reflects the automaton's actual current position at
    /// every step (a constraint's state does not change on its own between calls).
    private func greedyDecodeConstrained(
        priority: [Int], eosId: Int, maxSteps: Int, constraint: inout JSONObjectTokenConstraint
    ) throws -> [Int] {
        var emitted: [Int] = []
        for _ in 0..<maxSteps {
            let allowedSet = Set(try constraint.allowedTokenIds())
            guard let pick = priority.first(where: { allowedSet.contains($0) }) else { break }
            emitted.append(pick)
            if pick == eosId { break }
            try constraint.advance(token: pick)
        }
        return emitted
    }

    private func text(for ids: [Int], classifications: [TokenByteClassification]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if case .bytes(let tokenBytes) = classifications[id] {
                bytes.append(contentsOf: tokenBytes)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    func testAdversarialScorerControlProducesInvalidJSONWithoutConstraint() {
        // CONTROL: with no constraint, "allowed" is unconstrained (the whole vocab, always), so
        // the scorer immediately picks its top preference, EOS, producing empty (invalid) output.
        let ids = greedyDecode(
            priority: AdversarialVocab.priority, eosId: AdversarialVocab.eos, maxSteps: 50,
            allowed: { Array(0..<AdversarialVocab.classifications.count) })
        let output = text(for: ids, classifications: AdversarialVocab.classifications)
        let data = Data(output.utf8)
        XCTAssertThrowsError(try JSONSerialization.jsonObject(with: data)) { error in
            _ = error  // any failure proves the control: empty/invalid output is not a JSON object
        }
    }

    func testAdversarialScorerWithConstraintProducesValidJSONEndingInEOS() throws {
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)

        let ids = try greedyDecodeConstrained(
            priority: AdversarialVocab.priority, eosId: AdversarialVocab.eos, maxSteps: 50,
            constraint: &constraint)
        XCTAssertEqual(ids.last, AdversarialVocab.eos, "generation must end by picking EOS")

        let output = text(
            for: ids.filter { $0 != AdversarialVocab.eos }, classifications: AdversarialVocab.classifications)
        let data = Data(output.utf8)
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data))
        let parsed = try JSONSerialization.jsonObject(with: data)
        XCTAssertTrue(parsed is [String: Any], "top-level value must be an object")
    }

    func testAdversarialScorerMutationEOSAlwaysAllowedProducesInvalidOutput() throws {
        // MUTATION: flip the EOS-completeness gate off. The same adversarial scorer (which always
        // tries EOS first) must now immediately terminate with empty/invalid output again, proving
        // the gate — not some other mechanism — is what made the unmutated test above pass.
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table, eosAlwaysAllowedForTesting: true)

        let ids = try greedyDecodeConstrained(
            priority: AdversarialVocab.priority, eosId: AdversarialVocab.eos, maxSteps: 50,
            constraint: &constraint)
        XCTAssertEqual(ids, [AdversarialVocab.eos])

        let output = text(
            for: ids.filter { $0 != AdversarialVocab.eos }, classifications: AdversarialVocab.classifications)
        XCTAssertThrowsError(try JSONSerialization.jsonObject(with: Data(output.utf8)))
    }

    // MARK: - Whitespace-preferring scorer

    func testWhitespacePreferringScorerTerminatesWithCompleteObjectWithinBoundedSteps() throws {
        // Prefers whitespace above EVERYTHING else that's legal at each position, including the
        // closing brace and EOS. Ranking `closeBrace`/`eos` immediately after `space` (rather than
        // after key/value content tokens) keeps this scorer out of any open string — inside a
        // string, byte 0x20 is legal CONTENT with no length cap (RFC 8259 strings may contain
        // arbitrary runs of spaces), so a scorer that also outranks string content with space
        // would never terminate; that is correct JSON behavior, not a gap in the cap. Only the
        // STRUCTURAL whitespace between tokens is capped at 16 bytes, and this scorer stays
        // entirely in structural positions (before `{`, before `}`, after the top-level object),
        // so the cap alone is what forces it to eventually pick `{`, `}`, and `EOS`.
        let priority: [Int] = [
            AdversarialVocab.space, AdversarialVocab.closeBrace, AdversarialVocab.eos,
            AdversarialVocab.openBrace, AdversarialVocab.quote, AdversarialVocab.colon,
            AdversarialVocab.comma, AdversarialVocab.a, AdversarialVocab.b, AdversarialVocab.one,
            AdversarialVocab.openBraceQuote, AdversarialVocab.quoteColon,
            AdversarialVocab.quoteCloseBrace, AdversarialVocab.banned,
        ]
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)

        let cap = JSONObjectAutomaton.defaultMaxConsecutiveWhitespace
        // Three structural gaps a scorer this greedy will pad to the cap: before `{`, before `}`,
        // and after the top-level object closes (before EOS) — plus the three non-whitespace
        // picks (`{`, `}`, EOS) themselves.
        let maxSteps = cap * 3 + 10
        let ids = try greedyDecodeConstrained(
            priority: priority, eosId: AdversarialVocab.eos, maxSteps: maxSteps,
            constraint: &constraint)

        XCTAssertLessThan(ids.count, maxSteps, "must terminate before exhausting the step budget")
        XCTAssertEqual(ids.last, AdversarialVocab.eos)
        XCTAssertEqual(ids.filter { $0 == AdversarialVocab.openBrace }.count, 1)
        XCTAssertEqual(ids.filter { $0 == AdversarialVocab.closeBrace }.count, 1)
        XCTAssertLessThanOrEqual(
            ids.filter { $0 == AdversarialVocab.space }.count, cap * 3,
            "the whitespace cap must have actually bounded every run of spaces")

        let output = text(
            for: ids.filter { $0 != AdversarialVocab.eos }, classifications: AdversarialVocab.classifications)
        let parsed = try JSONSerialization.jsonObject(with: Data(output.utf8))
        XCTAssertTrue(parsed is [String: Any])
    }

    func testWhitespacePreferringScorerWithoutCapWouldNeverTerminate() {
        // Positive control: with the production cap, an unbounded run of whitespace bytes stops
        // exactly at the cap. Structural whitespace only exists INSIDE the object (defect B, slice
        // 1f: none is accepted before the top-level `{`), so both automatons here are first driven
        // past `{` before the run of spaces under test.
        var capped = JSONObjectAutomaton()
        XCTAssertTrue(capped.advance(byte: UInt8(ascii: "{")))
        var cappedConsumed = 0
        for _ in 0..<(JSONObjectAutomaton.defaultMaxConsecutiveWhitespace * 10) {
            guard capped.advance(byte: 0x20) else { break }
            cappedConsumed += 1
        }
        XCTAssertEqual(cappedConsumed, JSONObjectAutomaton.defaultMaxConsecutiveWhitespace)

        // MUTATION: actually disable the cap via the injectable parameter (Int.max), rather than
        // just asserting the capped behavior again. A whitespace-preferring scorer facing this
        // automaton would never terminate: every one of a large bounded step budget must still be
        // accepted, proving nothing else in the automaton independently bounds a whitespace run —
        // the production cap is the only thing that does. (All bytes here are plain spaces, so the
        // separate one-line-break-per-run limit is never in play.)
        var uncapped = JSONObjectAutomaton(maxConsecutiveWhitespace: Int.max)
        XCTAssertTrue(uncapped.advance(byte: UInt8(ascii: "{")))
        let stepBudget = JSONObjectAutomaton.defaultMaxConsecutiveWhitespace * 1000
        var uncappedConsumed = 0
        for _ in 0..<stepBudget {
            guard uncapped.advance(byte: 0x20) else { break }
            uncappedConsumed += 1
        }
        XCTAssertEqual(
            uncappedConsumed, stepBudget,
            "without the cap, whitespace must never be rejected within the step budget")
    }

    // MARK: - Cache

    func testRepeatedAutomatonStateProducesACacheHit() throws {
        // `warmCache: false`: the table-build-time warm-up (see `testCommonStatesAreWarmedAtTableBuild`)
        // pre-populates exactly this state (the initial, before-top-level state), which would
        // otherwise turn the "first call must miss" assertion below into a false failure.
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications, warmCache: false)
        let constraint = JSONObjectTokenConstraint(table: table)

        let missesBefore = table.cacheMissCount
        let hitsBefore = table.cacheHitCount

        _ = try constraint.allowedTokenIds()
        XCTAssertEqual(table.cacheMissCount, missesBefore + 1)
        XCTAssertEqual(table.cacheHitCount, hitsBefore)

        _ = try constraint.allowedTokenIds()
        XCTAssertEqual(table.cacheMissCount, missesBefore + 1, "second call at the same state must not miss again")
        XCTAssertEqual(table.cacheHitCount, hitsBefore + 1)
    }

    func testDifferentAutomatonStatesProduceDistinctCacheEntries() throws {
        // `warmCache: false`: both the initial state and the after-`{` state are in the table-
        // build warm-up's curated list (see `testCommonStatesAreWarmedAtTableBuild`), which would
        // otherwise make the post-`{` lookup below a hit, not the miss this test asserts.
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications, warmCache: false)
        var constraint = JSONObjectTokenConstraint(table: table)

        _ = try constraint.allowedTokenIds()  // state: before top level
        try constraint.advance(token: AdversarialVocab.openBrace)
        let missesBefore = table.cacheMissCount
        _ = try constraint.allowedTokenIds()  // a genuinely different state
        XCTAssertEqual(table.cacheMissCount, missesBefore + 1)
    }

    func testCommonStatesAreWarmedAtTableBuild() throws {
        // Positive control for `JSONObjectConstraintTable.warmCommonStates()`: the DEFAULT
        // (warm-up enabled) initializer must leave the two canonical states every real request
        // starts at — the initial, before-top-level state, and the state right after `{` —
        // already cached, so a request's very FIRST lookup is a cache HIT, not the O(vocab) miss
        // defect 3 measured at 78.5ms p99. Uses `FullASCIIVocab` (one token per printable ASCII
        // byte) so every warm-up prefix is reachable-with-continuation, unlike the tiny
        // `AdversarialVocab` fixture the cache-mechanics tests above deliberately opt out of
        // warming (`warmCache: false`) to test raw miss/hit behavior in isolation.
        let table = JSONObjectConstraintTable(classifications: FullASCIIVocab.classifications)
        XCTAssertEqual(table.cacheMissCount, 0, "warm-up must not be counted as a real miss")
        XCTAssertEqual(table.cacheHitCount, 0, "warm-up must not be counted as a real hit either")

        var constraint = JSONObjectTokenConstraint(table: table)
        _ = try constraint.allowedTokenIds()  // the very FIRST real lookup, at the initial state
        XCTAssertEqual(table.cacheMissCount, 0, "the initial state must already be warm")
        XCTAssertEqual(table.cacheHitCount, 1)

        try constraint.advance(token: FullASCIIVocab.id(for: 0x7B))  // '{'
        _ = try constraint.allowedTokenIds()
        XCTAssertEqual(table.cacheMissCount, 0, "the post-'{' state must already be warm")
        XCTAssertEqual(table.cacheHitCount, 2)
    }

    func testCacheByteBudgetBindsOnLargeSyntheticVocab() throws {
        // A large synthetic vocab (50k ids) so each cached bitset entry is non-trivially sized:
        // `ceil(50000/64) * 8` = 6,256 bytes per cached automaton state, regardless of how many of
        // those 50k ids are actually allowed at that state.
        let vocabSize = 50_000
        enum Ids {
            static let openBrace = 0, keyQuote = 1, keyChar = 2, colon = 3, openBracket = 4
        }
        var classifications: [TokenByteClassification] = [
            .bytes([0x7B]), .bytes([0x22]), .bytes([0x6B]), .bytes([0x3A]), .bytes([0x5B]),
            // A single token at least `maxDepth` bytes long: the mask cache key now truncates a
            // state's visible stack to the vocab's longest `.bytes` token length (see
            // `JSONObjectAutomaton.maskCacheKey(maxTokenBytes:)`), since no shorter token could
            // ever observe deeper frames. Without an anchor token this long, every nesting depth
            // past that bound would collapse onto the SAME cache key — which is the intended,
            // exact cache-efficiency win this slice adds, but it would make the ~60 distinct
            // states this test relies on stop being distinct, defeating the eviction control
            // below. Present only to keep `maxTokenBytes` >= `maxDepth` for this test; never
            // advanced through.
            .bytes(Array(repeating: UInt8(0x5A), count: JSONObjectAutomaton.maxDepth)),
        ]
        while classifications.count < vocabSize {
            classifications.append(.bytes([UInt8(0x41 + classifications.count % 26)]))
        }
        let entryBytes = ((vocabSize + 63) / 64) * 8
        // Enough budget for a handful of entries, nowhere near the ~60 distinct nesting depths the
        // walk below will produce.
        let smallBudget = entryBytes * 5
        let table = JSONObjectConstraintTable(classifications: classifications, cacheByteBudget: smallBudget)
        var constraint = JSONObjectTokenConstraint(table: table)

        try constraint.advance(token: Ids.openBrace)
        try constraint.advance(token: Ids.keyQuote)
        try constraint.advance(token: Ids.keyChar)
        try constraint.advance(token: Ids.keyQuote)
        try constraint.advance(token: Ids.colon)

        // Open nested arrays one at a time: each nesting depth is a genuinely distinct automaton
        // state (up to `maxDepth`), so every step below is a cache MISS inserting a new,
        // large (~6 KiB) bitset entry — far more entries than `smallBudget` can hold at once.
        for _ in 0..<(JSONObjectAutomaton.maxDepth - 1) {
            _ = try constraint.allowedTokenIds()
            try constraint.advance(token: Ids.openBracket)
        }

        XCTAssertLessThanOrEqual(
            table.cachedBytes, smallBudget + entryBytes,
            "cache must stay within its byte budget (plus at most the one entry always let in)")
        XCTAssertGreaterThan(
            table.cacheEvictionCount, 0,
            "control: a tiny budget against ~60 distinct states must actually evict something")
    }

    // MARK: - Precondition: at least one id is always allowed

    func testAllowedTokenIdsNeverEmptyAcrossAnAdversarialTrace() throws {
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        for _ in 0..<40 {
            let allowed = try constraint.allowedTokenIds()
            XCTAssertFalse(allowed.isEmpty)
            guard let pick = allowed.first(where: { $0 != AdversarialVocab.eos }) ?? allowed.first else {
                break
            }
            if pick == AdversarialVocab.eos { break }
            try constraint.advance(token: pick)
        }
    }

    /// A single-byte token per printable ASCII value (0x20...0x7E) plus EOS and a banned id —
    /// enough coverage to drive a real JSON trace through strings, escapes, numbers, and nested
    /// arrays token-by-token.
    private enum FullASCIIVocab {
        static let printableBytes: [UInt8] = Array(0x20...0x7E)
        static let eos = printableBytes.count
        static let banned = eos + 1
        static let classifications: [TokenByteClassification] = {
            var out: [TokenByteClassification] = printableBytes.map { .bytes([$0]) }
            out.append(.eos)
            out.append(.banned)
            return out
        }()
        static func id(for byte: UInt8) -> Int {
            Int(byte) - Int(printableBytes[0])
        }
    }

    func testAllowedTokenIdsNeverEmptyAcrossManyReachableGrammarStates() throws {
        // A scripted trace exercising: object/key/value strings, an escape sequence, numbers
        // (minus, digits, fraction, exponent with an explicit sign), nested arrays, and both `:`-
        // and `,`-separated positions — asserting the mask is never empty at any byte boundary.
        let table = JSONObjectConstraintTable(classifications: FullASCIIVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        let trace = Array(#"{"a":"x\n","b":[1,-2.5e+3],"c":123}"#.utf8)
        for byte in trace {
            let allowed = try constraint.allowedTokenIds()
            XCTAssertFalse(allowed.isEmpty, "unreachable state before byte 0x\(String(byte, radix: 16))")
            try constraint.advance(token: FullASCIIVocab.id(for: byte))
        }
        let finalAllowed = try constraint.allowedTokenIds()
        XCTAssertFalse(finalAllowed.isEmpty)
        XCTAssertTrue(constraint.isComplete)
    }

    // MARK: - `advance` throws on disallowed tokens

    func testAdvanceThrowsForBannedToken() {
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        XCTAssertThrowsError(try constraint.advance(token: AdversarialVocab.banned)) { error in
            XCTAssertEqual(error as? JSONObjectConstraintError, .tokenDisallowed(AdversarialVocab.banned))
        }
    }

    func testAdvanceThrowsForEOSBeforeComplete() {
        let table = JSONObjectConstraintTable(classifications: AdversarialVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        XCTAssertThrowsError(try constraint.advance(token: AdversarialVocab.eos)) { error in
            XCTAssertEqual(error as? JSONObjectConstraintError, .tokenDisallowed(AdversarialVocab.eos))
        }
    }

    // MARK: - Release micro-benchmark (skipped by default)

    /// Synthetic 248,000-token vocab + a recorded 500-token JSON trace, reporting mask p50/p99
    /// for cache hit/miss and the hit rate. Skipped unless `RUN_JSON_CONSTRAINT_BENCH=1`, since
    /// building/walking a realistic-scale trie is multi-second work that should not slow down the
    /// default `swift test` loop. Run explicitly with:
    ///   RUN_JSON_CONSTRAINT_BENCH=1 swift test -c release --package-path spike \
    ///     --filter ServingCoreTests.JSONObjectTokenConstraintTests/testReleaseMicroBenchmark
    func testReleaseMicroBenchmark() throws {
        guard ProcessInfo.processInfo.environment["RUN_JSON_CONSTRAINT_BENCH"] == "1" else {
            throw XCTSkip("set RUN_JSON_CONSTRAINT_BENCH=1 to run the perf micro-benchmark")
        }

        let vocabSize = 248_000
        var classifications: [TokenByteClassification] = []
        classifications.reserveCapacity(vocabSize)

        // All 256 single bytes first (mirrors a real byte-level vocab's base alphabet).
        for byte in 0...255 {
            classifications.append(.bytes([UInt8(byte)]))
        }

        // Deterministic pseudo-random printable strings, length 1...12, filling the rest.
        var state: UInt64 = 0x9E3779B97F4A7C15
        func nextRandom() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        let printable: [UInt8] = Array(UInt8(0x20)...UInt8(0x7E))
        while classifications.count < vocabSize {
            let length = 1 + Int(nextRandom() % 12)
            var bytes: [UInt8] = []
            bytes.reserveCapacity(length)
            for _ in 0..<length {
                bytes.append(printable[Int(nextRandom() % UInt64(printable.count))])
            }
            classifications.append(.bytes(bytes))
        }
        classifications.append(.eos)
        let eosId = classifications.count - 1
        _ = eosId

        // A recorded ~900-byte JSON trace: a flat object with 100 `"k<i>": "v<i>"` pairs, each
        // separated by `", "` — string VALUES (not just keys) and structural whitespace, so the
        // trace actually walks the expensive states (deep inside an open string, and the
        // whitespace-run state) rather than only short numeric-value states. Replayed byte-by-byte
        // through `advance` using whichever allowed id matches (falls back to feeding raw bytes
        // directly against the automaton where no vocab token happens to match, which is realistic
        // for a synthetic vocab).
        var traceJSON = "{ "
        for i in 0..<100 {
            if i > 0 { traceJSON += ", " }
            traceJSON += "\"k\(i)\": \"v\(i)\""
        }
        traceJSON += " }"

        struct BenchResult {
            var tableBuildMs: Double
            var warmupMs: Double
            var missDurations: [Double]
            var hitDurations: [Double]
            var hitRate: Double
        }

        // Replays `traceJSON` once against a freshly built table (`warmCache` controls whether the
        // table-build-time cache warm — see `JSONObjectConstraintTable.warmCommonStates()` — runs
        // first). Factored out so the SAME trace/measurement logic backs both the `warmCache: false`
        // run (isolates the allocation-free trie walk's own per-miss improvement, since it forces
        // every state in the trace to actually miss at least once — the production default would
        // otherwise pre-warm all of them and report zero misses here) and the `warmCache: true` run
        // (the actual production default, reported alongside it).
        func runBenchmark(warmCache: Bool) throws -> BenchResult {
            let tableBuildStart = DispatchTime.now()
            let table = JSONObjectConstraintTable(
                classifications: classifications, warmCache: warmCache)
            let tableBuildMs =
                Double(DispatchTime.now().uptimeNanoseconds - tableBuildStart.uptimeNanoseconds) / 1_000_000
            var constraint = JSONObjectTokenConstraint(table: table)

            var missDurations: [Double] = []
            var hitDurations: [Double] = []
            var warmupMs: Double?

            for byte in traceJSON.utf8 {
                let missesBefore = table.cacheMissCount
                let start = DispatchTime.now()
                let allowed = try constraint.allowedTokenIds()
                let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
                if warmupMs == nil { warmupMs = elapsedMs }
                if table.cacheMissCount > missesBefore {
                    missDurations.append(elapsedMs)
                } else {
                    hitDurations.append(elapsedMs)
                }
                XCTAssertFalse(allowed.isEmpty)
                // Best-effort: find a single-byte token id matching this trace byte and advance by it.
                if let id = classifications.firstIndex(where: {
                    if case .bytes(let b) = $0 { return b == [byte] }
                    return false
                }), allowed.contains(id) {
                    try constraint.advance(token: id)
                }
            }

            let hitRate = Double(table.cacheHitCount) / Double(max(1, table.cacheHitCount + table.cacheMissCount))
            return BenchResult(
                tableBuildMs: tableBuildMs, warmupMs: warmupMs ?? .nan,
                missDurations: missDurations, hitDurations: hitDurations, hitRate: hitRate)
        }

        func percentile(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return .nan }
            let sorted = values.sorted()
            let index = min(sorted.count - 1, Int(Double(sorted.count - 1) * p))
            return sorted[index]
        }

        // `nowarm`: isolates the allocation-free trie walk / truncated cache key's own per-miss
        // improvement, on the SAME trace, by forcing every visited state to actually miss once
        // (table-build warm-up disabled). `warm` (unlabeled, matches the metric name used by the
        // 1d/1e predeclaration): the actual production default — reported alongside it, not in
        // place of it, since a real deployment always uses the warmed default.
        let nowarm = try runBenchmark(warmCache: false)
        let warm = try runBenchmark(warmCache: true)

        func line(_ label: String, _ result: BenchResult) -> String {
            """
            [json-constraint-bench\(label)] vocabSize=\(vocabSize) traceBytes=\(traceJSON.utf8.count) \
            misses=\(result.missDurations.count) hits=\(result.hitDurations.count) hitRate=\(result.hitRate) \
            missP50ms=\(percentile(result.missDurations, 0.50)) missP99ms=\(percentile(result.missDurations, 0.99)) \
            hitP50ms=\(percentile(result.hitDurations, 0.50)) hitP99ms=\(percentile(result.hitDurations, 0.99)) \
            tableBuildMs=\(result.tableBuildMs) warmupMs=\(result.warmupMs)
            """
        }
        print(line("-nowarm", nowarm))
        print(line("", warm))
    }
}
