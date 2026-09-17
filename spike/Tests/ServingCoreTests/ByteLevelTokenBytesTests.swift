import XCTest

@testable import ServingCore

final class ByteLevelTokenBytesTests: XCTestCase {
    // MARK: - Known byte mappings

    func testSpaceMapsToGpt2SpaceGlyph() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x20], Unicode.Scalar(0x0120))
        XCTAssertEqual(ByteLevelTokenBytes.bytes(forVocabString: "Ġ"), [0x20])
    }

    func testNewlineMapsToGpt2NewlineGlyph() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x0A], Unicode.Scalar(0x010A))
        XCTAssertEqual(ByteLevelTokenBytes.bytes(forVocabString: "Ċ"), [0x0A])
    }

    // MARK: - Pinned bytes_to_unicode entries (HF GPT-2 `bytes_to_unicode()`)
    //
    // Printable ranges 0x21...0x7E, 0xA1...0xAC, 0xAE...0xFF map to themselves. Every other byte
    // (0x00...0x20, 0x7F...0xA0, 0xAD) is assigned, in ascending byte order, to U+0100, U+0101, ...
    // The non-identity assignments below are computed from that rule:
    //   ascending non-identity bytes: 0,1,...,32, 127,128,...,160, 173
    //   0x00 (index 0)  -> U+0100        0x20 (index 32) -> U+0120
    //   0x7F (index 33) -> U+0121        0xA0 (index 66) -> U+0142
    //   0xAD (index 67) -> U+0143

    func testNonIdentityByteZeroMapsToU0100() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x00], Unicode.Scalar(0x0100))
    }

    func testNonIdentityByteSpaceMapsToU0120() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x20], Unicode.Scalar(0x0120))
    }

    func testNonIdentityByteDELMapsToU0121() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x7F], Unicode.Scalar(0x0121))
    }

    func testNonIdentityByteA0MapsToU0142() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0xA0], Unicode.Scalar(0x0142))
    }

    func testNonIdentityByteADMapsToU0143() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0xAD], Unicode.Scalar(0x0143))
    }

    func testIdentityByteA1MapsToItself() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0xA1], Unicode.Scalar(0xA1))
    }

    func testIdentityByteAEMapsToItself() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0xAE], Unicode.Scalar(0xAE))
    }

    func testIdentityByteExclamationMapsToItself() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x21], Unicode.Scalar(0x21))
    }

    func testIdentityByteTildeMapsToItself() {
        XCTAssertEqual(ByteLevelTokenBytes.byteToUnicodeScalar[0x7E], Unicode.Scalar(0x7E))
    }

    func testHighByteRoundTripsThroughItsGlyph() {
        // 0xC3 is not printable ASCII/Latin-1-identity range, so it's an "extra" assigned glyph.
        let byte: UInt8 = 0xC3
        guard let scalar = ByteLevelTokenBytes.byteToUnicodeScalar[byte] else {
            return XCTFail("expected a mapping for byte 0xC3")
        }
        XCTAssertEqual(ByteLevelTokenBytes.unicodeScalarToByte[scalar], byte)
    }

    func testAllTwoFiftySixBytesRoundTrip() {
        for byteValue in 0...255 {
            let byte = UInt8(byteValue)
            guard let scalar = ByteLevelTokenBytes.byteToUnicodeScalar[byte] else {
                XCTFail("byte \(byteValue) has no glyph mapping")
                continue
            }
            XCTAssertEqual(
                ByteLevelTokenBytes.unicodeScalarToByte[scalar], byte,
                "byte \(byteValue) did not round-trip through its glyph")
            let recovered = ByteLevelTokenBytes.bytes(forVocabString: String(scalar))
            XCTAssertEqual(recovered, [byte], "byte \(byteValue) failed bytes(forVocabString:) round trip")
        }
        XCTAssertEqual(Set(ByteLevelTokenBytes.byteToUnicodeScalar.values).count, 256)
    }

    func testMultiByteVocabStringRecoversAllUnderlyingBytes() {
        // "hello" byte-level-encoded is itself since all are printable ASCII identity-mapped.
        XCTAssertEqual(ByteLevelTokenBytes.bytes(forVocabString: "hello"), Array("hello".utf8))
    }

    func testUnmappableScalarReturnsNil() {
        // U+FFFD is not in the byte-level glyph table (it's not one of the assigned glyphs).
        XCTAssertNil(ByteLevelTokenBytes.bytes(forVocabString: "\u{FFFD}"))
    }

    // MARK: - Classification

    func testClassifyPrioritizesEOSOverAddedTokenMembership() {
        // id 5 is both "added" and "eos" -> must classify as .eos, matching the DECISION doc's
        // stated EOS-set membership rule.
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 6,
            vocabString: { id in id == 5 ? "<|endoftext|>" : "a" },
            addedTokenIds: [5],
            eosTokenIds: [5])
        XCTAssertEqual(classifications[5], .eos)
    }

    func testClassifyBansAddedNonEOSTokens() {
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 3,
            vocabString: { id in id == 1 ? "<|im_end|>" : "a" },
            addedTokenIds: [1],
            eosTokenIds: [])
        XCTAssertEqual(classifications[1], .banned)
    }

    func testClassifyBansUnmappableVocabStrings() {
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 2,
            vocabString: { _ in "\u{FFFD}" },
            addedTokenIds: [],
            eosTokenIds: [])
        XCTAssertEqual(classifications, [.banned, .banned])
    }

    func testClassifyBansNilVocabStrings() {
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 2,
            vocabString: { id in id == 0 ? "a" : nil },
            addedTokenIds: [],
            eosTokenIds: [])
        XCTAssertEqual(classifications[0], .bytes([0x61]))
        XCTAssertEqual(classifications[1], .banned)
    }

    func testClassifyBansEmptyByteVocabStrings() {
        // An empty vocab string recovers to `[]` bytes (not `nil`): `advance` would treat such an
        // id as a free no-op token (its byte loop never runs), letting a decoder emit it forever
        // without making grammar progress. `classify` must ban it instead of allowing `.bytes([])`.
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 1,
            vocabString: { _ in "" },
            addedTokenIds: [],
            eosTokenIds: [])
        XCTAssertEqual(classifications[0], .banned)
    }

    func testClassifyRecoversBytesForOrdinaryTokens() {
        let classifications = ByteLevelTokenBytes.classify(
            vocabSize: 1,
            vocabString: { _ in "Ġhi" },
            addedTokenIds: [],
            eosTokenIds: [])
        XCTAssertEqual(classifications[0], .bytes([0x20, 0x68, 0x69]))
    }

    // MARK: - Self-check

    func testSelfCheckReportsNoMismatchesWhenDecodeAgrees() {
        let classifications: [TokenByteClassification] = [.bytes(Array("hi".utf8))]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: classifications,
            decode: { _ in "hi" })
        XCTAssertTrue(mismatches.isEmpty)
    }

    func testSelfCheckReportsMismatchWithoutCrashing() {
        let classifications: [TokenByteClassification] = [.bytes(Array("hi".utf8))]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: classifications,
            decode: { _ in "WRONG" })
        XCTAssertEqual(mismatches.count, 1)
        XCTAssertEqual(mismatches[0].id, 0)
        XCTAssertEqual(mismatches[0].recoveredUTF8, "hi")
        XCTAssertEqual(mismatches[0].decoded, "WRONG")
    }

    func testSelfCheckReportsMismatchWhenDecodeReturnsNil() {
        let classifications: [TokenByteClassification] = [.bytes(Array("hi".utf8))]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: classifications,
            decode: { _ in nil })
        XCTAssertEqual(mismatches.count, 1)
        XCTAssertEqual(mismatches[0].decoded, "<nil>")
    }

    func testSelfCheckSkipsIdsWhoseBytesAreNotStandaloneValidUTF8() {
        // A split-CJK head token's raw bytes (E4 BD, the first two bytes of "你" = E4 BD A0) are
        // NOT valid UTF-8 on their own, so the self-check must skip it rather than report a false
        // mismatch — decode([id]) legitimately returns U+FFFD for a partial sequence.
        let classifications: [TokenByteClassification] = [.bytes([0xE4, 0xBD])]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: classifications,
            decode: { _ in "\u{FFFD}" })
        XCTAssertTrue(mismatches.isEmpty)
    }

    func testSelfCheckIgnoresNonBytesClassifications() {
        let classifications: [TokenByteClassification] = [.eos, .banned]
        let mismatches = ByteLevelTokenBytes.selfCheckMismatches(
            classifications: classifications,
            decode: { _ in "should not matter" })
        XCTAssertTrue(mismatches.isEmpty)
    }

    // MARK: - BOM self-check exclusion (slice 1e item 2)
    //
    // `recoveredUTF8` is computed with Foundation's `String(bytes:encoding:.utf8)`, which silently
    // drops a leading U+FEFF byte-order mark; the tokenizer's own byte-level `decode([id])` path
    // does NOT drop it. A token whose raw bytes are (or start with) the 3-byte UTF-8 BOM therefore
    // self-checks as a false mismatch — `decoded` legitimately carries the leading U+FEFF that
    // `recoveredUTF8` lost. `excludingKnownBOMArtifacts` must recognize exactly this shape, refuse
    // to report it as a mismatch, and reclassify the id `.banned` so no emitted text can diverge
    // from what the automaton validated.

    func testExcludingKnownBOMArtifactsBansPureBOMTokenWithoutReportingMismatch() {
        // Raw bytes EF BB BF are the UTF-8 encoding of U+FEFF. Foundation's
        // `String(bytes:encoding:.utf8)` strips it, producing "" for `recoveredUTF8`; the
        // tokenizer's decode path does not strip it, producing "\u{FEFF}" for `decoded` — mirrors
        // the real Qwen vocab id 3121 (`self_check_mismatch id=3121`).
        let classifications: [TokenByteClassification] = [.bytes([0xEF, 0xBB, 0xBF])]
        let result = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
            classifications: classifications,
            decode: { _ in "\u{FEFF}" })
        XCTAssertTrue(result.mismatches.isEmpty, "a BOM-only-strip divergence must not self-check as a mismatch")
        XCTAssertEqual(result.classifications[0], .banned, "the BOM token must be excluded, never allowed")
    }

    func testExcludingKnownBOMArtifactsBansBOMPrefixedTokenWithoutReportingMismatch() {
        // Raw bytes EF BB BF 61 are BOM + "a". `String(bytes:encoding:.utf8)` strips the leading
        // BOM and returns "a"; a decode that mirrors the same stripping legitimately returns "a"
        // too, so `decoded == "a"` here IS a BOM-prefixed match on the decode side (this is the
        // "\u{FEFF}" + decoded == recoveredUTF8 shape from the opposite direction — decode already
        // agrees with the stripped form). Use a decode that keeps the BOM, matching the real
        // tokenizer's byte-level decoder, so `decoded == "\u{FEFF}a"` while `recoveredUTF8 == "a"`.
        let classifications: [TokenByteClassification] = [.bytes([0xEF, 0xBB, 0xBF, 0x61])]
        let result = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
            classifications: classifications,
            decode: { _ in "\u{FEFF}a" })
        XCTAssertTrue(result.mismatches.isEmpty)
        XCTAssertEqual(result.classifications[0], .banned)
    }

    func testExcludingKnownBOMArtifactsStillReportsUnrelatedMismatches() {
        // Control: a decode divergence that is NOT the BOM-stripping shape must still be reported
        // and must NOT be silently banned.
        let classifications: [TokenByteClassification] = [.bytes(Array("hi".utf8))]
        let result = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
            classifications: classifications,
            decode: { _ in "WRONG" })
        XCTAssertEqual(result.mismatches.count, 1)
        XCTAssertEqual(result.mismatches[0].id, 0)
        XCTAssertEqual(result.mismatches[0].recoveredUTF8, "hi")
        XCTAssertEqual(result.mismatches[0].decoded, "WRONG")
        XCTAssertEqual(result.classifications[0], .bytes(Array("hi".utf8)), "an unrelated mismatch must not be reclassified")
    }

    func testExcludingKnownBOMArtifactsDoesNotBanWhenDecodedHasNoLeadingBOM() {
        // Control: `decoded` differs from `recoveredUTF8` but does NOT carry a leading U+FEFF that
        // `recoveredUTF8` lacks — this must still be a real, reported mismatch, not an exclusion.
        let classifications: [TokenByteClassification] = [.bytes(Array("a".utf8))]
        let result = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
            classifications: classifications,
            decode: { _ in "b" })
        XCTAssertEqual(result.mismatches.count, 1)
        XCTAssertEqual(result.classifications[0], .bytes(Array("a".utf8)))
    }

    func testExcludingKnownBOMArtifactsLeavesAgreeingTokensUntouched() {
        let classifications: [TokenByteClassification] = [.bytes(Array("hi".utf8)), .eos, .banned]
        let result = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
            classifications: classifications,
            decode: { id in id == 0 ? "hi" : "should not matter" })
        XCTAssertTrue(result.mismatches.isEmpty)
        XCTAssertEqual(result.classifications, classifications)
    }
}
