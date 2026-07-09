import XCTest
@testable import HarnessCore

final class MeasurementCorpusTests: XCTestCase {
    func sampleJSON(text2: String = "second entry") -> Data {
        let json = """
        {
          "corpusId": "test-corpus-v1",
          "entries": [
            {"id": "b-entry", "tag": "code", "text": "\(text2)"},
            {"id": "a-entry", "tag": "prose", "text": "first entry"}
          ]
        }
        """
        return Data(json.utf8)
    }

    func testLoadsEntriesAndCorpusId() throws {
        let corpus = try MeasurementCorpusLoader.load(from: sampleJSON())
        XCTAssertEqual(corpus.corpusId, "test-corpus-v1")
        XCTAssertEqual(corpus.entries.count, 2)
        XCTAssertEqual(corpus.entries(tagged: .prose).map(\.id), ["a-entry"])
        XCTAssertEqual(corpus.entries(tagged: .code).map(\.id), ["b-entry"])
    }

    func testContentHashStableAcrossEntryOrder() throws {
        let corpusA = try MeasurementCorpusLoader.load(from: sampleJSON())
        // Same entries, JSON array order swapped — hash sorts by id internally, must match.
        let swapped = """
        {"corpusId": "test-corpus-v1", "entries": [
          {"id": "a-entry", "tag": "prose", "text": "first entry"},
          {"id": "b-entry", "tag": "code", "text": "second entry"}
        ]}
        """
        let corpusB = try MeasurementCorpusLoader.load(from: Data(swapped.utf8))
        XCTAssertEqual(corpusA.contentHash, corpusB.contentHash)
    }

    func testContentHashChangesWhenTextChanges() throws {
        let corpusA = try MeasurementCorpusLoader.load(from: sampleJSON())
        let corpusB = try MeasurementCorpusLoader.load(from: sampleJSON(text2: "a different second entry"))
        XCTAssertNotEqual(corpusA.contentHash, corpusB.contentHash)
    }

    func testContentHashIsAPinnedLiteral() throws {
        // Regression guard: the hash ALGORITHM must not silently drift. If this fails after an
        // intentional algorithm change, recompute and update the literal deliberately.
        let corpus = try MeasurementCorpusLoader.load(from: sampleJSON())
        XCTAssertEqual(corpus.contentHash, "a01654715415f718")
    }

    func testEmptyEntriesThrows() {
        let json = Data("""
        {"corpusId": "empty-v1", "entries": []}
        """.utf8)
        XCTAssertThrowsError(try MeasurementCorpusLoader.load(from: json)) { error in
            guard case MeasurementCorpusError.empty = error else {
                return XCTFail("expected .empty, got \(error)")
            }
        }
    }

    func testMalformedJSONThrowsDecodeFailed() {
        let json = Data("not json".utf8)
        XCTAssertThrowsError(try MeasurementCorpusLoader.load(from: json)) { error in
            guard case MeasurementCorpusError.decodeFailed = error else {
                return XCTFail("expected .decodeFailed, got \(error)")
            }
        }
    }

    func testHostileBytesToleratedInEntryText() throws {
        // Null bytes, RTL override, emoji, combining marks — must round-trip without crashing and
        // must not corrupt the hash delimiter scheme (fields are \0-delimited; a literal \0 inside
        // text could in principle collide two different entries onto the same hash input). Built
        // via JSONSerialization (not string interpolation) so the control bytes are properly
        // JSON-escaped rather than embedded raw, which would itself be invalid JSON.
        let hostile = "\u{0000}\u{202E}garbage\u{0301} \u{1F600} done."
        let obj: [String: Any] = [
            "corpusId": "test-corpus-v1",
            "entries": [
                ["id": "a-entry", "tag": "prose", "text": "first entry"],
                ["id": "b-entry", "tag": "code", "text": hostile],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: obj)
        let corpus = try MeasurementCorpusLoader.load(from: data)
        XCTAssertEqual(corpus.entries(tagged: .code).first?.text, hostile)
        XCTAssertFalse(corpus.contentHash.isEmpty)
    }

    func testRealCheckedInCorpusLoadsAndHasLongContextEntry() throws {
        // Walk up from the test bundle to the repo's spike/ root to find the checked-in file —
        // this test proves the ACTUAL shipped corpus loads, not just a synthetic fixture.
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL?
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("corpus/measurement-corpus-v1.json")
            if FileManager.default.fileExists(atPath: candidate.path) { found = candidate; break }
            dir = dir.deletingLastPathComponent()
        }
        guard let path = found else {
            return XCTFail("could not locate spike/corpus/measurement-corpus-v1.json by walking up from \(#filePath)")
        }
        let data = try Data(contentsOf: path)
        let corpus = try MeasurementCorpusLoader.load(from: data)
        XCTAssertEqual(corpus.corpusId, "measurement-corpus-v1")
        XCTAssertFalse(corpus.entries(tagged: .prose).isEmpty)
        XCTAssertFalse(corpus.entries(tagged: .code).isEmpty)
        let longEntries = corpus.entries(tagged: .longContext)
        XCTAssertFalse(longEntries.isEmpty, "corpus must include at least one long-context entry")
        // Character-count proxy for token count (real tokenizer check happens on llmbench where
        // the model's tokenizer is available) — well over the 4x margin needed to clear 4K tokens.
        for e in longEntries {
            XCTAssertGreaterThan(e.text.count, 16_000, "\(e.id) too short to plausibly clear 4K tokens")
        }
    }

    func testRealCheckedInCorpusV2AddsA16KTokenEntry() throws {
        // v2 = v1's entries + one >=16K-TOKEN natural-prose entry (the TurboQuant regime needs
        // context an order of magnitude past v1's ~5.8K-token ceiling-era entry). Same
        // character-count proxy: >=64K chars comfortably clears 16K tokens for technical prose.
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL?
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("corpus/measurement-corpus-v2.json")
            if FileManager.default.fileExists(atPath: candidate.path) { found = candidate; break }
            dir = dir.deletingLastPathComponent()
        }
        guard let path = found else {
            return XCTFail("could not locate spike/corpus/measurement-corpus-v2.json by walking up from \(#filePath)")
        }
        let corpus = try MeasurementCorpusLoader.load(from: try Data(contentsOf: path))
        XCTAssertEqual(corpus.corpusId, "measurement-corpus-v2")
        XCTAssertFalse(corpus.entries(tagged: .prose).isEmpty)
        XCTAssertFalse(corpus.entries(tagged: .code).isEmpty)
        let longEntries = corpus.entries(tagged: .longContext)
        XCTAssertEqual(longEntries.count, 2, "v2 keeps v1's long entry and adds the 16K one")
        XCTAssertTrue(
            longEntries.contains { $0.text.count >= 64_000 },
            "v2 must contain a >=64K-char (>=16K-token) long-context entry")
    }
}

final class PositionSamplingTests: XCTestCase {
    func testReturnsAllPositionsWhenSampleSizeCoversTotal() {
        XCTAssertEqual(evenlySpacedPositions(total: 5, sampleSize: 10), [0, 1, 2, 3, 4])
        XCTAssertEqual(evenlySpacedPositions(total: 5, sampleSize: 0), [0, 1, 2, 3, 4])
    }

    func testEmptyTotal() {
        XCTAssertEqual(evenlySpacedPositions(total: 0, sampleSize: 10), [])
    }

    func testSingleSampleReturnsFirstPosition() {
        XCTAssertEqual(evenlySpacedPositions(total: 100, sampleSize: 1), [0])
    }

    func testSampleIsAscendingDedupedAndBounded() {
        let positions = evenlySpacedPositions(total: 4096, sampleSize: 64)
        XCTAssertEqual(positions, positions.sorted())
        XCTAssertEqual(positions.count, Set(positions).count, "must be deduped")
        XCTAssertLessThanOrEqual(positions.count, 64)
        XCTAssertEqual(positions.first, 0)
        XCTAssertEqual(positions.last, 4095, "endpoints included so the sample spans the full sequence")
        for p in positions { XCTAssertTrue((0..<4096).contains(p)) }
    }

    func testSmallRangeDedupesWithoutCrashing() {
        // sampleSize close to total forces rounding collisions — must dedupe, not crash or
        // produce out-of-range/duplicate indices.
        let positions = evenlySpacedPositions(total: 10, sampleSize: 9)
        XCTAssertEqual(positions, positions.sorted())
        XCTAssertEqual(positions.count, Set(positions).count)
        for p in positions { XCTAssertTrue((0..<10).contains(p)) }
    }
}
