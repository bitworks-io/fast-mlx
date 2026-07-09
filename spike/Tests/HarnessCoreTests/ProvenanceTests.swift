import XCTest
@testable import HarnessCore

final class ProvenanceTests: XCTestCase {
    func testFnv1a64IsStableAndPinned() {
        // Regression guard on the algorithm itself — if this ever fails after a deliberate change,
        // recompute and update the literal deliberately (it changes every downstream hash).
        XCTAssertEqual(fnv1a64(Array("hello".utf8)), fnv1a64(Array("hello".utf8)))
        XCTAssertEqual(fnv1a64(Array("hello".utf8)), "a430d84680aabd0b")
        XCTAssertNotEqual(fnv1a64(Array("hello".utf8)), fnv1a64(Array("hellp".utf8)))
    }

    func testFnv1a64OfEmptyBytesIsTheOffsetBasis() {
        XCTAssertEqual(fnv1a64([] as [UInt8]), "cbf29ce484222325")
    }

    // MARK: ModelQuantInfo

    func testModelQuantInfoParsesBitsAndGroupSize() {
        let json = Data("""
        {"quantization": {"bits": 4, "group_size": 64}}
        """.utf8)
        let info = ModelQuantInfoLoader.load(from: json)
        XCTAssertEqual(info.bits, 4)
        XCTAssertEqual(info.groupSize, 64)
        XCTAssertEqual(info.label, "int4")
    }

    func testModelQuantInfoDefaultsToFP16WhenNoQuantizationBlock() {
        let json = Data("""
        {"model_type": "qwen3", "hidden_size": 4096}
        """.utf8)
        let info = ModelQuantInfoLoader.load(from: json)
        XCTAssertNil(info.bits)
        XCTAssertEqual(info.label, "fp16")
    }

    func testModelQuantInfoDefaultsToFP16OnMalformedJSON() {
        let info = ModelQuantInfoLoader.load(from: Data("not json at all".utf8))
        XCTAssertNil(info.bits)
        XCTAssertEqual(info.label, "fp16")
    }

    // MARK: Provenance + JSONL

    func sampleProvenance(nonce: String = "abc123") -> Provenance {
        Provenance(
            date: "2026-07-09T10:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1024 * 1024 * 1024,
            hardwareOS: "macOS 14.5",
            harnessGitSHA: "deadbeef",
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: "0.30.1",
            referenceMLXLMVersion: "0.28.0",
            modelPath: "/models/Qwen3-32B-4bit",
            modelConfigHash: fnv1a64(Array("config-bytes".utf8)),
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: "measurement-corpus-v1",
            corpusContentHash: "57dacafb4a7317be",
            nonce: nonce)
    }

    struct SamplePayload: Codable, Sendable, Equatable {
        let klMedian: Double
        let positions: Int
    }

    func testResultRecordEncodesAsSingleValidJSONLine() throws {
        let record = ResultRecord(subcommand: "kl", provenance: sampleProvenance(), payload: SamplePayload(klMedian: 0.0012, positions: 96))
        let line = try record.jsonLine()
        XCTAssertFalse(line.contains("\n"))
        let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        XCTAssertEqual(obj?["subcommand"] as? String, "kl")
        let payload = obj?["payload"] as? [String: Any]
        XCTAssertEqual(payload?["positions"] as? Int, 96)
        let provenance = obj?["provenance"] as? [String: Any]
        XCTAssertEqual(provenance?["corpusId"] as? String, "measurement-corpus-v1")
        XCTAssertEqual(provenance?["mlxSwiftVersion"] as? String, "0.31.6")
    }

    func testResultRecordRoundTripsThroughDecoding() throws {
        let original = ResultRecord(subcommand: "verify", provenance: sampleProvenance(), payload: SamplePayload(klMedian: 0.5, positions: 10))
        let line = try original.jsonLine()
        let decoded = try JSONDecoder().decode(ResultRecord<SamplePayload>.self, from: Data(line.utf8))
        XCTAssertEqual(decoded.subcommand, "verify")
        XCTAssertEqual(decoded.payload, SamplePayload(klMedian: 0.5, positions: 10))
        XCTAssertEqual(decoded.provenance, sampleProvenance())
    }

    func testResultRecordIsAppendOnlyFriendlyAcrossMultipleLines() throws {
        // Two records concatenated with a real newline must each independently parse as JSON —
        // i.e. neither record's own encoding embeds a stray newline that would corrupt the file.
        let a = try ResultRecord(subcommand: "kl", provenance: sampleProvenance(nonce: "n1"), payload: SamplePayload(klMedian: 0.1, positions: 1)).jsonLine()
        let b = try ResultRecord(subcommand: "kl", provenance: sampleProvenance(nonce: "n2"), payload: SamplePayload(klMedian: 0.2, positions: 2)).jsonLine()
        let file = a + "\n" + b + "\n"
        let lines = file.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(lines.count, 2)
        for line in lines {
            XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(line.utf8)))
        }
    }

    func testDifferentProvenanceProducesDifferentEncodedLines() throws {
        let a = try ResultRecord(subcommand: "kl", provenance: sampleProvenance(nonce: "n1"), payload: SamplePayload(klMedian: 0.1, positions: 1)).jsonLine()
        let b = try ResultRecord(subcommand: "kl", provenance: sampleProvenance(nonce: "n2"), payload: SamplePayload(klMedian: 0.1, positions: 1)).jsonLine()
        XCTAssertNotEqual(a, b, "nonce differs -> encoded record must differ")
    }
}
