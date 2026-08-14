import XCTest
import Dispatch
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

    func testRequiredWriterAppendsCompleteDurableJSONLines() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("evidence.jsonl")
        let a = ResultRecord(
            subcommand: "kl", provenance: sampleProvenance(nonce: "n1"),
            payload: SamplePayload(klMedian: 0.1, positions: 1))
        let b = ResultRecord(
            subcommand: "kl", provenance: sampleProvenance(nonce: "n2"),
            payload: SamplePayload(klMedian: 0.2, positions: 2))

        try RequiredJSONLWriter.append(a, to: url)
        try RequiredJSONLWriter.append(b, to: url)

        let contents = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(contents.hasSuffix("\n"))
        let lines = contents.split(separator: "\n")
        XCTAssertEqual(lines.count, 2)
        for line in lines {
            XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(line.utf8)))
        }
    }

    func testRequiredWriterRefusesToAppendAfterCorruptPartialLine() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("evidence.jsonl")
        try Data("partial".utf8).write(to: url)
        let record = ResultRecord(
            subcommand: "kl", provenance: sampleProvenance(),
            payload: SamplePayload(klMedian: 0.1, positions: 1))

        XCTAssertThrowsError(try RequiredJSONLWriter.append(record, to: url))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "partial")
    }

    func testRequiredWriterRefusesMalformedCompleteExistingLine() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("evidence.jsonl")
        try Data("garbage\n".utf8).write(to: url)
        let record = ResultRecord(
            subcommand: "kl", provenance: sampleProvenance(),
            payload: SamplePayload(klMedian: 0.1, positions: 1))

        XCTAssertThrowsError(try RequiredJSONLWriter.append(record, to: url)) {
            XCTAssertEqual(
                $0 as? RequiredJSONLWriterError,
                .malformedExistingLine(line: 1))
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "garbage\n")
    }

    func testRequiredWriterRefusesJSONThatIsNotAResultRecordEnvelope() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("evidence.jsonl")
        try Data("{}\n".utf8).write(to: url)
        let record = ResultRecord(
            subcommand: "kl", provenance: sampleProvenance(),
            payload: SamplePayload(klMedian: 0.1, positions: 1))

        XCTAssertThrowsError(try RequiredJSONLWriter.append(record, to: url)) {
            XCTAssertEqual(
                $0 as? RequiredJSONLWriterError,
                .malformedExistingLine(line: 1))
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "{}\n")
    }

    func testRequiredWriterSerializesConcurrentAppendsWithoutLosingRows() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("evidence.jsonl")
        let writes = 128
        let template = sampleProvenance()

        DispatchQueue.concurrentPerform(iterations: writes) { index in
            let provenance = Provenance(
                date: template.date, hardwareChip: template.hardwareChip,
                hardwareRAMBytes: template.hardwareRAMBytes,
                hardwareOS: template.hardwareOS,
                harnessGitSHA: template.harnessGitSHA,
                mlxSwiftVersion: template.mlxSwiftVersion,
                referenceMLXVersion: template.referenceMLXVersion,
                referenceMLXLMVersion: template.referenceMLXLMVersion,
                modelPath: template.modelPath,
                modelConfigHash: template.modelConfigHash,
                modelCheckpointManifestHash: template.modelCheckpointManifestHash,
                modelQuant: template.modelQuant,
                corpusId: template.corpusId,
                corpusContentHash: template.corpusContentHash,
                nonce: "n\(index)")
            let record = ResultRecord(
                subcommand: "kl", provenance: provenance,
                payload: SamplePayload(klMedian: Double(index), positions: index + 1))
            try! RequiredJSONLWriter.append(record, to: url)
        }

        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.last, 0x0a)
        let records = try data.split(separator: 0x0a).map {
            try JSONDecoder().decode(ResultRecord<SamplePayload>.self, from: Data($0))
        }
        XCTAssertEqual(records.count, writes)
        XCTAssertEqual(Set(records.map(\.provenance.nonce)).count, writes)
        XCTAssertEqual(Set(records.map(\.payload.positions)).count, writes)
    }

    // MARK: harness git SHA resolution (deployed hosts have no .git; the deploy step writes
    // a .harness-sha file the binary reads — env var stays the explicit override)

    func testSHAResolutionPrefersEnv() {
        XCTAssertEqual(
            resolveHarnessGitSHA(env: "aaa111", shaFile: "bbb222\n", gitOutput: "ccc333"),
            "aaa111")
    }

    func testSHAResolutionFallsBackToShaFileWhenLiveGitIsUnavailable() {
        XCTAssertEqual(resolveHarnessGitSHA(env: nil, shaFile: "bbb222\n", gitOutput: nil), "bbb222")
        XCTAssertEqual(resolveHarnessGitSHA(env: nil, shaFile: nil, gitOutput: "ccc333\n"), "ccc333")
    }

    func testSHAResolutionPrefersLiveGitOverStaleDeployedStamp() {
        XCTAssertEqual(
            resolveHarnessGitSHA(
                env: nil, shaFile: "stale-stamp\n", gitOutput: "live-head-dirty\n"),
            "live-head-dirty")
    }

    func testSHAResolutionTrimsAndRejectsEmptyOrMultilineValues() {
        XCTAssertEqual(resolveHarnessGitSHA(env: "  ", shaFile: "  bbb222  \n", gitOutput: nil), "bbb222")
        // a multi-line "SHA" is not a SHA (e.g. an error message captured into the file)
        XCTAssertEqual(resolveHarnessGitSHA(env: nil, shaFile: "fatal: not a git repo\nbbb\n", gitOutput: nil), "unknown")
        XCTAssertEqual(resolveHarnessGitSHA(env: nil, shaFile: nil, gitOutput: nil), "unknown")
    }

    func testSHAResolutionKeepsDirtySuffix() {
        XCTAssertEqual(
            resolveHarnessGitSHA(env: nil, shaFile: "60d84fa-dirty\n", gitOutput: nil),
            "60d84fa-dirty")
    }
}
