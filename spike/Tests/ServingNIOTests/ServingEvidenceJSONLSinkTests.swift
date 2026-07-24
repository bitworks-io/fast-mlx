import Foundation
import XCTest

@testable import ServingCore
@testable import ServingNIO

final class ServingEvidenceJSONLSinkTests: XCTestCase {
    func testFreshSinkWritesOneCanonicalPromptFreeLineAndRefusesReuse() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("serving-evidence.jsonl")
        let promptSentinel = "PROMPT-SENTINEL-jsonl"
        let generatedSentinel = "GENERATED-SENTINEL-jsonl"
        let evidence = try makeEvidence(
            promptSentinel: promptSentinel,
            generatedSentinel: generatedSentinel)
        let sink = try ServingEvidenceJSONLSink(path: output.path)

        try await sink.record(evidence)
        try await sink.finish()

        let bytes = try Data(contentsOf: output)
        let canonical = try evidence.canonicalJSONData()
        var expected = canonical
        expected.append(0x0A)
        XCTAssertEqual(bytes, expected)
        let text = try XCTUnwrap(String(data: bytes, encoding: .utf8))
        XCTAssertFalse(text.contains(promptSentinel))
        XCTAssertFalse(text.contains(generatedSentinel))
        let permissions = try XCTUnwrap(
            try FileManager.default.attributesOfItem(
                atPath: output.path)[.posixPermissions] as? NSNumber)
        XCTAssertEqual(permissions.intValue & 0o777, 0o600)

        XCTAssertThrowsError(
            try ServingEvidenceJSONLSink(path: output.path)
        ) { error in
            XCTAssertEqual(
                error as? ServingEvidenceJSONLSink.Error,
                .outputAlreadyExists)
        }
    }

    func testSinkRejectsRelativeAndMissingParentPaths() {
        XCTAssertThrowsError(
            try ServingEvidenceJSONLSink(path: "relative.jsonl")
        ) { error in
            XCTAssertEqual(
                error as? ServingEvidenceJSONLSink.Error,
                .pathMustBeAbsolute)
        }
        XCTAssertThrowsError(
            try ServingEvidenceJSONLSink(
                path: "/tmp/\(UUID().uuidString)/missing/evidence.jsonl")
        ) { error in
            XCTAssertEqual(
                error as? ServingEvidenceJSONLSink.Error,
                .parentDirectoryMissing)
        }
    }

    func testPartialWriteRollsBackAndLatchesSinkFailure() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("serving-evidence.jsonl")
        let evidence = try makeEvidence(
            promptSentinel: "PROMPT-SENTINEL-partial",
            generatedSentinel: "GENERATED-SENTINEL-partial")
        let sink = try ServingEvidenceJSONLSink(
            path: output.path,
            simulatedWriteFailureAfterBytes: 32)

        do {
            try await sink.record(evidence)
            XCTFail("A simulated short write must fail closed")
        } catch {
            XCTAssertEqual(
                error as? ServingEvidenceJSONLSink.Error,
                .writeFailed)
        }
        XCTAssertEqual(try Data(contentsOf: output), Data())

        do {
            try await sink.record(evidence)
            XCTFail("A failed sink must remain terminal")
        } catch {
            XCTAssertEqual(
                error as? ServingEvidenceJSONLSink.Error,
                .writeFailed)
        }
    }
}

private func makeEvidence(
    promptSentinel: String,
    generatedSentinel: String
) throws -> ServingEvidence {
    try ServingEvidence(
        request: ServingEvidence.Request(
            method: "POST",
            path: "/v1/chat/completions",
            headers: [],
            body: Data(promptSentinel.utf8),
            stream: false,
            messageCount: 1,
            maxCompletionTokens: 1),
        response: ServingEvidence.Response(
            status: 200,
            durationMilliseconds: 1,
            chunkCount: 1,
            body: Data(generatedSentinel.utf8)))
}
