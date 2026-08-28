import XCTest

@testable import fastmlx_harness

final class QwenMTPSampledTraceCLITests: XCTestCase {
    func testArgumentsRequireExactPairAndFreshEvidence() throws {
        let parsed = try parseQwenMTPSampledTraceArguments([
            "--target", "target",
            "--drafter", "drafter",
            "--evidence", "trace.jsonl",
        ])

        XCTAssertEqual(parsed.targetPath, "target")
        XCTAssertEqual(parsed.drafterPath, "drafter")
        XCTAssertEqual(parsed.evidencePath, "trace.jsonl")
    }

    func testArgumentsFailClosedWithoutEchoingValues() {
        XCTAssertThrowsError(try parseQwenMTPSampledTraceArguments([
            "--target", "private-input/target",
            "--drafter", "private-input/drafter",
        ])) { error in
            XCTAssertEqual(error as? QwenMTPSampledTraceCLIError, .missingFlag("--evidence"))
            XCTAssertFalse(qwenMTPSampledTraceExternalDiagnostic(error).contains("private-input"))
        }

        XCTAssertThrowsError(try parseQwenMTPSampledTraceArguments([
            "--target", "target",
            "--drafter", "drafter",
            "--evidence", "trace.jsonl",
            "--force", "true",
        ])) { error in
            XCTAssertEqual(error as? QwenMTPSampledTraceCLIError, .unknownFlag)
        }
    }

    func testValidatedWriterLeavesNoArtifactForInvalidFinalRecord() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        let evidence = directory.appendingPathComponent("trace.jsonl")

        XCTAssertThrowsError(
            try writeValidatedFreshSampledTrace(Data("{}\n".utf8), to: evidence))
        XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path))
    }
}
