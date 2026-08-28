import XCTest
import HarnessCore
import MLXLMCommon

@testable import fastmlx_harness

final class QwenMTPSampledBlockTraceCLITests: XCTestCase {
    func testArgumentsRequireExactPairAndFreshEvidence() throws {
        let parsed = try parseQwenMTPSampledBlockTraceArguments([
            "--target", "target",
            "--drafter", "drafter",
            "--evidence", "block-trace.jsonl",
        ])

        XCTAssertEqual(parsed.targetPath, "target")
        XCTAssertEqual(parsed.drafterPath, "drafter")
        XCTAssertEqual(parsed.evidencePath, "block-trace.jsonl")
    }

    func testArgumentsFailClosedWithoutEchoingValues() {
        XCTAssertThrowsError(try parseQwenMTPSampledBlockTraceArguments([
            "--target", "private-input/target",
            "--drafter", "private-input/drafter",
        ])) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceCLIError, .missingFlag("--evidence"))
            XCTAssertFalse(qwenMTPSampledBlockTraceExternalDiagnostic(error).contains("private-input"))
        }

        XCTAssertThrowsError(try parseQwenMTPSampledBlockTraceArguments([
            "--target", "target",
            "--drafter", "drafter",
            "--evidence", "block-trace.jsonl",
            "--force", "true",
        ])) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceCLIError, .unknownFlag)
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
        let evidence = directory.appendingPathComponent("block-trace.jsonl")

        XCTAssertThrowsError(
            try writeValidatedFreshSampledBlockTrace(Data("{}\n".utf8), to: evidence))
        XCTAssertFalse(FileManager.default.fileExists(atPath: evidence.path))
    }

    func testExternalDiagnosticDoesNotRouteToOldSampledTraceName() {
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(QwenMTPSampledBlockTraceCLIError.releaseBuildRequired),
            "sampled MTP block trace requires a Release build")
    }

    func testExternalDiagnosticReportsSanitizedDiagnosticInvariant() {
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(
                SampledMTPBlockDiagnosticError.recurrentRewindFailed),
            "sampled MTP block diagnostic failed: recurrentRewindFailed")
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(
                SampledMTPBlockDiagnosticError.distributionsMatch(stepIndex: 1)),
            "sampled MTP block diagnostic failed: distributionsMatch(stepIndex: 1)")
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(
                QwenMTPSampledBlockTraceGateError.cacheMismatch(
                    outcome: .rejectSecond, layer: 7)),
            "sampled MTP block trace gate failed: cacheMismatch(rejectSecond, layer: 7)")
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(
                SampledMTPBlockAcceptanceError.missingAcceptanceUniform(index: 1)),
            "sampled MTP block acceptance failed: missingAcceptanceUniform(index: 1)")
        XCTAssertEqual(
            qwenMTPSampledBlockTraceExternalDiagnostic(
                SampledMTPResidualCorrectionError.invalidResidualUniform),
            "sampled MTP residual correction failed: invalidResidualUniform")
    }
}
