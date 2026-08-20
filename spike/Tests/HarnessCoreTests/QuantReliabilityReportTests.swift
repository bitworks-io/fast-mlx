import XCTest
@testable import HarnessCore

/// TDD for a pure per-quant tool-call-reliability report (build #1 of the per-quant reliability
/// rows work): given measured `ToolCallReliabilityScore`s for several quant candidates of the same
/// model, render one bits-labeled summary line each. Mirrors `QuantPickResult.summaryLines()` in
/// style and reuses `QuantAutoPicker`'s ordering semantics (bits desc, nil/unquantized first,
/// repoID tiebreak) so a reliability report reads consistently with a fit-check announce.
final class QuantReliabilityReportTests: XCTestCase {
    private let corpus = ToolCallReliabilityScore.defaultToolCallReliabilityCorpus

    /// A perfect observation set: every expected call fires with the correct name and complete
    /// arguments, the chit-chat case emits nothing, and nothing needed repair.
    private func cleanObservations() -> [ToolCallObservation] {
        [
            ToolCallObservation(
                id: "weather-single-city", emittedToolName: "get_weather",
                argumentsJSON: #"{"location":"Boston"}"#, neededRepair: false),
            ToolCallObservation(
                id: "verify-identity-two-arg", emittedToolName: "verify_identity",
                argumentsJSON: #"{"email":"a@b.com","order_number":"123"}"#, neededRepair: false),
            ToolCallObservation(
                id: "chit-chat-no-call", emittedToolName: nil, argumentsJSON: nil, neededRepair: false),
            ToolCallObservation(
                id: "list-supported-timezones", emittedToolName: "list_timezones",
                argumentsJSON: "{}", neededRepair: false),
        ]
    }

    /// A degraded observation set: the two-argument call fires but drops a required key (fails
    /// argument validity) and needed repair.
    private func degradedObservations() -> [ToolCallObservation] {
        [
            ToolCallObservation(
                id: "weather-single-city", emittedToolName: "get_weather",
                argumentsJSON: #"{"location":"Boston"}"#, neededRepair: false),
            ToolCallObservation(
                id: "verify-identity-two-arg", emittedToolName: "verify_identity",
                argumentsJSON: #"{"email":"a@b.com"}"#, neededRepair: true),
            ToolCallObservation(
                id: "chit-chat-no-call", emittedToolName: nil, argumentsJSON: nil, neededRepair: false),
            ToolCallObservation(
                id: "list-supported-timezones", emittedToolName: "list_timezones",
                argumentsJSON: "{}", neededRepair: false),
        ]
    }

    // MARK: - one line per candidate, bits-labeled, fields present

    func testOneLinePerCandidateWithBitsLabelAndScoreFields() {
        let cleanScore = ToolCallReliabilityScore.score(expectations: corpus, observations: cleanObservations())
        let degradedScore = ToolCallReliabilityScore.score(expectations: corpus, observations: degradedObservations())
        XCTAssertLessThan(degradedScore.argumentValidityRate, cleanScore.argumentValidityRate)
        XCTAssertGreaterThan(degradedScore.repairRate, cleanScore.repairRate)

        let report = QuantReliabilityReport(rows: [
            QuantReliabilityRow(repoID: "repo-8bit", quantBits: 8, score: cleanScore),
            QuantReliabilityRow(repoID: "repo-4bit", quantBits: 4, score: degradedScore),
        ])
        let lines = report.summaryLines()

        // one line per candidate (no header/footer expected beyond the rows themselves being
        // exactly the candidate count)
        let candidateLines = lines.filter { $0.contains("repo-8bit") || $0.contains("repo-4bit") }
        XCTAssertEqual(candidateLines.count, 2)

        let eightBitLine = lines.first { $0.contains("repo-8bit") }
        let fourBitLine = lines.first { $0.contains("repo-4bit") }
        XCTAssertNotNil(eightBitLine)
        XCTAssertNotNil(fourBitLine)
        XCTAssertTrue(eightBitLine!.contains("[8-bit]"))
        XCTAssertTrue(fourBitLine!.contains("[4-bit]"))

        // score fields present on each line
        for line in [eightBitLine!, fourBitLine!] {
            XCTAssertTrue(line.contains("trigger"), "expected trigger field in: \(line)")
            XCTAssertTrue(line.contains("name"), "expected name field in: \(line)")
            XCTAssertTrue(line.contains("args"), "expected args field in: \(line)")
            XCTAssertTrue(line.contains("repair"), "expected repair field in: \(line)")
            XCTAssertTrue(line.contains("n="), "expected n= field in: \(line)")
        }
    }

    // MARK: - deterministic ordering: bits desc, nil (unquantized) first, repoID tiebreak

    func testOrderingMatchesQualityRankSemantics() {
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: cleanObservations())
        let report = QuantReliabilityReport(rows: [
            QuantReliabilityRow(repoID: "repo-8bit", quantBits: 8, score: score),
            QuantReliabilityRow(repoID: "repo-4bit", quantBits: 4, score: score),
            QuantReliabilityRow(repoID: "repo-unquantized", quantBits: nil, score: score),
        ])
        let lines = report.summaryLines()
        let candidateLines = lines.filter {
            $0.contains("repo-8bit") || $0.contains("repo-4bit") || $0.contains("repo-unquantized")
        }
        XCTAssertEqual(candidateLines.count, 3)
        // nil (unquantized) ranks first, then descending bits, matching QuantAutoPicker.qualityRank.
        XCTAssertTrue(candidateLines[0].contains("repo-unquantized"))
        XCTAssertTrue(candidateLines[1].contains("repo-8bit"))
        XCTAssertTrue(candidateLines[2].contains("repo-4bit"))
        XCTAssertTrue(candidateLines[0].contains("unquantized"))
    }

    // MARK: - empty report renders an explicit no-measurements line, never a fabricated row

    func testEmptyReportRendersNoMeasurementsLine() {
        let report = QuantReliabilityReport(rows: [])
        let lines = report.summaryLines()
        XCTAssertTrue(
            lines.contains { $0.contains("no measurements") },
            "expected an explicit 'no measurements' line, got: \(lines)")
    }
}
