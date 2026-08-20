import XCTest
@testable import HarnessCore

/// TDD for the per-quant reliability artifact decoder (build #2 of the per-quant reliability rows
/// work). Build #1 rendered rows the caller hand-built; build #2 decodes them from a probe artifact
/// so build #3's live multi-quant collection (M5-gated) has a stable schema to emit into. The
/// artifact is a SEPARATE JSON file (not the bench CSV), chosen over widening `BenchRow`'s CSV header
/// so the schema is additive and reversible — see docs/task-inbox/2026-08-18-per-quant-tool-call-reliability-rows.md.
///
/// Honesty invariant under test: `scripts/bench-tool-calling.py` measures no repair data, so a quant
/// entry may omit `repair_rate`; the decoder must render `repair=n/a`, never fabricate `0.00`.
final class QuantReliabilityArtifactTests: XCTestCase {

    /// A two-quant artifact: an 8-bit entry WITH a measured repair_rate and a 4-bit entry WITHOUT
    /// one (the shape the current probe emits). Field names match the Python probe's `reliability`
    /// block (`cases`, `trigger_rate`, `name_accuracy`, `arg_validity_rate`, optional `repair_rate`).
    private let twoQuantJSON = Data(#"""
    {
      "schema": "quant-reliability/v1",
      "model": "Qwen3-8B",
      "quants": [
        {
          "repo_id": "org/Qwen3-8B-MLX-4bit",
          "quant_bits": 4,
          "reliability": {
            "cases": 4,
            "trigger_rate": 1.0,
            "name_accuracy": 1.0,
            "arg_validity_rate": 0.5
          }
        },
        {
          "repo_id": "org/Qwen3-8B-MLX-8bit",
          "quant_bits": 8,
          "reliability": {
            "cases": 4,
            "trigger_rate": 1.0,
            "name_accuracy": 1.0,
            "arg_validity_rate": 1.0,
            "repair_rate": 0.0
          }
        }
      ]
    }
    """#.utf8)

    func testDecodesTwoQuantFixtureIntoSortedReportRows() throws {
        let artifact = try QuantReliabilityArtifact.decode(from: twoQuantJSON)
        XCTAssertEqual(artifact.model, "Qwen3-8B")
        XCTAssertEqual(artifact.quants.count, 2)

        let rows = artifact.rows()
        XCTAssertEqual(rows.count, 2)

        let report = QuantReliabilityReport(rows: rows)
        let lines = report.summaryLines()
        let candidateLines = lines.filter { $0.contains("Qwen3-8B-MLX") }
        XCTAssertEqual(candidateLines.count, 2)

        // Ordering: 8-bit ranks ahead of 4-bit (bits desc), same as QuantAutoPicker.qualityRank.
        XCTAssertTrue(candidateLines[0].contains("8bit"), "expected 8-bit first, got: \(candidateLines)")
        XCTAssertTrue(candidateLines[1].contains("4bit"), "expected 4-bit second, got: \(candidateLines)")

        // Honesty: the 4-bit entry omitted repair_rate → render n/a, NOT 0.00.
        let fourBitLine = lines.first { $0.contains("4bit") }
        let eightBitLine = lines.first { $0.contains("8bit") }
        XCTAssertNotNil(fourBitLine)
        XCTAssertNotNil(eightBitLine)
        XCTAssertTrue(fourBitLine!.contains("repair=n/a"), "unmeasured repair must render n/a, got: \(fourBitLine!)")
        XCTAssertFalse(fourBitLine!.contains("repair=0.00"), "must not fabricate a 0.00 repair rate: \(fourBitLine!)")
        // The 8-bit entry DID measure repair (0.0) → render the number, not n/a.
        XCTAssertTrue(eightBitLine!.contains("repair=0.00"), "measured repair must render its value, got: \(eightBitLine!)")
    }

    /// The score fields survive the round trip (args=0.50 for the degraded 4-bit entry).
    func testDecodedRowsCarryMeasuredRates() throws {
        let artifact = try QuantReliabilityArtifact.decode(from: twoQuantJSON)
        let rows = artifact.rows()
        let fourBit = try XCTUnwrap(rows.first { $0.quantBits == 4 })
        XCTAssertEqual(fourBit.score.argumentValidityRate, 0.5, accuracy: 1e-9)
        XCTAssertEqual(fourBit.score.caseCount, 4)
        XCTAssertFalse(fourBit.repairRateIsMeasured)

        let eightBit = try XCTUnwrap(rows.first { $0.quantBits == 8 })
        XCTAssertTrue(eightBit.repairRateIsMeasured)
    }
}
