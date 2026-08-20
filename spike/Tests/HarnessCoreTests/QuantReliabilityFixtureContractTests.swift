import XCTest
@testable import HarnessCore

/// Cross-language contract for the `quant-reliability/v1` artifact. A single checked-in fixture
/// (`Fixtures/quant-reliability-v1-sample.json`) is the shared truth both sides must agree on: this
/// Swift half proves the renderer decodes that exact fixture into the expected rows, and the Python
/// half (`scripts/tests/test_bench_tool_calling.py`) proves the emitter reproduces the same file
/// byte-for-byte. Schema drift on EITHER side turns one of these red. The fixture is what the Python
/// collector actually emits, so — honestly — it carries NO `repair_rate` (the probe measures none);
/// both quants therefore render `repair=n/a`.
final class QuantReliabilityFixtureContractTests: XCTestCase {

    private func fixtureData() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "quant-reliability-v1-sample", withExtension: "json"),
            "fixture quant-reliability-v1-sample.json must be bundled (Package.swift resources)")
        return try Data(contentsOf: url)
    }

    /// The fixture decodes to the two known quants with their measured rates intact.
    func testFixtureDecodesToExpectedRows() throws {
        let artifact = try QuantReliabilityArtifact.decode(from: fixtureData())
        XCTAssertEqual(artifact.schema, "quant-reliability/v1")
        XCTAssertEqual(artifact.model, "Qwen3-8B")

        let rows = artifact.rows()
        XCTAssertEqual(rows.count, 2)
        let fourBit = try XCTUnwrap(rows.first { $0.quantBits == 4 })
        XCTAssertEqual(fourBit.repoID, "org/Qwen3-8B-MLX-4bit")
        XCTAssertEqual(fourBit.score.argumentValidityRate, 0.5, accuracy: 1e-9)
        XCTAssertEqual(fourBit.score.caseCount, 4)
        // The emitter never writes repair_rate → every fixture row is honestly unmeasured.
        XCTAssertFalse(fourBit.repairRateIsMeasured)
        let eightBit = try XCTUnwrap(rows.first { $0.quantBits == 8 })
        XCTAssertFalse(eightBit.repairRateIsMeasured)
    }

    /// The renderer turns the fixture into the operator-facing rows: 8-bit ranks first, and both
    /// carry `repair=n/a` (unmeasured, never fabricated).
    func testFixtureRendersHonestRows() throws {
        let lines = try QuantReliabilityArtifactRenderer.renderLines(from: fixtureData())
        XCTAssertTrue(lines.first?.contains("Qwen3-8B") == true)
        let candidates = lines.filter { $0.contains("Qwen3-8B-MLX") }
        XCTAssertEqual(candidates.count, 2)
        XCTAssertTrue(candidates[0].contains("8bit"), "8-bit ranks first: \(candidates)")
        XCTAssertTrue(candidates.allSatisfy { $0.contains("repair=n/a") },
            "emitter measures no repair → every row is n/a: \(candidates)")
    }
}
