import XCTest
@testable import HarnessCore

/// TDD for the artifact RENDERER (the off-box CLI seam for roadmap #4's per-quant reliability rows).
/// Build #2 gave a permissive decoder (`QuantReliabilityArtifact`, schema optional); this renderer is
/// the layer a CLI (`fastmlx-capacity --quant-reliability <file>`) calls to turn artifact BYTES into
/// display lines. It adds the one thing the frozen permissive decoder deliberately does NOT do:
/// FAIL CLOSED on a foreign `schema` tag, so a `quant-reliability/v2` file is never silently rendered
/// as if it were v1. Absent schema stays allowed (forward-compat with hand-built/older artifacts).
final class QuantReliabilityRenderingTests: XCTestCase {

    private let v1JSON = Data(#"""
    {
      "schema": "quant-reliability/v1",
      "model": "Qwen3-8B",
      "quants": [
        {
          "repo_id": "org/Qwen3-8B-MLX-4bit",
          "quant_bits": 4,
          "reliability": {
            "cases": 4, "trigger_rate": 1.0, "name_accuracy": 1.0, "arg_validity_rate": 0.5
          }
        },
        {
          "repo_id": "org/Qwen3-8B-MLX-8bit",
          "quant_bits": 8,
          "reliability": {
            "cases": 4, "trigger_rate": 1.0, "name_accuracy": 1.0, "arg_validity_rate": 1.0,
            "repair_rate": 0.0
          }
        }
      ]
    }
    """#.utf8)

    /// Happy path: a provenance header names the model, and the two candidate rows render below it
    /// in the report's quality-rank order (8-bit ahead of 4-bit).
    func testRendersProvenanceHeaderAndRows() throws {
        let lines = try QuantReliabilityArtifactRenderer.renderLines(from: v1JSON)
        let header = try XCTUnwrap(lines.first)
        XCTAssertTrue(header.contains("Qwen3-8B"), "header must name the model, got: \(header)")

        let candidateLines = lines.filter { $0.contains("Qwen3-8B-MLX") }
        XCTAssertEqual(candidateLines.count, 2)
        XCTAssertTrue(candidateLines[0].contains("8bit"), "8-bit ranks first, got: \(candidateLines)")
        XCTAssertTrue(candidateLines[1].contains("4bit"), "4-bit ranks second, got: \(candidateLines)")
    }

    /// Honesty invariant survives the renderer: the 4-bit entry omitted `repair_rate` → `repair=n/a`,
    /// never a fabricated `0.00`; the 8-bit entry measured `0.0` → renders `repair=0.00`.
    func testRendererPreservesRepairHonesty() throws {
        let lines = try QuantReliabilityArtifactRenderer.renderLines(from: v1JSON)
        let fourBit = try XCTUnwrap(lines.first { $0.contains("4bit") })
        let eightBit = try XCTUnwrap(lines.first { $0.contains("8bit") })
        XCTAssertTrue(fourBit.contains("repair=n/a"), "unmeasured repair must render n/a: \(fourBit)")
        XCTAssertFalse(fourBit.contains("repair=0.00"), "must not fabricate 0.00 repair: \(fourBit)")
        XCTAssertTrue(eightBit.contains("repair=0.00"), "measured 0.0 renders its value: \(eightBit)")
    }

    /// An absent `schema` is allowed (older/hand-built artifacts) and still renders.
    func testAbsentSchemaIsAccepted() throws {
        let noSchema = Data(#"""
        {
          "model": "M",
          "quants": [
            { "repo_id": "org/M", "quant_bits": 4,
              "reliability": { "cases": 1, "trigger_rate": 1.0, "name_accuracy": 1.0, "arg_validity_rate": 1.0 } }
          ]
        }
        """#.utf8)
        let lines = try QuantReliabilityArtifactRenderer.renderLines(from: noSchema)
        XCTAssertTrue(lines.contains { $0.contains("org/M") })
    }

    /// A FOREIGN schema tag must fail closed — never render a v2 artifact as if it were v1.
    func testUnsupportedSchemaFailsClosed() {
        let v2 = Data(#"""
        { "schema": "quant-reliability/v2", "model": "M", "quants": [] }
        """#.utf8)
        XCTAssertThrowsError(try QuantReliabilityArtifactRenderer.renderLines(from: v2)) { error in
            guard case QuantReliabilityArtifactRenderer.RenderError.unsupportedSchema(let tag) = error else {
                return XCTFail("expected unsupportedSchema, got \(error)")
            }
            XCTAssertEqual(tag, "quant-reliability/v2")
        }
    }

    /// Malformed JSON must throw the renderer's own error (not leak a raw decoding error type),
    /// so the CLI can print one clean message and exit non-zero.
    func testMalformedJSONThrowsRenderError() {
        let junk = Data("{ not json".utf8)
        XCTAssertThrowsError(try QuantReliabilityArtifactRenderer.renderLines(from: junk)) { error in
            guard case QuantReliabilityArtifactRenderer.RenderError.malformedJSON = error else {
                return XCTFail("expected malformedJSON, got \(error)")
            }
        }
    }
}
