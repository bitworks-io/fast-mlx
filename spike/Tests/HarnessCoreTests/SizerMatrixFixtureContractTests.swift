import XCTest
@testable import HarnessCore

/// Contract test for the `sizer-matrix/v1` artifact — mirrors
/// `QuantReliabilityFixtureContractTests`. `Fixtures/sizer-matrix-v1-sample.json` is the checked-in
/// wire-format truth: this proves the decoder (fail-closed on a foreign schema) round-trips it, and
/// that `SizerMatrixArtifact.build`'s live encode path stays consistent with the same decoder.
final class SizerMatrixFixtureContractTests: XCTestCase {

    private func fixtureData() throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "sizer-matrix-v1-sample", withExtension: "json"),
            "fixture sizer-matrix-v1-sample.json must be bundled (Package.swift resources)")
        return try Data(contentsOf: url)
    }

    func testFixtureDecodesValidated() throws {
        let artifact = try SizerMatrixArtifact.decodeValidated(from: fixtureData())
        XCTAssertEqual(artifact.schema, SizerMatrixArtifact.schemaTag)
        XCTAssertEqual(artifact.host.label, "m5Max128")
        XCTAssertTrue(artifact.host.wiredLimitIsMeasured)
        XCTAssertEqual(artifact.kvQuant, "fp16")
        XCTAssertEqual(artifact.concurrency, 1)
        XCTAssertEqual(artifact.rows.count, 4)

        let eightBit = try XCTUnwrap(
            artifact.rows.first { $0.modelID == "Qwen3-8B" && $0.weightBits == 8 })
        XCTAssertEqual(eightBit.classification, "yellow")
        XCTAssertTrue(eightBit.fits)

        let bigRed = try XCTUnwrap(
            artifact.rows.first { $0.modelID == "DeepSeek-V3-671B" && $0.weightBits == 8 })
        XCTAssertFalse(bigRed.fits)
        XCTAssertFalse(bigRed.estimateIsMeasured)
    }

    func testForeignSchemaFailsClosed() throws {
        let foreign = """
        {"schema":"sizer-matrix/v99","host":{"label":"x","ramBytes":1,"wiredLimitBytes":1,"wiredLimitIsMeasured":true},"kvQuant":"fp16","concurrency":1,"rows":[]}
        """
        let data = try XCTUnwrap(foreign.data(using: .utf8))
        XCTAssertThrowsError(try SizerMatrixArtifact.decodeValidated(from: data))
    }

    func testRoundTripEncodeDecode() throws {
        let built = SizerMatrixArtifact.build(
            box: .m5Max128, boxLabel: "m5Max128", context: nil, kvQuant: .fp16, concurrency: 1)
        let json = built.encodedJSON()
        let data = try XCTUnwrap(json.data(using: String.Encoding.utf8))
        let decoded = try SizerMatrixArtifact.decodeValidated(from: data)

        XCTAssertEqual(decoded.schema, SizerMatrixArtifact.schemaTag)
        XCTAssertFalse(decoded.rows.isEmpty)
        XCTAssertEqual(decoded.rows.count, built.rows.count)
        XCTAssertEqual(decoded.rows.first?.modelID, built.rows.first?.modelID)
        XCTAssertEqual(decoded.rows.first?.totalPeakBytes, built.rows.first?.totalPeakBytes)
        XCTAssertEqual(decoded.host.label, "m5Max128")
    }
}
