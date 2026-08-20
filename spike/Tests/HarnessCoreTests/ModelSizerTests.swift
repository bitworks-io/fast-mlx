import XCTest
@testable import HarnessCore

/// "Model sizer v1" (llmfit-like moat pillar): `ModelSizer.report` wraps the existing
/// `CapacityModel` math — these tests assert the wiring (fit transitions, honesty flag,
/// `detectHost()` shape), not the underlying capacity arithmetic (already covered by
/// `CapacityModelTests`, ±3% tolerance style reused where byte numbers are asserted directly).
final class ModelSizerTests: XCTestCase {
    private let gib = 1024.0 * 1024.0 * 1024.0

    private func fit(_ rows: [ModelFit], modelID: String, weightBits: Int) -> ModelFit {
        guard let row = rows.first(where: { $0.modelID == modelID && $0.weightBits == weightBits }) else {
            XCTFail("missing ModelFit row for \(modelID) @\(weightBits)-bit")
            fatalError("missing ModelFit row for \(modelID) @\(weightBits)-bit")
        }
        return row
    }

    // MARK: - report() basics

    func testReport_M5Max128_ReturnsOneOrMoreFits() {
        let rows = ModelSizer.report(box: .m5Max128, context: 32768)
        XCTAssertGreaterThan(rows.count, 0)
        // Default weightBitOptions is [4, 8]: every cataloged model should get both rows.
        XCTAssertEqual(rows.count, ModelArchProfile.catalog.count * 2)
    }

    /// A large model (GLM-4.6, ~177.5 GiB @4-bit weights alone) does NOT fit at 8-bit on the
    /// 128GB bench box; a small model (Mistral-Small-3.2-24B, ~12 GiB @4-bit) DOES fit at 8-bit —
    /// the boolean `fits` must transition rather than reporting a uniform verdict.
    func testFitsTransitions_LargeModelDoesNotFit_SmallModelDoes_At8Bit() {
        let rows = ModelSizer.report(box: .m5Max128, context: 32768)

        let large = fit(rows, modelID: "GLM-4.6", weightBits: 8)
        XCTAssertFalse(large.fits, "GLM-4.6 @8-bit must not fit on the 128GB bench box")

        let small = fit(rows, modelID: "Mistral-Small-3.2-24B", weightBits: 8)
        XCTAssertTrue(small.fits, "Mistral-Small-3.2-24B @8-bit must fit on the 128GB bench box")
    }

    /// Weight-bit scaling is linear off the catalog's 4-bit estimate: 8-bit weights ≈ 2x 4-bit.
    func testWeightsBytesScalesLinearlyWithBits() {
        let rows = ModelSizer.report(box: .m5Max128, context: 32768)
        let bit4 = fit(rows, modelID: "Qwen3-32B", weightBits: 4)
        let bit8 = fit(rows, modelID: "Qwen3-32B", weightBits: 8)
        XCTAssertEqual(Double(bit8.weightsBytes), Double(bit4.weightsBytes) * 2.0, accuracy: 1.0)
    }

    // MARK: - estimateIsMeasured (the honest measured-vs-modeled flag)

    func testEstimateIsMeasured_TrueForM5Max128WithFp16() {
        let rows = ModelSizer.report(box: .m5Max128, context: 32768, kvQuant: .fp16)
        XCTAssertTrue(rows.allSatisfy(\.estimateIsMeasured), "m5Max128 + fp16 must report measured")
    }

    func testEstimateIsMeasured_FalseForDetectedHost() {
        let host = SystemProfile.detectHost()
        let rows = ModelSizer.report(box: host, context: 32768, kvQuant: .fp16)
        XCTAssertTrue(rows.allSatisfy { !$0.estimateIsMeasured }, "a detected host's wired limit is always estimated")
    }

    func testEstimateIsMeasured_FalseForPlaceholderKVTier() {
        // m5Max128's wiredLimit IS measured, but tq2_5 is an unmeasured placeholder KV tier —
        // estimateIsMeasured must still go false (either condition can sink the honesty flag).
        let rows = ModelSizer.report(box: .m5Max128, context: 32768, kvQuant: .tq2_5)
        XCTAssertTrue(rows.allSatisfy { !$0.estimateIsMeasured }, "tq2_5 is an unmeasured placeholder tier")
    }

    // MARK: - SystemProfile.detectHost()

    func testDetectHost_PositiveRAMAndUnmeasuredWiredLimit() {
        let host = SystemProfile.detectHost()
        XCTAssertGreaterThan(host.totalRAMBytes, 0, "detected RAM must be positive")
        XCTAssertFalse(host.wiredLimitIsMeasured, "a detected host's wired limit is always an estimate")
        XCTAssertGreaterThan(host.wiredLimitBytes, 0, "estimated wired limit must be positive")
        XCTAssertLessThanOrEqual(host.wiredLimitBytes, host.totalRAMBytes, "estimate must not exceed total RAM")
    }

    /// `estimatedWiredLimitBytes` reconciles the two data points this repo already has: ~0.90 at
    /// the 128GB scale (m5Max128's measured 115/128 ≈ 0.898), ~0.75 at 256GB+ (m3Ultra256/512's
    /// own documented planning assumption). ±3% tolerance, matching CapacityModelTests' style.
    func testEstimatedWiredLimitBytes_MatchesDocumentedEndpoints() {
        let at128 = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: 128 * Int(gib))
        XCTAssertEqual(Double(at128) / gib, 0.90 * 128, accuracy: 0.03 * 0.90 * 128)

        let at256 = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: 256 * Int(gib))
        XCTAssertEqual(Double(at256) / gib, 0.75 * 256, accuracy: 0.03 * 0.75 * 256)

        let at512 = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: 512 * Int(gib))
        XCTAssertEqual(Double(at512) / gib, 0.75 * 512, accuracy: 0.03 * 0.75 * 512)
    }

    /// Monotonic: a bigger box never gets a higher fraction-of-RAM wired-limit estimate.
    func testEstimatedWiredLimitBytes_FractionIsMonotonicNonIncreasing() {
        let ramSizesGiB = [64, 128, 160, 192, 224, 256, 384, 512]
        let fractions = ramSizesGiB.map { ramGiB -> Double in
            let bytes = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: ramGiB * Int(gib))
            return Double(bytes) / (Double(ramGiB) * gib)
        }
        for i in 1..<fractions.count {
            XCTAssertLessThanOrEqual(fractions[i], fractions[i - 1] + 1e-9, "fraction must not increase with RAM")
        }
    }
}
