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

    func testEstimateIsMeasured_TrueForMeasuredDedicatedCeilingWithFp16() {
        let measuredDedicated = SystemProfile(
            chip: "measured-dedicated",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true,
            hostUse: .operatorAssertedDedicatedServing())
        let rows = ModelSizer.report(box: measuredDedicated, context: 32768, kvQuant: .fp16)
        XCTAssertTrue(rows.allSatisfy(\.estimateIsMeasured), "a measured dedicated ceiling + fp16 must report measured")
    }

    func testEstimateIsMeasured_FalseForSharedPolicyAndMetalCeilings() {
        let sharedPolicy = SystemProfile(
            chip: "shared-policy",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true)
        let metalBound = SystemProfile(
            chip: "metal-bound",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 90 * Int(gib),
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 80 * Int(gib))

        XCTAssertTrue(ModelSizer.report(box: sharedPolicy, context: 32768).allSatisfy { !$0.estimateIsMeasured })
        XCTAssertTrue(ModelSizer.report(box: metalBound, context: 32768).allSatisfy { !$0.estimateIsMeasured })
    }

    func testProvenanceNotesDistinguishSharedAndMetalCeilingsFromExperimentalKV() {
        let sharedPolicy = SystemProfile(
            chip: "shared-policy",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true)
        let metalBound = SystemProfile(
            chip: "metal-bound",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 90 * Int(gib),
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 80 * Int(gib))

        let sharedNotes = ModelSizer.provenanceNotes(box: sharedPolicy, kvQuant: .fp16)
        let metalNotes = ModelSizer.provenanceNotes(box: metalBound, kvQuant: .fp16)
        let experimentalNotes = ModelSizer.provenanceNotes(box: sharedPolicy, kvQuant: .tq2_5)

        XCTAssertEqual(sharedNotes, [
            "NOTE: the effective memory ceiling is synthesized by shared policy (not measured) — headroom numbers are approximate."
        ])
        XCTAssertEqual(metalNotes, [
            "NOTE: the effective memory ceiling is Metal's advisory recommended working set (not a measured hard limit) — headroom numbers are approximate."
        ])
        XCTAssertTrue(experimentalNotes.contains { $0.contains("EXPERIMENTAL/UNMEASURED placeholder KV tier") })
        XCTAssertFalse(sharedNotes.contains { $0.contains("EXPERIMENTAL/UNMEASURED placeholder KV tier") })
        XCTAssertFalse(metalNotes.contains { $0.contains("EXPERIMENTAL/UNMEASURED placeholder KV tier") })
    }

    func testEstimateIsMeasured_FalseForDetectedHost() {
        let host = SystemProfile.detectHost()
        let rows = ModelSizer.report(box: host, context: 32768, kvQuant: .fp16)
        XCTAssertTrue(rows.allSatisfy { !$0.estimateIsMeasured }, "a detected host's wired limit is always estimated")
    }

    func testEstimateIsMeasured_FalseForPlaceholderKVTier() {
        // The dedicated host's wired limit IS measured, but tq2_5 is an unmeasured placeholder KV tier —
        // estimateIsMeasured must still go false (either condition can sink the honesty flag).
        let host = SystemProfile(
            chip: "measured-dedicated",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true,
            hostUse: .operatorAssertedDedicatedServing())
        let rows = ModelSizer.report(box: host, context: 32768, kvQuant: .tq2_5)
        XCTAssertTrue(rows.allSatisfy { !$0.estimateIsMeasured }, "tq2_5 is an unmeasured placeholder tier")
    }

    // MARK: - SystemProfile.detectHost()

    func testDetectHost_PositiveRAMAndUnmeasuredWiredLimit() {
        let host = SystemProfile.detectHost()
        XCTAssertGreaterThan(host.totalRAMBytes, 0, "detected RAM must be positive")
        XCTAssertFalse(host.wiredLimitIsMeasured, "a detected host's wired limit is always an estimate")
        XCTAssertGreaterThan(host.wiredLimitBytes, 0, "estimated wired limit must be positive")
        XCTAssertLessThanOrEqual(host.wiredLimitBytes, host.totalRAMBytes, "estimate must not exceed total RAM")
        XCTAssertEqual(host.hostUse.rawValue, "shared")
        XCTAssertEqual(host.hostUse.source.rawValue, "automatic")
        XCTAssertEqual(host.hostUse.policyVersion, HostUseClassification.currentPolicyVersion)
    }

    func testHostUseDefaultsToSharedDefaultPolicy() {
        let host = SystemProfile(
            chip: "test", totalRAMBytes: 24 * Int(gib), wiredLimitBytes: 18 * Int(gib),
            wiredLimitIsMeasured: false)

        XCTAssertEqual(host.hostUse.rawValue, "shared")
        XCTAssertEqual(host.hostUse.source.rawValue, "default")
        XCTAssertEqual(host.hostUse.policyVersion, HostUseClassification.currentPolicyVersion)
    }

    func testDedicatedServingRequiresOperatorAssertionFactory() {
        let hostUse = HostUseClassification.operatorAssertedDedicatedServing()

        XCTAssertEqual(hostUse.rawValue, "dedicated-serving")
        XCTAssertEqual(hostUse.source.rawValue, "operator-assertion")
        XCTAssertEqual(hostUse.policyVersion, HostUseClassification.currentPolicyVersion)
    }

    func testHostUseDecodingRejectsInconsistentDedicatedServingSource() {
        let data = """
        {"use":"dedicated-serving","source":"automatic","policyVersion":"host-use/v1"}
        """.data(using: .utf8)!

        XCTAssertThrowsError(try JSONDecoder().decode(HostUseClassification.self, from: data))
    }

    func testHostUseDecodingRejectsUnknownPolicyVersion() {
        let data = """
        {"use":"shared","source":"default","policyVersion":"host-use/v999"}
        """.data(using: .utf8)!

        XCTAssertThrowsError(try JSONDecoder().decode(HostUseClassification.self, from: data))
    }

    func testEstimatedWiredLimitBytes_IsExactlySeventyFivePercentForSharedAutoHosts() {
        for ramGiB in [24, 128, 256] {
            let totalRAMBytes = ramGiB * Int(gib)
            XCTAssertEqual(
                SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: totalRAMBytes),
                (totalRAMBytes * 3) / 4,
                "\(ramGiB) GiB auto/default shared hosts must use floor(75% RAM)")
        }
    }

    func testEstimatedWiredLimitBytes_FractionNeverExceedsSeventyFivePercent() {
        let ramSizesGiB = [64, 128, 160, 192, 224, 256, 384, 512]
        for ramGiB in ramSizesGiB {
            let bytes = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: ramGiB * Int(gib))
            XCTAssertLessThanOrEqual(Double(bytes) / (Double(ramGiB) * gib), 0.75)
        }
    }

    func testSharedEffectiveCeilingUsesExactSeventyFivePercentWhenWiredIsHigherAndMetalAbsent() {
        let host = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * Int(gib),
            wiredLimitBytes: 115 * Int(gib),
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: nil)

        XCTAssertEqual(host.effectiveMemoryCeiling.bytes, 96 * Int(gib))
        XCTAssertEqual(host.effectiveMemoryCeiling.source, .sharedPolicy)
        XCTAssertTrue(
            host.wiredLimitIsMeasured,
            "effective shared-policy provenance must not rewrite the measured wired input fact")
    }
}
