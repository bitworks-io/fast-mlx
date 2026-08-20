import XCTest
@testable import HarnessCore

/// Unit tests for the measured-vs-modeled drift report — the last unshipped clause of the
/// fit-checked-serve differentiator. Pure/off-box: constructs a `CapacityPrediction` and feeds plain
/// measured byte counts, no MLX. Each test maps to the acceptance criterion "surface modeled vs
/// measured with an honest drift verdict."
final class FitCheckMeasuredReportTests: XCTestCase {
    private let gib = 1_073_741_824

    /// A modeled prediction with the given per-term GiB (whole-GiB inputs keep the arithmetic exact).
    private func prediction(weightsGiB: Double, kvGiB: Double, transientGiB: Double, headroomGiB: Double,
                            derivable: Bool = true) -> CapacityPrediction {
        let g = Double(gib)
        return CapacityPrediction(
            modelID: "test", modelType: .uniformGQA, nativeMaxContext: 131072, context: 32768,
            concurrency: 1, weightsBytes: weightsGiB * g, kvBytes: kvGiB * g,
            transientPrefillPeakBytes: transientGiB * g, allocatorHeadroomBytes: headroomGiB * g,
            derivable: derivable)
    }

    // Modeled peak = 10 + 4 + 1 + 2 = 17 GiB for all drift-band tests below.
    private func peak17() -> CapacityPrediction {
        prediction(weightsGiB: 10, kvGiB: 4, transientGiB: 1, headroomGiB: 2)
    }

    /// measured peak well BELOW modeled → the sizer reserved more than the run used → `.conservative`
    /// (the fit-check's intended safe bias), negative drift fraction.
    func testConservative_measuredBelowModeled() {
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 12 * gib, measuredActiveBytes: 11 * gib,
            measuredCacheBytes: 1 * gib)
        XCTAssertEqual(r.modeledPeakBytes, 17 * gib)
        XCTAssertEqual(r.drift, .conservative)
        XCTAssertEqual(r.deltaBytes, (12 - 17) * gib)
        XCTAssertEqual(r.deltaFraction, Double(-5 * gib) / Double(17 * gib), accuracy: 1e-9)
    }

    /// measured peak ABOVE modeled beyond tolerance → the run used more than modeled → `.underpredicted`
    /// (the calibration concern the line exists to surface loudly), positive drift fraction.
    func testUnderpredicted_measuredAboveModeled() {
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 20 * gib, measuredActiveBytes: 18 * gib,
            measuredCacheBytes: 2 * gib)
        XCTAssertEqual(r.drift, .underpredicted)
        XCTAssertEqual(r.deltaBytes, 3 * gib)
        XCTAssertGreaterThan(r.deltaFraction, 0.10)
    }

    /// measured within ±tolerance of modeled → `.accurate`.
    func testAccurate_withinTolerance() {
        // 16 GiB vs 17 GiB modeled → −5.9%, inside the default ±10% band.
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 16 * gib, measuredActiveBytes: 15 * gib,
            measuredCacheBytes: 1 * gib)
        XCTAssertEqual(r.drift, .accurate)
    }

    /// The tolerance band is exclusive at exactly ±tolerance: a fraction of exactly the tolerance reads
    /// `.accurate` (only strictly beyond flips the verdict), so callers get a stable center band.
    func testToleranceBoundary_exactlyAtEdgeIsAccurate() {
        // modeled 10 GiB, measured 11 GiB → +10.0% exactly with tolerance 0.10.
        let p = prediction(weightsGiB: 6, kvGiB: 2, transientGiB: 1, headroomGiB: 1)  // 10 GiB
        let r = FitCheckMeasuredReport(
            prediction: p, measuredPeakBytes: 11 * gib, measuredActiveBytes: 10 * gib,
            measuredCacheBytes: 1 * gib, toleranceFraction: 0.10)
        XCTAssertEqual(r.deltaFraction, 0.10, accuracy: 1e-9)
        XCTAssertEqual(r.drift, .accurate, "exactly +tolerance is not yet underpredicted")
    }

    /// A non-derivable / zero-modeled prediction (unsupported/novel arch the sizer refused to size)
    /// yields `.indeterminate` and a 0 fraction — no false verdict from a divide-by-zero.
    func testIndeterminate_zeroModeledPeak() {
        let zero = prediction(weightsGiB: 0, kvGiB: 0, transientGiB: 0, headroomGiB: 0, derivable: false)
        let r = FitCheckMeasuredReport(
            prediction: zero, measuredPeakBytes: 5 * gib, measuredActiveBytes: 5 * gib,
            measuredCacheBytes: 0)
        XCTAssertEqual(r.drift, .indeterminate)
        XCTAssertEqual(r.deltaFraction, 0)
        XCTAssertEqual(r.measuredPeakBytes, 5 * gib, "measured is still reported without a verdict")
    }

    /// The modeled term breakdown mirrors the prediction (the "term-by-term breakdown" recon flagged as
    /// missing from the serve output).
    func testModeledTermBreakdown_matchesPrediction() {
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 12 * gib, measuredActiveBytes: 11 * gib,
            measuredCacheBytes: 1 * gib)
        XCTAssertEqual(r.modeledWeightsBytes, 10 * gib)
        XCTAssertEqual(r.modeledKVBytes, 4 * gib)
        XCTAssertEqual(r.modeledTransientBytes, 1 * gib)
        XCTAssertEqual(r.modeledHeadroomBytes, 2 * gib)
        XCTAssertEqual(r.modeledWeightsBytes + r.modeledKVBytes + r.modeledTransientBytes + r.modeledHeadroomBytes,
            r.modeledPeakBytes, "terms sum to the modeled peak")
    }

    /// The machine-readable line carries the frozen keys a gate script parses.
    func testMachineReadableFields_carriesKeys() {
        let fields = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 20 * gib, measuredActiveBytes: 18 * gib,
            measuredCacheBytes: 2 * gib).machineReadableFields()
        for key in ["fit_modeled_peak_bytes=", "fit_measured_peak_bytes=", "fit_drift=underpredicted",
                    "fit_drift_frac=", "fit_modeled_weights_bytes=", "fit_modeled_kv_bytes=",
                    "fit_modeled_transient_bytes=", "fit_modeled_headroom_bytes=",
                    "fit_measured_active_bytes=", "fit_measured_cache_bytes="] {
            XCTAssertTrue(fields.contains(key), "missing \(key) in: \(fields)")
        }
    }

    /// The human line names the verdict and formats GiB.
    func testSummaryLine_humanReadable() {
        let line = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 12 * gib, measuredActiveBytes: 11 * gib,
            measuredCacheBytes: 1 * gib).summaryLine()
        XCTAssertTrue(line.contains("CONSERVATIVE"), line)
        XCTAssertTrue(line.contains("measured=12.00 GiB"), line)
        XCTAssertTrue(line.contains("modeled=17.00 GiB"), line)
        XCTAssertTrue(line.contains("weights=10.00 GiB"), line)
    }

    /// End-to-end through the real predictor: a measured snapshot at exactly the modeled peak is accurate.
    func testFromRealPredictPeakBytes_selfConsistentAtModeledPeak() {
        guard let profile = ModelArchProfile.catalog.first(where: { $0.isKVDerivable }) else {
            return XCTFail("no derivable catalog profile")
        }
        let host = SystemProfile.detectHost()
        let pred = CapacityModel.predictPeakBytes(
            model: profile, context: 32768, concurrency: 1, kvQuant: .fp16, profile: host)
        let modeledPeak = Int(pred.totalBytes.rounded())
        let r = FitCheckMeasuredReport(
            prediction: pred, measuredPeakBytes: modeledPeak, measuredActiveBytes: modeledPeak,
            measuredCacheBytes: 0)
        XCTAssertEqual(r.drift, .accurate, "measured == modeled peak is accurate")
        XCTAssertEqual(r.deltaBytes, 0)
    }
}
