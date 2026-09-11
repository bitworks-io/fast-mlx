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

    /// measured FOOTPRINT (peak+cache) well BELOW modeled → the sizer reserved more than the run
    /// used → `.conservative` (the fit-check's intended safe bias), negative drift fraction.
    /// `deltaBytes`/`deltaFraction` are footprint-based (12+1-17 GiB), not peak-only (12-17 GiB) —
    /// this is the coupling this increment introduces (see `testFootprintCoupling_...` below for the
    /// case where that distinction actually flips the verdict).
    func testConservative_measuredBelowModeled() {
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 12 * gib, measuredActiveBytes: 11 * gib,
            measuredCacheBytes: 1 * gib)
        XCTAssertEqual(r.modeledPeakBytes, 17 * gib)
        XCTAssertEqual(r.measuredFootprintBytes, 13 * gib)
        XCTAssertEqual(r.drift, .conservative)
        XCTAssertEqual(r.deltaBytes, (13 - 17) * gib)
        XCTAssertEqual(r.deltaFraction, Double(-4 * gib) / Double(17 * gib), accuracy: 1e-9)
    }

    /// measured FOOTPRINT ABOVE modeled beyond tolerance → the run used more than modeled →
    /// `.underpredicted` (the calibration concern the line exists to surface loudly), positive drift
    /// fraction. `deltaBytes` is footprint-based (20+2-17 GiB), not peak-only (20-17 GiB).
    func testUnderpredicted_measuredAboveModeled() {
        let r = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 20 * gib, measuredActiveBytes: 18 * gib,
            measuredCacheBytes: 2 * gib)
        XCTAssertEqual(r.measuredFootprintBytes, 22 * gib)
        XCTAssertEqual(r.drift, .underpredicted)
        XCTAssertEqual(r.deltaBytes, 5 * gib)
        XCTAssertGreaterThan(r.deltaFraction, 0.10)
    }

    // MARK: - Behavior 1: drift must be classified against the measured FOOTPRINT (peak+cache), not
    // peak alone — MLX's `Memory.peakMemory` structurally excludes cache/pool bytes, so a run whose
    // cache footprint is large enough can classify `.conservative`/`.accurate` "for free" against
    // peak alone while it is genuinely `.underpredicted` against what MLX actually allocated.

    /// The regression this increment fixes, demonstrated end to end: the SAME measured numbers read
    /// `.conservative` if you (wrongly) classify against `measuredPeakBytes` alone, and
    /// `.underpredicted` once `measuredCacheBytes` is folded into the comparison.
    ///
    /// TDD note for reviewers: before this increment's `classify`-against-`measuredFootprintBytes`
    /// change landed, this exact assertion (`r.drift == .underpredicted`) FAILED — the code computed
    /// `.conservative` instead (peak alone: (76−90)/90 ≈ −15.6%, inside the safe/negative band).
    /// That failure was the red bar; folding `measuredCacheBytes` into `deltaBytes`/`drift` is what
    /// turns it green. Numbers are illustrative (not a literal reproduction of any specific
    /// production host), chosen with comfortable margins on both sides of ±10%.
    func testFootprintCoupling_cacheRevealsUnderpredictionPeakAloneWouldHide() {
        let modeled = prediction(weightsGiB: 60, kvGiB: 20, transientGiB: 8, headroomGiB: 2) // 90 GiB
        let r = FitCheckMeasuredReport(
            prediction: modeled, measuredPeakBytes: 76 * gib, measuredActiveBytes: 74 * gib,
            measuredCacheBytes: 24 * gib)
        XCTAssertEqual(r.measuredFootprintBytes, 100 * gib)

        // Peak alone would have read `.conservative` (this is the bug this increment fixes — kept
        // here as an explicit, permanent witness, not just asserted transiently during TDD).
        let peakOnlyFrac = Double(r.measuredPeakBytes - r.modeledPeakBytes) / Double(r.modeledPeakBytes)
        XCTAssertLessThan(peakOnlyFrac, -0.10, "precondition: peak alone would misreport .conservative")

        XCTAssertEqual(r.drift, .underpredicted,
            "peak+cache footprint exceeds modeled peak by >10% — the run used more than modeled")
    }

    /// Behavior 1, acceptance criterion 2: `measuredCacheBytes == 0` must render drift IDENTICAL to
    /// classifying against `measuredPeakBytes` alone — no silent regression for callers that don't
    /// (yet) supply a cache sample. `measuredFootprintBytes` collapses to `measuredPeakBytes` exactly.
    func testFootprintCoupling_zeroCacheIsIdenticalToPeakOnlyBaseline() {
        let modeled = peak17()
        let peakOnly = Double(20 * gib - 17 * gib) / Double(17 * gib)
        let r = FitCheckMeasuredReport(
            prediction: modeled, measuredPeakBytes: 20 * gib, measuredActiveBytes: 18 * gib,
            measuredCacheBytes: 0)
        XCTAssertEqual(r.measuredFootprintBytes, r.measuredPeakBytes, "cache=0 ⇒ footprint == peak")
        XCTAssertEqual(r.deltaFraction, peakOnly, accuracy: 1e-9)
        XCTAssertEqual(r.drift, .underpredicted, "unchanged from the pre-increment peak-only verdict")
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
    /// `measuredCacheBytes: 0` here is deliberate (not an oversight): the boundary math below is
    /// stated in terms of the measured PEAK alone, so a nonzero cache would shift the footprint off
    /// the exact +10% edge this test exists to pin — cache=0 keeps footprint == peak.
    func testToleranceBoundary_exactlyAtEdgeIsAccurate() {
        // modeled 10 GiB, measured 11 GiB → +10.0% exactly with tolerance 0.10.
        let p = prediction(weightsGiB: 6, kvGiB: 2, transientGiB: 1, headroomGiB: 1)  // 10 GiB
        let r = FitCheckMeasuredReport(
            prediction: p, measuredPeakBytes: 11 * gib, measuredActiveBytes: 10 * gib,
            measuredCacheBytes: 0, toleranceFraction: 0.10)
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
                    "fit_measured_active_bytes=", "fit_measured_cache_bytes=",
                    "fit_measured_footprint_bytes="] {
            XCTAssertTrue(fields.contains(key), "missing \(key) in: \(fields)")
        }
    }

    /// The human line names the verdict, formats GiB, and reports the footprint (peak+cache) that
    /// `drift` was actually classified against, plus the raw peak/active/cache terms it was built
    /// from — so an operator reading the line can reconstruct why the verdict landed where it did.
    func testSummaryLine_humanReadable() {
        let line = FitCheckMeasuredReport(
            prediction: peak17(), measuredPeakBytes: 12 * gib, measuredActiveBytes: 11 * gib,
            measuredCacheBytes: 1 * gib).summaryLine()
        XCTAssertTrue(line.contains("CONSERVATIVE"), line)
        XCTAssertTrue(line.contains("footprint(peak+cache) measured=13.00 GiB"), line)
        XCTAssertTrue(line.contains("modeled=17.00 GiB"), line)
        XCTAssertTrue(line.contains("weights=10.00 GiB"), line)
        XCTAssertTrue(line.contains("peak=12.00 GiB"), line)
        XCTAssertTrue(line.contains("cache=1.00 GiB"), line)
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

    // MARK: - Behavior 2: ServingFitPlanner.decide() names the allocator-headroom vs cache-limit gap
    //
    // Colocated in this file rather than ServingFitPlannerTests.swift because this increment's write
    // set names only this test file — see the increment's task packet. `ServingFitPlanner`/
    // `CapacityModel` are in the same `HarnessCore` module, so `@testable import HarnessCore` above
    // covers them.

    private func fitPlannerProfile(_ id: String) -> ModelArchProfile {
        guard let m = ModelArchProfile.catalog.first(where: { $0.id == id }) else {
            fatalError("missing catalog entry \(id)")
        }
        return m
    }

    /// The production-shaped geometry the defect was found against: a shared 128 GiB host with a
    /// measured wired limit ABOVE the 75%-RAM shared cap (100 GiB observed vs 96 GiB shared budget —
    /// mirrors the real production host, see `WiredCeilingOvercommitGuard`'s doc comment), so the
    /// shared policy's exact-75% cap binds at exactly 96 GiB. `CapacityModel.predictPeakBytes` never
    /// receives an `allocatorHeadroomBytes` override at this call site, so it prices the allocator at
    /// its hardcoded 2 GiB default, while `recommendedCacheLimitBytes(96 GiB)` == 12 GiB (`96/8`,
    /// inside the 4–24 GiB band) — 2 GiB priced vs 12 GiB entitled, a 10 GiB gap.
    private var productionShapedHost: SystemProfile {
        SystemProfile(chip: "test shared production", totalRAMBytes: 128 * gib, wiredLimitBytes: 100 * gib,
            wiredLimitIsMeasured: true)
    }

    /// Advisory fires (`under_entitled`) exactly on the geometry the defect was found against:
    /// headroom (2 GiB, hardcoded default) < cache limit (12 GiB, derived from the 96 GiB shared
    /// budget). Asserts on the machine-readable token, not prose — a wording edit must not silently
    /// break the log-scraper contract.
    func testAllocatorHeadroomAdvisory_firesUnderEntitled_productionShapedGap() {
        let d = ServingFitPlanner.decide(
            profile: fitPlannerProfile("Qwen3.8-Flash-Next (n-gram offload)"), weightsAreMeasured: true,
            host: productionShapedHost, requestedContext: 262_144)
        guard case .underEntitled(let headroom, let cacheLimit, let gap) = d.allocatorHeadroomState else {
            return XCTFail("expected .underEntitled, got \(d.allocatorHeadroomState)")
        }
        XCTAssertEqual(headroom, 2 * gib)
        XCTAssertEqual(cacheLimit, 12 * gib)
        XCTAssertEqual(gap, 10 * gib)

        let lines = ServingFitPlanner.allocatorHeadroomAdvisoryLines(for: d.allocatorHeadroomState)
        let text = lines.joined(separator: "\n")
        XCTAssertTrue(text.contains("allocator_headroom_state=under_entitled"), text)
        XCTAssertFalse(text.contains("allocator_headroom_state=consistent"), text)
        XCTAssertTrue(text.contains("2147483648"), "raw headroom bytes present: \(text)") // 2 GiB
        XCTAssertTrue(text.contains("12884901888"), "raw cache-limit bytes present: \(text)") // 12 GiB
    }

    /// Advisory does NOT fire `under_entitled` — and DOES emit the `consistent` line — when the
    /// modeled headroom is at/above the runtime's cache entitlement. A tiny host whose derived cache
    /// limit falls below the 2 GiB allocator-headroom default (`recommendedCacheLimitBytes` floors at
    /// `wiredLimitBytes/2` before its normal 4 GiB floor) exercises the other branch. Asserting the
    /// `consistent` line is emitted (not just the absence of `under_entitled`) is the point: a silent
    /// gate here would be indistinguishable from a gate that never ran.
    func testAllocatorHeadroomAdvisory_consistentWhenHeadroomAtOrAboveCacheLimit() {
        let tinyHost = SystemProfile(
            chip: "test tiny dedicated", totalRAMBytes: 4 * gib, wiredLimitBytes: 3 * gib,
            wiredLimitIsMeasured: true, hostUse: .operatorAssertedDedicatedServing())
        let cacheLimit = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: 3 * gib)
        XCTAssertLessThan(cacheLimit, 2 * gib, "precondition: this host's cache limit undercuts the 2 GiB headroom default")

        let d = ServingFitPlanner.decide(
            profile: fitPlannerProfile("Phi-4-14B"), weightsAreMeasured: true, host: tinyHost, force: true)
        guard case .consistent(let headroom, let reportedCacheLimit) = d.allocatorHeadroomState else {
            return XCTFail("expected .consistent, got \(d.allocatorHeadroomState)")
        }
        XCTAssertEqual(headroom, 2 * gib)
        XCTAssertEqual(reportedCacheLimit, cacheLimit)

        let lines = ServingFitPlanner.allocatorHeadroomAdvisoryLines(for: d.allocatorHeadroomState)
        let text = lines.joined(separator: "\n")
        XCTAssertTrue(text.contains("allocator_headroom_state=consistent"), text)
        XCTAssertFalse(text.contains("allocator_headroom_state=under_entitled"), text)
        XCTAssertFalse(lines.isEmpty, "the healthy state must still render a line, not silence")
    }

    // MARK: - Non-arming pin
    //
    // This increment must NOT change `CapacityModel.classify`/`shouldProceed`/the verdict colour, or
    // the value `predictPeakBytes` returns — folding the real cache entitlement into the GATING
    // prediction is a separate, operator-coordinated increment. This test pins today's verdict on the
    // exact production-shaped geometry the advisory above fires on, so a future arming change cannot
    // silently flip it without this test failing first.

    /// Production-shaped decision (96 GiB effective shared ceiling, ~75.8 GiB weights, context
    /// 262144): `shouldProceed`/`color` are pinned to their CURRENT values. If a future change makes
    /// this test fail, that is either (a) an accidental regression, or (b) the deliberate arming
    /// increment — which must update this pin explicitly, not incidentally.
    func testNonArmingPin_productionShapedDecision_verdictUnchanged() {
        let d = ServingFitPlanner.decide(
            profile: fitPlannerProfile("Qwen3.8-Flash-Next (n-gram offload)"), weightsAreMeasured: true,
            host: productionShapedHost, requestedContext: 262_144)
        XCTAssertEqual(d.color, .green, "pinned: today's verdict for this exact geometry")
        XCTAssertTrue(d.shouldProceed, "pinned: today's verdict for this exact geometry")
        XCTAssertFalse(d.proceedingUnderForce, "pinned: today's verdict for this exact geometry")
        XCTAssertEqual(d.contextWasCapped, false, "pinned: today's verdict for this exact geometry")
        XCTAssertEqual(d.servedContext, 262_144, "pinned: today's verdict for this exact geometry")
        // The advisory fires on this same decision (see the test above) WITHOUT moving any of the
        // pinned fields above — the whole point of "advisory only".
        XCTAssertEqual(d.allocatorHeadroomState, .underEntitled(
            headroomBytes: 2 * gib, cacheLimitBytes: 12 * gib, gapBytes: 10 * gib))
    }
}
