import Foundation

/// The last unshipped clause of the fit-checked-serve differentiator: surface the sizer's MODELED peak
/// (and its term breakdown) against the MEASURED live allocator bytes, so an operator sees not just
/// "we modeled it" but "here is proof the model was right (or where it drifted)."
///
/// PURE and MLX-FREE by construction: it consumes a `CapacityPrediction` (HarnessCore) plus plain
/// measured byte counts (the caller reads those from the MLX allocator snapshot —
/// `Memory.snapshot().{peakMemory,activeMemory,cacheMemory}` — and passes them in), so it unit-tests
/// off-box with no MLX dependency, mirroring `ServingFitDecision`.
///
/// Drift semantics (measured peak vs modeled peak, the safety-relevant direction):
/// - `.conservative`  — measured came in BELOW the modeled peak by more than the tolerance band: the
///   sizer reserved more than the run used. Safe (the fit-check's intended bias) — never a surprise OOM.
/// - `.underpredicted` — measured EXCEEDED the modeled peak by more than the tolerance: the run used
///   more than the model predicted. The calibration concern — if the model said GREEN and the true
///   peak overshoots, the host envelope was closer than reported. Worth surfacing loudly.
/// - `.accurate`      — within ±tolerance: the model tracked reality.
/// - `.indeterminate` — the modeled peak is not derivable (0), so no ratio can be formed (e.g. an
///   unsupported/novel arch the sizer refused to size). Report the measured bytes without a verdict.
public struct FitCheckMeasuredReport: Sendable {
    public enum Drift: String, Sendable {
        case accurate
        case conservative
        case underpredicted
        case indeterminate
    }

    /// Modeled peak = `CapacityPrediction.totalBytes` (weights + KV + transient prefill + allocator
    /// headroom), rounded to whole bytes.
    public let modeledPeakBytes: Int
    public let modeledWeightsBytes: Int
    public let modeledKVBytes: Int
    public let modeledTransientBytes: Int
    public let modeledHeadroomBytes: Int

    /// Measured live-allocator bytes the caller sampled after load (peak is the run high-water mark).
    public let measuredPeakBytes: Int
    public let measuredActiveBytes: Int
    public let measuredCacheBytes: Int

    /// `measuredPeakBytes − modeledPeakBytes` (signed): positive = the run used MORE than modeled.
    public let deltaBytes: Int
    /// `deltaBytes / modeledPeakBytes` (0 when indeterminate). Positive = underpredicted direction.
    public let deltaFraction: Double
    public let toleranceFraction: Double
    public let drift: Drift

    /// - Parameters:
    ///   - prediction: the sizer's modeled peak + term breakdown (from `CapacityModel.predictPeakBytes`).
    ///   - measuredPeakBytes/measuredActiveBytes/measuredCacheBytes: the live MLX allocator snapshot.
    ///   - toleranceFraction: the ± band (of the modeled peak) within which drift reads `.accurate`.
    ///     Default 0.10 (10%) — wider than allocator jitter, narrow enough to catch a real mis-model.
    public init(
        prediction: CapacityPrediction,
        measuredPeakBytes: Int,
        measuredActiveBytes: Int,
        measuredCacheBytes: Int,
        toleranceFraction: Double = 0.10
    ) {
        let modeledPeak = Int(prediction.totalBytes.rounded())
        self.modeledPeakBytes = modeledPeak
        self.modeledWeightsBytes = Int(prediction.weightsBytes.rounded())
        self.modeledKVBytes = Int(prediction.kvBytes.rounded())
        self.modeledTransientBytes = Int(prediction.transientPrefillPeakBytes.rounded())
        self.modeledHeadroomBytes = Int(prediction.allocatorHeadroomBytes.rounded())
        self.measuredPeakBytes = measuredPeakBytes
        self.measuredActiveBytes = measuredActiveBytes
        self.measuredCacheBytes = measuredCacheBytes
        self.toleranceFraction = toleranceFraction

        let delta = measuredPeakBytes - modeledPeak
        self.deltaBytes = delta
        if modeledPeak <= 0 {
            self.deltaFraction = 0
            self.drift = .indeterminate
        } else {
            let frac = Double(delta) / Double(modeledPeak)
            self.deltaFraction = frac
            if frac > toleranceFraction {
                self.drift = .underpredicted
            } else if frac < -toleranceFraction {
                self.drift = .conservative
            } else {
                self.drift = .accurate
            }
        }
    }

    private static func gib(_ b: Int) -> String { String(format: "%.2f GiB", Double(b) / 1_073_741_824.0) }

    /// Operator-facing one-liner, GiB-formatted, for the serve startup/snapshot announce.
    public func summaryLine() -> String {
        let pct = String(format: "%+.1f%%", deltaFraction * 100)
        let verdict = drift.rawValue.uppercased()
        return "  measured-vs-modeled: peak measured=\(Self.gib(measuredPeakBytes)) "
            + "modeled=\(Self.gib(modeledPeakBytes)) (Δ \(pct), \(verdict)); "
            + "modeled terms: weights=\(Self.gib(modeledWeightsBytes)) kv=\(Self.gib(modeledKVBytes)) "
            + "transient=\(Self.gib(modeledTransientBytes)) headroom=\(Self.gib(modeledHeadroomBytes)); "
            + "measured: active=\(Self.gib(measuredActiveBytes)) cache=\(Self.gib(measuredCacheBytes))"
    }

    /// Machine-readable `fit_*` fields for the startup line (space-separated `key=value`, matching
    /// `ServingFitDecision.machineReadableFields()` conventions). Additive — a gate script can parse the
    /// drift verdict and the modeled/measured peaks without the human line.
    public func machineReadableFields() -> String {
        let frac = String(format: "%+.4f", deltaFraction)
        return "fit_modeled_peak_bytes=\(modeledPeakBytes) fit_measured_peak_bytes=\(measuredPeakBytes) "
            + "fit_measured_active_bytes=\(measuredActiveBytes) fit_measured_cache_bytes=\(measuredCacheBytes) "
            + "fit_drift=\(drift.rawValue) fit_drift_frac=\(frac) "
            + "fit_modeled_weights_bytes=\(modeledWeightsBytes) fit_modeled_kv_bytes=\(modeledKVBytes) "
            + "fit_modeled_transient_bytes=\(modeledTransientBytes) fit_modeled_headroom_bytes=\(modeledHeadroomBytes)"
    }
}
