import Foundation

/// The pure judgment kernel for the MTP self-speculative-decode spike (roadmap #3 latency bet — the
/// fix for the measured 18 tok/s / thinking-on-45s-timeout). MTP self-speculation carries the SAME
/// failure mode that shelved EAGLE-3 (the qwen3-32b EAGLE-3 preflight verdict
/// — failed greedy exactness at 4/8-bit): a fast draft head that quietly changes the emitted tokens.
///
/// The on-box measurement is M5-gated (the Qwen3.8-27B-MTP checkpoints are 27B-class), so this file
/// pre-builds everything EXCEPT the numbers: the eventual spike run only has to feed measured token
/// streams + a modeled speedup in and read the verdict out. No decision logic is written under the
/// time pressure of a live run — exactly how the sizer's honesty seams are kept measure-before-commit.
///
/// Exact by construction: greedy speculative decoding (`SpecAccept.walk`) emits byte-identical tokens
/// to the non-speculative target. So a divergence between the MTP stream and the non-MTP baseline is
/// never a tuning knob — it is a broken integration, and no throughput number buys it back.

/// Compares an MTP self-speculative greedy token stream against the non-MTP greedy baseline from the
/// SAME target model.
public enum MTPStreamExactness {
    public struct Result: Sendable, Equatable {
        /// True only when the streams match token-for-token AND have equal length.
        public let exact: Bool
        /// Number of leading positions actually compared (`min(candidate.count, baseline.count)`).
        public let comparedTokens: Int
        /// First index at which the compared prefix diverges; `nil` when the whole prefix matched
        /// (a pure length mismatch leaves this `nil` but sets `exact`/`lengthMatched` false).
        public let firstDivergenceIndex: Int?
        /// True when the two streams have the same length.
        public let lengthMatched: Bool

        public init(exact: Bool, comparedTokens: Int, firstDivergenceIndex: Int?, lengthMatched: Bool) {
            self.exact = exact
            self.comparedTokens = comparedTokens
            self.firstDivergenceIndex = firstDivergenceIndex
            self.lengthMatched = lengthMatched
        }
    }

    public static func compare(candidate: [Int], baseline: [Int]) -> Result {
        let compared = min(candidate.count, baseline.count)
        var divergence: Int?
        for i in 0..<compared where candidate[i] != baseline[i] {
            divergence = i
            break
        }
        let lengthMatched = candidate.count == baseline.count
        let exact = lengthMatched && divergence == nil
        return Result(
            exact: exact, comparedTokens: compared,
            firstDivergenceIndex: divergence, lengthMatched: lengthMatched)
    }
}

/// The promote / shelve decision for the MTP spike, combining the exactness veto with the acceptance
/// economics into a dated machine-readable evidence line (the project's evidence-artifact pattern).
public struct MTPSpikeGate: Sendable, Equatable {
    public enum Verdict: String, Sendable {
        case promote
        case shelve
        /// Not enough signal to decide (e.g. no tokens were compared).
        case indeterminate
    }

    public let exactness: MTPStreamExactness.Result
    public let acceptance: SpeculativeAcceptanceSummary
    /// The modeled end-to-end speedup vs the non-MTP loop, typically `SpeculativeEconomics.projectedSpeedup`
    /// (draft-head cost folded in). Kept as an input so this kernel stays a pure combiner over the
    /// already-tested economics types rather than re-deriving them.
    public let modeledSpeedup: Double
    public let verdict: Verdict

    public init(
        exactness: MTPStreamExactness.Result,
        acceptance: SpeculativeAcceptanceSummary,
        modeledSpeedup: Double
    ) {
        self.exactness = exactness
        self.acceptance = acceptance
        self.modeledSpeedup = modeledSpeedup
        self.verdict = Self.decide(
            exactness: exactness, modeledSpeedup: modeledSpeedup)
    }

    private static func decide(
        exactness: MTPStreamExactness.Result, modeledSpeedup: Double
    ) -> Verdict {
        // No data compared → cannot judge exactness at all.
        guard exactness.comparedTokens > 0 else { return .indeterminate }
        // The EAGLE-3 invariant: any greedy divergence shelves, regardless of speedup.
        guard exactness.exact else { return .shelve }
        // Exact — promote only when the modeled speedup actually beats the non-MTP loop.
        guard modeledSpeedup.isFinite, modeledSpeedup > 1 else { return .shelve }
        return .promote
    }

    /// Machine-readable evidence line (space-separated `key=value`, matching the `fit_*` conventions in
    /// `FitCheckMeasuredReport.machineReadableFields`). Frozen keys: `mtp_exact`, `mtp_accept_rate`,
    /// `mtp_modeled_speedup`, `mtp_verdict` (plus `mtp_first_divergence`/`mtp_length_matched` context).
    public func evidenceLine() -> String {
        let acceptRate = acceptance.proposalAcceptanceRate.map { String(format: "%.4f", $0) } ?? "na"
        let speedup = modeledSpeedup.isFinite ? String(format: "%.4f", modeledSpeedup) : "na"
        let divergence = exactness.firstDivergenceIndex.map(String.init) ?? "-1"
        return "mtp_exact=\(exactness.exact) mtp_first_divergence=\(divergence) "
            + "mtp_length_matched=\(exactness.lengthMatched) mtp_compared_tokens=\(exactness.comparedTokens) "
            + "mtp_accept_rate=\(acceptRate) mtp_modeled_speedup=\(speedup) mtp_verdict=\(verdict.rawValue)"
    }
}
