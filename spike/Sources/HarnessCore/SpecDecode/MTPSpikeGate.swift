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
/// What is guaranteed: every emitted token is the target model's OWN argmax as evaluated by the
/// speculative verify forward (see `spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/
/// MTPSpeculativeTokenIterator.swift:715-736` — the drafter's proposal is only ever used as an equality
/// predicate, never substituted for the target's own choice). What is NOT guaranteed: token-identity
/// with the non-MTP scalar route. The verify forward evaluates `blockSize + 1` positions per round
/// while the scalar route evaluates one position per forward, so the two routes never share forward
/// geometry after prefill — a real-weight A/B measured a deterministic, divergent, shorter stream
/// (289 vs 318 tokens) at temperature 0 (cycle 79, `docs/task-inbox/
/// 2026-09-07-mtp-scalar-route-divergence-DECISION.md`). Route-vs-scalar exactness is therefore reported
/// as a diagnostic (`MTPStreamExactness`), not enforced as an acceptance criterion.

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

/// The promote / shelve decision for the MTP spike, combining the acceptance economics with a
/// vacuous-run guard into a dated machine-readable evidence line (the project's evidence-artifact
/// pattern). Route-vs-scalar token exactness is reported (`exactness`/`evidenceLine()`) but is NOT
/// a promote/shelve input — see the file header and `decide` below for why.
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
            exactness: exactness, acceptance: acceptance, modeledSpeedup: modeledSpeedup)
    }

    private static func decide(
        exactness: MTPStreamExactness.Result, acceptance: SpeculativeAcceptanceSummary,
        modeledSpeedup: Double
    ) -> Verdict {
        // No data compared → cannot judge anything at all.
        guard exactness.comparedTokens > 0 else { return .indeterminate }
        // A run where the drafter never proposed anything is scalar-vs-scalar by construction, so it
        // would trivially satisfy route-vs-scalar token agreement without exercising speculation at
        // all. That is a vacuous promote, not evidence the MTP route works — catch it before the
        // economics check (cycle-79 fail-closed-gates-enumerate-legitimate-refusals correction).
        guard acceptance.proposedDraftTokens > 0 else { return .indeterminate }
        // Promote only when the modeled speedup actually beats the non-MTP loop. Route-vs-scalar token
        // agreement (`exactness.exact`) is NOT a promote/shelve criterion: the MTP verify forward and
        // the scalar route never share forward geometry after prefill, so exactness is architecturally
        // unavailable even for a correct integration (cycle 79 measurement, see file header).
        guard modeledSpeedup.isFinite, modeledSpeedup > 1 else { return .shelve }
        return .promote
    }

    /// Machine-readable evidence line (space-separated `key=value`, matching the `fit_*` conventions in
    /// `FitCheckMeasuredReport.machineReadableFields`). Frozen keys: `mtp_route_token_agreement`,
    /// `mtp_accept_rate`, `mtp_modeled_speedup`, `mtp_verdict` (plus `mtp_first_divergence`/
    /// `mtp_length_matched` context). `mtp_route_token_agreement` is a REPORTED diagnostic — it is NOT
    /// a promote/shelve criterion (see `decide` above).
    public func evidenceLine() -> String {
        let acceptRate = acceptance.proposalAcceptanceRate.map { String(format: "%.4f", $0) } ?? "na"
        let speedup = modeledSpeedup.isFinite ? String(format: "%.4f", modeledSpeedup) : "na"
        let divergence = exactness.firstDivergenceIndex.map(String.init) ?? "-1"
        return "mtp_route_token_agreement=\(exactness.exact) mtp_first_divergence=\(divergence) "
            + "mtp_length_matched=\(exactness.lengthMatched) mtp_compared_tokens=\(exactness.comparedTokens) "
            + "mtp_accept_rate=\(acceptRate) mtp_modeled_speedup=\(speedup) mtp_verdict=\(verdict.rawValue)"
    }
}
