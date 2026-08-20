import Foundation

/// One quant candidate's measured tool-call reliability, ready to render into a report line.
/// `quantBits` mirrors `QuantCandidateEvaluation.quantBits` (`nil` = unquantized) rather than being
/// folded into `ToolCallReliabilityScore` itself — the score type stays a pure measurement, agnostic
/// of which candidate produced it.
public struct QuantReliabilityRow: Sendable {
    public let repoID: String
    public let quantBits: Int?
    public let score: ToolCallReliabilityScore
    /// `false` when the source measurement did not record repair data (e.g. the current
    /// `scripts/bench-tool-calling.py` probe, which measures no repairs). The renderer then emits
    /// `repair=n/a` rather than the score's placeholder `0.0` — never present an unmeasured field as
    /// a measured `0.00`. Defaults to `true` so hand-built rows (build #1) render the score's value.
    public let repairRateIsMeasured: Bool

    public init(
        repoID: String, quantBits: Int?, score: ToolCallReliabilityScore,
        repairRateIsMeasured: Bool = true
    ) {
        self.repoID = repoID
        self.quantBits = quantBits
        self.score = score
        self.repairRateIsMeasured = repairRateIsMeasured
    }
}

/// Pure renderer for per-quant tool-call-reliability rows (build #1 of the per-quant reliability
/// rows work): given measured scores for each quant candidate of a model, render one bits-labeled
/// summary line each. Mirrors `QuantPickResult.summaryLines()` in style and reuses
/// `QuantAutoPicker`'s ordering semantics (bits desc, `nil`/unquantized ranks first, repoID
/// tiebreak) so a reliability report reads consistently with a fit-check announce. Nothing wires
/// this up to a live measurement path yet — it renders whatever rows it is handed and never
/// fabricates one.
public struct QuantReliabilityReport: Sendable {
    public let rows: [QuantReliabilityRow]

    public init(rows: [QuantReliabilityRow]) {
        self.rows = rows
    }

    /// One line per row (bits, trigger/name/args/repair rates, sample count), sorted by the same
    /// quality-rank ordering as `QuantAutoPicker` (bits desc, `nil` first, repoID tiebreak). An
    /// empty report renders an explicit "no measurements" line rather than fabricating a row.
    public func summaryLines() -> [String] {
        guard !rows.isEmpty else {
            return ["quant reliability report: no measurements"]
        }
        var lines: [String] = ["quant reliability report: \(rows.count) candidate(s)"]
        for row in rows.sorted(by: Self.isBetter) {
            let bits = row.quantBits.map { "\($0)-bit" } ?? "unquantized"
            let s = row.score
            let repair = row.repairRateIsMeasured ? rate(s.repairRate) : "n/a"
            lines.append(
                "  \(row.repoID) [\(bits)] trigger=\(rate(s.triggerRate)) name=\(rate(s.nameAccuracy)) "
                    + "args=\(rate(s.argumentValidityRate)) repair=\(repair) n=\(s.caseCount)")
        }
        return lines
    }

    private func rate(_ v: Double) -> String { String(format: "%.2f", v) }

    /// `true` when `lhs` should rank ahead of `rhs`: higher quant bits first, `nil` (unquantized)
    /// ranking highest, repoID as the deterministic tiebreak. Mirrors
    /// `QuantAutoPicker.qualityRank`/`isBetter`'s bits-and-repoID ordering.
    private static func isBetter(_ lhs: QuantReliabilityRow, _ rhs: QuantReliabilityRow) -> Bool {
        let lq = qualityRank(lhs.quantBits), rq = qualityRank(rhs.quantBits)
        if lq != rq { return lq > rq }
        return lhs.repoID < rhs.repoID
    }

    /// Higher = better quality. `nil` (no quantization block) is an unquantized checkpoint — the
    /// highest quality — so it ranks above any finite bit width. Mirrors
    /// `QuantAutoPicker.qualityRank`.
    private static func qualityRank(_ bits: Int?) -> Int { bits ?? Int.max }
}
