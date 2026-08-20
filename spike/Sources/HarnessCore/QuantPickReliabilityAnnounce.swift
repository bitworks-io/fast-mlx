import Foundation

/// A DISPLAY-ONLY reliability overlay for a quant auto-pick announce (roadmap #4). Given a
/// `QuantPickResult` (which candidate quant the fit-checker would serve) and the measured tool-call
/// reliability rows from an off-box `quant-reliability/v1` artifact, it annotates each candidate the
/// picker considered with its measured reliability so an operator sees fit AND quality side by side.
///
/// Two deliberate invariants:
///
///  - **Join by quant bits, never repo id.** In the serve path a pick candidate's `repoID` is its
///    absolute on-disk directory path (`QuantCandidateResolver` keys on `dir.path`), while the
///    artifact's repo id is the HF repo id — different namespaces that would never match. Quant bits
///    is the only reliable join key. An unmatched candidate is honestly marked `n/a (no measurement)`;
///    two measurements sharing a candidate's bits are `n/a (ambiguous)` — the composer never guesses.
///
///  - **Display-only.** This produces a SEPARATE block; it does not reorder candidates, pick a winner,
///    or restate the pick's winner line. Ranking and selection stay entirely in `QuantAutoPicker`. The
///    caller emits this block to STDERR so the `--quant-pick-only` STDOUT winner-line contract is
///    byte-identical with or without the overlay.
public enum QuantPickReliabilityAnnounce {

    /// Compose the overlay lines: a provenance header naming the artifact's model, then one annotated
    /// line per pick candidate (bits-joined to a measurement, or an explicit n/a).
    public static func compose(
        pick: QuantPickResult, artifactModel: String?, rows: [QuantReliabilityRow]
    ) -> [String] {
        let model = artifactModel ?? "(unspecified)"
        var lines = [
            "quant reliability overlay — measured on model: \(model) "
                + "(\(rows.count) measurement(s), joined by quant bits; display-only, ranking unchanged)"
        ]
        for e in pick.evaluations {
            let bitsLabel = e.quantBits.map { "\($0)-bit" } ?? "unquantized"
            let matches = rows.filter { $0.quantBits == e.quantBits }
            let detail: String
            switch matches.count {
            case 1:
                let row = matches[0]
                let s = row.score
                let repair = row.repairRateIsMeasured ? rate(s.repairRate) : "n/a"
                detail = "trigger=\(rate(s.triggerRate)) name=\(rate(s.nameAccuracy)) "
                    + "args=\(rate(s.argumentValidityRate)) repair=\(repair) n=\(s.caseCount)"
            case 0:
                detail = "reliability: n/a (no measurement)"
            default:
                detail = "reliability: n/a (ambiguous: \(matches.count) measurements for \(bitsLabel))"
            }
            lines.append("  \(e.repoID) [\(bitsLabel)] \(detail)")
        }
        return lines
    }

    private static func rate(_ v: Double) -> String { String(format: "%.2f", v) }
}
