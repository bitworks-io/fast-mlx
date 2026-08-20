import XCTest
@testable import HarnessCore

/// TDD for the display-only reliability overlay on a quant auto-pick announce (roadmap #4). The
/// operator has an off-box `quant-reliability/v1` artifact and runs `--quant-pick-only` for the same
/// model's candidates; this composer annotates each candidate the picker considered with its measured
/// tool-call reliability. Two design invariants under test:
///
/// 1. **Join by quant bits, not repo id.** In the serve path a pick candidate's `repoID` is its
///    absolute directory PATH (`QuantCandidateResolver` uses `dir.path`), while the artifact's repo id
///    is the HF repo id — different namespaces that never match. Quant bits is the only reliable key.
/// 2. **Display-only.** The overlay is a SEPARATE block; it never contains the pick's winner line and
///    never reorders candidates, so ranking/winner selection is untouched.
final class QuantPickReliabilityAnnounceTests: XCTestCase {

    /// A pick over a 4-bit and an 8-bit candidate (repoIDs are absolute paths, as the serve path
    /// produces them — deliberately unrelated to the artifact's HF repo ids).
    private func samplePick() -> QuantPickResult {
        QuantPickResult(
            shouldProceed: true,
            winnerRepoID: "/models/q8",
            winnerDecision: nil,
            evaluations: [
                QuantCandidateEvaluation(repoID: "/models/q4", quantBits: 4, decision: nil, exclusionReason: "x"),
                QuantCandidateEvaluation(repoID: "/models/q8", quantBits: 8, decision: nil, exclusionReason: "x"),
            ])
    }

    private func score(_ args: Double, repair: Double? = nil) -> ToolCallReliabilityScore {
        ToolCallReliabilityScore(
            caseCount: 4, triggerRate: 1.0, nameAccuracy: 1.0, argumentValidityRate: args,
            repairRate: repair ?? 0.0)
    }

    /// Each candidate is annotated with the reliability row whose bits match — 8-bit path gets the
    /// 8-bit measurement (args=1.00), 4-bit path gets the 4-bit measurement (args=0.50) — even though
    /// the repo ids differ. The header names the artifact's model.
    func testJoinsByQuantBitsAcrossDifferingRepoIDs() {
        let rows = [
            QuantReliabilityRow(repoID: "hf/M-4bit", quantBits: 4, score: score(0.5), repairRateIsMeasured: false),
            QuantReliabilityRow(repoID: "hf/M-8bit", quantBits: 8, score: score(1.0, repair: 0.0)),
        ]
        let lines = QuantPickReliabilityAnnounce.compose(pick: samplePick(), artifactModel: "Qwen3-8B", rows: rows)
        XCTAssertTrue(lines.first?.contains("Qwen3-8B") == true, "header names model: \(lines)")

        let q4 = try! XCTUnwrap(lines.first { $0.contains("/models/q4") })
        let q8 = try! XCTUnwrap(lines.first { $0.contains("/models/q8") })
        XCTAssertTrue(q4.contains("args=0.50"), "4-bit path joins the 4-bit measurement: \(q4)")
        XCTAssertTrue(q8.contains("args=1.00"), "8-bit path joins the 8-bit measurement: \(q8)")
        // Honesty preserved through the join: unmeasured repair → n/a, measured 0.0 → 0.00.
        XCTAssertTrue(q4.contains("repair=n/a"), "4-bit unmeasured repair renders n/a: \(q4)")
        XCTAssertTrue(q8.contains("repair=0.00"), "8-bit measured repair renders value: \(q8)")
    }

    /// A candidate whose bits have no measurement is honestly marked n/a, not dropped or fabricated.
    func testUnmatchedCandidateRendersNoMeasurement() {
        let rows = [QuantReliabilityRow(repoID: "hf/M-8bit", quantBits: 8, score: score(1.0))]
        let lines = QuantPickReliabilityAnnounce.compose(pick: samplePick(), artifactModel: "M", rows: rows)
        let q4 = try! XCTUnwrap(lines.first { $0.contains("/models/q4") })
        XCTAssertTrue(q4.contains("n/a (no measurement)"), "unmeasured bits → n/a: \(q4)")
    }

    /// Two measurements sharing the same bits are ambiguous — the composer refuses to guess and marks
    /// n/a with the count, rather than silently picking one.
    func testAmbiguousBitsRendersNoMeasurement() {
        let rows = [
            QuantReliabilityRow(repoID: "hf/M-4bit-a", quantBits: 4, score: score(0.5)),
            QuantReliabilityRow(repoID: "hf/M-4bit-b", quantBits: 4, score: score(0.9)),
        ]
        let lines = QuantPickReliabilityAnnounce.compose(pick: samplePick(), artifactModel: "M", rows: rows)
        let q4 = try! XCTUnwrap(lines.first { $0.contains("/models/q4") })
        XCTAssertTrue(q4.contains("ambiguous"), "two 4-bit measurements → ambiguous n/a: \(q4)")
        XCTAssertFalse(q4.contains("args=0.50"), "must not silently pick one of the ambiguous rows: \(q4)")
    }

    /// Display-only: the overlay block never contains the pick's WINNER line, and the pick's own
    /// machine-readable winner surface is unchanged by composing an overlay.
    func testOverlayIsDisplayOnlyAndOmitsWinnerLine() {
        let pick = samplePick()
        let rows = [QuantReliabilityRow(repoID: "hf/M-8bit", quantBits: 8, score: score(1.0))]
        let overlay = QuantPickReliabilityAnnounce.compose(pick: pick, artifactModel: "M", rows: rows)
        XCTAssertFalse(overlay.contains { $0.contains("WINNER") }, "overlay must not restate the winner: \(overlay)")
        // The pick's machine-readable line is a pure function of the pick; composing the overlay
        // cannot change it (value semantics), asserted here as a regression guard.
        XCTAssertEqual(pick.machineReadableWinnerLine(), samplePick().machineReadableWinnerLine())
    }
}
