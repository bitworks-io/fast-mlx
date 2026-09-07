import XCTest
@testable import HarnessCore

/// TDD for the MTP self-speculative-decode spike gate (roadmap #3 latency bet). The on-box MEASUREMENT
/// is M5-gated (the Qwen3.8-27B-MTP checkpoints are 27B-class), but the JUDGMENT — the promote/shelve
/// kernel — is pure Swift and must exist BEFORE the spike runs, so the eventual M5 run is
/// measurement-only with zero decision code written under time pressure.
///
/// Cycle 79 corrected the invariant this file used to pin. A real-weight A/B proved the in-checkpoint
/// MTP route is NOT, and architecturally cannot be, token-identical to the non-MTP scalar route: the
/// verify forward evaluates `blockSize + 1` positions per round while the scalar route evaluates one,
/// so the two never share forward geometry after prefill (289 vs 318 tokens, deterministic, at
/// temperature 0 — see `docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`). Every
/// emitted token is still the target model's own argmax; route-vs-scalar exactness is reported
/// (`MTPStreamExactness`) but is no longer a promote/shelve criterion. The new invariant pinned here:
/// a run where the drafter never proposed anything (`proposedDraftTokens == 0`) is scalar-vs-scalar by
/// construction and must NOT promote — that is a vacuous pass, not evidence the MTP route works.
final class MTPSpikeGateTests: XCTestCase {

    // MARK: stream-exactness comparator

    func testIdenticalStreamsAreExact() {
        let r = MTPStreamExactness.compare(candidate: [5, 9, 2, 7], baseline: [5, 9, 2, 7])
        XCTAssertTrue(r.exact)
        XCTAssertNil(r.firstDivergenceIndex)
        XCTAssertTrue(r.lengthMatched)
        XCTAssertEqual(r.comparedTokens, 4)
    }

    func testSingleDivergenceReportsItsIndexAndIsNotExact() {
        let r = MTPStreamExactness.compare(candidate: [5, 9, 4, 7], baseline: [5, 9, 2, 7])
        XCTAssertFalse(r.exact)
        XCTAssertEqual(r.firstDivergenceIndex, 2)
        XCTAssertTrue(r.lengthMatched)
    }

    func testLengthMismatchIsNotExactEvenWhenTheCommonPrefixMatches() {
        // Candidate is a clean prefix of baseline: no in-prefix divergence, but a length mismatch is
        // still a failure (greedy MTP must stop at the same place the baseline does).
        let r = MTPStreamExactness.compare(candidate: [5, 9, 2], baseline: [5, 9, 2, 7])
        XCTAssertFalse(r.exact)
        XCTAssertNil(r.firstDivergenceIndex)
        XCTAssertFalse(r.lengthMatched)
        XCTAssertEqual(r.comparedTokens, 3)
    }

    func testEmptyStreamsCompareZeroTokens() {
        let r = MTPStreamExactness.compare(candidate: [], baseline: [])
        XCTAssertEqual(r.comparedTokens, 0)
    }

    // MARK: promote / shelve verdict kernel

    private func acceptance(_ rate: Double) -> SpeculativeAcceptanceSummary {
        // 100 proposed, `rate*100` accepted across 40 rounds — a plausible spike aggregate.
        SpeculativeAcceptanceSummary(
            proposedDraftTokens: 100, acceptedDraftTokens: Int((rate * 100).rounded()), verifyRounds: 40)
    }

    func testExactWithSpeedupAboveOnePromotes() {
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 2, 3], baseline: [1, 2, 3]),
            acceptance: acceptance(0.7),
            modeledSpeedup: 1.8)
        XCTAssertEqual(gate.verdict, .promote)
    }

    func testCycle79DivergentShorterStreamPromotes() {
        // The cycle-79 real-weight A/B: MTP-on produced 289 tokens vs the 318-token scalar baseline,
        // each arm internally deterministic at temperature 0, with a real mid-stream divergence and a
        // measured 1.478x token-rate speedup (docs/task-inbox/
        // 2026-09-07-mtp-scalar-route-divergence-DECISION.md). BEFORE this change, `decide` vetoed on
        // `exactness.exact == false` and this population returned `.shelve` — a correct integration
        // shelved for a property (route-vs-scalar token identity) the architecture never offered.
        let candidate = Array(0..<289)
        var baseline = Array(0..<318)
        baseline[200] = -1  // a real divergence, mirroring the empirically-observed ~token-200 onset
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: candidate, baseline: baseline),
            acceptance: acceptance(0.7),
            modeledSpeedup: 1.478)
        XCTAssertFalse(gate.exactness.exact, "the population must actually be divergent, not accidentally exact")
        XCTAssertEqual(gate.exactness.firstDivergenceIndex, 200)
        XCTAssertFalse(gate.exactness.lengthMatched)
        XCTAssertEqual(gate.verdict, .promote)
    }

    func testVacuousPassthroughRunIsIndeterminateNotPromote() {
        // A run where the drafter never proposed anything is scalar-vs-scalar by construction: the two
        // streams trivially agree, which is not evidence the MTP route works. BEFORE this change,
        // `decide` had no proposed-draft-tokens check and this population returned `.promote` — proof
        // the new clause is not inert (it flips this exact case).
        let sticky = SpeculativeAcceptanceSummary(
            proposedDraftTokens: 0, acceptedDraftTokens: 0, verifyRounds: 0)
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 2, 3], baseline: [1, 2, 3]),
            acceptance: sticky,
            modeledSpeedup: 2.0)
        XCTAssertTrue(gate.exactness.exact, "sticky passthrough is scalar-vs-scalar, trivially exact")
        XCTAssertEqual(gate.verdict, .indeterminate)
    }

    func testExactButNoSpeedupShelves() {
        for speedup in [0.9, 1.0] {
            let gate = MTPSpikeGate(
                exactness: MTPStreamExactness.compare(candidate: [1, 2, 3], baseline: [1, 2, 3]),
                acceptance: acceptance(0.4),
                modeledSpeedup: speedup)
            XCTAssertEqual(gate.verdict, .shelve, "speedup \(speedup) is not > 1 → no economic case")
        }
    }

    func testZeroComparedTokensIsIndeterminate() {
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [], baseline: []),
            acceptance: acceptance(0.5),
            modeledSpeedup: 2.0)
        XCTAssertEqual(gate.verdict, .indeterminate)
    }

    func testEvidenceLineCarriesFrozenMachineKeys() {
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 2, 3], baseline: [1, 2, 3]),
            acceptance: acceptance(0.7),
            modeledSpeedup: 1.8)
        let line = gate.evidenceLine()
        XCTAssertTrue(line.contains("mtp_route_token_agreement=true"), line)
        XCTAssertTrue(line.contains("mtp_accept_rate="), line)
        XCTAssertTrue(line.contains("mtp_modeled_speedup="), line)
        XCTAssertTrue(line.contains("mtp_verdict=promote"), line)
    }

    func testShelveEvidenceLineNamesTheDivergence() {
        // Divergent AND no economic case (speedup <= 1): shelves for the speedup reason, but the
        // divergence must still be visible in the reported diagnostic even though it is no longer the
        // verdict input (route-vs-scalar exactness is reported, not gated — see file header).
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 9, 3], baseline: [1, 2, 3]),
            acceptance: acceptance(0.9),
            modeledSpeedup: 0.9)
        let line = gate.evidenceLine()
        XCTAssertTrue(line.contains("mtp_route_token_agreement=false"), line)
        XCTAssertTrue(line.contains("mtp_first_divergence=1"), line)
        XCTAssertTrue(line.contains("mtp_verdict=shelve"), line)
    }
}
