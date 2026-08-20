import XCTest
@testable import HarnessCore

/// TDD for the MTP self-speculative-decode spike gate (roadmap #3 latency bet). The on-box MEASUREMENT
/// is M5-gated (the Qwen3.8-27B-MTP checkpoints are 27B-class), but the JUDGMENT — the exactness veto
/// and the promote/shelve kernel — is pure Swift and must exist BEFORE the spike runs, so the eventual
/// M5 run is measurement-only with zero decision code written under time pressure.
///
/// The invariant pinned here is the one that shelved EAGLE-3 (failed greedy exactness at 4/8-bit):
/// ANY greedy divergence from the non-MTP baseline ⇒ SHELVE, regardless of how large the speedup is.
/// Speculative decoding is exact BY CONSTRUCTION (SpecAccept.walk), so a divergence is not a tuning
/// knob — it is a broken integration, and no throughput number can buy it back.
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

    func testAnyDivergenceShelvesRegardlessOfSpeedup() {
        // The EAGLE-3 invariant: a huge speedup cannot buy back a broken (non-exact) stream.
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 9, 3], baseline: [1, 2, 3]),
            acceptance: acceptance(0.95),
            modeledSpeedup: 3.0)
        XCTAssertEqual(gate.verdict, .shelve)
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
        XCTAssertTrue(line.contains("mtp_exact=true"), line)
        XCTAssertTrue(line.contains("mtp_accept_rate="), line)
        XCTAssertTrue(line.contains("mtp_modeled_speedup="), line)
        XCTAssertTrue(line.contains("mtp_verdict=promote"), line)
    }

    func testShelveEvidenceLineNamesTheDivergence() {
        let gate = MTPSpikeGate(
            exactness: MTPStreamExactness.compare(candidate: [1, 9, 3], baseline: [1, 2, 3]),
            acceptance: acceptance(0.9),
            modeledSpeedup: 3.0)
        let line = gate.evidenceLine()
        XCTAssertTrue(line.contains("mtp_exact=false"), line)
        XCTAssertTrue(line.contains("mtp_first_divergence=1"), line)
        XCTAssertTrue(line.contains("mtp_verdict=shelve"), line)
    }
}
