import XCTest
@testable import HarnessCore

/// Pure-logic TDD for prompt-lookup speculative decoding (PLD): the drafter, the greedy
/// accept-walk (distribution-preserving — the headline property), and the enable/disable gate.
/// All Foundation-only, zero MLX; this is the framework core the engine (Phase 2) will call.
final class SpecDecodeTests: XCTestCase {

    // MARK: - Task 1: PromptLookupDrafter

    /// A repeated phrase yields the correct continuation drawn from the earlier occurrence.
    func testDrafter_repeatedPhrase_yieldsCorrectContinuation() {
        // context: 1 2 3 4 5 ... 1 2 3 <propose next>
        let context = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3]
        let drafter = PromptLookupDrafter(ngram: 3)
        // suffix [1,2,3] matches earlier at index 0..2 (i=2), continuation is context[3...] = [4,5,9,9]
        let draft = drafter.propose(context: context, maxDraft: 4)
        XCTAssertEqual(draft, [4, 5, 9, 9])
    }

    /// When there are two earlier occurrences of the matching n-gram, the MOST RECENT wins.
    func testDrafter_mostRecentOccurrenceWins() {
        // context: [1,2,3] [5,5,5] [0] [1,2,3] [8,8,8] [1,2,3]
        //  indices:  0 1 2   3 4 5   6   7 8 9  10 11 12 13 14 15
        // suffix [1,2,3] (indices 13..15) occurs earlier at 0..2 (continuation [5,5,5])
        // and again at 7..9 (continuation [8,8,8]) — the most recent earlier match must win.
        let context = [1, 2, 3, 5, 5, 5, 0, 1, 2, 3, 8, 8, 8, 1, 2, 3]
        let drafter = PromptLookupDrafter(ngram: 3)
        let draft = drafter.propose(context: context, maxDraft: 3)
        XCTAssertEqual(draft, [8, 8, 8])
    }

    /// No earlier occurrence of the suffix n-gram yields an empty proposal.
    func testDrafter_noMatch_yieldsEmpty() {
        let context = [1, 2, 3, 4, 5, 6]
        let drafter = PromptLookupDrafter(ngram: 3)
        let draft = drafter.propose(context: context, maxDraft: 4)
        XCTAssertEqual(draft, [])
    }

    /// A match whose continuation runs off the end of the context is clamped to what's available.
    func testDrafter_matchNearEnd_clampsToAvailable() {
        // suffix [1,2,3] (indices 5..7) matches earlier at 0..2; the continuation after that
        // match is context[3...7] = [4,5,1,2,3] — only 5 tokens available, even though
        // maxDraft asks for 10.
        let context = [1, 2, 3, 4, 5, 1, 2, 3]
        let drafter = PromptLookupDrafter(ngram: 3)
        let draft = drafter.propose(context: context, maxDraft: 10)
        XCTAssertEqual(draft, [4, 5, 1, 2, 3])
    }

    /// `ngram` longer than the context yields no proposal (there's no suffix of that length to match).
    func testDrafter_ngramLongerThanContext_yieldsEmpty() {
        let context = [1, 2]
        let drafter = PromptLookupDrafter(ngram: 3)
        let draft = drafter.propose(context: context, maxDraft: 4)
        XCTAssertEqual(draft, [])
    }

    /// maxDraft of 0 yields no proposal even with a match.
    func testDrafter_zeroMaxDraft_yieldsEmpty() {
        let context = [1, 2, 3, 4, 5, 1, 2, 3]
        let drafter = PromptLookupDrafter(ngram: 3)
        let draft = drafter.propose(context: context, maxDraft: 0)
        XCTAssertEqual(draft, [])
    }

    // MARK: - Task 2: SpecAccept.walk — the exactness property

    /// Full-accept: all K drafts match the target's argmax at each position -> emits K+1 tokens
    /// (the bonus token is the target's own next-token pick after the last accepted draft).
    func testAcceptWalk_fullAccept_emitsKPlus1() {
        let draft = [10, 11, 12]
        let verifyArgmax = [10, 11, 12, 99]
        let result = SpecAccept.walk(draft: draft, verifyArgmax: verifyArgmax)
        XCTAssertEqual(result.accepted, 3)
        XCTAssertEqual(result.bonus, 99)
        assertExactness(draft: draft, verifyArgmax: verifyArgmax, result: result)
    }

    /// Zero-accept: the first draft mismatches the target's argmax -> emits just verifyArgmax[0],
    /// identical to a plain (non-speculative) greedy decode step.
    func testAcceptWalk_zeroAccept_emitsPlainGreedyStep() {
        let draft = [10, 11, 12]
        let verifyArgmax = [7, 11, 12, 99]
        let result = SpecAccept.walk(draft: draft, verifyArgmax: verifyArgmax)
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.bonus, 7)
        assertExactness(draft: draft, verifyArgmax: verifyArgmax, result: result)
    }

    /// Partial-accept: mismatch at position m -> emits m+1 tokens (m accepted drafts + bonus).
    func testAcceptWalk_partialAccept_emitsMPlus1() {
        let draft = [10, 11, 12, 13]
        let verifyArgmax = [10, 11, 5, 13, 99] // mismatch at position 2
        let result = SpecAccept.walk(draft: draft, verifyArgmax: verifyArgmax)
        XCTAssertEqual(result.accepted, 2)
        XCTAssertEqual(result.bonus, 5)
        assertExactness(draft: draft, verifyArgmax: verifyArgmax, result: result)
    }

    /// Empty draft: no drafted tokens at all -> emits just verifyArgmax[0].
    func testAcceptWalk_emptyDraft_emitsSingleToken() {
        let draft: [Int] = []
        let verifyArgmax = [42]
        let result = SpecAccept.walk(draft: draft, verifyArgmax: verifyArgmax)
        XCTAssertEqual(result.accepted, 0)
        XCTAssertEqual(result.bonus, 42)
        assertExactness(draft: draft, verifyArgmax: verifyArgmax, result: result)
    }

    /// Property-style sweep over several hand-built cases: the emitted sequence
    /// (draft[0..<accepted] + [bonus]) must equal verifyArgmax truncated at the first
    /// draft/argmax divergence, then that argmax value — i.e. exactly what plain greedy
    /// decoding would have produced from the same target distribution.
    func testAcceptWalk_exactness_matchesPlainGreedyPrefix_acrossCases() {
        let cases: [(draft: [Int], verifyArgmax: [Int])] = [
            ([1, 2, 3], [1, 2, 3, 4]),
            ([1, 2, 3], [1, 9, 3, 4]),
            ([1, 2, 3], [9, 2, 3, 4]),
            ([5], [5, 6]),
            ([5], [7, 6]),
            ([], [3]),
            ([1, 1, 1, 1], [1, 1, 1, 1, 1]),
            ([1, 1, 1, 1], [1, 1, 2, 1, 1]),
        ]
        for c in cases {
            let result = SpecAccept.walk(draft: c.draft, verifyArgmax: c.verifyArgmax)
            assertExactness(draft: c.draft, verifyArgmax: c.verifyArgmax, result: result)
        }
    }

    /// Helper: the load-bearing exactness assertion — compares the FULL emitted token sequence,
    /// not just accepted/bonus counts, against the plain-greedy prefix derived independently
    /// from `verifyArgmax` by walking to the first draft/argmax divergence.
    private func assertExactness(draft: [Int], verifyArgmax: [Int], result: (accepted: Int, bonus: Int), file: StaticString = #filePath, line: UInt = #line) {
        var plainGreedyIndex = 0
        while plainGreedyIndex < draft.count && draft[plainGreedyIndex] == verifyArgmax[plainGreedyIndex] {
            plainGreedyIndex += 1
        }
        let expectedEmitted = Array(verifyArgmax[0...plainGreedyIndex])
        let actualEmitted = Array(draft[0..<result.accepted]) + [result.bonus]
        XCTAssertEqual(actualEmitted, expectedEmitted, "emitted sequence must equal the plain-greedy prefix", file: file, line: line)
        XCTAssertEqual(result.accepted, plainGreedyIndex, file: file, line: line)
    }

    // MARK: - Task 3: PLDGate

    /// A sustained low-acceptance sequence (well below minAcceptPerStep) disables the gate.
    func testGate_sustainedLowAcceptance_disables() {
        var gate = PLDGate(window: 8, minAcceptPerStep: 0.25, cooldown: 4)
        XCTAssertTrue(gate.isEnabled)
        for _ in 0..<8 { gate.record(accepted: 0) }
        XCTAssertFalse(gate.isEnabled)
    }

    /// After a disable, the gate re-enables (probes) once `cooldown` normal steps have elapsed.
    func testGate_reenablesAfterCooldown() {
        var gate = PLDGate(window: 4, minAcceptPerStep: 0.25, cooldown: 3)
        for _ in 0..<4 { gate.record(accepted: 0) }
        XCTAssertFalse(gate.isEnabled)
        // Fewer than cooldown normal steps: still disabled.
        gate.record(accepted: 0)
        gate.record(accepted: 0)
        XCTAssertFalse(gate.isEnabled)
        // The cooldown-th normal step re-enables (probes) PLD again.
        gate.record(accepted: 0)
        XCTAssertTrue(gate.isEnabled)
    }

    /// A sustained high-acceptance sequence keeps the gate enabled throughout.
    func testGate_sustainedHighAcceptance_staysEnabled() {
        var gate = PLDGate(window: 8, minAcceptPerStep: 0.25, cooldown: 4)
        for _ in 0..<32 {
            gate.record(accepted: 3)
            XCTAssertTrue(gate.isEnabled)
        }
    }

    /// The windowed mean is computed correctly at the window boundary: once more than `window`
    /// samples have been recorded, only the most recent `window` contribute to the mean.
    func testGate_windowMeanComputedAtBoundary() {
        var gate = PLDGate(window: 4, minAcceptPerStep: 0.25, cooldown: 4)
        // First 4 samples: mean = (4+4+4+4)/4 = 4.0, well above threshold -> enabled.
        for _ in 0..<4 { gate.record(accepted: 4) }
        XCTAssertTrue(gate.isEnabled)
        // Next 4 samples all 0: once the window has fully rolled over (4 more samples),
        // mean = (0+0+0+0)/4 = 0.0 -> disabled. The stale high values must have rolled off.
        for _ in 0..<4 { gate.record(accepted: 0) }
        XCTAssertFalse(gate.isEnabled)
    }
}
