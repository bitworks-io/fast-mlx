import XCTest
@testable import HarnessCore

/// TDD for the cold-snapshot block-alignment planner — the pure, slot-ownership-agnostic arithmetic
/// half of the SSD-tiered-KV feature (`docs/task-inbox/2026-07-12-exact-prefix-session-cache.md`,
/// roadmap #3a). The capture's acceptance signal: "full-block-only SSD snapshots recompute the
/// uncached tail exactly: cover blockSize-1, blockSize+1, and several-block-plus-tail prompts, put
/// the final system/user instruction wholly in that tail."
///
/// Scope guard: this is the planner ONLY. The on-disk snapshot manifest, digest scheme, and
/// commit/restore orchestration are deliberately NOT built here — the capture's Next Step defers
/// them behind continuous-batching slot ownership (`implementation_ready: false`), and committing a
/// persisted format before that design is throwaway + a security-relevant surface. Nothing here
/// touches MLX arrays, serving, or any announce line.
final class ExactPrefixColdSnapshotTests: XCTestCase {

    // MARK: - the three acceptance-named edges

    /// blockSize-1 tokens → no full block persists; the whole prompt is the recomputed tail.
    func testBelowOneBlock_allTail_noPersistedBlocks() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 255, blockSize: 256)
        XCTAssertEqual(plan.persistedBlockCount, 0)
        XCTAssertEqual(plan.persistedTokenCount, 0)
        XCTAssertEqual(plan.recomputeTailTokenCount, 255)
    }

    /// An exact block multiple → every token is in a full persisted block; the tail is empty.
    func testExactBlockMultiple_tailIsZero() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 768, blockSize: 256)
        XCTAssertEqual(plan.persistedBlockCount, 3)
        XCTAssertEqual(plan.persistedTokenCount, 768)
        XCTAssertEqual(plan.recomputeTailTokenCount, 0)
    }

    /// blockSize+1 → exactly one full block persists, one token recomputes.
    func testOneOverABlock_oneBlockOneTail() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 257, blockSize: 256)
        XCTAssertEqual(plan.persistedBlockCount, 1)
        XCTAssertEqual(plan.persistedTokenCount, 256)
        XCTAssertEqual(plan.recomputeTailTokenCount, 1)
    }

    /// several-block-plus-tail → floor(1000/256)=3 blocks (768), 232-token tail.
    func testSeveralBlocksPlusTail() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 1000, blockSize: 256)
        XCTAssertEqual(plan.persistedBlockCount, 3)
        XCTAssertEqual(plan.persistedTokenCount, 768)
        XCTAssertEqual(plan.recomputeTailTokenCount, 232)
    }

    // MARK: - "final instruction wholly in the tail" (the mistaken-partial-block failure case)

    /// The capture's correctness property + its named failure ("mistaking an intentionally
    /// recomputed partial block for dropped context"): a final instruction whose length fits within
    /// the tail is recomputed exactly this turn — none of it was frozen into a persisted block.
    func testFinalInstructionFitsInTail_isFullyRecomputed() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 1000, blockSize: 256)
        // tail is 232 tokens; a 200-token final instruction lies wholly inside it.
        XCTAssertTrue(plan.suffixIsFullyRecomputed(suffixLength: 200))
        XCTAssertTrue(plan.suffixIsFullyRecomputed(suffixLength: plan.recomputeTailTokenCount))
    }

    /// An instruction one token longer than the tail would bleed into the last persisted block, so
    /// it is NOT wholly recomputed — the planner must report that honestly (else a restore could
    /// serve a stale trailing instruction).
    func testInstructionLongerThanTail_isNotFullyRecomputed() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 1000, blockSize: 256)
        XCTAssertFalse(plan.suffixIsFullyRecomputed(suffixLength: plan.recomputeTailTokenCount + 1))
    }

    // MARK: - degenerate + invalid inputs (fail closed)

    func testZeroPrompt_emptyPlan() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 0, blockSize: 256)
        XCTAssertEqual(plan.persistedBlockCount, 0)
        XCTAssertEqual(plan.persistedTokenCount, 0)
        XCTAssertEqual(plan.recomputeTailTokenCount, 0)
    }

    func testNonPositiveBlockSize_throws() {
        for bad in [0, -1, -256] {
            XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: 100, blockSize: bad)) {
                XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidBlockSize(bad))
            }
        }
    }

    func testNegativePromptTokenCount_throws() {
        XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: -1, blockSize: 256)) {
            XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidPromptTokenCount(-1))
        }
    }

    // MARK: - invariants across a grid (the arithmetic can never violate these)

    func testPlanInvariants_holdAcrossGrid() throws {
        let blockSizes = [1, 16, 128, 256, 4096]
        let prompts = [0, 1, 15, 16, 17, 255, 256, 257, 1000, 32768, 262_144]
        for blockSize in blockSizes {
            for prompt in prompts {
                let p = try ExactPrefixColdSnapshotPlanner.plan(promptTokenCount: prompt, blockSize: blockSize)
                XCTAssertEqual(p.persistedTokenCount % blockSize, 0, "persisted region is always whole blocks (bs=\(blockSize), n=\(prompt))")
                XCTAssertEqual(p.persistedBlockCount * blockSize, p.persistedTokenCount, "block count and token count agree")
                XCTAssertGreaterThanOrEqual(p.recomputeTailTokenCount, 0, "tail never negative")
                XCTAssertLessThan(p.recomputeTailTokenCount, blockSize, "tail is a strict partial block")
                XCTAssertEqual(p.persistedTokenCount + p.recomputeTailTokenCount, prompt, "persisted + tail reconstructs the prompt exactly (no dropped/duplicated context)")
            }
        }
    }

    // MARK: - restore reconciliation: reuse a stored whole-block snapshot against a NEW prompt

    /// Helper: a token run `0, 1, ..., n-1` (distinct tokens so equality tracks position).
    private func seq(_ n: Int) -> [Int] { Array(0..<n) }

    /// Identical prompt, exactly the stored region → every stored block is reusable, no recompute
    /// of the persisted region (only whatever tail the plan already excluded).
    func testRestore_identicalPrompt_reusesAllStoredBlocks() throws {
        let stored = seq(512)   // 2 whole blocks of 256
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: stored, blockSize: 256)
        XCTAssertEqual(plan.restoredBlockCount, 2)
        XCTAssertEqual(plan.restoredTokenCount, 512)
        XCTAssertEqual(plan.recomputeTokenCount, 0)
        XCTAssertEqual(plan.commonPrefixTokenCount, 512)
    }

    /// Prompt extends the stored prefix (same leading 512, then new tokens) → reuse both stored
    /// blocks, recompute only the appended remainder.
    func testRestore_promptExtendsStored_reusesStoredCapsAtStoredBlocks() throws {
        let stored = seq(512)
        var prompt = seq(512); prompt += [900, 901, 902]   // 515 tokens, first 512 identical
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.restoredBlockCount, 2)
        XCTAssertEqual(plan.restoredTokenCount, 512)
        XCTAssertEqual(plan.recomputeTokenCount, 3)
        XCTAssertEqual(plan.commonPrefixTokenCount, 512)
    }

    /// Divergence mid-block-k: common prefix 300 tokens (diverges at index 300) → only block 0
    /// (0..<256) is wholly common; block 1 contains the divergent token and must be recomputed.
    func testRestore_divergenceMidBlock_reusesOnlyWhollyCommonBlocks() throws {
        let stored = seq(512)
        var prompt = seq(300); prompt.append(9999); prompt += Array(301..<520)  // diverges at index 300
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.commonPrefixTokenCount, 300)
        XCTAssertEqual(plan.restoredBlockCount, 1)
        XCTAssertEqual(plan.restoredTokenCount, 256)
        XCTAssertEqual(plan.recomputeTokenCount, prompt.count - 256)
    }

    /// Divergence exactly at a block boundary (index 256): block 0 is wholly common and reused;
    /// the token at 256 diverges, so block 1 is recomputed.
    func testRestore_divergenceAtBlockBoundary_reusesPrecedingBlocksOnly() throws {
        let stored = seq(512)
        var prompt = seq(256); prompt.append(9999); prompt += Array(257..<512)  // diverges at index 256
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.commonPrefixTokenCount, 256)
        XCTAssertEqual(plan.restoredBlockCount, 1)
        XCTAssertEqual(plan.restoredTokenCount, 256)
    }

    /// Prompt shorter than the stored region (identical prefix) → reuse only the whole blocks the
    /// shorter prompt fully covers; the rest is recompute.
    func testRestore_promptShorterThanStored_reusesCoveredBlocksOnly() throws {
        let stored = seq(768)   // 3 whole blocks
        let prompt = seq(300)   // identical first 300
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.commonPrefixTokenCount, 300)
        XCTAssertEqual(plan.restoredBlockCount, 1)
        XCTAssertEqual(plan.restoredTokenCount, 256)
        XCTAssertEqual(plan.recomputeTokenCount, 44)
    }

    /// Total divergence at token 0 → nothing reusable, the whole prompt recomputes.
    func testRestore_noCommonPrefix_recomputesEverything() throws {
        let stored = seq(512)
        let prompt = Array(5000..<5300)   // shares no leading token
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.commonPrefixTokenCount, 0)
        XCTAssertEqual(plan.restoredBlockCount, 0)
        XCTAssertEqual(plan.restoredTokenCount, 0)
        XCTAssertEqual(plan.recomputeTokenCount, 300)
    }

    /// Empty stored snapshot → nothing to reuse (cold start).
    func testRestore_emptyStored_recomputesEverything() throws {
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: [], promptTokens: seq(300), blockSize: 256)
        XCTAssertEqual(plan.restoredBlockCount, 0)
        XCTAssertEqual(plan.recomputeTokenCount, 300)
    }

    // MARK: - whole-snapshot-only granularity (recurrent/linear state can't rewind)

    /// A hybrid/recurrent snapshot diverging mid-way reuses NOTHING under `.wholeSnapshotOnly`: the
    /// recurrent state exists only at the stored token count and can't be rewound to an earlier
    /// block boundary. The paired `.blockAligned` assertion proves the parameter is load-bearing
    /// (dense attention DOES reuse the preceding whole blocks here).
    func testRestore_wholeSnapshotOnly_midSnapshotDivergence_reusesNothing() throws {
        let stored = seq(64)                       // 4 blocks @ bs 16
        var prompt = seq(40); prompt += Array(1000..<1024)  // diverges at token 40, len 64
        let whole = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16, granularity: .wholeSnapshotOnly)
        XCTAssertEqual(whole.commonPrefixTokenCount, 40)
        XCTAssertEqual(whole.restoredBlockCount, 0, "no mid-snapshot rewind for recurrent state")
        XCTAssertEqual(whole.restoredTokenCount, 0)
        XCTAssertEqual(whole.recomputeTokenCount, prompt.count)
        // Same inputs, dense granularity → the two whole blocks before the divergence ARE reusable.
        let dense = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16, granularity: .blockAligned)
        XCTAssertEqual(dense.restoredBlockCount, 2)
        XCTAssertEqual(dense.restoredTokenCount, 32)
    }

    /// A prompt that is a proper prefix of the stored snapshot (no divergence, just shorter) still
    /// reuses nothing under `.wholeSnapshotOnly` — the state can't be replayed to a mid-point.
    func testRestore_wholeSnapshotOnly_promptShorterButFullyCommon_reusesNothing() throws {
        let stored = seq(64)
        let prompt = seq(48)   // identical first 48, but stops short of the 64-token state
        let whole = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16, granularity: .wholeSnapshotOnly)
        XCTAssertEqual(whole.commonPrefixTokenCount, 48)
        XCTAssertEqual(whole.restoredBlockCount, 0)
        XCTAssertEqual(whole.recomputeTokenCount, 48)
    }

    /// When the prompt covers the ENTIRE stored snapshot (matches then extends), the whole snapshot
    /// is reusable — that is the one case recurrent state supports.
    func testRestore_wholeSnapshotOnly_promptExtendsStored_reusesWholeSnapshot() throws {
        let stored = seq(64)
        let prompt = seq(80)   // stored 64 exactly, plus 16 new tokens
        let whole = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 16, granularity: .wholeSnapshotOnly)
        XCTAssertEqual(whole.commonPrefixTokenCount, 64)
        XCTAssertEqual(whole.restoredBlockCount, 4)
        XCTAssertEqual(whole.restoredTokenCount, 64)
        XCTAssertEqual(whole.recomputeTokenCount, 16)
    }

    /// An exact match reuses the whole snapshot under both granularities (the boundary case where
    /// common prefix == stored length).
    func testRestore_wholeSnapshotOnly_exactMatch_reusesWholeSnapshot() throws {
        let stored = seq(64)
        let whole = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: seq(64), blockSize: 16, granularity: .wholeSnapshotOnly)
        XCTAssertEqual(whole.restoredBlockCount, 4)
        XCTAssertEqual(whole.recomputeTokenCount, 0)
    }

    // fail-closed inputs

    func testRestore_nonPositiveBlockSize_throws() {
        for bad in [0, -1, -256] {
            XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.planRestore(
                storedPrefixTokens: seq(256), promptTokens: seq(256), blockSize: bad)) {
                XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidBlockSize(bad))
            }
        }
    }

    /// A stored snapshot that is not a whole-block multiple violates the full-block-only invariant —
    /// treating its trailing partial block as reusable would restore a block that was never persisted.
    func testRestore_nonWholeBlockStored_throws() {
        XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: seq(300), promptTokens: seq(300), blockSize: 256)) {
            XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .snapshotNotWholeBlock(storedTokenCount: 300, blockSize: 256))
        }
    }

    // MARK: - restore-side suffixIsFullyRecomputed (the ≥1-recompute question, surfaced not hidden)

    /// A restore whose recompute region covers the final instruction reports it fully recomputed —
    /// the restore-side twin of the block-plan property.
    func testRestorePlan_suffixWithinRecompute_isFullyRecomputed() throws {
        let stored = seq(512)
        var prompt = seq(512); prompt += Array(900..<950)  // 50-token appended tail recomputes
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(plan.recomputeTokenCount, 50)
        XCTAssertTrue(plan.suffixIsFullyRecomputed(suffixLength: 50))
        XCTAssertTrue(plan.suffixIsFullyRecomputed(suffixLength: 10))
        XCTAssertFalse(plan.suffixIsFullyRecomputed(suffixLength: 51), "one token bleeds into a restored block")
        XCTAssertFalse(plan.suffixIsFullyRecomputed(suffixLength: -1), "negative length is not a suffix")
    }

    /// On a full cache hit (recompute == 0) any positive suffix is NOT fully recomputed — the honest
    /// surface of the deferred ≥1-recompute orchestration question, not a hidden zero.
    func testRestorePlan_fullHit_positiveSuffixIsNotFullyRecomputed() throws {
        let stored = seq(512)
        let plan = try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: stored, promptTokens: stored, blockSize: 256)
        XCTAssertEqual(plan.recomputeTokenCount, 0)
        XCTAssertTrue(plan.suffixIsFullyRecomputed(suffixLength: 0))
        XCTAssertFalse(plan.suffixIsFullyRecomputed(suffixLength: 1))
    }

    // MARK: - planBestRestore: pick which stored snapshot to reuse among several for one key

    /// Deepest-common-prefix wins: candidate A shares 512 tokens with the prompt (2 blocks), B shares
    /// only 256 (1 block, then diverges). Under dense granularity A restores more → A wins.
    func testBestRestore_blockAligned_picksDeepestCommonPrefixCandidate() throws {
        let a = seq(512)                               // fully common with the prompt's first 512
        var b = seq(256); b += Array(700..<956)        // diverges at token 256
        var prompt = seq(512); prompt += [900, 901]
        let sel = try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [b, a], promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(sel?.candidateIndex, 1, "index of candidate a")
        XCTAssertEqual(sel?.plan.restoredTokenCount, 512)
        XCTAssertEqual(sel?.plan.restoredBlockCount, 2)
    }

    /// Under `.wholeSnapshotOnly` the winner can flip: a LONGER candidate that diverges mid-snapshot
    /// restores nothing, while a SHORTER fully-covered candidate restores in full and wins.
    func testBestRestore_wholeSnapshotOnly_shorterFullyCoveredCandidateWins() throws {
        let shortFull = seq(32)                          // 2 blocks @ bs 16 — prompt covers it entirely
        let longWhole = seq(16) + Array(9000..<9048)     // 64 tokens (4 blocks) — diverges at token 16
        var prompt = seq(32); prompt += Array(80..<96)   // covers shortFull fully; diverges from longWhole at 16
        let sel = try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [longWhole, shortFull], promptTokens: prompt, blockSize: 16,
            granularity: .wholeSnapshotOnly)
        XCTAssertEqual(sel?.candidateIndex, 1, "the shorter, fully-covered snapshot")
        XCTAssertEqual(sel?.plan.restoredTokenCount, 32, "restores in full; longWhole restores nothing")
    }

    /// Two identical winning candidates → the lowest index is chosen (deterministic tie-break).
    func testBestRestore_tieBreak_isLowestIndex() throws {
        let a = seq(512)
        let prompt = seq(512)
        let sel = try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [a, a], promptTokens: prompt, blockSize: 256)
        XCTAssertEqual(sel?.candidateIndex, 0)
        XCTAssertEqual(sel?.plan.restoredTokenCount, 512)
    }

    /// No candidate restores anything (all diverge at token 0) and the empty list → `nil`, the honest
    /// "nothing to reuse" outcome rather than a zero-token plan.
    func testBestRestore_noUsefulCandidate_returnsNil() throws {
        let a = Array(5000..<5512)
        let b = Array(6000..<6512)
        let prompt = seq(300)
        let none = try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [a, b], promptTokens: prompt, blockSize: 256)
        XCTAssertNil(none)
        let empty = try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [], promptTokens: prompt, blockSize: 256)
        XCTAssertNil(empty)
    }

    /// A corrupt (non-whole-block) candidate fails the WHOLE selection, even when another candidate
    /// would otherwise win — its presence signals corruption, not a losing candidate to skip.
    func testBestRestore_corruptCandidate_throwsWholeSelection() {
        let good = seq(512)
        let corrupt = seq(300)   // not a 256-multiple
        XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: [good, corrupt], promptTokens: seq(512), blockSize: 256)) {
            XCTAssertEqual($0 as? ExactPrefixColdSnapshotError,
                .snapshotNotWholeBlock(storedTokenCount: 300, blockSize: 256))
        }
    }

    /// A non-positive block size fails closed before any candidate is examined.
    func testBestRestore_nonPositiveBlockSize_throws() {
        for bad in [0, -1] {
            XCTAssertThrowsError(try ExactPrefixColdSnapshotPlanner.planBestRestore(
                candidateStoredPrefixes: [seq(256)], promptTokens: seq(256), blockSize: bad)) {
                XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidBlockSize(bad))
            }
        }
    }

    /// The selection always equals the max over per-candidate `planRestore` calls — the aggregator
    /// adds no arithmetic of its own.
    func testBestRestore_matchesMaxOverPerCandidatePlanRestore() throws {
        let candidates = [seq(256), seq(512), seq(768), Array(9000..<9256)]
        var prompt = seq(600); prompt += Array(20000..<20050)
        for granularity in [ColdSnapshotReuseGranularity.blockAligned, .wholeSnapshotOnly] {
            let sel = try ExactPrefixColdSnapshotPlanner.planBestRestore(
                candidateStoredPrefixes: candidates, promptTokens: prompt, blockSize: 256, granularity: granularity)
            // Compute the expected winner independently.
            var expectedIndex = -1
            var expectedBest = 0
            for (i, c) in candidates.enumerated() {
                let p = try ExactPrefixColdSnapshotPlanner.planRestore(
                    storedPrefixTokens: c, promptTokens: prompt, blockSize: 256, granularity: granularity)
                if p.restoredTokenCount > expectedBest {
                    expectedBest = p.restoredTokenCount
                    expectedIndex = i
                }
            }
            if expectedIndex == -1 {
                XCTAssertNil(sel, "no candidate restores anything under \(granularity)")
            } else {
                XCTAssertEqual(sel?.candidateIndex, expectedIndex, "granularity \(granularity)")
                XCTAssertEqual(sel?.plan.restoredTokenCount, expectedBest)
            }
        }
    }

    /// Invariants across a grid of stored/prompt/blockSize combinations: restored region is always
    /// whole blocks, never exceeds the common prefix, and restore + recompute reconstructs the prompt.
    func testRestore_invariants_holdAcrossGrid() throws {
        let blockSizes = [1, 16, 256]
        for blockSize in blockSizes {
            let storedBlocks = [0, 1, 3]
            let divergeAts = [0, 1, blockSize, blockSize + 1, 2 * blockSize, 5 * blockSize]
            let promptExtras = [0, 1, blockSize - 1, blockSize]
            for sb in storedBlocks {
                let storedLen = sb * blockSize
                let stored = seq(storedLen)
                for d in divergeAts {
                    for extra in promptExtras {
                        // prompt shares min(d, storedLen) leading tokens then diverges/extends.
                        let common = min(d, storedLen)
                        var prompt = seq(common)
                        // append tokens guaranteed not to match the stored continuation.
                        let extraLen = max(0, d - common) + extra + 1
                        prompt += Array(100_000..<(100_000 + extraLen))
                        for granularity in [ColdSnapshotReuseGranularity.blockAligned, .wholeSnapshotOnly] {
                            let p = try ExactPrefixColdSnapshotPlanner.planRestore(
                                storedPrefixTokens: stored, promptTokens: prompt, blockSize: blockSize, granularity: granularity)
                            XCTAssertEqual(p.restoredTokenCount % blockSize, 0, "restored region is whole blocks (bs=\(blockSize),sb=\(sb),d=\(d),g=\(granularity))")
                            XCTAssertEqual(p.restoredBlockCount * blockSize, p.restoredTokenCount, "block count agrees")
                            XCTAssertLessThanOrEqual(p.restoredTokenCount, p.commonPrefixTokenCount, "never restore past the common prefix")
                            XCTAssertLessThanOrEqual(p.restoredBlockCount, sb, "never restore more blocks than were stored")
                            XCTAssertGreaterThanOrEqual(p.recomputeTokenCount, 0, "recompute never negative")
                            XCTAssertEqual(p.restoredTokenCount + p.recomputeTokenCount, prompt.count, "restore + recompute reconstructs the prompt exactly")
                            if granularity == .wholeSnapshotOnly {
                                // Whole-snapshot-only reuses either nothing or the entire stored snapshot — never a middle.
                                XCTAssertTrue(p.restoredBlockCount == 0 || p.restoredBlockCount == sb,
                                    "wholeSnapshotOnly is all-or-nothing (bs=\(blockSize),sb=\(sb),d=\(d)) got \(p.restoredBlockCount)")
                            }
                        }
                    }
                }
            }
        }
    }
}
