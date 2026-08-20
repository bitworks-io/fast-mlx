import XCTest
@testable import HarnessCore

/// TDD for the cold-tier SSD store admission/eviction PLANNER — the pure decision half of "does a
/// new cold snapshot fit the SSD budget, and if not, which stored snapshots must be evicted to make
/// room" (`docs/task-inbox/2026-07-12-exact-prefix-session-cache.md`, roadmap #3a). It pairs with
/// `modeledSnapshotKVBytes` (which produces the candidate's byte size) to complete the SSD budget
/// arithmetic, and deliberately mirrors the hot-path `ExactPrefixCache` reservation semantics:
///   - a candidate larger than the WHOLE budget is rejected WITHOUT evicting anything (never clear
///     the store for a snapshot that still won't fit — `ExactPrefixCommitSkipReason
///     .snapshotExceedsBudget`);
///   - eviction is least-recently-used first, tie-broken by lowest id (deterministic), stopping as
///     soon as the candidate fits (`ExactPrefixCache.leastRecentlyUsedEntryID`).
///
/// Scope guard: this owns NO state, writes NO files, and commits NO on-disk format — it returns
/// which entry ids to evict, exactly as `ExactPrefixReservationDecision` does. Persistence/
/// orchestration stay deferred behind continuous-batching slot ownership.
final class ExactPrefixColdSnapshotStorePlannerTests: XCTestCase {

    private func entry(_ id: UInt64, bytes: Int, tick: UInt64) -> ColdSnapshotStoredEntry {
        ColdSnapshotStoredEntry(id: id, bytes: bytes, lastUsedTick: tick)
    }

    /// Candidate fits in the free space → admit with no evictions.
    func testAdmitWithinBudget_noEviction() throws {
        let existing = [entry(1, bytes: 100, tick: 1), entry(2, bytes: 100, tick: 2)]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 50, existing: existing, budgetBytes: 300)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: []))
    }

    /// Store is full; the candidate needs room. Evict least-recently-used first, and stop as soon as
    /// the candidate fits — do NOT over-evict. Here budget 300, used 300 (three 100-byte entries),
    /// candidate 100 → evicting the single LRU entry (tick 1 → id 1) frees exactly enough.
    func testEvictsLeastRecentlyUsedUntilCandidateFits() throws {
        let existing = [
            entry(1, bytes: 100, tick: 1),  // LRU
            entry(2, bytes: 100, tick: 5),
            entry(3, bytes: 100, tick: 9),
        ]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 100, existing: existing, budgetBytes: 300)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: [1]))
    }

    /// Two entries must go when one isn't enough — still LRU order, still STOPPING the moment it
    /// fits (the third survives). budget 400, used 300, candidate 250: evict tick 1 (used→200,
    /// 200+250=450>400) then tick 5 (used→100, 100+250=350≤400, stop). Entry with tick 9 survives —
    /// proving the loop does not over-evict.
    func testEvictsMultipleInLRUOrder_stopsWhenFits() throws {
        let existing = [
            entry(3, bytes: 100, tick: 9),
            entry(1, bytes: 100, tick: 1),  // LRU
            entry(2, bytes: 100, tick: 5),
        ]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 250, existing: existing, budgetBytes: 400)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: [1, 2]))
    }

    /// A candidate larger than the whole budget is rejected WITHOUT evicting anything — clearing the
    /// store would not help it fit. Mirrors the hot path's `snapshotExceedsBudget`.
    func testCandidateExceedingWholeBudgetRejectedWithoutEviction() throws {
        let existing = [entry(1, bytes: 100, tick: 1)]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 301, existing: existing, budgetBytes: 300)
        XCTAssertEqual(decision, .reject(.candidateExceedsBudget))
    }

    /// A candidate exactly equal to the budget is admissible — it evicts everything, then fits.
    func testCandidateEqualToBudget_evictsAll() throws {
        let existing = [entry(7, bytes: 120, tick: 3), entry(4, bytes: 80, tick: 1)]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 300, existing: existing, budgetBytes: 300)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: [4, 7]))
    }

    /// Equal `lastUsedTick` must tie-break deterministically by lowest id, matching
    /// `ExactPrefixCache.leastRecentlyUsedEntryID`. budget 200, used 200, candidate 100 → one 100-B
    /// eviction; both entries share tick 5, so id 2 (lower) goes first.
    func testEqualLastUsedTick_tieBreakByLowestId() throws {
        let existing = [entry(9, bytes: 100, tick: 5), entry(2, bytes: 100, tick: 5)]
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 100, existing: existing, budgetBytes: 200)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: [2]))
    }

    /// An empty store with a fitting candidate admits with no evictions.
    func testEmptyStore_admitsWithoutEviction() throws {
        let decision = try ColdSnapshotStorePlanner.planCommit(
            candidateBytes: 100, existing: [], budgetBytes: 300)
        XCTAssertEqual(decision, .admit(evictedEntryIDs: []))
    }

    /// Non-positive candidate bytes or budget are invalid inputs — throw rather than make a bogus
    /// admit/reject decision.
    func testNonPositiveCandidateBytes_orBudget_throws() {
        XCTAssertThrowsError(
            try ColdSnapshotStorePlanner.planCommit(
                candidateBytes: 0, existing: [], budgetBytes: 300)
        ) { XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidCandidateBytes(0)) }
        XCTAssertThrowsError(
            try ColdSnapshotStorePlanner.planCommit(
                candidateBytes: -5, existing: [], budgetBytes: 300)
        ) { XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidCandidateBytes(-5)) }
        XCTAssertThrowsError(
            try ColdSnapshotStorePlanner.planCommit(
                candidateBytes: 100, existing: [], budgetBytes: 0)
        ) { XCTAssertEqual($0 as? ExactPrefixColdSnapshotError, .invalidStoreBudgetBytes(0)) }
    }
}
