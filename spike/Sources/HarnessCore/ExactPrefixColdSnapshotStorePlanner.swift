import Foundation

/// One stored cold (SSD) snapshot as the store planner sees it: a stable `id`, its persisted `bytes`
/// (from `modeledSnapshotKVBytes` at write time), and a monotonic `lastUsedTick` for LRU ordering.
/// The planner owns no state — the caller supplies the current set of stored entries each call — so
/// this is a plain value, not a live handle to disk.
public struct ColdSnapshotStoredEntry: Equatable, Sendable {
    public let id: UInt64
    public let bytes: Int
    public let lastUsedTick: UInt64

    public init(id: UInt64, bytes: Int, lastUsedTick: UInt64) {
        self.id = id
        self.bytes = bytes
        self.lastUsedTick = lastUsedTick
    }
}

/// Why a cold-store commit was refused outright (as opposed to admitted after eviction).
public enum ColdSnapshotStoreRejectReason: String, Equatable, Sendable {
    /// The candidate alone is larger than the WHOLE budget — evicting everything still would not
    /// make it fit, so nothing is evicted. Mirrors `ExactPrefixCommitSkipReason.snapshotExceedsBudget`.
    case candidateExceedsBudget = "candidate-exceeds-budget"
}

/// The planner's verdict for one candidate commit: either admit it (naming the stored entries that
/// must be evicted first — possibly none) or reject it. `evictedEntryIDs` is in eviction order
/// (least-recently-used first); an empty list means the candidate fit in free space. Mirrors the
/// hot-path `ExactPrefixReservationDecision` (skipReason + evictedEntryIDs), decoupled from any live
/// cache.
public enum ColdSnapshotStoreDecision: Equatable, Sendable {
    case admit(evictedEntryIDs: [UInt64])
    case reject(ColdSnapshotStoreRejectReason)
}

/// Pure admission/eviction PLANNER for the cold (SSD) snapshot tier — the decision half of "does a
/// new snapshot fit the SSD budget, and if not, what must be evicted". It computes a decision from
/// the candidate size, the current stored set, and the budget; it performs no I/O, holds no state,
/// and commits no on-disk format (persistence stays deferred behind continuous-batching slot
/// ownership, like the rest of this plane).
///
/// The policy is transcribed from the audited hot-path `ExactPrefixCache` so the cold tier can't
/// drift from the warm one:
///   - reject-without-evict when the candidate exceeds the whole budget (never clear the store for a
///     snapshot that still won't fit);
///   - evict least-recently-used first, tie-broken by lowest id (deterministic), stopping the moment
///     the candidate fits (no over-eviction).
public enum ColdSnapshotStorePlanner {

    /// Decide whether `candidateBytes` may be committed to a cold store currently holding `existing`
    /// under a hard `budgetBytes` ceiling.
    ///
    /// - Throws `invalidCandidateBytes`/`invalidStoreBudgetBytes` for non-positive inputs (a snapshot
    ///   worth persisting has positive size; a store needs a positive budget).
    /// - Returns `.reject(.candidateExceedsBudget)` — with NO evictions — when the candidate alone is
    ///   larger than the budget.
    /// - Otherwise returns `.admit(evictedEntryIDs:)` where the ids are the least-recently-used
    ///   entries (fewest possible) whose removal makes room, in eviction order.
    public static func planCommit(
        candidateBytes: Int, existing: [ColdSnapshotStoredEntry], budgetBytes: Int
    ) throws -> ColdSnapshotStoreDecision {
        guard budgetBytes > 0 else {
            throw ExactPrefixColdSnapshotError.invalidStoreBudgetBytes(budgetBytes)
        }
        guard candidateBytes > 0 else {
            throw ExactPrefixColdSnapshotError.invalidCandidateBytes(candidateBytes)
        }
        // A candidate larger than the whole budget can never fit — refuse without touching the store.
        guard candidateBytes <= budgetBytes else {
            return .reject(.candidateExceedsBudget)
        }

        var usedBytes = existing.reduce(0) { $0 + $1.bytes }
        // Fits in the current free space — admit with no evictions.
        if usedBytes + candidateBytes <= budgetBytes {
            return .admit(evictedEntryIDs: [])
        }

        // Evict least-recently-used first (tie-break: lowest id), stopping as soon as the candidate
        // fits. Terminates because candidateBytes <= budgetBytes: evicting every entry drives
        // usedBytes to 0, and 0 + candidateBytes <= budgetBytes.
        let byLRU = existing.sorted {
            $0.lastUsedTick == $1.lastUsedTick ? $0.id < $1.id : $0.lastUsedTick < $1.lastUsedTick
        }
        var evicted: [UInt64] = []
        for entry in byLRU {
            evicted.append(entry.id)
            usedBytes -= entry.bytes
            if usedBytes + candidateBytes <= budgetBytes { break }
        }
        return .admit(evictedEntryIDs: evicted)
    }
}
