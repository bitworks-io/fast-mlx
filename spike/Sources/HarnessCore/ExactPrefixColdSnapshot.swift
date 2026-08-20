import Foundation

/// Pure block-alignment planner for cold (SSD) prefix snapshots — the slot-ownership-agnostic
/// arithmetic half of the exact-prefix/session-cache feature
/// (`docs/task-inbox/2026-07-12-exact-prefix-session-cache.md`, roadmap #3a). It answers one
/// question with no side effects: given a prompt of `promptTokenCount` tokens and a `blockSize`,
/// how many WHOLE blocks may be persisted, and how long is the partial tail that must be recomputed
/// exactly this turn?
///
/// The "full-block-only" rule is the correctness spine (capture line 39 + its "mistaking an
/// intentionally recomputed partial block for dropped context" failure case): a snapshot persists
/// only complete blocks, so the trailing partial block — which is where a final system/user
/// instruction lands when the prompt is not a block multiple — is always recomputed from live
/// tokens, never restored from disk. That is why cache-on/off temperature-0 output can be
/// byte-identical.
///
/// Deliberately NOT here: the on-disk manifest, its digest scheme, and commit/restore
/// orchestration. The capture defers those behind continuous-batching slot ownership
/// (`implementation_ready: false`); this plane commits no persisted format.
public enum ExactPrefixColdSnapshotError: Error, Equatable, Sendable {
    /// `blockSize <= 0` — a block must hold at least one token.
    case invalidBlockSize(Int)
    /// `promptTokenCount < 0` — a prompt cannot have negative length.
    case invalidPromptTokenCount(Int)
    /// A stored snapshot whose token count is not a whole-block multiple — it violates the
    /// full-block-only invariant (`plan` only ever persists complete blocks), so a partial trailing
    /// block could never have been written. Fail closed rather than treat it as reusable.
    case snapshotNotWholeBlock(storedTokenCount: Int, blockSize: Int)
    /// `persistedTokenCount < 0` passed to `modeledSnapshotKVBytes` — a snapshot cannot hold a
    /// negative number of tokens.
    case invalidPersistedTokenCount(Int)
    /// `modeledSnapshotKVBytes` was asked to size a snapshot for an arch class whose cold-restore
    /// arithmetic is NOT audited safe off-box (anything other than `.uniformGQA`/`.hybridLinear` —
    /// exactly the classes `ServingSnapshotBridge.predictedSnapshotReuse` refuses). Fail closed
    /// rather than return a fabricated or zero size that could make an unservable snapshot look like
    /// it fits the SSD budget.
    case snapshotSizingUnsupported(ArchClass)
    /// `candidateBytes <= 0` passed to `ColdSnapshotStorePlanner.planCommit` — a snapshot worth
    /// persisting has positive size (size it with `modeledSnapshotKVBytes` first).
    case invalidCandidateBytes(Int)
    /// `budgetBytes <= 0` passed to `ColdSnapshotStorePlanner.planCommit` — an SSD store with no
    /// budget can hold nothing; a caller must supply a positive budget.
    case invalidStoreBudgetBytes(Int)
}

/// The whole-block split of one prompt: `persistedBlockCount` complete blocks
/// (`persistedTokenCount == persistedBlockCount * blockSize` tokens) that a cold snapshot may hold,
/// and a `recomputeTailTokenCount`-token partial tail (`0 ..< blockSize`) that must be recomputed.
///
/// Invariants (see `ExactPrefixColdSnapshotTests`): `persistedTokenCount % blockSize == 0`;
/// `0 <= recomputeTailTokenCount < blockSize`; `persistedTokenCount + recomputeTailTokenCount ==
/// promptTokenCount` (no context is dropped or duplicated).
public struct ColdSnapshotBlockPlan: Equatable, Sendable {
    public let promptTokenCount: Int
    public let blockSize: Int
    public let persistedBlockCount: Int
    public let persistedTokenCount: Int
    public let recomputeTailTokenCount: Int

    /// Whether the final `suffixLength` prompt tokens lie wholly inside the recomputed tail — i.e.
    /// none of them was frozen into a persisted block, so they are recomputed exactly this turn
    /// rather than restored from a snapshot. This is the capture's "put the final system/user
    /// instruction wholly in that tail" property as arithmetic: since both the suffix and the tail
    /// are suffixes of the same token sequence, the instruction is fully recomputed exactly when it
    /// is no longer than the tail. Returns `false` for a negative `suffixLength` (not a suffix).
    public func suffixIsFullyRecomputed(suffixLength: Int) -> Bool {
        suffixLength >= 0 && suffixLength <= recomputeTailTokenCount
    }
}

/// The reuse reconciliation of one stored whole-block snapshot against a NEW prompt that may diverge
/// from it: how many persisted blocks are exactly reusable, and how many prompt tokens must be
/// recomputed this turn.
///
/// The reuse boundary is the LONGEST common token prefix, floored to whole blocks and capped at the
/// stored block count — a block is reusable only if every one of its tokens matches the stored
/// snapshot (the first block containing a divergent token, and everything after, is recomputed). This
/// is why a restore can be byte-identical to a cold run at temperature 0: no partially-divergent
/// block is ever served from disk.
///
/// Invariants (see `ExactPrefixColdSnapshotTests`): `restoredTokenCount % blockSize == 0`;
/// `restoredBlockCount <= storedBlockCount`; `restoredTokenCount <= commonPrefixTokenCount`;
/// `restoredTokenCount + recomputeTokenCount == promptTokenCount` (no context dropped or duplicated).
///
/// CONSUMER INVARIANT (not enforced here — this plane is pure arithmetic): a serving restore must
/// recompute at least one token even on a full cache hit. When the prompt exactly matches the stored
/// snapshot this plan reports `recomputeTokenCount == 0`, but K/V is cached while the last position's
/// LOGITS are not, so a consumer that forwarded zero tokens could not produce the first output token.
/// Whether the ≥1-recompute is taken by dropping the last restored block or just the last token is a
/// deliberate design choice deferred to the orchestration layer
/// (`docs/task-inbox/2026-08-19-cold-snapshot-restore-min-recompute.md`).
public struct ColdSnapshotRestorePlan: Equatable, Sendable {
    public let blockSize: Int
    /// Whole blocks in the stored snapshot (`storedTokenCount / blockSize`).
    public let storedBlockCount: Int
    /// The new prompt's total token count.
    public let promptTokenCount: Int
    /// Length of the longest common leading token run between the stored snapshot and the prompt.
    public let commonPrefixTokenCount: Int
    /// Whole persisted blocks that are exactly reusable this turn.
    public let restoredBlockCount: Int
    /// Tokens restored from the snapshot (`restoredBlockCount * blockSize`).
    public let restoredTokenCount: Int
    /// Prompt tokens that must be recomputed this turn (`promptTokenCount - restoredTokenCount`).
    public let recomputeTokenCount: Int

    /// Whether the final `suffixLength` prompt tokens lie wholly inside the recomputed region — i.e.
    /// none of them was served from a restored block, so they are recomputed exactly this turn rather
    /// than restored from the snapshot. This is the restore-side twin of
    /// `ColdSnapshotBlockPlan.suffixIsFullyRecomputed`: since both the suffix and the recompute region
    /// are suffixes of the same prompt, the instruction is fully recomputed exactly when it is no
    /// longer than `recomputeTokenCount`. Returns `false` for a negative `suffixLength` (not a suffix).
    ///
    /// Note the honest edge this surfaces rather than hides: on a full cache hit
    /// (`recomputeTokenCount == 0`) this returns `false` for ANY positive suffix, because nothing is
    /// recomputed — which is exactly the ≥1-recompute question the CONSUMER INVARIANT defers to the
    /// orchestration layer (`docs/task-inbox/2026-08-19-cold-snapshot-restore-min-recompute.md`).
    public func suffixIsFullyRecomputed(suffixLength: Int) -> Bool {
        suffixLength >= 0 && suffixLength <= recomputeTokenCount
    }
}

/// The winning candidate when several stored snapshots share one semantic key and a restore must pick
/// exactly one to reuse — the cold-plane twin of `ExactPrefixCache`'s longest-prefix `lookup` over its
/// per-key entry set. Carries the winning candidate's index (into the input array) and its restore plan
/// so a consumer can both act on the arithmetic and know which stored snapshot it chose.
public struct ColdSnapshotRestoreSelection: Equatable, Sendable {
    /// Index into the `candidateStoredPrefixes` array of the chosen snapshot.
    public let candidateIndex: Int
    /// The restore plan for the chosen snapshot against the prompt.
    public let plan: ColdSnapshotRestorePlan
}

/// How much of a stored snapshot may be reused when a new prompt diverges from it — a property of
/// the model's cache layout, not the prompt.
///
/// - `.blockAligned`: dense attention (`KVCacheSimple`/`StandardKVCache`). Each whole block's K/V is
///   independent, so a mid-snapshot divergence still reuses every block BEFORE the first divergent
///   one. This is the original `planRestore` behavior and stays the default.
/// - `.wholeSnapshotOnly`: recurrent/linear-attention state (`MambaCache`/GatedDeltaNet — the hybrid
///   families like qwen3_5, fast-mlx's flagship serving line). The recurrent state exists at EXACTLY
///   the stored token count and cannot be rewound to an earlier block boundary without replaying, so
///   a snapshot is reusable only in FULL: the prompt's common prefix must cover the entire stored
///   snapshot, otherwise nothing is reused. This bakes in the capture's warning that "hybrid/SSM
///   checkpoints are explicit boundaries; arbitrary trim is not assumed" before orchestration
///   consumes the wrong (dense) arithmetic. (The `plan()`-side concern — capturing recurrent state
///   at a block boundary so a whole-snapshot restore is even possible — remains an orchestration
///   matter, deferred with the rest of the persistence layer.)
public enum ColdSnapshotReuseGranularity: Equatable, Sendable {
    case blockAligned
    case wholeSnapshotOnly
}

public enum ExactPrefixColdSnapshotPlanner {
    /// Split `promptTokenCount` into whole persistable blocks plus a recomputed tail. Fails closed
    /// on a non-positive `blockSize` or a negative `promptTokenCount` rather than returning a
    /// nonsense plan.
    public static func plan(promptTokenCount: Int, blockSize: Int) throws -> ColdSnapshotBlockPlan {
        guard blockSize > 0 else {
            throw ExactPrefixColdSnapshotError.invalidBlockSize(blockSize)
        }
        guard promptTokenCount >= 0 else {
            throw ExactPrefixColdSnapshotError.invalidPromptTokenCount(promptTokenCount)
        }
        let persistedBlockCount = promptTokenCount / blockSize
        let persistedTokenCount = persistedBlockCount * blockSize
        let recomputeTailTokenCount = promptTokenCount - persistedTokenCount
        return ColdSnapshotBlockPlan(
            promptTokenCount: promptTokenCount,
            blockSize: blockSize,
            persistedBlockCount: persistedBlockCount,
            persistedTokenCount: persistedTokenCount,
            recomputeTailTokenCount: recomputeTailTokenCount)
    }

    /// Reconcile a stored whole-block snapshot against a new prompt: reuse the persisted blocks that
    /// are wholly within the two sequences' common leading token prefix, recompute the rest. Fails
    /// closed on a non-positive `blockSize` and on a stored snapshot that is not a whole-block
    /// multiple (which could never have been written by `plan`).
    public static func planRestore(
        storedPrefixTokens: [Int], promptTokens: [Int], blockSize: Int,
        granularity: ColdSnapshotReuseGranularity = .blockAligned
    ) throws -> ColdSnapshotRestorePlan {
        guard blockSize > 0 else {
            throw ExactPrefixColdSnapshotError.invalidBlockSize(blockSize)
        }
        guard storedPrefixTokens.count % blockSize == 0 else {
            throw ExactPrefixColdSnapshotError.snapshotNotWholeBlock(
                storedTokenCount: storedPrefixTokens.count, blockSize: blockSize)
        }
        let storedBlockCount = storedPrefixTokens.count / blockSize

        // Longest common leading token run between the stored snapshot and the new prompt.
        var commonPrefix = 0
        let limit = min(storedPrefixTokens.count, promptTokens.count)
        while commonPrefix < limit && storedPrefixTokens[commonPrefix] == promptTokens[commonPrefix] {
            commonPrefix += 1
        }

        let restoredBlockCount: Int
        switch granularity {
        case .blockAligned:
            // Dense attention: any block wholly inside the common prefix is reusable; cap at stored.
            restoredBlockCount = min(commonPrefix / blockSize, storedBlockCount)
        case .wholeSnapshotOnly:
            // Recurrent/linear state can't rewind to an earlier boundary: the snapshot is reusable
            // only when the common prefix covers ALL of it, otherwise nothing (a mid-snapshot
            // divergence — or a prompt that stops short of the stored length — reuses zero blocks).
            restoredBlockCount = commonPrefix >= storedPrefixTokens.count ? storedBlockCount : 0
        }
        let restoredTokenCount = restoredBlockCount * blockSize
        let recomputeTokenCount = promptTokens.count - restoredTokenCount
        return ColdSnapshotRestorePlan(
            blockSize: blockSize,
            storedBlockCount: storedBlockCount,
            promptTokenCount: promptTokens.count,
            commonPrefixTokenCount: commonPrefix,
            restoredBlockCount: restoredBlockCount,
            restoredTokenCount: restoredTokenCount,
            recomputeTokenCount: recomputeTokenCount)
    }

    /// Choose which stored snapshot to restore when several share one semantic key. Runs `planRestore`
    /// per candidate under the model's `granularity` and returns the candidate that recovers the most
    /// tokens (`restoredTokenCount`), so a consumer never has to re-derive the reuse arithmetic to pick
    /// a winner. This mirrors `ExactPrefixCache.lookup`, which already selects the longest-prefix entry
    /// among the multiple snapshots stored per key.
    ///
    /// Policy (deliberately in the arithmetic plane, committing no persisted format):
    /// - Objective: maximize `restoredTokenCount`. Under `.blockAligned` this is the deepest common
    ///   whole-block prefix; under `.wholeSnapshotOnly` only candidates the prompt fully covers can
    ///   score above zero, so a shorter fully-covered snapshot can beat a longer one that diverges.
    /// - Tie-break: lowest `candidateIndex`, so the result is deterministic and stable.
    /// - Returns `nil` when nothing is reusable (empty list, or every candidate restores zero tokens):
    ///   "nothing to reuse" is a real outcome the consumer must handle, not a zero-token plan to act on.
    /// - Fails closed for the WHOLE selection if any candidate is not a whole-block multiple — such a
    ///   snapshot could never have been written by `plan`, so its presence signals corruption rather
    ///   than a losing candidate to skip (same stance `planRestore` takes for one snapshot).
    public static func planBestRestore(
        candidateStoredPrefixes: [[Int]], promptTokens: [Int], blockSize: Int,
        granularity: ColdSnapshotReuseGranularity = .blockAligned
    ) throws -> ColdSnapshotRestoreSelection? {
        guard blockSize > 0 else {
            throw ExactPrefixColdSnapshotError.invalidBlockSize(blockSize)
        }
        var best: ColdSnapshotRestoreSelection?
        for (index, stored) in candidateStoredPrefixes.enumerated() {
            // planRestore fails closed on a non-whole-block candidate — propagate for the whole
            // selection rather than silently dropping a corrupt snapshot.
            let plan = try planRestore(
                storedPrefixTokens: stored, promptTokens: promptTokens, blockSize: blockSize,
                granularity: granularity)
            guard plan.restoredTokenCount > 0 else { continue }
            // Strictly-greater keeps the first (lowest-index) candidate on ties — deterministic.
            if best == nil || plan.restoredTokenCount > best!.plan.restoredTokenCount {
                best = ColdSnapshotRestoreSelection(candidateIndex: index, plan: plan)
            }
        }
        return best
    }

    /// The modeled KV size, in bytes, of a cold snapshot holding `persistedTokenCount` tokens for
    /// `profile` — the byte-axis companion to `plan`'s token-axis split, used by the SSD budget to
    /// answer "how big is this snapshot / does it fit". Single-source-locked to `CapacityModel` (the
    /// one KV formula), so it inherits the class-specific assembly for free: uniformGQA grows every
    /// layer, hybridLinear grows only its attention subset and adds its fixed recurrent-state term.
    ///
    /// FAIL-CLOSED on two axes:
    /// - Arch class: only `.uniformGQA` and `.hybridLinear` — exactly the classes
    ///   `ServingSnapshotBridge.predictedSnapshotReuse` marks reusable — may be sized. Every other
    ///   class throws `snapshotSizingUnsupported` rather than returning a fabricated or zero size
    ///   that could make an unservable snapshot look like it fits the budget. The `switch` is
    ///   exhaustive over `ArchClass`, so a new arch breaks compilation here (it must be classified,
    ///   not silently sized).
    /// - Token count: a negative count throws `invalidPersistedTokenCount`; zero is valid and
    ///   returns just the fixed-state term (an empty snapshot still carries recurrent state).
    ///
    /// `.fp16` is pinned deliberately: the runtime allocates fp16 KV, so a snapshot of live KV is
    /// fp16 on disk. A `kvQuant` parameter here would let a caller name a size the write never has —
    /// the same phantom-sizing hazard the tier-dial capture blocks. This is a MODELED estimate from
    /// confirmed geometry, never a claim about a file's on-disk byte length.
    public static func modeledSnapshotKVBytes(
        profile: ModelArchProfile, persistedTokenCount: Int
    ) throws -> Double {
        guard persistedTokenCount >= 0 else {
            throw ExactPrefixColdSnapshotError.invalidPersistedTokenCount(persistedTokenCount)
        }
        switch profile.modelType {
        case .uniformGQA, .hybridLinear:
            return CapacityModel.kvBytesForContext(
                profile, context: persistedTokenCount, kvQuant: .fp16, concurrency: 1)
        case .interleavedSWA, .dualGeometrySWA, .mlaAsImplemented, .hybridMamba2MoE, .novelCompressedUnsupported:
            throw ExactPrefixColdSnapshotError.snapshotSizingUnsupported(profile.modelType)
        }
    }
}
