import Foundation
import MLX
import MLXLMCommon

/// S2 of the hybrid continuous-batching build (design of record: the continuous-batching
/// heterogeneous-cache design, §2.3). The
/// recurrent-kind sibling of `BatchedCompiledKVCache`: it merges the fixed-size recurrent state
/// (GatedDeltaNet / Mamba: conv tail + fp32 SSM state) of N per-request rows into one batched state
/// carrying the cohort on dim 0, and slices it back out for eviction/spill.
///
/// It is deliberately MUCH simpler than the dense `BatchedCompiledKVCache`: recurrent state is
/// length-independent and fixed-size, so every row's arrays share one shape `[1, …]`. There is no
/// capacity to pad to, no right-padded tail, no per-row logical length inside the array — merge is a
/// plain `concatenate(axis: 0)` and extract is a plain dim-0 slice. That ABSENCE of padding is exactly
/// what makes the round-trip bit-identical (`merge(rows).extract(i)` == `rows[i]`, the S2 gate).
///
/// The per-request row (design name `RecurrentScalarRowCache`) is the vendored `MambaCache` itself,
/// conformed to `ContinuousScalarRowCache` below — mirroring how the dense arms conform
/// `CompiledKVCache`/`AffineKVCache` directly rather than wrapping them, so there is no `KVCache`
/// delegation surface to keep in sync with upstream.

/// The recurrent per-request row: the vendored `MambaCache` (an `ArraysCache(size: 2)` — slot 0 conv
/// tail, slot 1 fp32 SSM state). Its `offset` is the committed token count, but the vendored
/// `MambaCache.advance(_:)` adjusts only transient `lengths`/`leftPadding` and never `BaseKVCache.offset`
/// — so the continuous-batch driver, not the model step, maintains this row's `offset` (S3 requirement).
public typealias RecurrentScalarRowCache = MambaCache

/// A recurrent row must be usable as `compile(inputs:outputs:)` state so the compiled decode step
/// tracks the SSM/conv arrays across calls (otherwise MLX freezes them as traced constants and the
/// recurrence runs silently stateless). `MambaCache` already satisfies `Updatable`'s sole
/// requirement — `innerState()` via `Evaluatable`/`KVCache` — so this conformance is witness-free.
/// Retroactive because both `MambaCache` and `Updatable` are vendored; the runtime owns the seam.
extension MambaCache: @retroactive Updatable {}

extension MambaCache: ContinuousScalarRowCache {
    var continuousLogicalOffset: Int { offset }

    /// Growth is a provable no-op for a recurrent row: the state is fixed-size and never grows with
    /// context (mirrors `BatchedRecurrentStateCache.grow(by:)` on the batched side).
    func grow(by chunk: Int) {
        precondition(chunk > 0, "cache growth must be positive")
        // Intentionally empty: recurrent state footprint is length-independent (design §4.1).
    }
}

/// Validation failures at the recurrent scalar-row / batched-state membership boundary. Recoverable
/// scheduler/driver errors, not preconditions: a malformed cohort fails closed before any row is mutated.
public enum BatchedRecurrentStateCacheError: Error, Equatable {
    case emptyBatch
    case lengthCount(expected: Int, actual: Int)
    case lengthMismatch(slot: Int, expected: Int, actual: Int)
    case uninitializedSlot(Int)
    case incompatibleShape(slot: Int, expected: [Int], actual: [Int])
    case incompatibleDType(slot: Int)
    case invalidSlot(index: Int, batchSize: Int)
}

/// Batched recurrent state for one linear layer: a conv-tail array `[N, k-1, convDim]` (model dtype)
/// and an SSM-state array `[N, Hv, Dv, Dk]` (fp32 by vendored construction), plus each row's logical
/// token count. `N` is the cohort size on dim 0.
public final class BatchedRecurrentStateCache {
    /// The single authoritative batched recurrent state, stored on an OWNED inner `MambaCache`
    /// (`ArraysCache(size: 2)`: slot 0 conv tail `[N, k-1, convDim]` model dtype, slot 1 SSM state
    /// `[N, Hv, Dv, Dk]` fp32), cohort `N` on dim 0. This is the object the runtime hands the model at a
    /// linear layer's index (`modelCache`): the model reads it via the concrete `cache as? MambaCache`
    /// downcast (`Qwen35.swift:487,537`) and writes updated state back through the SAME object
    /// (`:254,:289`), so composing on an inner `MambaCache` makes those write-backs mutate this wrapper's
    /// authoritative state BY CONSTRUCTION — no raw-array duplicate to fall stale (S3 finding, design §2.3:
    /// "a MambaCache whose arrays carry B = cohort size on dim 0"). The bespoke fail-closed
    /// merging/extract validation, per-row `logicalOffsets`, and the error enum are retained; the inner
    /// `MambaCache`'s own `advance`/`extend`/`filter` are NOT used (they violate fail-closed discipline).
    private let inner: MambaCache

    /// Conv-tail state, cohort on dim 0: `[N, convKernelSize-1, convDim]`, model dtype. Computed read of
    /// the inner `MambaCache` slot 0, so a model-side write-back is reflected here with no re-sync.
    public var convState: MLXArray {
        // Force-unwrap is safe: `merging` always initializes both slots and never clears them.
        inner[0]!
    }
    /// SSM state, cohort on dim 0: `[N, Hv, Dv, Dk]`, fp32. Computed read of the inner slot 1.
    public var ssmState: MLXArray { inner[1]! }
    /// Per-row committed token count, in row order.
    public private(set) var logicalOffsets: [Int]

    /// The model-visible recurrent cache for this linear layer: the owned inner `MambaCache` itself. The
    /// runtime places THIS object at the linear layer's index in the model's `[any KVCache]` array so the
    /// model's `cache as? MambaCache` downcast succeeds and its write-backs land on the authoritative
    /// state. Typed as `any KVCache` to witness the `ContinuousBatchedRowCache.modelCache` seam (Swift
    /// forbids a covariant concrete witness); its dynamic type is always `MambaCache`. Identity is stable
    /// for the wrapper's lifetime (it is a stored `let`). `Updatable` (via `MambaCache`) so the runtime
    /// can reuse it as `compile` state and track the recurrent arrays across steps.
    public var modelCache: any (KVCache & Updatable) { inner }

    /// Cohort size (rows currently merged into this batched state).
    public var batchSize: Int { convState.dim(0) }

    /// Per-row logical token counts — the recurrent analogue of the dense
    /// `ContinuousBatchedKVCache.continuousLogicalOffsets`.
    public var continuousLogicalOffsets: [Int] { logicalOffsets }

    /// Highest logical token count across rows. Recurrent state has no physical write frontier of its
    /// own (it is fixed-size); this mirrors the dense `continuousPhysicalWrittenEnd` for accounting.
    public var continuousPhysicalWrittenEnd: Int { logicalOffsets.max() ?? 0 }

    private init(convState: MLXArray, ssmState: MLXArray, logicalOffsets: [Int]) {
        let inner = MambaCache()
        inner[0] = convState
        inner[1] = ssmState
        self.inner = inner
        self.logicalOffsets = logicalOffsets
    }

    /// Merge initialized recurrent scalar rows into one batched state WITHOUT modifying them.
    ///
    /// Optional `lengths` are an assertion from a driver that already tracks committed positions; they
    /// must equal each row's authoritative `offset`. Rows must share suffix shape and dtype for both the
    /// conv tail and the SSM state, and each must carry a single sequence (dim 0 == 1).
    public static func merging(
        _ rows: [RecurrentScalarRowCache], lengths explicitLengths: [Int]? = nil
    ) throws -> BatchedRecurrentStateCache {
        guard !rows.isEmpty else {
            throw BatchedRecurrentStateCacheError.emptyBatch
        }
        if let explicitLengths, explicitLengths.count != rows.count {
            throw BatchedRecurrentStateCacheError.lengthCount(
                expected: rows.count, actual: explicitLengths.count)
        }

        let authoritativeLengths = rows.map(\.offset)
        if let explicitLengths {
            for slot in rows.indices where explicitLengths[slot] != authoritativeLengths[slot] {
                throw BatchedRecurrentStateCacheError.lengthMismatch(
                    slot: slot,
                    expected: authoritativeLengths[slot],
                    actual: explicitLengths[slot])
            }
        }

        guard let firstConv = rows[0][0], let firstSSM = rows[0][1] else {
            throw BatchedRecurrentStateCacheError.uninitializedSlot(0)
        }
        let convSuffix = Array(firstConv.shape.dropFirst())
        let ssmSuffix = Array(firstSSM.shape.dropFirst())

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(rows.count)
        ssmRows.reserveCapacity(rows.count)

        for (slot, row) in rows.enumerated() {
            guard let conv = row[0], let ssm = row[1] else {
                throw BatchedRecurrentStateCacheError.uninitializedSlot(slot)
            }
            guard conv.dim(0) == 1, Array(conv.shape.dropFirst()) == convSuffix else {
                throw BatchedRecurrentStateCacheError.incompatibleShape(
                    slot: slot, expected: [1] + convSuffix, actual: conv.shape)
            }
            guard ssm.dim(0) == 1, Array(ssm.shape.dropFirst()) == ssmSuffix else {
                throw BatchedRecurrentStateCacheError.incompatibleShape(
                    slot: slot, expected: [1] + ssmSuffix, actual: ssm.shape)
            }
            guard conv.dtype == firstConv.dtype, ssm.dtype == firstSSM.dtype else {
                throw BatchedRecurrentStateCacheError.incompatibleDType(slot: slot)
            }
            convRows.append(conv)
            ssmRows.append(ssm)
        }

        return BatchedRecurrentStateCache(
            convState: concatenated(convRows, axis: 0),
            ssmState: concatenated(ssmRows, axis: 0),
            logicalOffsets: authoritativeLengths)
    }

    /// Return one independent recurrent scalar row (`MambaCache`) for the given slot, sliced out of the
    /// batched state along dim 0. Bit-identical to the row that was merged in — no padding, no capacity.
    public func extract(slot: Int) throws -> RecurrentScalarRowCache {
        guard slot >= 0, slot < batchSize else {
            throw BatchedRecurrentStateCacheError.invalidSlot(index: slot, batchSize: batchSize)
        }
        let row = MambaCache()
        row[0] = convState[slot ..< (slot + 1)]
        row[1] = ssmState[slot ..< (slot + 1)]
        row.offset = logicalOffsets[slot]
        return row
    }

    /// Growth is a provable no-op for recurrent state: it is fixed-size and never grows with context.
    /// Present so the batched-cache caller can treat every layer uniformly; it touches nothing.
    public func grow(by chunk: Int) {
        precondition(chunk > 0, "cache growth must be positive")
        // Intentionally empty: recurrent state footprint is length-independent (design §4.1).
    }

    /// Advance every row's committed token count in lockstep. The continuous-batch driver commits
    /// exactly `delta` tokens per batched decode step, but nothing in the vendored model advances a
    /// `MambaCache`'s logical offset (`ArraysCache.advance` touches only lengths/leftPadding; S3
    /// finding 3) — so the driver maintains the batched-row offsets here. Without this, `extract`
    /// restores a stale merge-time offset and the runtime's `validateCacheLengths` throws on every
    /// spill/re-merge. Fail-closed on a non-positive delta.
    public func advanceOffsets(by delta: Int) {
        precondition(delta > 0, "offset advance must be positive")
        logicalOffsets = logicalOffsets.map { $0 + delta }
    }
}

/// Join the kind-agnostic batched-row seam so a hybrid runtime can hold recurrent and dense batched
/// caches side by side in one `[any ContinuousBatchedRowCache]` per-layer array. The recurrent side is a
/// fixed-size state, so it deliberately does NOT join `ContinuousBatchedKVCache` (no capacity, no
/// growing KV) — only the kind-agnostic super-protocol.
extension BatchedRecurrentStateCache: ContinuousBatchedRowCache {
    func extractContinuousRow(slot: Int) throws -> any ContinuousScalarRowCache {
        try extract(slot: slot)
    }
}
