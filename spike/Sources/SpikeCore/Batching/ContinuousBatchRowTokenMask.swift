import HarnessCore
import MLX

/// Dense additive row built from a `ContinuousBatchRowTokenMask` bitset (the protocol itself is
/// declared in HarnessCore beside `ContinuousBatchSubmission`, which carries it).
enum ContinuousBatchRowTokenMaskRow {
    /// `width` Float values: `0` for every set bit with `id < width`, `-inf` everywhere else. That
    /// includes every column at or beyond the bitset's own width (a padded vocabulary) and ignores
    /// set bits at or beyond `width`. Word-scan with `trailingZeroBitCount`, exactly like
    /// `ConstraintMaskingLogitProcessor.process`, so a sparse bitset never materializes an id list.
    ///
    /// Returns `nil` when NO set bit has `id < width`. Such a bitset would build an all-`-inf` row,
    /// the batch readback would see the `-1` sentinel and throw `invalidTokenID`, and the
    /// coordinator answers any runtime throw with `failAll(terminal: true)`: one row's bad mask
    /// would fail every user's request and shut the route down. The protocol contract is that one
    /// row's failure can never fail a shared step, so the runtime treats this case exactly as a
    /// `nil` bitset (unconstrained for this step). The row's conformer then rejects the emitted
    /// token in `advance` and records its own failure, which fails that one request only.
    static func values(bitset: [UInt64], width: Int) -> [Float]? {
        var values = [Float](repeating: -Float.infinity, count: width)
        var hasInWidthColumn = false
        for (wordIndex, word) in bitset.enumerated() {
            guard word != 0 else { continue }
            var bits = word
            while bits != 0 {
                let id = wordIndex * 64 + bits.trailingZeroBitCount
                if id < width {
                    values[id] = 0
                    hasInWidthColumn = true
                }
                bits &= bits - 1
            }
        }
        return hasInWidthColumn ? values : nil
    }

    /// The `[1, width]` additive row in the logits dtype, or `nil` when the bitset has no column
    /// inside `width` (see `values`): the caller then treats the row as unconstrained.
    static func row(bitset: [UInt64], width: Int, dtype: DType) -> MLXArray? {
        values(bitset: bitset, width: width).map {
            MLXArray($0).reshaped([1, width]).asType(dtype)
        }
    }

    /// Greedy reduction of `logits + additive` where `additive` carries `-inf` mask columns.
    ///
    /// `CompiledMLXDecoder.greedyTokenOrInvalidSentinel` treats every non-finite logit as invalid,
    /// which would reject a legitimately masked row. This variant keeps the same fail-closed
    /// contract on the model's own output: the sentinel `-1` is returned when the RAW logits hold
    /// any non-finite value (same all-rows scope as the original), or when a row's masked maximum
    /// is not finite (a mask that allows no logits column).
    static func maskedGreedyToken(logits: MLXArray, additive: MLXArray) -> MLXArray {
        let masked = logits + additive
        let token = argMax(masked, axis: -1).asType(.int32)
        let valid = isFinite(logits).all() & isFinite(masked.max(axis: -1))
        return MLX.where(valid, token, MLXArray(Int32(-1)))
    }
}
