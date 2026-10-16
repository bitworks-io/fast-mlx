import Foundation

import MLX
import MLXLMCommon
import SpikeCore

/// A request's `logit_bias` map referenced a token id outside the logits row the decoder actually
/// produced. Admission (`ScalarServingBackend.start`) refuses such a request with a 400 before any
/// generation starts; this error exists so a mismatch that still reaches the decode loop (a
/// configured vocabulary size that disagrees with the real logits width) is a typed, thrown failure
/// rather than an out-of-bounds MLX index or a silently skipped bias.
public enum LogitBiasProcessorError: Error, Equatable, Sendable {
    case tokenIDOutOfRange(tokenID: Int, logitsWidth: Int)
}

/// OpenAI `logit_bias` for the plain SCALAR route: adds each configured bias to the logits of its
/// token id BEFORE sampling (so it affects greedy and sampled generation alike), then hands the
/// biased logits to an optional `inner` processor -- the `response_format` grammar mask.
///
/// ORDER IS THE CONTRACT: bias first, mask LAST. A response-format mask is the authority on which
/// tokens may be emitted at all, so a +100 bias must never resurrect a masked token. (The shipped
/// masks add `-inf`, which absorbs any finite bias in either order; running the mask last is what
/// keeps the guarantee for a mask that replaces rather than adds.) Inside `MLXDecoder` this whole
/// composite occupies the single "constraint" slot, which already runs AFTER the penalty
/// processor, so the full order is penalties -> bias -> mask.
///
/// `didSample`, `prompt` and `recordedFailure` are forwarded to / surfaced from `inner`, so the
/// decoder's existing post-sample failure check keeps working for a masked request.
///
/// Admission guarantees every id is below the model's vocabulary size; if a bias id is nevertheless
/// beyond the logits width, no partial bias is applied and a `LogitBiasProcessorError` is recorded
/// (the decoder throws it right after sampling).
///
/// `@unchecked Sendable`: one instance is built per request and then driven only from the single
/// decode loop (same justification as `ConstraintMaskingLogitProcessor`).
public final class LogitBiasLogitProcessor: LogitProcessor, ConstraintProcessorFailureReporting,
    @unchecked Sendable
{
    private let bias: [Int: Float]
    private let maximumBiasID: Int
    private var inner: (any LogitProcessor & ConstraintProcessorFailureReporting & Sendable)?
    private var ownFailure: Error?
    /// The dense `[1, width]` bias row, built once per (width, dtype) and reused on every step so
    /// the per-token cost is one elementwise add rather than a host rebuild of a vocab-sized array.
    private var cachedRow: (width: Int, dtype: DType, row: MLXArray)?

    public var recordedFailure: Error? { ownFailure ?? inner?.recordedFailure }

    public init(
        bias: [Int: Float],
        then inner: (any LogitProcessor & ConstraintProcessorFailureReporting & Sendable)? = nil
    ) {
        self.bias = bias
        self.maximumBiasID = bias.keys.max() ?? -1
        self.inner = inner
    }

    public func prompt(_ prompt: MLXArray) {
        inner?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let width = logits.dim(-1)
        var biased = logits
        if !bias.isEmpty {
            if maximumBiasID >= width {
                if ownFailure == nil {
                    ownFailure = LogitBiasProcessorError.tokenIDOutOfRange(
                        tokenID: maximumBiasID, logitsWidth: width)
                }
            } else {
                biased = logits + biasRow(width: width, dtype: logits.dtype)
            }
        }
        // Mask LAST: the grammar mask has the final word over the biased logits.
        return inner?.process(logits: biased) ?? biased
    }

    public func didSample(token: MLXArray) {
        inner?.didSample(token: token)
    }

    private func biasRow(width: Int, dtype: DType) -> MLXArray {
        if let cachedRow, cachedRow.width == width, cachedRow.dtype == dtype {
            return cachedRow.row
        }
        var values = [Float](repeating: 0, count: width)
        for (id, value) in bias {
            values[id] = value
        }
        let row = MLXArray(values).reshaped([1, width]).asType(dtype)
        cachedRow = (width, dtype, row)
        return row
    }
}
