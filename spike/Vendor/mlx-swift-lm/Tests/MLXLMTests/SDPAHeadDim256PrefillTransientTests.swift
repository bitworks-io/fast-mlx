// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

// Root cause and question this suite answers:
// docs/task-inbox (fast-mlx) -- does `scaledDotProductAttention` at
// head_dim 256 materialize an O(qL x kL) score tensor during prefill on
// the pinned mlx-core v0.31.1, and at how many bytes per score element?
//
// `scaled_dot_product_attention.cpp:623-625` (pinned mlx-core v0.31.1)
// gates the FUSED full-attention path to `head_dim in {64, 80, 128}`; the
// head_dim-256 vector path additionally requires `query_sequence_length <=
// 8` (`:634`). The production hybrid-attention geometry this repo serves
// is head_dim 256 with prefill
// query lengths far above 8, so production prefill attention is expected
// to fall back to an unfused `matmul(q, swapaxes(k, -1, -2))` that
// materializes the full `[queryHeads, qL, kL]` score tensor. This suite
// measures that transient directly rather than trusting the source-read
// gate, on this repo's default device (GPU/Metal on Apple Silicon --
// deliberately NOT `Device.withDefaultDevice(.cpu)`, since the dispatch
// gate under test lives in the SDPA backend selection that production
// actually exercises on GPU).
//
// Measurement idiom mirrors
// the drafter-priming chunking suite's subquadratic peak-memory test and
// the sparse-indexer suite's `measurePeakBytes` (~line 2218): measure
// the INCREMENT above a live baseline, because raw `Memory.peakMemory` is
// order-dependent and not reproducible read alone. This suite additionally
// subtracts the known input/output buffer bytes from that increment, per
// the task's stated idiom, to isolate the SDPA call's own TRANSIENT
// (the intermediate score tensor and any other scratch it allocates)
// rather than reporting the cost of merely holding q/k/v/out live.

private struct SDPACell {
    let headDim: Int
    let queryLength: Int
    let keyLength: Int
    let transientBytes: Int
    let bytesPerScoreElement: Double
}

@Suite(.serialized)
struct SDPAHeadDim256PrefillTransientTests {

    private static let batch = 1
    private static let queryHeads = 24
    private static let kvHeads = 2
    private static let keyLength = 8_192

    /// Builds `[batch, heads, length, headDim]` float16 query/key/value
    /// tensors with fixed, non-degenerate random content (SDPA's transient
    /// scratch allocation should not depend on values, only on shape/dtype,
    /// but non-constant input avoids masking a defect that only shows up
    /// on non-trivial data, e.g. a fast-path short-circuit on all-zero
    /// input).
    private static func makeQKV(headDim: Int, queryLength: Int, keyLength: Int) -> (
        q: MLXArray, k: MLXArray, v: MLXArray
    ) {
        let q = MLXRandom.normal([batch, queryHeads, queryLength, headDim]).asType(.float16)
        let k = MLXRandom.normal([batch, kvHeads, keyLength, headDim]).asType(.float16)
        let v = MLXRandom.normal([batch, kvHeads, keyLength, headDim]).asType(.float16)
        eval(q, k, v)
        return (q, k, v)
    }

    /// Measures the transient bytes allocated by a single SDPA call, above
    /// holding its inputs and output live, using the increment-above-a-
    /// live-baseline idiom shared by
    /// the drafter-priming chunking and sparse-indexer suites, plus the
    /// input/output-byte subtraction
    /// specified for this task.
    private static func measureTransientBytes(
        query: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> Int {
        // Warmup iteration (discarded): lets MLX/Metal perform any one-time
        // kernel-compile / buffer-pool allocations before the measured
        // call, so the measured peak reflects only this SDPA call's own
        // transient rather than first-call warm-up noise.
        // GPU (Metal) work is dispatched asynchronously: `eval` enqueues
        // and waits for the array's own command buffer, but the
        // allocator's active/peak-memory bookkeeping for buffers freed at
        // the end of that buffer can lag slightly behind `eval` returning.
        // An explicit `Stream().synchronize()` after every `eval` (used
        // nowhere in the CPU-only reference tests this idiom is borrowed
        // from, since CPU execution has no such lag) makes the
        // active/peak reads below deterministic instead of racing the
        // GPU queue -- confirmed necessary here: without it this suite's
        // measured transient was 0 B on 2 of 3 head_dim=256 cells despite
        // an identical call shape to the one cell that measured non-zero.
        do {
            let warm = MLXFast.scaledDotProductAttention(
                queries: query, keys: keys, values: values, scale: scale, mask: nil)
            eval(warm)
            Stream().synchronize()
        }

        Memory.clearCache()
        let before = Memory.activeMemory
        Memory.peakMemory = 0

        let out = MLXFast.scaledDotProductAttention(
            queries: query, keys: keys, values: values, scale: scale, mask: nil)
        eval(out)  // force materialization INSIDE the measured region
        Stream().synchronize()

        let peak = max(Memory.peakMemory, before)
        let ioBytes = query.nbytes + keys.nbytes + values.nbytes + out.nbytes
        let transient = peak - before - ioBytes
        Memory.clearCache()
        return max(0, transient)
    }

    private static func measureCell(headDim: Int, queryLength: Int, keyLength: Int) -> SDPACell {
        let (q, k, v) = makeQKV(headDim: headDim, queryLength: queryLength, keyLength: keyLength)
        let scale = 1.0 / Float(headDim).squareRoot()
        let transient = measureTransientBytes(query: q, keys: k, values: v, scale: scale)
        let scoreElementCount = Double(queryHeads) * Double(queryLength) * Double(keyLength)
        let bytesPerScoreElement = Double(transient) / scoreElementCount
        print(
            "SDPA-TRANSIENT | headDim=\(headDim) qL=\(queryLength) kL=\(keyLength) "
                + "transient=\(transient) B perScoreElem=\(bytesPerScoreElement)"
        )
        return SDPACell(
            headDim: headDim, queryLength: queryLength, keyLength: keyLength,
            transientBytes: transient, bytesPerScoreElement: bytesPerScoreElement)
    }

    @Test
    func testSDPAHeadDim256PrefillTransientBytesPerScoreElement() throws {
        let queryLengths = [512, 1_024, 2_048]

        var cellsByHeadDim: [Int: [SDPACell]] = [:]
        for headDim in [128, 256] {
            var cells: [SDPACell] = []
            for queryLength in queryLengths {
                cells.append(
                    Self.measureCell(
                        headDim: headDim, queryLength: queryLength, keyLength: Self.keyLength))
            }
            cellsByHeadDim[headDim] = cells
        }

        let cells128 = try #require(cellsByHeadDim[128])
        let cells256 = try #require(cellsByHeadDim[256])
        try #require(cells128.count == 3 && cells256.count == 3)

        // --- Assertion 1: ANTI-VACUITY CONTROL ---
        // head_dim 128 is gated to mlx-core's FUSED full-attention path
        // (`scaled_dot_product_attention.cpp:623-625`), which should not
        // materialize an O(qL*kL) score tensor, so its measured
        // bytes-per-score-element should be small and roughly CONSTANT
        // (near zero, not scaling with geometry). head_dim 256 falls
        // outside that gate and is expected to materialize the score
        // tensor, so its bytes-per-score-element should be a real,
        // non-negligible constant. If the two head dims produce
        // indistinguishable bytes-per-score-element profiles, this
        // instrument is measuring nothing (e.g. a bound backend, a
        // constant-folded call, or a broken transient calculation) and the
        // test MUST fail loudly rather than silently pass. This assertion
        // is unconditional -- it always runs, never behind a feature flag
        // or skip.
        let meanPerElem128 = cells128.map(\.bytesPerScoreElement).reduce(0, +) / 3
        let meanPerElem256 = cells256.map(\.bytesPerScoreElement).reduce(0, +) / 3
        let controlRatio = meanPerElem256 / max(meanPerElem128, 1e-12)
        let controlMessage =
            "ANTI-VACUITY CONTROL FIRED: headDim=128 (fused path, mean=\(meanPerElem128) B/elem) "
            + "and headDim=256 (mean=\(meanPerElem256) B/elem) produced indistinguishable "
            + "bytes-per-score-element profiles (ratio=\(controlRatio)x). This instrument is "
            + "measuring nothing -- the head_dim-256 result below cannot be trusted until this "
            + "control actually discriminates the two dispatch paths."
        #expect(controlRatio > 10, "\(controlMessage)")

        // --- Assertion 2: scaling discrimination for head_dim 256 ---
        // If head_dim 256 materializes the full [queryHeads, qL, kL] score
        // tensor, transient bytes grow LINEARLY in queryLength at fixed
        // keyLength (score tensor element count == queryHeads * qL * kL,
        // linear in qL), so bytes-per-score-element should be roughly
        // CONSTANT across the three queryLengths. Tolerance is 35%:
        // generous enough to absorb allocator rounding, MLX's buffer-pool
        // reuse behavior, and any small O(qL) (not O(qL*kL)) terms SDPA
        // also allocates (e.g. an O(qL) softmax denominator/rowmax
        // buffer), which shift the ratio more at the smallest queryLength
        // (512) where those lower-order terms are a larger fraction of the
        // dominant O(qL*kL) term -- while still being tight enough that a
        // non-linear (e.g. quadratic-in-qL, which would be a distinct and
        // more alarming defect) growth pattern would violate it.
        let perElem256 = cells256.map(\.bytesPerScoreElement)
        let minPerElem256 = try #require(perElem256.min())
        let maxPerElem256 = try #require(perElem256.max())
        #expect(minPerElem256 > 0, "expected a non-zero measured transient at headDim=256")
        let spread = (maxPerElem256 - minPerElem256) / minPerElem256
        let spreadMessage =
            "headDim=256 bytes-per-score-element was not roughly constant across queryLength "
            + "in \(queryLengths): values=\(perElem256), spread=\(spread) exceeds 0.35 "
            + "tolerance -- transient does not scale linearly in queryLength as expected for "
            + "a materialized [queryHeads, qL, kL] score tensor"
        #expect(spread <= 0.35, "\(spreadMessage)")
    }
}
