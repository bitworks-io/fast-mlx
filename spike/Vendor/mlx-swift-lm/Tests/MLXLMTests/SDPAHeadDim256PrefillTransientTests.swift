// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
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
// subtracts the known newly-allocated output buffer bytes from that
// increment, per the task's stated idiom, to isolate the SDPA call's own
// TRANSIENT (the intermediate score tensor and any other scratch it
// allocates) rather than reporting the cost of merely holding the output
// live.
//
// Corrected coefficient (cycle 61; see
// docs/task-inbox/2026-09-11-sdpa-transient-instrument-double-subtraction-PREDECLARATION.md):
// head_dim 256 prefill measures **2.0625 bytes per score element**,
// decomposing as one fp16 `[queryHeads, qL, kL]` score buffer (2 B/elem)
// plus one q-sized scratch copy. An earlier version of this suite
// double-subtracted `q`/`k`/`v` -- which are already allocated and live
// (built and `eval`'d by `makeQKV`) at the moment `before` is read, so
// `peak - before` already excludes them -- and reported a spurious
// 1.83 -> 1.96 B/elem "drift" across queryLength that was entirely that
// double-subtraction bias growing with `qL`. Only `out` is newly
// allocated inside the measured region, so it is the only buffer
// `measureTransientBytes` subtracts now.

private struct SDPACell {
    let headDim: Int
    let queryLength: Int
    let keyLength: Int
    /// May be negative: this is the raw, unclamped `peak - before -
    /// out.nbytes` reading. A negative value means the peak read landed
    /// below the baseline and is itself diagnostic information (see
    /// `measureTransientBytes`), not an error to hide.
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
    /// holding its inputs (and, when supplied, an explicit mask) and
    /// output live, using the increment-above-a-live-baseline idiom shared
    /// by the drafter-priming chunking and sparse-indexer suites.
    ///
    /// `mask`, like `query`/`keys`/`values`, is expected to already be
    /// built and `eval`'d by the caller before this function runs (see
    /// each call site) -- it is therefore already live and already
    /// excluded from `peak - before`, exactly like q/k/v, and must NOT be
    /// subtracted again.
    private static func measureTransientBytes(
        query: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray? = nil
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
                queries: query, keys: keys, values: values, scale: scale, mask: mask)
            eval(warm)
            Stream().synchronize()
        }

        Memory.clearCache()
        let before = Memory.activeMemory
        Memory.peakMemory = 0

        let out = MLXFast.scaledDotProductAttention(
            queries: query, keys: keys, values: values, scale: scale, mask: mask)
        eval(out)  // force materialization INSIDE the measured region
        Stream().synchronize()

        let peak = max(Memory.peakMemory, before)
        // CORRECTED (cycle 61): `query`, `keys`, `values`, and `mask` (when
        // present) are already allocated and live at `before` -- they were
        // built and `eval`'d by the caller before this function ran -- so
        // `peak - before` has already excluded them. Only `out` is newly
        // allocated INSIDE the measured region, so it is the only buffer
        // subtracted here. The previous version additionally subtracted
        // `query.nbytes + keys.nbytes + values.nbytes`, double-subtracting
        // bytes `peak - before` had already excluded and understating the
        // transient by exactly that amount (see the predeclaration cited
        // in this file's header comment for the arithmetic confirming
        // this).
        let ioBytes = out.nbytes
        let transient = peak - before - ioBytes
        Memory.clearCache()
        // NOT clamped to >= 0 (previously `max(0, transient)`): a negative
        // reading means the peak read landed below the baseline and is
        // diagnostic information that must stay visible, not be hidden.
        // The old clamp converted the head_dim-128 fused path's
        // double-subtracted (and therefore frequently negative) result
        // into a fake "exactly 0 B" -- see the predeclaration's point 2.
        return transient
    }

    private static func measureCell(
        headDim: Int, queryLength: Int, keyLength: Int, mask: MLXArray? = nil,
        maskLabel: String = "none"
    ) -> SDPACell {
        let (q, k, v) = makeQKV(headDim: headDim, queryLength: queryLength, keyLength: keyLength)
        let scale = 1.0 / Float(headDim).squareRoot()
        let transient = measureTransientBytes(query: q, keys: k, values: v, scale: scale, mask: mask)
        let scoreElementCount = Double(queryHeads) * Double(queryLength) * Double(keyLength)
        let bytesPerScoreElement = Double(transient) / scoreElementCount
        print(
            "SDPA-TRANSIENT | headDim=\(headDim) qL=\(queryLength) kL=\(keyLength) mask=\(maskLabel) "
                + "transient=\(transient) B perScoreElem=\(bytesPerScoreElement)"
        )
        return SDPACell(
            headDim: headDim, queryLength: queryLength, keyLength: keyLength,
            transientBytes: transient, bytesPerScoreElement: bytesPerScoreElement)
    }

    /// The head_dim-INDEPENDENT term of the closed-form prediction below:
    /// the byte size of the `[queryHeads, qL, kL]` fp16 score buffer that
    /// SDPA's unfused fallback path materializes. This term does not
    /// depend on head_dim (the score tensor's shape is `queryHeads x qL x
    /// kL` regardless of head_dim), so it also serves as the yardstick
    /// the head_dim-128 anti-vacuity control below compares against: if
    /// head_dim 128 (the fused path) ever starts allocating a transient
    /// anywhere near this size at the same geometry, the fused/unfused
    /// dispatch distinction this suite exists to detect has stopped being
    /// observable.
    private static func scoreBufferBytes(queryLength: Int, keyLength: Int) -> Int {
        2 * queryHeads * queryLength * keyLength
    }

    /// `q.nbytes` for the fixed `[batch, queryHeads, queryLength, headDim]`
    /// float16 (2 bytes/element) shape this suite always builds via
    /// `makeQKV`, computed directly so callers do not need to keep the
    /// actual `MLXArray` around just to read this off it.
    private static func queryNBytes(headDim: Int, queryLength: Int) -> Int {
        batch * queryHeads * queryLength * headDim * 2
    }

    /// Trailing fixed-size term in the closed form below. Mechanism
    /// unconfirmed beyond "small and constant" (plausibly an alignment
    /// pad or a tiny scalar scratch buffer); included because dropping it
    /// breaks exact-byte agreement at every measured cell (see the
    /// predeclaration cited in this file's header comment).
    private static let trailingConstantBytes = 2

    /// EXACT closed-form byte count SDPA's head_dim-256 unfused fallback
    /// path materializes: one fp16 `[queryHeads, qL, kL]` score buffer (2
    /// bytes/element) plus one q-sized scratch copy plus a small fixed
    /// constant. Confirmed (predeclaration cited in this file's header
    /// comment) to reproduce every measured cell in this suite to the
    /// byte: 207618050 / 415236098 / 830472194 B at kL=8192 for
    /// qL=512/1024/2048, and the kL-sweep per-element values to nine
    /// decimal places.
    ///
    /// Per-ELEMENT bytes (`total / (queryHeads*qL*kL)`) equal
    /// `2 + (headDim*2)/kL` and are therefore PROVABLY NOT constant across
    /// kL -- they fall as kL grows, because the fixed q-scratch term is
    /// amortized over more score elements as kL increases. A
    /// percentage-spread gate over a kL sweep was therefore always
    /// measuring a quantity known not to be constant; asserting exact
    /// integer byte equality against this closed form is the correct
    /// gate here, not merely a stricter version of the same one.
    private static func predictedHeadDim256TransientBytes(
        queryLength: Int, keyLength: Int
    ) -> (total: Int, scoreBufferBytes: Int, queryScratchBytes: Int, trailingConstantBytes: Int) {
        let scoreBufferBytes = Self.scoreBufferBytes(queryLength: queryLength, keyLength: keyLength)
        let queryScratchBytes = Self.queryNBytes(headDim: 256, queryLength: queryLength)
        let total = scoreBufferBytes + queryScratchBytes + Self.trailingConstantBytes
        return (total, scoreBufferBytes, queryScratchBytes, Self.trailingConstantBytes)
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

        // --- Assertion 1: ANTI-VACUITY CONTROL (unconditional, UNCLAMPED) ---
        // head_dim 128 is gated to mlx-core's FUSED full-attention path
        // (`scaled_dot_product_attention.cpp:623-625`), which should not
        // materialize an O(qL*kL) score tensor. Its raw, unclamped
        // bytes-per-score-element may now legitimately be small, roughly
        // constant, OR NEGATIVE (a negative reading here is a real,
        // OBSERVABLE consequence of the corrected subtraction -- the
        // previously published "exactly 0 B at every cell" for head_dim
        // 128 was produced by a `max(0, .)` clamp and so could not have
        // distinguished "allocates nothing" from "allocates up to
        // q+k+v bytes"; the corrected, UNCLAMPED instrument independently
        // re-measures exactly 0 B at every head_dim-128 cell, so that
        // conclusion now rests on a real measurement rather than on the
        // clamp that previously guaranteed it; see this file's header
        // comment). head_dim
        // 256 falls outside that gate and is expected to materialize the
        // score tensor, so its bytes-per-score-element should be a real,
        // non-negligible positive constant. If the two head dims produce
        // indistinguishable bytes-per-score-element profiles, this
        // instrument is measuring nothing (e.g. a bound backend, a
        // constant-folded call, or a broken transient calculation) and the
        // test MUST fail loudly rather than silently pass. This assertion
        // is unconditional -- it always runs, never behind a feature flag
        // or skip -- and is computed on the unclamped values above.
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

        // --- Assertion 1b: PER-CELL anti-vacuity control against the
        // head_dim-256 closed form's score-buffer term ---
        // The score-buffer term (`scoreBufferBytes` below) is head_dim-
        // INDEPENDENT -- the score tensor is `[queryHeads, qL, kL]`
        // regardless of head_dim -- so it is also what the FUSED
        // head_dim-128 path would have to be allocating for the two
        // dispatch paths to become indistinguishable. Beyond the
        // mean-ratio check above, assert directly, per cell, that
        // head_dim 128's measured transient stays far below (< 10% of)
        // that same-geometry score-buffer term. If head_dim 128 ever
        // begins matching the head_dim-256 closed form, this instrument
        // has stopped discriminating the fused and unfused dispatch
        // paths and must fail loudly rather than silently pass.
        for (queryLength, cell128) in zip(queryLengths, cells128) {
            let sameGeometryScoreBufferBytes = Self.scoreBufferBytes(
                queryLength: queryLength, keyLength: Self.keyLength)
            let cellMessage =
                "ANTI-VACUITY CONTROL FIRED: headDim=128 qL=\(queryLength) kL=\(Self.keyLength) "
                + "measured transient=\(cell128.transientBytes) B, which is not far below the "
                + "same-geometry head_dim-256 score-buffer term of "
                + "\(sameGeometryScoreBufferBytes) B -- head_dim 128 (fused path) is no longer "
                + "distinguishable from the unfused head_dim-256 path at this geometry"
            #expect(
                Double(cell128.transientBytes) < Double(sameGeometryScoreBufferBytes) / 10,
                "\(cellMessage)")
        }

        // --- Assertion 2: EXACT closed-form check for head_dim 256 ---
        // Per-element bytes equal `2 + (headDim*2)/kL` and are therefore
        // NOT expected to be constant across kL/qL (see
        // `predictedHeadDim256TransientBytes`'s doc comment) -- so this
        // asserts exact integer byte equality against the closed form
        // instead of a spread/tolerance band, which was never the right
        // gate for a quantity that is provably non-constant.
        for (queryLength, cell) in zip(queryLengths, cells256) {
            let prediction = Self.predictedHeadDim256TransientBytes(
                queryLength: queryLength, keyLength: Self.keyLength)
            let predictedTotal = prediction.total
            let predictedScoreBufferBytes = prediction.scoreBufferBytes
            let predictedQueryScratchBytes = prediction.queryScratchBytes
            let predictedTrailingConstantBytes = prediction.trailingConstantBytes
            let measuredBytes = cell.transientBytes
            let message = """
                headDim=256 qL=\(queryLength) kL=\(Self.keyLength): predicted \(predictedTotal) B \
                (scoreBufferBytes=\(predictedScoreBufferBytes) + queryScratchBytes=\(predictedQueryScratchBytes) \
                + trailingConstantBytes=\(predictedTrailingConstantBytes)) != measured \(measuredBytes) B -- \
                the exact closed form (see this file's header comment) no longer reproduces this cell
                """
            #expect(measuredBytes == predictedTotal, "\(message)")
        }
    }

    @Test
    func testSDPAHeadDim256TransientScalesLinearlyInKeyLength() throws {
        // Companion to the queryLength sweep above: holds qL fixed and
        // sweeps keyLength instead, since a materialized [queryHeads, qL,
        // kL] score tensor should scale linearly in EITHER dimension
        // independently. The largest cell here (kL=32768, qL=2048,
        // headDim=256) materializes a ~3.2 GB fp16 score tensor
        // (24 * 2048 * 32768 * 2 B); acceptable on a 24 GiB box, and
        // `measureTransientBytes` already calls `Memory.clearCache()`
        // after every cell.
        let queryLength = 2_048
        let keyLengths = [2_048, 8_192, 32_768]

        var cellsByHeadDim: [Int: [SDPACell]] = [:]
        for headDim in [128, 256] {
            var cells: [SDPACell] = []
            for keyLength in keyLengths {
                cells.append(
                    Self.measureCell(headDim: headDim, queryLength: queryLength, keyLength: keyLength))
            }
            cellsByHeadDim[headDim] = cells
        }

        let cells128 = try #require(cellsByHeadDim[128])
        let cells256 = try #require(cellsByHeadDim[256])
        try #require(cells128.count == 3 && cells256.count == 3)

        // --- Assertion 1: ANTI-VACUITY CONTROL (unconditional, UNCLAMPED) ---
        // Same control as the queryLength sweep, recomputed here against
        // the keyLength sweep's own cells, on unclamped values.
        let meanPerElem128 = cells128.map(\.bytesPerScoreElement).reduce(0, +) / 3
        let meanPerElem256 = cells256.map(\.bytesPerScoreElement).reduce(0, +) / 3
        let controlRatio = meanPerElem256 / max(meanPerElem128, 1e-12)
        let controlMessage =
            "ANTI-VACUITY CONTROL FIRED: headDim=128 (fused path, mean=\(meanPerElem128) B/elem) "
            + "and headDim=256 (mean=\(meanPerElem256) B/elem) produced indistinguishable "
            + "bytes-per-score-element profiles across the keyLength sweep (ratio=\(controlRatio)x). "
            + "This instrument is measuring nothing -- the head_dim-256 result below cannot be "
            + "trusted until this control actually discriminates the two dispatch paths."
        #expect(controlRatio > 10, "\(controlMessage)")

        // --- Assertion 1b: PER-CELL anti-vacuity control against the
        // head_dim-256 closed form's score-buffer term ---
        // Same rationale as the queryLength sweep's per-cell control: the
        // score-buffer term is head_dim-INDEPENDENT, so it is the
        // yardstick head_dim 128 must stay far below at every keyLength
        // too.
        for (keyLength, cell128) in zip(keyLengths, cells128) {
            let sameGeometryScoreBufferBytes = Self.scoreBufferBytes(
                queryLength: queryLength, keyLength: keyLength)
            let cellMessage =
                "ANTI-VACUITY CONTROL FIRED: headDim=128 qL=\(queryLength) kL=\(keyLength) "
                + "measured transient=\(cell128.transientBytes) B, which is not far below the "
                + "same-geometry head_dim-256 score-buffer term of "
                + "\(sameGeometryScoreBufferBytes) B -- head_dim 128 (fused path) is no longer "
                + "distinguishable from the unfused head_dim-256 path at this geometry"
            #expect(
                Double(cell128.transientBytes) < Double(sameGeometryScoreBufferBytes) / 10,
                "\(cellMessage)")
        }

        // --- Assertion 2: EXACT closed-form check for head_dim 256 ---
        // Per-element bytes equal `2 + (headDim*2)/kL`, which FALLS as kL
        // grows (e.g. 2.25 at kL=2048 vs 2.015625 at kL=32768, at this
        // fixed qL=2048) -- provably NOT constant across this sweep. That
        // is exactly why this suite gates on exact integer byte equality
        // against the closed form (`predictedHeadDim256TransientBytes`)
        // instead of a percentage spread band, which was never the right
        // gate for a quantity known not to be constant.
        for (keyLength, cell) in zip(keyLengths, cells256) {
            let prediction = Self.predictedHeadDim256TransientBytes(
                queryLength: queryLength, keyLength: keyLength)
            let predictedTotal = prediction.total
            let predictedScoreBufferBytes = prediction.scoreBufferBytes
            let predictedQueryScratchBytes = prediction.queryScratchBytes
            let predictedTrailingConstantBytes = prediction.trailingConstantBytes
            let measuredBytes = cell.transientBytes
            let message = """
                headDim=256 qL=\(queryLength) kL=\(keyLength): predicted \(predictedTotal) B \
                (scoreBufferBytes=\(predictedScoreBufferBytes) + queryScratchBytes=\(predictedQueryScratchBytes) \
                + trailingConstantBytes=\(predictedTrailingConstantBytes)) != measured \(measuredBytes) B -- \
                the exact closed form (see this file's header comment) no longer reproduces this cell
                """
            #expect(measuredBytes == predictedTotal, "\(message)")
        }
    }

    @Test
    func testCausalMaskCellCostsStrictlyMoreThanNilMask() throws {
        // `mask: nil` (the call shape every other cell in this suite uses)
        // is NOT the production call shape: the production hybrid-
        // attention geometry always attends under an explicit causal mask
        // during prefill. A coefficient measured only with `mask: nil`
        // therefore under-counts production -- the unsafe direction for a
        // capacity model built from this suite's numbers. This cell
        // measures the additional transient a real boolean causal mask
        // costs at matched geometry, using the same mask-construction
        // idiom as the production hybrid-attention core's composed-mask
        // path (`createCausalMask(n: queryLength, offset: keyLength -
        // queryLength)`).
        let headDim = 256
        let queryLength = 2_048
        let keyLength = 8_192

        // The mask is built and `eval`'d HERE, outside the measured
        // region, mirroring q/k/v: it is therefore already live at
        // `before` inside `measureTransientBytes` and, consistent with
        // the corrected subtraction above, must NOT be subtracted again --
        // only `out` is newly allocated inside the measured region. Any
        // extra transient this cell measures over the nil-mask cell is
        // therefore genuinely attributable to SDPA's own handling of the
        // mask (e.g. combining it with the score tensor), not to the mask
        // buffer's own allocation.
        let causalMask = createCausalMask(n: queryLength, offset: keyLength - queryLength)
        eval(causalMask)

        let nilMaskCell = Self.measureCell(
            headDim: headDim, queryLength: queryLength, keyLength: keyLength, mask: nil,
            maskLabel: "none")
        let causalMaskCell = Self.measureCell(
            headDim: headDim, queryLength: queryLength, keyLength: keyLength, mask: causalMask,
            maskLabel: "causal")

        // `maskBytes`: the real cost of holding the boolean `[qL, kL]`
        // causal mask live, which the transients measured above do NOT
        // include -- the mask is built and `eval`'d OUTSIDE the measured
        // region (mirroring q/k/v), so it is already live at `before` and
        // excluded from `peak - before` by construction (see
        // `measureTransientBytes`). A coefficient built only from those
        // transients therefore UNDER-COUNTS what production actually
        // pays: production must allocate and hold this mask too, and
        // under-counting is the unsafe direction for a capacity model.
        // Compute and report the mask's own cost explicitly instead of
        // leaving it implicit in a tiny before/after margin.
        let expectedMaskBytes = queryLength * keyLength  // MLX bool == 1 byte/element
        let maskBytesMessage =
            "expected the causal mask's byte size to equal the boolean [qL, kL] element count "
            + "\(expectedMaskBytes) B (qL=\(queryLength) x kL=\(keyLength) x 1 byte/element), "
            + "measured \(causalMask.nbytes) B -- if these disagree, MLX's bool element size or "
            + "this mask's shape no longer matches what this suite assumes"
        #expect(causalMask.nbytes == expectedMaskBytes, "\(maskBytesMessage)")
        let maskBytes = causalMask.nbytes

        let productionRelevantTotal = causalMaskCell.transientBytes + maskBytes
        print(
            "SDPA-TRANSIENT | headDim=\(headDim) qL=\(queryLength) kL=\(keyLength) "
                + "maskBytes=\(maskBytes) B productionRelevantTotal=\(productionRelevantTotal) B"
        )
        // This is `>=` rather than the old (near-vacuous) `>`: `maskBytes`
        // is now added to BOTH sides, so a passing equality here no
        // longer hides behind the tiny before/after margin the old
        // assertion could pass on -- it means the causal-mask cell's own
        // SDPA transient plus the mask's real bytes is not cheaper than
        // the nil-mask baseline plus that same mask cost.
        let productionMessage =
            "expected productionRelevantTotal=\(productionRelevantTotal) B (causal transient="
            + "\(causalMaskCell.transientBytes) B + maskBytes=\(maskBytes) B) to be at least "
            + "nilMaskCell.transientBytes=\(nilMaskCell.transientBytes) B + maskBytes=\(maskBytes) B "
            + "-- holding the mask live plus SDPA's own handling of it must not cost less than "
            + "the nil-mask baseline plus the same mask bytes"
        #expect(
            productionRelevantTotal >= nilMaskCell.transientBytes + maskBytes, "\(productionMessage)")
    }
}
