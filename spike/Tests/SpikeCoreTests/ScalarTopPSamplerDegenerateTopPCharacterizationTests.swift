import MLX
import MLXLMCommon
import XCTest

/// Empirical characterization of `TopPSampler.sample(logits:)` (public,
/// `spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`) at
/// pathologically small `topP` values, over a production-scale vocabulary.
///
/// This is the SCALAR (non-speculative) sampling path: it is what
/// `GenerateParameters.sampler()` returns whenever `temperature != 0`
/// (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` around
/// line 165) and is what `MLXDecoder` calls once per decode step in normal,
/// non-speculative serving (`spike/Sources/SpikeCore/MLXDecoder.swift:113,134`).
/// `SampledMTPDegenerateTopPCharacterizationTests.swift` already established
/// that the SPECULATIVE arm's `truncatedSamplingProbabilities` helper goes
/// NaN / zero-support at this exact `(topP, V)` combination and that the
/// speculative bridge fails closed (`return []`) on that NaN. This suite asks
/// the DIFFERENT question that matters for currently-served traffic: does the
/// scalar path fail closed too, or does it silently hand back a token from a
/// degenerate/undefined distribution?
///
/// ## Reading `TopPSampler.sample(logits:)`
///
/// (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:257-279`)
///
/// 1. Casts `bfloat16` logits to `float32` (no-op here; the fixture is
///    already `float32`).
/// 2. `logSoftmax(logits)` — converts raw logits to log-probabilities.
/// 3. Filters are applied in Python-`mlx_lm`-compatible order: top-p, then
///    min-p, then top-k. At the deployed preset (`topK: 0, minP: 0`) only
///    top-p is active; `applyTopPFilter` (lines 284-293) sorts log-probs
///    ascending, exponentiates, takes a running `cumsum`, and keeps only the
///    entries whose cumulative probability exceeds `1 - topP`, scattering
///    everything else to `-Float.infinity` via `MLX.where`. This is exactly
///    the same nucleus construction the speculative helper's degenerate row
///    was built from, so the SAME float32 rounding collapse applies: once
///    `1 - topP` rounds indistinguishably from `1.0` at this vocabulary
///    width, `cumulativeProbs .> (1 - topP)` is false everywhere and the
///    ENTIRE row becomes `-infinity`.
/// 4. Unlike the speculative helper, there is no explicit `isFinite`/`> 0`
///    guard afterward. The (possibly all-`-infinity`) log-probability row is
///    fed straight into `categorical(logprobs * (1 / temp))` — an MLX
///    Gumbel-max categorical draw (add Gumbel noise to each log-prob, return
///    the `argmax` index) — and whatever `categorical` returns for an
///    all-`-infinity` row becomes the token that the scalar decode step
///    emits, unfiltered and unchecked, to the caller. This suite measures
///    what that draw actually is.
///
/// Every assertion below pins an OBSERVED outcome from the printed
/// `[scalar-topp]` diagnostic lines this suite emits, not a pre-run guess.
final class ScalarTopPSamplerDegenerateTopPCharacterizationTests: XCTestCase {

    /// Production decoder vocabulary width used by the deployed model family
    /// (matches `SampledMTPDegenerateTopPCharacterizationTests`).
    private static let productionVocabularySize = 151_936

    /// The deployed truncation preset (temperature 1, topP 0.95, topK 0,
    /// minP 0) — same preset `SampledMTPDegenerateTopPCharacterizationTests`
    /// uses as its anti-vacuity control.
    private static let deployedTopP: Float = 0.95

    /// A fixed seed used to construct a FRESH `TopPSampler` per trial.
    /// `TopPSampler` owns its own `MLXRandom.RandomState`
    /// (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:237,
    /// 254`), seeded from `MLXRandom.key(seed)` at `init` time and advanced
    /// (`RandomState.next()`, `spike/.build/checkouts/mlx-swift/Source/MLX/
    /// State.swift:50-55`) once per `categorical` draw *inside that
    /// instance's task-local scope* — it does NOT consult or get seeded by
    /// the process-global `MLXRandom.seed(_:)`. So, unlike a plain global
    /// seed call, reproducibility here requires constructing a NEW
    /// `TopPSampler(seed: fixedSeed)` for every trial (each instance starts
    /// its `RandomState` fresh at the same key and this suite draws exactly
    /// once per instance); reusing one instance across repeated calls would
    /// legitimately advance its state and change the draw each time. This
    /// matches how `GenerateParameters.seed` is documented to make
    /// `(seed, prompt, parameters)` reproducible in production
    /// (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:93-96`).
    private static let fixedSeed: UInt64 = 42

    // MARK: - Fixture construction

    /// Same deterministic, monotonically decaying logits row as
    /// `SampledMTPDegenerateTopPCharacterizationTests.decayingLogitsRow`:
    /// `logit[i] = -Float(i) * decayStep`, index 0 is the single
    /// highest-logit token, no randomness.
    private func decayingLogitsRow(vocabularySize: Int, decayStep: Float = 0.01) -> MLXArray {
        precondition(vocabularySize > 0)
        let values = (0 ..< vocabularySize).map { -Float($0) * decayStep }
        return MLXArray(values).reshaped([1, vocabularySize])
    }

    /// PERMUTED variant of `decayingLogitsRow`: same decaying magnitudes,
    /// but centered on `peakIndex` instead of index 0 —
    /// `logit[i] = -Float((i - peakIndex + V) % V) * decayStep`. The unique
    /// maximum (distance 0) sits at `peakIndex`, and probability decays away
    /// from it circularly in both directions. Used to discriminate "the
    /// sampler tracked the real peak" from "the sampler fell back to index
    /// 0 regardless of where the peak actually is" — a confound the
    /// unpermuted fixture (whose peak always happens to sit at index 0)
    /// cannot rule out on its own.
    private func permutedDecayingLogitsRow(
        vocabularySize: Int,
        peakIndex: Int,
        decayStep: Float = 0.01
    ) -> MLXArray {
        precondition(vocabularySize > 0)
        precondition((0 ..< vocabularySize).contains(peakIndex))
        let values = (0 ..< vocabularySize).map { i -> Float in
            let distance = (i - peakIndex + vocabularySize) % vocabularySize
            return -Float(distance) * decayStep
        }
        return MLXArray(values).reshaped([1, vocabularySize])
    }

    /// Builds a fresh `TopPSampler` seeded with `Self.fixedSeed` (temperature
    /// 1, topK 0, minP 0 — the deployed preset's fixed knobs, varying only
    /// `topP`) and draws exactly one token from the given pre-built logits
    /// row. Returns the token id as `Int` and whether it falls in the valid
    /// `0..<vocabularySize` range.
    private func sampledToken(
        logits: MLXArray,
        vocabularySize: Int,
        topP: Float
    ) -> (tokenId: Int, inRange: Bool) {
        let sampler = TopPSampler(temperature: 1, topP: topP, topK: 0, minP: 0, seed: Self.fixedSeed)
        let result = sampler.sample(logits: logits)
        eval(result)
        let tokenId = result.item(Int.self)
        return (tokenId, (0 ..< vocabularySize).contains(tokenId))
    }

    /// Same as above, but builds the unpermuted decaying fixture internally
    /// — kept for the existing (index-0-peaked) test methods below.
    private func sampledToken(
        vocabularySize: Int,
        topP: Float,
        decayStep: Float = 0.01
    ) -> (tokenId: Int, inRange: Bool) {
        let logits = decayingLogitsRow(vocabularySize: vocabularySize, decayStep: decayStep)
        return sampledToken(logits: logits, vocabularySize: vocabularySize, topP: topP)
    }

    // MARK: - Q1 & Q2: valid token id, or crash, or silent garbage?

    /// At the exact `(topP, V)` combination that made the speculative arm's
    /// helper go NaN (`topP = 1e-9`, `V = 151936`), does the scalar sampler
    /// return a valid, deterministic token id, or something else?
    ///
    /// On Q2 ("does it crash"): the fact that this test method — and every
    /// other method in this file, all exercising the identical degenerate
    /// `(topP=1e-9, V=151936)` combination — runs to completion and reports
    /// a PASS/FAIL verdict rather than aborting the process IS the answer:
    /// `TopPSampler.sample` does not fatal-error or hang on this input. It
    /// returns SILENTLY. (Per the task's crash-safety guidance, no assertion
    /// is written that presumes a crash either way; this is a factual
    /// observation about the run, not a pinned expectation.)
    ///
    /// Measured on 2026-09-09 against this exact fixture and seed: all three
    /// trials return token id `0` (see the `[scalar-topp] degenerate ...`
    /// line this test prints). `0` is index 0 of the decaying fixture — the
    /// single highest-logit ("most likely") token — which LOOKS sensible,
    /// but is almost certainly coincidental rather than a real nucleus draw:
    /// per `applyTopPFilter`'s construction (see the file-level doc comment
    /// above) the entire log-probability row collapses to `-infinity` at
    /// this `(topP, V)`, exactly as `SampledMTPDegenerateTopPCharacterizationTests`
    /// measured for the shared helper. `categorical` on an all-`-infinity`
    /// row is an `argmax` over a fully-tied (or NaN, if Gumbel-max ever
    /// produces `-inf + +inf`) input with no real signal left to distinguish
    /// indices; index `0` winning is consistent with a first-index tie
    /// fallback, not with "the sampler correctly identified token 0 as most
    /// likely." Reordering the fixture so a DIFFERENT index held the highest
    /// logit would be needed to fully rule out signal survival, but the
    /// key operational fact stands either way: the scalar path does not
    /// detect the degenerate row and does not refuse — it emits a token,
    /// deterministically, silently, from an internally undefined
    /// distribution.
    func testDegenerateTopPReturnsValidDeterministicTokenId() {
        let vocabularySize = Self.productionVocabularySize
        let degenerateTopP: Float = 1e-9

        let trial1 = sampledToken(vocabularySize: vocabularySize, topP: degenerateTopP)
        let trial2 = sampledToken(vocabularySize: vocabularySize, topP: degenerateTopP)
        let trial3 = sampledToken(vocabularySize: vocabularySize, topP: degenerateTopP)

        print(
            "[scalar-topp] degenerate V=\(vocabularySize) topP=\(degenerateTopP) "
                + "trial1=\(trial1.tokenId) trial2=\(trial2.tokenId) trial3=\(trial3.tokenId) "
                + "inRange=\(trial1.inRange)")

        // Q1: valid index.
        XCTAssertTrue(trial1.inRange)
        // Deterministic across repeated (freshly-seeded) trials.
        XCTAssertEqual(trial1.tokenId, trial2.tokenId)
        XCTAssertEqual(trial1.tokenId, trial3.tokenId)
    }

    // MARK: - Q3: anti-vacuity control at the deployed preset

    /// Proves the harness is capable of producing a healthy, sensible
    /// (low-index, high-probability) draw at the DEPLOYED preset (`topP =
    /// 0.95`) before trusting any reading at the degenerate preset. Over the
    /// monotonically decaying fixture, a healthy nucleus draw should land
    /// close to index 0 (the highest-probability token), not scattered
    /// across the full `151936`-wide vocabulary.
    ///
    /// Measured on 2026-09-09: token id `125` (see the
    /// `[scalar-topp] deployed ...` line this test prints) — well within the
    /// nucleus, confirming the harness and fixture are capable of producing
    /// a real, non-degenerate categorical draw when the input is healthy.
    func testDeployedPresetReturnsLowIndexToken() {
        let vocabularySize = Self.productionVocabularySize
        let (tokenId, inRange) = sampledToken(vocabularySize: vocabularySize, topP: Self.deployedTopP)

        print(
            "[scalar-topp] deployed V=\(vocabularySize) topP=\(Self.deployedTopP) "
                + "token=\(tokenId) inRange=\(inRange)")

        XCTAssertTrue(inRange)
        // Anti-vacuity: the deployed preset must draw from the peaked head
        // of the distribution, not an arbitrary/garbage index. 500 is a
        // generous bound over a V=151936 vocabulary decaying at 0.01/index
        // (nucleus mass at topP=0.95 is concentrated far below that).
        XCTAssertLessThan(tokenId, 500)
    }

    // MARK: - Q4: sweep, crossover visibility

    /// Sweeps `topP` from healthy to maximally degenerate at production
    /// width and prints the returned token id at every point, so the
    /// crossover in behavior (if any) between the healthy and degenerate
    /// regimes is visible in the test log rather than just asserted.
    ///
    /// Measured on 2026-09-09, BEFORE the `applyTopPFilter` `min_tokens_to_keep=1`
    /// fix (see the `[scalar-topp] sweep ...` lines this test prints):
    ///
    /// | topP                    | token id |
    /// |--------------------------|---------:|
    /// | 1.0                      |      322 |
    /// | 0.95 (deployed)          |      125 |
    /// | 1e-7 (usable per the shared-helper boundary) | 0 |
    /// | 1e-9 (NaN per the shared-helper boundary)     | 0 |
    /// | `Float.leastNormalMagnitude` (NaN per the shared-helper boundary) | 0 |
    ///
    /// There is no visible crash, hang, or out-of-range index anywhere in
    /// the sweep — every point returns a valid, in-range token id.
    ///
    /// CORRECTION (superseding an earlier reading of this same table): this
    /// suite originally read `1e-7` returning `0` here as evidence the
    /// scalar path's own degeneracy crossover sits EARLIER (in `topP` terms)
    /// than the shared helper's NaN boundary. `testPermutedFixtureDistinguishesTieBreakFromRealPeakTracking`
    /// below has since RULED THAT OUT directly: run on a fixture whose true
    /// peak is NOT at index 0, `topP=1e-7` returns the TRUE peak, exactly
    /// matching the shared helper's own measurement that support size is 1
    /// (a single legitimately-surviving token) at that exact `(topP, V)`.
    /// So `1e-7` returning `0` here was never a sign of degeneracy — this
    /// UNPERMUTED fixture's true peak simply also sits at index 0, so a
    /// correct narrow-nucleus draw and a Gumbel-max tie-break fallback are
    /// indistinguishable from the returned token id alone at THIS topP. The
    /// scalar path's degeneracy crossover coincided with the shared helper's
    /// NaN boundary (between `1e-7` and `1e-9`), not before it. `1e-9` and
    /// `Float.leastNormalMagnitude` were the confirmed-degenerate scalar-path
    /// reads (see the permuted test's decisive case) BEFORE the fix below.
    ///
    /// POST-FIX (this suite's decisive pin is
    /// `testPermutedFixtureDistinguishesTieBreakFromRealPeakTracking`, not
    /// this table): `applyTopPFilter` now force-keeps the row's argmax
    /// (HF `min_tokens_to_keep=1`), so the `1e-9` /
    /// `Float.leastNormalMagnitude` rows no longer collapse to an
    /// all-`-infinity` Gumbel-max tie-break. This unpermuted, index-0-peaked
    /// fixture cannot distinguish "correctly kept the peak" from "still
    /// tie-breaking to 0" on its own — this table is left as the historical
    /// pre-fix measurement rather than re-asserted post-fix; the permuted
    /// test below is the one pinned against the fix.
    func testProductionWidthTopPSweepReturnedTokenIds() {
        let vocabularySize = Self.productionVocabularySize
        let sweep: [Float] = [1.0, 0.95, 1e-7, 1e-9, Float.leastNormalMagnitude]

        var observed: [Float: Int] = [:]
        for topP in sweep {
            let (tokenId, inRange) = sampledToken(vocabularySize: vocabularySize, topP: topP)
            observed[topP] = tokenId
            print(
                "[scalar-topp] sweep V=\(vocabularySize) topP=\(topP) token=\(tokenId) inRange=\(inRange)")
            XCTAssertTrue(inRange)
        }

        XCTAssertEqual(observed.count, sweep.count)
    }

    // MARK: - Q5: closing the tie-break-vs-real-draw confound

    /// Every test above uses a fixture whose true peak sits at index `0` —
    /// which meant the degenerate-topP reading ("returns token `0`") could
    /// not distinguish "correctly identified the peak" from "fell back to
    /// the Gumbel-max tie-break index for an all-`-infinity` row, which
    /// happens to also be `0`". This test closes that confound with a
    /// PERMUTED fixture whose true peak sits at `peakIndex = 12345`
    /// (`permutedDecayingLogitsRow`), so a tie-break-to-index-0 fallback and
    /// a real peak-tracking draw are now visibly different outcomes.
    func testPermutedFixtureDistinguishesTieBreakFromRealPeakTracking() {
        let vocabularySize = Self.productionVocabularySize
        let peakIndex = 12_345
        let logits = permutedDecayingLogitsRow(vocabularySize: vocabularySize, peakIndex: peakIndex)

        // Anti-vacuity: confirm the fixture's true argmax really is
        // `peakIndex`, not `0`. If a later edit to `permutedDecayingLogitsRow`
        // broke the permutation (e.g. silently reverted to an index-0 peak),
        // this assertion fails loudly instead of the rest of the test
        // quietly re-measuring the unpermuted case under a different name.
        let trueArgmax = argMax(logits, axis: -1)
        eval(trueArgmax)
        let trueArgmaxIndex = trueArgmax.item(Int.self)
        print("[scalar-topp] permuted fixture trueArgmax=\(trueArgmaxIndex) expectedPeak=\(peakIndex)")
        XCTAssertEqual(trueArgmaxIndex, peakIndex)

        // Case 1 — healthy control (topP=0.95): must land near the REAL
        // peak and must NOT be 0. If this fails, the permutation is not
        // actually being honored by the sampler and nothing below can be
        // trusted either.
        let healthy = sampledToken(logits: logits, vocabularySize: vocabularySize, topP: 0.95)
        print("[scalar-topp] permuted healthy peakIndex=\(peakIndex) topP=0.95 token=\(healthy.tokenId)")
        XCTAssertTrue(healthy.inRange)
        XCTAssertNotEqual(healthy.tokenId, 0)
        XCTAssertLessThan(abs(healthy.tokenId - peakIndex), 500)

        // Case 2 — legitimately-narrow control (topP=1e-7): the shared
        // helper (`SampledMTPDegenerateTopPCharacterizationTests`) measured
        // support size 1 at this exact (topP, V), so the single correct
        // answer is `peakIndex` itself, not a tie-break artifact.
        let narrow = sampledToken(logits: logits, vocabularySize: vocabularySize, topP: 1e-7)
        print("[scalar-topp] permuted narrow peakIndex=\(peakIndex) topP=1e-7 token=\(narrow.tokenId)")
        XCTAssertTrue(narrow.inRange)
        XCTAssertEqual(narrow.tokenId, peakIndex)

        // Case 3 — the decisive case (topP=1e-9), and the PRIMARY ACCEPTANCE
        // TEST for the `applyTopPFilter` `min_tokens_to_keep=1` fix.
        //
        // BEFORE the fix (measured 2026-09-09, root-cause characterization):
        // token id 0, NOT peakIndex. That confirmed the earlier token-0
        // reading was a Gumbel-max tie-break artifact of an all-`-infinity`
        // row, independent of where the true peak sits — a genuinely wrong,
        // silently-returned token. `applyTopPFilter` masked the ENTIRE row
        // to `-infinity` at this `(topP, V)` because `1 - topP` rounds to
        // exactly `1.0f` in float32 and the cumulative-probability test was
        // false everywhere, including at the row's true argmax.
        //
        // AFTER the fix: `applyTopPFilter` force-keeps the row's argmax
        // (HF `TopPLogitsWarper`'s `min_tokens_to_keep=1` floor) whenever the
        // cumulative-mass test would otherwise mask every position. The
        // permuted fixture's true peak sits at `peakIndex`, so the fixed
        // sampler must now return `peakIndex`, not `0` — this is the exact
        // discriminator that proved the defect, inverted to prove the
        // repair: a sampler still tie-breaking to a fixed index-0 fallback
        // would fail this assertion just as visibly as it failed before the
        // fix confirmed the bug.
        let degenerate = sampledToken(logits: logits, vocabularySize: vocabularySize, topP: 1e-9)
        print("[scalar-topp] permuted degenerate peakIndex=\(peakIndex) topP=1e-9 token=\(degenerate.tokenId)")
        XCTAssertTrue(degenerate.inRange)
        XCTAssertEqual(degenerate.tokenId, peakIndex)
        XCTAssertNotEqual(degenerate.tokenId, 0)
    }
}
