import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Empirical characterization of `truncatedSamplingProbabilities` (the shared
/// top-p/min-p/top-k sampling-probability helper vended by
/// `MLXLMCommon/Evaluate.swift`, `spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`)
/// at pathologically small `topP` values, over a production-scale
/// vocabulary. `SampledMTPBlockRuntimeBridge.normalizedProbabilities`
/// (private, `spike/Sources/SpikeCore/SampledMTPBlockRuntimeBridge.swift`
/// around line 765) calls this exact function with the same pre-processing
/// (`logits.asType(.float32).reshaped([-1, logits.dim(-1)])`) and treats a
/// non-finite or non-positive row sum as "no usable distribution"
/// (`return []`). These tests call the shared helper DIRECTLY -- it is
/// `public`, so no `@testable` import of `SpikeCore` is needed -- to
/// determine, from real measurement rather than code inspection alone,
/// whether and where that degenerate path is actually reachable.
///
/// This is a MEASUREMENT/CHARACTERIZATION suite: every assertion below pins
/// an observed outcome, not a hypothesized one. Where iteration against the
/// first real run changed an assertion from an initial guess, the comment
/// says so.
final class SampledMTPDegenerateTopPCharacterizationTests: XCTestCase {

    /// Production decoder vocabulary width used by the deployed model family.
    private static let productionVocabularySize = 151_936

    /// The deployed truncation preset (see `SampledMTPSamplingTruncation`
    /// call sites in `SampledMTPBlockRuntimeBridge.swift`): temperature 1,
    /// topP 0.95, topK 0, minP 0.
    private static let deployedTopP: Float = 0.95

    // MARK: - Fixture construction

    /// Builds a deterministic, monotonically decaying logits row of the given
    /// width: `logit[i] = -Float(i) * decayStep`. Index 0 is the single
    /// highest-logit ("most likely") token; probability falls off smoothly
    /// with index rather than spiking to one token or staying perfectly flat,
    /// which is representative of a real decoder head's peaked-but-not-
    /// degenerate output. No randomness is used, so the fixture is exactly
    /// reproducible from the (vocabularySize, decayStep) pair alone.
    private func decayingLogitsRow(vocabularySize: Int, decayStep: Float = 0.01) -> MLXArray {
        precondition(vocabularySize > 0)
        let values = (0 ..< vocabularySize).map { -Float($0) * decayStep }
        return MLXArray(values).reshaped([1, vocabularySize])
    }

    /// Reproduces the bridge's exact pre-processing
    /// (`logits.asType(.float32).reshaped([-1, logits.dim(-1)])`) and calls
    /// the shared helper with the bridge's fixed `temperature: 1, topK: 0,
    /// minP: 0`, varying only `topP`. Returns the row sum computed the same
    /// way the bridge's own guard computes it (`Float` results widened to
    /// `Double` and summed) plus the count of strictly-positive entries
    /// ("support size").
    private func measure(
        vocabularySize: Int,
        topP: Float,
        decayStep: Float = 0.01
    ) -> (sum: Double, supportSize: Int) {
        let logits = decayingLogitsRow(vocabularySize: vocabularySize, decayStep: decayStep)
        let row = logits.asType(.float32).reshaped([-1, logits.dim(-1)])
        let probabilities = truncatedSamplingProbabilities(
            logits: row,
            temperature: 1,
            topP: topP,
            topK: 0,
            minP: 0
        ).flattened()
        eval(probabilities)
        let raw = probabilities.asArray(Float.self)
        let sum = raw.map(Double.init).reduce(0, +)
        let supportSize = raw.reduce(0) { $1 > 0 ? $0 + 1 : $0 }
        return (sum, supportSize)
    }

    private func isUsable(_ sum: Double) -> Bool {
        sum.isFinite && sum > 0
    }

    // MARK: - Pre-fix reference (T3 bit-identity gate helper)

    /// Frozen snapshot of `applyTopPFilter`'s PRE-FIX predicate, written
    /// independently in this test file using the same MLX op sequence
    /// (`argSort` -> `takeAlong` -> `exp` -> `cumsum` -> `where` ->
    /// `putAlong`) the vendored function used before the
    /// `min_tokens_to_keep=1` fix, MINUS the fix's keep-last-sorted-position
    /// OR. This is deliberately NOT a call into the (now fixed) vendored
    /// helper -- the whole point of `testTopPFilterFixIsInertAcrossNonDegenerateRange`
    /// below is to compare the fixed helper's real output against what the
    /// unfixed predicate would have produced, bit for bit, on the exact same
    /// MLX device and kernels.
    private func preFixTopPFiltered(_ logprobs: MLXArray, topP: Float) -> MLXArray {
        let negInf = MLXArray(-Float.infinity)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Frozen snapshot of `truncatedSamplingProbabilities`'s PRE-FIX
    /// behavior at the bridge's fixed `minP: 0, topK: 0` preset (the only
    /// preset the bit-identity gate below exercises), built on
    /// `preFixTopPFiltered` above instead of the live (fixed) filter.
    private func preFixTruncatedProbabilities(
        logits: MLXArray, temperature: Float, topP: Float
    ) -> MLXArray {
        var logprobs = logSoftmax(logits)
        if topP > 0, topP < 1 {
            logprobs = preFixTopPFiltered(logprobs, topP: topP)
        }
        return softmax(logprobs * (1 / temperature), axis: -1)
    }

    /// Counts indices where two equal-length `Float` arrays differ under
    /// exact `==` (never `allClose`/`rtol`/`atol` -- the gate below needs
    /// bit-for-bit identity). Deliberately does NOT use
    /// `XCTAssertEqual(arrayA, arrayB)`: a failing array `==` on a
    /// large collection triggers a diff that dumps both arrays and can wedge
    /// the run. Returns the diff count plus the first differing index (if
    /// any) and its two values, for a compact failure message instead.
    private func countExactDifferences(
        _ a: [Float], _ b: [Float]
    ) -> (diffCount: Int, firstDiffIndex: Int?, firstDiffA: Float?, firstDiffB: Float?) {
        precondition(a.count == b.count)
        var diffCount = 0
        var firstDiffIndex: Int?
        var firstDiffA: Float?
        var firstDiffB: Float?
        for index in a.indices where a[index] != b[index] {
            diffCount += 1
            if firstDiffIndex == nil {
                firstDiffIndex = index
                firstDiffA = a[index]
                firstDiffB = b[index]
            }
        }
        return (diffCount, firstDiffIndex, firstDiffA, firstDiffB)
    }

    // MARK: - Q1 & Q2: did it degenerate, and does the fix close it?
    //
    // NOTE ON THIS SECTION'S HISTORY: every test below in this section
    // originally pinned a degeneracy boundary in `applyTopPFilter`
    // (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`):
    // at production width, `topP <= 1e-9` produced a NaN sum (every entry
    // masked to `-infinity`, i.e. the final softmax computes `0/0`). That
    // was a genuine fail-open defect -- `ScalarTopPSamplerDegenerateTopPCharacterizationTests`
    // showed the SCALAR sampling path built on the same filter does not
    // fail closed on this input, it silently returns index 0 regardless of
    // where the row's true peak sits. `applyTopPFilter` has since been
    // fixed to force-keep the row's single most-probable token whenever the
    // cumulative-mass test would otherwise mask everything (HF
    // `TopPLogitsWarper`'s `min_tokens_to_keep=1` floor). Every assertion
    // below is now inverted from its original (pre-fix) reading to match
    // the fixed, measured behavior -- this suite still pins observed
    // outcomes, not hypothesized ones.

    /// Sweeps `topP` at production width and prints the measured
    /// (sum, finite?, support size) for every point, then asserts the
    /// observed usable/degenerate verdict at each point individually so the
    /// sweep is pinned rather than just eyeballed.
    func testProductionWidthTopPSweepUsableAfterMinTokensToKeepFix() {
        let sweep: [Float] = [1.0, 0.95, 0.5, 0.1, 1e-3, 1e-5, 1e-7, 1e-9, Float.leastNormalMagnitude]

        var observedUsable: [Float: Bool] = [:]
        var observedSupport: [Float: Int] = [:]
        for topP in sweep {
            let (sum, supportSize) = measure(vocabularySize: Self.productionVocabularySize, topP: topP)
            let usable = isUsable(sum)
            observedUsable[topP] = usable
            observedSupport[topP] = supportSize
            print(
                "[sweep] V=\(Self.productionVocabularySize) topP=\(topP) sum=\(sum) "
                    + "finite=\(sum.isFinite) support=\(supportSize) usable=\(usable)")
        }

        // PRE-FIX (measured 2026-09-09, before the `min_tokens_to_keep=1`
        // fix): `topP <= 1e-7` stayed usable (support collapses to a single
        // token, the expected nucleus-sampling behavior at a vanishingly
        // small topP, not a bug); `topP <= 1e-9` produced a NaN sum (every
        // entry masked to -infinity before the final softmax). The boundary
        // sat between 1e-7 and 1e-9.
        //
        // POST-FIX (this pin): `1e-9` and `Float.leastNormalMagnitude` are
        // now ALSO usable, with support forced to exactly 1 (the row's
        // single most-probable token, kept unconditionally). There is no
        // remaining degeneracy boundary anywhere in this swept range.
        XCTAssertEqual(observedUsable[1.0], true)
        XCTAssertEqual(observedUsable[0.95], true)
        XCTAssertEqual(observedUsable[0.5], true)
        XCTAssertEqual(observedUsable[0.1], true)
        XCTAssertEqual(observedUsable[1e-3], true)
        XCTAssertEqual(observedUsable[1e-5], true)
        XCTAssertEqual(observedUsable[1e-7], true)
        XCTAssertEqual(observedUsable[1e-9], true)
        XCTAssertEqual(observedUsable[Float.leastNormalMagnitude], true)
        XCTAssertEqual(observedSupport[1e-9], 1)
        XCTAssertEqual(observedSupport[Float.leastNormalMagnitude], 1)
    }

    /// Restates the two points the sweep above previously located as a
    /// degeneracy boundary (`1e-7` and `1e-9`) as a dedicated, two-point pin.
    /// Both are now usable with support forced to exactly 1 by the
    /// `min_tokens_to_keep=1` fix -- kept as its own focused test (rather
    /// than relying solely on the sweep) since the original boundary claim
    /// had its own dedicated regression guard.
    func testProductionWidthBothFormerlyDegeneratePointsNowUsable() {
        let (sumAtSmallestUsable, supportAtSmallestUsable) =
            measure(vocabularySize: Self.productionVocabularySize, topP: 1e-7)
        let (sumAtFormerlyDegenerate, supportAtFormerlyDegenerate) =
            measure(vocabularySize: Self.productionVocabularySize, topP: 1e-9)
        print(
            "[boundary] topP=1e-7 sum=\(sumAtSmallestUsable) support=\(supportAtSmallestUsable) "
                + "usable=\(isUsable(sumAtSmallestUsable))")
        print(
            "[boundary] topP=1e-9 sum=\(sumAtFormerlyDegenerate) support=\(supportAtFormerlyDegenerate) "
                + "usable=\(isUsable(sumAtFormerlyDegenerate))")

        XCTAssertTrue(isUsable(sumAtSmallestUsable))
        // PRE-FIX this was `XCTAssertFalse` (NaN sum, 0 support). POST-FIX
        // the `min_tokens_to_keep=1` floor keeps exactly one token here too.
        XCTAssertTrue(isUsable(sumAtFormerlyDegenerate))
        XCTAssertEqual(supportAtFormerlyDegenerate, 1)
    }

    // MARK: - Q3: is vocabulary width still a driver, post-fix?

    /// Repeats the single formerly-most-degenerate swept `topP` (`1e-9`) at
    /// three widths -- 8, 1000, and the production 151936 -- to determine
    /// whether the fix closes the degeneracy uniformly across widths or only
    /// at some of them.
    func testFormerlyDegenerateTopPUsableAcrossVocabularyWidths() {
        let formerlyMostDegenerateTopP: Float = 1e-9
        let widths = [8, 1_000, Self.productionVocabularySize]

        var observed: [Int: (sum: Double, usable: Bool, support: Int)] = [:]
        for width in widths {
            let (sum, supportSize) = measure(vocabularySize: width, topP: formerlyMostDegenerateTopP)
            let usable = isUsable(sum)
            observed[width] = (sum, usable, supportSize)
            print(
                "[width] V=\(width) topP=\(formerlyMostDegenerateTopP) sum=\(sum) "
                    + "support=\(supportSize) usable=\(usable)")
        }

        // PRE-FIX (measured 2026-09-09): the narrowest row (V=8, sum=1.0
        // exactly, support=1) stayed usable at topP=1e-9, while V=1000 and
        // the production V=151936 both degenerated (NaN sum, 0 support) at
        // that SAME topP. The suite's original comment here additionally
        // speculated that this split was caused by `cumsum`'s float32
        // running total landing on different sides of `1.0` depending on
        // how many terms were summed (fractionally above for V=8,
        // fractionally below for V>=1000). That mechanism was NEVER
        // independently measured beyond the boolean usable/degenerate
        // outcome itself -- it was inferred, not observed, and is
        // RETRACTED here as a factual claim rather than repeated.
        //
        // POST-FIX (this pin): the `min_tokens_to_keep=1` floor closes the
        // degeneracy uniformly -- ALL THREE widths are now usable, each
        // with support forced to exactly 1 (the row's most-probable token).
        // Width no longer determines usability at this topP; it never
        // needed to, on the structural argument in `applyTopPFilter`'s doc
        // comment (the OR only ever fires in the previously-all-masked
        // case, at any width).
        XCTAssertTrue(observed[8]!.usable)
        XCTAssertTrue(observed[1_000]!.usable)
        XCTAssertTrue(observed[Self.productionVocabularySize]!.usable)
        XCTAssertEqual(observed[8]!.support, 1)
        XCTAssertEqual(observed[1_000]!.support, 1)
        XCTAssertEqual(observed[Self.productionVocabularySize]!.support, 1)
    }

    // MARK: - Q4: anti-vacuity

    /// Proves the harness itself is capable of producing a healthy result:
    /// at the DEPLOYED preset (topP 0.95) and production width, the row must
    /// yield a finite, strictly-positive sum and non-empty support. If this
    /// failed, a degenerate reading anywhere else in this file could not be
    /// trusted as a real signal (it could just as easily be a broken
    /// fixture).
    func testDeployedPresetYieldsHealthyDistributionAtProductionWidth() {
        let (sum, supportSize) = measure(vocabularySize: Self.productionVocabularySize, topP: Self.deployedTopP)
        print("[anti-vacuity] V=\(Self.productionVocabularySize) topP=\(Self.deployedTopP) sum=\(sum) support=\(supportSize)")

        XCTAssertTrue(sum.isFinite)
        XCTAssertGreaterThan(sum, 0)
        XCTAssertGreaterThan(supportSize, 0)
    }

    // MARK: - T3: bit-identity gate -- the fix is inert outside the degenerate band

    /// Proves, by exact `Float` comparison (not `allClose`), that the
    /// `min_tokens_to_keep=1` fix changes NOTHING across the entire
    /// non-degenerate `topP` range, at three vocabulary widths. This is not
    /// an empirical hope -- it is structural: `sortedProbs = exp(sortedLogprobs)`
    /// is elementwise non-negative, and IEEE-754 round-to-nearest addition is
    /// monotonic in a non-negative operand, so `cumulativeProbs` is
    /// non-decreasing in float32 and its LAST sorted position (the row's
    /// argmax) always holds the array's maximum. So whenever the unmodified
    /// predicate keeps ANY token at all, it already keeps the last one, and
    /// OR-ing in "always keep the last sorted position" changes nothing.
    /// `topP = 1.0` is skipped (the filter is never invoked at `topP >= 1`,
    /// per both `applyTopPFilter` call sites' `topP > 0 && topP < 1` guard).
    ///
    /// NaN rows are explicitly excluded from this gate (asserted absent, not
    /// silently tolerated) because the fix is NOT structurally inert on a
    /// NaN-poisoned row -- see `testNaNPoisonedRowCharacterization` below,
    /// which characterizes that separately-diverging case on its own. None
    /// of the fixtures here are expected to produce NaN (the decaying
    /// fixture's logits are all finite and `topP` never saturates `1 - topP`
    /// to exactly `1.0f` in this swept range), so the assertion is a
    /// precondition check on the measurement, not new coverage of the NaN
    /// path.
    func testTopPFilterFixIsInertAcrossNonDegenerateRange() {
        // 1.0 is deliberately excluded -- see the doc comment above.
        let nonDegenerateTopPSweep: [Float] = [0.95, 0.8, 0.5, 0.1, 1e-3, 1e-5, 1e-7]
        let widths = [Self.productionVocabularySize, 8, 1_000]

        for width in widths {
            let logits = decayingLogitsRow(vocabularySize: width)
            for topP in nonDegenerateTopPSweep {
                let actual = truncatedSamplingProbabilities(
                    logits: logits, temperature: 1, topP: topP, topK: 0, minP: 0
                ).flattened()
                let expected = preFixTruncatedProbabilities(
                    logits: logits, temperature: 1, topP: topP
                ).flattened()
                eval(actual, expected)
                let actualValues = actual.asArray(Float.self)
                let expectedValues = expected.asArray(Float.self)

                // Precondition, not the gate itself: this sweep must never
                // land in NaN territory (see the doc comment above). Fail
                // loudly and specifically if it ever does, rather than
                // letting `!=` silently treat NaN != NaN as a "difference"
                // indistinguishable from a real regression.
                let actualNaNCount = actualValues.filter { $0.isNaN }.count
                let expectedNaNCount = expectedValues.filter { $0.isNaN }.count
                XCTAssertEqual(
                    actualNaNCount, 0,
                    "unexpected NaN in fixed output at V=\(width) topP=\(topP); "
                        + "this sweep is meant to stay entirely in the non-degenerate band")
                XCTAssertEqual(
                    expectedNaNCount, 0,
                    "unexpected NaN in pre-fix reference at V=\(width) topP=\(topP)")

                let (diffCount, firstIndex, firstA, firstB) =
                    countExactDifferences(actualValues, expectedValues)
                print(
                    "[inert] V=\(width) topP=\(topP) diffCount=\(diffCount) "
                        + "firstDiffIndex=\(String(describing: firstIndex)) "
                        + "fixed=\(String(describing: firstA)) preFix=\(String(describing: firstB))")
                XCTAssertEqual(
                    diffCount, 0,
                    "V=\(width) topP=\(topP): \(diffCount) of \(width) entries differ between the "
                        + "fixed helper and the pre-fix reference; first at index "
                        + "\(String(describing: firstIndex)) (fixed=\(String(describing: firstA)), "
                        + "preFix=\(String(describing: firstB)))")
            }
        }
    }

    // MARK: - T4: the fix is NOT inert on a NaN-poisoned row

    /// The inertness proof above assumes every entry is finite. It does not
    /// hold once a raw logit is NaN: `logSoftmax` propagates NaN (a NaN
    /// logit poisons the shared `logsumexp` normalizer, so the ENTIRE
    /// log-probability row becomes NaN, not just the poisoned position),
    /// `cumsum` of `exp(NaN) = NaN` is NaN from that point on, and
    /// `NaN > x` is always `false` -- so PRE-FIX, the row masks fully to
    /// `-infinity` (no position's comparison against the threshold is ever
    /// `true`) exactly like the ordinary degenerate case. POST-FIX, the
    /// keep-last-sorted-position floor force-keeps whichever position
    /// `argSort` places last; since every position is NaN here, `argSort`'s
    /// tie-break ordering (not a magnitude comparison) decides which
    /// original index that is.
    ///
    /// This test does not attempt to FIX NaN handling -- out of scope for
    /// the `min_tokens_to_keep=1` change. It exists to PIN whatever the
    /// fixed helper actually does on this input, from a real run, so a
    /// future change to either the fix or to NaN handling has a concrete
    /// regression guard instead of an unmeasured guess. A NaN logit
    /// reaching this helper at all indicates an upstream failure: the
    /// independent oracle in `InCheckpointSampledMTPAcceptanceCLI.swift`
    /// (`inCheckpointSampledMTPIndependentSoftmax`, `:405-410`, and
    /// `independentTruncatedProbabilities`, `:463-468`) refuses NaN logits
    /// up front, before any truncation arithmetic runs. The vendored shared
    /// helper exercised here has no equivalent upfront refusal.
    ///
    /// Measured on 2026-09-09 (see the "[nan-row] ..." line this test
    /// prints): the fixed helper's output row is entirely NaN (every entry,
    /// not just the poisoned one) -- both pre-fix and post-fix, this
    /// particular row is unusable garbage, but the mechanism differs as
    /// documented above (all-`-infinity` pre-fix vs. a NaN kept at one
    /// scattered position post-fix, which the final `softmax`'s
    /// `-infinity - (-infinity) = NaN` step then spreads back across the
    /// whole row regardless).
    func testNaNPoisonedRowCharacterization() {
        let vocabularySize = 64
        var values = (0 ..< vocabularySize).map { -Float($0) * 0.01 }
        let poisonedIndex = 7
        values[poisonedIndex] = Float.nan
        let logits = MLXArray(values).reshaped([1, vocabularySize])

        let probabilities = truncatedSamplingProbabilities(
            logits: logits, temperature: 1, topP: Self.deployedTopP, topK: 0, minP: 0
        ).flattened()
        eval(probabilities)
        let raw = probabilities.asArray(Float.self)
        let nanCount = raw.filter { $0.isNaN }.count
        let sum = raw.map(Double.init).reduce(0, +)
        print(
            "[nan-row] V=\(vocabularySize) poisonedIndex=\(poisonedIndex) nanCount=\(nanCount) "
                + "allNaN=\(nanCount == raw.count) sum=\(sum)")

        XCTAssertEqual(raw.count, vocabularySize)
        // Pinned observation: a single NaN raw logit poisons the ENTIRE
        // output row via `logSoftmax`'s shared normalizer, so every entry
        // -- not only the originally-poisoned index -- comes back NaN.
        XCTAssertEqual(nanCount, vocabularySize)
        XCTAssertTrue(sum.isNaN)
    }
}
