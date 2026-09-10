import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Verifies (or falsifies) the rank-genericity claim documented on
/// `applyTopPFilter` in
/// `spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:316-336`:
///
/// > "Built rank-generically on axis -1 via broadcasting (a `[V]`-shaped
/// > boolean mask broadcasts against any `[..., V]`-shaped `cumulativeProbs`)
/// > -- this function has no rank precondition, unlike `applyTopKFilter`
/// > below, which does."
///
/// `applyTopPFilter` is `public`, so it is called directly here (no
/// `@testable import` needed). Both live call sites
/// (`CategoricalSampler.sample` / `TopPSampler` in the same file) only ever
/// pass rank-2 (`[1, V]`) logprobs, and every existing suite that exercises
/// top-p behavior goes through `truncatedSamplingProbabilities` (which itself
/// preconditions rank >= 2) or `TopPSampler.sample()` -- so before this file,
/// the rank-1 branch of the doc comment's claim had never actually been run.
///
/// This is a MEASUREMENT suite: it runs the real vendored code at rank 1 and
/// rank 2 and pins what is actually observed, rather than trusting the
/// comment.
final class TopPFilterRankGenericityTests: XCTestCase {

    /// Vocabulary width for the fixture row. Deliberately modest (not
    /// production-scale) since this suite is about rank, not width; a small,
    /// exactly-inspectable width keeps failure messages legible.
    private static let vocabularySize = 64

    /// Vocabulary index the fixture's peak (argmax) is permuted to. Deliberately
    /// interior -- neither `0` nor `vocabularySize - 1`.
    ///
    /// WHY THE PEAK IS PERMUTED (do not "simplify" this back to a plain
    /// monotonic row): with an unpermuted monotonically-decaying row, THREE
    /// distinct things collapse onto the very same vocabulary index -- the
    /// argmax, the last *sorted* position (the position the
    /// `keepLastSortedPosition` floor in `applyTopPFilter` actually
    /// force-keeps), and the literal index `0`. A defect that simply
    /// hardcodes "return index 0" then passes every assertion in this file
    /// identically to the correct implementation, because all three land on
    /// index 0 together. That is exactly the historical defect the floor was
    /// added to fix: per
    /// `docs/task-inbox/2026-09-09-top-p-nucleus-min-tokens-to-keep-ROOT-FIX.md`,
    /// pre-fix, `topP=1e-9` returned index `0` regardless of where the row's
    /// true peak sat -- even with the peak permuted to index `12345`, at
    /// V=151936. Permuting the peak to a mid-vocabulary index here (away from
    /// both `0` and the last index) is what makes this fixture able to tell
    /// the correct implementation apart from that "always returns 0" defect;
    /// removing the permutation silently destroys that discrimination while
    /// every assertion below keeps reading as if it still proves something.
    private static let peakIndex = 37

    /// Production decoder vocabulary width (matches
    /// `SampledMTPDegenerateTopPCharacterizationTests.productionVocabularySize`),
    /// used ONLY by the floor-binding test below
    /// (`testRankOneFloorBindsWhenCumulativeMassTestKeepsNothing`).
    ///
    /// WHY THAT TEST NEEDS PRODUCTION WIDTH, NOT `vocabularySize` (64): at
    /// `topP = Float.leastNormalMagnitude` and this file's narrower fixture,
    /// `cumsum`'s float32 summation rounding pushes the LAST sorted
    /// position's cumulative probability fractionally ABOVE `1.0f`
    /// (measured `1.0000002`, bits `1065353218`), which is already strictly
    /// greater than `1 - Float.leastNormalMagnitude` (which itself rounds to
    /// exactly `1.0f`). That means the bare `cumulativeProbs .> (1 - topP)`
    /// predicate keeps the argmax BY ITSELF at width 64 -- the
    /// `keepLastSortedPosition` floor is never the operative mechanism
    /// there, even though an earlier version of this file's floor test
    /// claimed (in its name, comment, and assertions) that it was. Deleting
    /// `.|| keepLastSortedPosition` from `applyTopPFilter` and rerunning at
    /// width 64 leaves that claim's test GREEN, which is exactly backwards
    /// for a test whose entire point is to prove the floor is load-bearing.
    ///
    /// At this production width with `decayStep = 0.01`, the rounding lands
    /// on the OTHER side of `1.0f` instead: `SampledMTPDegenerateTopPCharacterizationTests`
    /// independently measured (via `truncatedSamplingProbabilities`, which
    /// calls this same `applyTopPFilter`) that the PRE-FIX bare predicate at
    /// V=151936, topP=1e-9 produces a fully-masked row (NaN softmax sum, 0
    /// support) -- i.e. the bare predicate keeps NOTHING there. `topP=1e-9`
    /// and `topP=Float.leastNormalMagnitude` round `1 - topP` to the
    /// identical `1.0f` (both are many orders of magnitude below float32's
    /// ~1.19e-7 ULP at 1.0), so that measurement transfers directly to the
    /// `Float.leastNormalMagnitude` fixture used here. This width/decayStep
    /// combination is therefore in the actual floor-binding regime; see the
    /// PRECONDITION assertion inside the floor test itself, which measures
    /// this directly rather than only citing the sibling suite.
    private static let productionVocabularySize = 151_936

    /// Peak permutation target for the floor-binding fixture (see
    /// `productionVocabularySize`'s doc comment). Matches the interior index
    /// (`12345`) `docs/task-inbox/2026-09-09-top-p-nucleus-min-tokens-to-keep-ROOT-FIX.md`
    /// used to reproduce the historical "returns token 0 regardless of the
    /// true peak" defect at this same vocabulary width -- neither `0` nor
    /// `productionVocabularySize - 1`, and (since the peak is always the
    /// last *sorted* position by construction -- sorting is ascending by
    /// logprob, and the peak holds the maximum logprob) this is also
    /// necessarily distinct from the literal vocabulary index at the last
    /// sorted position landing on `0` or `productionVocabularySize - 1`.
    private static let productionPeakIndex = 12_345

    /// Deterministic logit row built from the monotonically decaying shape
    /// `logit[i] = -Float(i) * decayStep` (same construction as the sibling
    /// suite `SampledMTPDegenerateTopPCharacterizationTests.decayingLogitsRow`,
    /// at a steeper decay step so that a moderate `topP` (0.5) actually
    /// truncates part of the tail at this narrower width), with index `0`
    /// and index `peakIndex` swapped afterward so the row's argmax sits at
    /// the interior index `peakIndex` instead of at index `0`. See
    /// `peakIndex`'s doc comment for why that permutation is required for
    /// this fixture to discriminate at all. The decaying shape (a real,
    /// non-flat distribution, not degenerate by construction) is otherwise
    /// unchanged. No randomness is used, so the fixture reproduces
    /// byte-for-byte from (vocabularySize, decayStep, peakIndex) alone.
    private func decayingLogitValues(
        vocabularySize: Int, decayStep: Float = 0.05,
        peakIndex: Int = TopPFilterRankGenericityTests.peakIndex
    ) -> [Float] {
        precondition(vocabularySize > 0)
        precondition(peakIndex > 0 && peakIndex < vocabularySize - 1)
        var values = (0 ..< vocabularySize).map { -Float($0) * decayStep }
        values.swapAt(0, peakIndex)
        return values
    }

    /// Counts indices where two equal-length `Float` arrays differ under
    /// exact `==` (never `allClose`/`rtol`/`atol`). Mirrors
    /// `SampledMTPDegenerateTopPCharacterizationTests.countExactDifferences`:
    /// deliberately does NOT `XCTAssertEqual` two large arrays directly --
    /// a failing array `==` on a large collection triggers a diff that dumps
    /// both arrays and can wedge the run. Returns a compact scalar summary
    /// instead.
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

    /// Runs `applyTopPFilter` on the given raw logit values at the requested
    /// rank (1 -> shape `[V]`, 2 -> shape `[1, V]`), reproducing the
    /// pre-processing every live call site performs first (`logSoftmax`
    /// along the last axis) before the filter itself runs. Returns the
    /// flattened `Float` output so rank-1 and rank-2 results compare
    /// directly.
    private func runTopPFilter(
        values: [Float], topP: Float, rank: Int
    ) -> [Float] {
        let logits: MLXArray
        switch rank {
        case 1:
            logits = MLXArray(values)
        case 2:
            logits = MLXArray(values).reshaped([1, values.count])
        default:
            preconditionFailure("this fixture only builds rank 1 or rank 2 inputs")
        }
        let logprobs = logSoftmax(logits, axis: -1)
        let filtered = applyTopPFilter(
            logprobs, topP: MLXArray(topP), negInf: MLXArray(-Float.infinity)
        )
        eval(filtered)
        return filtered.flattened().asArray(Float.self)
    }

    // MARK: - T1: rank 1 executes and agrees with rank 2, exactly

    /// Calls `applyTopPFilter` on the SAME logit values twice -- once at
    /// rank 1 (`[V]`), once at rank 2 (`[1, V]`) -- and requires the rank-1
    /// output to equal row 0 of the rank-2 output EXACTLY. This is the
    /// direct measurement of the doc comment's rank-genericity claim:
    /// identical op sequence on identical values along the same axis, so any
    /// difference at all is a real defect (hence exact equality, not
    /// `allClose`).
    func testRankOneAgreesExactlyWithRankTwo() {
        let values = decayingLogitValues(vocabularySize: Self.vocabularySize)
        let topP: Float = 0.5

        let rank1Output = runTopPFilter(values: values, topP: topP, rank: 1)
        let rank2Output = runTopPFilter(values: values, topP: topP, rank: 2)

        XCTAssertEqual(rank1Output.count, Self.vocabularySize)
        XCTAssertEqual(rank2Output.count, Self.vocabularySize)

        let (diffCount, firstIndex, firstA, firstB) =
            countExactDifferences(rank1Output, rank2Output)
        print(
            "[rank-agree] V=\(Self.vocabularySize) topP=\(topP) diffCount=\(diffCount) "
                + "firstDiffIndex=\(String(describing: firstIndex)) "
                + "rank1=\(String(describing: firstA)) rank2=\(String(describing: firstB))")
        XCTAssertEqual(
            diffCount, 0,
            "\(diffCount) of \(Self.vocabularySize) entries differ between rank-1 and rank-2 output; "
                + "first at index \(String(describing: firstIndex)) (rank1=\(String(describing: firstA)), "
                + "rank2=\(String(describing: firstB))). Identical op sequence on identical values along "
                + "the same axis should produce identical results at any rank.")

        // Separately pin that the two ranks agree on WHICH positions were
        // masked to -infinity, not just on the overall diff count (the diff
        // count already covers this since -inf == -inf in Swift, but the
        // task calls for this as its own explicit scalar check).
        let rank1NegInfCount = rank1Output.filter { $0 == -Float.infinity }.count
        let rank2NegInfCount = rank2Output.filter { $0 == -Float.infinity }.count
        print("[rank-agree] rank1NegInfCount=\(rank1NegInfCount) rank2NegInfCount=\(rank2NegInfCount)")
        XCTAssertEqual(rank1NegInfCount, rank2NegInfCount)
    }

    // MARK: - T2: anti-vacuity -- the fixture actually filters, and the floor actually binds

    /// `testRankOneAgreesExactlyWithRankTwo` would pass trivially if
    /// `applyTopPFilter` were a no-op at both ranks (every position finite,
    /// nothing ever masked). This test proves the `topP = 0.5` fixture used
    /// there is not vacuous: at rank 1, at least one position must be
    /// `-infinity` (the filter genuinely truncated something).
    func testRankOneFixtureActuallyFilters() {
        let values = decayingLogitValues(vocabularySize: Self.vocabularySize)
        let topP: Float = 0.5

        let rank1Output = runTopPFilter(values: values, topP: topP, rank: 1)
        let negInfCount = rank1Output.filter { $0 == -Float.infinity }.count
        print("[anti-vacuity] V=\(Self.vocabularySize) topP=\(topP) negInfCount=\(negInfCount)")

        XCTAssertGreaterThan(
            negInfCount, 0,
            "topP=\(topP) truncated nothing at rank 1 -- the T1 fixture would then be agreeing on a "
                + "no-op, which proves nothing about rank genericity under real filtering.")
    }

    /// Exercises the specific line the rank-generic `keepLastSortedPosition`
    /// mask exists for, at rank 1: a `topP` small enough that `1 - topP`
    /// rounds to exactly `1.0f` in float32, AND a fixture width/decayStep
    /// combination where `cumsum`'s float32 rounding does NOT overshoot
    /// `1.0f` at the last sorted position, so `cumulativeProbs .> (1 - topP)`
    /// is false at EVERY sorted position. The cumulative-mass test alone
    /// then keeps nothing at all; the ONLY reason anything survives is the
    /// unconditional "keep the last sorted position" floor.
    ///
    /// USES PRODUCTION WIDTH (`productionVocabularySize` /
    /// `productionPeakIndex`), NOT this file's `vocabularySize` (64): see
    /// `productionVocabularySize`'s doc comment for the measured reason --
    /// at width 64, `cumsum` rounds fractionally ABOVE `1.0f` at the final
    /// sorted position, so the bare predicate ALREADY keeps the argmax by
    /// itself there and this test would be passing for the wrong reason (a
    /// prior version of this test made exactly that mistake). The
    /// PRECONDITION assertion immediately below measures, at these actual
    /// parameters, that the bare predicate really does keep nothing -- do
    /// not trust this doc comment over that assertion.
    func testRankOneFloorBindsWhenCumulativeMassTestKeepsNothing() {
        let values = decayingLogitValues(
            vocabularySize: Self.productionVocabularySize, decayStep: 0.01,
            peakIndex: Self.productionPeakIndex)
        let topP = Float.leastNormalMagnitude
        let expectedArgmaxIndex = values.indices.max { values[$0] < values[$1] }!

        // --- PRECONDITION: the bare cumulative-mass predicate, evaluated on
        // its own (never through `applyTopPFilter`), keeps NOTHING at these
        // parameters. `applyTopPFilter`'s internal pipeline is
        // `logSoftmax` -> `argSort` -> `takeAlong` -> `exp` -> `cumsum`;
        // this precondition reruns exactly that pipeline directly, then
        // checks the resulting cumulative array's MAXIMUM against
        // `1 - topP`. `cumulativeProbs` is non-decreasing in float32 (see
        // `applyTopPFilter`'s doc comment: summing non-negative
        // probabilities via IEEE-754 round-to-nearest addition cannot
        // decrease the running total), so the maximum is sufficient -- if
        // it does not clear the threshold, no earlier position does either.
        //
        // If this assertion fails, the fixture has drifted OUT of the
        // floor-binding regime (most likely: `cumsum` now rounds fractionally
        // above `1.0f` at production width, the same way it already does at
        // this file's narrower `vocabularySize` -- see that doc comment) and
        // everything below proves nothing about the `keepLastSortedPosition`
        // floor at all; pick different fixture parameters instead of
        // trusting the assertions below.
        let logits = MLXArray(values)
        let logprobs = logSoftmax(logits, axis: -1)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let maxCumulativeProb = cumulativeProbs.max().item(Float.self)
        let threshold = Float(1) - topP
        print(
            "[floor-precondition] V=\(Self.productionVocabularySize) topP=\(topP) "
                + "maxCumulativeProb=\(maxCumulativeProb) threshold=\(threshold)")
        XCTAssertFalse(
            maxCumulativeProb > threshold,
            "ANTI-VACUITY FAILURE: the bare predicate `cumulativeProbs .> (1 - topP)` does NOT keep "
                + "nothing at these parameters -- the maximum cumulative probability "
                + "(\(maxCumulativeProb)) already exceeds the threshold (\(threshold)) without the "
                + "`keepLastSortedPosition` floor. testRankOneFloorBindsWhenCumulativeMassTestKeepsNothing "
                + "proves nothing about the floor unless this precondition holds -- the fixture has "
                + "drifted out of the floor-binding regime; pick different (vocabularySize, decayStep, "
                + "topP) parameters rather than trusting the assertions below.")

        let rank1Output = runTopPFilter(values: values, topP: topP, rank: 1)
        let finiteIndices = rank1Output.indices.filter { rank1Output[$0].isFinite }
        print(
            "[floor] V=\(Self.productionVocabularySize) topP=\(topP) finiteCount=\(finiteIndices.count) "
                + "finiteIndices=\(finiteIndices) expectedArgmaxIndex=\(expectedArgmaxIndex)")

        XCTAssertEqual(
            finiteIndices.count, 1,
            "expected exactly one surviving finite position (the keep-last-sorted-position floor), "
                + "got \(finiteIndices.count): \(finiteIndices)")
        if let onlySurvivor = finiteIndices.first {
            XCTAssertEqual(
                onlySurvivor, expectedArgmaxIndex,
                "the single surviving position must be the input row's argmax (index "
                    + "\(expectedArgmaxIndex)), got \(onlySurvivor)")
            // The fixture's peak is permuted away from index 0 specifically
            // so this assertion can fail: pre-fix, `applyTopPFilter` returned
            // index 0 unconditionally once `1 - topP` saturated to `1.0f`,
            // regardless of where the row's true peak sat (see
            // `docs/task-inbox/2026-09-09-top-p-nucleus-min-tokens-to-keep-ROOT-FIX.md`).
            // With an unpermuted (monotonic) fixture, `expectedArgmaxIndex`
            // would itself be 0, so a "hardcode index 0" defect would satisfy
            // the assertion above too. This assertion names that historical
            // "returned token 0" defect explicitly so it stays caught even if
            // `expectedArgmaxIndex` is ever miscomputed.
            XCTAssertNotEqual(
                onlySurvivor, 0,
                "the surviving position is index 0, which is the historical degenerate-topP defect "
                    + "(applyTopPFilter returning token 0 unconditionally once `1 - topP` saturates to "
                    + "1.0f, independent of the row's true peak) -- this fixture's peak is deliberately "
                    + "permuted to index \(Self.productionPeakIndex) so that defect cannot hide behind an "
                    + "argmax-happens-to-be-0 coincidence.")
        }
    }

    // MARK: - T3: the contrast with `applyTopKFilter` is real, not assumed

    /// `applyTopPFilter`'s doc comment asserts an asymmetry: it has no rank
    /// precondition, "unlike `applyTopKFilter` below, which does." That
    /// precondition is a Swift `precondition(...)` (a fatal trap), not a
    /// throwing check -- triggering it at rank 1 would crash this entire
    /// test process, not raise a catchable error. This suite therefore does
    /// NOT attempt to trigger the rank-1 trap in-process; doing so would
    /// take down the whole test bundle rather than produce a red test.
    ///
    /// What this test instead confirms, from the real vendored source (not
    /// an assumption): `applyTopKFilter`'s declaration is guarded by exactly
    /// the `precondition(logprobs.ndim >= 2, ...)` its doc comment describes
    /// (checked below by grepping the vendored file itself, so the check
    /// stays honest if the source ever changes), and that the function DOES
    /// execute successfully and correctly at the rank (2) its precondition
    /// permits -- i.e. the asymmetry is a real, present-in-source rank gate
    /// on `applyTopKFilter`, not a comment describing behavior nobody wrote.
    func testTopKFilterHasRankPreconditionSourceAndWorksAtPermittedRank() {
        // --- 1. Confirm the precondition is actually present in source. ---
        let evaluateSwiftPath =
            "\(FileManager.default.currentDirectoryPath)/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift"
        let source: String
        if let contents = try? String(contentsOfFile: evaluateSwiftPath, encoding: .utf8) {
            source = contents
        } else {
            // Fall back to a path relative to this test bundle's package
            // root, in case the working directory differs across
            // invocations (e.g. `swift test --package-path spike` vs. an
            // Xcode test run).
            let fallbackPath =
                "\(#filePath)"
                    .replacingOccurrences(of: "Tests/SpikeCoreTests/TopPFilterRankGenericityTests.swift", with: "")
                + "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift"
            source = (try? String(contentsOfFile: fallbackPath, encoding: .utf8)) ?? ""
        }
        XCTAssertTrue(
            source.contains("precondition(") && source.contains("logprobs.ndim >= 2"),
            "expected to find `precondition(logprobs.ndim >= 2, ...)` in the vendored "
                + "Evaluate.swift near `applyTopKFilter`; if this fails, either the source moved or "
                + "the precondition text changed and this check needs updating -- do not weaken it to "
                + "a structural assumption instead of a real source check.")

        // --- 2. Confirm it actually works, correctly, at the permitted rank (2). ---
        let values = decayingLogitValues(vocabularySize: Self.vocabularySize)
        let logits = MLXArray(values).reshaped([1, values.count])
        let logprobs = logSoftmax(logits, axis: -1)
        let topK = 5
        let filtered = applyTopKFilter(logprobs, topK: topK, negInf: MLXArray(-Float.infinity))
        eval(filtered)
        let output = filtered.flattened().asArray(Float.self)
        let finiteCount = output.filter { $0.isFinite }.count
        print("[topk-rank2] V=\(Self.vocabularySize) topK=\(topK) finiteCount=\(finiteCount)")

        XCTAssertEqual(
            finiteCount, topK,
            "applyTopKFilter at rank 2 should keep exactly topK=\(topK) finite positions, got "
                + "\(finiteCount)")
    }
}
