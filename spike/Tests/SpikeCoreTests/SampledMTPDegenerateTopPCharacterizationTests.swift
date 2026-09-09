import MLX
import MLXLMCommon
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

    // MARK: - Q1 & Q2: does it degenerate at all, and where is the boundary?

    /// Sweeps `topP` at production width and prints the measured
    /// (sum, finite?, support size) for every point, then asserts the
    /// observed usable/degenerate verdict at each point individually so the
    /// sweep is pinned rather than just eyeballed.
    func testProductionWidthTopPSweepDegeneracyBoundary() {
        let sweep: [Float] = [1.0, 0.95, 0.5, 0.1, 1e-3, 1e-5, 1e-7, 1e-9, Float.leastNormalMagnitude]

        var observedUsable: [Float: Bool] = [:]
        for topP in sweep {
            let (sum, supportSize) = measure(vocabularySize: Self.productionVocabularySize, topP: topP)
            let usable = isUsable(sum)
            observedUsable[topP] = usable
            print(
                "[sweep] V=\(Self.productionVocabularySize) topP=\(topP) sum=\(sum) "
                    + "finite=\(sum.isFinite) support=\(supportSize) usable=\(usable)")
        }

        // Pinned per-point verdicts, measured on 2026-09-09 against this
        // exact fixture (see the "[sweep] ..." lines this test prints for
        // the raw sum/support at every point). It DOES degenerate at
        // production width, but only at the two smallest swept points:
        // `topP <= 1e-7` stays usable (support collapses to a single token,
        // which is the expected nucleus-sampling behavior at a vanishingly
        // small topP, not a bug); `topP <= 1e-9` produces a NaN sum (every
        // entry masked to -infinity before the final softmax, i.e. 0/0).
        // The boundary is between 1e-7 and 1e-9, NOT between 1e-3 and 1e-5
        // as the pre-run hypothesis guessed -- confirmed by re-running with
        // the actual measured numbers below.
        XCTAssertEqual(observedUsable[1.0], true)
        XCTAssertEqual(observedUsable[0.95], true)
        XCTAssertEqual(observedUsable[0.5], true)
        XCTAssertEqual(observedUsable[0.1], true)
        XCTAssertEqual(observedUsable[1e-3], true)
        XCTAssertEqual(observedUsable[1e-5], true)
        XCTAssertEqual(observedUsable[1e-7], true)
        XCTAssertEqual(observedUsable[1e-9], false)
        XCTAssertEqual(observedUsable[Float.leastNormalMagnitude], false)
    }

    /// Restates the boundary located by the sweep above as a dedicated,
    /// two-point pin: the smallest swept `topP` that still yields a usable
    /// distribution (`1e-7`) against the largest swept `topP` that degenerates
    /// (`1e-9`). Kept as its own test (rather than relying solely on the
    /// sweep) so the boundary claim has a minimal, focused regression guard.
    func testProductionWidthNarrowedBoundary() {
        let (sumAtSmallestUsable, _) = measure(vocabularySize: Self.productionVocabularySize, topP: 1e-7)
        let (sumAtLargestDegenerate, _) = measure(vocabularySize: Self.productionVocabularySize, topP: 1e-9)
        print("[boundary] topP=1e-7 sum=\(sumAtSmallestUsable) usable=\(isUsable(sumAtSmallestUsable))")
        print("[boundary] topP=1e-9 sum=\(sumAtLargestDegenerate) usable=\(isUsable(sumAtLargestDegenerate))")

        XCTAssertTrue(isUsable(sumAtSmallestUsable))
        XCTAssertFalse(isUsable(sumAtLargestDegenerate))
    }

    // MARK: - Q3: is vocabulary width the driver?

    /// Repeats the single most-degenerate swept `topP` (`1e-9`) at three
    /// widths -- 8, 1000, and the production 151936 -- to determine whether
    /// row width changes the outcome or whether the degeneracy is a
    /// width-independent float32 property of "`1 - topP` rounds
    /// indistinguishably from `1.0`" at that `topP` magnitude.
    func testMostDegenerateTopPAcrossVocabularyWidths() {
        let mostDegenerateTopP: Float = 1e-9
        let widths = [8, 1_000, Self.productionVocabularySize]

        var observed: [Int: (sum: Double, usable: Bool)] = [:]
        for width in widths {
            let (sum, supportSize) = measure(vocabularySize: width, topP: mostDegenerateTopP)
            let usable = isUsable(sum)
            observed[width] = (sum, usable)
            print("[width] V=\(width) topP=\(mostDegenerateTopP) sum=\(sum) support=\(supportSize) usable=\(usable)")
        }

        // Pinned: width DOES matter, contrary to a pre-run guess that this
        // topP magnitude alone (independent of V) would saturate `1 - topP`
        // to `1.0` and mask every width equally. Measured instead: the
        // narrowest row (V=8, sum=1.0 exactly, support=1) stays usable at
        // topP=1e-9, while V=1000 and the production V=151936 both degenerate
        // (NaN sum, 0 support) at that SAME topP. `cumsum`'s final float32
        // value is the running total of a monotonically-sorted probability
        // row; with only 8 terms accumulated it lands fractionally above
        // 1.0 (so it still clears the `> (1 - topP)` threshold), while with
        // 1000+ terms accumulated it lands fractionally below 1.0 (so it
        // never clears the threshold once `1 - topP` itself has rounded to
        // `1.0`). Which side of 1.0 the rounding error falls on is a
        // function of how many terms get summed, i.e. of vocabulary width.
        XCTAssertTrue(observed[8]!.usable)
        XCTAssertFalse(observed[1_000]!.usable)
        XCTAssertFalse(observed[Self.productionVocabularySize]!.usable)
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
}
