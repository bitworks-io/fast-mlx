import MLX
import MLXLMCommon
import XCTest

@testable import fastmlx_harness

/// Independent hand-computed checks of the PURE arithmetic in
/// `InCheckpointSampledMTPAcceptanceCLI.swift` -- softmax/sigmaMinPQ, per-step acceptance,
/// break-even, and the Wilson-interval/agreement cross-check (Fix 5). These functions are the
/// measurement's own independent cross-check -- if they are wrong, the whole measurement's
/// validity check is wrong -- so every expected value here is worked out by hand in the comment
/// beside the assertion, never by calling the function under test and never copied from a prior
/// run's output. No model load and no network, but ONE test (the Fix 6 integration test at the
/// bottom of this class) does exercise real `MLXArray`/Metal compute -- it is documented as an
/// exception where it appears, not folded silently into this "pure" claim.
final class InCheckpointSampledMTPAcceptanceArithmeticTests: XCTestCase {

    // MARK: - inCheckpointSampledMTPIndependentSoftmax

    func testSoftmaxOfEqualLogitsIsExactlyOneHalfEach() throws {
        // exp(0-0)=1, exp(0-0)=1; sum=2; probs = [1/2, 1/2] exactly (0.5 is exactly representable).
        let probabilities = try inCheckpointSampledMTPIndependentSoftmax([0, 0])

        XCTAssertEqual(probabilities.count, 2)
        XCTAssertEqual(probabilities[0], 0.5)
        XCTAssertEqual(probabilities[1], 0.5)
    }

    func testSoftmaxMapsNegativeInfinityLogitToExactlyZeroProbability() throws {
        // Masked media-sentinel case: logits [0, -inf, 0]. max = 0.
        // exp(0-0)=1, exp(-inf-0)=exp(-inf)=0 exactly, exp(0-0)=1. sum = 2.
        // probs = [1/2, 0/2, 1/2] = [0.5, 0.0, 0.5] exactly.
        let probabilities = try inCheckpointSampledMTPIndependentSoftmax([0, -Double.infinity, 0])

        XCTAssertEqual(probabilities.count, 3)
        XCTAssertEqual(probabilities[0], 0.5)
        XCTAssertEqual(probabilities[1], 0.0)
        XCTAssertEqual(probabilities[2], 0.5)
    }

    func testSoftmaxRejectsNaN() {
        XCTAssertThrowsError(try inCheckpointSampledMTPIndependentSoftmax([0, Double.nan]))
    }

    func testSoftmaxRejectsPositiveInfinity() {
        XCTAssertThrowsError(try inCheckpointSampledMTPIndependentSoftmax([0, Double.infinity]))
    }

    func testSoftmaxRejectsEmptyVocabulary() {
        XCTAssertThrowsError(try inCheckpointSampledMTPIndependentSoftmax([]))
    }

    func testSoftmaxRejectsAllNegativeInfinity() {
        // No finite maxLogit exists to subtract; every entry is -inf.
        XCTAssertThrowsError(
            try inCheckpointSampledMTPIndependentSoftmax([-Double.infinity, -Double.infinity]))
    }

    func testSoftmaxIsNumericallyStableUnderLargeSharedOffset() throws {
        // Max-subtraction: [1000, 1000] -> subtract max=1000 -> [0, 0] -> exp -> [1,1] -> sum=2
        // -> [0.5, 0.5] exactly. Without max-subtraction, exp(1000) would overflow to +inf.
        let probabilities = try inCheckpointSampledMTPIndependentSoftmax([1000, 1000])

        XCTAssertEqual(probabilities.count, 2)
        XCTAssertEqual(probabilities[0], 0.5)
        XCTAssertEqual(probabilities[1], 0.5)
    }

    // MARK: - inCheckpointSampledMTPSigmaMinPQ

    func testSigmaMinPQOfIdenticalDistributionsIsExactlyOne() {
        // Sum of min(p,p) over identical entries is just Sum p = 1.0 (a valid distribution).
        let value = inCheckpointSampledMTPSigmaMinPQ(target: [0.25, 0.75], draft: [0.25, 0.75])

        XCTAssertEqual(value, 1.0)
    }

    func testSigmaMinPQOfDisjointSupportsIsExactlyZero() {
        // p puts all mass on index 0, q puts all mass on index 1: min(1,0)+min(0,1) = 0+0 = 0.
        let value = inCheckpointSampledMTPSigmaMinPQ(target: [1.0, 0.0], draft: [0.0, 1.0])

        XCTAssertEqual(value, 0.0)
    }

    func testSigmaMinPQPartialOverlapHandComputed() {
        // p=[0.5,0.5], q=[0.25,0.75]:
        // min(0.5,0.25) = 0.25
        // min(0.5,0.75) = 0.5
        // total = 0.25 + 0.5 = 0.75
        let value = inCheckpointSampledMTPSigmaMinPQ(target: [0.5, 0.5], draft: [0.25, 0.75])

        XCTAssertEqual(value, 0.75)
    }

    // MARK: - inCheckpointSampledMTPPerStepAcceptance
    //
    // Semantics restated from the source: with `stepCount` steps (0-based index i), a block with
    // `proposedCount`/`acceptedDraftCount` contributes to step i only if i < proposedCount, and
    // then:
    //   reached[i]  iff i == 0 OR acceptedDraftCount >= i
    //   accepted[i] iff acceptedDraftCount > i
    //
    // Hand-derive per-block, per-step (proposedCount=2, stepCount=2, so steps {0,1} both always
    // in range since i < 2 for i in {0,1}):
    //
    //   Block A: acceptedDraftCount=0 (rejected immediately at step 0)
    //     step 0: reached (i==0) ; accepted? 0>0 false -> NOT accepted
    //     step 1: reached? i==0? no. acceptedDraftCount>=1? 0>=1 false -> NOT reached
    //             (accepted only counted if reached; here it's not reached, so not accepted either)
    //
    //   Block B: acceptedDraftCount=1 (accepted step 0, rejected at step 1)
    //     step 0: reached (i==0); accepted? 1>0 true -> accepted
    //     step 1: reached? acceptedDraftCount>=1 -> 1>=1 true -> reached
    //             accepted? 1>1 false -> NOT accepted
    //
    //   Block C: acceptedDraftCount=2 (accepted both drafted positions, full block)
    //     step 0: reached (i==0); accepted? 2>0 true -> accepted
    //     step 1: reached? 2>=1 true -> reached; accepted? 2>1 true -> accepted
    //
    // Pooling A+B+C:
    //   step 0: reached = 3 (A,B,C all reach step 0); accepted = 2 (B,C accepted; A did not)
    //   step 1: reached = 2 (B,C reach step 1; A does NOT); accepted = 1 (only C)
    func testPerStepAcceptanceHandDerivedAcrossThreeBlocks() {
        let outcomes = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 0),  // Block A
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 1),  // Block B
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 2),  // Block C
        ]

        let steps = inCheckpointSampledMTPPerStepAcceptance(outcomes, stepCount: 2)

        XCTAssertEqual(steps.count, 2)
        XCTAssertEqual(steps[0].stepIndex, 0)
        XCTAssertEqual(steps[0].reachedCount, 3)
        XCTAssertEqual(steps[0].acceptedCount, 2)
        XCTAssertEqual(steps[1].stepIndex, 1)
        XCTAssertEqual(steps[1].reachedCount, 2)
        XCTAssertEqual(steps[1].acceptedCount, 1)
    }

    func testPerStepAcceptanceStep1IsNotReachedWhenStep0Rejected() {
        // CRITICAL DISCRIMINATING CASE: a single block with proposedCount=2, acceptedDraftCount=0
        // (rejected immediately at step 0). By the real semantics, step 1's reached condition is
        // `acceptedDraftCount >= 1`, i.e. `0 >= 1`, which is FALSE -- step 1 is NOT reached, and
        // reachedCount for step 1 must be exactly 0.
        //
        // A NAIVE implementation that instead counted every step index < proposedCount as
        // "reached" (ignoring the consecutive-prefix semantics -- i.e. treating step 1 as reached
        // purely because proposedCount=2 covers it) would report reachedCount=1 for step 1 here
        // instead of the correct 0. That wrong number is exactly what this test guards against.
        let outcomes = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 0)
        ]

        let steps = inCheckpointSampledMTPPerStepAcceptance(outcomes, stepCount: 2)

        XCTAssertEqual(steps.count, 2)
        XCTAssertEqual(steps[0].reachedCount, 1)
        XCTAssertEqual(steps[0].acceptedCount, 0)
        XCTAssertEqual(steps[1].reachedCount, 0)  // NOT 1 -- see comment above.
        XCTAssertEqual(steps[1].acceptedCount, 0)
    }

    func testPerStepAcceptanceRateIsNilWhenUnreachedNeverZero() {
        // Same single-block case as above: step 1 has reachedCount=0, so acceptanceRate must be
        // nil -- an unreached step has no rate. Fabricating 0.0 would understate depth decay by
        // making an unreached step look like a reached-but-always-rejected step.
        let outcomes = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 0)
        ]

        let steps = inCheckpointSampledMTPPerStepAcceptance(outcomes, stepCount: 2)

        XCTAssertNil(steps[1].acceptanceRate)
        // Step 0 IS reached (reachedCount=1) and accepted 0 of 1: rate = 0/1 = 0.0 exactly.
        XCTAssertEqual(steps[0].acceptanceRate, 0.0)
    }

    func testPerStepAcceptanceWithZeroStepCountIsEmpty() {
        let outcomes = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 2)
        ]

        let steps = inCheckpointSampledMTPPerStepAcceptance(outcomes, stepCount: 0)

        XCTAssertEqual(steps.count, 0)
    }

    // MARK: - inCheckpointSampledMTPPooledAcceptance

    func testPooledAcceptanceIsNilWhenNothingWasProposed() {
        // Zero-block run: proposedCount sums to 0. Must be nil, NOT 0.0 -- a zero-block run must
        // not read like a clean run that simply accepted nothing.
        let outcomes: [InCheckpointSampledMTPBlockOutcome] = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 0, acceptedDraftCount: 0)
        ]

        XCTAssertNil(inCheckpointSampledMTPPooledAcceptance(outcomes))
        XCTAssertNil(inCheckpointSampledMTPPooledAcceptance([]))
    }

    func testPooledAcceptanceHandComputedOverSeveralBlocks() throws {
        // proposed: 2 + 2 + 2 = 6
        // accepted: 0 + 1 + 2 = 3
        // pooled ratio = 3 / 6 = 0.5 exactly.
        let outcomes = [
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 0),
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 1),
            InCheckpointSampledMTPBlockOutcome(proposedCount: 2, acceptedDraftCount: 2),
        ]

        let pooled = try XCTUnwrap(inCheckpointSampledMTPPooledAcceptance(outcomes))

        XCTAssertEqual(pooled, 0.5)
    }

    // MARK: - inCheckpointSampledMTPBreakEvenAcceptance
    //
    // Solves a from a^2 + a = 2*delta + D/T, i.e. 1 + a + a^2 = 1 + 2*delta + D/T. Verified here
    // by SUBSTITUTION -- plugging the returned `a` back into `1 + a + a*a` and checking it equals
    // `1 + 2*delta + dOverT` -- rather than by reimplementing the quadratic formula, which would
    // just duplicate the function under test instead of independently checking it.

    func testBreakEvenAcceptanceAtZeroDeltaAndZeroDOverTIsExactlyZero() {
        // rightHandSide = 2*0 + 0 = 0; discriminant = 1 + 4*0 = 1; sqrt(1) = 1;
        // a = (-1 + 1) / 2 = 0 exactly.
        let a = try? XCTUnwrap(inCheckpointSampledMTPBreakEvenAcceptance(delta: 0, dOverT: 0))

        XCTAssertEqual(a, 0.0)
    }

    func testBreakEvenAcceptanceSatisfiesQuadraticBySubstitution() throws {
        let delta = 0.1
        let dOverT = 0.3
        let a = try XCTUnwrap(inCheckpointSampledMTPBreakEvenAcceptance(delta: delta, dOverT: dOverT))

        let lhs = 1 + a + a * a
        let rhs = 1 + 2 * delta + dOverT
        XCTAssertLessThanOrEqual(abs(lhs - rhs), 1e-12)  // floating-point solve, not exact arithmetic
    }

    func testBreakEvenAcceptanceSatisfiesQuadraticBySubstitutionAtLargerInputs() throws {
        let delta = 1.25
        let dOverT = 4.0
        let a = try XCTUnwrap(inCheckpointSampledMTPBreakEvenAcceptance(delta: delta, dOverT: dOverT))

        let lhs = 1 + a + a * a
        let rhs = 1 + 2 * delta + dOverT
        XCTAssertLessThanOrEqual(abs(lhs - rhs), 1e-12)  // floating-point solve, not exact arithmetic
    }

    // MARK: - inCheckpointSampledMTPSummarize

    func testSummarizeOfEmptyIsNil() {
        XCTAssertNil(inCheckpointSampledMTPSummarize([]))
    }

    func testSummarizeHandComputedOverFiveValues() throws {
        // values: [1, 2, 3, 4, 10]
        // sum = 1+2+3+4+10 = 20; count = 5; mean = 20/5 = 4.0 exactly
        // min = 1.0, max = 10.0
        let summary = try XCTUnwrap(inCheckpointSampledMTPSummarize([1, 2, 3, 4, 10]))

        XCTAssertEqual(summary.mean, 4.0)
        XCTAssertEqual(summary.min, 1.0)
        XCTAssertEqual(summary.max, 10.0)
        XCTAssertEqual(summary.count, 5)
    }

    // MARK: - inCheckpointSampledMTPStepReached
    //
    // Predicate restated from the source: `step == 0 || acceptedDraftCount >= step`. Step 0 is
    // reached unconditionally (every block evaluates the first drafted position); step i>0 is
    // reached iff the block's consecutive accepted-prefix count is at least i.

    func testStepReachedStep0IsAlwaysTrueRegardlessOfAcceptedDraftCount() {
        // step == 0 short-circuits the predicate to true no matter what acceptedDraftCount is.
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 0, acceptedDraftCount: 0))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 0, acceptedDraftCount: 1))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 0, acceptedDraftCount: 2))
    }

    func testStepReachedStep1BoundaryOnAcceptedDraftCount() {
        // step 1 reached iff acceptedDraftCount >= 1: 0 -> false, 1 -> true (boundary), 2 -> true.
        XCTAssertFalse(inCheckpointSampledMTPStepReached(step: 1, acceptedDraftCount: 0))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 1, acceptedDraftCount: 1))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 1, acceptedDraftCount: 2))
    }

    func testStepReachedStep2BoundaryOnAcceptedDraftCount() {
        // step 2 reached iff acceptedDraftCount >= 2: 1 -> false (boundary just below),
        // 2 -> true (boundary exactly at), 3 -> true (comfortably above).
        XCTAssertFalse(inCheckpointSampledMTPStepReached(step: 2, acceptedDraftCount: 1))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 2, acceptedDraftCount: 2))
        XCTAssertTrue(inCheckpointSampledMTPStepReached(step: 2, acceptedDraftCount: 3))
    }

    // MARK: - inCheckpointSampledMTPSummarizeStepSigma

    func testSummarizeStepSigmaAtStepZeroMeansCoincideByInvariantNotCoincidence() throws {
        // Step 0 is reached by every block (see inCheckpointSampledMTPStepReached), so the
        // "conditioned on reached" filter keeps every sample -- conditionedMean and
        // unconditionalMean are computed over the IDENTICAL multiset of sigma values, and their
        // counts are identical too. This must hold structurally, not as a numeric fluke: it is
        // the direct consequence of step 0's reached predicate being unconditionally true.
        //
        // samples (sigma, acceptedDraftCount): (0.1, 0), (0.4, 1), (0.9, 2)
        // hand-computed mean = (0.1 + 0.4 + 0.9) / 3 = 1.4 / 3 = 0.466666...
        let samples: [InCheckpointSampledMTPStepSigmaSample] = [
            (sigma: 0.1, acceptedDraftCount: 0),
            (sigma: 0.4, acceptedDraftCount: 1),
            (sigma: 0.9, acceptedDraftCount: 2),
        ]

        let summary = inCheckpointSampledMTPSummarizeStepSigma(samples, stepIndex: 0)

        let unconditionalMean = try XCTUnwrap(summary.unconditionalMean)
        let conditionedMean = try XCTUnwrap(summary.conditionedMean)
        XCTAssertEqual(unconditionalMean, conditionedMean)  // identical multiset -> bit-identical, not just close
        XCTAssertEqual(summary.unconditionalCount, summary.conditionedCount)
        XCTAssertEqual(summary.unconditionalCount, 3)
        XCTAssertEqual(unconditionalMean, 0.466666666666667, accuracy: 1e-9)  // 1.4/3 is not exact in binary
    }

    func testSummarizeStepSigmaAtStepOneMeansDiffer() throws {
        // THE DISCRIMINATING TEST: blocks that rejected at step 0 (acceptedDraftCount == 0) carry
        // a deliberately different sigma from blocks that reached step 1, so pooling
        // unconditionally vs. conditioning on "reached step 1" must produce different numbers.
        // This is the test that would fail if someone "simplified" the two means into one.
        //
        // samples (sigma, acceptedDraftCount):
        //   (0.10, 0)  <- did NOT reach step 1 (acceptedDraftCount < 1)
        //   (0.80, 1)  <- reached step 1 (acceptedDraftCount >= 1)
        //   (0.90, 2)  <- reached step 1 (acceptedDraftCount >= 1)
        //
        // unconditionalMean pools all three: (0.10 + 0.80 + 0.90) / 3 = 1.8 / 3 = 0.6
        // conditionedMean pools only the reached two: (0.80 + 0.90) / 2 = 1.7 / 2 = 0.85
        // unconditionalCount = 3, conditionedCount = 2
        let samples: [InCheckpointSampledMTPStepSigmaSample] = [
            (sigma: 0.10, acceptedDraftCount: 0),
            (sigma: 0.80, acceptedDraftCount: 1),
            (sigma: 0.90, acceptedDraftCount: 2),
        ]

        let summary = inCheckpointSampledMTPSummarizeStepSigma(samples, stepIndex: 1)

        let unconditionalMean = try XCTUnwrap(summary.unconditionalMean)
        let conditionedMean = try XCTUnwrap(summary.conditionedMean)
        XCTAssertEqual(unconditionalMean, 0.6, accuracy: 1e-9)  // 0.10+0.80+0.90 not exact in binary
        XCTAssertEqual(conditionedMean, 0.85, accuracy: 1e-9)  // 0.80+0.90 not exact in binary
        XCTAssertNotEqual(unconditionalMean, conditionedMean)
        XCTAssertEqual(summary.unconditionalCount, 3)
        XCTAssertEqual(summary.conditionedCount, 2)
    }

    func testSummarizeStepSigmaConditionedMeanIsNilNotZeroWhenNoBlockReachedStep() throws {
        // Every block rejected at step 0 (acceptedDraftCount == 0 throughout), so NONE reach
        // step 1 -- conditionedMean must be nil, never a fabricated 0.0. A fabricated 0.0 would
        // understate depth decay by making "no data at this depth" look like "always rejected at
        // this depth", which is precisely the wrong direction for the accept/reject decision.
        //
        // samples (sigma, acceptedDraftCount): (0.1, 0), (0.5, 0), (0.9, 0)
        // unconditionalMean (still pools everything) = (0.1 + 0.5 + 0.9) / 3 = 1.5 / 3 = 0.5
        let samples: [InCheckpointSampledMTPStepSigmaSample] = [
            (sigma: 0.1, acceptedDraftCount: 0),
            (sigma: 0.5, acceptedDraftCount: 0),
            (sigma: 0.9, acceptedDraftCount: 0),
        ]

        let summary = inCheckpointSampledMTPSummarizeStepSigma(samples, stepIndex: 1)

        XCTAssertNil(summary.conditionedMean)
        XCTAssertEqual(summary.conditionedCount, 0)
        let unconditionalMean = try XCTUnwrap(summary.unconditionalMean)
        XCTAssertEqual(unconditionalMean, 0.5, accuracy: 1e-9)  // 0.1+0.5+0.9 not exact in binary
        XCTAssertEqual(summary.unconditionalCount, 3)
    }

    func testSummarizeStepSigmaOfEmptyInputIsNilForBothMeans() {
        let summary = inCheckpointSampledMTPSummarizeStepSigma([], stepIndex: 1)

        XCTAssertNil(summary.unconditionalMean)
        XCTAssertNil(summary.conditionedMean)
        XCTAssertEqual(summary.unconditionalCount, 0)
        XCTAssertEqual(summary.conditionedCount, 0)
    }

    // MARK: - inCheckpointSampledMTPWilsonInterval (Fix 5)
    //
    // Expected values below were computed independently with a standalone Python script (the
    // textbook Wilson score formula, z=1.959963985) -- NOT by calling the function under test --
    // and are reproduced here to 1e-9. See the function's doc comment for why Wilson (not Wald) is
    // the chosen interval.

    func testWilsonIntervalAtFiftyFiftyIsSymmetric() throws {
        // success=50, trial=100: independently computed (lower, upper) =
        // (0.4038315303442626, 0.5961684696557374).
        let interval = try XCTUnwrap(
            inCheckpointSampledMTPWilsonInterval(successCount: 50, trialCount: 100))

        XCTAssertEqual(interval.lower, 0.4038315303442626, accuracy: 1e-9)
        XCTAssertEqual(interval.upper, 0.5961684696557374, accuracy: 1e-9)
    }

    func testWilsonIntervalNearPredeclaredFloorBlockCount() throws {
        // success=175, trial=250 (the predeclared floor's minimum block count): independently
        // computed (lower, upper) = (0.6405184593037334, 0.7534282208952906).
        let interval = try XCTUnwrap(
            inCheckpointSampledMTPWilsonInterval(successCount: 175, trialCount: 250))

        XCTAssertEqual(interval.lower, 0.6405184593037334, accuracy: 1e-9)
        XCTAssertEqual(interval.upper, 0.7534282208952906, accuracy: 1e-9)
    }

    func testWilsonIntervalClampsAtZeroAndOneBoundaries() throws {
        // success=0, trial=10: independently computed (lower, upper) = (0.0, 0.27753279995699603)
        // -- the raw lower bound is already exactly 0 here, so the `Swift.max(0, ...)` clamp is
        // inert for this case but the boundary is still exercised.
        let allRejected = try XCTUnwrap(
            inCheckpointSampledMTPWilsonInterval(successCount: 0, trialCount: 10))
        XCTAssertEqual(allRejected.lower, 0.0, accuracy: 1e-9)
        XCTAssertEqual(allRejected.upper, 0.27753279995699603, accuracy: 1e-9)

        // success=10, trial=10: independently computed raw upper = 1.0 exactly (the
        // `Swift.min(1, ...)` clamp is likewise inert here, not the thing under test -- this case
        // exercises the all-accepted boundary itself).
        let allAccepted = try XCTUnwrap(
            inCheckpointSampledMTPWilsonInterval(successCount: 10, trialCount: 10))
        XCTAssertEqual(allAccepted.lower, 0.7224672000430039, accuracy: 1e-9)
        XCTAssertEqual(allAccepted.upper, 1.0, accuracy: 1e-9)
    }

    func testWilsonIntervalIsNilForZeroTrials() {
        XCTAssertNil(inCheckpointSampledMTPWilsonInterval(successCount: 0, trialCount: 0))
    }

    // MARK: - inCheckpointSampledMTPAgreementCheck (Fix 5)

    func testAgreementCheckAgreesWhenSigmaFallsInsideWilsonInterval() {
        // success=50, trial=100 -> interval (0.4038315303442626, 0.5961684696557374) (hand-derived
        // above). sigmaMean=0.5 falls inside -> AGREE.
        let check = inCheckpointSampledMTPAgreementCheck(
            stepIndex1Based: 1, acceptedCount: 50, reachedCount: 100, sigmaMean: 0.5)

        XCTAssertEqual(check.verdict, .agree)
        XCTAssertEqual(try XCTUnwrap(check.empiricalAcceptanceRate), 0.5, accuracy: 1e-9)
    }

    func testAgreementCheckDisagreesWhenSigmaFallsOutsideWilsonInterval() {
        // THE DISCRIMINATING CASE this fix exists for. success=175, trial=250 -> interval
        // (0.6405184593037334, 0.7534282208952906) (hand-derived above). sigmaMean=0.9 is well
        // outside the upper bound -> DISAGREE. This is the shape of a real mispairing defect: a
        // plausible-looking empirical ratio next to an incompatible independent cross-check.
        let check = inCheckpointSampledMTPAgreementCheck(
            stepIndex1Based: 1, acceptedCount: 175, reachedCount: 250, sigmaMean: 0.9)

        XCTAssertEqual(check.verdict, .disagree)
    }

    func testAgreementCheckIsNoDataNotDisagreeWhenStepUnreached() {
        // reachedCount=0 -- no empirical trials at all. Must be NO_DATA, never fabricated as
        // AGREE or DISAGREE.
        let check = inCheckpointSampledMTPAgreementCheck(
            stepIndex1Based: 2, acceptedCount: 0, reachedCount: 0, sigmaMean: nil)

        XCTAssertEqual(check.verdict, .noData)
        XCTAssertNil(check.empiricalAcceptanceRate)
        XCTAssertNil(check.wilsonLower)
        XCTAssertNil(check.wilsonUpper)
    }

    // MARK: - Fix 6 integration test: the real per-block call sequence through the measuring
    // provider (the regression test whose absence let the Fix 1 defect ship).
    //
    // Drives `InCheckpointMeasuringSampledMTPBlockRuntimeProvider` through TWO blocks, reproducing
    // the REAL call sequence verified at source (see the file-header comment above
    // `InCheckpointSampledMTPTeeingLogitSampler` in `InCheckpointSampledMTPAcceptanceCLI.swift`):
    //   1. ONE seed `sample(logits:)` call before the first block (`prepareDrafterState`).
    //   2. Per block: ONE draft-loop `sample(logits:)` call (BEFORE `decide()`), then `decide()`,
    //      then ONE commit-seed `sample(logits:)` call (AFTER `decide()` returns -- this becomes
    //      the FIRST row of the NEXT block's queue).
    // So at each `decide()` call the queue holds EXACTLY `needed` (== numDraft == 2) rows,
    // leading-aligned with `proposedTokens`. This is the exact shape `expectedCapturedCount =
    // needed + 1` (the pre-fix defect) could never satisfy -- verified by temporarily reverting
    // Fix 1 locally and re-running this test: it fails with
    // `draftLogitCaptureShapeMismatch(expectedRows: 3, observedRows: 2)` thrown out of the FIRST
    // `decide()` call, matching the live-run defect exactly (see the PR/report for the exact
    // captured failure text).
    //
    // Note: `InCheckpointMeasuringSampledMTPBlockRuntimeProvider` and
    // `InCheckpointSampledMTPTeeingLogitSampler` are declared at top level in
    // `InCheckpointSampledMTPAcceptanceCLI.swift`, OUTSIDE the `#if DEBUG / #else / #endif` guard
    // that wraps only `runInCheckpointSampledMTPAcceptance`'s body (confirmed by grepping the file
    // for `#if`: the only occurrence brackets that one function's implementation) -- so nothing
    // needed to move out of `#if DEBUG` for this test to reach them; they already compile in this
    // DEBUG test target via `@testable import fastmlx_harness`.
    func testTwoBlockRealCallSequenceSucceedsWithCorrectlyPairedSigma() throws {
        let inner = StubSampledMTPBlockRuntimeProvider(acceptedDraftCountsToReturn: [1, 2])
        // Identity sampling parameters: this test's expected sigma values are
        // hand-computed from the PLAIN softmax, so the measuring provider must
        // predict the untruncated overlap. These are stated explicitly rather
        // than defaulted, so a future truncating configuration cannot silently
        // inherit an expectation that is only valid at identity.
        let provider = InCheckpointMeasuringSampledMTPBlockRuntimeProvider(
            inner: inner,
            targetTemperature: 1, targetTopP: 1, targetTopK: 0, targetMinP: 0)

        // Hand-computed logits/softmax/sigma -- see the assertions below for the arithmetic.
        let flatZero = MLXArray([Float(0), Float(0)])  // softmax [0.5, 0.5]
        let flatLog3 = MLXArray([Float(0), Float(log(3.0))])  // softmax [0.25, 0.75]
        let flatLog9 = MLXArray([Float(0), Float(log(9.0))])  // softmax [0.1, 0.9]

        // --- prepareDrafterState's ONE seed call, before any block ---
        _ = provider.proposalSampler.sample(logits: flatZero)  // seed_0

        // --- block 1 ---
        _ = provider.proposalSampler.sample(logits: flatLog3)  // draft_1 (draft-loop, BEFORE decide)
        let decision1 = try provider.decide(
            proposedTokens: [10, 11],
            targetLogits: [flatZero, flatZero],
            bonusTargetLogits: flatZero)
        XCTAssertEqual(decision1.acceptedDraftCount, 1)
        _ = provider.proposalSampler.sample(logits: flatZero)  // commit seed_1 (AFTER decide)

        // --- block 2 ---
        _ = provider.proposalSampler.sample(logits: flatLog9)  // draft_2 (draft-loop, BEFORE decide)
        let decision2 = try provider.decide(
            proposedTokens: [20, 21],
            targetLogits: [flatZero, flatZero],
            bonusTargetLogits: flatZero)
        XCTAssertEqual(decision2.acceptedDraftCount, 2)
        _ = provider.proposalSampler.sample(logits: flatZero)  // commit seed_2 (unused by any decide() here)

        // Both `decide()` calls above SUCCEEDED (this is the regression assertion for Fix 1: under
        // `expectedCapturedCount = needed + 1`, the first call above throws
        // `draftLogitCaptureShapeMismatch(expectedRows: 3, observedRows: 2)` and this test fails
        // before reaching here).
        XCTAssertEqual(provider.blockOutcomes.count, 2)
        XCTAssertEqual(provider.blockOutcomes[0].proposedCount, 2)
        XCTAssertEqual(provider.blockOutcomes[0].acceptedDraftCount, 1)
        XCTAssertEqual(provider.blockOutcomes[1].proposedCount, 2)
        XCTAssertEqual(provider.blockOutcomes[1].acceptedDraftCount, 2)

        // Step 0 (index 0): both blocks pair target=[0,0] (softmax [0.5,0.5]) against a draft row
        // that is ALSO [0,0] (softmax [0.5,0.5]) -- identical distributions, so
        // sigma = min(0.5,0.5)+min(0.5,0.5) = 1.0 exactly, for both blocks.
        let step0Samples = try XCTUnwrap(provider.sigmaMinPQByStep[0])
        XCTAssertEqual(step0Samples.count, 2)
        XCTAssertEqual(step0Samples[0].sigma, 1.0, accuracy: 1e-9)
        XCTAssertEqual(step0Samples[1].sigma, 1.0, accuracy: 1e-9)

        // Step 1 (index 1): block 1's draft row is flatLog3, softmax([0, ln3]) = [0.25, 0.75]
        // (max=ln3: exp(0-ln3)=1/3, exp(ln3-ln3)=1, sum=4/3, probs=[(1/3)/(4/3),1/(4/3)]=[.25,.75]),
        // against target=[0,0] (softmax [0.5,0.5]):
        //   sigma = min(0.5,0.25) + min(0.5,0.75) = 0.25 + 0.5 = 0.75
        // block 2's draft row is flatLog9, softmax([0, ln9]) = [0.1, 0.9]
        // (max=ln9: exp(0-ln9)=1/9, exp(ln9-ln9)=1, sum=10/9, probs=[(1/9)/(10/9),1/(10/9)]=[.1,.9]),
        // against the SAME target=[0,0]:
        //   sigma = min(0.5,0.1) + min(0.5,0.9) = 0.1 + 0.5 = 0.6
        // These two values are DELIBERATELY different so a row-pairing bug (e.g. block 2 reading
        // block 1's draft row, or the seed/draft rows swapped) would be caught here. Accuracy is
        // 1e-6, not 1e-9, here (unlike the pure-Double tests above): `flatLog3`/`flatLog9` round
        // `log(3.0)`/`log(9.0)` through `MLXArray`'s Float32 storage before this file's softmax
        // (which itself operates on `Double`) ever sees them, so ~1e-7-relative Float32 rounding is
        // real and expected, not a bug -- measured deviation was ~3.7e-9 absolute, well inside this
        // margin.
        let step1Samples = try XCTUnwrap(provider.sigmaMinPQByStep[1])
        XCTAssertEqual(step1Samples.count, 2)
        XCTAssertEqual(step1Samples[0].sigma, 0.75, accuracy: 1e-6)
        XCTAssertEqual(step1Samples[1].sigma, 0.6, accuracy: 1e-6)

        // decideDurationsSeconds grew by exactly one real wall-clock sample per successful
        // `decide()` call -- count only (a scalar), never the timing values themselves.
        XCTAssertEqual(provider.decideDurationsSeconds.count, 2)
    }
}

// MARK: - Fix 6 test doubles
//
// `StubInnerLogitSampler` and `StubSampledMTPBlockRuntimeProvider` conform to the SAME public
// protocols (`LogitSampler`, `SampledMTPBlockRuntimeDeciding`) the real vendor iterator and the
// real seeded/nondeterministic providers use (`MLXLMCommon`), so
// `InCheckpointMeasuringSampledMTPBlockRuntimeProvider` is driven through its real interface, not a
// bespoke test-only seam.

private final class StubInnerLogitSampler: LogitSampler {
    func sample(logits: MLXArray) -> MLXArray { MLXArray([Int32(0)]) }
}

private final class StubSampledMTPBlockRuntimeProvider: SampledMTPBlockRuntimeDeciding {
    let proposalSampler: any LogitSampler = StubInnerLogitSampler()
    /// One entry per `decide()` call, in call order -- the `acceptedDraftCount` this stub returns
    /// for that call.
    private var acceptedDraftCountsToReturn: [Int]
    private var callIndex = 0

    init(acceptedDraftCountsToReturn: [Int]) {
        self.acceptedDraftCountsToReturn = acceptedDraftCountsToReturn
    }

    func supports(parameters: GenerateParameters) -> Bool { true }

    func decide(
        proposedTokens: [Int], targetLogits: [MLXArray], bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        let accepted = acceptedDraftCountsToReturn[callIndex]
        callIndex += 1
        var outputTokens = Array(proposedTokens.prefix(accepted))
        outputTokens.append(999)  // residual/bonus placeholder token; value is not asserted on
        return SampledMTPBlockRuntimeDecision(outputTokens: outputTokens, acceptedDraftCount: accepted)
    }
}
