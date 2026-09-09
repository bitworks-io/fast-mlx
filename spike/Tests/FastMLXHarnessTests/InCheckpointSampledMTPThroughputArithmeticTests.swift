import XCTest

@testable import fastmlx_harness

/// Independent hand-computed checks of the PURE arithmetic in
/// `InCheckpointSampledMTPThroughputCLI.swift` -- tok/s, median ratio, the predeclared band
/// classification, and each unconditional control's boundary. Every expected value here is worked
/// out by hand in the comment beside the assertion, never by calling the function under test. No
/// model load, no network, no MLX.
final class InCheckpointSampledMTPThroughputArithmeticTests: XCTestCase {

    // MARK: - inCheckpointSampledMTPThroughputTokensPerSecond

    func testTokensPerSecondUsesEmittedMinusOneNumerator() {
        // 257 tokens emitted (256 decoded after the excluded-from-timing first token) over 10s:
        // (257 - 1) / 10 = 25.6 exactly.
        let rate = inCheckpointSampledMTPThroughputTokensPerSecond(
            emittedTokenCount: 257, elapsedSeconds: 10)

        XCTAssertEqual(rate, 25.6)
    }

    func testTokensPerSecondIsNilWhenFewerThanTwoTokensEmitted() {
        // Nothing was ever timed (the first token is excluded from the numerator by definition).
        XCTAssertNil(
            inCheckpointSampledMTPThroughputTokensPerSecond(emittedTokenCount: 1, elapsedSeconds: 5))
        XCTAssertNil(
            inCheckpointSampledMTPThroughputTokensPerSecond(emittedTokenCount: 0, elapsedSeconds: 5))
    }

    func testTokensPerSecondIsNilForNonPositiveElapsedSeconds() {
        // A non-positive elapsed window is a clock/ordering defect, not a real zero-cost decode --
        // must not be fabricated as +infinity or a large finite rate.
        XCTAssertNil(
            inCheckpointSampledMTPThroughputTokensPerSecond(emittedTokenCount: 10, elapsedSeconds: 0))
        XCTAssertNil(
            inCheckpointSampledMTPThroughputTokensPerSecond(emittedTokenCount: 10, elapsedSeconds: -1))
    }

    // MARK: - inCheckpointSampledMTPThroughputMedianRatio

    func testMedianRatioOfEmptyIsNil() {
        XCTAssertNil(inCheckpointSampledMTPThroughputMedianRatio([]))
    }

    func testMedianRatioOddCountIsTheMiddleSortedValue() throws {
        // sorted: [1.0, 1.2, 1.5, 1.8, 2.0] -- odd count (5), middle index 2 -> 1.5.
        let median = try XCTUnwrap(
            inCheckpointSampledMTPThroughputMedianRatio([2.0, 1.0, 1.5, 1.8, 1.2]))

        XCTAssertEqual(median, 1.5)
    }

    func testMedianRatioEvenCountAveragesTheTwoCentralValues() throws {
        // sorted: [1.0, 1.2, 1.6, 2.0] -- even count (4): average of index 1 (1.2) and index 2
        // (1.6) = (1.2 + 1.6) / 2 = 1.4 exactly.
        let median = try XCTUnwrap(
            inCheckpointSampledMTPThroughputMedianRatio([2.0, 1.0, 1.6, 1.2]))

        XCTAssertEqual(median, 1.4, accuracy: 1e-12)
    }

    func testMedianRatioOfSingleValueIsThatValue() throws {
        let median = try XCTUnwrap(inCheckpointSampledMTPThroughputMedianRatio([1.75]))

        XCTAssertEqual(median, 1.75)
    }

    // MARK: - inCheckpointSampledMTPThroughputBand
    //
    // Predeclared bands (docs/task-inbox/2026-09-09-sampled-mtp-direct-throughput-
    // PREDECLARATION.md, "Predeclared bands"):
    //   REJECT: median <= 1.10
    //   ACCEPT: median >= 1.30 AND every per-prompt ratio >= 1.10
    //   GATED:  everything else

    func testBandAcceptsExactlyAtBothBoundaries() {
        // median == 1.30 (the ACCEPT floor, inclusive) AND minPerPromptRatio == 1.10 (the
        // per-prompt floor, inclusive) -- both boundaries hit exactly.
        let band = inCheckpointSampledMTPThroughputBand(medianRatio: 1.30, minPerPromptRatio: 1.10)

        XCTAssertEqual(band, .accept)
    }

    func testBandIsGatedJustBelowTheAcceptMedianFloor() {
        // median = 1.2999999... (just under 1.30) with a healthy min ratio -- must NOT accept.
        let band = inCheckpointSampledMTPThroughputBand(
            medianRatio: 1.2999999999, minPerPromptRatio: 1.20)

        XCTAssertEqual(band, .gated)
    }

    func testBandRejectsExactlyAtTheRejectCeiling() {
        // median == 1.10 exactly -- REJECT (<=), regardless of what the min ratio is.
        let band = inCheckpointSampledMTPThroughputBand(medianRatio: 1.10, minPerPromptRatio: 1.10)

        XCTAssertEqual(band, .reject)
    }

    func testBandRejectsJustBelowTheRejectCeiling() {
        let band = inCheckpointSampledMTPThroughputBand(
            medianRatio: 1.0999999999, minPerPromptRatio: 1.05)

        XCTAssertEqual(band, .reject)
    }

    func testBandIsGatedNotAcceptWhenMedianClearsFloorButAnyPerPromptRatioIsBelowTheFloor() {
        // THE DISCRIMINATING CASE the per-prompt floor exists for: median comfortably clears 1.30
        // (a strong pooled result), but at least one individual prompt regressed below 1.10 --
        // "a median carried by two prompts while others regress is not a speedup a user
        // experiences" (predeclaration). This must be GATED, never ACCEPT.
        let band = inCheckpointSampledMTPThroughputBand(medianRatio: 1.50, minPerPromptRatio: 1.05)

        XCTAssertEqual(band, .gated)
    }

    func testBandIsGatedInTheOpenIntervalBetweenBoundaries() {
        // median strictly between 1.10 and 1.30, min ratio irrelevant to this case (healthy) --
        // the direction stays open but is not licensed.
        let band = inCheckpointSampledMTPThroughputBand(medianRatio: 1.20, minPerPromptRatio: 1.20)

        XCTAssertEqual(band, .gated)
    }

    // MARK: - C1: inCheckpointSampledMTPThroughputSpeculationEngaged

    func testSpeculationEngagedRequiresBothProposedCountPositiveAndNilPassthrough() {
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputSpeculationEngaged(
                proposedCount: 1, passthroughReason: nil))
        // proposedCount == 0 (boundary) -- must fail even with no passthrough reason.
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSpeculationEngaged(
                proposedCount: 0, passthroughReason: nil))
        // A non-nil passthrough reason fails the control even if proposedCount is positive (a
        // stream can propose tokens before going sticky mid-stream).
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSpeculationEngaged(
                proposedCount: 4, passthroughReason: "went sticky"))
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSpeculationEngaged(
                proposedCount: 0, passthroughReason: "drafter does not support this prompt input"))
    }

    // MARK: - C2: inCheckpointSampledMTPThroughputImpliedAcceptance /
    // inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses

    func testImpliedAcceptanceIsNilWhenNothingWasProposed() {
        XCTAssertNil(
            inCheckpointSampledMTPThroughputImpliedAcceptance(proposedCount: 0, acceptedCount: 0))
    }

    func testImpliedAcceptanceHandComputed() throws {
        // accepted=6879, proposed=10000 -> 0.6879 exactly.
        let observed = try XCTUnwrap(
            inCheckpointSampledMTPThroughputImpliedAcceptance(
                proposedCount: 10_000, acceptedCount: 6_879))

        XCTAssertEqual(observed, 0.6879, accuracy: 1e-12)
    }

    // Boundary tests below use `expected`/`toleranceAbsolute` values that are EXACTLY
    // representable in binary floating point (halves and quarters), and derive `observed` from
    // them by addition/subtraction -- NOT the real predeclared constants (`0.6879`/`0.05`, hand-
    // computed against in `testImpliedAcceptanceHandComputed` above). Decimal fractions like
    // `0.6879 + 0.05` do not round-trip back to a double bit-identical to the literal `0.05`
    // (floating-point addition/subtraction is not exactly invertible for non-representable
    // fractions), which would make an "exactly at the boundary" test flaky on the specific
    // constant rather than on the `<=` inclusivity this test actually targets. Representable
    // values isolate that.
    func testImpliedAcceptanceControlPassesExactlyAtPlusTolerance() {
        let expected = 0.5
        let toleranceAbsolute = 0.25
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(
                observed: 0.75, expected: expected, toleranceAbsolute: toleranceAbsolute))
    }

    func testImpliedAcceptanceControlFailsJustAbovePlusTolerance() {
        let expected = 0.5
        let toleranceAbsolute = 0.25
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(
                observed: 0.7501, expected: expected, toleranceAbsolute: toleranceAbsolute))
    }

    func testImpliedAcceptanceControlPassesExactlyAtMinusTolerance() {
        let expected = 0.5
        let toleranceAbsolute = 0.25
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(
                observed: 0.25, expected: expected, toleranceAbsolute: toleranceAbsolute))
    }

    func testImpliedAcceptanceControlFailsJustBelowMinusTolerance() {
        let expected = 0.5
        let toleranceAbsolute = 0.25
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(
                observed: 0.2499, expected: expected, toleranceAbsolute: toleranceAbsolute))
    }

    // MARK: - C3: inCheckpointSampledMTPThroughputScalarControlPasses
    //
    // Same construction rationale as the C2 boundary tests above: `expected`/`toleranceFraction`
    // are exactly representable in binary (100 and a quarter), so `expected * (1 +/-
    // toleranceFraction)` is exact and the boundary comparison tests `<=` inclusivity rather than
    // float rounding on the real predeclared constants.
    func testScalarControlPassesExactlyAtPlusTwentyFivePercent() {
        let expected = 100.0
        let toleranceFraction = 0.25
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputScalarControlPasses(
                observedMeanTokensPerSecond: 125.0, expected: expected,
                toleranceFraction: toleranceFraction))
    }

    func testScalarControlFailsJustAbovePlusTwentyFivePercent() {
        let expected = 100.0
        let toleranceFraction = 0.25
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputScalarControlPasses(
                observedMeanTokensPerSecond: 125.001, expected: expected,
                toleranceFraction: toleranceFraction))
    }

    func testScalarControlPassesExactlyAtMinusTwentyFivePercent() {
        let expected = 100.0
        let toleranceFraction = 0.25
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputScalarControlPasses(
                observedMeanTokensPerSecond: 75.0, expected: expected,
                toleranceFraction: toleranceFraction))
    }

    func testScalarControlFailsJustBelowMinusTwentyFivePercent() {
        let expected = 100.0
        let toleranceFraction = 0.25
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputScalarControlPasses(
                observedMeanTokensPerSecond: 74.999, expected: expected,
                toleranceFraction: toleranceFraction))
    }

    // MARK: - C4: inCheckpointSampledMTPThroughputSampleSizeControlPasses

    func testSampleSizeControlPassesExactlyAtBothFloors() {
        XCTAssertTrue(
            inCheckpointSampledMTPThroughputSampleSizeControlPasses(
                promptCount: 8, maxTokens: 256, requiredPromptCount: 8, requiredMaxTokens: 256))
    }

    func testSampleSizeControlFailsJustBelowThePromptCountFloor() {
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSampleSizeControlPasses(
                promptCount: 7, maxTokens: 256, requiredPromptCount: 8, requiredMaxTokens: 256))
    }

    func testSampleSizeControlFailsJustBelowTheMaxTokensFloor() {
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSampleSizeControlPasses(
                promptCount: 8, maxTokens: 255, requiredPromptCount: 8, requiredMaxTokens: 256))
    }

    func testSampleSizeControlFailsWhenBothAreBelowFloor() {
        XCTAssertFalse(
            inCheckpointSampledMTPThroughputSampleSizeControlPasses(
                promptCount: 1, maxTokens: 8, requiredPromptCount: 8, requiredMaxTokens: 256))
    }

    // MARK: - inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst

    func testSpeculativeArmRunsFirstOnEvenPromptIndicesOnly() {
        XCTAssertTrue(inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(promptIndex: 0))
        XCTAssertFalse(inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(promptIndex: 1))
        XCTAssertTrue(inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(promptIndex: 2))
        XCTAssertFalse(inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(promptIndex: 3))
    }

    // MARK: - inCheckpointSampledMTPThroughputSummarize

    func testSummarizeOfEmptyIsNil() {
        XCTAssertNil(inCheckpointSampledMTPThroughputSummarize([]))
    }

    func testSummarizeHandComputedOverFourValues() throws {
        // values: [20.0, 24.0, 28.0, 32.0] -- sum=104, count=4, mean=26.0 exactly.
        let summary = try XCTUnwrap(
            inCheckpointSampledMTPThroughputSummarize([20.0, 24.0, 28.0, 32.0]))

        XCTAssertEqual(summary.mean, 26.0)
        XCTAssertEqual(summary.min, 20.0)
        XCTAssertEqual(summary.max, 32.0)
        XCTAssertEqual(summary.count, 4)
    }
}
