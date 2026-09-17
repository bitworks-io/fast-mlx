import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

/// A `LogitProcessor` that ALWAYS records a failure the first time `didSample` runs, regardless of
/// which token was sampled — a minimal double standing in for a real masking processor's failure
/// path (`SpikeServingAdapters.JSONObjectMaskingLogitProcessor` is exercised for the REAL grammar
/// separately in `JSONObjectMaskingLogitProcessorTests`; this file pins the SpikeCore half of the
/// contract: `MLXDecoder` must turn ANY conforming processor's recorded failure into a real thrown
/// error, independent of what specific grammar produced it).
private final class AlwaysFailingConstraintProcessor: LogitProcessor, ConstraintProcessorFailureReporting {
    struct Sentinel: Error, Equatable {}
    private(set) var recordedFailure: Error?
    private(set) var processCallCount = 0

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        processCallCount += 1
        return logits
    }

    func didSample(token: MLXArray) {
        recordedFailure = Sentinel()
    }
}

/// A `LogitProcessor` that records which logits it was asked to process, without altering them —
/// used to prove ORDERING (penalties before the constraint) rather than any failure behavior.
private final class RecordingPassthroughProcessor: LogitProcessor {
    private(set) var seenLogits: [[Float]] = []

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        seenLogits.append(logits.asArray(Float.self))
        return logits
    }

    func didSample(token: MLXArray) {}
}

/// Minimal `LanguageModel` fixture returning the SAME fixed logits at every position, ignoring
/// input — mirrors `MLXDecoderEvaluateThrowingPropagationTests.swift`'s `FixedLogitsModel` idiom
/// (kept private/local here rather than shared, matching that file's own no-shared-fixture style).
private final class FixedLogitsModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = [1]
    let logits: [Float]

    init(logits: [Float]) { self.logits = logits }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        let seqLen = input.tokens.dim(1)
        let row = MLXArray(logits)
        let full = MLXArray.zeros([1, seqLen, logits.count]) + row
        return LMOutput(logits: full)
    }
}

final class MLXDecoderResponseFormatConstraintFailureTests: XCTestCase {
    /// The core failure-surfacing contract (response-format design item #3): a constraint
    /// processor that records a failure from `didSample` turns `prefill`'s NEXT decode step into a
    /// real thrown Swift error — never a silent success — because `didSample` itself cannot throw
    /// (see `ConstraintProcessorFailureReporting`'s doc comment).
    func testRecordedConstraintFailureIsThrownFromPrefill() {
        var decoder = MLXDecoder(model: FixedLogitsModel(logits: [1, 2, 3, 4]), cache: [KVCacheSimple()])
        let failing = AlwaysFailingConstraintProcessor()
        decoder.setResponseFormatConstraint(failing)

        XCTAssertThrowsError(try decoder.prefill([1])) {
            XCTAssertEqual($0 as? AlwaysFailingConstraintProcessor.Sentinel, .init())
        }
    }

    /// Same failure contract on `step`, independent of `prefill`'s own throwing seam — mirrors
    /// `MLXDecoderEvaluateThrowingPropagationTests`'s existing prefill/step split for the same
    /// reason: a regression that only wires one call site should still be caught.
    func testRecordedConstraintFailureIsThrownFromStep() throws {
        var decoder = MLXDecoder(model: FixedLogitsModel(logits: [1, 2, 3, 4]), cache: [KVCacheSimple()])
        // First `prefill` with NO constraint configured, so it succeeds and populates
        // `pendingLogits`, then attach the failing constraint before the `step` under test.
        let first = try decoder.prefill([1])
        let failing = AlwaysFailingConstraintProcessor()
        decoder.setResponseFormatConstraint(failing)

        XCTAssertThrowsError(try decoder.step(last: first)) {
            XCTAssertEqual($0 as? AlwaysFailingConstraintProcessor.Sentinel, .init())
        }
    }

    /// `reset()` clears any previously configured constraint — the SAME clearing contract
    /// `setPenalties`/`setSampling` already have (see `MLXDecoder.reset()`), proven here by
    /// showing a decode AFTER `reset()` no longer invokes (or fails via) the old processor.
    func testResetClearsAnyPreviouslyConfiguredConstraint() throws {
        var decoder = MLXDecoder(model: FixedLogitsModel(logits: [1, 2, 3, 4]), cache: [KVCacheSimple()])
        let failing = AlwaysFailingConstraintProcessor()
        decoder.setResponseFormatConstraint(failing)
        decoder.reset()

        XCTAssertNoThrow(try decoder.prefill([1]))
    }

    /// Composition-order contract (response-format design: "penalties first, grammar mask last"):
    /// the constraint processor's `process(logits:)` must see the ALREADY-penalized logits, not the
    /// raw model output.
    func testConstraintProcessorReceivesLogitsAfterPenaltiesApplied() throws {
        var decoder = MLXDecoder(model: FixedLogitsModel(logits: [1, 2, 3, 4]), cache: [KVCacheSimple()])
        // `frequencyPenalty` alone is enough to make `setPenalties` build a real, non-nil
        // `PenaltyProcessor` — its exact numeric effect on these logits does not matter here, only
        // that SOME transformation happened before the constraint saw them.
        decoder.setPenalties(DecoderPenalties(frequencyPenalty: 1.0))
        let recorder = RecordingPassthroughProcessor()
        decoder.setResponseFormatConstraint(recorder)

        _ = try decoder.prefill([1])

        XCTAssertEqual(recorder.seenLogits.count, 1)
        // The raw model output is exactly `[1, 2, 3, 4]`; if the constraint were (incorrectly)
        // seeing PRE-penalty logits, this would be `[1, 2, 3, 4]` unchanged. A frequency penalty on
        // a prompt that already contains token id 1 changes at least that entry.
        XCTAssertNotEqual(recorder.seenLogits[0], [1, 2, 3, 4])
    }
}
