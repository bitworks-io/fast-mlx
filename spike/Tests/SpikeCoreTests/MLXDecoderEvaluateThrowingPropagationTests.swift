import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

/// Sentinel error the fixture model throws from `evaluateThrowing`. Distinct from
/// `preconditionFailure`/`fatalError` (which still abort the process for genuine caller bugs —
/// see `MLXDecoder.step`'s "called before prefill" guard) — this is what a MODEL EVALUATION
/// failure (e.g. `qwen4_exp`'s cache/PLE-ownership validation) is supposed to look like once it
/// reaches a caller: an ordinary, catchable Swift error.
private struct FixtureThrowingEvaluationError: Error, Equatable {}

/// Minimal `LanguageModel` overriding ONLY `evaluateThrowing` — mirrors the minimal-fake shape of
/// `CacheFactorySpyModel` (`MLXDecoderCacheFactoryTests.swift`): the rest of `LanguageModel` has
/// default implementations via protocol extension, and nothing here ever needs a real forward
/// pass. Deliberately does NOT implement `callAsFunction(_:cache:)`: the protocol extension's
/// default for that overload is `fatalError("callAsFunction(inputs:cache:) not implemented ...")`
/// (`LanguageModel.swift`), so if `MLXDecoder` ever called the non-throwing entry point instead of
/// `evaluateThrowing`, this fixture crashes the test process rather than silently passing. That
/// crash IS the mutation-proof signal for acceptance criterion (d).
private final class AlwaysThrowingEvaluationModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = [1]

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        throw FixtureThrowingEvaluationError()
    }
}

/// Proves the increment-B wiring: `MLXDecoder.prefill`/`.step` call `LanguageModel.evaluateThrowing`
/// (the increment-A throwing seam), not the non-throwing `callAsFunction`, so a model whose own
/// step-time validation fails reports an ordinary Swift error a caller can catch — instead of
/// aborting the process via `preconditionFailure`/`fatalError` inside `callAsFunction` (the defect
/// this two-part fix exists to close: the first real-weight `qwen4_exp` run is exactly when that
/// validation is most likely to fail, and today that kills the process after a multi-minute weight
/// load, discarding the run).
final class MLXDecoderEvaluateThrowingPropagationTests: XCTestCase {
    func testPrefillPropagatesModelEvaluationFailureInsteadOfAborting() {
        var decoder = MLXDecoder(
            model: AlwaysThrowingEvaluationModel(),
            cache: [KVCacheSimple()])

        XCTAssertThrowsError(try decoder.prefill([1, 2, 3])) {
            XCTAssertEqual($0 as? FixtureThrowingEvaluationError, FixtureThrowingEvaluationError())
        }
    }

    /// `step` runs its own `evaluateThrowing` call independently of `prefill` (the submit-first
    /// lookahead forward): cover it separately so a regression that only wires one of the two
    /// call sites is still caught.
    func testStepPropagatesModelEvaluationFailureInsteadOfAborting() throws {
        // `step` reads `pendingLogits`, which only a successful `prefill` populates, so this uses
        // a model that succeeds its first two `evaluateThrowing` calls (prefill's chunk forward
        // plus its submit-first lookahead forward) to reach a real `step` call, then fails on
        // every call after — exercising `step`'s own throwing seam independently of `prefill`'s.
        var decoder = MLXDecoder(
            model: SucceedsOnceThenThrowsEvaluationModel(),
            cache: [KVCacheSimple()])

        _ = try decoder.prefill([1])

        XCTAssertThrowsError(try decoder.step(last: 0)) {
            XCTAssertEqual($0 as? FixtureThrowingEvaluationError, FixtureThrowingEvaluationError())
        }
    }
}

/// Succeeds its first two `evaluateThrowing` calls (prefill's chunk forward + submit-first
/// lookahead forward), then throws on every call after — lets a test drive `step` (which needs a
/// populated `pendingLogits` from a real `prefill`) into the throwing path without ever reaching a
/// real MLX forward pass.
private final class SucceedsOnceThenThrowsEvaluationModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    let kvHeads: [Int] = [1]
    private var callCount = 0

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        callCount += 1
        guard callCount <= 2 else {
            throw FixtureThrowingEvaluationError()
        }
        let vocab = 4
        let logits = MLXArray.zeros([1, input.tokens.dim(1), vocab])
        return LMOutput(logits: logits)
    }
}
