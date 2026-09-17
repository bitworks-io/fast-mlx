import Foundation
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

/// Covers `MLXDecoder`'s `LogprobDecoding` conformance (`prefillWithLogprob`/`stepWithLogprob`):
/// acceptance criteria (a) the reported logprob and top-N candidates are the REAL `logSoftmax` of
/// the raw (pre-processor) logits, sorted strictly descending, finite-only, at most `topN` long;
/// and (b) turning logprobs on changes nothing about which tokens get SELECTED — same seed/
/// temperature/top-p must produce the identical sampled sequence as the no-logprobs path. Reuses
/// this file's `AlwaysThrowingEvaluationModel`-style minimal `LanguageModel` fixture idiom.
final class MLXDecoderLogprobComputationTests: XCTestCase {
    /// Deliberately distinct, non-monotonic values (including a negative one) so argmax, the top-3
    /// ranking, and the manual `logSoftmax` oracle below all pin down a UNIQUE, unambiguous answer.
    private let fixedLogits: [Float] = [0.2, 3.0, -5.0, 1.0, 4.0, 2.5]

    /// Pure-Swift oracle for `logSoftmax`, computed independently of MLX/MLXNN so this test does
    /// not validate the production computation against itself.
    private func manualLogSoftmax(_ logits: [Float]) -> [Float] {
        let maxValue = logits.max()!
        let shifted = logits.map { $0 - maxValue }
        let sumExp = shifted.reduce(Float(0)) { $0 + Foundation.exp($1) }
        let logSumExp = Foundation.log(sumExp)
        return shifted.map { $0 - logSumExp }
    }

    func testPrefillWithLogprobMatchesManualLogSoftmaxAndSortsTopNDescending() throws {
        var decoder = MLXDecoder(
            model: FixedLogitsModel(logits: fixedLogits),
            cache: [KVCacheSimple()])

        let (token, logprob) = try decoder.prefillWithLogprob([1], topN: 3)
        let expected = manualLogSoftmax(fixedLogits)

        // The default sampler is `ArgMaxSampler`: it selects the largest raw logit — index 4 (4.0).
        XCTAssertEqual(token, 4)
        XCTAssertEqual(logprob.tokenID, token)
        XCTAssertEqual(Double(logprob.logprob), Double(expected[4]), accuracy: 1e-5)

        XCTAssertLessThanOrEqual(logprob.top.count, 3)
        XCTAssertTrue(logprob.top.allSatisfy { $0.logprob.isFinite })
        for i in 1..<logprob.top.count {
            XCTAssertGreaterThan(logprob.top[i - 1].logprob, logprob.top[i].logprob)
        }
        // The top 3 raw logits are indices 4 (4.0), 1 (3.0), 5 (2.5), in that order.
        XCTAssertEqual(logprob.top.map(\.tokenID), [4, 1, 5])
        for candidate in logprob.top {
            XCTAssertEqual(
                Double(candidate.logprob), Double(expected[candidate.tokenID]), accuracy: 1e-5)
        }
    }

    func testStepWithLogprobMatchesManualLogSoftmax() throws {
        var decoder = MLXDecoder(
            model: FixedLogitsModel(logits: fixedLogits),
            cache: [KVCacheSimple()])
        let first = try decoder.prefill([1])

        let (token, logprob) = try decoder.stepWithLogprob(last: first, topN: 2)

        let expected = manualLogSoftmax(fixedLogits)
        XCTAssertEqual(token, 4)
        XCTAssertEqual(Double(logprob.logprob), Double(expected[4]), accuracy: 1e-5)
        XCTAssertEqual(logprob.top.map(\.tokenID), [4, 1])
    }

    func testTopNZeroReturnsNoCandidatesButStillReportsSampledLogprob() throws {
        var decoder = MLXDecoder(
            model: FixedLogitsModel(logits: fixedLogits),
            cache: [KVCacheSimple()])

        let (token, logprob) = try decoder.prefillWithLogprob([1], topN: 0)

        XCTAssertEqual(token, 4)
        XCTAssertTrue(logprob.top.isEmpty)
        let expected = manualLogSoftmax(fixedLogits)
        XCTAssertEqual(Double(logprob.logprob), Double(expected[4]), accuracy: 1e-5)
    }

    /// Acceptance (b): logprobs on vs off, same seed/temperature/top-p, must select the IDENTICAL
    /// token sequence. `prefillWithLogprob`/`stepWithLogprob` SHARE `selectSampleAndAdvance` with
    /// `prefill`/`step` (see `MLXDecoder.swift`), so this is a regression guard against a future
    /// edit that duplicates instead of shares that code and drifts RNG consumption.
    func testLogprobsOnVsOffProduceIdenticalSampledTokenSequence() throws {
        let sampling = DecoderSampling.sampled(
            temperature: 1.0, topP: 1.0, topK: nil, minP: nil, seed: 42)

        var withoutLogprobs = MLXDecoder(
            model: FixedLogitsModel(logits: fixedLogits),
            cache: [KVCacheSimple()])
        withoutLogprobs.setSampling(sampling)
        var withoutTokens: [Int] = []
        withoutTokens.append(try withoutLogprobs.prefill([1]))
        withoutTokens.append(try withoutLogprobs.step(last: withoutTokens[0]))
        withoutTokens.append(try withoutLogprobs.step(last: withoutTokens[1]))

        var withLogprobs = MLXDecoder(
            model: FixedLogitsModel(logits: fixedLogits),
            cache: [KVCacheSimple()])
        withLogprobs.setSampling(sampling)
        var withTokens: [Int] = []
        let first = try withLogprobs.prefillWithLogprob([1], topN: 3)
        withTokens.append(first.token)
        let second = try withLogprobs.stepWithLogprob(last: withTokens[0], topN: 3)
        withTokens.append(second.token)
        let third = try withLogprobs.stepWithLogprob(last: withTokens[1], topN: 3)
        withTokens.append(third.token)

        XCTAssertEqual(withoutTokens, withTokens)
    }
}

/// A `LanguageModel` fixture returning the SAME fixed per-token logit vector at every position and
/// every call, ignoring input entirely — deliberately simpler than a real forward pass so a test
/// can assert an EXACT `logSoftmax` value against hand-computed Swift floating point, and so two
/// independently constructed decoders driven through different call sequences (`prefill`/`step` vs
/// `prefillWithLogprob`/`stepWithLogprob`) see byte-identical logits at every step.
private final class FixedLogitsModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = [1]
    let logits: [Float]

    init(logits: [Float]) {
        self.logits = logits
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
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
