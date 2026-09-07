// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// MARK: - Synthetic mock for plain-TokenIterator plumbing

/// Test-only error used to assert `nextThrowing()` propagates the specific
/// error surfaced by `evaluateThrowing`, not merely "some error".
private struct TokenIteratorThrowingTestError: Error, Equatable {}

/// `LanguageModel` whose `callAsFunction` behaves normally (never aborts) but
/// whose `evaluateThrowing` override throws
/// `TokenIteratorThrowingTestError` starting at the configured 1-based call
/// index. Stands in for a target whose internal validation fails: on the
/// ordinary `callAsFunction` route such a model has no way to report that
/// short of a process abort, but a caller that goes through
/// `evaluateThrowing` (via `TokenIterator.nextThrowing()`) gets a catchable
/// error instead. `throwFromCall == nil` disables the override's throwing
/// behavior entirely (every call succeeds), which is what makes this mock
/// double as the non-vacuity control for
/// ``testNextThrowingMatchesNextForNonThrowingTarget()``: the same call
/// sequence must produce the same tokens whether the iterator is driven
/// through `next()` or `nextThrowing()`.
///
/// `TokenIterator.prepare(input:windowSize:)` always calls the non-throwing
/// `step(previous:)`, which never calls `evaluateThrowing` -- only
/// `stepThrowing(previous:)` (driven by `nextThrowing()`) does. So this
/// model's `evaluateThrowingCallCount` only advances on `nextThrowing()`
/// calls, never on `next()` calls or on iterator construction.
private final class ThrowingTokenModel: Module, LanguageModel {
    var nextLogitTokens: [Int32]
    var perPositionIndex = 0
    var throwFromCall: Int?
    private(set) var evaluateThrowingCallCount = 0
    private(set) var callAsFunctionCallCount = 0

    init(nextLogitTokens: [Int32], throwFromCall: Int? = nil) {
        self.nextLogitTokens = nextLogitTokens
        self.throwFromCall = throwFromCall
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(positions: inputs.dim(-1))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callAsFunctionCallCount += 1
        return LMOutput(logits: makeLogits(positions: input.tokens.dim(-1)))
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        evaluateThrowingCallCount += 1
        if let throwFromCall, evaluateThrowingCallCount >= throwFromCall {
            throw TokenIteratorThrowingTestError()
        }
        return callAsFunction(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private func makeLogits(positions: Int) -> MLXArray {
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let tokIdx = perPositionIndex + i
            let tok = tokIdx < nextLogitTokens.count ? Int(nextLogitTokens[tokIdx]) : 0
            data[i * vocab + tok] = 100
        }
        perPositionIndex += positions
        return MLXArray(data, [1, positions, vocab])
    }
}

@Suite
struct TokenIteratorThrowingEntryPointTests {

    @Test
    func testNextThrowingPropagatesEvaluateThrowingErrorOnFirstDecodeStep() throws {
        // Prepare's own priming forward always goes through the non-throwing
        // `step`, so `throwFromCall: 1` cannot fire until the first
        // `nextThrowing()` call performs the first decode-time
        // `stepThrowing` -- production site Evaluate.swift's
        // `TokenIterator.stepThrowing(previous:)`.
        let model = ThrowingTokenModel(nextLogitTokens: [0, 0, 7, 8, 9, 10], throwFromCall: 1)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            model: model,
            parameters: GenerateParameters(maxTokens: 4))

        #expect(throws: TokenIteratorThrowingTestError.self) {
            _ = try iterator.nextThrowing()
        }
        #expect(model.evaluateThrowingCallCount == 1)
    }

    @Test
    func testNextThrowingMatchesNextForNonThrowingTarget() throws {
        // Non-vacuity control: `nextThrowing()` must reproduce exactly the
        // token sequence `next()` already produces for a target that never
        // fails validation. Without this control, a `nextThrowing()` that
        // always threw would still pass the propagation test above.
        let logits: [Int32] = [0, 0, 7, 8, 9, 10]
        let promptTokens = MLXArray([Int32(1), 2, 3])
        let parameters = GenerateParameters(maxTokens: 4)

        let modelForNext = ThrowingTokenModel(nextLogitTokens: logits, throwFromCall: nil)
        var iterViaNext = try TokenIterator(
            input: LMInput(tokens: promptTokens), model: modelForNext, parameters: parameters)
        var tokensViaNext = [Int]()
        while let token = iterViaNext.next() {
            tokensViaNext.append(token)
        }

        let modelForThrowing = ThrowingTokenModel(nextLogitTokens: logits, throwFromCall: nil)
        var iterViaThrowing = try TokenIterator(
            input: LMInput(tokens: promptTokens), model: modelForThrowing, parameters: parameters)
        var tokensViaThrowing = [Int]()
        while let token = try iterViaThrowing.nextThrowing() {
            tokensViaThrowing.append(token)
        }

        // Compare counts, then scalars -- never `Array ==` on a large
        // collection (a failing compare there dumps both arrays via a Myers
        // diff and has wedged a run for minutes).
        #expect(tokensViaNext.count == tokensViaThrowing.count)
        for (a, b) in zip(tokensViaNext, tokensViaThrowing) {
            #expect(a == b)
        }
    }

    @Test
    func testNextIsUnchangedOnSucceedingModel() throws {
        // Pins that `next()` still works exactly as before on a model that
        // never fails validation, and that its token sequence is the
        // expected, independently-computed one -- not just "some tokens".
        // The prompt (3 tokens) is consumed by prepare's single priming
        // forward, which yields the bonus token at prompt-local index 2
        // (`nextLogitTokens[2] == 7`); each subsequent `next()` call decodes
        // one more position (indices 3, 4, 5 -> 8, 9, 10).
        let model = ThrowingTokenModel(
            nextLogitTokens: [0, 0, 7, 8, 9, 10], throwFromCall: nil)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            model: model,
            parameters: GenerateParameters(maxTokens: 4))

        let expected: [Int] = [7, 8, 9, 10]
        var tokens = [Int]()
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens.count == expected.count)
        for (actual, wanted) in zip(tokens, expected) {
            #expect(actual == wanted)
        }
        // `next()` never touches `evaluateThrowing`.
        #expect(model.evaluateThrowingCallCount == 0)
        // One priming forward from `prepare` plus one decode forward per
        // returned token.
        #expect(model.callAsFunctionCallCount == expected.count + 1)
    }
}
