import XCTest
@testable import SpikeCore

final class InferenceActorTests: XCTestCase {
    func testStreamsExpectedGreedyTokensFromFakeModel() async throws {
        // Fake decoder returns a fixed script, proving the actor's loop/streaming
        // is correct independent of MLX. Real model wired in Task 5.
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 7, /*eos*/ 2], eos: 2))
        var got: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1, 2, 3], maxTokens: 10) {
            got.append(tok)
        }
        XCTAssertEqual(got, [5, 6, 7]) // eos consumed, not emitted
    }

    func testMaxTokensStopsGenerationBeforeEOS() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 7, 8, 9], eos: 2))
        var got: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 3) {
            got.append(tok)
        }
        XCTAssertEqual(got, [5, 6, 7])
    }

    func testResetForNewRunReplaysFromStartOfScript() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        var first: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 10) {
            first.append(tok)
        }
        try await actor.resetForNewRun()
        var second: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 10) {
            second.append(tok)
        }
        XCTAssertEqual(first, [5, 6])
        XCTAssertEqual(second, [5, 6]) // same script from the top, not continuing past eos
    }

    func testBoundedGenerationWaitsForConsumerBeforeAdvancingDecoder() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 7, 2], eos: 2))
        let observed = TokenRecorder()
        let releaseFirstToken = AsyncGate()

        let task = Task {
            try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2
            ) { token in
                await observed.append(token)
                if token == 5 {
                    try await releaseFirstToken.wait()
                }
                return .continueGeneration
            }
        }

        try await waitUntil { await observed.values == [5] }
        try await Task.sleep(for: .milliseconds(20))
        let blockedValues = await observed.values
        XCTAssertEqual(blockedValues, [5])

        await releaseFirstToken.open()
        let summary = try await task.value
        let completedValues = await observed.values
        XCTAssertEqual(completedValues, [5, 6, 7])
        XCTAssertEqual(
            summary,
            InferenceRunSummary(
                promptTokenCount: 1,
                generatedTokenCount: 3,
                finishReason: .endOfSequence))
    }

    func testBoundedGenerationCancellationResetsDecoderForRecovery() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        let observed = TokenRecorder()

        let cancelled = Task {
            try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2
            ) { token in
                await observed.append(token)
                try await Task.sleep(for: .seconds(60))
                return .continueGeneration
            }
        }

        try await waitUntil { await observed.values == [5] }
        cancelled.cancel()
        do {
            _ = try await cancelled.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // Expected.
        }

        let recovered = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2
        ) { token in
            await recovered.append(token)
            return .continueGeneration
        }
        let recoveredValues = await recovered.values
        XCTAssertEqual(recoveredValues, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    func testBoundedGenerationRejectsReentrantRequestWithoutMutatingActiveDecoder() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        let observed = TokenRecorder()
        let release = AsyncGate()

        let active = Task {
            try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2
            ) { token in
                await observed.append(token)
                if token == 5 {
                    try await release.wait()
                }
                return .continueGeneration
            }
        }
        try await waitUntil { await observed.values == [5] }

        do {
            _ = try await actor.generateBounded(
                promptTokens: [9],
                maxTokens: 1,
                eos: 2
            ) { _ in
                .continueGeneration
            }
            XCTFail("expected a reentrant-generation rejection")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .generationAlreadyActive)
        }

        await release.open()
        _ = try await active.value
        let completedValues = await observed.values
        XCTAssertEqual(completedValues, [5, 6])
    }

    func testLegacyStreamCannotMutateSuspendedBoundedGeneration() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        let observed = TokenRecorder()
        let release = AsyncGate()

        let active = Task {
            try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2
            ) { token in
                await observed.append(token)
                if token == 5 {
                    try await release.wait()
                }
                return .continueGeneration
            }
        }
        try await waitUntil { await observed.values == [5] }

        do {
            for try await _ in await actor.submit(
                promptTokens: [9],
                maxTokens: 1)
            {}
            XCTFail("expected a reentrant-generation rejection")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .generationAlreadyActive)
        }

        await release.open()
        _ = try await active.value
        let completedValues = await observed.values
        XCTAssertEqual(completedValues, [5, 6])
    }

    func testResetCannotMutateSuspendedBoundedGeneration() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        let observed = TokenRecorder()
        let release = AsyncGate()

        let active = Task {
            try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2
            ) { token in
                await observed.append(token)
                if token == 5 {
                    try await release.wait()
                }
                return .continueGeneration
            }
        }
        try await waitUntil { await observed.values == [5] }

        do {
            try await actor.resetForNewRun()
            XCTFail("expected a reentrant-reset rejection")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .generationAlreadyActive)
        }

        await release.open()
        _ = try await active.value
        let completedValues = await observed.values
        XCTAssertEqual(completedValues, [5, 6])
    }

    func testBoundedGenerationConsumerStopAndLengthAreTyped() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [5, 6, 7, 8], eos: 2))

        let stopped = try await actor.generateBounded(
            promptTokens: [1, 2],
            maxTokens: 4,
            eos: 2
        ) { _ in
            .stopGeneration
        }
        XCTAssertEqual(
            stopped,
            InferenceRunSummary(
                promptTokenCount: 2,
                generatedTokenCount: 1,
                finishReason: .consumerStop))

        let lengthLimited = try await actor.generateBounded(
            promptTokens: [1, 2],
            maxTokens: 2,
            eos: 2
        ) { _ in
            .continueGeneration
        }
        XCTAssertEqual(
            lengthLimited,
            InferenceRunSummary(
                promptTokenCount: 2,
                generatedTokenCount: 2,
                finishReason: .length))
    }

    func testBoundedGenerationStopsOnAnyConfiguredTokenID() async throws {
        let actor = InferenceActor(
            decoder: ScriptedDecoder(script: [4, 77, 5], eos: 99))
        let recorder = TokenRecorder()

        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 3,
            stopTokenIDs: [77, 99]
        ) { token in
            await recorder.append(token)
            return .continueGeneration
        }
        let values = await recorder.values

        XCTAssertEqual(values, [4])
        XCTAssertEqual(summary.generatedTokenCount, 1)
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    // MARK: - Logprobs plumbing (`logprobTopN`/`onLogprob`)

    /// A non-nil `logprobTopN` against a decoder that does not conform to `LogprobDecoding`
    /// (`ScriptedDecoder`) is refused with the specific error case, before any state mutation —
    /// mirroring `DecoderSamplingCapabilityTests`'s R1/R2 shape for the sampling/penalties guards.
    func testLogprobsRequestAgainstNonConformingDecoderThrowsLogprobsUnsupported() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                logprobTopN: 3
            ) { _ in .continueGeneration }
            XCTFail("expected logprobsUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .logprobsUnsupportedByDecoder)
        }
    }

    /// The SAME actor remains usable after a refused logprobs request — mirrors
    /// `DecoderSamplingCapabilityTests.testActorRemainsUsableAfterARefusedSampledRequest`: the
    /// guard must sit before `boundedGenerationActive = true`, never bricking the actor.
    func testActorRemainsUsableAfterARefusedLogprobsRequest() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1], maxTokens: 10, eos: 2, logprobTopN: 3
            ) { _ in .continueGeneration }
            XCTFail("expected logprobsUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .logprobsUnsupportedByDecoder)
        }

        let recorder = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1], maxTokens: 10, eos: 2
        ) { token in
            await recorder.append(token)
            return .continueGeneration
        }
        let values = await recorder.values
        XCTAssertEqual(values, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    /// A `logprobTopN: nil` request (every call site predating this parameter) against a decoder
    /// that does NOT conform to `LogprobDecoding` still admits and generates normally — the guard
    /// must gate on `logprobTopN` being non-nil, not merely on the decoder's capability, so the
    /// zero-cost/unchanged default path is never accidentally narrowed to logprob-capable decoders
    /// only.
    func testNilLogprobTopNAgainstNonConformingDecoderStillAdmits() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        let recorder = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1], maxTokens: 10, eos: 2, logprobTopN: nil
        ) { token in
            await recorder.append(token)
            return .continueGeneration
        }
        let values = await recorder.values
        XCTAssertEqual(values, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    /// A `LogprobDecoding`-conforming decoder admits a logprobs request, and `onLogprob` fires
    /// EXACTLY once per GENERATED token (never for the intercepted stop token), each call
    /// immediately BEFORE that token's own `consume` — the ordering the streaming carry-once rule
    /// downstream (`ServingCore`/`ServingNIO`, already tested end-to-end against a fake backend)
    /// depends on.
    func testConformingDecoderDeliversOnLogprobBeforeConsumeForEveryGeneratedToken() async throws {
        let actor = InferenceActor(
            decoder: ScriptedLogprobDecoder(script: [5, 6, 2], eos: 2))
        let events = EventRecorder()

        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            logprobTopN: 2,
            onLogprob: { logprob in
                await events.append("logprob:\(logprob.tokenID)")
            }
        ) { token in
            await events.append("consume:\(token)")
            return .continueGeneration
        }

        let values = await events.values
        XCTAssertEqual(
            values,
            ["logprob:5", "consume:5", "logprob:6", "consume:6"])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
        XCTAssertEqual(summary.generatedTokenCount, 2)
    }
}

private actor TokenRecorder {
    private var tokens: [Int] = []

    var values: [Int] {
        tokens
    }

    func append(_ token: Int) {
        tokens.append(token)
    }
}

/// Records ordered string events from both `onLogprob` and `consume`, so a test can assert their
/// RELATIVE ordering (not just that each fired) — see
/// `testConformingDecoderDeliversOnLogprobBeforeConsumeForEveryGeneratedToken`.
private actor EventRecorder {
    private var events: [String] = []

    var values: [String] {
        events
    }

    func append(_ event: String) {
        events.append(event)
    }
}

/// Test double conforming to `LogprobDecoding`: replays a fixed script like `ScriptedDecoder`
/// (never a real MLX computation — proving `InferenceActor`'s plumbing is correct independent of
/// MLX, mirroring `ScriptedDecoder`'s own doc comment), and reports a synthetic
/// `DecodedTokenLogprob` for each generated token so a test can assert `onLogprob` actually
/// receives the token this decoder just selected.
private struct ScriptedLogprobDecoder: LogprobDecoding {
    let script: [Int]
    let eos: Int
    var i = 0

    init(script: [Int], eos: Int) {
        self.script = script
        self.eos = eos
    }

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func step(last: Int) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func reset() { i = 0 }

    mutating func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let token = prefill(promptTokens)
        return (token, DecodedTokenLogprob(tokenID: token, logprob: Float(-token)))
    }

    mutating func stepWithLogprob(
        last: Int, topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let token = step(last: last)
        return (token, DecodedTokenLogprob(tokenID: token, logprob: Float(-token)))
    }
}

private actor AsyncGate {
    private var isOpen = false
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func wait() async throws {
        try Task.checkCancellation()
        guard !isOpen else {
            return
        }
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                continuations.append(continuation)
            }
        } onCancel: {
            Task {
                await self.open()
            }
        }
        try Task.checkCancellation()
    }

    func open() {
        guard !isOpen else {
            return
        }
        isOpen = true
        let waiting = continuations
        continuations.removeAll()
        for continuation in waiting {
            continuation.resume()
        }
    }
}

private enum WaitError: Error {
    case timedOut
}

private func waitUntil(
    timeout: Duration = .seconds(1),
    condition: @escaping @Sendable () async -> Bool
) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while !(await condition()) {
        guard clock.now < deadline else {
            throw WaitError.timedOut
        }
        try await Task.sleep(for: .milliseconds(1))
    }
}
