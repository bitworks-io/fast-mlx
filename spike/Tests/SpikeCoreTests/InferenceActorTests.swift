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
