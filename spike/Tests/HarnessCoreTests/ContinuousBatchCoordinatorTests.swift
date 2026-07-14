import XCTest

@testable import HarnessCore

private enum ScriptedBatchRuntimeError: Error {
    case unknownPrompt(Int)
    case invalidPrefill(BatchRequestID)
    case decodeBeforeReady(BatchRequestID)
    case drainWithoutPending(BatchRequestID)
    case batchWithPending(BatchRequestID)
}

private final class ScriptedBatchRuntime: ContinuousBatchRuntime {
    private struct Slot {
        var processedTokens = 0
        var ready = false
        var hasPendingSoloLookahead = false
        var script: [Int]
        var cursor = 0
    }

    private let scriptsByPromptHead: [Int: [Int]]
    private let reverseBatchResults: Bool
    private let omittedPromptHead: Int?
    private let speculativeSoloWidth: Int
    private let resources: ContinuousBatchRuntimeResourceSnapshot?
    private var slots: [BatchRequestID: Slot] = [:]
    private var promptHeadByID: [BatchRequestID: Int] = [:]

    init(
        scriptsByPromptHead: [Int: [Int]],
        reverseBatchResults: Bool = false,
        omittedPromptHead: Int? = nil,
        speculativeSoloWidth: Int = 1,
        resources: ContinuousBatchRuntimeResourceSnapshot? = nil
    ) {
        self.scriptsByPromptHead = scriptsByPromptHead
        self.reverseBatchResults = reverseBatchResults
        self.omittedPromptHead = omittedPromptHead
        self.speculativeSoloWidth = speculativeSoloWidth
        self.resources = resources
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? { resources }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        var slot: Slot
        if let current = slots[work.id] {
            slot = current
        } else {
            guard work.startToken == 0, let head = work.tokens.first,
                let script = scriptsByPromptHead[head]
            else {
                throw ScriptedBatchRuntimeError.invalidPrefill(work.id)
            }
            slot = Slot(script: script)
            promptHeadByID[work.id] = head
        }

        guard work.startToken == slot.processedTokens else {
            throw ScriptedBatchRuntimeError.invalidPrefill(work.id)
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(_ action: BatchDecodeAction) throws -> [ContinuousBatchRuntimeDecodeResult] {
        let ids: [BatchRequestID]
        let pendingAfter: Bool
        let resultWidth: Int
        switch action {
        case .solo(let id, let speculationAllowed):
            ids = [id]
            pendingAfter = true
            resultWidth = speculationAllowed ? speculativeSoloWidth : 1
        case .drainSoloPipeline(let id):
            guard slots[id]?.hasPendingSoloLookahead == true else {
                throw ScriptedBatchRuntimeError.drainWithoutPending(id)
            }
            ids = [id]
            pendingAfter = false
            resultWidth = 1
        case .batch(let batchIDs, _):
            for id in batchIDs where slots[id]?.hasPendingSoloLookahead == true {
                throw ScriptedBatchRuntimeError.batchWithPending(id)
            }
            ids = batchIDs
            pendingAfter = false
            resultWidth = 1
        }

        var results: [ContinuousBatchRuntimeDecodeResult] = []
        for id in ids {
            guard var slot = slots[id], slot.ready else {
                throw ScriptedBatchRuntimeError.decodeBeforeReady(id)
            }
            if promptHeadByID[id] == omittedPromptHead {
                continue
            }
            var tokens: [Int] = []
            let finished: Bool
            if slot.cursor < slot.script.count {
                let end = min(slot.script.count, slot.cursor + resultWidth)
                tokens = Array(slot.script[slot.cursor ..< end])
                slot.cursor = end
                finished = false
            } else {
                finished = true
            }
            slot.hasPendingSoloLookahead = pendingAfter
            slots[id] = slot
            results.append(
                ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: tokens,
                    finished: finished,
                    hasPendingSoloLookahead: pendingAfter))
        }
        if reverseBatchResults, case .batch = action {
            results.reverse()
        }
        return results
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
        promptHeadByID[id] = nil
    }
}

final class ContinuousBatchCoordinatorTests: XCTestCase {
    private func configuration(
        active: Int = 4, prefill: Int = 2, chunk: Int = 4
    ) -> ContinuousBatchConfiguration {
        try! ContinuousBatchConfiguration(
            maxActiveSlots: active,
            maxPrefillSlots: prefill,
            prefillChunkSize: chunk)
    }

    private func submission(
        _ prompt: [Int], maxTokens: Int = 16, eos: Int = 2,
        architecture: BatchArchitectureClass = .denseAttention,
        speculation: Bool = false
    ) -> ContinuousBatchSubmission {
        ContinuousBatchSubmission(
            promptTokens: prompt,
            maxOutputTokens: maxTokens,
            eosToken: eos,
            architecture: architecture,
            requestsSpeculation: speculation)
    }

    private func drain(_ coordinator: ContinuousBatchCoordinator) async throws {
        while try await coordinator.runOneTick() {}
    }

    private func collect(_ stream: AsyncThrowingStream<Int, Error>) async throws -> [Int] {
        var tokens: [Int] = []
        for try await token in stream {
            tokens.append(token)
        }
        return tokens
    }

    func testSharedDecodeDemultiplexesReverseOrderedRuntimeResultsAndConsumesEOS() async throws {
        let runtime = ScriptedBatchRuntime(
            scriptsByPromptHead: [10: [101, 102, 2], 20: [201, 2]],
            reverseBatchResults: true)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: runtime,
            automaticDrive: false,
            traceLimit: 64)
        let handles = try await coordinator.submitBatch([
            submission([10, 11]),
            submission([20, 21]),
        ])

        try await drain(coordinator)

        let firstTokens = try await collect(handles[0].tokens)
        let secondTokens = try await collect(handles[1].tokens)
        XCTAssertEqual(firstTokens, [101, 102])
        XCTAssertEqual(secondTokens, [201])
        let sharedEvents = await coordinator.executionTrace()
        let operations = sharedEvents.compactMap { event in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        XCTAssertTrue(
            operations.contains(
                .decode(.batch([handles[0].id, handles[1].id], speculationAllowed: false))))
    }

    func testAutomaticDriveCompletesAStreamAndWaitUntilIdleJoinsThePump() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2]]),
            traceLimit: 16)
        let handle = try await coordinator.submit(submission([10, 11]))

        await coordinator.waitUntilIdle()

        let tokens = try await collect(handle.tokens)
        XCTAssertEqual(tokens, [101])
        let remaining = await coordinator.snapshots()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testOutputBudgetClipsTheRuntimeStreamAndFinishesTheSlot() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 102, 2]]),
            automaticDrive: false)
        let handle = try await coordinator.submit(
            submission([10], maxTokens: 1))

        try await drain(coordinator)

        let tokens = try await collect(handle.tokens)
        let snapshot = await coordinator.snapshot(for: handle.id)
        XCTAssertEqual(tokens, [101])
        XCTAssertNil(snapshot)
    }

    func testBatchSubmissionValidationIsAtomicAndDoesNotConsumeIDs() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false)

        do {
            _ = try await coordinator.submitBatch([
                submission([10]),
                submission([20], architecture: .mixtureOfExperts),
            ])
            XCTFail("unsupported burst member should reject the whole batch")
        } catch {
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .unsupportedArchitecture(BatchRequestID(2), .mixtureOfExperts))
        }
        let snapshots = await coordinator.snapshots()
        XCTAssertTrue(snapshots.isEmpty)

        let accepted = try await coordinator.submit(submission([10]))
        XCTAssertEqual(accepted.id, BatchRequestID(1))
        await coordinator.shutdown()
    }

    func testDecodeRunsBeforeLongPrefillThenDrainCommitsBeforeBatchJoin() async throws {
        let runtime = ScriptedBatchRuntime(
            scriptsByPromptHead: [10: [101, 2], 20: [201, 202, 2]])
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 2, prefill: 2, chunk: 2),
            runtime: runtime,
            automaticDrive: false,
            traceLimit: 64)
        let handles = try await coordinator.submitBatch([
            submission([10, 11, 12, 13]),
            submission([20, 21], speculation: true),
        ])

        let afterFirst = try await coordinator.runOneTick()
        let afterSecond = try await coordinator.runOneTick()
        let afterDrain = try await coordinator.runOneTick()
        let afterShared = try await coordinator.runOneTick()
        XCTAssertTrue(afterFirst) // both prefill; short becomes ready
        XCTAssertTrue(afterSecond) // solo(short), then long prefill
        XCTAssertTrue(afterDrain) // drain(short), never batch same tick
        XCTAssertTrue(afterShared) // shared batch

        let transitionEvents = await coordinator.executionTrace()
        let operations = transitionEvents.compactMap { event in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        XCTAssertEqual(
            Array(operations.prefix(6)),
            [
                .prefill(BatchPrefillSlice(id: handles[0].id, startToken: 0, count: 2)),
                .prefill(BatchPrefillSlice(id: handles[1].id, startToken: 0, count: 2)),
                .decode(.solo(handles[1].id, speculationAllowed: true)),
                .prefill(BatchPrefillSlice(id: handles[0].id, startToken: 2, count: 2)),
                .decode(.drainSoloPipeline(handles[1].id)),
                .decode(.batch([handles[0].id, handles[1].id], speculationAllowed: false)),
            ])

        try await drain(coordinator)
        let longTokens = try await collect(handles[0].tokens)
        let shortTokens = try await collect(handles[1].tokens)
        XCTAssertEqual(longTokens, [101])
        XCTAssertEqual(shortTokens, [201, 202])
    }

    func testMultiTokenSpeculativeSoloResultConsumesEOSAndHonorsBudget() async throws {
        let eosCoordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 102, 2, 999]],
                speculativeSoloWidth: 4),
            automaticDrive: false)
        let eosHandle = try await eosCoordinator.submit(
            submission([10], speculation: true))

        try await drain(eosCoordinator)

        let eosTokens = try await collect(eosHandle.tokens)
        XCTAssertEqual(eosTokens, [101, 102])

        let budgetCoordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [20: [201, 202, 203]],
                speculativeSoloWidth: 3),
            automaticDrive: false)
        let budgetHandle = try await budgetCoordinator.submit(
            submission([20], maxTokens: 2, speculation: true))

        try await drain(budgetCoordinator)

        let budgetTokens = try await collect(budgetHandle.tokens)
        XCTAssertEqual(budgetTokens, [201, 202])
    }

    func testExplicitCancellationRemovesQueuedPrefillingReadyAndDecodingSlots() async throws {
        func makeCoordinator() -> ContinuousBatchCoordinator {
            ContinuousBatchCoordinator(
                configuration: configuration(active: 1, prefill: 1, chunk: 1),
                runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 102, 2]]),
                automaticDrive: false,
                traceLimit: 16)
        }

        let queued = makeCoordinator()
        let queuedHandle = try await queued.submit(submission([10, 11, 12]))
        let queuedCancellation = await queued.cancel(queuedHandle.id)
        XCTAssertEqual(
            queuedCancellation,
            .cancelled(id: queuedHandle.id, previousPhase: .queued))

        let prefilling = makeCoordinator()
        let prefillHandle = try await prefilling.submit(submission([10, 11, 12]))
        _ = try await prefilling.runOneTick()
        let prefillCancellation = await prefilling.cancel(prefillHandle.id)
        XCTAssertEqual(
            prefillCancellation,
            .cancelled(
                id: prefillHandle.id,
                previousPhase: .prefilling(processedTokens: 1, totalTokens: 3)))

        let ready = makeCoordinator()
        let readyHandle = try await ready.submit(submission([10]))
        _ = try await ready.runOneTick()
        let readyCancellation = await ready.cancel(readyHandle.id)
        XCTAssertEqual(
            readyCancellation,
            .cancelled(id: readyHandle.id, previousPhase: .ready))

        let decoding = makeCoordinator()
        let decodingHandle = try await decoding.submit(submission([10]))
        _ = try await decoding.runOneTick()
        _ = try await decoding.runOneTick()
        let decodingCancellation = await decoding.cancel(decodingHandle.id)
        XCTAssertEqual(
            decodingCancellation,
            .cancelled(
                id: decodingHandle.id,
                previousPhase: .decoding(
                    emittedTokens: 1, hasPendingSoloLookahead: true)))
    }

    func testDecodingCancellationReusesSlotAndReformsSharedBatch() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 2, prefill: 2, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [
                    10: [101, 102, 103, 104, 2],
                    20: [201, 202, 2],
                    30: [301, 302, 2],
                ]),
            automaticDrive: false,
            traceLimit: 32)
        let initial = try await coordinator.submitBatch([
            submission([10]), submission([20]),
        ])
        let disconnectedConsumer = Task {
            for try await _ in initial[1].tokens {}
        }
        await Task.yield()
        _ = try await coordinator.runOneTick() // both prefills
        _ = try await coordinator.runOneTick() // first B=2 token
        let replacement = try await coordinator.submit(submission([30]))
        guard let queuedReplacement = await coordinator.snapshot(for: replacement.id) else {
            return XCTFail("replacement disappeared before cancellation")
        }
        XCTAssertEqual(queuedReplacement.phase, .queued)

        disconnectedConsumer.cancel()
        do {
            try await disconnectedConsumer.value
        } catch is CancellationError {
            // Expected for the disconnect path under test.
        }
        for _ in 0 ..< 100 {
            if await coordinator.snapshot(for: initial[1].id) == nil { break }
            await Task.yield()
        }
        let cancellationTrace = await coordinator.executionTrace()
        let cancellation = cancellationTrace.compactMap {
            if case .cancelled(let result) = $0, result.id == initial[1].id { return result }
            return nil
        }.first
        XCTAssertEqual(
            cancellation,
            .cancelled(
                id: initial[1].id,
                previousPhase: .decoding(
                    emittedTokens: 1,
                    hasPendingSoloLookahead: false)))
        try await drain(coordinator)

        let survivorTokens = try await collect(initial[0].tokens)
        let replacementTokens = try await collect(replacement.tokens)
        XCTAssertEqual(survivorTokens, [101, 102, 103, 104])
        XCTAssertEqual(replacementTokens, [301, 302])
        let trace = await coordinator.executionTrace()
        let operations = trace.compactMap {
            if case .operation(let operation) = $0 { return operation }
            return nil
        }
        XCTAssertTrue(
            operations.contains(
                .decode(.batch([initial[0].id, initial[1].id], speculationAllowed: false))))
        XCTAssertTrue(
            operations.contains(
                .decode(.batch([initial[0].id, replacement.id], speculationAllowed: false))))
    }

    func testConsumerTaskCancellationEnqueuesQueuedSlotRemoval() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 1),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false,
            traceLimit: 16)
        let handle = try await coordinator.submit(submission([10, 11]))
        let consumer = Task {
            for try await _ in handle.tokens {}
        }

        await Task.yield()
        consumer.cancel()
        do {
            try await consumer.value
        } catch {
            // Task cancellation is the event under test.
        }
        for _ in 0 ..< 100 {
            if await coordinator.snapshot(for: handle.id) == nil { break }
            await Task.yield()
        }

        let cancelledSnapshot = await coordinator.snapshot(for: handle.id)
        let cancellationTrace = await coordinator.executionTrace()
        XCTAssertNil(cancelledSnapshot)
        XCTAssertTrue(
            cancellationTrace.contains(
                .cancelled(
                    .cancelled(id: handle.id, previousPhase: .queued))))
    }

    func testShutdownFinishesStreamsRejectsNewWorkAndIsIdempotent() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2], 20: [201, 2]]),
            automaticDrive: false,
            traceLimit: 16)
        let handles = try await coordinator.submitBatch([
            submission([10, 11]), submission([20, 21]),
        ])

        await coordinator.shutdown()
        await coordinator.shutdown()

        for handle in handles {
            do {
                _ = try await collect(handle.tokens)
                XCTFail("shutdown stream should throw cancellation")
            } catch is CancellationError {
                // expected
            }
        }
        do {
            _ = try await coordinator.submit(submission([10]))
            XCTFail("shutdown coordinator accepted new work")
        } catch {
            XCTAssertEqual(error as? ContinuousBatchCoordinatorError, .shuttingDown)
        }
    }

    func testMalformedRuntimeOutcomeFailsAtomicallyWithoutYieldingTokens() async throws {
        let runtime = ScriptedBatchRuntime(
            scriptsByPromptHead: [10: [101, 2], 20: [201, 2]],
            omittedPromptHead: 20)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: runtime,
            automaticDrive: false,
            traceLimit: 16)
        let handles = try await coordinator.submitBatch([
            submission([10]), submission([20]),
        ])
        _ = try await coordinator.runOneTick() // prefill commits

        do {
            _ = try await coordinator.runOneTick()
            XCTFail("missing batch outcome should fail")
        } catch {
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidDecodeOutcomeIDs(
                    expected: [handles[0].id, handles[1].id],
                    actual: [handles[0].id]))
        }
        for handle in handles {
            do {
                let tokens = try await collect(handle.tokens)
                XCTFail("failed tick yielded tokens: \(tokens)")
            } catch {
                // coordinator forwards the scheduler failure to every stream
            }
        }
        let remaining = await coordinator.snapshots()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testTakeExecutionTraceReturnsCommittedEventsOnce() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false,
            traceLimit: 16)
        _ = try await coordinator.submit(submission([10]))
        _ = try await coordinator.runOneTick()

        let first = await coordinator.takeExecutionTrace()
        let second = await coordinator.takeExecutionTrace()

        XCTAssertEqual(
            first,
            [.operation(.prefill(BatchPrefillSlice(id: BatchRequestID(1), startToken: 0, count: 1)))])
        XCTAssertTrue(second.isEmpty)
    }

    func testTakeTimingTraceStampsActorYieldBoundaryAndClearsInterval() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false,
            traceLimit: 16)
        let handle = try await coordinator.submit(submission([10]))
        _ = try await coordinator.runOneTick() // prefill only
        let prefillTiming = await coordinator.takeTimingTrace()
        XCTAssertTrue(prefillTiming.isEmpty)

        _ = try await coordinator.runOneTick() // one visible token
        let first = await coordinator.takeTimingTrace()
        let second = await coordinator.takeTimingTrace()

        XCTAssertEqual(first.count, 1)
        guard case .emitted(let id, let timestamp) = first[0] else {
            return XCTFail("expected emitted timing event, got \(first[0])")
        }
        XCTAssertEqual(id, handle.id)
        XCTAssertTrue(timestamp.isFinite)
        XCTAssertGreaterThan(timestamp, 0)
        XCTAssertTrue(second.isEmpty)
    }

    func testRuntimeResourceSnapshotCrossesActorBoundaryAsValue() async {
        let expected = ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 262_144,
            reservedKVBytes: 2_147_491_968,
            maxReservedKVBytes: 34_359_738_368)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [:],
                resources: expected),
            automaticDrive: false)

        let actual = await coordinator.runtimeResourceSnapshot()
        XCTAssertEqual(actual, expected)
        await coordinator.shutdown()
    }
}
