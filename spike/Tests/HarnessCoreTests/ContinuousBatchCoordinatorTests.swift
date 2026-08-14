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
        var soloPipelineState: BatchSoloPipelineState = .canonical
        var script: [Int]
        var cursor = 0
    }

    private let scriptsByPromptHead: [Int: [Int]]
    private let reverseBatchResults: Bool
    private let omittedPromptHead: Int?
    private let speculativeSoloWidth: Int
    private let soloPipelineStateAfterSolo: BatchSoloPipelineState
    private let outputlessSpeculativeDrain: Bool
    private let resources: ContinuousBatchRuntimeResourceSnapshot?
    private let cohortByPromptHead: [Int: BatchDecodeCohort]
    private var slots: [BatchRequestID: Slot] = [:]
    private var promptHeadByID: [BatchRequestID: Int] = [:]

    init(
        scriptsByPromptHead: [Int: [Int]],
        reverseBatchResults: Bool = false,
        omittedPromptHead: Int? = nil,
        speculativeSoloWidth: Int = 1,
        soloPipelineStateAfterSolo: BatchSoloPipelineState = .pipelinedLookahead,
        outputlessSpeculativeDrain: Bool = false,
        resources: ContinuousBatchRuntimeResourceSnapshot? = nil,
        cohortByPromptHead: [Int: BatchDecodeCohort] = [:]
    ) {
        self.scriptsByPromptHead = scriptsByPromptHead
        self.reverseBatchResults = reverseBatchResults
        self.omittedPromptHead = omittedPromptHead
        self.speculativeSoloWidth = speculativeSoloWidth
        self.soloPipelineStateAfterSolo = soloPipelineStateAfterSolo
        self.outputlessSpeculativeDrain = outputlessSpeculativeDrain
        self.resources = resources
        self.cohortByPromptHead = cohortByPromptHead
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? { resources }

    func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        guard let head = admission.submission.promptTokens.first else {
            throw ScriptedBatchRuntimeError.invalidPrefill(admission.id)
        }
        return cohortByPromptHead[head] ?? .unrestricted
    }

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
        let soloPipelineStateAfter: BatchSoloPipelineState
        let resultWidth: Int
        switch action {
        case .solo(let id, let speculationAllowed):
            ids = [id]
            soloPipelineStateAfter = speculationAllowed
                ? soloPipelineStateAfterSolo
                : .pipelinedLookahead
            resultWidth = speculationAllowed ? speculativeSoloWidth : 1
        case .drainSoloPipeline(let id):
            guard let state = slots[id]?.soloPipelineState, state.requiresDrain else {
                throw ScriptedBatchRuntimeError.drainWithoutPending(id)
            }
            ids = [id]
            soloPipelineStateAfter = .canonical
            resultWidth = state == .speculative && outputlessSpeculativeDrain ? 0 : 1
        case .batch(let batchIDs, _):
            for id in batchIDs where slots[id]?.soloPipelineState.requiresDrain == true {
                throw ScriptedBatchRuntimeError.batchWithPending(id)
            }
            ids = batchIDs
            soloPipelineStateAfter = .canonical
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
            slot.soloPipelineState = soloPipelineStateAfter
            slots[id] = slot
            results.append(
                ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: tokens,
                    finished: finished,
                    soloPipelineState: soloPipelineStateAfter))
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
        stopTokenIDs: Set<Int>? = nil,
        architecture: BatchArchitectureClass = .denseAttention,
        speculation: Bool = false
    ) -> ContinuousBatchSubmission {
        ContinuousBatchSubmission(
            promptTokens: prompt,
            maxOutputTokens: maxTokens,
            stopTokenIDs: stopTokenIDs ?? [eos],
            architecture: architecture,
            requestsSpeculation: speculation)
    }

    private func drain(_ coordinator: ContinuousBatchCoordinator) async throws {
        while try await coordinator.runOneTick() {}
    }

    private func collect(_ stream: ContinuousBatchTokenStream) async throws -> [Int] {
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

    func testRuntimeDerivedDecodeCohortsAreAppliedBeforeSchedulerAdmission() async throws {
        let runtime = ScriptedBatchRuntime(
            scriptsByPromptHead: [
                10: [101, 102, 103, 2],
                20: [201, 202, 203, 2],
                30: [301, 302, 303, 2],
            ],
            cohortByPromptHead: [
                10: .fixedKVCapacity(256),
                20: .fixedKVCapacity(256),
                30: .fixedKVCapacity(512),
            ])
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 3, prefill: 3, chunk: 8),
            runtime: runtime,
            automaticDrive: false,
            traceLimit: 64)
        let handles = try await coordinator.submitBatch([
            submission([10]),
            submission([20]),
            submission([30]),
        ])

        try await drain(coordinator)

        let operations = await coordinator.executionTrace().compactMap { event in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        XCTAssertTrue(
            operations.contains(
                .decode(
                    .batch(
                        [handles[0].id, handles[1].id],
                        speculationAllowed: false))))
        XCTAssertTrue(
            operations.contains(
                .decode(
                    .solo(
                        handles[2].id,
                        speculationAllowed: false))))
        XCTAssertFalse(
            operations.contains {
                guard case .decode(.batch(let ids, _)) = $0 else {
                    return false
                }
                return ids.contains(handles[2].id)
            })
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

    func testConfiguredStopTokenSetSuppressesEveryTerminalToken() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 2, prefill: 2, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [
                    10: [101, 99, 102],
                    20: [201, 2, 202],
                ]),
            automaticDrive: false)
        let handles = try await coordinator.submitBatch([
            submission([10], stopTokenIDs: [2, 99]),
            submission([20], stopTokenIDs: [2, 99]),
        ])

        try await drain(coordinator)

        let firstTokens = try await collect(handles[0].tokens)
        let secondTokens = try await collect(handles[1].tokens)
        XCTAssertEqual(firstTokens, [101])
        XCTAssertEqual(secondTokens, [201])
        let remaining = await coordinator.snapshots()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testInvalidStopTokenSetFailsBeforeAdmissionAndDoesNotConsumeID() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [20: [201, 2]]),
            automaticDrive: false)

        do {
            _ = try await coordinator.submit(
                submission([10], stopTokenIDs: []))
            XCTFail("Expected empty stop-token rejection")
        } catch let error as ContinuousBatchCoordinatorError {
            XCTAssertEqual(error, .invalidStopTokenIDs)
        }

        let accepted = try await coordinator.submit(
            submission([20], stopTokenIDs: [2]))
        XCTAssertEqual(accepted.id, BatchRequestID(1))
        await coordinator.shutdown()
    }

    func testAutomaticDriveWaitsForBoundedPublicationCapacityBeforeNextDecode() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 102, 2]]),
            publicationCapacity: 1,
            traceLimit: 32)
        let handle = try await coordinator.submit(submission([10]))

        for _ in 0 ..< 1_000 {
            if await handle.tokens.snapshot().waitingProducers == 1 {
                break
            }
            await Task.yield()
        }

        let blocked = await handle.tokens.snapshot()
        let blockedTrace = await coordinator.executionTrace()
        let blockedDecodes = blockedTrace.filter {
            if case .operation(.decode) = $0 { return true }
            return false
        }
        XCTAssertEqual(blocked.bufferedTokens, 1)
        XCTAssertEqual(blocked.reservedTokens, 0)
        XCTAssertEqual(blocked.waitingProducers, 1)
        XCTAssertEqual(blockedDecodes.count, 1)

        var iterator = handle.tokens.makeAsyncIterator()
        let first = try await iterator.next()
        let second = try await iterator.next()
        let terminal = try await iterator.next()
        XCTAssertEqual(first, 101)
        XCTAssertEqual(second, 102)
        XCTAssertNil(terminal)
        await coordinator.waitUntilIdle()

        let finished = await handle.tokens.snapshot()
        let remaining = await coordinator.snapshots()
        XCTAssertEqual(finished.terminal, .finished)
        XCTAssertTrue(remaining.isEmpty)
    }

    func testCancellationWhilePublicationCapacityBlockedReleasesSlotAndPump() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [
                    10: [101, 102, 103, 2],
                    20: [201, 2],
                ]),
            publicationCapacity: 1,
            traceLimit: 32)
        let blockedHandle = try await coordinator.submit(submission([10]))

        for _ in 0 ..< 1_000 {
            if await blockedHandle.tokens.snapshot().waitingProducers == 1 {
                break
            }
            await Task.yield()
        }
        let blocked = await blockedHandle.tokens.snapshot()
        XCTAssertEqual(blocked.waitingProducers, 1)

        let cancellation = await coordinator.cancel(blockedHandle.id)
        XCTAssertEqual(
            cancellation,
            .cancelled(
                id: blockedHandle.id,
                previousPhase: .decoding(
                    emittedTokens: 1,
                    soloPipelineState: .pipelinedLookahead)))
        await coordinator.waitUntilIdle()
        let cancelledSlot = await coordinator.snapshot(for: blockedHandle.id)
        let cancelledTokens = await blockedHandle.tokens.snapshot()
        XCTAssertNil(cancelledSlot)
        XCTAssertEqual(cancelledTokens.terminal, .cancelled)

        let replacement = try await coordinator.submit(submission([20]))
        var replacementIterator = replacement.tokens.makeAsyncIterator()
        let replacementToken = try await replacementIterator.next()
        let replacementTerminal = try await replacementIterator.next()
        XCTAssertEqual(replacementToken, 201)
        XCTAssertNil(replacementTerminal)
        await coordinator.waitUntilIdle()
        let remaining = await coordinator.snapshots()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testBatchReservationReleasesPartialCapacityWhileAnotherClientIsBlocked() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 2, prefill: 2, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [
                    10: [101, 102, 2],
                    20: [201, 202, 2],
                ]),
            publicationCapacity: 1,
            traceLimit: 32)
        let handles = try await coordinator.submitBatch([
            submission([10]),
            submission([20]),
        ])

        for _ in 0 ..< 1_000 {
            let first = await handles[0].tokens.snapshot()
            if first.bufferedTokens == 1, first.waitingProducers == 1 {
                break
            }
            await Task.yield()
        }

        var firstIterator = handles[0].tokens.makeAsyncIterator()
        var secondIterator = handles[1].tokens.makeAsyncIterator()
        let firstToken = try await firstIterator.next()
        XCTAssertEqual(firstToken, 101)

        for _ in 0 ..< 1_000 {
            let second = await handles[1].tokens.snapshot()
            if second.waitingProducers == 1 { break }
            await Task.yield()
        }

        let firstWhileSecondBlocked = await handles[0].tokens.snapshot()
        let secondBlocked = await handles[1].tokens.snapshot()
        XCTAssertEqual(firstWhileSecondBlocked.reservedTokens, 0)
        XCTAssertEqual(secondBlocked.waitingProducers, 1)

        let secondToken = try await secondIterator.next()
        let firstNext = try await firstIterator.next()
        let secondNext = try await secondIterator.next()
        let firstTerminal = try await firstIterator.next()
        let secondTerminal = try await secondIterator.next()
        XCTAssertEqual(secondToken, 201)
        XCTAssertEqual(firstNext, 102)
        XCTAssertEqual(secondNext, 202)
        XCTAssertNil(firstTerminal)
        XCTAssertNil(secondTerminal)
        await coordinator.waitUntilIdle()
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

    func testOutputlessSpeculativeDrainDoesNotWaitForPublicationCapacityBeforeBatchJoin()
        async throws
    {
        let runtime = ScriptedBatchRuntime(
            scriptsByPromptHead: [
                10: [101, 102, 2],
                20: [201, 202, 2],
            ],
            soloPipelineStateAfterSolo: .speculative,
            outputlessSpeculativeDrain: true)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 2, prefill: 2, chunk: 2),
            runtime: runtime,
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 64)
        let handles = try await coordinator.submitBatch([
            submission([10, 11, 12, 13]),
            submission([20, 21], speculation: true),
        ])

        _ = try await coordinator.runOneTick() // both prefill; short becomes ready
        _ = try await coordinator.runOneTick() // solo(short), then long prefill
        let blockedShort = await handles[1].tokens.snapshot()
        XCTAssertEqual(blockedShort.bufferedTokens, 1)

        let drainTask = Task { try await coordinator.runOneTick() }
        for _ in 0 ..< 1_000 {
            let snapshot = await coordinator.snapshot(for: handles[1].id)
            if snapshot?.phase == .decoding(
                emittedTokens: 1,
                soloPipelineState: .canonical)
            {
                break
            }
            await Task.yield()
        }

        guard
            let drained = await coordinator.snapshot(for: handles[1].id),
            drained.phase == .decoding(
                emittedTokens: 1,
                soloPipelineState: .canonical)
        else {
            drainTask.cancel()
            do {
                _ = try await drainTask.value
            } catch {
                // Expected on the pre-fix path: drain is incorrectly blocked by output capacity.
            }
            return XCTFail(
                "outputless speculative drain should not wait for stream publication capacity")
        }
        let workRemainsAfterDrain = try await drainTask.value
        XCTAssertTrue(workRemainsAfterDrain)

        var shortIterator = handles[1].tokens.makeAsyncIterator()
        let shortToken = try await shortIterator.next()
        XCTAssertEqual(shortToken, 201)

        _ = try await coordinator.runOneTick()
        let operations = await coordinator.executionTrace().compactMap { event in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        XCTAssertTrue(
            operations.contains(
                .decode(.drainSoloPipeline(handles[1].id))))
        XCTAssertTrue(
            operations.contains(
                .decode(
                    .batch(
                        [handles[0].id, handles[1].id],
                        speculationAllowed: false))))
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
                    emittedTokens: 1,
                    soloPipelineState: .pipelinedLookahead)))
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
                    soloPipelineState: .canonical)))
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

    func testDroppingIteratorWithoutPendingNextCancelsAndReleasesQueuedSlot() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 1),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 2], 20: [201, 2]]),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)

        let abandonedID: BatchRequestID
        do {
            let abandoned = try await coordinator.submit(submission([10, 11]))
            abandonedID = abandoned.id
            _ = abandoned.tokens.makeAsyncIterator()
        }

        for _ in 0 ..< 1_000 {
            if await coordinator.snapshot(for: abandonedID) == nil { break }
            await Task.yield()
        }

        let abandonedSnapshot = await coordinator.snapshot(for: abandonedID)
        XCTAssertNil(abandonedSnapshot)
        let replacement = try await coordinator.submit(submission([20]))
        XCTAssertEqual(replacement.id, BatchRequestID(2))
        await coordinator.shutdown()
    }

    func testIteratorKeepsConsumerAliveAfterTheStreamHandleIsDropped() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)

        let id: BatchRequestID
        var iterator: ContinuousBatchTokenStream.AsyncIterator
        do {
            let handle = try await coordinator.submit(submission([10]))
            id = handle.id
            iterator = handle.tokens.makeAsyncIterator()
        }

        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let retainedSnapshot = await coordinator.snapshot(for: id)
        XCTAssertNotNil(retainedSnapshot)

        _ = try await coordinator.runOneTick()
        _ = try await coordinator.runOneTick()
        let visibleToken = try await iterator.next()
        try await drain(coordinator)
        let terminal = try await iterator.next()
        XCTAssertEqual(visibleToken, 101)
        XCTAssertNil(terminal)
    }

    func testDroppingOneOfTwoIteratorsKeepsTheRemainingConsumerAlive() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 2]]),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)
        let handle = try await coordinator.submit(submission([10]))
        var retainedIterator = handle.tokens.makeAsyncIterator()
        do {
            _ = handle.tokens.makeAsyncIterator()
        }

        for _ in 0 ..< 100 {
            await Task.yield()
        }
        let retainedSnapshot = await coordinator.snapshot(for: handle.id)
        XCTAssertNotNil(retainedSnapshot)

        _ = try await coordinator.runOneTick()
        _ = try await coordinator.runOneTick()
        let visibleToken = try await retainedIterator.next()
        try await drain(coordinator)
        let terminal = try await retainedIterator.next()
        XCTAssertEqual(visibleToken, 101)
        XCTAssertNil(terminal)
    }

    func testCancelledManualPumpThrowsWithoutPoisoningCoordinator() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 102, 2], 20: [201, 2]]),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)
        let blocked = try await coordinator.submit(submission([10]))
        _ = try await coordinator.runOneTick()
        _ = try await coordinator.runOneTick()

        let blockedPump = Task { try await coordinator.runOneTick() }
        for _ in 0 ..< 1_000 {
            if await blocked.tokens.snapshot().waitingProducers == 1 { break }
            await Task.yield()
        }
        blockedPump.cancel()
        do {
            _ = try await blockedPump.value
            XCTFail("cancelled manual pump should throw CancellationError")
        } catch is CancellationError {
            // Expected: no runtime work executes after the cancelled reservation wait.
        }

        let coordinatorIsShutDown = await coordinator.isShutDown()
        XCTAssertFalse(coordinatorIsShutDown)
        _ = await coordinator.cancel(blocked.id)
        let replacement = try await coordinator.submit(submission([20]))
        let replacementStream = replacement.tokens
        let collector = Task {
            var tokens: [Int] = []
            for try await token in replacementStream {
                tokens.append(token)
            }
            return tokens
        }
        try await drain(coordinator)
        let replacementTokens = try await collector.value
        let finalSnapshots = await coordinator.snapshots()
        XCTAssertEqual(replacementTokens, [201])
        XCTAssertTrue(finalSnapshots.isEmpty)
    }

    func testCancelledPumpFinishesCommittedMultiTokenPublicationBeforeThrowing() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration(active: 1, prefill: 1, chunk: 8),
            runtime: ScriptedBatchRuntime(
                scriptsByPromptHead: [10: [101, 102, 103, 2]],
                speculativeSoloWidth: 3),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 16)
        let handle = try await coordinator.submit(
            submission([10], maxTokens: 3, speculation: true))
        _ = try await coordinator.runOneTick()

        let pump = Task { try await coordinator.runOneTick() }
        for _ in 0 ..< 1_000 {
            if await handle.tokens.snapshot().waitingProducers == 1 { break }
            await Task.yield()
        }
        pump.cancel()
        for _ in 0 ..< 100 {
            await Task.yield()
        }

        let committedPublication = await handle.tokens.snapshot()
        guard committedPublication.waitingProducers == 1 else {
            _ = await coordinator.cancel(handle.id)
            _ = try? await pump.value
            return XCTFail(
                "task cancellation dropped an already committed multi-token publication")
        }

        var iterator = handle.tokens.makeAsyncIterator()
        let first = try await iterator.next()
        let second = try await iterator.next()
        let third = try await iterator.next()
        let terminal = try await iterator.next()
        do {
            _ = try await pump.value
            XCTFail("the pump should report cancellation after committed publication")
        } catch is CancellationError {
            // The scheduler/publication transaction completed before cancellation surfaced.
        }

        let streamSnapshot = await handle.tokens.snapshot()
        let finalSnapshots = await coordinator.snapshots()
        XCTAssertEqual([first, second, third].compactMap { $0 }, [101, 102, 103])
        XCTAssertNil(terminal)
        XCTAssertEqual(streamSnapshot.terminal, .finished)
        XCTAssertTrue(finalSnapshots.isEmpty)
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
                XCTAssertEqual(
                    error as? ContinuousBatchSchedulerError,
                    .invalidDecodeOutcomeIDs(
                        expected: [handles[0].id, handles[1].id],
                        actual: [handles[0].id]))
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
