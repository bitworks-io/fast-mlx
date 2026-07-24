import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ContinuousServingBackendTests: XCTestCase {
    func testContinuousRoutePublishesExactTextUsageAndResolvedStopSet() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 4),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 99, 3]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["hello": [10, 11]],
            pieces: [1: "hel", 2: "lo", 3: "hidden"],
            stopTokenIDs: [2_048, 99])

        let handle = try await backend.start(
            request(text: "hello", maxTokens: 8))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        XCTAssertEqual(
            events,
            [
                .text("hel"),
                .text("lo"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 2,
                            completionTokens: 2))),
            ])
        XCTAssertTrue(recorder.decodeActions.allSatisfy(\.speculationDisabled))
        await waitUntil {
            await backend.snapshot().activeRequests == 0
        }
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(
            finalSnapshot,
            ContinuousServingBackendSnapshot(
                activeRequests: 0,
                coordinatorSlots: 0,
                reservedKVBytes: 0,
                maxReservedKVBytes: 4_096))
    }

    func testRequestStopSplitAcrossTokenChunksCancelsCoordinatorNormally() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop": [10]],
            pieces: [
                1: "hello<",
                2: "stop",
                3: ">hidden",
                4: "tail",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop",
                maxTokens: 8,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("hello"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
        let finalSlots = await coordinator.snapshots()
        XCTAssertTrue(finalSlots.isEmpty)
    }

    func testRequestStopAtOutputBudgetReportsStopInsteadOfLength() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["stop-at-budget": [10]],
            pieces: [
                1: "a",
                2: "b<",
                3: "stop>",
            ],
            stopTokenIDs: [99])

        let handle = try await backend.start(
            request(
                text: "stop-at-budget",
                maxTokens: 3,
                stop: ["<stop>"]))
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(
            events,
            [
                .text("a"),
                .text("b"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 3))),
            ])
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }
    }

    func testDetectedStopReleasesCoordinatorBeforeBlockedPrefixPublication() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 99]],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["blocked-stop": [10]],
            pieces: [
                1: "buffer",
                2: "prefix<stop>",
                3: "hidden",
            ],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 64))

        let handle = try await backend.start(
            request(
                text: "blocked-stop",
                maxTokens: 4,
                stop: ["<stop>"]))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        XCTAssertTrue(slots.isEmpty)
        XCTAssertEqual(resources?.reservedKVBytes, 0)

        let cancelled = await handle.lease.cancel(.clientDisconnected)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .clientDisconnected)
    }

    func testQueueExhaustionDisconnectAndReplacementRecoverOneSharedCoordinator() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 99],
                    20: [3, 4, 99],
                    30: [5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
                "replacement": [30],
            ],
            pieces: [
                1: "a",
                2: "b",
                3: "c",
                4: "d",
                5: "r",
            ],
            stopTokenIDs: [99])

        let active = try await backend.start(
            request(text: "active", maxTokens: 2))
        _ = try await coordinator.runOneTick()
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))

        do {
            _ = try await backend.start(
                request(text: "replacement", maxTokens: 1))
            XCTFail("Expected queue-full rejection")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .queueFull(retryAfterSeconds: 2))
        }

        let queuedCancelled = await queued.lease.cancel(.clientDisconnected)
        XCTAssertTrue(queuedCancelled)
        await assertMailboxCancelled(
            queued.mailbox,
            reason: .clientDisconnected)
        let activeCancelled = await active.lease.cancel(.clientDisconnected)
        XCTAssertTrue(activeCancelled)
        await assertMailboxCancelled(
            active.mailbox,
            reason: .clientDisconnected)
        await waitUntil {
            await coordinator.snapshots().isEmpty
        }

        let replacement = try await backend.start(
            request(text: "replacement", maxTokens: 1))
        let collector = Task { try await collect(replacement.mailbox) }
        while try await coordinator.runOneTick() {}
        let replacementEvents = try await collector.value
        XCTAssertEqual(
            replacementEvents,
            [
                .text("r"),
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .length,
                        usage: OpenAIChatUsage(
                            promptTokens: 1,
                            completionTokens: 1))),
            ])
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(
            finalSnapshot,
            ContinuousServingBackendSnapshot(
                activeRequests: 0,
                coordinatorSlots: 0,
                reservedKVBytes: 0,
                maxReservedKVBytes: 4_096))
    }

    func testRuntimeCapacityFailuresAreTypedBeforeHTTPGenerationStarts()
        async throws
    {
        let singleCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 1,
                error: .contextLimitExceeded(
                    BatchRequestID(1),
                    requested: 4_097,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let singleBackend = makeBackend(
            coordinator: singleCoordinator,
            promptByText: ["oversized": [10]],
            pieces: [:],
            stopTokenIDs: [99])

        do {
            _ = try await singleBackend.start(
                request(text: "oversized", maxTokens: 1))
            XCTFail("Expected a typed request-size admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }

        let sharedCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .aggregateKVByteLimitExceeded(
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let sharedBackend = makeBackend(
            coordinator: sharedCoordinator,
            promptByText: [
                "active": [10],
                "blocked": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let active = try await sharedBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await sharedCoordinator.runOneTick()

        do {
            _ = try await sharedBackend.start(
                request(text: "blocked", maxTokens: 1))
            XCTFail("Expected a typed aggregate-capacity admission failure")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(
                error,
                .capacityExceeded(retryAfterSeconds: 2))
        }
        _ = await active.lease.cancel(.clientDisconnected)
        await sharedBackend.shutdown()

        let permanentCoordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 1),
            runtime: AdmissionFailureContinuousRuntime(
                failureOnAdmission: 2,
                error: .requestReservedKVByteLimitExceeded(
                    BatchRequestID(2),
                    requested: 8_192,
                    limit: 4_096)),
            automaticDrive: false,
            publicationCapacity: 1)
        let permanentBackend = makeBackend(
            coordinator: permanentCoordinator,
            promptByText: [
                "active": [10],
                "permanent": [20],
            ],
            pieces: [:],
            stopTokenIDs: [99])
        let permanentActive = try await permanentBackend.start(
            request(text: "active", maxTokens: 1))
        _ = try await permanentCoordinator.runOneTick()

        do {
            _ = try await permanentBackend.start(
                request(text: "permanent", maxTokens: 1))
            XCTFail("Expected permanent request oversize to remain non-retryable")
        } catch let error as ServingBackendAdmissionError {
            XCTAssertEqual(error, .requestTooLarge())
        }
        _ = await permanentActive.lease.cancel(.clientDisconnected)
        await permanentBackend.shutdown()
    }

    func testBoundedDiagnosticsExposeCoordinatorMembershipAndTraceOnly()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 2, queued: 2),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 99],
                    20: [2, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 1,
            traceLimit: 8)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "one": [10],
                "two": [20],
            ],
            pieces: [1: "a", 2: "b"],
            stopTokenIDs: [99])

        let one = try await backend.start(
            request(text: "one", maxTokens: 1))
        let two = try await backend.start(
            request(text: "two", maxTokens: 1))

        let requestIDs = await backend.diagnosticCoordinatorRequestIDs()
        XCTAssertEqual(
            requestIDs,
            [BatchRequestID(1), BatchRequestID(2)])
        let queuedSnapshots = await backend
            .diagnosticCoordinatorSnapshots()
        XCTAssertEqual(queuedSnapshots.map(\.request.id), [
            BatchRequestID(1), BatchRequestID(2),
        ])
        _ = try await coordinator.runOneTick()
        let trace = await backend.diagnosticCoordinatorExecutionTrace()
        XCTAssertTrue(trace.contains {
            if case .operation(.prefill(let slice)) = $0 {
                return slice.id == BatchRequestID(1)
                    || slice.id == BatchRequestID(2)
            }
            return false
        })

        _ = await one.lease.cancel(.clientDisconnected)
        _ = await two.lease.cancel(.clientDisconnected)
        await backend.shutdown()
    }

    func testServingMailboxBackpressureStopsFurtherCoordinatorDecode() async throws {
        let recorder = ContinuousRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [10: [1, 2, 3, 4, 99]],
                recorder: recorder),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: ["slow": [10]],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let handle = try await backend.start(
            request(text: "slow", maxTokens: 4))
        await waitUntil {
            let mailbox = await handle.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
                && recorder.decodeActions.count >= 3
        }
        let blockedDecodeCount = recorder.decodeActions.count
        for _ in 0 ..< 100 {
            await Task.yield()
        }
        XCTAssertEqual(recorder.decodeActions.count, blockedDecodeCount)

        let cancelled = await handle.lease.cancel(.backpressureTimeout)
        XCTAssertTrue(cancelled)
        await assertMailboxCancelled(
            handle.mailbox,
            reason: .backpressureTimeout)
        await waitUntil {
            let snapshot = await backend.snapshot()
            return snapshot.activeRequests == 0
                && snapshot.coordinatorSlots == 0
                && snapshot.reservedKVBytes == 0
        }
    }

    func testShutdownCancelsEveryAcceptedRequestAndRejectsNewWork() async throws {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try configuration(active: 1, queued: 1),
            runtime: FixtureContinuousRuntime(
                scriptsByPromptHead: [
                    10: [1, 2, 3, 99],
                    20: [4, 5, 99],
                ],
                recorder: ContinuousRuntimeRecorder()),
            publicationCapacity: 1,
            traceLimit: 32)
        let backend = makeBackend(
            coordinator: coordinator,
            promptByText: [
                "active": [10],
                "queued": [20],
            ],
            pieces: [1: "a", 2: "b", 3: "c", 4: "d", 5: "e"],
            stopTokenIDs: [99],
            mailboxCapacity: .init(maxDeltas: 1, maxBytes: 8))

        let active = try await backend.start(
            request(text: "active", maxTokens: 3))
        let queued = try await backend.start(
            request(text: "queued", maxTokens: 2))
        await waitUntil {
            let mailbox = await active.mailbox.snapshot()
            return mailbox.bufferedDeltas == 1
                && mailbox.waitingProducers == 1
        }

        await backend.shutdown()
        await assertMailboxCancelled(active.mailbox, reason: .shutdown)
        await assertMailboxCancelled(queued.mailbox, reason: .shutdown)
        let activeState = await active.lease.state
        let queuedState = await queued.lease.state
        XCTAssertEqual(activeState, .cancelled(.shutdown))
        XCTAssertEqual(queuedState, .cancelled(.shutdown))
        let finalSnapshot = await backend.snapshot()
        XCTAssertEqual(
            finalSnapshot,
            ContinuousServingBackendSnapshot(
                activeRequests: 0,
                coordinatorSlots: 0,
                reservedKVBytes: 0,
                maxReservedKVBytes: 4_096))

        do {
            _ = try await backend.start(
                request(text: "active", maxTokens: 1))
            XCTFail("Expected shutdown admission rejection")
        } catch let error as ContinuousServingBackendError {
            XCTAssertEqual(error, .shuttingDown)
        }
    }
}

private extension BatchDecodeAction {
    var speculationDisabled: Bool {
        switch self {
        case .drainSoloPipeline:
            true
        case .solo(_, let speculationAllowed),
            .batch(_, let speculationAllowed):
            !speculationAllowed
        }
    }
}

private final class ContinuousRuntimeRecorder: Sendable {
    private struct State: Sendable {
        var decodeActions: [BatchDecodeAction] = []
        var activeRequests = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var decodeActions: [BatchDecodeAction] {
        state.withLock { $0.decodeActions }
    }

    func record(_ action: BatchDecodeAction) {
        state.withLock { $0.decodeActions.append(action) }
    }

    func admitted(_ count: Int) {
        state.withLock { $0.activeRequests += count }
    }

    func removed() {
        state.withLock { state in
            state.activeRequests = max(0, state.activeRequests - 1)
        }
    }

    var resources: ContinuousBatchRuntimeResourceSnapshot {
        state.withLock { state in
            ContinuousBatchRuntimeResourceSnapshot(
                kvBytesPerToken: 64,
                reservedKVBytes: state.activeRequests * 1_024,
                maxReservedKVBytes: 4_096)
        }
    }
}

private final class FixtureContinuousRuntime: ContinuousBatchRuntime {
    private struct Slot {
        var processedTokens = 0
        var ready = false
        var pendingSoloLookahead = false
        var script: [Int]
        var cursor = 0
    }

    private let scriptsByPromptHead: [Int: [Int]]
    private let recorder: ContinuousRuntimeRecorder
    private var admittedIDs: Set<BatchRequestID> = []
    private var promptHeadByID: [BatchRequestID: Int] = [:]
    private var slots: [BatchRequestID: Slot] = [:]

    init(
        scriptsByPromptHead: [Int: [Int]],
        recorder: ContinuousRuntimeRecorder
    ) {
        self.scriptsByPromptHead = scriptsByPromptHead
        self.recorder = recorder
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admittedIDs.formUnion(admissions.map(\.id))
        recorder.admitted(admissions.count)
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        recorder.resources
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        var slot: Slot
        if let existing = slots[work.id] {
            slot = existing
        } else {
            guard work.startToken == 0,
                let head = work.tokens.first,
                let script = scriptsByPromptHead[head]
            else {
                throw FixtureContinuousRuntimeError.invalidPrefill
            }
            slot = Slot(script: script)
            promptHeadByID[work.id] = head
        }
        guard work.startToken == slot.processedTokens else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        recorder.record(action)
        let ids: [BatchRequestID]
        let pendingAfter: Bool
        switch action {
        case .solo(let id, let speculationAllowed):
            guard !speculationAllowed else {
                throw FixtureContinuousRuntimeError.speculation
            }
            ids = [id]
            pendingAfter = true
        case .drainSoloPipeline(let id):
            guard slots[id]?.pendingSoloLookahead == true else {
                throw FixtureContinuousRuntimeError.invalidDrain
            }
            ids = [id]
            pendingAfter = false
        case .batch(let batchIDs, let speculationAllowed):
            guard !speculationAllowed else {
                throw FixtureContinuousRuntimeError.speculation
            }
            guard batchIDs.allSatisfy({
                slots[$0]?.pendingSoloLookahead == false
            }) else {
                throw FixtureContinuousRuntimeError.invalidBatch
            }
            ids = batchIDs
            pendingAfter = false
        }

        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw FixtureContinuousRuntimeError.decodeBeforeReady
            }
            let tokens: [Int]
            let finished: Bool
            if slot.cursor < slot.script.count {
                tokens = [slot.script[slot.cursor]]
                slot.cursor += 1
                finished = false
            } else {
                tokens = []
                finished = true
            }
            slot.pendingSoloLookahead = pendingAfter
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: tokens,
                finished: finished,
                hasPendingSoloLookahead: pendingAfter)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
        if admittedIDs.remove(id) != nil {
            recorder.removed()
        }
        promptHeadByID[id] = nil
    }
}

private final class AdmissionFailureContinuousRuntime:
    ContinuousBatchRuntime
{
    private let failureOnAdmission: Int
    private let error: DenseContinuousBatchRuntimeError
    private var admissionCount = 0

    init(
        failureOnAdmission: Int,
        error: DenseContinuousBatchRuntimeError
    ) {
        self.failureOnAdmission = failureOnAdmission
        self.error = error
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        admissionCount += 1
        if admissionCount == failureOnAdmission {
            throw error
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}

private enum FixtureContinuousRuntimeError: Error {
    case invalidPrefill
    case decodeBeforeReady
    case invalidDrain
    case invalidBatch
    case speculation
}

private struct FixtureContinuousTextCodec: ScalarServingTextCodec {
    let promptByText: [String: [Int]]
    let pieces: [Int: String]

    func render(messages: [OpenAIChatMessage]) throws -> [Int] {
        guard let text = messages.last?.text,
            let prompt = promptByText[text]
        else {
            throw FixtureContinuousRuntimeError.invalidPrefill
        }
        return prompt
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureContinuousDetokenizer(pieces: pieces)
    }
}

private struct FixtureContinuousDetokenizer: ScalarServingDetokenizer {
    let pieces: [Int: String]
    private var pending: String?

    init(pieces: [Int: String]) {
        self.pieces = pieces
    }

    mutating func append(token: Int) {
        pending = pieces[token]
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

private func configuration(
    active: Int,
    queued: Int
) throws -> ContinuousBatchConfiguration {
    try ContinuousBatchConfiguration(
        maxActiveSlots: active,
        maxPrefillSlots: active,
        prefillChunkSize: 8,
        maxQueuedRequests: queued)
}

private func makeBackend(
    coordinator: ContinuousBatchCoordinator,
    promptByText: [String: [Int]],
    pieces: [Int: String],
    stopTokenIDs: Set<Int>,
    mailboxCapacity: BoundedDeltaMailbox.Capacity = .init(
        maxDeltas: 2,
        maxBytes: 4_096)
) -> ContinuousServingBackend {
    ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: FixtureContinuousTextCodec(
            promptByText: promptByText,
            pieces: pieces),
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: mailboxCapacity))
}

private func request(
    text: String,
    maxTokens: Int,
    stop: [String] = []
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture",
        messages: [
            OpenAIChatMessage(role: .user, text: text),
        ],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop)
}

private func collect(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private func assertMailboxCancelled(
    _ mailbox: BoundedDeltaMailbox,
    reason: ServingCancellationReason,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await mailbox.next()
        XCTFail("Expected mailbox cancellation", file: file, line: line)
    } catch let error as ServingMailboxError {
        XCTAssertEqual(
            error,
            .cancelled(reason),
            file: file,
            line: line)
    } catch {
        XCTFail(
            "Unexpected mailbox error: \(error)",
            file: file,
            line: line)
    }
}

private func waitUntil(
    attempts: Int = 10_000,
    _ predicate: () async -> Bool
) async {
    for _ in 0 ..< attempts {
        if await predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}
