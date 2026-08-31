import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@_spi(ProductionRouteEvidence) import SpikeServingAdapters

final class ContinuousServingProductionRouteEvidenceTests: XCTestCase {
    func testManualLoadedWrapperHasNoProductionRouteEvidenceAuthorization()
        async throws
    {
        let loaded = LoadedContinuousServingModel(
            backend: makeEvidenceBackend(),
            startupReport: makeEvidenceStartupReport())

        XCTAssertThrowsError(
            try loaded.productionRouteEvidenceAuthorization()
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingProductionRouteEvidenceError,
                .missingLoadedModelProvenance)
        }
    }

    func testLoaderProvenanceAuthorizesPromptFreeBoundedOutputTokenTrace()
        async throws
    {
        let recorder = EvidenceRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 8,
                maxQueuedRequests: 1),
            runtime: EvidenceRuntime(recorder: recorder),
            automaticDrive: false,
            publicationCapacity: 4,
            traceLimit: 16)
        let backend = makeEvidenceBackend(coordinator: coordinator)
        let loaded = LoadedContinuousServingModel
            .testingLoadedContinuousServingModelWithLoaderProvenance(
                backend: backend,
                startupReport: makeEvidenceStartupReport())
        let authorization = try loaded.productionRouteEvidenceAuthorization()

        let handle = try await backend.startProductionRouteEvidence(
            evidenceRequest(),
            authorization: authorization,
            tokenTrace: .outputTokenIDs(
                maxCompletedRequests: 1,
                maxTokensPerRequest: 1))
        while try await coordinator.runOneTick() {
            await Task.yield()
        }
        _ = try await collectEvidence(handle.mailbox)
        let traces = try await backend.takeProductionRouteCompletedTokenTraces(
            authorization: authorization)

        XCTAssertEqual(traces.count, 1)
        XCTAssertEqual(traces[0].responseID, handle.responseID)
        XCTAssertEqual(traces[0].coordinatorRequestID, BatchRequestID(1))
        XCTAssertEqual(traces[0].outputTokenIDs, [10])
        XCTAssertEqual(traces[0].completionTokenCount, 2)
        XCTAssertTrue(traces[0].truncated)
        XCTAssertFalse(String(describing: traces).contains("sensitive prompt"))
        let consumed = try await backend.takeProductionRouteCompletedTokenTraces(
            authorization: authorization)
        XCTAssertEqual(consumed, [])
        XCTAssertEqual(recorder.removedCount, 1)
        await backend.shutdown()
    }

    func testNormalStartDoesNotRecordTokenTraceByDefaultEvenWithProvenance()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 8,
                maxQueuedRequests: 1),
            runtime: EvidenceRuntime(recorder: EvidenceRuntimeRecorder()),
            automaticDrive: false,
            publicationCapacity: 4,
            traceLimit: 16)
        let backend = makeEvidenceBackend(coordinator: coordinator)
        let loaded = LoadedContinuousServingModel
            .testingLoadedContinuousServingModelWithLoaderProvenance(
                backend: backend,
                startupReport: makeEvidenceStartupReport())
        let authorization = try loaded.productionRouteEvidenceAuthorization()

        let handle = try await backend.start(evidenceRequest())
        while try await coordinator.runOneTick() {
            await Task.yield()
        }
        _ = try await collectEvidence(handle.mailbox)

        let traces = try await backend.takeProductionRouteCompletedTokenTraces(
            authorization: authorization)
        XCTAssertEqual(traces, [])
        await backend.shutdown()
    }
}

private func makeEvidenceBackend(
    coordinator: ContinuousBatchCoordinator? = nil
) -> ContinuousServingBackend {
    let coordinator = coordinator ?? ContinuousBatchCoordinator(
        configuration: try! ContinuousBatchConfiguration(
            maxActiveSlots: 1,
            maxPrefillSlots: 1,
            prefillChunkSize: 8,
            maxQueuedRequests: 1),
        runtime: EvidenceRuntime(recorder: EvidenceRuntimeRecorder()),
        automaticDrive: false,
        publicationCapacity: 4,
        traceLimit: 16)
    return ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: EvidenceCodec(),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 4,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 4_096)))
}

private func makeEvidenceStartupReport() -> ContinuousServingModelStartupReport {
    ContinuousServingModelStartupReport(
        launchedModel: "fixture",
        route: .continuousBatchNoSpec,
        memoryLimitBytes: 1,
        cacheLimitBytes: 1,
        maxReservedKVBytes: 1,
        maxContextTokens: 16,
        maxReservedContextTokens: 16,
        modelFamily: .qwen3,
        modelConfigurationSHA256: String(repeating: "a", count: 64),
        layerCount: 1,
        keyValueHeadCount: 1,
        headDimension: 1,
        stopTokenCount: 1,
        stopStringCount: 0,
        nativeCacheKinds: [.denseAttention],
        startupPromptTokenCount: 1,
        startupGeneratedTokenCount: 1,
        maxActiveSlots: 1,
        maxPrefillSlots: 1,
        prefillChunkSize: 8,
        maxQueuedRequests: 1,
        publicationCapacity: 4,
        soloPLDPolicy: nil,
        modelProofVerified: true)
}

private func evidenceRequest() -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture",
        messages: [
            OpenAIChatMessage(role: .user, text: "sensitive prompt"),
        ],
        maxCompletionTokens: 4,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: [])
}

private func collectEvidence(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private final class EvidenceRuntimeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var removed = 0

    var removedCount: Int {
        lock.withLock { removed }
    }

    func remove() {
        lock.withLock { removed += 1 }
    }
}

private final class EvidenceRuntime: ContinuousBatchRuntime {
    private var ready = false
    private var cursor = 0
    private let recorder: EvidenceRuntimeRecorder

    init(recorder: EvidenceRuntimeRecorder) {
        self.recorder = recorder
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 1,
            reservedKVBytes: ready ? 1 : 0,
            maxReservedKVBytes: 8)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        ready = work.isFinal
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        guard case .solo(let id, speculationAllowed: false) = action, ready else {
            return []
        }
        let script = [10, 11, 99]
        let token = script[cursor]
        cursor += 1
        return [
            ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                soloPipelineState: .canonical),
        ]
    }

    func remove(_ id: BatchRequestID) {
        ready = false
        recorder.remove()
    }
}

private struct EvidenceCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        [1]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        EvidenceDetokenizer()
    }
}

private struct EvidenceDetokenizer: ScalarServingDetokenizer {
    private var pending: Int?

    mutating func append(token: Int) {
        pending = token
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending.map(String.init)
    }
}
