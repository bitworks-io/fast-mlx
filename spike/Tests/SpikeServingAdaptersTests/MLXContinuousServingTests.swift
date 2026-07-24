import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class MLXContinuousServingTests: XCTestCase {
    func testStartupValidationExecutesRuntimeAndReleasesEveryReservation()
        async throws
    {
        let recorder = StartupValidationRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 2,
                maxQueuedRequests: 1),
            runtime: StartupValidationRuntime(recorder: recorder),
            automaticDrive: true,
            publicationCapacity: 1,
            traceLimit: 8)

        let generated = try await validateContinuousServingStartup(
            coordinator: coordinator,
            promptTokens: [10, 11, 12],
            stopTokenIDs: [99])

        XCTAssertEqual(generated, 1)
        XCTAssertEqual(recorder.prefillSlices, [0 ..< 2, 2 ..< 3])
        XCTAssertEqual(recorder.decodeCount, 1)
        XCTAssertEqual(recorder.removeCount, 1)
        let finalSlots = await coordinator.snapshots()
        let finalResources = await coordinator.runtimeResourceSnapshot()
        XCTAssertTrue(finalSlots.isEmpty)
        XCTAssertEqual(finalResources?.reservedKVBytes, 0)
    }

    func testLoaderRejectsUnsupportedArchitectureBeforeWeightLoading()
        async throws
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "fastmlx-continuous-unsupported-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: directory)
        }
        try Data(
            """
            {
              "model_type": "llama",
              "max_position_embeddings": 8192,
              "vocab_size": 128256
            }
            """.utf8
        ).write(to: directory.appendingPathComponent("config.json"))

        do {
            _ = try await loadContinuousServingModel(
                configuration: ContinuousServingModelLoadConfiguration(
                    launchedModel: "unsupported-source-locked",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 4_096,
                    coordinatorConfiguration: try ContinuousBatchConfiguration(
                        maxActiveSlots: 1,
                        maxPrefillSlots: 1,
                        prefillChunkSize: 128,
                        maxQueuedRequests: 1),
                    publicationCapacity: 1,
                    backendConfiguration:
                        ContinuousServingBackendConfiguration(
                            defaultMaximumCompletionTokens: 8,
                            queueRetryAfterSeconds: 1,
                            mailboxCapacity: .init(
                                maxDeltas: 2,
                                maxBytes: 4_096))))
            XCTFail("Unsupported architecture must fail before model loading")
        } catch let error as DenseContinuousBatchRuntimeError {
            XCTAssertEqual(error, .unsupportedModelFamily("llama"))
        }
    }

    func testLoaderConfigurationAcceptsBoundedDenseServingPolicy() throws {
        let configuration = ContinuousServingModelLoadConfiguration(
            launchedModel: "qwen3-source-locked",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 8_192,
            cacheLimitBytes: 1_024,
            maxReservedKVBytes: 4_096,
            coordinatorConfiguration: try ContinuousBatchConfiguration(
                maxActiveSlots: 4,
                maxPrefillSlots: 2,
                prefillChunkSize: 512,
                maxQueuedRequests: 8),
            publicationCapacity: 1,
            backendConfiguration: ContinuousServingBackendConfiguration(
                defaultMaximumCompletionTokens: 512,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(
                    maxDeltas: 8,
                    maxBytes: 32 * 1_024)))

        let validated = try validateContinuousServingModelLoadConfiguration(
            configuration)

        XCTAssertEqual(validated.launchedModel, "qwen3-source-locked")
        XCTAssertEqual(validated.modelDirectory.path, "/tmp")
        XCTAssertEqual(validated.memoryLimitBytes, 8_192)
        XCTAssertEqual(validated.cacheLimitBytes, 1_024)
        XCTAssertEqual(validated.maxReservedKVBytes, 4_096)
        XCTAssertEqual(validated.coordinatorConfiguration.maxActiveSlots, 4)
        XCTAssertEqual(validated.publicationCapacity, 1)
    }

    func testLoaderConfigurationRejectsUnboundedOrUnsafeKVPolicy() throws {
        let coordinator = try ContinuousBatchConfiguration(
            maxActiveSlots: 2,
            maxPrefillSlots: 1,
            prefillChunkSize: 128,
            maxQueuedRequests: 2)
        let backend = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 32,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096))

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 0,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .invalidReservedKVLimit)
        }

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 16_384,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .reservedKVLimitExceedsMemoryLimit)
        }
    }
}

private final class StartupValidationRuntimeRecorder: Sendable {
    private struct State: Sendable {
        var prefillSlices: [Range<Int>] = []
        var decodeCount = 0
        var removeCount = 0
        var reservedKVBytes = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var prefillSlices: [Range<Int>] {
        state.withLock { $0.prefillSlices }
    }

    var decodeCount: Int {
        state.withLock { $0.decodeCount }
    }

    var removeCount: Int {
        state.withLock { $0.removeCount }
    }

    var reservedKVBytes: Int {
        state.withLock { $0.reservedKVBytes }
    }

    func admitted() {
        state.withLock { $0.reservedKVBytes = 1_024 }
    }

    func prefilling(_ range: Range<Int>) {
        state.withLock { $0.prefillSlices.append(range) }
    }

    func decoded() {
        state.withLock { $0.decodeCount += 1 }
    }

    func removed() {
        state.withLock {
            $0.removeCount += 1
            $0.reservedKVBytes = 0
        }
    }
}

private final class StartupValidationRuntime: ContinuousBatchRuntime {
    private let recorder: StartupValidationRuntimeRecorder

    init(recorder: StartupValidationRuntimeRecorder) {
        self.recorder = recorder
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        XCTAssertEqual(admissions.count, 1)
        recorder.admitted()
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: recorder.reservedKVBytes,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        recorder.prefilling(
            work.startToken ..< work.startToken + work.tokens.count)
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        guard case .solo(let id, speculationAllowed: false) = action else {
            XCTFail("Startup validation must use one non-speculative solo decode")
            return []
        }
        recorder.decoded()
        return [
            ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [42],
                finished: true,
                hasPendingSoloLookahead: false)
        ]
    }

    func remove(_ id: BatchRequestID) {
        recorder.removed()
    }
}
