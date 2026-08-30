import XCTest
import os

import HarnessCore
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ExactQwen35MTPServeCompositionTests: XCTestCase {
    func testCompositeFitProfileCountsTwoTargetsAndMeasuredDrafterWeights() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(FileManager.default.createFile(
            atPath: directory.appendingPathComponent("model.safetensors").path,
            contents: Data(repeating: 7, count: 25)))

        let target = compositeTargetParsed(weights: 100, measured: true)
        let composite = try ExactQwen35MTPCompositeFitProfile.make(
            target: target,
            drafterDirectory: directory)

        XCTAssertEqual(composite.profile.weightsBytes4bitEstimate, 225)
        XCTAssertTrue(composite.weightsAreMeasured)
        XCTAssertFalse(composite.weightsAreDeclared)
        XCTAssertEqual(composite.profile.id, "fixture-target+exact-qwen35-mtp-composition")
        XCTAssertEqual(composite.profile.nLayers, target.profile.nLayers)
        XCTAssertEqual(composite.profile.fixedStateBytes, target.profile.fixedStateBytes)
    }

    func testCompositeFitProfileFailsClosedWhenDrafterWeightsAreUnavailable() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(try ExactQwen35MTPCompositeFitProfile.make(
            target: compositeTargetParsed(weights: 100, measured: true),
            drafterDirectory: directory)) { error in
            XCTAssertEqual(
                error as? ExactQwen35MTPCompositeFitProfileError,
                .drafterWeightsUnavailable)
        }
    }

    func testCompositionLoadsScalarFallbackBeforeExactRuntimeAndUsesExactCodec() async throws {
        let events = LockedEvents()
        let runner = RecordingMTPRunner()
        let scalar = ScriptedCompositionScalarBackend()

        let loaded = try await loadExactQwen35MTPServeComposition(
            configuration: compositionConfiguration(),
            scalarLoader: { configuration in
                events.append("scalar:\(configuration.modelDirectory.path)")
                return ExactQwen35MTPLoadedScalarFallback(
                    backend: scalar,
                    startupReport: scalarReport(memory: configuration.memoryLimitBytes, cache: configuration.cacheLimitBytes))
            },
            runtimeLoader: { configuration in
                events.append("exact:\(configuration.targetDirectory.path):\(configuration.drafterDirectory.path)")
                return ExactQwen35MTPServingRuntimeComponents(
                    runner: runner,
                    codec: FixtureCompositionCodec(promptTokens: [91, 92]),
                    descriptor: exactDescriptor())
            })

        XCTAssertEqual(events.snapshot(), [
            "scalar:/models/qwen35-target",
            "exact:/models/qwen35-target:/models/qwen35-drafter",
        ])
        XCTAssertEqual(
            loaded.exactStartupReport.status,
            ExactQwen35MTPServeStartupStatus.exactSuccess)
        XCTAssertEqual(loaded.exactStartupReport.descriptor, exactDescriptor())

        let handle = try await loaded.backend.start(compositionRequest())
        _ = try await collectComposition(handle.mailbox)

        XCTAssertEqual(handle.route, ServingExecutionRoute.exactQwen35MTP)
        XCTAssertEqual(runner.snapshot().lastPromptTokens, [91, 92])
        XCTAssertEqual(scalar.snapshot().startCount, 0)
    }

    func testCompositionForwardsExplicitQwen38ArtifactSelectionIntoExactRuntimeLoad() async throws {
        let events = LockedEvents()
        let runner = RecordingMTPRunner(descriptor: exactDescriptor(selection: .qwen38_27BMXFP8Depth1))

        let loaded = try await loadExactQwen35MTPServeComposition(
            configuration: compositionConfiguration(selection: .qwen38_27BMXFP8Depth1),
            scalarLoader: { configuration in
                ExactQwen35MTPLoadedScalarFallback(
                    backend: ScriptedCompositionScalarBackend(),
                    startupReport: scalarReport(
                        memory: configuration.memoryLimitBytes,
                        cache: configuration.cacheLimitBytes))
            },
            runtimeLoader: { configuration in
                events.append("selection:\(configuration.selection.rawValue)")
                return ExactQwen35MTPServingRuntimeComponents(
                    runner: runner,
                    codec: FixtureCompositionCodec(promptTokens: [91, 92]),
                    descriptor: exactDescriptor(selection: configuration.selection))
            })

        XCTAssertEqual(events.snapshot(), ["selection:qwen38-27b-mxfp8-depth1"])
        XCTAssertEqual(loaded.exactStartupReport.descriptor?.artifactSelection, .qwen38_27BMXFP8Depth1)
        XCTAssertTrue(
            loaded.exactStartupReport.machineReadableFields()
                .contains("exact_qwen35_mtp_selection=qwen38-27b-mxfp8-depth1"))
    }

    func testLocalSnapshotDownloaderUsesSelectedQwen38LockAndRejectsLegacyLock() async throws {
        let targetDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let drafterDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: targetDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: drafterDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: targetDirectory)
            try? FileManager.default.removeItem(at: drafterDirectory)
        }

        let downloader = ExactQwen35MTPLocalSnapshotDownloader(
            selection: .qwen38_27BMXFP8Depth1,
            targetDirectory: targetDirectory,
            drafterDirectory: drafterDirectory)
        let selected = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        let legacy = QwenMTPKnownArtifactLocks.qwen35_9BDepth1

        let target = try await downloader.download(
            id: selected.targetIdentity.modelID,
            revision: selected.targetIdentity.revision,
            matching: ["*.safetensors", "*.json", "*.jinja"],
            useLatest: false,
            progressHandler: { _ in })
        let drafter = try await downloader.download(
            id: selected.drafterIdentity.modelID,
            revision: selected.drafterIdentity.revision,
            matching: ["*.safetensors", "*.json", "*.jinja"],
            useLatest: false,
            progressHandler: { _ in })

        XCTAssertEqual(target, targetDirectory)
        XCTAssertEqual(drafter, drafterDirectory)
        do {
            _ = try await downloader.download(
                id: legacy.targetIdentity.modelID,
                revision: legacy.targetIdentity.revision,
                matching: ["*.safetensors", "*.json", "*.jinja"],
                useLatest: false,
                progressHandler: { _ in })
            XCTFail("expected legacy lock request to fail closed")
        } catch ExactQwen35MTPLocalSnapshotDownloaderError.unexpectedRequest {
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testCompositionForwardsOneModelCapabilityIntoExactRunner() async throws {
        let capabilities = try ServingModelCapabilities(
            model: "fixture-model",
            nativeMaxContextTokens: 12,
            effectiveMaxContextTokens: 6,
            requestedDefaultCompletionTokens: 4,
            maximumNonStreamingCompletionTokens: 4,
            completionLimitPolicy: .clamp)
        let runner = RecordingMTPRunner()

        let loaded = try await loadExactQwen35MTPServeComposition(
            configuration: compositionConfiguration(modelCapabilities: capabilities),
            scalarLoader: { configuration in
                ExactQwen35MTPLoadedScalarFallback(
                    backend: ScriptedCompositionScalarBackend(),
                    startupReport: scalarReport(
                        memory: configuration.memoryLimitBytes,
                        cache: configuration.cacheLimitBytes))
            },
            runtimeLoader: { _ in
                ExactQwen35MTPServingRuntimeComponents(
                    runner: runner,
                    codec: FixtureCompositionCodec(promptTokens: [91, 92]),
                    descriptor: exactDescriptor())
            })

        let handle = try await loaded.backend.start(
            compositionRequest(maxCompletionTokens: 8))
        _ = try await collectComposition(handle.mailbox)

        XCTAssertEqual(handle.completionBudgetResolution?.appliedCompletionTokens, 4)
        XCTAssertTrue(handle.completionBudgetResolution?.wasClamped == true)
        XCTAssertEqual(runner.snapshot().maximumCompletionTokens, 4)
    }

    func testExactRuntimeFailureReturnsScalarFallbackWithPathFreeStatus() async throws {
        let events = LockedEvents()
        let scalar = ScriptedCompositionScalarBackend()

        let loaded = try await loadExactQwen35MTPServeComposition(
            configuration: compositionConfiguration(selection: .qwen38_27BMXFP8Depth1),
            scalarLoader: { configuration in
                events.append("scalar")
                return ExactQwen35MTPLoadedScalarFallback(
                    backend: scalar,
                    startupReport: scalarReport(memory: configuration.memoryLimitBytes, cache: configuration.cacheLimitBytes))
            },
            runtimeLoader: { _ in
                events.append("exact")
                throw CompositionFixtureError.exactLoadFailed(path: "/secret/local/snapshot")
            })

        XCTAssertEqual(events.snapshot(), ["scalar", "exact"])
        XCTAssertEqual(
            loaded.exactStartupReport.status,
            ExactQwen35MTPServeStartupStatus.scalarFallback(
                reason: .exactRuntimeUnavailable))
        let line = loaded.exactStartupReport.machineReadableFields()
        XCTAssertTrue(line.contains("exact_qwen35_mtp_status=scalar_fallback"))
        XCTAssertTrue(line.contains("exact_qwen35_mtp_selection=qwen38-27b-mxfp8-depth1"))
        XCTAssertTrue(line.contains("exact_qwen35_mtp_reason=exact_runtime_unavailable"))
        XCTAssertFalse(line.contains("/secret"))
        XCTAssertFalse(line.contains("/models"))

        let handle = try await loaded.backend.start(compositionRequest())
        _ = try await collectComposition(handle.mailbox)

        XCTAssertEqual(handle.route, ServingExecutionRoute.scalarGreedy)
        XCTAssertEqual(scalar.snapshot().startCount, 1)
    }

    func testScalarLoadFailureDoesNotAttemptExactRuntime() async {
        let events = LockedEvents()

        do {
            _ = try await loadExactQwen35MTPServeComposition(
                configuration: compositionConfiguration(),
                scalarLoader: { _ in
                    events.append("scalar")
                    throw CompositionFixtureError.scalarLoadFailed
                },
                runtimeLoader: { _ in
                    events.append("exact")
                    return ExactQwen35MTPServingRuntimeComponents(
                        runner: RecordingMTPRunner(),
                        codec: FixtureCompositionCodec(promptTokens: [1]),
                        descriptor: exactDescriptor())
                })
            XCTFail("expected scalar load failure")
        } catch CompositionFixtureError.scalarLoadFailed {
            XCTAssertEqual(events.snapshot(), ["scalar"])
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testExactRuntimeCancellationShutsDownScalarAndPropagates() async {
        let scalar = ScriptedCompositionScalarBackend()

        do {
            _ = try await loadExactQwen35MTPServeComposition(
                configuration: compositionConfiguration(),
                scalarLoader: { configuration in
                    ExactQwen35MTPLoadedScalarFallback(
                        backend: scalar,
                        startupReport: scalarReport(
                            memory: configuration.memoryLimitBytes,
                            cache: configuration.cacheLimitBytes))
                },
                runtimeLoader: { _ in throw CancellationError() })
            XCTFail("expected cancellation to propagate")
        } catch is CancellationError {
            XCTAssertEqual(scalar.snapshot().shutdownCount, 1)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testLoadedRealCompositionWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN35_RUN_LIVE_COMPOSITION"] == "1" else {
            throw XCTSkip("live exact Qwen3.5 composition requires FAST_MLX_QWEN35_RUN_LIVE_COMPOSITION=1")
        }
        guard let target = environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"],
            let drafter = environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        let memoryLimit = Int(environment["FAST_MLX_QWEN35_COMPOSITION_MEMORY_LIMIT_BYTES"] ?? "")
            ?? 96 * 1_024 * 1_024 * 1_024
        let cacheLimit = Int(environment["FAST_MLX_QWEN35_COMPOSITION_CACHE_LIMIT_BYTES"] ?? "")
            ?? 8 * 1_024 * 1_024 * 1_024
        let loaded = try await loadExactQwen35MTPServeComposition(
            configuration: ExactQwen35MTPServeCompositionConfiguration(
                launchedModel: "live-exact-qwen35-composition",
                targetDirectory: URL(fileURLWithPath: target, isDirectory: true),
                drafterDirectory: URL(fileURLWithPath: drafter, isDirectory: true),
                memoryLimitBytes: memoryLimit,
                cacheLimitBytes: cacheLimit,
                scalarBackendConfiguration: .init(
                    defaultMaximumCompletionTokens: 16,
                    maximumQueuedRequests: 1,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(maxDeltas: 8, maxBytes: 64 * 1_024))))

        XCTAssertEqual(
            loaded.exactStartupReport.status,
            ExactQwen35MTPServeStartupStatus.exactSuccess)
        XCTAssertEqual(loaded.exactStartupReport.descriptor?.runtimeBlockSize, 3)
        XCTAssertEqual(loaded.exactStartupReport.descriptor?.maximumAcceptedDraftTokens, 2)
        await loaded.backend.shutdown()
    }
}

private func compositionConfiguration(
    selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1,
    modelCapabilities: ServingModelCapabilities? = nil
) -> ExactQwen35MTPServeCompositionConfiguration {
    ExactQwen35MTPServeCompositionConfiguration(
        selection: selection,
        launchedModel: "fixture-model",
        targetDirectory: URL(fileURLWithPath: "/models/qwen35-target", isDirectory: true),
        drafterDirectory: URL(fileURLWithPath: "/models/qwen35-drafter", isDirectory: true),
        memoryLimitBytes: 64,
        cacheLimitBytes: 16,
        scalarBackendConfiguration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 1,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            modelCapabilities: modelCapabilities))
}

private func compositeTargetParsed(
    weights: Int,
    measured: Bool,
    declared: Bool = false
) -> ParsedModelArch {
    ParsedModelArch(
        profile: ModelArchProfile(
            id: "fixture-target",
            modelType: .hybridLinear,
            nLayers: 8,
            nAttnLayers: 2,
            nKVHeads: 4,
            headDim: 16,
            fixedStateBytes: 32,
            nativeMaxContext: 4_096,
            weightsBytes4bitEstimate: weights,
            license: "fixture"),
        weightsAreMeasured: measured,
        weightsAreDeclared: declared,
        quantBits: 4)
}

private func scalarReport(memory: Int, cache: Int) -> ScalarServingModelStartupReport {
    ScalarServingModelStartupReport(
        launchedModel: "fixture-model",
        route: .scalarGreedy,
        memoryLimitBytes: memory,
        cacheLimitBytes: cache,
        stopTokenCount: 1,
        stopStringCount: 0,
        nativeCacheKinds: [.denseAttention],
        startupPromptTokenCount: 1,
        startupGeneratedTokenCount: 1,
        resetParityVerified: true)
}

private func exactDescriptor(
    selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1
) -> ExactQwen35MTPServingDescriptor {
    let lock: QwenMTPArtifactLock
    switch selection {
    case .qwen35_9BDepth1:
        lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1
    case .qwen38_27BMXFP8Depth1:
        lock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
    }
    return ExactQwen35MTPServingDescriptor(
        artifactSelection: selection,
        targetModelID: lock.targetIdentity.modelID,
        drafterModelID: lock.drafterIdentity.modelID,
        targetRevision: lock.targetIdentity.revision,
        drafterRevision: lock.drafterIdentity.revision,
        sourceRevision: lock.sourceRevision,
        architecture: lock.architecture,
        runtimeBlockSize: 3,
        maximumAcceptedDraftTokens: 2)
}

private func compositionRequest(
    maxCompletionTokens: Int = 2
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [OpenAIChatMessage(role: .user, text: "hello")],
        maxCompletionTokens: maxCompletionTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: [])
}

private func collectComposition(_ mailbox: BoundedDeltaMailbox) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private enum CompositionFixtureError: Error, Equatable {
    case scalarLoadFailed
    case exactLoadFailed(path: String)
}

private final class LockedEvents: Sendable {
    private let values = OSAllocatedUnfairLock(initialState: [String]())

    func append(_ value: String) {
        values.withLock { $0.append(value) }
    }

    func snapshot() -> [String] {
        values.withLock { $0 }
    }
}

private final class ScriptedCompositionScalarBackend: ServingGenerationBackend, Sendable {
    struct Snapshot: Sendable {
        let startCount: Int
        let shutdownCount: Int
    }

    private struct State {
        var startCount = 0
        var shutdownCount = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let sequence = state.withLock { value in
            value.startCount += 1
            return value.startCount
        }
        let mailbox = BoundedDeltaMailbox(capacity: .init(maxDeltas: 4, maxBytes: 1_024))
        Task {
            do {
                try await mailbox.send(.text("scalar"))
                try await mailbox.send(
                    .completion(
                        ServingGenerationCompletion(
                            finishReason: .stop,
                            usage: OpenAIChatUsage(promptTokens: 1, completionTokens: 1))))
                await mailbox.finish()
            } catch {}
        }
        return ServingGenerationHandle(
            responseID: "chatcmpl-scalar-\(sequence)",
            created: 1,
            model: request.model,
            route: .scalarGreedy,
            mailbox: mailbox,
            lease: ServingRequestLease(id: ServingRequestID("scalar-\(sequence)")))
    }

    func shutdown() async {
        state.withLock { $0.shutdownCount += 1 }
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                startCount: $0.startCount,
                shutdownCount: $0.shutdownCount)
        }
    }
}

private final class RecordingMTPRunner: ExactQwen35MTPServingRunner, Sendable {
    struct Snapshot: Sendable {
        let lastPromptTokens: [Int]
        let maximumCompletionTokens: Int
    }

    let binding: QwenMTPArtifactBinding?

    init(descriptor: ExactQwen35MTPServingDescriptor = exactDescriptor()) {
        self.binding = QwenMTPArtifactBinding(
            targetModelID: descriptor.targetModelID,
            drafterModelID: descriptor.drafterModelID,
            targetRevision: descriptor.targetRevision,
            drafterRevision: descriptor.drafterRevision,
            sourceRevision: descriptor.sourceRevision,
            architecture: descriptor.architecture,
            runtimeBlockSize: descriptor.runtimeBlockSize,
            maximumAcceptedDraftTokens: descriptor.maximumAcceptedDraftTokens)
    }

    private struct State: Sendable {
        var promptTokens: [Int] = []
        var maximumCompletionTokens = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        state.withLock {
            $0.promptTokens = request.promptTokens
            $0.maximumCompletionTokens = request.maximumCompletionTokens
        }
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let task = Task {
            continuation.yield(.chunk("exact"))
            continuation.yield(
                .info(
                    GenerateCompletionInfo(
                        promptTokenCount: request.promptTokens.count,
                        generationTokenCount: 1,
                        promptTime: 0,
                        generationTime: 0,
                        stopReason: .stop)))
            continuation.finish()
        }
        return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
    }

    func snapshot() -> Snapshot {
        state.withLock {
            Snapshot(
                lastPromptTokens: $0.promptTokens,
                maximumCompletionTokens: $0.maximumCompletionTokens)
        }
    }
}

private struct FixtureCompositionCodec: ScalarServingTextCodec {
    let promptTokens: [Int]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        promptTokens
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureCompositionDetokenizer()
    }
}

private struct FixtureCompositionDetokenizer: ScalarServingDetokenizer {
    private var emitted = false

    mutating func append(token: Int) {}

    mutating func next() -> String? {
        guard !emitted else { return nil }
        emitted = true
        return "exact"
    }
}
