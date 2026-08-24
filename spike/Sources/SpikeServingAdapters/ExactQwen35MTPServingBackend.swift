import Foundation

import HarnessCore
import MLX
@_spi(FastMLXExactMTP) import MLXLLM
import MLXLMCommon
import ServingCore
import SpikeCore

public struct ExactQwen35MTPServingRunnerRequest: Sendable {
    public let promptTokens: [Int]
    public let maximumCompletionTokens: Int
    public let stopStrings: Set<String>
    public let descriptor: ExactQwen35MTPServingDescriptor

    public init(
        promptTokens: [Int],
        maximumCompletionTokens: Int,
        stopStrings: Set<String>,
        descriptor: ExactQwen35MTPServingDescriptor
    ) {
        self.promptTokens = promptTokens
        self.maximumCompletionTokens = maximumCompletionTokens
        self.stopStrings = stopStrings
        self.descriptor = descriptor
    }
}

public struct ExactQwen35MTPServingRunnerHandle: Sendable {
    public let stream: AsyncStream<Generation>
    public let task: Task<Void, Never>

    public init(stream: AsyncStream<Generation>, task: Task<Void, Never>) {
        self.stream = stream
        self.task = task
    }
}

public protocol ExactQwen35MTPServingRunner: Sendable {
    /// Binding derived from the same loaded target/drafter pair this runner owns.
    var binding: QwenMTPArtifactBinding? { get }

    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle
}

public enum ExactQwen35MTPScalarFallbackIsolation: Equatable, Sendable {
    /// The scalar fallback owns a different raw target/dependency from the MTP runner.
    case strictlySeparateRawTarget
    /// The scalar fallback may share the MTP runner's raw target. This is rejected by the backend.
    case sharedRawTarget
}

public enum ExactQwen35MTPServingBackendConstructionError: Error, Equatable, Sendable {
    case scalarFallbackMustUseSeparateRawTarget
}

public actor ExactQwen35MTPMLXServingRunner: ExactQwen35MTPServingRunner {
    public nonisolated let binding: QwenMTPArtifactBinding?
    private let context: ModelContext
    private let mtpDrafter: any MTPDrafterModel
    private let wiredMemoryTicket: WiredMemoryTicket?

    public init(
        pair: sending Qwen35ExactMTPLoadedPair,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) throws {
        self.binding = try Qwen35ExactMTPRuntimeFactory.servingBinding(for: pair)
        self.context = pair.target
        self.mtpDrafter = pair.drafter.model
        self.wiredMemoryTicket = wiredMemoryTicket
    }

    public func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        var context = context
        if !request.stopStrings.isEmpty {
            context.configuration.stopStrings =
                context.configuration.effectiveStopStrings.union(request.stopStrings)
        }
        let parameters = GenerateParameters(
            maxTokens: request.maximumCompletionTokens,
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0)
        let (stream, task) = try generateTask(
            input: LMInput(tokens: MLXArray(request.promptTokens)),
            parameters: parameters,
            context: context,
            mtpDrafter: mtpDrafter,
            blockSize: request.descriptor.runtimeBlockSize,
            wiredMemoryTicket: wiredMemoryTicket,
            parseToolCalls: false)
        return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
    }
}

public struct ExactQwen35MTPServingBackendConfiguration: Sendable {
    public let defaultMaximumCompletionTokens: Int
    public let mailboxCapacity: BoundedDeltaMailbox.Capacity
    public let disableThinkingWhenToolsActive: Bool
    public let thinksByDefault: Bool

    public init(
        defaultMaximumCompletionTokens: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity,
        disableThinkingWhenToolsActive: Bool = false,
        thinksByDefault: Bool = false
    ) {
        precondition(
            defaultMaximumCompletionTokens > 0,
            "defaultMaximumCompletionTokens must be positive")
        self.defaultMaximumCompletionTokens = defaultMaximumCompletionTokens
        self.mailboxCapacity = mailboxCapacity
        self.disableThinkingWhenToolsActive = disableThinkingWhenToolsActive
        self.thinksByDefault = thinksByDefault
    }
}

public struct ExactQwen35MTPServingBackendSnapshot: Equatable, Sendable {
    public let activeMTPReservations: Int
    public let pendingMTPStartups: Int
    public let acceptingRequests: Bool

    public init(
        activeMTPReservations: Int,
        pendingMTPStartups: Int,
        acceptingRequests: Bool
    ) {
        self.activeMTPReservations = activeMTPReservations
        self.pendingMTPStartups = pendingMTPStartups
        self.acceptingRequests = acceptingRequests
    }
}

public enum ExactQwen35MTPServingBackendError: Error, Equatable, Sendable {
    case shuttingDown
    case unexpectedToolCall
}

private actor ExactQwen35MTPStartupLatch {
    private var finished = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !finished else { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func finish() {
        guard !finished else { return }
        finished = true
        let current = waiters
        waiters.removeAll(keepingCapacity: false)
        for waiter in current {
            waiter.resume()
        }
    }
}

public actor ExactQwen35MTPSerialOwner {
    private var reservationID: UUID?

    public init() {}

    public func activeReservationCount() -> Int {
        reservationID == nil ? 0 : 1
    }

    fileprivate func acquire() -> ExactQwen35MTPReservation? {
        guard reservationID == nil else {
            return nil
        }
        let id = UUID()
        reservationID = id
        return ExactQwen35MTPReservation(id: id, owner: self)
    }

    fileprivate func release(id: UUID) {
        guard reservationID == id else {
            return
        }
        reservationID = nil
    }
}

fileprivate struct ExactQwen35MTPReservation: Sendable {
    private let id: UUID
    private let owner: ExactQwen35MTPSerialOwner

    fileprivate init(id: UUID, owner: ExactQwen35MTPSerialOwner) {
        self.id = id
        self.owner = owner
    }

    fileprivate func release() async {
        await owner.release(id: id)
    }
}

public actor ExactQwen35MTPServingBackend: ServingGenerationBackend {
    private struct ActiveMTPRequest {
        let id: ServingRequestID
        let mailbox: BoundedDeltaMailbox
        let lease: ServingRequestLease
        let upstreamTask: Task<Void, Never>
        let task: Task<Void, Never>
        var cancellationReason: ServingCancellationReason?
    }

    private let launchedModel: String
    private let enabled: Bool
    private let runner: any ExactQwen35MTPServingRunner
    private let scalarFallback: any ServingGenerationBackend
    private let codec: any ScalarServingTextCodec
    private let configuration: ExactQwen35MTPServingBackendConfiguration
    private let serialOwner: ExactQwen35MTPSerialOwner

    private var active: ActiveMTPRequest?
    private var pendingStartups: [UUID: ExactQwen35MTPStartupLatch] = [:]
    private var acceptingRequests = true

    public init(
        launchedModel: String,
        enabled: Bool,
        runner: any ExactQwen35MTPServingRunner,
        scalarFallback: any ServingGenerationBackend,
        scalarFallbackIsolation: ExactQwen35MTPScalarFallbackIsolation,
        codec: sending any ScalarServingTextCodec,
        configuration: ExactQwen35MTPServingBackendConfiguration,
        serialOwner: ExactQwen35MTPSerialOwner = ExactQwen35MTPSerialOwner()
    ) throws {
        precondition(!launchedModel.isEmpty, "launchedModel must be non-empty")
        guard scalarFallbackIsolation == .strictlySeparateRawTarget else {
            throw ExactQwen35MTPServingBackendConstructionError
                .scalarFallbackMustUseSeparateRawTarget
        }
        self.launchedModel = launchedModel
        self.enabled = enabled
        self.runner = runner
        self.scalarFallback = scalarFallback
        self.codec = codec
        self.configuration = configuration
        self.serialOwner = serialOwner
    }

    public func start(
        _ request: OpenAIChatCompletionRequest
    ) async throws -> ServingGenerationHandle {
        guard acceptingRequests else {
            throw ExactQwen35MTPServingBackendError.shuttingDown
        }
        guard request.model == launchedModel else {
            return try await scalarFallback.start(request)
        }

        let sampling = try Self.resolveSampling(request)
        let penalties = Self.decoderPenalties(request)
        let startupID = UUID()
        let startupLatch = ExactQwen35MTPStartupLatch()
        pendingStartups[startupID] = startupLatch

        do {
            let handle = try await startTracked(
                request,
                sampling: sampling,
                penalties: penalties)
            pendingStartups.removeValue(forKey: startupID)
            await startupLatch.finish()
            return handle
        } catch {
            pendingStartups.removeValue(forKey: startupID)
            await startupLatch.finish()
            throw error
        }
    }

    private func startTracked(
        _ request: OpenAIChatCompletionRequest,
        sampling: ServingSamplingPolicy,
        penalties: DecoderPenalties
    ) async throws -> ServingGenerationHandle {
        let activeReservationCount = await serialOwner.activeReservationCount()
        guard acceptingRequests else {
            throw ExactQwen35MTPServingBackendError.shuttingDown
        }
        let decision = ExactQwen35MTPServingAdmissionPolicy.decide(
            enabled: enabled,
            binding: runner.binding,
            sampling: sampling,
            penalties: penalties,
            hasActiveTools: !request.activeTools.isEmpty,
            speculativeRequestCount: activeReservationCount)

        guard case .eligible(let descriptor) = decision else {
            return try await scalarFallback.start(request)
        }
        let reservation = await serialOwner.acquire()
        guard acceptingRequests else {
            await reservation?.release()
            throw ExactQwen35MTPServingBackendError.shuttingDown
        }
        guard let reservation else {
            return try await scalarFallback.start(request)
        }

        let promptTokens: [Int]
        do {
            promptTokens = try codec.render(
                messages: request.messages,
                tools: [],
                enableThinking: request.resolvedEnableThinking(
                    disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive),
                reasoningEffort: request.reasoningEffort)
        } catch {
            await reservation.release()
            guard acceptingRequests else {
                throw ExactQwen35MTPServingBackendError.shuttingDown
            }
            return try await scalarFallback.start(request)
        }
        guard !promptTokens.isEmpty else {
            await reservation.release()
            guard acceptingRequests else {
                throw ExactQwen35MTPServingBackendError.shuttingDown
            }
            return try await scalarFallback.start(request)
        }

        let runnerHandle: ExactQwen35MTPServingRunnerHandle
        do {
            runnerHandle = try await runner.start(
                ExactQwen35MTPServingRunnerRequest(
                    promptTokens: promptTokens,
                    maximumCompletionTokens: request.maxCompletionTokens
                        ?? configuration.defaultMaximumCompletionTokens,
                    stopStrings: Set(request.stop),
                    descriptor: descriptor))
        } catch {
            await reservation.release()
            guard acceptingRequests else {
                throw ExactQwen35MTPServingBackendError.shuttingDown
            }
            return try await scalarFallback.start(request)
        }

        guard acceptingRequests else {
            runnerHandle.task.cancel()
            await runnerHandle.task.value
            await reservation.release()
            throw ExactQwen35MTPServingBackendError.shuttingDown
        }
        return launch(
            request: request,
            runnerHandle: runnerHandle,
            reservation: reservation,
            requestStopStrings: Set(request.stop))
    }

    public func shutdown() async {
        acceptingRequests = false
        let startupLatches = Array(pendingStartups.values)
        var current = active
        if var running = current {
            running.cancellationReason = .shutdown
            active = running
            current = running
            running.upstreamTask.cancel()
            running.task.cancel()
            _ = await running.lease.cancelFromBackend(.shutdown)
            await running.mailbox.cancel(.shutdown)
        }
        await scalarFallback.shutdown()
        for latch in startupLatches {
            await latch.wait()
        }
        await current?.task.value
    }

    public func snapshot() async -> ExactQwen35MTPServingBackendSnapshot {
        ExactQwen35MTPServingBackendSnapshot(
            activeMTPReservations: await serialOwner.activeReservationCount(),
            pendingMTPStartups: pendingStartups.count,
            acceptingRequests: acceptingRequests)
    }

    private func launch(
        request: OpenAIChatCompletionRequest,
        runnerHandle: ExactQwen35MTPServingRunnerHandle,
        reservation: ExactQwen35MTPReservation,
        requestStopStrings: Set<String>
    ) -> ServingGenerationHandle {
        let id = ServingRequestID("exact-mtp-\(UUID().uuidString)")
        let mailbox = BoundedDeltaMailbox(capacity: configuration.mailboxCapacity)
        let lease = ServingRequestLease(
            id: id,
            onCancelWithReason: { [weak self] reason in
                await self?.cancel(id: id, reason: reason)
            })
        let task = Task { [weak self] in
            guard let self else {
                runnerHandle.task.cancel()
                await runnerHandle.task.value
                await reservation.release()
                return
            }
            await self.execute(
                id: id,
                runnerHandle: runnerHandle,
                reservation: reservation,
                requestStopStrings: requestStopStrings)
        }
        active = ActiveMTPRequest(
            id: id,
            mailbox: mailbox,
            lease: lease,
            upstreamTask: runnerHandle.task,
            task: task,
            cancellationReason: nil)
        return ServingGenerationHandle(
            responseID: "chatcmpl-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: launchedModel,
            route: .exactQwen35MTP,
            mailbox: mailbox,
            lease: lease,
            separatesReasoning: servingSeparatesReasoning(
                thinksByDefault: configuration.thinksByDefault,
                resolvedEnableThinking: request.resolvedEnableThinking(
                    disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive)))
    }

    private func execute(
        id: ServingRequestID,
        runnerHandle: ExactQwen35MTPServingRunnerHandle,
        reservation: ExactQwen35MTPReservation,
        requestStopStrings: Set<String>
    ) async {
        let mailbox = active?.mailbox
        var completed = false
        var stopFilter = ServingStopStringFilter(stopStrings: requestStopStrings)
        var requestStopEncountered = false
        do {
            for await event in runnerHandle.stream {
                try Task.checkCancellation()
                switch event {
                case .chunk(let text):
                    let output = stopFilter.process(text)
                    if let text = output.text {
                        try await mailbox?.send(.text(text))
                    }
                    if output.stopped {
                        requestStopEncountered = true
                    }
                case .toolCall:
                    throw ExactQwen35MTPServingBackendError.unexpectedToolCall
                case .info(let info):
                    if !requestStopEncountered, let tail = stopFilter.finish() {
                        try await mailbox?.send(.text(tail))
                    }
                    try await mailbox?.send(
                        .completion(
                            ServingGenerationCompletion(
                                finishReason: requestStopEncountered
                                    ? .stop
                                    : Self.finishReason(from: info.stopReason),
                                usage: OpenAIChatUsage(
                                    promptTokens: info.promptTokenCount,
                                    completionTokens: info.generationTokenCount))))
                    completed = true
                }
            }
            if Task.isCancelled {
                throw CancellationError()
            }
            await runnerHandle.task.value
            if completed {
                await mailbox?.finish()
            } else {
                await mailbox?.fail(.backend("exact MTP generation finished without completion"))
            }
        } catch is CancellationError {
            runnerHandle.task.cancel()
            await runnerHandle.task.value
            let reason = active?.cancellationReason ?? .clientDisconnected
            await mailbox?.cancel(reason)
        } catch let error as ServingMailboxError {
            runnerHandle.task.cancel()
            await runnerHandle.task.value
            switch error {
            case .cancelled(let reason):
                await mailbox?.cancel(reason)
            case .backend:
                await mailbox?.fail(.backend("exact MTP generation failed"))
            }
        } catch {
            runnerHandle.task.cancel()
            await runnerHandle.task.value
            await mailbox?.fail(.backend("exact MTP generation failed"))
        }

        await reservation.release()
        finish(id: id)
    }

    private func cancel(
        id: ServingRequestID,
        reason: ServingCancellationReason
    ) async {
        guard var current = active, current.id == id else {
            return
        }
        if reason == .shutdown {
            await shutdown()
            return
        }
        current.cancellationReason = reason
        active = current
        current.upstreamTask.cancel()
        current.task.cancel()
        await current.mailbox.cancel(reason)
    }

    private func finish(id: ServingRequestID) {
        guard active?.id == id else {
            return
        }
        active = nil
    }

    private static func finishReason(
        from stopReason: GenerateStopReason
    ) -> OpenAIChatFinishReason {
        switch stopReason {
        case .length:
            return .length
        case .stop, .cancelled:
            return .stop
        }
    }

    private static func decoderPenalties(
        _ request: OpenAIChatCompletionRequest
    ) -> DecoderPenalties {
        DecoderPenalties(
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            repetitionPenalty: request.repetitionPenalty)
    }

    private static func resolveSampling(
        _ request: OpenAIChatCompletionRequest
    ) throws -> ServingSamplingPolicy {
        do {
            return try ServingSamplingPolicy.resolve(from: request)
        } catch let error as ServingSamplingPolicyError {
            throw openAIError(for: error)
        }
    }

    private static func openAIError(
        for error: ServingSamplingPolicyError
    ) -> OpenAIServingError {
        switch error {
        case .nonFiniteTemperature, .temperatureOutOfRange:
            return .invalidRequest("temperature must be finite and in (0, 2]", param: "temperature")
        case .nonFiniteTopP, .topPOutOfRange:
            return .invalidRequest("top_p must be finite and in [0, 1]", param: "top_p")
        case .topKOutOfRange:
            return .invalidRequest("top_k must be greater than zero", param: "top_k")
        case .nonFiniteMinP, .minPOutOfRange:
            return .invalidRequest("min_p must be finite and in [0, 1]", param: "min_p")
        }
    }
}
