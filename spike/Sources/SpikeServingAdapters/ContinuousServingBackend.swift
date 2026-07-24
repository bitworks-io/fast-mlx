import Foundation

import HarnessCore
import ServingCore

public struct ContinuousServingBackendConfiguration: Sendable {
    public let defaultMaximumCompletionTokens: Int
    public let queueRetryAfterSeconds: Int
    public let mailboxCapacity: BoundedDeltaMailbox.Capacity

    public init(
        defaultMaximumCompletionTokens: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity
    ) {
        precondition(
            defaultMaximumCompletionTokens > 0,
            "defaultMaximumCompletionTokens must be positive")
        precondition(
            (1...3_600).contains(queueRetryAfterSeconds),
            "queueRetryAfterSeconds must be between 1 and 3600")
        self.defaultMaximumCompletionTokens = defaultMaximumCompletionTokens
        self.queueRetryAfterSeconds = queueRetryAfterSeconds
        self.mailboxCapacity = mailboxCapacity
    }
}

public struct ContinuousServingBackendSnapshot: Equatable, Sendable {
    public let activeRequests: Int
    public let coordinatorSlots: Int
    public let reservedKVBytes: Int
    public let maxReservedKVBytes: Int

    public init(
        activeRequests: Int,
        coordinatorSlots: Int,
        reservedKVBytes: Int,
        maxReservedKVBytes: Int
    ) {
        self.activeRequests = activeRequests
        self.coordinatorSlots = coordinatorSlots
        self.reservedKVBytes = reservedKVBytes
        self.maxReservedKVBytes = maxReservedKVBytes
    }
}

public enum ContinuousServingBackendError: Error, Equatable, Sendable {
    case emptyRenderedPrompt
    case invalidStopTokenIDs
    case shuttingDown
}

/// Production-serving bridge for one actor-confined continuous-batch coordinator.
///
/// The coordinator remains the sole scheduler and model/cache owner. This actor only
/// detokenizes each request's bounded token stream into the transport-facing bounded
/// mailbox and maps transport lease cancellation back to the exact coordinator slot.
public actor ContinuousServingBackend: ServingGenerationBackend {
    private struct ActiveRequest {
        let coordinatorID: BatchRequestID
        let promptTokenCount: Int
        let maximumCompletionTokens: Int
        let mailbox: BoundedDeltaMailbox
        let lease: ServingRequestLease
        var task: Task<Void, Never>?
        var cancellationReason: ServingCancellationReason?
        var generatedTokenCount = 0
        var detokenizer: any ScalarServingDetokenizer
        var stopFilter: ServingStopStringFilter
    }

    private let launchedModel: String
    private let coordinator: ContinuousBatchCoordinator
    private let codec: any ScalarServingTextCodec
    private let stopTokenIDs: Set<Int>
    private let modelStopStrings: Set<String>
    private let configuration: ContinuousServingBackendConfiguration

    private var requests: [ServingRequestID: ActiveRequest] = [:]
    private var acceptingRequests = true

    public init(
        launchedModel: String,
        coordinator: ContinuousBatchCoordinator,
        codec: sending any ScalarServingTextCodec,
        stopTokenIDs: Set<Int>,
        modelStopStrings: Set<String>,
        configuration: ContinuousServingBackendConfiguration
    ) {
        precondition(!launchedModel.isEmpty, "launchedModel must be non-empty")
        self.launchedModel = launchedModel
        self.coordinator = coordinator
        self.codec = codec
        self.stopTokenIDs = stopTokenIDs
        self.modelStopStrings = modelStopStrings
        self.configuration = configuration
    }

    public func start(
        _ request: OpenAIChatCompletionRequest
    ) async throws -> ServingGenerationHandle {
        guard acceptingRequests else {
            throw ContinuousServingBackendError.shuttingDown
        }
        try request.requireLaunchedModel(launchedModel)
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw ContinuousServingBackendError.invalidStopTokenIDs
        }

        let promptTokens = try codec.render(messages: request.messages)
        guard !promptTokens.isEmpty else {
            throw ContinuousServingBackendError.emptyRenderedPrompt
        }
        let maximumCompletionTokens =
            request.maxCompletionTokens
            ?? configuration.defaultMaximumCompletionTokens

        let coordinatorHandle: ContinuousBatchRequestHandle
        do {
            coordinatorHandle = try await coordinator.submit(
                ContinuousBatchSubmission(
                    promptTokens: promptTokens,
                    maxOutputTokens: maximumCompletionTokens,
                    stopTokenIDs: stopTokenIDs,
                    architecture: .denseAttention,
                    requestsSpeculation: false))
        } catch ContinuousBatchSchedulerError.queueCapacityExceeded {
            throw ServingBackendAdmissionError.queueFull(
                retryAfterSeconds: configuration.queueRetryAfterSeconds)
        }

        guard acceptingRequests else {
            _ = await coordinator.cancel(coordinatorHandle.id)
            throw ContinuousServingBackendError.shuttingDown
        }

        let id = ServingRequestID("continuous-\(UUID().uuidString)")
        let mailbox = BoundedDeltaMailbox(capacity: configuration.mailboxCapacity)
        let lease = ServingRequestLease(
            id: id,
            onCancelWithReason: { [weak self] reason in
                await self?.cancel(id: id, reason: reason)
            })
        requests[id] = ActiveRequest(
            coordinatorID: coordinatorHandle.id,
            promptTokenCount: promptTokens.count,
            maximumCompletionTokens: maximumCompletionTokens,
            mailbox: mailbox,
            lease: lease,
            task: nil,
            cancellationReason: nil,
            detokenizer: codec.makeDetokenizer(),
            stopFilter: ServingStopStringFilter(
                stopStrings: modelStopStrings.union(request.stop)))

        let task = Task { [weak self] in
            guard let self else { return }
            await self.execute(
                id: id,
                tokens: coordinatorHandle.tokens)
        }
        requests[id]?.task = task

        return ServingGenerationHandle(
            responseID: "chatcmpl-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: launchedModel,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }

    public func snapshot() async -> ContinuousServingBackendSnapshot {
        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        return ContinuousServingBackendSnapshot(
            activeRequests: requests.count,
            coordinatorSlots: slots.count,
            reservedKVBytes: resources?.reservedKVBytes ?? 0,
            maxReservedKVBytes: resources?.maxReservedKVBytes ?? 0)
    }

    public func shutdown() async {
        if !acceptingRequests, requests.isEmpty {
            await coordinator.shutdown()
            return
        }
        acceptingRequests = false

        struct ShutdownRequest {
            let lease: ServingRequestLease
            let mailbox: BoundedDeltaMailbox
            let task: Task<Void, Never>?
        }

        let ids = Array(requests.keys)
        var shutdownRequests: [ShutdownRequest] = []
        shutdownRequests.reserveCapacity(ids.count)
        for id in ids {
            guard var request = requests[id] else { continue }
            request.cancellationReason = .shutdown
            request.task?.cancel()
            requests[id] = request
            shutdownRequests.append(
                ShutdownRequest(
                    lease: request.lease,
                    mailbox: request.mailbox,
                    task: request.task))
        }

        await coordinator.shutdown()
        for request in shutdownRequests {
            _ = await request.lease.cancelFromBackend(.shutdown)
            await request.mailbox.cancel(.shutdown)
        }
        for request in shutdownRequests {
            await request.task?.value
        }
    }

    private func execute(
        id: ServingRequestID,
        tokens: ContinuousBatchTokenStream
    ) async {
        do {
            var stoppedByString = false
            for try await token in tokens {
                try Task.checkCancellation()
                if try await publish(token: token, for: id) {
                    stoppedByString = true
                    break
                }
            }

            try Task.checkCancellation()
            if !stoppedByString {
                try await flushStopFilter(for: id)
            }
            guard let request = requests[id] else {
                throw CancellationError()
            }
            let finishReason: OpenAIChatFinishReason
            if stoppedByString {
                finishReason = .stop
            } else if request.generatedTokenCount >= request.maximumCompletionTokens {
                finishReason = .length
            } else {
                finishReason = .stop
            }
            try await request.mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: finishReason,
                        usage: OpenAIChatUsage(
                            promptTokens: request.promptTokenCount,
                            completionTokens: request.generatedTokenCount))))
            await request.mailbox.finish()
        } catch is CancellationError {
            guard let request = requests[id] else { return }
            let reason =
                request.cancellationReason
                ?? request.lease.terminalCancellationReason
                ?? .clientDisconnected
            await request.mailbox.cancel(reason)
        } catch let error as ServingMailboxError {
            guard let request = requests[id] else { return }
            _ = await coordinator.cancel(request.coordinatorID)
            await request.mailbox.fail(error)
        } catch {
            guard let request = requests[id] else { return }
            _ = await coordinator.cancel(request.coordinatorID)
            await request.mailbox.fail(
                .backend("continuous generation failed: \(error)"))
        }

        requests[id] = nil
    }

    private func publish(
        token: Int,
        for id: ServingRequestID
    ) async throws -> Bool {
        try Task.checkCancellation()
        guard var request = requests[id] else {
            throw CancellationError()
        }

        request.generatedTokenCount += 1
        request.detokenizer.append(token: token)
        let chunk = request.detokenizer.next()
        let output = chunk.map { request.stopFilter.process($0) }
        requests[id] = request

        let stopped = output?.stopped == true
        if stopped {
            _ = await coordinator.cancel(request.coordinatorID)
            try Task.checkCancellation()
        }
        if let text = output?.text {
            try await request.mailbox.send(.text(text))
        }
        try Task.checkCancellation()
        return stopped
    }

    private func flushStopFilter(for id: ServingRequestID) async throws {
        guard var request = requests[id] else {
            throw CancellationError()
        }
        let tail = request.stopFilter.finish()
        requests[id] = request
        if let tail {
            try await request.mailbox.send(.text(tail))
        }
    }

    private func cancel(
        id: ServingRequestID,
        reason: ServingCancellationReason
    ) async {
        if reason == .shutdown {
            await shutdown()
            return
        }
        guard var request = requests[id] else { return }
        request.cancellationReason = reason
        request.task?.cancel()
        requests[id] = request
        _ = await coordinator.cancel(request.coordinatorID)
        await request.mailbox.cancel(reason)
    }
}
