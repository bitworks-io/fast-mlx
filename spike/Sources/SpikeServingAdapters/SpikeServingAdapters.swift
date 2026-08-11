import Foundation

import ServingCore
import SpikeCore

public protocol ScalarServingDetokenizer {
    mutating func append(token: Int)
    mutating func next() -> String?
}

public protocol ScalarServingTextCodec: Sendable {
    func render(messages: [OpenAIChatMessage]) throws -> [Int]
    func makeDetokenizer() -> any ScalarServingDetokenizer
}

public struct ScalarServingBackendConfiguration: Sendable {
    public let defaultMaximumCompletionTokens: Int
    public let maximumQueuedRequests: Int
    public let queueRetryAfterSeconds: Int
    public let mailboxCapacity: BoundedDeltaMailbox.Capacity

    public init(
        defaultMaximumCompletionTokens: Int,
        maximumQueuedRequests: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity
    ) {
        precondition(
            defaultMaximumCompletionTokens > 0,
            "defaultMaximumCompletionTokens must be positive")
        precondition(
            maximumQueuedRequests >= 0,
            "maximumQueuedRequests must be non-negative")
        precondition(
            (1...3_600).contains(queueRetryAfterSeconds),
            "queueRetryAfterSeconds must be between 1 and 3600")
        self.defaultMaximumCompletionTokens = defaultMaximumCompletionTokens
        self.maximumQueuedRequests = maximumQueuedRequests
        self.queueRetryAfterSeconds = queueRetryAfterSeconds
        self.mailboxCapacity = mailboxCapacity
    }
}

public struct ScalarServingBackendSnapshot: Equatable, Sendable {
    public let activeRequests: Int
    public let queuedRequests: Int

    public init(activeRequests: Int, queuedRequests: Int) {
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
    }
}

public enum ScalarServingBackendError: Error, Equatable, Sendable {
    case emptyRenderedPrompt
    case invalidStopTokenIDs
    case shuttingDown
}

/// Serial, actor-confined scalar generation with a bounded FIFO and bounded publication.
public actor ScalarServingBackend: ServingGenerationBackend {
    private struct PendingRequest: Sendable {
        let id: ServingRequestID
        let request: OpenAIChatCompletionRequest
        let promptTokens: [Int]
        let maximumCompletionTokens: Int
        let mailbox: BoundedDeltaMailbox
        let lease: ServingRequestLease
    }

    private struct ActiveRequest {
        let request: PendingRequest
        var task: Task<Void, Never>?
        var cancellationReason: ServingCancellationReason?
        var detokenizer: any ScalarServingDetokenizer
        var stopFilter: ServingStopStringFilter
    }

    private let launchedModel: String
    private let inference: InferenceActor
    private let codec: any ScalarServingTextCodec
    private let stopTokenIDs: Set<Int>
    private let modelStopStrings: Set<String>
    private let configuration: ScalarServingBackendConfiguration

    private var active: ActiveRequest?
    private var queue: [PendingRequest] = []
    private var acceptingRequests = true

    public init(
        launchedModel: String,
        inference: InferenceActor,
        codec: sending any ScalarServingTextCodec,
        stopTokenIDs: Set<Int>,
        modelStopStrings: Set<String>,
        configuration: ScalarServingBackendConfiguration
    ) {
        precondition(!launchedModel.isEmpty, "launchedModel must be non-empty")
        self.launchedModel = launchedModel
        self.inference = inference
        self.codec = codec
        self.stopTokenIDs = stopTokenIDs
        self.modelStopStrings = modelStopStrings
        self.configuration = configuration
    }

    public func start(
        _ request: OpenAIChatCompletionRequest
    ) async throws -> ServingGenerationHandle {
        guard acceptingRequests else {
            throw ScalarServingBackendError.shuttingDown
        }
        guard request.model == launchedModel else {
            throw OpenAIServingError.invalidRequest(
                "The requested model is not loaded by this server",
                param: "model")
        }
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw ScalarServingBackendError.invalidStopTokenIDs
        }

        if active != nil, queue.count >= configuration.maximumQueuedRequests {
            throw ServingBackendAdmissionError.queueFull(
                retryAfterSeconds: configuration.queueRetryAfterSeconds)
        }
        let promptTokens = try codec.render(messages: request.messages)
        guard !promptTokens.isEmpty else {
            throw ScalarServingBackendError.emptyRenderedPrompt
        }

        let id = ServingRequestID("scalar-\(UUID().uuidString)")
        let mailbox = BoundedDeltaMailbox(
            capacity: configuration.mailboxCapacity)
        let lease = ServingRequestLease(
            id: id,
            onCancelWithReason: { [weak self] reason in
                await self?.cancel(id: id, reason: reason)
            })
        let pending = PendingRequest(
            id: id,
            request: request,
            promptTokens: promptTokens,
            maximumCompletionTokens:
                request.maxCompletionTokens
                ?? configuration.defaultMaximumCompletionTokens,
            mailbox: mailbox,
            lease: lease)

        if active == nil {
            launch(pending)
        } else {
            queue.append(pending)
        }

        return ServingGenerationHandle(
            responseID: "chatcmpl-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: launchedModel,
            route: .scalarGreedy,
            mailbox: mailbox,
            lease: lease)
    }

    public func snapshot() -> ScalarServingBackendSnapshot {
        ScalarServingBackendSnapshot(
            activeRequests: active == nil ? 0 : 1,
            queuedRequests: queue.count)
    }

    /// Stop admission and cancel every active or queued request before returning.
    public func shutdown() async {
        acceptingRequests = false

        let queued = queue
        queue.removeAll(keepingCapacity: false)
        var current = active
        if var running = current {
            running.cancellationReason = .shutdown
            running.task?.cancel()
            active = running
            current = running
        }

        if let current {
            _ = await current.request.lease.cancelFromBackend(.shutdown)
            await current.request.mailbox.cancel(.shutdown)
        }
        for request in queued {
            _ = await request.lease.cancelFromBackend(.shutdown)
            await request.mailbox.cancel(.shutdown)
        }

        guard let current else {
            return
        }
        await current.task?.value
    }

    private func launch(_ request: PendingRequest) {
        let stops = modelStopStrings.union(request.request.stop)
        active = ActiveRequest(
            request: request,
            task: nil,
            cancellationReason: nil,
            detokenizer: codec.makeDetokenizer(),
            stopFilter: ServingStopStringFilter(stopStrings: stops))
        let task = Task { [weak self] in
            guard let self else {
                return
            }
            await self.execute(id: request.id)
        }
        active?.task = task
    }

    private func execute(id: ServingRequestID) async {
        guard let request = active?.request, request.id == id else {
            return
        }

        do {
            let summary = try await inference.generateBounded(
                promptTokens: request.promptTokens,
                maxTokens: request.maximumCompletionTokens,
                stopTokenIDs: stopTokenIDs
            ) { [weak self] token in
                guard let self else {
                    throw CancellationError()
                }
                return try await self.publish(token: token, for: id)
            }
            try Task.checkCancellation()
            try await flushStopFilter(for: id)
            let finishReason: OpenAIChatFinishReason =
                summary.finishReason == .length ? .length : .stop
            try await request.mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: finishReason,
                        usage: OpenAIChatUsage(
                            promptTokens: summary.promptTokenCount,
                            completionTokens: summary.generatedTokenCount))))
            await request.mailbox.finish()
        } catch is CancellationError {
            let reason = active?.cancellationReason ?? .clientDisconnected
            await request.mailbox.cancel(reason)
        } catch let error as ServingMailboxError {
            switch error {
            case .cancelled(let reason):
                await request.mailbox.cancel(reason)
            case .backend:
                await request.mailbox.fail(
                    .backend("scalar generation failed"))
            }
        } catch {
            await request.mailbox.fail(
                .backend("scalar generation failed"))
        }

        await finish(id: id)
    }

    private func publish(
        token: Int,
        for id: ServingRequestID
    ) async throws -> InferenceTokenDisposition {
        try Task.checkCancellation()
        guard var current = active, current.request.id == id else {
            throw CancellationError()
        }

        current.detokenizer.append(token: token)
        let chunk = current.detokenizer.next()
        let output = chunk.map { current.stopFilter.process($0) }
        active = current

        if let text = output?.text {
            try await current.request.mailbox.send(.text(text))
        }
        try Task.checkCancellation()
        return output?.stopped == true ? .stopGeneration : .continueGeneration
    }

    private func flushStopFilter(for id: ServingRequestID) async throws {
        guard var current = active, current.request.id == id else {
            throw CancellationError()
        }
        let tail = current.stopFilter.finish()
        active = current
        if let tail {
            try await current.request.mailbox.send(.text(tail))
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

        if var current = active, current.request.id == id {
            current.cancellationReason = reason
            active = current
            current.task?.cancel()
            await current.request.mailbox.cancel(reason)
            return
        }

        guard let index = queue.firstIndex(where: { $0.id == id }) else {
            return
        }
        let request = queue.remove(at: index)
        await request.mailbox.cancel(reason)
    }

    private func finish(id: ServingRequestID) async {
        guard let current = active, current.request.id == id else {
            return
        }
        if current.cancellationReason == .shutdown
            || current.request.lease.terminalCancellationReason == .shutdown
        {
            acceptingRequests = false
            active = nil
            let queued = queue
            queue.removeAll(keepingCapacity: false)
            for request in queued {
                _ = await request.lease.cancelFromBackend(.shutdown)
                await request.mailbox.cancel(.shutdown)
            }
            return
        }
        guard active?.request.id == id else {
            return
        }
        active = nil
        if acceptingRequests, !queue.isEmpty {
            launch(queue.removeFirst())
        }
    }
}
