import Foundation

import MLXLMCommon
import ServingCore
import SpikeCore

/// Map vendored `MLXLMCommon.ToolCall`s (parsed by `ToolCallProcessor`) into the OpenAI wire
/// shape, where `function.arguments` is always a JSON *string*.
func openAIToolCalls(from calls: [MLXLMCommon.ToolCall]) -> [OpenAIToolCall] {
    calls.enumerated().map { index, call in
        let object = call.function.arguments.mapValues { $0.anyValue }
        let arguments: String
        if JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
            let string = String(data: data, encoding: .utf8) {
            arguments = string
        } else {
            arguments = "{}"
        }
        let id = call.id ?? "call_\(index)"
        return OpenAIToolCall(id: id, function: .init(name: call.function.name, arguments: arguments))
    }
}

/// Resolve the wire format a serving backend should parse tool calls with. The vendored loader
/// infers the model's format from `config.json` (`.xmlFunction` for qwen3_5, `.glm4` for GLM4, …)
/// and leaves it `nil` for the JSON-standard families (Llama/Qwen). We fall back to `.json` so the
/// default behavior is unchanged while non-JSON families parse correctly.
public func servingToolCallFormat(inferred: ToolCallFormat?) -> ToolCallFormat {
    inferred ?? .json
}

public protocol ScalarServingDetokenizer {
    mutating func append(token: Int)
    mutating func next() -> String?
}

public protocol ScalarServingTextCodec: Sendable {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int]
    func makeDetokenizer() -> any ScalarServingDetokenizer
}

public struct ScalarServingBackendConfiguration: Sendable {
    public let defaultMaximumCompletionTokens: Int
    public let maximumQueuedRequests: Int
    public let queueRetryAfterSeconds: Int
    public let mailboxCapacity: BoundedDeltaMailbox.Capacity
    /// Wire format the model uses to emit tool calls. Defaults to `.json` (Llama/Qwen-standard);
    /// the loader overrides it with the model's inferred format (e.g. `.xmlFunction` for qwen3_5)
    /// so tool calls parse correctly for non-JSON families.
    public var toolCallFormat: ToolCallFormat
    /// Legacy tool-thinking workaround: force `enable_thinking:false` when tools are attached and
    /// the client did not set it. Set ONLY for very old dense Qwen3 (QwenLM/Qwen3 #1817); the
    /// agentic qwen3_5 family (Qwen3.5/3.6/3.8) leaves this off and respects the template default.
    public var disableThinkingWhenToolsActive: Bool
    /// Whether the loaded model's family emits its reasoning block by DEFAULT with no leading `<think>`
    /// opener (reasoning from token 0). Set by the loader from `servingThinksByDefault(route:)`; folds
    /// with the per-request resolved thinking flag into `handle.separatesReasoning`. Defaults false
    /// (dense/compiled + any family not yet live-attested → today's passthrough, zero regression).
    public var thinksByDefault: Bool

    public init(
        defaultMaximumCompletionTokens: Int,
        maximumQueuedRequests: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity,
        toolCallFormat: ToolCallFormat = .json,
        disableThinkingWhenToolsActive: Bool = false,
        thinksByDefault: Bool = false
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
        self.toolCallFormat = toolCallFormat
        self.disableThinkingWhenToolsActive = disableThinkingWhenToolsActive
        self.thinksByDefault = thinksByDefault
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
        let sampling: DecoderSampling
        let penalties: DecoderPenalties
        let activeTools: [OpenAIToolSpec]
        let mailbox: BoundedDeltaMailbox
        let lease: ServingRequestLease
    }

    private struct ActiveRequest {
        let request: PendingRequest
        var task: Task<Void, Never>?
        var cancellationReason: ServingCancellationReason?
        var detokenizer: any ScalarServingDetokenizer
        var stopFilter: ServingStopStringFilter
        let toolCallProcessor: ToolCallProcessor?
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
        let activeTools = request.activeTools
        // Resolve thinking ONCE and use the SAME value for both the prompt render and the streaming
        // reasoning gate — rendering and gating must not desync (a closed <think></think> in the prompt
        // with a splitter still engaged would mislabel the answer).
        let resolvedEnableThinking = request.resolvedEnableThinking(
            disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive)
        let promptTokens = try codec.render(
            messages: request.messages,
            tools: activeTools,
            enableThinking: resolvedEnableThinking,
            reasoningEffort: request.reasoningEffort)
        guard !promptTokens.isEmpty else {
            throw ScalarServingBackendError.emptyRenderedPrompt
        }
        // Resolve + validate sampling at admission so an out-of-range temperature/top_p rejects
        // with a clean 400 here rather than failing mid-generation in the detached task.
        let sampling = try Self.resolveDecoderSampling(request)
        let penalties = Self.decoderPenalties(request)

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
            sampling: sampling,
            penalties: penalties,
            activeTools: activeTools,
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
            lease: lease,
            separatesReasoning: servingSeparatesReasoning(
                thinksByDefault: configuration.thinksByDefault,
                resolvedEnableThinking: resolvedEnableThinking))
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

    /// Resolve the request's sampling policy (ServingCore) into the decoder-runtime policy
    /// (SpikeCore), mapping a policy validation failure to a 400-class serving error so an
    /// out-of-range temperature/top_p/etc. is rejected at admission with an honest param.
    /// Bridge the request's OpenAI penalty fields to the decoder-runtime penalties (SpikeCore).
    private static func decoderPenalties(
        _ request: OpenAIChatCompletionRequest
    ) -> DecoderPenalties {
        DecoderPenalties(
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            repetitionPenalty: request.repetitionPenalty)
    }

    private static func resolveDecoderSampling(
        _ request: OpenAIChatCompletionRequest
    ) throws -> DecoderSampling {
        let policy: ServingSamplingPolicy
        do {
            policy = try ServingSamplingPolicy.resolve(from: request)
        } catch let error as ServingSamplingPolicyError {
            throw openAIError(for: error)
        }
        switch policy {
        case .greedy:
            return .greedy
        case let .sampled(temperature, topP, topK, minP, seed):
            return .sampled(
                temperature: temperature, topP: topP, topK: topK, minP: minP, seed: seed)
        }
    }

    private static func openAIError(for error: ServingSamplingPolicyError) -> OpenAIServingError {
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

    private func launch(_ request: PendingRequest) {
        let stops = modelStopStrings.union(request.request.stop)
        let toolCallProcessor: ToolCallProcessor? =
            request.activeTools.isEmpty
            ? nil
            : ToolCallProcessor(
                format: configuration.toolCallFormat,
                tools: request.activeTools.compactMap { $0.raw.asObjectSendable })
        active = ActiveRequest(
            request: request,
            task: nil,
            cancellationReason: nil,
            detokenizer: codec.makeDetokenizer(),
            stopFilter: ServingStopStringFilter(stopStrings: stops),
            toolCallProcessor: toolCallProcessor)
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
                stopTokenIDs: stopTokenIDs,
                sampling: request.sampling,
                penalties: request.penalties
            ) { [weak self] token in
                guard let self else {
                    throw CancellationError()
                }
                return try await self.publish(token: token, for: id)
            }
            try Task.checkCancellation()
            try await flushStopFilter(for: id)
            var finishReason: OpenAIChatFinishReason =
                summary.finishReason == .length ? .length : .stop
            if let processor = active?.toolCallProcessor {
                // Preserve any residual buffered text that turned out NOT to be a tool call
                // (e.g. the model ended mid-`<tool_call` or emitted malformed JSON) so it is
                // surfaced as content rather than silently dropped.
                if let residual = processor.processEOS(returnBufferedText: true), !residual.isEmpty {
                    try await request.mailbox.send(.text(residual))
                }
                if !processor.toolCalls.isEmpty {
                    try await request.mailbox.send(
                        .toolCalls(openAIToolCalls(from: processor.toolCalls)))
                    finishReason = .toolCalls
                }
            }
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
            if let processor = current.toolCallProcessor {
                if let display = processor.processChunk(text), !display.isEmpty {
                    try await current.request.mailbox.send(.text(display))
                }
            } else {
                try await current.request.mailbox.send(.text(text))
            }
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
