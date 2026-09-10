import Foundation

import MLX
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

/// Map a loaded model to the prompt token IDs its OWN input validation rejects, converted to the
/// `Int` representation `ScalarServingBackendConfiguration.rejectedPromptTokenIDs` screens against.
/// Returns `[]` when the model does not conform to `UnsupportedInputTokenReporting` — every family
/// today except the ones that opt in — so behavior is unchanged for them. When it does conform, a
/// reported ID that cannot round-trip through `Int(exactly:)` is SKIPPED rather than treated as a
/// failure: rendered prompt tokens are always `[Int]`, so a value that cannot be represented as
/// `Int` can never actually appear in a prompt, and skipping it therefore cannot under-screen a
/// token a real request could contain.
public func servingRejectedPromptTokenIDs(model: any LanguageModel) -> Set<Int> {
    guard let reporting = model as? any UnsupportedInputTokenReporting else {
        return []
    }
    return Set(reporting.unsupportedInputTokenIDs.compactMap { Int(exactly: $0) })
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
    /// Immutable model/host-fit capability. Nil preserves the legacy library behavior for existing
    /// embedders; `fastmlx-serve` always supplies it for production routes.
    public let modelCapabilities: ServingModelCapabilities?
    /// Prompt token IDs the loaded model's OWN input validation rejects. Some model families
    /// `preconditionFailure` on specific input token IDs inside their forward path; on the scalar
    /// route that kills the server process. Generated tokens are already masked away from these IDs,
    /// but prompt tokens are not, so an admitted chat request that tokenizes to one of them would be
    /// an HTTP-reachable process kill. Set by the loader from the model's own reported unsupported
    /// input tokens. Defaults to `[]`, meaning the model rejects none — today's behavior, unchanged,
    /// for every family that does not report any.
    public var rejectedPromptTokenIDs: Set<Int>
    /// Artifact-sourced sampling fallback (`--default-sampling generation-config`) applied at
    /// admission (`resolveDecoderSampling`) to a request that omits a sampling field. Set by the
    /// scalar loader ONLY when `ScalarServingModelLoadConfiguration.defaultSampling ==
    /// .generationConfig` resolved successfully -- see `loadScalarServingModel`'s doc comment at the
    /// assignment site for why this is confined to `ScalarServingBackend` admission rather than the
    /// decoder/`InferenceActor` layer. Defaults `nil`, preserving today's `defaults: nil` behavior
    /// (and every other serving route's construction, which never sets this field) byte-for-byte.
    public var samplingDefaults: ServingSamplingDefaults?

    public init(
        defaultMaximumCompletionTokens: Int,
        maximumQueuedRequests: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity,
        toolCallFormat: ToolCallFormat = .json,
        disableThinkingWhenToolsActive: Bool = false,
        thinksByDefault: Bool = false,
        modelCapabilities: ServingModelCapabilities? = nil,
        rejectedPromptTokenIDs: Set<Int> = [],
        samplingDefaults: ServingSamplingDefaults? = nil
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
        self.modelCapabilities = modelCapabilities
        self.rejectedPromptTokenIDs = rejectedPromptTokenIDs
        self.samplingDefaults = samplingDefaults
    }
}

public struct ScalarServingBackendSnapshot: Equatable, Sendable {
    public let activeRequests: Int
    public let queuedRequests: Int
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int

    public init(
        activeRequests: Int,
        queuedRequests: Int,
        mlxActiveBytes: Int = 0,
        mlxCacheBytes: Int = 0,
        mlxPeakBytes: Int = 0
    ) {
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
    }
}

extension ScalarServingBackendSnapshot {
    /// The live process-wide MLX allocator sample, with no in-flight request counts.
    ///
    /// `Memory.snapshot()` is process-global, so the allocator figures are correct no matter which
    /// backend type is serving. Routes whose concrete backend publishes no snapshot shape use this
    /// rather than reporting a fabricated zero for an allocator that has a model resident — a zero
    /// there would also poison the measured-vs-modeled drift comparison, which divides by the
    /// modeled peak.
    public static func processAllocatorSample() -> ScalarServingBackendSnapshot {
        let memory = Memory.snapshot()
        return ScalarServingBackendSnapshot(
            activeRequests: 0,
            queuedRequests: 0,
            mlxActiveBytes: memory.activeMemory,
            mlxCacheBytes: memory.cacheMemory,
            mlxPeakBytes: memory.peakMemory)
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
        let completionBudgetResolution: ServingCompletionBudgetResolution?
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
        precondition(
            configuration.modelCapabilities == nil
                || configuration.modelCapabilities?.model == launchedModel,
            "modelCapabilities must describe the launched model")
    }

    public func start(
        _ request: OpenAIChatCompletionRequest
    ) async throws -> ServingGenerationHandle {
        try await start(request, preservedResolution: nil)
    }

    public func start(
        _ request: OpenAIChatCompletionRequest,
        resolvedCompletionBudget: ServingCompletionBudgetResolution
    ) async throws -> ServingGenerationHandle {
        try await start(request, preservedResolution: resolvedCompletionBudget)
    }

    private func start(
        _ request: OpenAIChatCompletionRequest,
        preservedResolution: ServingCompletionBudgetResolution?
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

        let activeTools = request.activeTools
        // Resolve thinking ONCE and use the SAME value for both the prompt render and the streaming
        // reasoning gate — rendering and gating must not desync (a closed <think></think> in the prompt
        // with a splitter still engaged would mislabel the answer).
        let resolvedEnableThinking = request.resolvedEnableThinking(
            disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive)
        var promptTokens: [Int]?
        var completionBudgetResolution: ServingCompletionBudgetResolution?
        if let capabilities = configuration.modelCapabilities {
            let rendered = try codec.render(
                messages: request.messages,
                tools: activeTools,
                enableThinking: resolvedEnableThinking,
                reasoningEffort: request.reasoningEffort)
            guard !rendered.isEmpty else {
                throw ScalarServingBackendError.emptyRenderedPrompt
            }
            // Screen here too (in addition to the shared screen below) so that a rejected token is a
            // terminal 400 EVEN WHEN the queue is saturated. The queue-full check below throws a
            // retryable error; if a permanently-invalid request reached it first, a client would retry
            // a request that can never succeed. See `screenRejectedPromptTokens` for the rule itself —
            // it lives in one place so the two call sites cannot drift apart.
            try Self.screenRejectedPromptTokens(rendered, configuration: configuration)
            let resolved = try capabilities.resolveCompletionBudget(
                requestedCompletionTokens: request.maxCompletionTokens,
                renderedPromptTokens: rendered.count,
                stream: request.stream)
            if let preservedResolution, preservedResolution != resolved {
                throw OpenAIServingError.server(
                    "The fallback route rendered a different prompt or completion budget",
                    code: "resolved_budget_fallback_mismatch")
            }
            promptTokens = rendered
            completionBudgetResolution = resolved
        } else if preservedResolution != nil {
            throw OpenAIServingError.server(
                "The fallback route has no model-aware capability for budget validation",
                code: "resolved_budget_fallback_unsupported")
        }

        if active != nil, queue.count >= configuration.maximumQueuedRequests {
            throw ServingBackendAdmissionError.queueFull(
                retryAfterSeconds: configuration.queueRetryAfterSeconds)
        }
        if promptTokens == nil {
            promptTokens = try codec.render(
                messages: request.messages,
                tools: activeTools,
                enableThinking: resolvedEnableThinking,
                reasoningEffort: request.reasoningEffort)
        }
        let renderedPromptTokens = promptTokens ?? []
        guard !renderedPromptTokens.isEmpty else {
            throw ScalarServingBackendError.emptyRenderedPrompt
        }
        // Screen for prompt token IDs the model's OWN input validation rejects, BEFORE the prompt
        // reaches the decoder. This is the single path every request converges on (both the
        // model-capabilities-aware branch above and the fallback branch assign into
        // `renderedPromptTokens`), so this screen covers every admission route. It is also called a
        // second time, earlier, inside the model-capabilities branch above — see the comment there for
        // why the check appears twice. Applying it again here is redundant but harmless for that
        // branch, and it is the ONLY screen for the fallback branch, whose render is deliberately
        // deferred past the queue-full check to avoid paying render cost for requests that queue-full
        // will reject anyway.
        try Self.screenRejectedPromptTokens(renderedPromptTokens, configuration: configuration)
        // Resolve + validate sampling at admission so an out-of-range temperature/top_p rejects
        // with a clean 400 here rather than failing mid-generation in the detached task.
        let sampling = try resolveDecoderSampling(request)
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
            promptTokens: renderedPromptTokens,
            maximumCompletionTokens:
                completionBudgetResolution?.appliedCompletionTokens
                ?? request.maxCompletionTokens
                ?? configuration.defaultMaximumCompletionTokens,
            completionBudgetResolution: completionBudgetResolution,
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
            completionBudgetResolution: completionBudgetResolution,
            separatesReasoning: servingSeparatesReasoning(
                thinksByDefault: configuration.thinksByDefault,
                resolvedEnableThinking: resolvedEnableThinking))
    }

    public func snapshot() -> ScalarServingBackendSnapshot {
        let memory = Memory.snapshot()
        return ScalarServingBackendSnapshot(
            activeRequests: active == nil ? 0 : 1,
            queuedRequests: queue.count,
            mlxActiveBytes: memory.activeMemory,
            mlxCacheBytes: memory.cacheMemory,
            mlxPeakBytes: memory.peakMemory)
    }

    /// Cumulative MTP speculative-decoding counters, forwarded from `InferenceActor`. `nil` means
    /// the bound decoder does not conform to `SpeculativeTelemetryProviding` (a plain `MLXDecoder`,
    /// not an `MTPSpeculativeDecoder`) — preserve that nil/non-nil distinction to callers.
    public func speculativeTelemetry() async -> SpeculativeTelemetrySnapshot? {
        await inference.speculativeTelemetry()
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

    /// Reject a rendered prompt that contains a token ID the loaded model's OWN input validation
    /// rejects. Some model families `preconditionFailure` on specific input token IDs inside their
    /// forward path; on the scalar route that kills the server process. Generated tokens are already
    /// masked away from these IDs, but prompt tokens are not, so an admitted chat request that merely
    /// tokenizes to one of them would otherwise be an HTTP-reachable way to kill the server. Called
    /// from two sites in `start(...)` — see the comments there for why — kept as one function so the
    /// rule cannot drift between them. Deliberately does not include the offending token ID or any
    /// decoded text in the message: echoing user-controlled content back in the response is avoided.
    private static func screenRejectedPromptTokens(
        _ tokens: [Int],
        configuration: ScalarServingBackendConfiguration
    ) throws {
        guard !configuration.rejectedPromptTokenIDs.isEmpty,
            tokens.contains(where: { configuration.rejectedPromptTokenIDs.contains($0) })
        else {
            return
        }
        throw OpenAIServingError.invalidRequestWithCode(
            "The request contains a token this model cannot accept",
            param: "messages",
            code: "unsupported_prompt_token")
    }

    /// Bridge the request's OpenAI penalty fields to the decoder-runtime penalties (SpikeCore).
    private static func decoderPenalties(
        _ request: OpenAIChatCompletionRequest
    ) -> DecoderPenalties {
        DecoderPenalties(
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            repetitionPenalty: request.repetitionPenalty)
    }

    /// Resolve the request's sampling policy (ServingCore) into the decoder-runtime policy
    /// (SpikeCore), mapping a policy validation failure to a 400-class serving error so an
    /// out-of-range temperature/top_p/etc. is rejected at admission with an honest param.
    ///
    /// An INSTANCE method (not `static`, unlike its sibling `decoderPenalties` above) specifically
    /// so it can read `configuration.samplingDefaults` -- the `--default-sampling generation-config`
    /// artifact-sourced fallback the scalar loader sets, `nil` for every other route/configuration
    /// (today's behavior, unchanged). Has exactly one caller, `start(_:preservedResolution:)` above.
    private func resolveDecoderSampling(
        _ request: OpenAIChatCompletionRequest
    ) throws -> DecoderSampling {
        let policy: ServingSamplingPolicy
        do {
            policy = try ServingSamplingPolicy.resolve(from: request, defaults: configuration.samplingDefaults)
        } catch let error as ServingSamplingPolicyError {
            throw Self.openAIError(for: error)
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
            return .invalidRequest("top_p must be finite and in (0, 1]", param: "top_p")
        case .topPTooSmallToTruncate:
            return .invalidRequest(
                "top_p is too small to select any tokens; use temperature 0 for greedy decoding",
                param: "top_p")
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

    /// Machine-readable per-request MTP telemetry line: space-separated `key=value`, matching the
    /// convention of `ScalarServingInCheckpointMTPStartupVerdict.machineReadableFields()`
    /// (`MLXScalarServing.swift`). Prefixed `mtp_request_` — distinct from BOTH the startup line's
    /// `in_checkpoint_mtp_*`/`exact_qwen35_mtp_*` keys and the offline gate's `mtp_*` keys
    /// (`mtp_exact`, `mtp_accept_rate`, ...), so a log scraper can never confuse a per-request line
    /// with either. All fields are always present, including zero-valued ones (`acceptedDraftTokens
    /// == 0` is a legitimate, meaningful outcome per `InferenceRunSpeculativeDelta`'s own doc
    /// comment — it must be visible, not silently omitted).
    private static func mtpRequestTelemetryLine(
        requestID: ServingRequestID, generatedTokenCount: Int, delta: InferenceRunSpeculativeDelta
    ) -> String {
        // `passthroughReason` strings contain spaces (e.g. "main model did not emit drafter
        // state" — see `MTPSpeculativeTokenIterator.switchToPassthrough`), which would otherwise
        // split a space-separated `key=value` line into multiple bogus tokens.
        let passthroughReason =
            delta.passthroughReason.map { $0.replacingOccurrences(of: " ", with: "_") } ?? "none"
        return [
            "mtp_request_telemetry=true",
            "mtp_request_id=\(requestID.rawValue)",
            "mtp_request_proposed_draft_tokens=\(delta.proposedDraftTokens)",
            "mtp_request_accepted_draft_tokens=\(delta.acceptedDraftTokens)",
            "mtp_request_generated_token_count=\(generatedTokenCount)",
            "mtp_request_passthrough_reason=\(passthroughReason)",
        ].joined(separator: " ")
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
            // Emitted for EVERY completed speculative request, including a zero-acceptance or
            // passthrough one (`delta` itself, not any of its fields, gates this) — a field
            // silently omitted on a legitimate zero has already shipped as a real bug in this
            // codebase once. `nil` (a non-speculative decoder/route) emits nothing, matching
            // `speculativeDelta`'s own absent-vs-zero contract. Uses the same bare `print` (stdout)
            // mechanism as `FastMLXServe.startupLine`, not a new logging path, so this line is
            // readable off a live serve the same way the startup line already is.
            if let delta = summary.speculativeDelta {
                print(
                    Self.mtpRequestTelemetryLine(
                        requestID: id, generatedTokenCount: summary.generatedTokenCount,
                        delta: delta))
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
