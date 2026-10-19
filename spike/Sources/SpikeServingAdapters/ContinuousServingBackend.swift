import Foundation

import HarnessCore
import MLXLMCommon
import ServingCore
import SpikeCore

public enum ContinuousServingAdmissionCoalescing: Equatable, Sendable {
    case automatic(Duration)
    case manualDiagnostic
}

public enum ContinuousServingAdmissionMode: Equatable, Sendable {
    case immediateBatchNoSpec
    case dynamic(
        configuration: ServingAdmissionConfiguration,
        coalescing: ContinuousServingAdmissionCoalescing)
}

public struct ContinuousServingBackendConfiguration: Sendable {
    public let defaultMaximumCompletionTokens: Int
    public let queueRetryAfterSeconds: Int
    public let mailboxCapacity: BoundedDeltaMailbox.Capacity
    public let admission: ContinuousServingAdmissionMode
    public var scorecardTraceCapacity: Int
    /// Wire format the model uses to emit tool calls. Defaults to `.json` (Llama/Qwen-standard);
    /// the loader overrides it with the model's inferred format (e.g. `.xmlFunction` for qwen3_5)
    /// so tool calls parse correctly for non-JSON families.
    public var toolCallFormat: ToolCallFormat
    /// Legacy tool-thinking workaround: force `enable_thinking:false` when tools are attached and
    /// the client did not set it. Set for the very old dense Qwen3 the continuous route serves
    /// (QwenLM/Qwen3 #1817). The agentic qwen3_5 family is served on the scalar route, which leaves
    /// this off and respects the template default.
    public var disableThinkingWhenToolsActive: Bool
    /// Whether the loaded family begins generation inside the reasoning block (see
    /// `servingThinksByDefault`). The loader computes it from the model's `model_type` and its own chat
    /// template; `start` folds it with the request's resolved `enable_thinking` into
    /// `ServingGenerationHandle.separatesReasoning`. Defaults to `false`: byte-identical passthrough.
    public var thinksByDefault: Bool
    /// Immutable model/host-fit capability. Production serving always supplies this; nil retains
    /// the legacy embedding contract for existing library callers and fixtures.
    public let modelCapabilities: ServingModelCapabilities?
    /// The model's logits width (`vocab_size` from `config.json`, the same number the dense runtime's
    /// verified proof carries), used to bound `logit_bias` token ids at admission. `nil` (the default,
    /// FAIL CLOSED) means the width is unknown, which keeps `ContinuousServingBackend.supportsLogitBias`
    /// `false` so a non-empty bias is refused rather than applied unchecked.
    public var logitBiasVocabularySize: Int?

    public init(
        defaultMaximumCompletionTokens: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity,
        scorecardTraceCapacity: Int = 0,
        admission: ContinuousServingAdmissionMode = .immediateBatchNoSpec,
        toolCallFormat: ToolCallFormat = .json,
        disableThinkingWhenToolsActive: Bool = false,
        thinksByDefault: Bool = false,
        modelCapabilities: ServingModelCapabilities? = nil,
        logitBiasVocabularySize: Int? = nil
    ) {
        precondition(
            defaultMaximumCompletionTokens > 0,
            "defaultMaximumCompletionTokens must be positive")
        precondition(
            (1...3_600).contains(queueRetryAfterSeconds),
            "queueRetryAfterSeconds must be between 1 and 3600")
        if case .dynamic(_, .automatic(let duration)) = admission {
            precondition(duration > .zero, "coalescing duration must be positive")
        }
        precondition(scorecardTraceCapacity >= 0, "scorecardTraceCapacity must be non-negative")
        self.defaultMaximumCompletionTokens = defaultMaximumCompletionTokens
        self.queueRetryAfterSeconds = queueRetryAfterSeconds
        self.mailboxCapacity = mailboxCapacity
        self.scorecardTraceCapacity = scorecardTraceCapacity
        self.admission = admission
        self.toolCallFormat = toolCallFormat
        self.disableThinkingWhenToolsActive = disableThinkingWhenToolsActive
        self.thinksByDefault = thinksByDefault
        self.modelCapabilities = modelCapabilities
        self.logitBiasVocabularySize = logitBiasVocabularySize
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
    private struct PreparedRequest {
        let id: ServingRequestID
        let promptTokens: [Int]
        let maximumCompletionTokens: Int
        let stopStrings: Set<String>
        /// `ignore_eos`: the coordinator submission must not end this request on a model stop id.
        let ignoreEOS: Bool
        /// The request's validated `logit_bias` (bounded by the vocabulary in `start`); carried into
        /// the coordinator submission, where the runtime applies it per row.
        let logitBias: [Int: Float]
        let activeTools: [OpenAIToolSpec]
        /// The request's `parallel_tool_calls` (`nil` = absent/null). Only an explicit `false`
        /// stops decoding at the first completed tool call.
        let parallelToolCalls: Bool?
        let completionBudgetResolution: ServingCompletionBudgetResolution?
        /// Streaming reasoning separation for this request (family `thinksByDefault` folded with the
        /// SAME resolved `enable_thinking` the prompt was rendered from).
        let separatesReasoning: Bool
        let outputTokenTraceLimit: Int?
        let completedTraceCapacity: Int
    }

    private struct PendingAdmission {
        let prepared: PreparedRequest
        let continuation: CheckedContinuation<ServingGenerationHandle, any Error>
    }

    private struct ActiveRequest {
        let coordinatorID: BatchRequestID
        let promptTokenCount: Int
        let maximumCompletionTokens: Int
        let mailbox: BoundedDeltaMailbox
        let lease: ServingRequestLease
        var task: Task<Void, Never>?
        var cancellationReason: ServingCancellationReason?
        var admissionReleased: Bool
        var generatedTokenCount = 0
        var detokenizer: any ScalarServingDetokenizer
        var stopFilter: ServingStopStringFilter
        let toolCallProcessor: ToolCallProcessor?
        /// Non-nil only when a tool processor exists AND the response separates reasoning: keeps the
        /// parser from seeing text the client is shown as `reasoning_content`.
        var reasoningGate: ReasoningPhaseToolCallGate?
        let parallelToolCalls: Bool?
        let responseID: String
        let outputTokenTraceLimit: Int?
        let completedTraceCapacity: Int
        var outputTokenTrace: [Int] = []
        var outputTokenTraceTruncated = false
    }

    private let launchedModel: String
    private let coordinator: ContinuousBatchCoordinator
    private let codec: any ScalarServingTextCodec
    private let stopTokenIDs: Set<Int>
    private let modelStopStrings: Set<String>
    private let configuration: ContinuousServingBackendConfiguration

    private var requests: [ServingRequestID: ActiveRequest] = [:]
    private var admissionReducer: ServingAdmissionReducer?
    private var pendingAdmissions: [ServingRequestID: PendingAdmission] = [:]
    /// Requests remain pending while the coordinator actor validates and commits their atomic
    /// submission. Mark that reentrant window explicitly so a later join can submit only its
    /// new request instead of resubmitting the cohort already awaiting coordinator admission.
    private var admittingRequestIDs: Set<ServingRequestID> = []
    /// Cancellation can arrive while the coordinator owns an in-flight admission. Keep the
    /// reducer reservation until the returned physical handle has been cancelled.
    private var cancelledAdmittingRequestIDs: Set<ServingRequestID> = []
    private var coalescingTask: Task<Void, Never>?
    private var acceptingRequests = true
    private var completedTokenTraces:
        [ContinuousServingCompletedRequestTokenTrace] = []

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
        precondition(
            configuration.modelCapabilities == nil
                || configuration.modelCapabilities?.model == launchedModel,
            "modelCapabilities must describe the launched model")
        if case .dynamic(let admissionConfiguration, _) = configuration.admission {
            self.admissionReducer = ServingAdmissionReducer(
                configuration: admissionConfiguration)
        }
    }

    /// `logit_bias`: the dense runtime applies the map per row, so a non-empty map is admissible only
    /// when the model's vocabulary width is known (`supportsLogitBias`), and every id must lie below it
    /// (an id at or above it would be an out-of-bounds index). Refused with a 400 (`param:
    /// "logit_bias"`, the scalar backend's texts) before any prompt render, admission or coordinator
    /// work. The HTTP layer gates on `supportsLogitBias` first; this is the same rule at the backend.
    private func rejectUnsupportedLogitBias(
        _ request: OpenAIChatCompletionRequest
    ) throws {
        guard !request.logitBias.isEmpty else { return }
        guard supportsLogitBias, let vocabularySize = configuration.logitBiasVocabularySize else {
            throw OpenAIServingError.invalidRequest(
                "logit_bias is not supported by the loaded model's decoding route",
                param: "logit_bias")
        }
        if let maximumID = request.logitBias.keys.max(), maximumID >= vocabularySize {
            throw OpenAIServingError.invalidRequest(
                "logit_bias token ids must be smaller than the model's vocabulary size "
                    + "(\(vocabularySize))",
                param: "logit_bias")
        }
    }

    /// Reject a request asking for decode controls the greedy-only continuous route can't apply
    /// (honesty: never accept sampling or penalties the compiled path would silently ignore). A
    /// policy-validation failure maps to a 400-class error.
    private static func rejectUnsupportedDecodeControls(
        _ request: OpenAIChatCompletionRequest
    ) throws {
        let policy: ServingSamplingPolicy
        do {
            policy = try ServingSamplingPolicy.resolve(from: request)
        } catch is ServingSamplingPolicyError {
            throw OpenAIServingError.invalidRequest(
                "invalid sampling parameters", param: "temperature")
        }
        if case .sampled = policy {
            throw OpenAIServingError.invalidRequest(
                "sampling (temperature > 0) is not supported on the continuous-batch route; "
                    + "send temperature 0 for greedy decoding",
                param: "temperature")
        }
        // `repetitionPenalty` is checked for non-neutrality via the shared `DecoderPenalties`
        // helper, not `(x ?? 0) != 0` like the additive presence/frequency knobs: the vendored
        // `RepetitionContext` is a MULTIPLICATIVE penalty (`x < 0 ? x * penalty : x / penalty`),
        // so its neutral element is `1`, not `0`. Both HF-recommended presets for the deployed
        // model send `repetition_penalty: 1.0`, which must stay admissible here.
        // `logit_bias` is NOT refused here: the dense runtime applies a per-row additive bias before
        // every greedy reduction. Its capability and vocabulary bound are enforced by
        // `rejectUnsupportedLogitBias(_:)` (called first in `start`); sampling and the penalties below
        // stay refused because the compiled path still has no processor for them.
        if (request.presencePenalty ?? 0) != 0 || (request.frequencyPenalty ?? 0) != 0
            || !DecoderPenalties.repetitionPenaltyIsNeutral(request.repetitionPenalty)
        {
            throw OpenAIServingError.invalidRequest(
                "presence/frequency/repetition penalties are not supported on the continuous-batch "
                    + "route (no logit processor on the compiled path)",
                param: "presence_penalty")
        }
    }

    /// `true` exactly when the model's vocabulary width is known (`logitBiasVocabularySize`), so every
    /// `logit_bias` id can be bounded before admission; the dense runtime applies the bias per row.
    /// An unknown width fails closed (the HTTP gate then refuses a non-empty map with a 400).
    public nonisolated var supportsLogitBias: Bool {
        configuration.logitBiasVocabularySize != nil
    }

    public func start(
        _ request: OpenAIChatCompletionRequest
    ) async throws -> ServingGenerationHandle {
        try await start(
            request,
            outputTokenTraceLimit: nil,
            completedTraceCapacity: 0)
    }

    @_spi(ProductionRouteEvidence)
    public func startProductionRouteEvidence(
        _ request: OpenAIChatCompletionRequest,
        authorization: ContinuousServingProductionRouteEvidenceAuthorization,
        tokenTrace: ContinuousServingOutputTokenTraceConfiguration
    ) async throws -> ServingGenerationHandle {
        guard authorization.authorizes(self) else {
            throw ContinuousServingProductionRouteEvidenceError
                .authorizationBackendMismatch
        }
        return try await start(
            request,
            outputTokenTraceLimit: try tokenTrace.traceLimit,
            completedTraceCapacity: tokenTrace.enabled
                ? tokenTrace.maxCompletedRequests
                : 0)
    }

    @_spi(ProductionRouteEvidence)
    public func takeProductionRouteCompletedTokenTraces(
        authorization: ContinuousServingProductionRouteEvidenceAuthorization
    ) throws -> [ContinuousServingCompletedRequestTokenTrace] {
        guard authorization.authorizes(self) else {
            throw ContinuousServingProductionRouteEvidenceError
                .authorizationBackendMismatch
        }
        let traces = completedTokenTraces
        completedTokenTraces.removeAll(keepingCapacity: true)
        return traces
    }

    @_spi(ProductionRouteEvidence)
    public func productionRouteScorecardTraceCapacity(
        authorization: ContinuousServingProductionRouteEvidenceAuthorization
    ) throws -> Int {
        guard authorization.authorizes(self) else {
            throw ContinuousServingProductionRouteEvidenceError
                .authorizationBackendMismatch
        }
        return configuration.scorecardTraceCapacity
    }

    private func start(
        _ request: OpenAIChatCompletionRequest,
        outputTokenTraceLimit: Int?,
        completedTraceCapacity: Int
    ) async throws -> ServingGenerationHandle {
        guard acceptingRequests else {
            throw ContinuousServingBackendError.shuttingDown
        }
        // The continuous-batch route only knows how to render a CHAT-templated prompt (it has no
        // tokenizer-only codepath); fail closed rather than silently mistreating raw completions text
        // as an already-templated chat prompt.
        guard request.promptInput == .chat else {
            throw OpenAIServingError.invalidRequestWithCode(
                "This server does not support the legacy text-completions route for the loaded backend",
                param: "prompt",
                code: "completions_unsupported")
        }
        // The continuous-batch route has no seam to observe a step's raw logits (its decode loop is
        // driven entirely inside the batched scheduler, with no per-request hook comparable to the
        // scalar route's detokenizer callback) — fail closed rather than silently omitting logprobs
        // from a response that claims to carry them.
        guard request.logprobsRequest == nil else {
            throw OpenAIServingError.invalidRequestWithCode(
                "This server does not support per-token logprobs on the loaded backend",
                param: "logprobs",
                code: "logprobs_unsupported")
        }
        // This route has no per-row constraint seam, so a named or `"required"` `tool_choice` cannot
        // be enforced: refuse it (400, `param: "tool_choice"`) before any admission or decode, never
        // answer in prose. Defense in depth behind the HTTP capability gate.
        try validateToolChoiceCapability(
            request: request, backendEnforcesToolChoice: enforcesForcingToolChoice)
        try request.requireLaunchedModel(launchedModel)
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw ContinuousServingBackendError.invalidStopTokenIDs
        }
        // `logit_bias` capability and vocabulary bound: before any prompt render or admission.
        try rejectUnsupportedLogitBias(request)
        // The continuous-batch route decodes greedily (the compiled step folds argmax); a sampled
        // request must be REJECTED here, never silently downgraded to greedy — the honesty
        // invariant. Native sampling is wired on the scalar route (hybrid models, e.g. Qwen3.8).
        try Self.rejectUnsupportedDecodeControls(request)

        let activeTools = request.activeTools
        // Resolve thinking ONCE and use the SAME value for the prompt render and the reasoning gate —
        // a closed `<think></think>` in the prompt with a splitter still engaged would mislabel the
        // answer. This route only accepts `.chat` prompts (raw-text is rejected above).
        let resolvedEnableThinking = request.resolvedEnableThinking(
            disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive)
        let separatesReasoning = servingSeparatesReasoning(
            thinksByDefault: configuration.thinksByDefault,
            resolvedEnableThinking: resolvedEnableThinking)
        let promptTokens = try codec.render(
            messages: request.messages,
            tools: activeTools,
            enableThinking: resolvedEnableThinking,
            reasoningEffort: request.reasoningEffort,
            addGenerationPrompt: request.addGenerationPrompt)
        guard !promptTokens.isEmpty else {
            throw ContinuousServingBackendError.emptyRenderedPrompt
        }
        let completionBudgetResolution = try configuration.modelCapabilities?
            .resolveCompletionBudget(
                requestedCompletionTokens: request.maxCompletionTokens,
                renderedPromptTokens: promptTokens.count,
                stream: request.stream)
        let maximumCompletionTokens =
            completionBudgetResolution?.appliedCompletionTokens
            ?? request.maxCompletionTokens
            ?? configuration.defaultMaximumCompletionTokens

        let prepared = PreparedRequest(
            id: ServingRequestID("continuous-\(UUID().uuidString)"),
            promptTokens: promptTokens,
            maximumCompletionTokens: maximumCompletionTokens,
            stopStrings: modelStopStrings.union(request.stop),
            ignoreEOS: request.ignoreEOS,
            logitBias: request.logitBias,
            activeTools: activeTools,
            parallelToolCalls: request.parallelToolCalls,
            completionBudgetResolution: completionBudgetResolution,
            separatesReasoning: separatesReasoning,
            outputTokenTraceLimit: outputTokenTraceLimit,
            completedTraceCapacity: completedTraceCapacity)
        switch configuration.admission {
        case .immediateBatchNoSpec:
            let handles = try await submitToCoordinator(
                [submission(for: prepared, route: .continuousBatchNoSpec)])
            guard let coordinatorHandle = handles.first else {
                throw ContinuousServingBackendError.shuttingDown
            }
            guard acceptingRequests else {
                _ = await coordinator.cancel(coordinatorHandle.id)
                throw ContinuousServingBackendError.shuttingDown
            }
            return activate(
                prepared,
                coordinatorHandle: coordinatorHandle,
                route: .continuousBatchNoSpec)

        case .dynamic:
            try await validateDynamicAdmission(prepared)
            return try await waitForDynamicAdmission(prepared)
        }
    }

    public func snapshot() async -> ContinuousServingBackendSnapshot {
        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        return ContinuousServingBackendSnapshot.current(
            activeRequests: requests.count,
            coordinatorSlots: slots.count,
            reservedKVBytes: resources?.reservedKVBytes ?? 0,
            maxReservedKVBytes: resources?.maxReservedKVBytes ?? 0)
    }

    public func takeScorecardContinuousRouteObservation()
        async -> ContinuousServingBackendScorecardObservation
    {
        await scorecardContinuousRouteObservation(consumingTraces: true)
    }

    public func scorecardContinuousRouteObservation()
        async -> ContinuousServingBackendScorecardObservation
    {
        await scorecardContinuousRouteObservation(consumingTraces: false)
    }

    private func scorecardContinuousRouteObservation(
        consumingTraces: Bool
    ) async -> ContinuousServingBackendScorecardObservation {
        let slots = await coordinator.snapshots()
        let resources = await coordinator.runtimeResourceSnapshot()
        let executionTrace: [ContinuousBatchCoordinatorEvent]
        let planObservations: [ContinuousBatchPlanObservation]
        let timingTrace: [ContinuousBatchTimingEvent]
        if consumingTraces {
            executionTrace = await coordinator.takeExecutionTrace()
            planObservations = await coordinator.takePlanObservations()
            timingTrace = await coordinator.takeTimingTrace()
        } else {
            executionTrace = await coordinator.executionTrace()
            planObservations = await coordinator.planObservations()
            timingTrace = await coordinator.timingObservations()
        }
        return ContinuousServingBackendScorecardObservation(
            activeRequests: requests.count,
            coordinatorRequestIDs: requests.values.map(\.coordinatorID).sorted(),
            coordinatorSlots: slots,
            runtimeResources: resources,
            executionTrace: executionTrace,
            planObservations: planObservations,
            timingTrace: timingTrace)
    }

    /// Package diagnostics deliberately expose only opaque coordinator IDs and bounded,
    /// prompt-free scheduler telemetry. They are used by authenticated serving evidence to
    /// prove cancellation and membership transitions without logging request content.
    func diagnosticCoordinatorRequestIDs() -> [BatchRequestID] {
        requests.values.map(\.coordinatorID).sorted()
    }

    func diagnosticPendingAdmissionRequestCount() -> Int {
        admissionReducer?.heldRequestIDs.count ?? 0
    }

    func diagnosticQueuedAdmissionRequestCount() -> Int {
        admissionReducer?.queuedRequestIDs.count ?? 0
    }

    func diagnosticExecutingAdmissionRequestCount() -> Int {
        admissionReducer?.executingRequestIDs.count ?? 0
    }

    func diagnosticAdmissionCoalescingWindowArmed() -> Bool {
        coalescingTask != nil
    }

    func diagnosticAcceptingRequests() -> Bool {
        acceptingRequests
    }

    func diagnosticExpireAdmissionCoalescingWindow() async {
        await expireAdmissionCoalescingWindow()
    }

    func diagnosticCoordinatorSnapshots() async -> [BatchSlotSnapshot] {
        await coordinator.snapshots()
    }

    func diagnosticCoordinatorExecutionTrace()
        async -> [ContinuousBatchCoordinatorEvent]
    {
        await coordinator.executionTrace()
    }

    func diagnosticRuntimeResourceSnapshot()
        async -> ContinuousBatchRuntimeResourceSnapshot?
    {
        await coordinator.runtimeResourceSnapshot()
    }

    @discardableResult
    func diagnosticTakeCoordinatorExecutionTrace()
        async -> [ContinuousBatchCoordinatorEvent]
    {
        await coordinator.takeExecutionTrace()
    }

    public func shutdown() async {
        if !acceptingRequests, requests.isEmpty, pendingAdmissions.isEmpty {
            await coordinator.shutdown()
            return
        }
        acceptingRequests = false
        coalescingTask?.cancel()
        coalescingTask = nil

        let pending = pendingAdmissions
        pendingAdmissions.removeAll()
        if var reducer = admissionReducer {
            for id in pending.keys {
                _ = reducer.cancel(id)
            }
            admissionReducer = reducer
        }
        for admission in pending.values {
            admission.continuation.resume(
                throwing: ContinuousServingBackendError.shuttingDown)
        }

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
            request.admissionReleased = true
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
        admittingRequestIDs.removeAll()
        cancelledAdmittingRequestIDs.removeAll()
    }

    private func waitForDynamicAdmission(
        _ prepared: PreparedRequest
    ) async throws -> ServingGenerationHandle {
        guard acceptingRequests else {
            throw ContinuousServingBackendError.shuttingDown
        }
        try Task.checkCancellation()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pendingAdmissions[prepared.id] = PendingAdmission(
                    prepared: prepared,
                    continuation: continuation)
                guard var reducer = admissionReducer else {
                    pendingAdmissions[prepared.id] = nil
                    continuation.resume(
                        throwing: ContinuousServingBackendError.shuttingDown)
                    return
                }
                let decision = reducer.submit(prepared.id)
                admissionReducer = reducer
                Task { [weak self] in
                    await self?.applyAdmissionDecision(decision)
                }
            }
        } onCancel: { [weak self] in
            Task {
                await self?.cancelPendingStart(id: prepared.id)
            }
        }
    }

    private func applyAdmissionDecision(
        _ decision: ServingAdmissionDecision
    ) async {
        guard acceptingRequests else { return }
        switch decision {
        case .held(let ids):
            if !ids.isEmpty { armAdmissionCoalescingWindowIfNeeded() }

        case .removedFromHold(let ids):
            if ids.isEmpty {
                disarmAdmissionCoalescingWindow()
            } else {
                armAdmissionCoalescingWindowIfNeeded()
            }

        case .queued, .noRouteChange, .idle:
            break

        case .rejected(let id, let reason):
            guard let pending = pendingAdmissions.removeValue(forKey: id) else {
                return
            }
            switch reason {
            case .queueFull:
                pending.continuation.resume(
                    throwing: ServingBackendAdmissionError.queueFull(
                        retryAfterSeconds: configuration.queueRetryAfterSeconds))
            case .duplicateRequest:
                pending.continuation.resume(
                    throwing: ServingBackendAdmissionError.capacityExceeded(
                        retryAfterSeconds: configuration.queueRetryAfterSeconds))
            }

        case .start(let route, let ids):
            await admitPending(ids, route: route)

        case .joinedContinuousBatch(let ids):
            await admitPending(
                ids.filter { pendingAdmissions[$0] != nil },
                route: .continuousBatchNoSpec)

        case .continuedExecution(let route, _, let replacements):
            await admitPending(replacements, route: route)

        case .removedFromExecution(_, let replacements, let nextHeld):
            if !replacements.isEmpty {
                await admitPending(
                    replacements,
                    route: .continuousBatchNoSpec)
            }
            if !nextHeld.isEmpty {
                armAdmissionCoalescingWindowIfNeeded()
            }
        }
    }

    private func armAdmissionCoalescingWindowIfNeeded() {
        guard coalescingTask == nil,
            admissionReducer?.currentExecutionRoute == nil,
            admissionReducer?.heldRequestIDs.isEmpty == false
        else {
            return
        }
        guard case .dynamic(_, let coalescing) = configuration.admission else {
            return
        }
        switch coalescing {
        case .manualDiagnostic:
            return
        case .automatic(let duration):
            coalescingTask = Task { [weak self] in
                do {
                    try await Task.sleep(for: duration)
                } catch {
                    return
                }
                await self?.expireAdmissionCoalescingWindow()
            }
        }
    }

    private func disarmAdmissionCoalescingWindow() {
        coalescingTask?.cancel()
        coalescingTask = nil
    }

    private func expireAdmissionCoalescingWindow() async {
        coalescingTask?.cancel()
        coalescingTask = nil
        guard acceptingRequests, var reducer = admissionReducer else { return }
        let decision = reducer.coalescingExpired()
        admissionReducer = reducer
        await applyAdmissionDecision(decision)
    }

    private func admitPending(
        _ ids: [ServingRequestID],
        route: ServingExecutionRoute
    ) async {
        var uniqueIDs: Set<ServingRequestID> = []
        let candidateIDs = ids.filter {
            uniqueIDs.insert($0).inserted
                && pendingAdmissions[$0] != nil
                && !admittingRequestIDs.contains($0)
        }
        let candidates = candidateIDs.compactMap { pendingAdmissions[$0] }
        guard !candidates.isEmpty else { return }
        admittingRequestIDs.formUnion(candidateIDs)
        let coordinatorHandles: [ContinuousBatchRequestHandle]
        do {
            coordinatorHandles = try await submitToCoordinator(
                candidates.map { submission(for: $0.prepared, route: route) })
        } catch {
            admittingRequestIDs.subtract(candidateIDs)
            cancelledAdmittingRequestIDs.subtract(candidateIDs)
            for candidate in candidates {
                if let pending = pendingAdmissions.removeValue(
                    forKey: candidate.prepared.id)
                {
                    pending.continuation.resume(throwing: error)
                }
            }
            await releaseAdmission(candidates.map(\.prepared.id))
            return
        }
        admittingRequestIDs.subtract(candidateIDs)

        guard acceptingRequests else {
            for handle in coordinatorHandles {
                _ = await coordinator.cancel(handle.id)
            }
            for candidate in candidates {
                if let pending = pendingAdmissions.removeValue(
                    forKey: candidate.prepared.id)
                {
                    pending.continuation.resume(
                        throwing: ContinuousServingBackendError.shuttingDown)
                }
            }
            cancelledAdmittingRequestIDs.subtract(candidateIDs)
            return
        }

        var cancelledIDs: [ServingRequestID] = []
        for (candidate, coordinatorHandle) in zip(
            candidates,
            coordinatorHandles)
        {
            guard let pending = pendingAdmissions.removeValue(
                forKey: candidate.prepared.id)
            else {
                _ = await coordinator.cancel(coordinatorHandle.id)
                if cancelledAdmittingRequestIDs.remove(
                    candidate.prepared.id
                ) != nil {
                    cancelledIDs.append(candidate.prepared.id)
                }
                continue
            }
            let handle = activate(
                pending.prepared,
                coordinatorHandle: coordinatorHandle,
                route: route)
            pending.continuation.resume(returning: handle)
        }
        for id in cancelledIDs {
            await cancelAdmission(id)
        }
    }

    private func submission(
        for prepared: PreparedRequest,
        route: ServingExecutionRoute
    ) -> ContinuousBatchSubmission {
        ContinuousBatchSubmission(
            promptTokens: prepared.promptTokens,
            maxOutputTokens: prepared.maximumCompletionTokens,
            stopTokenIDs: stopTokenIDs,
            architecture: .denseAttention,
            // A biased row is never speculative (the runtime refuses the pair, and the verify step must
            // never see a biased row): a biased request on the solo-PLD route decodes without
            // speculation. An unbiased request on that route still requests it.
            requestsSpeculation: route == .soloPLD && prepared.logitBias.isEmpty,
            ignoresStopTokens: prepared.ignoreEOS,
            logitBias: prepared.logitBias)
    }

    private func submitToCoordinator(
        _ submissions: [ContinuousBatchSubmission]
    ) async throws -> [ContinuousBatchRequestHandle] {
        do {
            return try await coordinator.submitBatch(submissions)
        } catch ContinuousBatchSchedulerError.queueCapacityExceeded {
            throw ServingBackendAdmissionError.queueFull(
                retryAfterSeconds: configuration.queueRetryAfterSeconds)
        } catch let runtimeError as DenseContinuousBatchRuntimeError {
            switch runtimeError {
            case .contextLimitExceeded, .positionOverflow,
                .kvByteAccountingOverflow,
                .requestReservedContextLimitExceeded,
                .requestReservedKVByteLimitExceeded:
                throw ServingBackendAdmissionError.requestTooLarge()
            case .aggregateContextLimitExceeded,
                .aggregateKVByteLimitExceeded:
                throw ServingBackendAdmissionError.capacityExceeded(
                    retryAfterSeconds:
                        configuration.queueRetryAfterSeconds)
            default:
                throw runtimeError
            }
        }
    }

    private func validateDynamicAdmission(
        _ prepared: PreparedRequest
    ) async throws {
        guard case .dynamic(let admission, _) = configuration.admission else {
            return
        }
        let route: ServingExecutionRoute = admission.soloPLDQualified
            ? .soloPLD
            : .continuousBatchNoSpec
        do {
            try await coordinator.validateSubmission(
                submission(for: prepared, route: route))
        } catch let runtimeError as DenseContinuousBatchRuntimeError {
            switch runtimeError {
            case .contextLimitExceeded, .positionOverflow,
                .kvByteAccountingOverflow,
                .requestReservedContextLimitExceeded,
                .requestReservedKVByteLimitExceeded:
                throw ServingBackendAdmissionError.requestTooLarge()
            case .aggregateContextLimitExceeded,
                .aggregateKVByteLimitExceeded:
                throw ServingBackendAdmissionError.capacityExceeded(
                    retryAfterSeconds:
                        configuration.queueRetryAfterSeconds)
            default:
                throw runtimeError
            }
        }
    }

    private func activate(
        _ prepared: PreparedRequest,
        coordinatorHandle: ContinuousBatchRequestHandle,
        route: ServingExecutionRoute
    ) -> ServingGenerationHandle {
        let id = prepared.id
        let responseID = "chatcmpl-\(UUID().uuidString)"
        let mailbox = BoundedDeltaMailbox(capacity: configuration.mailboxCapacity)
        let lease = ServingRequestLease(
            id: id,
            onCancelWithReason: { [weak self] reason in
                await self?.cancel(id: id, reason: reason)
            })
        let toolCallProcessor: ToolCallProcessor? =
            prepared.activeTools.isEmpty
            ? nil
            : ToolCallProcessor(
                format: configuration.toolCallFormat,
                tools: prepared.activeTools.compactMap { $0.raw.asObjectSendable })
        requests[id] = ActiveRequest(
            coordinatorID: coordinatorHandle.id,
            promptTokenCount: prepared.promptTokens.count,
            maximumCompletionTokens: prepared.maximumCompletionTokens,
            mailbox: mailbox,
            lease: lease,
            task: nil,
            cancellationReason: nil,
            admissionReleased: admissionReducer == nil,
            detokenizer: codec.makeDetokenizer(),
            stopFilter: ServingStopStringFilter(
                stopStrings: prepared.stopStrings),
            toolCallProcessor: toolCallProcessor,
            reasoningGate: toolCallProcessor != nil && prepared.separatesReasoning
                ? ReasoningPhaseToolCallGate() : nil,
            parallelToolCalls: prepared.parallelToolCalls,
            responseID: responseID,
            outputTokenTraceLimit: prepared.outputTokenTraceLimit,
            completedTraceCapacity: prepared.completedTraceCapacity)

        let task = Task { [weak self] in
            guard let self else { return }
            await self.execute(
                id: id,
                tokens: coordinatorHandle.tokens)
        }
        requests[id]?.task = task

        return ServingGenerationHandle(
            responseID: responseID,
            created: Int(Date().timeIntervalSince1970),
            model: launchedModel,
            route: route,
            mailbox: mailbox,
            lease: lease,
            completionBudgetResolution: prepared.completionBudgetResolution,
            separatesReasoning: prepared.separatesReasoning)
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
            // Held reasoning bytes and the stop-filter tail, in stream order, through gate and parser.
            try await flushEndOfStream(for: id, includeStopFilterTail: !stoppedByString)
            guard let request = requests[id] else {
                throw CancellationError()
            }
            var finishReason: OpenAIChatFinishReason
            if stoppedByString {
                finishReason = .stop
            } else if request.generatedTokenCount >= request.maximumCompletionTokens {
                finishReason = .length
            } else {
                finishReason = .stop
            }
            if let processor = request.toolCallProcessor {
                // Preserve residual buffered text that was not a tool call (see scalar backend).
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
                            promptTokens: request.promptTokenCount,
                            completionTokens: request.generatedTokenCount))))
            try recordCompletedTokenTraceIfNeeded(request)
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

        guard let completed = requests.removeValue(forKey: id) else { return }
        if !completed.admissionReleased {
            await releaseAdmission([id])
        }
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
        if let limit = request.outputTokenTraceLimit {
            if request.outputTokenTrace.count < limit {
                request.outputTokenTrace.append(token)
            } else {
                request.outputTokenTraceTruncated = true
            }
        }
        request.detokenizer.append(token: token)
        let chunk = request.detokenizer.next()
        let output = chunk.map { request.stopFilter.process($0) }
        requests[id] = request

        var stopped = output?.stopped == true
        if stopped {
            _ = await coordinator.cancel(request.coordinatorID)
            try Task.checkCancellation()
        }
        if let text = output?.text {
            if let processor = request.toolCallProcessor {
                var parserText = text
                var gated = false
                if var gate = request.reasoningGate {
                    // Reasoning (through the first `</think>`) is shown as content, never parsed.
                    let routed = gate.route(text)
                    requests[id]?.reasoningGate = gate
                    if !routed.passthrough.isEmpty {
                        try await request.mailbox.send(.text(routed.passthrough))
                    }
                    parserText = routed.toParser
                    gated = true
                }
                if gated && parserText.isEmpty {
                    // Nothing reached the parser (all reasoning, or a held tag prefix).
                } else if let display = processor.processChunk(parserText), !display.isEmpty
                {
                    try await request.mailbox.send(.text(display))
                }
            } else {
                try await request.mailbox.send(.text(text))
            }
        }
        // `parallel_tool_calls: false` caps the reply at the first call, so release the slot on
        // the token that closes it. Must run AFTER `processChunk` above: the call only becomes
        // "completed" once this token's text has been fed to the processor.
        if !stopped, let processor = request.toolCallProcessor,
            OpenAIChatCompletionRequest.shouldStopAfterToolCalls(
                parallelToolCalls: request.parallelToolCalls,
                completedCount: processor.toolCalls.count)
        {
            _ = await coordinator.cancel(request.coordinatorID)
            stopped = true
        }
        try Task.checkCancellation()
        return stopped
    }

    private func recordCompletedTokenTraceIfNeeded(
        _ request: ActiveRequest
    ) throws {
        guard request.outputTokenTraceLimit != nil else { return }
        guard completedTokenTraces.count < request.completedTraceCapacity else {
            throw ContinuousServingProductionRouteEvidenceError
                .completedTraceCapacityExceeded(
                    limit: request.completedTraceCapacity)
        }
        completedTokenTraces.append(
            ContinuousServingCompletedRequestTokenTrace(
                responseID: request.responseID,
                coordinatorRequestID: request.coordinatorID,
                outputTokenIDs: request.outputTokenTrace,
                completionTokenCount: request.generatedTokenCount,
                truncated: request.outputTokenTraceTruncated))
    }

    /// End of stream: the stop filter's held tail is the LAST text in stream order, so it is routed
    /// exactly like a chunk -- through the reasoning gate (which also releases a held `</think>`
    /// prefix) and then the tool-call processor -- rather than sent to the client directly. The
    /// processor's own residual (earlier text) is then released by `processEOS`, so the client sees
    /// the stream in order and the parser sees exactly what the client is shown as content.
    ///
    /// A request stopped by a stop string or by `parallel_tool_calls: false` (`stoppedByString`) never
    /// emitted its held tail, so the tail is not read there; only a held `</think>` prefix is released.
    private func flushEndOfStream(for id: ServingRequestID, includeStopFilterTail: Bool) async throws {
        guard var request = requests[id] else {
            throw CancellationError()
        }
        let tail = includeStopFilterTail ? request.stopFilter.finish() : nil
        var passthrough = ""
        var parserText = tail ?? ""
        if var gate = request.reasoningGate {
            let routed = gate.finish(tail: tail)
            request.reasoningGate = gate
            passthrough = routed.passthrough
            parserText = routed.toParser
        }
        requests[id] = request
        if let processor = request.toolCallProcessor {
            if !passthrough.isEmpty {
                try await request.mailbox.send(.text(passthrough))
            }
            if !parserText.isEmpty, let display = processor.processChunk(parserText), !display.isEmpty {
                try await request.mailbox.send(.text(display))
            }
        } else {
            let text = passthrough + parserText
            if !text.isEmpty {
                try await request.mailbox.send(.text(text))
            }
        }
    }

    private func cancelPendingStart(id: ServingRequestID) async {
        if let pending = pendingAdmissions.removeValue(forKey: id) {
            pending.continuation.resume(throwing: CancellationError())
            if admittingRequestIDs.contains(id) {
                cancelledAdmittingRequestIDs.insert(id)
                return
            }
            guard var reducer = admissionReducer else { return }
            let decision = reducer.cancel(id)
            admissionReducer = reducer
            await applyAdmissionDecision(decision)
            return
        }
        if requests[id] != nil {
            await cancel(id: id, reason: .clientDisconnected)
        }
    }

    private func releaseAdmission(
        _ ids: [ServingRequestID]
    ) async {
        guard var reducer = admissionReducer else { return }
        let decision = reducer.executionFinished(requests: ids)
        admissionReducer = reducer
        await applyAdmissionDecision(decision)
    }

    private func cancelAdmission(
        _ id: ServingRequestID
    ) async {
        guard var reducer = admissionReducer else { return }
        let decision = reducer.cancel(id)
        admissionReducer = reducer
        await applyAdmissionDecision(decision)
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
        let shouldReleaseAdmission = !request.admissionReleased
        request.admissionReleased = true
        request.task?.cancel()
        requests[id] = request
        _ = await coordinator.cancel(request.coordinatorID)
        if shouldReleaseAdmission {
            await cancelAdmission(id)
        }
        await request.mailbox.cancel(reason)
        requests[id] = nil
    }
}
