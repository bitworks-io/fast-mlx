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
    /// Immutable model/host-fit capability. Production serving always supplies this; nil retains
    /// the legacy embedding contract for existing library callers and fixtures.
    public let modelCapabilities: ServingModelCapabilities?

    public init(
        defaultMaximumCompletionTokens: Int,
        queueRetryAfterSeconds: Int,
        mailboxCapacity: BoundedDeltaMailbox.Capacity,
        scorecardTraceCapacity: Int = 0,
        admission: ContinuousServingAdmissionMode = .immediateBatchNoSpec,
        toolCallFormat: ToolCallFormat = .json,
        disableThinkingWhenToolsActive: Bool = false,
        modelCapabilities: ServingModelCapabilities? = nil
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
        self.modelCapabilities = modelCapabilities
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
        let activeTools: [OpenAIToolSpec]
        let completionBudgetResolution: ServingCompletionBudgetResolution?
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
        if (request.presencePenalty ?? 0) != 0 || (request.frequencyPenalty ?? 0) != 0
            || (request.repetitionPenalty ?? 0) != 0
        {
            throw OpenAIServingError.invalidRequest(
                "presence/frequency/repetition penalties are not supported on the continuous-batch "
                    + "route (no logit processor on the compiled path)",
                param: "presence_penalty")
        }
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
        try request.requireLaunchedModel(launchedModel)
        guard !stopTokenIDs.isEmpty, stopTokenIDs.allSatisfy({ $0 >= 0 }) else {
            throw ContinuousServingBackendError.invalidStopTokenIDs
        }
        // The continuous-batch route decodes greedily (the compiled step folds argmax); a sampled
        // request must be REJECTED here, never silently downgraded to greedy — the honesty
        // invariant. Native sampling is wired on the scalar route (hybrid models, e.g. Qwen3.8).
        try Self.rejectUnsupportedDecodeControls(request)

        let activeTools = request.activeTools
        let promptTokens = try codec.render(
            messages: request.messages,
            tools: activeTools,
            enableThinking: request.resolvedEnableThinking(
                disableThinkingWhenToolsActive: configuration.disableThinkingWhenToolsActive),
            reasoningEffort: request.reasoningEffort)
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
            activeTools: activeTools,
            completionBudgetResolution: completionBudgetResolution,
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
            requestsSpeculation: route == .soloPLD)
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
            completionBudgetResolution: prepared.completionBudgetResolution)
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

        let stopped = output?.stopped == true
        if stopped {
            _ = await coordinator.cancel(request.coordinatorID)
            try Task.checkCancellation()
        }
        if let text = output?.text {
            if let processor = request.toolCallProcessor {
                if let display = processor.processChunk(text), !display.isEmpty {
                    try await request.mailbox.send(.text(display))
                }
            } else {
                try await request.mailbox.send(.text(text))
            }
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
