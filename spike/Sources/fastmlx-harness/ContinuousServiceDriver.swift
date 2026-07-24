import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

enum ContinuousServiceDriverError: Error, CustomStringConvertible {
    case emptyBurst
    case emptyPrompt(Int)
    case invalidOutputBudget(Int)
    case invalidCancellationPromptCount(Int)
    case invalidCancellationWaitLimit(Double)
    case missingRuntimeResources
    case cancellationTargetNeverDecoded(BatchRequestID)
    case cancellationSharedBatchMissing
    case decodeFirstInterleaveMissing
    case initialSlotUnavailable(BatchRequestID)
    case unexpectedCancellationResult(BatchCancellationResult)
    case cancellationSlotRetained(BatchRequestID)
    case cancellationEventMissing(BatchRequestID)
    case replacementWasNotQueued(BatchRequestID)
    case replacementSlotMissing(BatchRequestID)
    case emptyRecoveryOutput(String)
    case missingCompletionTiming(BatchRequestID)
    case tokenTimingCountMismatch(BatchRequestID, expected: Int, actual: Int)

    var description: String {
        switch self {
        case .emptyBurst:
            return "continuous service burst must contain at least one request"
        case .emptyPrompt(let index):
            return "continuous service prompt \(index) is empty"
        case .invalidOutputBudget(let budget):
            return "continuous service output budget must be positive; actual=\(budget)"
        case .invalidCancellationPromptCount(let count):
            return "continuous cancellation recovery requires at least three prompts; actual=\(count)"
        case .invalidCancellationWaitLimit(let seconds):
            return "continuous cancellation wait limit must be finite and positive; actual=\(seconds)"
        case .missingRuntimeResources:
            return "continuous service runtime did not expose byte-admission resources"
        case .cancellationTargetNeverDecoded(let id):
            return "cancellation target \(id.rawValue) never reached decoding"
        case .cancellationSharedBatchMissing:
            return "cancellation scenario never formed the required shared batch"
        case .decodeFirstInterleaveMissing:
            return "short request never decoded beside long-request prefill"
        case .initialSlotUnavailable(let id):
            return "initial cancellation request \(id.rawValue) was not active at saturation"
        case .unexpectedCancellationResult(let result):
            return "unexpected cancellation result: \(result)"
        case .cancellationSlotRetained(let id):
            return "cancelled request \(id.rawValue) remained in the scheduler"
        case .cancellationEventMissing(let id):
            return "cancelled request \(id.rawValue) emitted no cancellation event"
        case .replacementWasNotQueued(let id):
            return "replacement request \(id.rawValue) did not wait for the occupied slot"
        case .replacementSlotMissing(let id):
            return "replacement request \(id.rawValue) did not reuse the released slot"
        case .emptyRecoveryOutput(let request):
            return "\(request) emitted no visible recovery tokens"
        case .missingCompletionTiming(let id):
            return "continuous service request \(id.rawValue) has no actor completion timestamp"
        case .tokenTimingCountMismatch(let id, let expected, let actual):
            return "continuous service request \(id.rawValue) published \(actual) timestamps for \(expected) tokens"
        }
    }
}

struct ContinuousServiceLoadConfiguration: Sendable {
    let maxActiveSlots: Int
    let maxPrefillSlots: Int
    let prefillChunkSize: Int
    let maxQueuedRequests: Int
    let maxReservedContextTokens: Int?
    let maxReservedKVBytes: Int
    let traceLimit: Int

    init(
        maxActiveSlots: Int,
        maxPrefillSlots: Int,
        prefillChunkSize: Int,
        maxQueuedRequests: Int = 256,
        maxReservedContextTokens: Int? = nil,
        maxReservedKVBytes: Int,
        traceLimit: Int
    ) {
        self.maxActiveSlots = maxActiveSlots
        self.maxPrefillSlots = maxPrefillSlots
        self.prefillChunkSize = prefillChunkSize
        self.maxQueuedRequests = maxQueuedRequests
        self.maxReservedContextTokens = maxReservedContextTokens
        self.maxReservedKVBytes = maxReservedKVBytes
        self.traceLimit = traceLimit
    }
}

struct ContinuousServiceRunObservation: Sendable {
    let metrics: ServiceRunMetrics
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
    let memorySamples: [ServiceMemorySample]
    let resources: ContinuousBatchRuntimeResourceSnapshot
    let outputTokens: [[Int]]
}

struct ContinuousServiceCancellationObservation: Sendable {
    let timeline: ServiceCancellationTimeline
    let previousPhase: String
    let slotRemoved: Bool
    let cancellationEventObserved: Bool
    let repeatedCancellationNotFound: Bool
    let initialActiveRequestCount: Int
    let replacementWasQueued: Bool
    let replacementSlotReused: Bool
    let sharedBatchObserved: Bool
    let decodeFirstInterleaveObserved: Bool
    let cancelledPrefixTokens: Int
    let survivorOutputTokens: Int
    let replacementOutputTokens: Int
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
    let resourcesAtAdmission: ContinuousBatchRuntimeResourceSnapshot
    let resourcesBeforeCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterReplacement: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
}

private struct CollectedServiceRequest: Sendable {
    let index: Int
    let id: BatchRequestID
    let promptTokenCount: Int
    let tokens: [Int]
}

/// Service-specific driver over the actor-confined continuous runtime. It is intentionally
/// separate from `EngineDriver`: that protocol describes one all-at-once request, while this
/// seam measures shared admission, bounded prompt work, and membership changes.
struct ContinuousSwiftServiceDriver: Sendable {
    let coordinator: ContinuousBatchCoordinator
    let eos: Int

    func runBurst(
        prompts: [[Int]],
        maxOutputTokens: Int
    ) async throws -> ContinuousServiceRunObservation {
        guard !prompts.isEmpty else { throw ContinuousServiceDriverError.emptyBurst }
        guard maxOutputTokens > 0 else {
            throw ContinuousServiceDriverError.invalidOutputBudget(maxOutputTokens)
        }
        for (index, prompt) in prompts.enumerated() where prompt.isEmpty {
            throw ContinuousServiceDriverError.emptyPrompt(index)
        }

        _ = await coordinator.takeExecutionTrace()
        _ = await coordinator.takeTimingTrace()
        // MLX peak is process-global and otherwise retains model-load/warmup history. Reset only
        // the counter (not active/cache allocations) so each run's peak describes that burst.
        Memory.peakMemory = 0
        guard let resourcesBeforeRun = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }
        let memoryStart = serviceMemorySample()
        let submittedAt = serviceClockSeconds()
        let handles = try await coordinator.submitBatch(
            prompts.map {
                ContinuousBatchSubmission(
                    promptTokens: $0,
                    maxOutputTokens: maxOutputTokens,
                    eosToken: eos,
                    architecture: .denseAttention)
            })
        guard let resources = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }

        var ticks: [ServiceTickObservation] = []
        var memory = [memoryStart]
        var tokenTimes: [BatchRequestID: [Double]] = [:]
        var completionTimes: [BatchRequestID: Double] = [:]
        while true {
            let remains = try await coordinator.runOneTick()
            let events = await coordinator.takeExecutionTrace()
            for timing in await coordinator.takeTimingTrace() {
                switch timing {
                case .emitted(let id, let timestamp):
                    tokenTimes[id, default: []].append(timestamp)
                case .finished(let id, let timestamp):
                    completionTimes[id] = timestamp
                }
            }
            let snapshots = await coordinator.snapshots()
            let queued = snapshots.filter {
                if case .queued = $0.phase { return true }
                return false
            }.count
            ticks.append(
                ServiceTickObservation(
                    activeSlots: snapshots.count - queued,
                    queuedSlots: queued,
                    operations: events.compactMap {
                        if case .operation(let operation) = $0 { return operation }
                        return nil
                    }))
            memory.append(serviceMemorySample())
            if !remains { break }
        }

        let collected = try await withThrowingTaskGroup(
            of: CollectedServiceRequest.self,
            returning: [CollectedServiceRequest].self
        ) { group in
            for (index, handle) in handles.enumerated() {
                let promptCount = prompts[index].count
                group.addTask {
                    var tokens: [Int] = []
                    for try await token in handle.tokens { tokens.append(token) }
                    return CollectedServiceRequest(
                        index: index,
                        id: handle.id,
                        promptTokenCount: promptCount,
                        tokens: tokens)
                }
            }
            var result: [CollectedServiceRequest] = []
            for try await request in group { result.append(request) }
            return result.sorted { $0.index < $1.index }
        }

        var timelines: [ServiceRequestTimeline] = []
        for request in collected {
            guard let completedAt = completionTimes[request.id] else {
                throw ContinuousServiceDriverError.missingCompletionTiming(request.id)
            }
            let times = tokenTimes[request.id, default: []]
            guard times.count == request.tokens.count else {
                throw ContinuousServiceDriverError.tokenTimingCountMismatch(
                    request.id, expected: request.tokens.count, actual: times.count)
            }
            timelines.append(
                ServiceRequestTimeline(
                    requestID: request.id,
                    promptTokenCount: request.promptTokenCount,
                    submittedAt: submittedAt,
                    tokenTimes: times,
                    completedAt: completedAt))
        }
        guard let completionResources = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }
        return ContinuousServiceRunObservation(
            metrics: try measureServiceRun(timelines),
            operations: summarizeServiceOperations(
                ticks,
                speculationStartSnapshot: resourcesBeforeRun.speculation,
                speculationEndSnapshot: completionResources.speculation),
            memory: try summarizeServiceMemory(memory),
            memorySamples: memory,
            resources: resources,
            outputTokens: collected.map(\.tokens))
    }

    /// Fills the configured active slots, queues a replacement, cancels one live batch row,
    /// then proves the queued request takes the released slot and joins a shared decode.
    /// The cancellation timestamp ends only after the coordinator has removed scheduler and
    /// runtime state, so it measures disconnect-to-removal rather than stream-consumer delay.
    func runCancellationRecovery(
        prompts: [[Int]],
        maxOutputTokens: Int,
        cancellationWaitLimitSeconds: Double
    ) async throws -> ContinuousServiceCancellationObservation {
        guard prompts.count >= 3 else {
            throw ContinuousServiceDriverError.invalidCancellationPromptCount(prompts.count)
        }
        guard maxOutputTokens > 1 else {
            throw ContinuousServiceDriverError.invalidOutputBudget(maxOutputTokens)
        }
        guard cancellationWaitLimitSeconds.isFinite, cancellationWaitLimitSeconds > 0 else {
            throw ContinuousServiceDriverError.invalidCancellationWaitLimit(
                cancellationWaitLimitSeconds)
        }
        for (index, prompt) in prompts.enumerated() where prompt.isEmpty {
            throw ContinuousServiceDriverError.emptyPrompt(index)
        }

        _ = await coordinator.takeExecutionTrace()
        _ = await coordinator.takeTimingTrace()
        Memory.peakMemory = 0
        guard let resourcesBeforeRun = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }
        var memory = [serviceMemorySample()]
        var ticks: [ServiceTickObservation] = []
        let initial = try await coordinator.submitBatch(
            prompts.dropLast().map {
                ContinuousBatchSubmission(
                    promptTokens: $0,
                    maxOutputTokens: maxOutputTokens,
                    eosToken: eos,
                    architecture: .denseAttention)
            })
        let survivor = initial[0]
        let target = initial[1]
        let targetConsumer = Task {
            for try await _ in target.tokens {}
        }
        await Task.yield()
        guard let resourcesAtAdmission = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }

        var sharedBatchObserved = false
        var decodeFirstInterleaveObserved = false
        while true {
            let observed = try await observedServiceTick(coordinator)
            ticks.append(observed.tick)
            memory.append(observed.memory)
            let hasPrefill = observed.tick.operations.contains {
                if case .prefill = $0 { return true }
                return false
            }
            let hasTargetDecode = observed.tick.operations.contains {
                switch $0 {
                case .decode(.solo(let id, _)), .decode(.drainSoloPipeline(let id)):
                    return id == target.id
                case .decode(.batch(let ids, _)):
                    return ids.contains(target.id)
                default:
                    return false
                }
            }
            decodeFirstInterleaveObserved = decodeFirstInterleaveObserved
                || (hasPrefill && hasTargetDecode)
            sharedBatchObserved = sharedBatchObserved || observed.tick.operations.contains {
                if case .decode(.batch(let ids, speculationAllowed: false)) = $0 {
                    return ids.contains(survivor.id) && ids.contains(target.id)
                }
                return false
            }
            if case .decoding(let emitted, _) = await coordinator.snapshot(for: target.id)?.phase,
                emitted >= 1,
                sharedBatchObserved
            {
                break
            }
            guard observed.remains else {
                throw ContinuousServiceDriverError.cancellationTargetNeverDecoded(target.id)
            }
        }
        guard sharedBatchObserved else {
            throw ContinuousServiceDriverError.cancellationSharedBatchMissing
        }
        guard decodeFirstInterleaveObserved else {
            throw ContinuousServiceDriverError.decodeFirstInterleaveMissing
        }
        for handle in initial {
            guard let snapshot = await coordinator.snapshot(for: handle.id) else {
                throw ContinuousServiceDriverError.initialSlotUnavailable(handle.id)
            }
            if case .queued = snapshot.phase {
                throw ContinuousServiceDriverError.initialSlotUnavailable(handle.id)
            }
        }

        let replacement = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: prompts.last!,
                maxOutputTokens: maxOutputTokens,
                eosToken: eos,
                architecture: .denseAttention))
        let replacementWasQueued: Bool
        if let snapshot = await coordinator.snapshot(for: replacement.id),
            case .queued = snapshot.phase
        {
            replacementWasQueued = true
        } else {
            throw ContinuousServiceDriverError.replacementWasNotQueued(replacement.id)
        }
        guard let resourcesBeforeCancellation = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }

        let requestedAt = serviceClockSeconds()
        targetConsumer.cancel()
        do {
            try await targetConsumer.value
        } catch is CancellationError {
            // Expected for a consumer-side disconnect.
        }
        while await coordinator.snapshot(for: target.id) != nil {
            guard serviceClockSeconds() - requestedAt <= cancellationWaitLimitSeconds else {
                throw ContinuousServiceDriverError.cancellationSlotRetained(target.id)
            }
            await Task.yield()
        }
        let removedAt = serviceClockSeconds()
        let cancellationEvents = await coordinator.takeExecutionTrace()
        guard let result = cancellationEvents.compactMap({ event -> BatchCancellationResult? in
            if case .cancelled(let result) = event, result.id == target.id { return result }
            return nil
        }).first else {
            throw ContinuousServiceDriverError.cancellationEventMissing(target.id)
        }
        let previousPhase: BatchSlotPhase
        guard case .cancelled(_, let phase) = result else {
            throw ContinuousServiceDriverError.unexpectedCancellationResult(result)
        }
        previousPhase = phase
        let slotRemoved = await coordinator.snapshot(for: target.id) == nil
        guard slotRemoved else {
            throw ContinuousServiceDriverError.cancellationSlotRetained(target.id)
        }
        guard let resourcesAfterCancellation = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }
        let cancellationEventObserved = cancellationEvents.contains(.cancelled(result))
        guard cancellationEventObserved else {
            throw ContinuousServiceDriverError.cancellationEventMissing(target.id)
        }
        let repeatedCancellationNotFound = await coordinator.cancel(target.id) == .notFound(target.id)
        memory.append(serviceMemorySample())

        var replacementSlotReused = false
        var resourcesAfterReplacement: ContinuousBatchRuntimeResourceSnapshot?
        while true {
            let observed = try await observedServiceTick(coordinator)
            ticks.append(observed.tick)
            memory.append(observed.memory)
            let replacementAdvanced = observed.tick.operations.contains { operation in
                switch operation {
                case .prefill(let slice):
                    return slice.id == replacement.id
                case .decode(.solo(let id, _)), .decode(.drainSoloPipeline(let id)):
                    return id == replacement.id
                case .decode(.batch(let ids, _)):
                    return ids.contains(replacement.id)
                }
            }
            if replacementAdvanced, resourcesAfterReplacement == nil {
                resourcesAfterReplacement = await coordinator.runtimeResourceSnapshot()
            }
            replacementSlotReused = replacementSlotReused || observed.tick.operations.contains {
                if case .decode(.batch(let ids, speculationAllowed: false)) = $0 {
                    return ids.contains(survivor.id) && ids.contains(replacement.id)
                }
                return false
            }
            if !observed.remains { break }
        }
        guard replacementSlotReused else {
            throw ContinuousServiceDriverError.replacementSlotMissing(replacement.id)
        }
        guard let resourcesAfterReplacement else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }

        var survivorOutputTokens = 0
        for (index, handle) in initial.enumerated() where index != 1 {
            let tokens = try await collectServiceTokens(handle.tokens)
            guard !tokens.isEmpty else {
                throw ContinuousServiceDriverError.emptyRecoveryOutput("survivor \(index)")
            }
            survivorOutputTokens += tokens.count
        }
        let replacementTokens = try await collectServiceTokens(replacement.tokens)
        guard !replacementTokens.isEmpty else {
            throw ContinuousServiceDriverError.emptyRecoveryOutput("replacement")
        }
        guard let resourcesAtEnd = await coordinator.runtimeResourceSnapshot() else {
            throw ContinuousServiceDriverError.missingRuntimeResources
        }
        memory.append(serviceMemorySample())

        return ContinuousServiceCancellationObservation(
            timeline: ServiceCancellationTimeline(
                requestedAt: requestedAt,
                removedAt: removedAt),
            previousPhase: servicePhaseDescription(previousPhase),
            slotRemoved: slotRemoved,
            cancellationEventObserved: cancellationEventObserved,
            repeatedCancellationNotFound: repeatedCancellationNotFound,
            initialActiveRequestCount: initial.count,
            replacementWasQueued: replacementWasQueued,
            replacementSlotReused: replacementSlotReused,
            sharedBatchObserved: sharedBatchObserved,
            decodeFirstInterleaveObserved: decodeFirstInterleaveObserved,
            cancelledPrefixTokens: emittedTokenCount(previousPhase),
            survivorOutputTokens: survivorOutputTokens,
            replacementOutputTokens: replacementTokens.count,
            operations: summarizeServiceOperations(
                ticks,
                speculationStartSnapshot: resourcesBeforeRun.speculation,
                speculationEndSnapshot: resourcesAtEnd.speculation),
            memory: try summarizeServiceMemory(memory),
            resourcesAtAdmission: resourcesAtAdmission,
            resourcesBeforeCancellation: resourcesBeforeCancellation,
            resourcesAfterCancellation: resourcesAfterCancellation,
            resourcesAfterReplacement: resourcesAfterReplacement,
            resourcesAtEnd: resourcesAtEnd)
    }
}

func loadContinuousSwiftServiceDriver(
    modelPath: String,
    configuration: ContinuousServiceLoadConfiguration
) async throws -> (
    driver: ContinuousSwiftServiceDriver,
    tokenizer: MLXLMCommon.Tokenizer,
    eos: Int
) {
    let modelURL = URL(fileURLWithPath: modelPath)
    let proof = try DenseContinuousBatchModelProof.verifying(modelDirectory: modelURL)

    // This process may run under a raised wired-memory ceiling. Keep MLX's reusable buffer
    // cache explicitly bounded so service evidence cannot depend on an implicit larger cache.
    Memory.cacheLimit = 8 << 30
    let context = try await loadModel(
        from: modelURL,
        using: #huggingFaceTokenizerLoader())
    let tokenizer = context.tokenizer
    let eos = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    let scheduler = try ContinuousBatchConfiguration(
        maxActiveSlots: configuration.maxActiveSlots,
        maxPrefillSlots: configuration.maxPrefillSlots,
        prefillChunkSize: configuration.prefillChunkSize,
        maxQueuedRequests: configuration.maxQueuedRequests)
    let runtime = try DenseContinuousBatchRuntime(
        model: context.model,
        verifiedBy: proof,
        maxReservedContextTokens: configuration.maxReservedContextTokens,
        maxReservedKVBytes: configuration.maxReservedKVBytes)
    let coordinator = ContinuousBatchCoordinator(
        configuration: scheduler,
        runtime: runtime,
        automaticDrive: false,
        traceLimit: configuration.traceLimit)
    return (ContinuousSwiftServiceDriver(coordinator: coordinator, eos: eos), tokenizer, eos)
}

private func serviceClockSeconds() -> Double {
    ProcessInfo.processInfo.systemUptime
}

private func observedServiceTick(
    _ coordinator: ContinuousBatchCoordinator
) async throws -> (
    remains: Bool,
    tick: ServiceTickObservation,
    memory: ServiceMemorySample
) {
    let remains = try await coordinator.runOneTick()
    let events = await coordinator.takeExecutionTrace()
    _ = await coordinator.takeTimingTrace()
    let snapshots = await coordinator.snapshots()
    let queued = snapshots.filter {
        if case .queued = $0.phase { return true }
        return false
    }.count
    return (
        remains,
        ServiceTickObservation(
            activeSlots: snapshots.count - queued,
            queuedSlots: queued,
            operations: events.compactMap {
                if case .operation(let operation) = $0 { return operation }
                return nil
            }),
        serviceMemorySample())
}

private func collectServiceTokens(
    _ stream: ContinuousBatchTokenStream
) async throws -> [Int] {
    var tokens: [Int] = []
    for try await token in stream { tokens.append(token) }
    return tokens
}

private func servicePhaseDescription(_ phase: BatchSlotPhase) -> String {
    switch phase {
    case .queued: "queued"
    case .prefilling: "prefilling"
    case .ready: "ready"
    case .decoding: "decoding"
    }
}

private func emittedTokenCount(_ phase: BatchSlotPhase) -> Int {
    if case .decoding(let emittedTokens, _) = phase { return emittedTokens }
    return 0
}

func serviceMemorySample() -> ServiceMemorySample {
    let snapshot = Memory.snapshot()
    return ServiceMemorySample(
        timestamp: serviceClockSeconds(),
        physicalFootprintBytes: physFootprintBytes(),
        mlxActiveBytes: snapshot.activeMemory,
        mlxCacheBytes: snapshot.cacheMemory,
        mlxPeakBytes: snapshot.peakMemory)
}
