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
    let traceLimit: Int

    init(
        maxActiveSlots: Int,
        maxPrefillSlots: Int,
        prefillChunkSize: Int,
        maxQueuedRequests: Int = 256,
        maxReservedContextTokens: Int? = nil,
        traceLimit: Int
    ) {
        self.maxActiveSlots = maxActiveSlots
        self.maxPrefillSlots = maxPrefillSlots
        self.prefillChunkSize = prefillChunkSize
        self.maxQueuedRequests = maxQueuedRequests
        self.maxReservedContextTokens = maxReservedContextTokens
        self.traceLimit = traceLimit
    }
}

struct ContinuousServiceRunObservation: Sendable {
    let metrics: ServiceRunMetrics
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
    let memorySamples: [ServiceMemorySample]
    let outputTokens: [[Int]]
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
        return ContinuousServiceRunObservation(
            metrics: try measureServiceRun(timelines),
            operations: summarizeServiceOperations(ticks),
            memory: try summarizeServiceMemory(memory),
            memorySamples: memory,
            outputTokens: collected.map(\.tokens))
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
        maxReservedContextTokens: configuration.maxReservedContextTokens)
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

func serviceMemorySample() -> ServiceMemorySample {
    let snapshot = Memory.snapshot()
    return ServiceMemorySample(
        timestamp: serviceClockSeconds(),
        physicalFootprintBytes: physFootprintBytes(),
        mlxActiveBytes: snapshot.activeMemory,
        mlxCacheBytes: snapshot.cacheMemory,
        mlxPeakBytes: snapshot.peakMemory)
}
