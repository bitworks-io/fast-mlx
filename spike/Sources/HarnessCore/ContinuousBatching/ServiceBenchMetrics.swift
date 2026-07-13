import Foundation

/// Client-submitted/server-stream lifecycle for one request in a shared service run. The caller
/// stamps `submittedAt` before admission; token/completion times are captured where the actor
/// publishes the stream, so queueing and prompt work count but consumer-task scheduling does not.
public struct ServiceRequestTimeline: Sendable, Equatable {
    public let requestID: BatchRequestID
    public let promptTokenCount: Int
    public let submittedAt: Double
    public let tokenTimes: [Double]
    public let completedAt: Double

    public init(
        requestID: BatchRequestID,
        promptTokenCount: Int,
        submittedAt: Double,
        tokenTimes: [Double],
        completedAt: Double
    ) {
        self.requestID = requestID
        self.promptTokenCount = promptTokenCount
        self.submittedAt = submittedAt
        self.tokenTimes = tokenTimes
        self.completedAt = completedAt
    }
}

public struct ServiceLatencyDistribution: Sendable, Codable, Equatable {
    public let p50Seconds: Double
    public let p95Seconds: Double

    public init(p50Seconds: Double, p95Seconds: Double) {
        self.p50Seconds = p50Seconds
        self.p95Seconds = p95Seconds
    }
}

/// Recomputable per-request values persisted beside every service headline.
public struct ServiceRequestMetrics: Sendable, Codable, Equatable {
    public let requestID: UInt64
    public let promptTokenCount: Int
    public let outputTokenCount: Int
    public let ttftSeconds: Double
    public let tpotSeconds: Double?
    public let completionSeconds: Double
    public let completionTokensPerSecond: Double

    public init(
        requestID: UInt64,
        promptTokenCount: Int,
        outputTokenCount: Int,
        ttftSeconds: Double,
        tpotSeconds: Double?,
        completionSeconds: Double,
        completionTokensPerSecond: Double
    ) {
        self.requestID = requestID
        self.promptTokenCount = promptTokenCount
        self.outputTokenCount = outputTokenCount
        self.ttftSeconds = ttftSeconds
        self.tpotSeconds = tpotSeconds
        self.completionSeconds = completionSeconds
        self.completionTokensPerSecond = completionTokensPerSecond
    }
}

public struct ServiceRunMetrics: Sendable, Codable, Equatable {
    public let requestCount: Int
    public let totalOutputTokens: Int
    public let wallSeconds: Double
    public let aggregateTokensPerSecond: Double
    public let ttft: ServiceLatencyDistribution
    public let tpot: ServiceLatencyDistribution?
    public let completion: ServiceLatencyDistribution
    public let jainCompletionRate: Double
    public let requests: [ServiceRequestMetrics]

    public init(
        requestCount: Int,
        totalOutputTokens: Int,
        wallSeconds: Double,
        aggregateTokensPerSecond: Double,
        ttft: ServiceLatencyDistribution,
        tpot: ServiceLatencyDistribution?,
        completion: ServiceLatencyDistribution,
        jainCompletionRate: Double,
        requests: [ServiceRequestMetrics]
    ) {
        self.requestCount = requestCount
        self.totalOutputTokens = totalOutputTokens
        self.wallSeconds = wallSeconds
        self.aggregateTokensPerSecond = aggregateTokensPerSecond
        self.ttft = ttft
        self.tpot = tpot
        self.completion = completion
        self.jainCompletionRate = jainCompletionRate
        self.requests = requests
    }
}

public struct ServiceRunAggregate: Sendable, Codable, Equatable {
    public let runCount: Int
    public let requestCount: Int
    public let meanAggregateTokensPerSecond: Double
    public let ttft: ServiceLatencyDistribution
    public let tpot: ServiceLatencyDistribution?
    public let completion: ServiceLatencyDistribution
    public let meanJainCompletionRate: Double
    public let minJainCompletionRate: Double
}

public enum ServiceBenchMetricsError: Error, Sendable, Equatable {
    case emptyRun
    case duplicateRequestID(BatchRequestID)
    case invalidPromptTokenCount(BatchRequestID, Int)
    case emptyTokenStream(BatchRequestID)
    case nonFiniteTimeline(BatchRequestID)
    case tokenBeforeSubmission(BatchRequestID)
    case nonMonotonicTokenTimes(BatchRequestID)
    case completionBeforeLastToken(BatchRequestID)
    case nonPositiveCompletionSpan(BatchRequestID)
    case tokenTimestampCountMismatch(tokens: Int, timestamps: Int)
    case emptyCancellationSamples
    case invalidCancellationTimeline(Int)
    case invalidKeepaliveSeconds(Double)
    case insufficientMemorySamples
    case invalidMemorySample(Int)
    case zeroMemoryBaseline
}

public struct VisibleServiceTokenStream: Sendable, Equatable {
    public let tokens: [Int]
    public let tokenTimes: [Double]

    public init(tokens: [Int], tokenTimes: [Double]) {
        self.tokens = tokens
        self.tokenTimes = tokenTimes
    }
}

/// The scalar/PLD driver includes terminal EOS while the continuous coordinator consumes it.
/// Normalize both policies to user-visible output before any throughput or latency statistic.
public func normalizeVisibleServiceTokens(
    tokens: [Int],
    tokenTimes: [Double],
    eosToken: Int
) throws -> VisibleServiceTokenStream {
    guard tokens.count == tokenTimes.count else {
        throw ServiceBenchMetricsError.tokenTimestampCountMismatch(
            tokens: tokens.count, timestamps: tokenTimes.count)
    }
    let end = tokens.firstIndex(of: eosToken) ?? tokens.endIndex
    return VisibleServiceTokenStream(
        tokens: Array(tokens[..<end]),
        tokenTimes: Array(tokenTimes[..<end]))
}

/// Aggregate throughput uses the whole client-observed burst makespan, including queueing and
/// prefill. TPOT is one mean inter-token interval per request, then p50/p95 across requests.
public func measureServiceRun(
    _ timelines: [ServiceRequestTimeline]
) throws -> ServiceRunMetrics {
    guard !timelines.isEmpty else { throw ServiceBenchMetricsError.emptyRun }
    var seen: Set<BatchRequestID> = []
    var requests: [ServiceRequestMetrics] = []
    requests.reserveCapacity(timelines.count)

    for timeline in timelines {
        guard seen.insert(timeline.requestID).inserted else {
            throw ServiceBenchMetricsError.duplicateRequestID(timeline.requestID)
        }
        guard timeline.promptTokenCount > 0 else {
            throw ServiceBenchMetricsError.invalidPromptTokenCount(
                timeline.requestID, timeline.promptTokenCount)
        }
        guard !timeline.tokenTimes.isEmpty else {
            throw ServiceBenchMetricsError.emptyTokenStream(timeline.requestID)
        }
        guard timeline.submittedAt.isFinite, timeline.completedAt.isFinite,
            timeline.tokenTimes.allSatisfy(\.isFinite)
        else {
            throw ServiceBenchMetricsError.nonFiniteTimeline(timeline.requestID)
        }
        guard timeline.tokenTimes[0] >= timeline.submittedAt else {
            throw ServiceBenchMetricsError.tokenBeforeSubmission(timeline.requestID)
        }
        guard zip(timeline.tokenTimes, timeline.tokenTimes.dropFirst()).allSatisfy({ $0 <= $1 })
        else {
            throw ServiceBenchMetricsError.nonMonotonicTokenTimes(timeline.requestID)
        }
        guard timeline.completedAt >= timeline.tokenTimes.last! else {
            throw ServiceBenchMetricsError.completionBeforeLastToken(timeline.requestID)
        }
        let completion = timeline.completedAt - timeline.submittedAt
        guard completion > 0 else {
            throw ServiceBenchMetricsError.nonPositiveCompletionSpan(timeline.requestID)
        }
        let tpot: Double?
        if timeline.tokenTimes.count > 1 {
            tpot = (timeline.tokenTimes.last! - timeline.tokenTimes[0])
                / Double(timeline.tokenTimes.count - 1)
        } else {
            tpot = nil
        }
        requests.append(
            ServiceRequestMetrics(
                requestID: timeline.requestID.rawValue,
                promptTokenCount: timeline.promptTokenCount,
                outputTokenCount: timeline.tokenTimes.count,
                ttftSeconds: timeline.tokenTimes[0] - timeline.submittedAt,
                tpotSeconds: tpot,
                completionSeconds: completion,
                completionTokensPerSecond: Double(timeline.tokenTimes.count) / completion))
    }

    let start = timelines.map(\.submittedAt).min()!
    let end = timelines.map(\.completedAt).max()!
    let wall = end - start
    guard wall > 0 else {
        throw ServiceBenchMetricsError.nonPositiveCompletionSpan(timelines[0].requestID)
    }
    let totalTokens = requests.map(\.outputTokenCount).reduce(0, +)
    let rates = requests.map(\.completionTokensPerSecond)
    let rateSum = rates.reduce(0, +)
    let rateSquareSum = rates.reduce(0) { $0 + ($1 * $1) }
    let fairness = (rateSum * rateSum) / (Double(rates.count) * rateSquareSum)
    let tpots = requests.compactMap(\.tpotSeconds)

    return ServiceRunMetrics(
        requestCount: requests.count,
        totalOutputTokens: totalTokens,
        wallSeconds: wall,
        aggregateTokensPerSecond: Double(totalTokens) / wall,
        ttft: distribution(requests.map(\.ttftSeconds)),
        tpot: tpots.isEmpty ? nil : distribution(tpots),
        completion: distribution(requests.map(\.completionSeconds)),
        jainCompletionRate: fairness,
        requests: requests)
}

/// Aggregates post-warmup runs while retaining their raw per-request records in the caller's
/// payload. Latency distributions pool requests (not run headlines); throughput and fairness
/// give each measured run equal weight.
public func aggregateServiceRuns(
    _ runs: [ServiceRunMetrics]
) throws -> ServiceRunAggregate {
    guard !runs.isEmpty else { throw ServiceBenchMetricsError.emptyRun }
    let requests = runs.flatMap(\.requests)
    let tpots = requests.compactMap(\.tpotSeconds)
    let fairness = runs.map(\.jainCompletionRate)
    return ServiceRunAggregate(
        runCount: runs.count,
        requestCount: requests.count,
        meanAggregateTokensPerSecond: runs.map(\.aggregateTokensPerSecond).reduce(0, +)
            / Double(runs.count),
        ttft: distribution(requests.map(\.ttftSeconds)),
        tpot: tpots.isEmpty ? nil : distribution(tpots),
        completion: distribution(requests.map(\.completionSeconds)),
        meanJainCompletionRate: fairness.reduce(0, +) / Double(fairness.count),
        minJainCompletionRate: fairness.min()!)
}

private func distribution(_ values: [Double]) -> ServiceLatencyDistribution {
    ServiceLatencyDistribution(
        p50Seconds: quantile(values, 0.5),
        p95Seconds: quantile(values, 0.95))
}

public struct ServiceTickObservation: Sendable, Equatable {
    public let activeSlots: Int
    public let queuedSlots: Int
    public let operations: [BatchSchedulerOperation]

    public init(
        activeSlots: Int,
        queuedSlots: Int,
        operations: [BatchSchedulerOperation]
    ) {
        self.activeSlots = activeSlots
        self.queuedSlots = queuedSlots
        self.operations = operations
    }
}

public struct ServiceOperationSummary: Sendable, Codable, Equatable {
    public let tickCount: Int
    public let maxActiveSlots: Int
    public let meanActiveSlots: Double
    public let maxQueuedSlots: Int
    public let decodeBatchSizeHistogram: [Int: Int]
    public let promptChunkCount: Int
    public let promptTokensProcessed: Int
    public let drainCount: Int
    public let speculationEngaged: Bool

    public init(
        tickCount: Int,
        maxActiveSlots: Int,
        meanActiveSlots: Double,
        maxQueuedSlots: Int,
        decodeBatchSizeHistogram: [Int: Int],
        promptChunkCount: Int,
        promptTokensProcessed: Int,
        drainCount: Int,
        speculationEngaged: Bool
    ) {
        self.tickCount = tickCount
        self.maxActiveSlots = maxActiveSlots
        self.meanActiveSlots = meanActiveSlots
        self.maxQueuedSlots = maxQueuedSlots
        self.decodeBatchSizeHistogram = decodeBatchSizeHistogram
        self.promptChunkCount = promptChunkCount
        self.promptTokensProcessed = promptTokensProcessed
        self.drainCount = drainCount
        self.speculationEngaged = speculationEngaged
    }
}

public func summarizeServiceOperations(
    _ observations: [ServiceTickObservation]
) -> ServiceOperationSummary {
    var histogram: [Int: Int] = [:]
    var promptChunks = 0
    var promptTokens = 0
    var drains = 0
    var speculation = false
    for observation in observations {
        for operation in observation.operations {
            switch operation {
            case .prefill(let slice):
                promptChunks += 1
                promptTokens += slice.count
            case .decode(.drainSoloPipeline):
                drains += 1
            case .decode(.solo(_, let allowed)):
                histogram[1, default: 0] += 1
                speculation = speculation || allowed
            case .decode(.batch(let ids, let allowed)):
                histogram[ids.count, default: 0] += 1
                speculation = speculation || allowed
            }
        }
    }
    let activeTotal = observations.map(\.activeSlots).reduce(0, +)
    return ServiceOperationSummary(
        tickCount: observations.count,
        maxActiveSlots: observations.map(\.activeSlots).max() ?? 0,
        meanActiveSlots: observations.isEmpty
            ? 0 : Double(activeTotal) / Double(observations.count),
        maxQueuedSlots: observations.map(\.queuedSlots).max() ?? 0,
        decodeBatchSizeHistogram: histogram,
        promptChunkCount: promptChunks,
        promptTokensProcessed: promptTokens,
        drainCount: drains,
        speculationEngaged: speculation)
}

public struct ServiceCancellationTimeline: Sendable, Equatable {
    public let requestedAt: Double
    public let removedAt: Double

    public init(requestedAt: Double, removedAt: Double) {
        self.requestedAt = requestedAt
        self.removedAt = removedAt
    }
}

public struct ServiceCancellationSummary: Sendable, Codable, Equatable {
    public let p50Seconds: Double
    public let p95Seconds: Double
    public let maxSeconds: Double
}

public struct ServiceCancellationGate: Sendable, Codable, Equatable {
    public let keepaliveSeconds: Double
    public let summary: ServiceCancellationSummary
    public let withinKeepalive: Bool
}

public func summarizeCancellationLatencies(
    _ timelines: [ServiceCancellationTimeline]
) throws -> ServiceCancellationSummary {
    guard !timelines.isEmpty else {
        throw ServiceBenchMetricsError.emptyCancellationSamples
    }
    var values: [Double] = []
    for (index, timeline) in timelines.enumerated() {
        guard timeline.requestedAt.isFinite, timeline.removedAt.isFinite,
            timeline.removedAt >= timeline.requestedAt
        else {
            throw ServiceBenchMetricsError.invalidCancellationTimeline(index)
        }
        values.append(timeline.removedAt - timeline.requestedAt)
    }
    return ServiceCancellationSummary(
        p50Seconds: quantile(values, 0.5),
        p95Seconds: quantile(values, 0.95),
        maxSeconds: values.max()!)
}

public func evaluateCancellationGate(
    _ timelines: [ServiceCancellationTimeline],
    keepaliveSeconds: Double
) throws -> ServiceCancellationGate {
    guard keepaliveSeconds.isFinite, keepaliveSeconds > 0 else {
        throw ServiceBenchMetricsError.invalidKeepaliveSeconds(keepaliveSeconds)
    }
    let summary = try summarizeCancellationLatencies(timelines)
    return ServiceCancellationGate(
        keepaliveSeconds: keepaliveSeconds,
        summary: summary,
        withinKeepalive: summary.maxSeconds <= keepaliveSeconds)
}

public struct ServiceMemorySample: Sendable, Codable, Equatable {
    public let timestamp: Double
    public let physicalFootprintBytes: UInt64
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int

    public init(
        timestamp: Double,
        physicalFootprintBytes: UInt64,
        mlxActiveBytes: Int,
        mlxCacheBytes: Int,
        mlxPeakBytes: Int
    ) {
        self.timestamp = timestamp
        self.physicalFootprintBytes = physicalFootprintBytes
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
    }
}

public struct ServiceMemorySummary: Sendable, Codable, Equatable {
    public let startFootprintBytes: UInt64
    public let endFootprintBytes: UInt64
    public let maxSampledFootprintBytes: UInt64
    public let endFootprintDriftPercent: Double
    public let maxFootprintDriftPercent: Double
    public let maxMLXActiveBytes: Int
    public let maxMLXCacheBytes: Int
    public let maxMLXPeakBytes: Int
}

public func summarizeServiceMemory(
    _ samples: [ServiceMemorySample]
) throws -> ServiceMemorySummary {
    guard samples.count >= 2 else {
        throw ServiceBenchMetricsError.insufficientMemorySamples
    }
    for (index, sample) in samples.enumerated() {
        guard sample.timestamp.isFinite, sample.mlxActiveBytes >= 0,
            sample.mlxCacheBytes >= 0, sample.mlxPeakBytes >= 0
        else {
            throw ServiceBenchMetricsError.invalidMemorySample(index)
        }
        if index > 0, sample.timestamp < samples[index - 1].timestamp {
            throw ServiceBenchMetricsError.invalidMemorySample(index)
        }
    }
    let start = samples[0].physicalFootprintBytes
    guard start > 0 else { throw ServiceBenchMetricsError.zeroMemoryBaseline }
    let end = samples.last!.physicalFootprintBytes
    let peak = samples.map(\.physicalFootprintBytes).max()!
    func drift(_ value: UInt64) -> Double {
        (Double(value) - Double(start)) / Double(start) * 100
    }
    return ServiceMemorySummary(
        startFootprintBytes: start,
        endFootprintBytes: end,
        maxSampledFootprintBytes: peak,
        endFootprintDriftPercent: drift(end),
        maxFootprintDriftPercent: drift(peak),
        maxMLXActiveBytes: samples.map(\.mlxActiveBytes).max()!,
        maxMLXCacheBytes: samples.map(\.mlxCacheBytes).max()!,
        maxMLXPeakBytes: samples.map(\.mlxPeakBytes).max()!)
}
