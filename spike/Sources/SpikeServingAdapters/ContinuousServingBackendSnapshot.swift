import Foundation

import HarnessCore
import MLX

public struct ContinuousServingBackendSnapshot:
    Codable, Equatable, Sendable
{
    public let activeRequests: Int
    public let coordinatorSlots: Int
    public let reservedKVBytes: Int
    public let maxReservedKVBytes: Int
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int

    public init(
        activeRequests: Int,
        coordinatorSlots: Int,
        reservedKVBytes: Int,
        maxReservedKVBytes: Int,
        mlxActiveBytes: Int = 0,
        mlxCacheBytes: Int = 0,
        mlxPeakBytes: Int = 0
    ) {
        self.activeRequests = activeRequests
        self.coordinatorSlots = coordinatorSlots
        self.reservedKVBytes = reservedKVBytes
        self.maxReservedKVBytes = maxReservedKVBytes
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
    }
}

public struct ContinuousServingBackendScorecardObservation:
    Equatable, Sendable
{
    public let activeRequests: Int
    public let coordinatorRequestIDs: [BatchRequestID]
    public let coordinatorSlots: [BatchSlotSnapshot]
    public let runtimeResources: ContinuousBatchRuntimeResourceSnapshot?
    public let executionTrace: [ContinuousBatchCoordinatorEvent]
    public let planObservations: [ContinuousBatchPlanObservation]
    public let timingTrace: [ContinuousBatchTimingEvent]

    public init(
        activeRequests: Int,
        coordinatorRequestIDs: [BatchRequestID],
        coordinatorSlots: [BatchSlotSnapshot],
        runtimeResources: ContinuousBatchRuntimeResourceSnapshot?,
        executionTrace: [ContinuousBatchCoordinatorEvent],
        planObservations: [ContinuousBatchPlanObservation],
        timingTrace: [ContinuousBatchTimingEvent]
    ) {
        self.activeRequests = activeRequests
        self.coordinatorRequestIDs = coordinatorRequestIDs
        self.coordinatorSlots = coordinatorSlots
        self.runtimeResources = runtimeResources
        self.executionTrace = executionTrace
        self.planObservations = planObservations
        self.timingTrace = timingTrace
    }
}

@_spi(ProductionRouteEvidence)
public enum ContinuousServingProductionRouteEvidenceError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case missingLoadedModelProvenance
    case authorizationBackendMismatch
    case invalidTraceLimit(Int)
    case missingOutputTokenTrace(responseID: String)
    case duplicateOutputTokenTrace(responseID: String)
    case unexpectedOutputTokenTrace(responseID: String)
    case outputTokenTraceTruncated(responseID: String)
    case completedTraceCapacityExceeded(limit: Int)
    case insufficientScorecardTraceCapacity(required: Int, observed: Int)

    public var description: String {
        switch self {
        case .missingLoadedModelProvenance:
            return "loaded model was not produced by loadContinuousServingModel"
        case .authorizationBackendMismatch:
            return "production-route evidence authorization does not match backend"
        case .invalidTraceLimit(let limit):
            return "invalid output token trace limit \(limit)"
        case .missingOutputTokenTrace(let responseID):
            return "missing output token trace for response \(responseID)"
        case .duplicateOutputTokenTrace(let responseID):
            return "duplicate output token trace for response \(responseID)"
        case .unexpectedOutputTokenTrace(let responseID):
            return "unexpected output token trace for response \(responseID)"
        case .outputTokenTraceTruncated(let responseID):
            return "output token trace truncated for response \(responseID)"
        case .completedTraceCapacityExceeded(let limit):
            return "completed output token trace capacity exceeded \(limit)"
        case .insufficientScorecardTraceCapacity(let required, let observed):
            return "scorecard trace capacity \(observed) is below required \(required)"
        }
    }
}

@_spi(ProductionRouteEvidence)
public struct ContinuousServingOutputTokenTraceConfiguration:
    Equatable, Sendable
{
    public static let disabled = Self(
        enabled: false,
        maxCompletedRequests: 0,
        maxTokensPerRequest: 0)

    public let enabled: Bool
    public let maxCompletedRequests: Int
    public let maxTokensPerRequest: Int

    public init(
        enabled: Bool,
        maxCompletedRequests: Int,
        maxTokensPerRequest: Int
    ) {
        self.enabled = enabled
        self.maxCompletedRequests = maxCompletedRequests
        self.maxTokensPerRequest = maxTokensPerRequest
    }

    public static func outputTokenIDs(
        maxCompletedRequests: Int,
        maxTokensPerRequest: Int
    ) -> Self {
        Self(
            enabled: true,
            maxCompletedRequests: maxCompletedRequests,
            maxTokensPerRequest: maxTokensPerRequest)
    }

    var traceLimit: Int? {
        get throws {
            guard enabled else { return nil }
            guard maxCompletedRequests > 0 else {
                throw ContinuousServingProductionRouteEvidenceError
                    .invalidTraceLimit(maxCompletedRequests)
            }
            guard maxTokensPerRequest > 0 else {
                throw ContinuousServingProductionRouteEvidenceError
                    .invalidTraceLimit(maxTokensPerRequest)
            }
            return maxTokensPerRequest
        }
    }
}

@_spi(ProductionRouteEvidence)
public struct ContinuousServingProductionRouteEvidenceAuthorization:
    Sendable
{
    let backend: ContinuousServingBackend

    init(backend: ContinuousServingBackend) {
        self.backend = backend
    }

    func authorizes(_ backend: ContinuousServingBackend) -> Bool {
        self.backend === backend
    }
}

@_spi(ProductionRouteEvidence)
public struct ContinuousServingCompletedRequestTokenTrace:
    Equatable, Sendable
{
    public let responseID: String
    public let coordinatorRequestID: BatchRequestID
    public let outputTokenIDs: [Int]
    public let completionTokenCount: Int
    public let truncated: Bool

    public init(
        responseID: String,
        coordinatorRequestID: BatchRequestID,
        outputTokenIDs: [Int],
        completionTokenCount: Int,
        truncated: Bool
    ) {
        self.responseID = responseID
        self.coordinatorRequestID = coordinatorRequestID
        self.outputTokenIDs = outputTokenIDs
        self.completionTokenCount = completionTokenCount
        self.truncated = truncated
    }
}

extension ContinuousServingBackendSnapshot {
    static func current(
        activeRequests: Int,
        coordinatorSlots: Int,
        reservedKVBytes: Int,
        maxReservedKVBytes: Int
    ) -> Self {
        let memory = Memory.snapshot()
        return Self(
            activeRequests: activeRequests,
            coordinatorSlots: coordinatorSlots,
            reservedKVBytes: reservedKVBytes,
            maxReservedKVBytes: maxReservedKVBytes,
            mlxActiveBytes: memory.activeMemory,
            mlxCacheBytes: memory.cacheMemory,
            mlxPeakBytes: memory.peakMemory)
    }
}
