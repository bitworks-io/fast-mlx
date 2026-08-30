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
