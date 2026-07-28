import Foundation

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
