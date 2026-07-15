import Foundation
import MLX

public enum KVarNMemoryEvidenceError: Error, Equatable, Sendable {
    case invalidProbeConfiguration
    case invalidMemoryCounters
    case allocatorCacheNotEmpty
    case cacheStorageUnavailable
    case unsupportedCacheStorageDType
    case cacheStorageDetachmentMismatch
    case unreconciledAllocatorMemory
    case invalidRetainedMemoryAccounting
    case invalidStructuralMemoryAccounting
}

public enum KVarNMemoryProbePhase: String, Codable, CaseIterable, Sendable {
    case encode
    case decode
    case cacheBoundary = "cache-boundary"
}

/// Closed qualification geometry for the model-free KVarN memory probe. The first matrix is
/// deliberately narrower than the codec: batch one, the runtime's K4V2-g128 layout, exact fp16
/// inputs, 8/16 balancing iterations, and a disabled MLX allocator cache.
public struct KVarNMemoryProbeConfiguration: Encodable, Equatable, Sendable {
    public let phase: KVarNMemoryProbePhase
    public let heads: Int
    public let headDimension: Int
    public let groupSize: Int
    public let iterations: Int
    public let capacity: Int
    public let cacheLimitBytes: Int
    public let run: Int

    public init(
        phase: KVarNMemoryProbePhase,
        heads: Int, headDimension: Int, groupSize: Int,
        iterations: Int, capacity: Int, cacheLimitBytes: Int, run: Int
    ) throws {
        guard heads == 8,
            headDimension == 128,
            groupSize == 128,
            iterations == 8 || iterations == 16,
            [256, 4_096, 24_192].contains(capacity),
            capacity <= Int(Int32.max),
            cacheLimitBytes == 0,
            (1 ... 3).contains(run)
        else { throw KVarNMemoryEvidenceError.invalidProbeConfiguration }
        let (rowElements, rowOverflow) = groupSize.multipliedReportingOverflow(
            by: headDimension)
        let (tileElements, tileOverflow) = rowElements.multipliedReportingOverflow(by: heads)
        let (capacityRows, capacityRowsOverflow) = capacity.multipliedReportingOverflow(
            by: heads)
        let (capacityElements, capacityElementsOverflow) = capacityRows
            .multipliedReportingOverflow(by: headDimension)
        let (_, capacityBytesOverflow) = capacityElements.multipliedReportingOverflow(by: 4)
        guard !rowOverflow, !tileOverflow, tileElements > 0,
            !capacityRowsOverflow, !capacityElementsOverflow,
            !capacityBytesOverflow, capacityElements > 0
        else {
            throw KVarNMemoryEvidenceError.invalidProbeConfiguration
        }
        self.phase = phase
        self.heads = heads
        self.headDimension = headDimension
        self.groupSize = groupSize
        self.iterations = iterations
        self.capacity = capacity
        self.cacheLimitBytes = cacheLimitBytes
        self.run = run
    }

    public var rows: Int { heads }

    public var tileElementCount: Int {
        heads * groupSize * headDimension
    }

    /// Number of post-sink complete tiles that exactly fill the declared capacity.
    public var completedTileCapacity: Int {
        (capacity - groupSize) / groupSize
    }
}

public struct KVarNMemoryCounters: Codable, Equatable, Sendable {
    public let activeBytes: Int
    public let cacheBytes: Int
    public let peakActiveBytes: Int

    public init(activeBytes: Int, cacheBytes: Int, peakActiveBytes: Int) {
        self.activeBytes = activeBytes
        self.cacheBytes = cacheBytes
        self.peakActiveBytes = peakActiveBytes
    }
}

/// Proves that a set of independently materialized arrays accounts for allocator-active memory.
/// `expectedAllocatorBytes` is the exact sum of MLX's per-array page rounding; only the separately
/// measured sub-page runtime baseline may sit above it. Any other byte invalidates the setup.
public struct KVarNMemoryReconciliation: Encodable, Equatable, Sendable {
    public let logicalBytes: Int
    public let expectedAllocatorBytes: Int
    public let activeBytes: Int
    public let runtimeBaselineBytes: Int
    public let arrayCount: Int
    public let allocatorPageBytes: Int
    public let maximumActiveBytes: Int
    public let activeAboveExpectedAllocatorBytes: Int

    public init(
        logicalBytes: Int, expectedAllocatorBytes: Int,
        activeBytes: Int, runtimeBaselineBytes: Int,
        arrayCount: Int, allocatorPageBytes: Int
    ) throws {
        guard logicalBytes >= 0,
            expectedAllocatorBytes >= logicalBytes,
            activeBytes >= expectedAllocatorBytes,
            runtimeBaselineBytes >= 0,
            runtimeBaselineBytes < allocatorPageBytes,
            arrayCount > 0,
            allocatorPageBytes > 0,
            (allocatorPageBytes & (allocatorPageBytes - 1)) == 0
        else { throw KVarNMemoryEvidenceError.unreconciledAllocatorMemory }
        let (maximumRounding, roundingOverflow) = (allocatorPageBytes - 1)
            .multipliedReportingOverflow(by: arrayCount)
        let allocatorRounding = expectedAllocatorBytes - logicalBytes
        let (maximumActive, activeOverflow) = expectedAllocatorBytes
            .addingReportingOverflow(runtimeBaselineBytes)
        guard !roundingOverflow, !activeOverflow,
            allocatorRounding <= maximumRounding,
            activeBytes <= maximumActive
        else {
            throw KVarNMemoryEvidenceError.unreconciledAllocatorMemory
        }
        self.logicalBytes = logicalBytes
        self.expectedAllocatorBytes = expectedAllocatorBytes
        self.activeBytes = activeBytes
        self.runtimeBaselineBytes = runtimeBaselineBytes
        self.arrayCount = arrayCount
        self.allocatorPageBytes = allocatorPageBytes
        self.maximumActiveBytes = maximumActive
        self.activeAboveExpectedAllocatorBytes = activeBytes - expectedAllocatorBytes
    }
}

/// Separates the logical arrays known to be retained after an operation from additional active
/// residency held by MLX's evaluated graph. The additional bytes are a measured result, not
/// mislabeled as transient scratch and not silently discarded by the evidence writer.
public struct KVarNRetainedMemoryAccounting: Encodable, Equatable, Sendable {
    public let minimumLogicalBytes: Int
    public let activeBytes: Int
    public let arrayCount: Int
    public let activeAboveMinimumLogicalBytes: Int

    public init(
        minimumLogicalBytes: Int, activeBytes: Int, arrayCount: Int
    ) throws {
        guard minimumLogicalBytes >= 0, activeBytes >= minimumLogicalBytes,
            arrayCount > 0
        else { throw KVarNMemoryEvidenceError.invalidRetainedMemoryAccounting }
        self.minimumLogicalBytes = minimumLogicalBytes
        self.activeBytes = activeBytes
        self.arrayCount = arrayCount
        self.activeAboveMinimumLogicalBytes = activeBytes - minimumLogicalBytes
    }
}

/// Structural references for one full-tile cache-boundary update. One concatenation output and
/// its reconstructed inputs are a guaranteed minimum live-set. The dual-concatenation value is a
/// useful pinned-evaluator reference, but is not treated as a floor because MLX may schedule and
/// release the K and V branches independently. A separate whole-decoder measurement remains
/// required before multiplying any transient result across model layers.
public struct KVarNCacheBoundaryStructuralMemory: Encodable, Equatable, Sendable {
    public let completedTileCount: Int
    public let materializedOutputArrayBytes: Int
    public let materializedOutputArrayAllocatorBytes: Int
    public let reconstructedTileArrayBytes: Int
    public let reconstructedTileArrayAllocatorBytes: Int
    public let minimumConcatIncrementBytes: Int
    public let minimumStructuralPeakActiveBytes: Int
    public let dualConcatReferenceIncrementBytes: Int
    public let dualConcatReferencePeakActiveBytes: Int
    public let observedPeakActiveBytes: Int
    public let observedPeakAboveMinimumStructuralBytes: Int
    public let observedPeakDeltaFromDualConcatReferenceBytes: Int

    public init(
        configuration: KVarNMemoryProbeConfiguration,
        allocatorPageBytes: Int,
        startActiveBytes: Int,
        observedPeakActiveBytes: Int
    ) throws {
        guard configuration.phase == .cacheBoundary,
            startActiveBytes >= 0, observedPeakActiveBytes >= 0
        else { throw KVarNMemoryEvidenceError.invalidStructuralMemoryAccounting }
        let materialized = try Self.checkedProduct([
            2, configuration.heads, configuration.capacity,
            configuration.headDimension,
        ])
        let reconstructed = try Self.checkedProduct([
            2, configuration.heads, configuration.groupSize,
            configuration.headDimension,
        ])
        let materializedAllocator = try KVarNMemoryEvidence.allocatorBytes(
            forLogicalBytes: materialized, pageBytes: allocatorPageBytes)
        let reconstructedAllocator = try KVarNMemoryEvidence.allocatorBytes(
            forLogicalBytes: reconstructed, pageBytes: allocatorPageBytes)
        let completedTiles = configuration.completedTileCapacity
        let oneBranchPieces = try Self.checkedProduct([
            completedTiles, reconstructedAllocator,
        ])
        let minimumIncrement = try Self.checkedSum([
            materializedAllocator, oneBranchPieces,
        ])
        let minimumPeak = try Self.checkedSum([
            startActiveBytes, minimumIncrement,
        ])
        let dualIncrement = try Self.checkedProduct([2, minimumIncrement])
        let dualReferencePeak = try Self.checkedSum([
            startActiveBytes, dualIncrement,
        ])
        guard observedPeakActiveBytes >= minimumPeak else {
            throw KVarNMemoryEvidenceError.invalidStructuralMemoryAccounting
        }
        self.completedTileCount = completedTiles
        self.materializedOutputArrayBytes = materialized
        self.materializedOutputArrayAllocatorBytes = materializedAllocator
        self.reconstructedTileArrayBytes = reconstructed
        self.reconstructedTileArrayAllocatorBytes = reconstructedAllocator
        self.minimumConcatIncrementBytes = minimumIncrement
        self.minimumStructuralPeakActiveBytes = minimumPeak
        self.dualConcatReferenceIncrementBytes = dualIncrement
        self.dualConcatReferencePeakActiveBytes = dualReferencePeak
        self.observedPeakActiveBytes = observedPeakActiveBytes
        self.observedPeakAboveMinimumStructuralBytes = observedPeakActiveBytes - minimumPeak
        self.observedPeakDeltaFromDualConcatReferenceBytes =
            observedPeakActiveBytes - dualReferencePeak
    }

    private static func checkedProduct(_ values: [Int]) throws -> Int {
        try values.reduce(into: 1) { result, value in
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw KVarNMemoryEvidenceError.invalidStructuralMemoryAccounting
            }
            result = next
        }
    }

    private static func checkedSum(_ values: [Int]) throws -> Int {
        try values.reduce(into: 0) { result, value in
            let (next, overflow) = result.addingReportingOverflow(value)
            guard !overflow else {
                throw KVarNMemoryEvidenceError.invalidStructuralMemoryAccounting
            }
            result = next
        }
    }
}

/// Derived active-memory high-water with stable retained residency separated from temporary
/// allocations. The MLX peak counter is reset immediately before `start`, so subtracting its old
/// value would be invalid; the derivation instead compares the absolute counters explicitly.
public struct KVarNMemoryHighWater: Encodable, Equatable, Sendable {
    public let start: KVarNMemoryCounters
    public let end: KVarNMemoryCounters
    public let observedPeakActiveBytes: Int
    public let retainedActiveBytes: Int
    public let transientActiveAboveRetainedBytes: Int
    public let incrementalPeakActiveBytes: Int

    public init(
        start: KVarNMemoryCounters, end: KVarNMemoryCounters
    ) throws {
        let values = [
            start.activeBytes, start.cacheBytes, start.peakActiveBytes,
            end.activeBytes, end.cacheBytes, end.peakActiveBytes,
        ]
        guard values.allSatisfy({ $0 >= 0 }) else {
            throw KVarNMemoryEvidenceError.invalidMemoryCounters
        }
        guard start.cacheBytes == 0, end.cacheBytes == 0 else {
            throw KVarNMemoryEvidenceError.allocatorCacheNotEmpty
        }
        guard start.peakActiveBytes == 0 else {
            throw KVarNMemoryEvidenceError.invalidMemoryCounters
        }
        let observed = max(start.activeBytes, end.activeBytes, end.peakActiveBytes)
        let retained = max(start.activeBytes, end.activeBytes)
        self.start = start
        self.end = end
        self.observedPeakActiveBytes = observed
        self.retainedActiveBytes = retained
        self.transientActiveAboveRetainedBytes = observed - retained
        self.incrementalPeakActiveBytes = observed - start.activeBytes
    }
}

public enum KVarNMemoryEvidence {
    public static func isCleanHarnessSHA(_ value: String) -> Bool {
        value.count == 40 && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    /// Mirrors MLX's Metal allocator: requests at or below one VM page retain their logical byte
    /// size; larger requests round up to the next page boundary.
    package static func allocatorBytes(
        forLogicalBytes logicalBytes: Int, pageBytes: Int
    ) throws -> Int {
        guard logicalBytes >= 0, pageBytes > 0,
            (pageBytes & (pageBytes - 1)) == 0
        else { throw KVarNMemoryEvidenceError.unreconciledAllocatorMemory }
        guard logicalBytes > pageBytes else { return logicalBytes }
        let (adjusted, additionOverflow) = logicalBytes.addingReportingOverflow(
            pageBytes - 1)
        guard !additionOverflow else {
            throw KVarNMemoryEvidenceError.unreconciledAllocatorMemory
        }
        let pages = adjusted / pageBytes
        let (rounded, multiplicationOverflow) = pages.multipliedReportingOverflow(
            by: pageBytes)
        guard !multiplicationOverflow else {
            throw KVarNMemoryEvidenceError.unreconciledAllocatorMemory
        }
        return rounded
    }

    /// Replaces every allocated KVarN cache array with a byte-identical host-round-tripped copy.
    /// This is intentionally evidence-only: it synchronizes the GPU and must never be inserted in
    /// the inference loop. It lets a boundary probe start from storage without retaining the graph
    /// that constructed that storage.
    package static func detachCacheStorage(
        _ cache: KVarNKVCache
    ) throws {
        guard let snapshot = cache.storageSnapshot() else {
            throw KVarNMemoryEvidenceError.cacheStorageUnavailable
        }

        func detached(_ array: MLXArray?) throws -> MLXArray {
            guard let array else {
                throw KVarNMemoryEvidenceError.cacheStorageUnavailable
            }
            switch array.dtype {
            case .uint8:
                return MLXArray(array.asArray(UInt8.self)).reshaped(array.shape)
            case .float16:
                return MLXArray(array.asArray(Float16.self)).reshaped(array.shape)
            case .int32:
                return MLXArray(array.asArray(Int32.self)).reshaped(array.shape)
            default:
                throw KVarNMemoryEvidenceError.unsupportedCacheStorageDType
            }
        }

        cache.kPayload = try detached(cache.kPayload)
        cache.kAbsorbedScales = try detached(cache.kAbsorbedScales)
        cache.kAbsorbedBiases = try detached(cache.kAbsorbedBiases)
        cache.kTokenScales = try detached(cache.kTokenScales)
        cache.vPayload = try detached(cache.vPayload)
        cache.vChannelScales = try detached(cache.vChannelScales)
        cache.vAbsorbedScales = try detached(cache.vAbsorbedScales)
        cache.vAbsorbedBiases = try detached(cache.vAbsorbedBiases)
        cache.sinkKeys = try detached(cache.sinkKeys)
        cache.sinkValues = try detached(cache.sinkValues)
        cache.tailKeys = try detached(cache.tailKeys)
        cache.tailValues = try detached(cache.tailValues)
        cache.offsetArr = try detached(cache.offsetArr)
        eval(cache.innerState())

        guard cache.storageSnapshot() == snapshot,
            cache.offsetArr.item(Int32.self) == Int32(cache.offset)
        else {
            throw KVarNMemoryEvidenceError.cacheStorageDetachmentMismatch
        }
    }
}
