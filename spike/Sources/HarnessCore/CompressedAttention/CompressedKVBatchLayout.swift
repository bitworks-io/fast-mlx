import Foundation

/// Recoverable validation failures for end-aligned compressed-batch transitions.
///
/// The value contract is intentionally MLX-free so a scheduler can prove every membership and
/// capacity transition before mutating actor-confined packed arrays.
public enum CompressedKVBatchLayoutError: Error, Equatable, Sendable {
    case emptyBatch
    case capacityCount(expected: Int, actual: Int)
    case invalidLogicalOffset(row: Int, offset: Int, capacity: Int)
    case unsupportedIndexWidth(Int)
    case emptySelection
    case invalidSelection([Int])
    case invalidTokenCount(Int)
    case invalidCapacity(current: Int, requested: Int, physicalWrittenEnd: Int)
    case arithmeticOverflow
    case insufficientCapacity(required: Int, actual: Int)
}

/// Immutable end-aligned layout for a ragged compressed KV batch.
///
/// `logicalOffsets` drive RoPE positions. `physicalWrittenEnd` is the shared append boundary in
/// packed storage and deliberately survives removal of the row that originally established it.
/// Conflating the two values can overwrite a survivor after longest-row compaction.
public struct CompressedKVBatchLayout: Equatable, Sendable {
    public let allocationCapacity: Int
    public let physicalWrittenEnd: Int
    public let logicalOffsets: [Int]

    public var leftPadding: [Int] {
        logicalOffsets.map { physicalWrittenEnd - $0 }
    }

    public var physicalValidRanges: [Range<Int>] {
        leftPadding.map { $0 ..< physicalWrittenEnd }
    }

    public init(logicalOffsets: [Int], capacities: [Int]) throws {
        guard !logicalOffsets.isEmpty else {
            throw CompressedKVBatchLayoutError.emptyBatch
        }
        guard capacities.count == logicalOffsets.count else {
            throw CompressedKVBatchLayoutError.capacityCount(
                expected: logicalOffsets.count,
                actual: capacities.count)
        }

        for row in logicalOffsets.indices {
            let logical = logicalOffsets[row]
            let capacity = capacities[row]
            guard logical >= 0, capacity >= 0, logical <= capacity else {
                throw CompressedKVBatchLayoutError.invalidLogicalOffset(
                    row: row, offset: logical, capacity: capacity)
            }
            guard capacity <= Int(Int32.max) else {
                throw CompressedKVBatchLayoutError.unsupportedIndexWidth(capacity)
            }
        }

        let allocationCapacity = capacities.max() ?? 0
        let physicalWrittenEnd = logicalOffsets.max() ?? 0
        self.init(
            allocationCapacity: allocationCapacity,
            physicalWrittenEnd: physicalWrittenEnd,
            logicalOffsets: logicalOffsets)
    }

    private init(
        allocationCapacity: Int,
        physicalWrittenEnd: Int,
        logicalOffsets: [Int]
    ) {
        self.allocationCapacity = allocationCapacity
        self.physicalWrittenEnd = physicalWrittenEnd
        self.logicalOffsets = logicalOffsets
    }

    /// Select rows without re-deriving or shrinking the physical append boundary.
    public func filtering(keeping indices: [Int]) throws -> Self {
        guard !indices.isEmpty else {
            throw CompressedKVBatchLayoutError.emptySelection
        }
        guard Set(indices).count == indices.count,
            indices.allSatisfy({ $0 >= 0 && $0 < logicalOffsets.count })
        else {
            throw CompressedKVBatchLayoutError.invalidSelection(indices)
        }
        return Self(
            allocationCapacity: allocationCapacity,
            physicalWrittenEnd: physicalWrittenEnd,
            logicalOffsets: indices.map { logicalOffsets[$0] })
    }

    /// Increase the reserved fixed width without changing logical or physical positions.
    public func growing(to requestedCapacity: Int) throws -> Self {
        guard requestedCapacity >= allocationCapacity,
            requestedCapacity >= physicalWrittenEnd
        else {
            throw CompressedKVBatchLayoutError.invalidCapacity(
                current: allocationCapacity,
                requested: requestedCapacity,
                physicalWrittenEnd: physicalWrittenEnd)
        }
        guard requestedCapacity <= Int(Int32.max) else {
            throw CompressedKVBatchLayoutError.unsupportedIndexWidth(requestedCapacity)
        }
        return Self(
            allocationCapacity: requestedCapacity,
            physicalWrittenEnd: physicalWrittenEnd,
            logicalOffsets: logicalOffsets)
    }

    /// Plan a shared append at the physical boundary. The result reports exact required capacity;
    /// the actor owner must grow packed arrays before applying the transition.
    public func planAppend(tokenCount: Int) throws -> CompressedKVBatchAppendPlan {
        guard tokenCount > 0 else {
            throw CompressedKVBatchLayoutError.invalidTokenCount(tokenCount)
        }
        let (newPhysicalEnd, physicalOverflow) = physicalWrittenEnd.addingReportingOverflow(
            tokenCount)
        guard !physicalOverflow else {
            throw CompressedKVBatchLayoutError.arithmeticOverflow
        }
        guard newPhysicalEnd <= Int(Int32.max) else {
            throw CompressedKVBatchLayoutError.unsupportedIndexWidth(newPhysicalEnd)
        }

        var updatedLogicalOffsets: [Int] = []
        updatedLogicalOffsets.reserveCapacity(logicalOffsets.count)
        for logical in logicalOffsets {
            let (updated, overflow) = logical.addingReportingOverflow(tokenCount)
            guard !overflow else {
                throw CompressedKVBatchLayoutError.arithmeticOverflow
            }
            guard updated <= Int(Int32.max) else {
                throw CompressedKVBatchLayoutError.unsupportedIndexWidth(updated)
            }
            updatedLogicalOffsets.append(updated)
        }

        let requiredCapacity = max(allocationCapacity, newPhysicalEnd)
        return CompressedKVBatchAppendPlan(
            physicalRange: physicalWrittenEnd ..< newPhysicalEnd,
            requiredCapacity: requiredCapacity,
            result: Self(
                allocationCapacity: requiredCapacity,
                physicalWrittenEnd: newPhysicalEnd,
                logicalOffsets: updatedLogicalOffsets))
    }

    /// Produce fixed-width causal ranges for an incoming block before it is written.
    public func maskPlan(incomingTokens: Int) throws -> CompressedKVBatchMaskPlan {
        let append = try planAppend(tokenCount: incomingTokens)
        guard append.requiredCapacity <= allocationCapacity else {
            throw CompressedKVBatchLayoutError.insufficientCapacity(
                required: append.requiredCapacity,
                actual: allocationCapacity)
        }
        let allowed = leftPadding.map { start in
            (0 ..< incomingTokens).map { query in
                start ..< (physicalWrittenEnd + query + 1)
            }
        }
        return CompressedKVBatchMaskPlan(
            shape: [logicalOffsets.count, 1, incomingTokens, allocationCapacity],
            physicalWrittenExtent: append.physicalRange.upperBound,
            allowedPhysicalRanges: allowed)
    }
}

public struct CompressedKVBatchAppendPlan: Equatable, Sendable {
    public let physicalRange: Range<Int>
    public let requiredCapacity: Int
    public let result: CompressedKVBatchLayout
}

public struct CompressedKVBatchMaskPlan: Equatable, Sendable {
    public let shape: [Int]
    public let physicalWrittenExtent: Int
    public let allowedPhysicalRanges: [[Range<Int>]]
}
