import Darwin
import Foundation
import HarnessCore
import Metal

package enum Qwen38MTPScorecardLiveAdapterError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case invalidInteger(String)
    case invalidMemoryBudget
    case unsafeOutput
    case outputExists
    case outputWriteFailed
    case invalidWorkerRole(Qwen38MTPPerformanceScorecardEngineRole)
    case invalidHandshake(Qwen38MTPPerformanceScorecardEngineRole)
    case exactnessLaunchBindingMismatch
    case workerEnvDrift(Qwen38MTPPerformanceScorecardEngineRole)
    case workerRestarted(Qwen38MTPPerformanceScorecardEngineRole)
    case workerExited(Qwen38MTPPerformanceScorecardEngineRole)
    case malformedWorkerResponse
    case duplicateWorkerResponse(Int)
    case outOfOrderWorkerResponse(expected: Int, actual: Int)
    case workerError

    package var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .invalidInteger(let flag): return "\(flag) must be a positive integer"
        case .invalidMemoryBudget: return "invalid explicit memory budget"
        case .unsafeOutput: return "output destination is unsafe"
        case .outputExists: return "output destination already exists"
        case .outputWriteFailed: return "failed to publish scorecard outputs"
        case .invalidWorkerRole: return "worker role mismatch"
        case .invalidHandshake: return "worker handshake failed validation"
        case .exactnessLaunchBindingMismatch:
            return "candidate exactness binding does not match candidate measurement binding"
        case .workerEnvDrift: return "worker environment drifted"
        case .workerRestarted: return "worker process identity changed"
        case .workerExited: return "worker exited"
        case .malformedWorkerResponse: return "worker returned malformed response"
        case .duplicateWorkerResponse: return "worker returned duplicate response"
        case .outOfOrderWorkerResponse: return "worker returned out-of-order response"
        case .workerError: return "worker command failed"
        }
    }
}

package struct Qwen38MTPScorecardLiveHostMemorySnapshot: Equatable, Sendable {
    package static let requiredScorecardChip = "Apple M3 Ultra"

    package let physicalRAMBytes: UInt64
    package let chipName: String
    package let wiredLimitMB: UInt64
    package let metalRecommendedMaxWorkingSetSizeBytes: UInt64?
    package let metalCurrentAllocatedSizeBytes: UInt64

    package init(
        physicalRAMBytes: UInt64,
        chipName: String,
        wiredLimitMB: UInt64,
        metalRecommendedMaxWorkingSetSizeBytes: UInt64?,
        metalCurrentAllocatedSizeBytes: UInt64
    ) {
        self.physicalRAMBytes = physicalRAMBytes
        self.chipName = chipName
        self.wiredLimitMB = wiredLimitMB
        self.metalRecommendedMaxWorkingSetSizeBytes = metalRecommendedMaxWorkingSetSizeBytes
        self.metalCurrentAllocatedSizeBytes = metalCurrentAllocatedSizeBytes
    }

    package static func live() throws -> Qwen38MTPScorecardLiveHostMemorySnapshot {
        guard let device = MTLCreateSystemDefaultDevice(),
            device.recommendedMaxWorkingSetSize > 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: ProcessInfo.processInfo.physicalMemory,
            chipName: device.name,
            wiredLimitMB: try qwen38MTPScorecardMeasuredWiredLimitMB(),
            metalRecommendedMaxWorkingSetSizeBytes: device.recommendedMaxWorkingSetSize,
            metalCurrentAllocatedSizeBytes: UInt64(max(device.currentAllocatedSize, 0)))
    }

    package var effectiveWiredLimitBytes: UInt64 {
        get throws {
            guard physicalRAMBytes > 0 else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            guard wiredLimitMB > 0 else { return physicalRAMBytes }
            let multiplied = wiredLimitMB.multipliedReportingOverflow(by: 1_048_576)
            guard !multiplied.overflow, multiplied.partialValue > 0 else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            return min(physicalRAMBytes, multiplied.partialValue)
        }
    }

    package var effectiveMetalCeilingBytes: UInt64 {
        get throws {
            guard let recommended = metalRecommendedMaxWorkingSetSizeBytes,
                recommended > 0
            else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            return min(try effectiveWiredLimitBytes, recommended)
        }
    }
}

package struct Qwen38MTPScorecardLiveMemoryBudget: Equatable, Codable, Sendable {
    package let memoryLimitBytes: UInt64
    package let cacheLimitBytes: UInt64
    package let reservedKVBytes: UInt64
    package let reservedIOBytes: UInt64
    package let reservedPrefetchBytes: UInt64
    package let osServiceReserveBytes: UInt64

    package init(
        memoryLimitBytes: UInt64,
        cacheLimitBytes: UInt64,
        reservedKVBytes: UInt64,
        reservedIOBytes: UInt64,
        reservedPrefetchBytes: UInt64,
        osServiceReserveBytes: UInt64
    ) throws {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.reservedKVBytes = reservedKVBytes
        self.reservedIOBytes = reservedIOBytes
        self.reservedPrefetchBytes = reservedPrefetchBytes
        self.osServiceReserveBytes = osServiceReserveBytes
        try validate()
    }

    package var liveExactnessBudget: Qwen38MTPLiveExactnessMLXMemoryBudget {
        Qwen38MTPLiveExactnessMLXMemoryBudget(
            memoryLimitBytes: Int(memoryLimitBytes),
            cacheLimitBytes: Int(cacheLimitBytes))
    }

    package func validate() throws {
        guard memoryLimitBytes > 0,
            cacheLimitBytes > 0,
            cacheLimitBytes <= memoryLimitBytes,
            reservedKVBytes > 0,
            reservedIOBytes > 0,
            reservedPrefetchBytes > 0,
            osServiceReserveBytes > 0,
            cacheLimitBytes + reservedKVBytes <= memoryLimitBytes
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }

    package var workerResidentPlanBytes: UInt64 {
        get throws {
            try qwen38MTPScorecardCheckedSum([
                memoryLimitBytes,
                reservedIOBytes,
                reservedPrefetchBytes,
            ])
        }
    }

    package func validateAgainstHost(_ snapshot: Qwen38MTPScorecardLiveHostMemorySnapshot) throws {
        let ceiling = try snapshot.effectiveMetalCeilingBytes
        guard ceiling >= memoryLimitBytes else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        let required = try qwen38MTPScorecardCheckedSum([
            workerResidentPlanBytes,
            osServiceReserveBytes,
            snapshot.metalCurrentAllocatedSizeBytes,
        ])
        guard required <= ceiling else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }

    package func validateTwoWorkerAdmission(
        _ snapshot: Qwen38MTPScorecardLiveHostMemorySnapshot,
        expectedChip: String
    ) throws {
        guard !expectedChip.isEmpty,
            expectedChip == Qwen38MTPScorecardLiveHostMemorySnapshot.requiredScorecardChip,
            snapshot.chipName == expectedChip,
            snapshot.physicalRAMBytes == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        try validateAgainstHost(snapshot)
        let ceiling = try snapshot.effectiveMetalCeilingBytes
        let workers = try workerResidentPlanBytes.multipliedReportingOverflow(by: 2)
        guard !workers.overflow else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        let required = try qwen38MTPScorecardCheckedSum([
            workers.partialValue,
            osServiceReserveBytes,
            snapshot.metalCurrentAllocatedSizeBytes,
        ])
        guard required <= ceiling else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }
}

package struct Qwen38MTPScorecardLiveRunArguments: Equatable, Sendable {
    package let targetPath: String
    package let drafterPath: String
    package let outputPath: String
    package let authorityOutputPath: String
    package let hostUse: String
    package let hostUseSource: String
    package let expectedChip: String
    package let memoryBudget: Qwen38MTPScorecardLiveMemoryBudget

    package init(
        targetPath: String,
        drafterPath: String,
        outputPath: String,
        authorityOutputPath: String,
        hostUse: String,
        hostUseSource: String,
        expectedChip: String,
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
    ) {
        self.targetPath = targetPath
        self.drafterPath = drafterPath
        self.outputPath = outputPath
        self.authorityOutputPath = authorityOutputPath
        self.hostUse = hostUse
        self.hostUseSource = hostUseSource
        self.expectedChip = expectedChip
        self.memoryBudget = memoryBudget
    }
}

/// Shared argv builder for the self-exec worker process: both the live adapter
/// (spawning through the current executable) and the proof runner must launch
/// the `qwen38-mtp-scorecard-worker` subcommand with byte-identical arguments,
/// so this is the single source of truth for that argument list.
package func qwen38MTPScorecardWorkerLaunchArguments(
    role: Qwen38MTPPerformanceScorecardEngineRole,
    arguments: Qwen38MTPScorecardLiveRunArguments
) -> [String] {
    [
        "qwen38-mtp-scorecard-worker",
        "--role", role.rawValue,
        "--target", arguments.targetPath,
        "--drafter", arguments.drafterPath,
        "--host-use", arguments.hostUse,
        "--host-use-source", arguments.hostUseSource,
        "--memory-limit-bytes", "\(arguments.memoryBudget.memoryLimitBytes)",
        "--cache-limit-bytes", "\(arguments.memoryBudget.cacheLimitBytes)",
        "--reserved-kv-bytes", "\(arguments.memoryBudget.reservedKVBytes)",
        "--reserved-io-bytes", "\(arguments.memoryBudget.reservedIOBytes)",
        "--reserved-prefetch-bytes", "\(arguments.memoryBudget.reservedPrefetchBytes)",
        "--os-service-reserve-bytes", "\(arguments.memoryBudget.osServiceReserveBytes)",
    ]
}

private func qwen38MTPScorecardMeasuredWiredLimitMB() throws -> UInt64 {
    var size = 0
    guard sysctlbyname("iogpu.wired_limit_mb", nil, &size, nil, 0) == 0,
        size > 0
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
    }
    if size == MemoryLayout<Int32>.size {
        var value: Int32 = 0
        guard sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value >= 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return UInt64(value)
    }
    if size == MemoryLayout<Int64>.size {
        var value: Int64 = 0
        guard sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value >= 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return UInt64(value)
    }
    throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
}

private func qwen38MTPScorecardCheckedSum(_ values: [UInt64]) throws -> UInt64 {
    var total: UInt64 = 0
    for value in values {
        let added = total.addingReportingOverflow(value)
        guard !added.overflow else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        total = added.partialValue
    }
    return total
}
