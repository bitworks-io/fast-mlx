import Darwin
import Foundation
import HarnessCore
import IOKit.ps
import MLX

enum BenchQualificationRuntimeError:
    Error, Equatable, CustomStringConvertible
{
    case wiredLimitUnavailable
    case wiredLimitOverflow
    case wiredLimitMismatch(expected: Int, actual: Int)
    case memorySettingsNotApplied
    case processMemoryUnavailable
    case modelIdentityChanged

    var description: String {
        switch self {
        case .wiredLimitUnavailable:
            "qualification bench could not read iogpu.wired_limit_mb"
        case .wiredLimitOverflow:
            "qualification bench wired-memory setting overflowed bytes"
        case .wiredLimitMismatch(let expected, let actual):
            "qualification bench wired-memory setting changed: expected=\(expected), actual=\(actual)"
        case .memorySettingsNotApplied:
            "qualification bench could not apply the declared MLX memory/cache limits"
        case .processMemoryUnavailable:
            "qualification bench could not capture process memory receipts"
        case .modelIdentityChanged:
            "qualification bench model config or checkpoint manifest changed during the isolated run"
        }
    }
}

struct BenchQualificationModelIdentity: Equatable {
    let configHash: String
    let checkpointManifestHash: String
    let tokenizerSHA256: String
}

func benchQualificationModelIdentity(
    modelPath: String
) throws -> BenchQualificationModelIdentity {
    let configHash = ProvenanceCLI.modelConfig(at: modelPath).hash
    guard configHash != "unknown" else {
        throw BenchQualificationRuntimeError.modelIdentityChanged
    }
    return try BenchQualificationModelIdentity(
        configHash: configHash,
        checkpointManifestHash:
            ProvenanceCLI.checkpointManifestHash(at: modelPath),
        tokenizerSHA256:
            ProvenanceCLI.tokenizerManifestSHA256(at: modelPath))
}

func validateBenchQualificationModelIdentity(
    _ identity: BenchQualificationModelIdentity,
    context: BenchQualificationContext
) throws {
    guard identity.tokenizerSHA256 == context.tokenizerSHA256 else {
        throw BenchQualificationRuntimeError.modelIdentityChanged
    }
}

/// Apply and read back the fixed MLX limits before tokenizer/model allocation. The wired limit is
/// independently read from the kernel rather than trusting the runner's command-line value.
func configureBenchQualificationMemory(
    _ context: BenchQualificationContext
) throws {
    let actualWiredLimit = try benchQualificationWiredLimitBytes()
    guard actualWiredLimit == context.wiredLimitBytes else {
        throw BenchQualificationRuntimeError.wiredLimitMismatch(
            expected: context.wiredLimitBytes,
            actual: actualWiredLimit)
    }
    Memory.memoryLimit = context.memoryLimitBytes
    Memory.cacheLimit = context.cacheLimitBytes
    guard Memory.memoryLimit == context.memoryLimitBytes,
        Memory.cacheLimit == context.cacheLimitBytes
    else {
        throw BenchQualificationRuntimeError.memorySettingsNotApplied
    }
}

func benchQualificationHostSnapshot()
    throws -> BenchQualificationHostSnapshot
{
    let process = ProcessInfo.processInfo
    let memory = try benchQualificationProcessMemory()
    return BenchQualificationHostSnapshot(
        monotonicTimestampSeconds: process.systemUptime,
        residentSizeBytes: memory.residentSizeBytes,
        physicalFootprintBytes: memory.physicalFootprintBytes,
        lowPowerModeEnabled: process.isLowPowerModeEnabled,
        powerSource: benchQualificationPowerSource(),
        thermalState: benchQualificationThermalState(
            process.thermalState))
}

private func benchQualificationWiredLimitBytes() throws -> Int {
    var size = 0
    guard sysctlbyname(
        "iogpu.wired_limit_mb", nil, &size, nil, 0) == 0,
        size > 0
    else {
        throw BenchQualificationRuntimeError.wiredLimitUnavailable
    }
    let megabytes: Int
    if size == MemoryLayout<Int32>.size {
        var value: Int32 = 0
        guard sysctlbyname(
            "iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value > 0
        else {
            throw BenchQualificationRuntimeError.wiredLimitUnavailable
        }
        megabytes = Int(value)
    } else if size == MemoryLayout<Int64>.size {
        var value: Int64 = 0
        guard sysctlbyname(
            "iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value > 0, value <= Int64(Int.max)
        else {
            throw BenchQualificationRuntimeError.wiredLimitUnavailable
        }
        megabytes = Int(value)
    } else {
        throw BenchQualificationRuntimeError.wiredLimitUnavailable
    }
    let (bytes, overflow) = megabytes.multipliedReportingOverflow(
        by: 1_048_576)
    guard !overflow else {
        throw BenchQualificationRuntimeError.wiredLimitOverflow
    }
    return bytes
}

private func benchQualificationProcessMemory() throws -> (
    residentSizeBytes: Int, physicalFootprintBytes: Int
) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size
            / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(
            to: integer_t.self, capacity: Int(count)
        ) {
            task_info(
                mach_task_self_,
                task_flavor_t(TASK_VM_INFO),
                $0,
                &count)
        }
    }
    guard result == KERN_SUCCESS, info.resident_size > 0,
        info.phys_footprint > 0,
        info.resident_size <= UInt64(Int.max),
        info.phys_footprint <= UInt64(Int.max)
    else {
        throw BenchQualificationRuntimeError.processMemoryUnavailable
    }
    return (
        residentSizeBytes: Int(info.resident_size),
        physicalFootprintBytes: Int(info.phys_footprint))
}

private func benchQualificationPowerSource()
    -> CompressedAttentionProbePowerSource
{
    guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
        let source = IOPSGetProvidingPowerSourceType(snapshot)?
            .takeUnretainedValue()
    else { return .unavailable }
    let value = ((source as NSString) as String).lowercased()
    if value.contains("battery") { return .battery }
    if value.contains("ac") || value.contains("ups") { return .acPower }
    return .unavailable
}

private func benchQualificationThermalState(
    _ state: ProcessInfo.ThermalState
) -> CompressedAttentionProbeThermalState {
    switch state {
    case .nominal: .nominal
    case .fair: .fair
    case .serious: .serious
    case .critical: .critical
    @unknown default: .unknown
    }
}
