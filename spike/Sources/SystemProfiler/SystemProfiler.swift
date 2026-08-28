import Darwin
import Foundation
import Metal
import HarnessCore

/// Real host introspection for the capacity advisor (spec §3). **MLX-free by design**: this target
/// imports only `Foundation`, `Darwin`, `Metal` (a system framework), and `HarnessCore` — no
/// `MLX`/`MLXNN`/`SpikeCore` — so the capacity advisor builds and runs on any Mac without the full
/// inference-engine toolchain.
public enum SystemProfiler {

    /// Shared/default planning never exceeds floor(75% of physical RAM). Keep this thin bridge to
    /// the pure policy so the live probe cannot drift through floating-point recomputation.
    static func defaultSharedWiredLimitBytes(totalRAMBytes: Int) -> Int {
        SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: totalRAMBytes)
    }

    /// Probe the live host and return everything the capacity advisor + operator-facing CLI need.
    /// Every field is read from a real syscall/Metal API — see the individual helpers below for the
    /// exact source per spec §3's table. Tolerates absent MIBs/APIs by returning `nil` (or a
    /// documented synthesized fallback for `wiredLimitBytes`) rather than crashing.
    public static func probe() -> HostReport {
        let device = MTLCreateSystemDefaultDevice()

        let chip = device?.name
            ?? sysctlString("machdep.cpu.brand_string")
            ?? sysctlString("hw.model")
            ?? "unknown"

        let totalRAMBytes = Int(ProcessInfo.processInfo.physicalMemory)

        // P/E core split: perflevel0 = performance cores, perflevel1 = efficiency cores on
        // Apple Silicon. Older/uniform hosts (or hosts without the MIB) fall back to
        // hw.physicalcpu as all-P, 0 E.
        let pCores: Int
        let eCores: Int
        if let p = sysctlInt("hw.perflevel0.physicalcpu") {
            pCores = p
            eCores = sysctlInt("hw.perflevel1.physicalcpu") ?? 0
        } else {
            pCores = sysctlInt("hw.physicalcpu") ?? 0
            eCores = 0
        }

        // iogpu.wired_limit_mb: 0 or absent means "system default", documented as ~66-75% of RAM
        // (spec §3) — NOT unlimited. Synthesize 75% of RAM as the planning number and flag it so
        // callers never mistake a guess for a reading.
        let wiredLimitMB = sysctlInt("iogpu.wired_limit_mb") ?? 0
        let wiredLimitBytes: Int
        let wiredLimitIsDefault: Bool
        if wiredLimitMB > 0 {
            wiredLimitBytes = wiredLimitMB * 1024 * 1024
            wiredLimitIsDefault = false
        } else {
            wiredLimitBytes = defaultSharedWiredLimitBytes(totalRAMBytes: totalRAMBytes)
            wiredLimitIsDefault = true
        }

        let currentGPUAllocBytes = device.map { Int($0.currentAllocatedSize) }
        let recommendedWorkingSetBytes = device.map { Int($0.recommendedMaxWorkingSetSize) }

        let (diskInternal, diskFreeBytes) = probeDisk()

        // TODO(spec §7 incremental): NVMe vs external interconnect detection via IOKit
        // (kIOPropertyMediumTypeKey / kIOPropertyPhysicalInterconnectTypeKey) — deferred, hard
        // (IORegistry + DiskArbitration), re-verify constant spellings vs current SDK first.
        // TODO(spec §7 incremental): os_proc_available_memory() process-headroom reading — needs a
        // tiny C/ObjC shim, not exposed in the Darwin Swift module.

        return HostReport(
            chip: chip,
            totalRAMBytes: totalRAMBytes,
            wiredLimitBytes: wiredLimitBytes,
            wiredLimitIsDefault: wiredLimitIsDefault,
            pCores: pCores,
            eCores: eCores,
            currentGPUAllocBytes: currentGPUAllocBytes,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            diskInternal: diskInternal,
            diskFreeBytes: diskFreeBytes,
            hostUse: .automaticShared
        )
    }

    private static func probeDisk() -> (Bool?, Int?) {
        let home = URL(fileURLWithPath: NSHomeDirectory())
        guard let values = try? home.resourceValues(forKeys: [
            .volumeIsInternalKey, .volumeAvailableCapacityForImportantUsageKey,
        ]) else {
            return (nil, nil)
        }
        let diskInternal = values.volumeIsInternal
        let diskFreeBytes = values.volumeAvailableCapacityForImportantUsage.map { Int($0) }
        return (diskInternal, diskFreeBytes)
    }

    // MARK: - sysctl helpers

    /// Reads an integer-valued sysctl (works for both `Int32`- and `Int64`/`quad`-sized MIBs by
    /// sizing the buffer to whatever `sysctlbyname` reports first).
    private static func sysctlInt(_ name: String) -> Int? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }

        if size == MemoryLayout<Int32>.size {
            var value: Int32 = 0
            guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
            return Int(value)
        } else {
            var value: Int64 = 0
            var size64 = MemoryLayout<Int64>.size
            guard sysctlbyname(name, &value, &size64, nil, 0) == 0 else { return nil }
            return Int(value)
        }
    }

    /// Reads a C-string-valued sysctl via the standard two-call pattern: query the size, then
    /// allocate a buffer of that size and read into it.
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        // Decode up to the first NUL as UTF-8 (the `[CChar]` `String(cString:)` overload is
        // deprecated in Swift 6.3): sysctl returns a NUL-terminated C string.
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}

/// The result of `SystemProfiler.probe()` — everything the capacity advisor + operator-facing CLI
/// need from the live host, plus the derived `HarnessCore.SystemProfile` the pure capacity model
/// consumes.
public struct HostReport: Sendable {
    public let chip: String
    public let totalRAMBytes: Int
    public let wiredLimitBytes: Int
    /// `true` when `wiredLimitBytes` was synthesized (sysctl read 0/absent) rather than read
    /// directly — spec §3: "0 = system default (~66-75% RAM), not unlimited". Callers must not
    /// present a synthesized value as a measured reading.
    public let wiredLimitIsDefault: Bool
    public let pCores: Int
    public let eCores: Int
    public let currentGPUAllocBytes: Int?
    public let recommendedWorkingSetBytes: Int?
    public let diskInternal: Bool?
    public let diskFreeBytes: Int?
    public let hostUse: HostUseClassification

    public init(
        chip: String, totalRAMBytes: Int, wiredLimitBytes: Int, wiredLimitIsDefault: Bool,
        pCores: Int, eCores: Int, currentGPUAllocBytes: Int?, recommendedWorkingSetBytes: Int?,
        diskInternal: Bool?, diskFreeBytes: Int?,
        hostUse: HostUseClassification = .defaultShared
    ) {
        self.chip = chip
        self.totalRAMBytes = totalRAMBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.wiredLimitIsDefault = wiredLimitIsDefault
        self.pCores = pCores
        self.eCores = eCores
        self.currentGPUAllocBytes = currentGPUAllocBytes
        self.recommendedWorkingSetBytes = recommendedWorkingSetBytes
        self.diskInternal = diskInternal
        self.diskFreeBytes = diskFreeBytes
        self.hostUse = hostUse
    }

    /// The `HarnessCore.SystemProfile` the pure capacity model consumes, stripped of fields that
    /// are CLI-display-only while preserving capacity provenance.
    public var systemProfile: SystemProfile {
        SystemProfile(
            chip: chip,
            totalRAMBytes: totalRAMBytes,
            wiredLimitBytes: wiredLimitBytes,
            wiredLimitIsMeasured: !wiredLimitIsDefault,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            hostUse: hostUse)
    }

    /// Apply an explicit operator classification to the already-probed snapshot. This must copy
    /// every observation instead of probing again: startup evidence and fit planning need one
    /// coherent point-in-time view of the host.
    public func applyingHostUse(_ selectedHostUse: HostUseClassification) -> HostReport {
        HostReport(
            chip: chip,
            totalRAMBytes: totalRAMBytes,
            wiredLimitBytes: wiredLimitBytes,
            wiredLimitIsDefault: wiredLimitIsDefault,
            pCores: pCores,
            eCores: eCores,
            currentGPUAllocBytes: currentGPUAllocBytes,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            diskInternal: diskInternal,
            diskFreeBytes: diskFreeBytes,
            hostUse: selectedHostUse)
    }

    /// Stable operator-facing fields for the serve startup record. Metal's recommended working set
    /// is advisory and the current allocation is a point-in-time observation; the names preserve
    /// those semantics instead of promoting either value to a hard or historical measurement.
    public func machineReadableMemoryFields() -> String {
        let profile = systemProfile
        let effective = profile.effectiveMemoryCeiling
        let wiredProvenance = wiredLimitIsDefault ? "synthesized" : "measured"
        let recommended = recommendedWorkingSetBytes.map(String.init) ?? "unavailable"
        let current = currentGPUAllocBytes.map(String.init) ?? "unavailable"
        return [
            "host_use=\(hostUse.rawValue)",
            "host_use_source=\(hostUse.source.rawValue)",
            "host_use_policy_version=\(hostUse.policyVersion)",
            "host_physical_ram_bytes=\(totalRAMBytes)",
            "host_wired_limit_bytes=\(wiredLimitBytes)",
            "host_wired_limit_provenance=\(wiredProvenance)",
            "host_metal_recommended_working_set_bytes=\(recommended)",
            "host_metal_current_allocated_bytes=\(current)",
            "host_effective_memory_ceiling_bytes=\(effective.bytes)",
            "host_effective_memory_ceiling_source=\(effective.source.rawValue)",
        ].joined(separator: " ")
    }

    /// Complete startup fragment: the one-probe host observations plus the final process limits.
    /// A scalar backend has no separately reserved KV pool, while a transport-only backend has no
    /// MLX allocator at all; spell both states explicitly instead of inventing a numeric budget.
    public func machineReadableServingFields(
        memoryLimitBytes: Int?,
        cacheLimitBytes: Int?,
        kvBudgetBytes: Int?,
        osServiceReserveBytes: Int
    ) -> String {
        let memory = memoryLimitBytes.map(String.init) ?? "not-applicable"
        let cache = cacheLimitBytes.map(String.init) ?? "not-applicable"
        let kv: String
        if let kvBudgetBytes {
            kv = String(kvBudgetBytes)
        } else if memoryLimitBytes != nil {
            kv = "not-separately-reserved"
        } else {
            kv = "not-applicable"
        }
        return machineReadableMemoryFields()
            + " host_os_service_reserve_bytes=\(osServiceReserveBytes)"
            + " mlx_memory_limit_bytes=\(memory)"
            + " mlx_cache_limit_bytes=\(cache)"
            + " mlx_kv_budget_bytes=\(kv)"
    }
}
