import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// A pure, injectable description of a host's memory topology. **No introspection here** —
/// reading `sysctlbyname`/Metal/IOKit live is deferred engine work (spec §3); this type is the
/// data shape the capacity model consumes, constructed by the caller (tests, or eventually the
/// real profiler) from whatever source is appropriate for that context.
///
/// `detectHost()` below is the one exception: a minimal, MLX-free `sysctlbyname` read for the
/// model-sizer CLI's `--auto` mode, kept intentionally narrow (RAM + chip string only) rather than
/// duplicating `SystemProfiler`'s fuller probe (which also needs `Metal` for GPU/disk fields).
public struct SystemProfile: Sendable {
    public let chip: String
    /// Total physical RAM, in bytes (`ProcessInfo.physicalMemory` / `hw.memsize` on the real box).
    public let totalRAMBytes: Int
    /// The effective GPU wired-memory ceiling, in bytes (`iogpu.wired_limit_mb` × 1024² on the
    /// real box; `0` from the sysctl means "system default", which callers must resolve to a
    /// concrete byte count before constructing this profile — this type only holds resolved values).
    public let wiredLimitBytes: Int
    /// `true` when `wiredLimitBytes` came from an actual `iogpu.wired_limit_mb` reading (or a
    /// hand-measured preset); `false` when it was synthesized/estimated from a fraction of RAM
    /// (`detectHost()`, or the unmeasured `.m3Ultra256`/`.m3Ultra512` presets). Callers (the model
    /// sizer, the CLI) use this to flag headroom numbers as approximate rather than presenting an
    /// estimate as a measured fact. Defaults to `true` to preserve existing call sites that
    /// construct a `SystemProfile` from a genuinely measured/assumed-authoritative value.
    public let wiredLimitIsMeasured: Bool

    public init(chip: String, totalRAMBytes: Int, wiredLimitBytes: Int, wiredLimitIsMeasured: Bool = true) {
        self.chip = chip
        self.totalRAMBytes = totalRAMBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.wiredLimitIsMeasured = wiredLimitIsMeasured
    }

    /// The bytes actually available to hold model weights + KV + transient overhead once the
    /// smaller of {wired limit, physical RAM} is reduced by the weights already resident and an
    /// OS/other-processes reserve. This is the `hardwareHolds` function from spec §4/§5 — the
    /// single headroom computation both the context ceiling and the capacity advisor share.
    public func hardwareHoldsBytes(weightsBytes: Int, osReserveBytes: Int) -> Int {
        min(wiredLimitBytes, totalRAMBytes) - weightsBytes - osReserveBytes
    }

    private static let gib = 1024 * 1024 * 1024

    /// Qualified 128 GiB bench-host profile with `iogpu.wired_limit_mb=117760` (115 GiB exactly),
    /// retained as an explicit measured value rather than a default-percentage guess.
    public static let m5Max128 = SystemProfile(
        chip: "Apple M5 Max",
        totalRAMBytes: 128 * gib,
        wiredLimitBytes: 115 * gib,
        wiredLimitIsMeasured: true
    )

    /// ⚠️ ASSUMPTION: no confirmed `iogpu.wired_limit_mb` reading exists yet for an M3 Ultra
    /// 256GB box (none is provisioned per `environments.md`). Spec §3 documents the *system
    /// default* wired limit as "~66–75% RAM" when the sysctl reads `0`; this preset takes the
    /// upper end of that range (75%) as a defensible planning number. Replace with a measured
    /// value once the box exists and is profiled.
    public static let m3Ultra256 = SystemProfile(
        chip: "Apple M3 Ultra",
        totalRAMBytes: 256 * gib,
        wiredLimitBytes: Int(0.75 * Double(256 * gib)),
        wiredLimitIsMeasured: false
    )

    /// ⚠️ ASSUMPTION: same caveat as `.m3Ultra256` — no measured wired limit for a 512GB M3
    /// Ultra; 75% of RAM used as the planning default per spec §3's "~66–75%" system default.
    public static let m3Ultra512 = SystemProfile(
        chip: "Apple M3 Ultra",
        totalRAMBytes: 512 * gib,
        wiredLimitBytes: Int(0.75 * Double(512 * gib)),
        wiredLimitIsMeasured: false
    )

    /// Estimated `wiredLimitBytes` for a host whose `iogpu.wired_limit_mb` cannot be read directly
    /// (no such sysctl exists — spec §3). NOT a measurement: a documented, monotonically
    /// non-increasing fraction of total RAM, chosen to agree with the two data points this repo
    /// actually has —
    ///   - `.m5Max128`'s hand-measured 115/128 GiB ≈ 0.898 (rounds to "~0.90" in the small-box case)
    ///   - `.m3Ultra256`/`.m3Ultra512`'s own documented planning assumption of 0.75 (spec §3's
    ///     "system default ~66–75% RAM", upper end)
    /// A single global fraction can't fit both: small boxes can dedicate a larger share of RAM to
    /// the GPU wired limit (less OS/other-process pressure in absolute terms), while big boxes
    /// reserve a larger absolute (though smaller relative) slice for the OS and other processes.
    /// This clamped linear interpolation is conservative in both directions (never guesses higher
    /// than either endpoint) and monotonic (larger RAM never yields a higher fraction):
    ///   RAM <= 128 GiB  -> 0.90
    ///   RAM >= 256 GiB  -> 0.75
    ///   in between      -> linear ramp from 0.90 down to 0.75
    public static func estimatedWiredLimitBytes(totalRAMBytes: Int) -> Int {
        let ramGiB = Double(totalRAMBytes) / Double(gib)
        let highFraction = 0.90, lowFraction = 0.75
        let highRAMGiB = 128.0, lowRAMGiB = 256.0
        let fraction: Double
        if ramGiB <= highRAMGiB {
            fraction = highFraction
        } else if ramGiB >= lowRAMGiB {
            fraction = lowFraction
        } else {
            let t = (ramGiB - highRAMGiB) / (lowRAMGiB - highRAMGiB)
            fraction = highFraction - t * (highFraction - lowFraction)
        }
        return Int(fraction * Double(totalRAMBytes))
    }

    /// Probe the live host's RAM + chip string via `sysctlbyname`/`ProcessInfo` and synthesize the
    /// rest (spec: "model sizer v1", operator-facing `--auto` mode). `wiredLimitBytes` is always
    /// estimated here (see `estimatedWiredLimitBytes`) — there is no direct sysctl for the GPU
    /// wired-memory ceiling — so `wiredLimitIsMeasured` is always `false` for a detected host.
    public static func detectHost() -> SystemProfile {
        var ramSize: Int64 = 0
        var ramSizeLen = MemoryLayout<Int64>.size
        var totalRAMBytes = Int(ProcessInfo.processInfo.physicalMemory)
        if sysctlbyname("hw.memsize", &ramSize, &ramSizeLen, nil, 0) == 0, ramSize > 0 {
            totalRAMBytes = Int(ramSize)
        }

        let chip = sysctlString("machdep.cpu.brand_string") ?? sysctlString("hw.model") ?? "unknown"
        let wiredLimitBytes = estimatedWiredLimitBytes(totalRAMBytes: totalRAMBytes)

        return SystemProfile(
            chip: chip, totalRAMBytes: totalRAMBytes, wiredLimitBytes: wiredLimitBytes,
            wiredLimitIsMeasured: false)
    }

    /// Minimal C-string sysctl read (mirrors `SystemProfiler.sysctlString`, duplicated narrowly
    /// here rather than pulling `SystemProfiler` — and the `Metal` import it requires — into
    /// HarnessCore, which must stay MLX/GPU-toolchain-free).
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
