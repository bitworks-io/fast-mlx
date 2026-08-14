import Foundation

/// A pure, injectable description of a host's memory topology. **No introspection here** —
/// reading `sysctlbyname`/Metal/IOKit live is deferred engine work (spec §3); this type is the
/// data shape the capacity model consumes, constructed by the caller (tests, or eventually the
/// real profiler) from whatever source is appropriate for that context.
public struct SystemProfile: Sendable {
    public let chip: String
    /// Total physical RAM, in bytes (`ProcessInfo.physicalMemory` / `hw.memsize` on the real box).
    public let totalRAMBytes: Int
    /// The effective GPU wired-memory ceiling, in bytes (`iogpu.wired_limit_mb` × 1024² on the
    /// real box; `0` from the sysctl means "system default", which callers must resolve to a
    /// concrete byte count before constructing this profile — this type only holds resolved values).
    public let wiredLimitBytes: Int

    public init(chip: String, totalRAMBytes: Int, wiredLimitBytes: Int) {
        self.chip = chip
        self.totalRAMBytes = totalRAMBytes
        self.wiredLimitBytes = wiredLimitBytes
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
        wiredLimitBytes: 115 * gib
    )

    /// ⚠️ ASSUMPTION: no confirmed `iogpu.wired_limit_mb` reading exists yet for an M3 Ultra
    /// 256GB box (none is provisioned per `environments.md`). Spec §3 documents the *system
    /// default* wired limit as "~66–75% RAM" when the sysctl reads `0`; this preset takes the
    /// upper end of that range (75%) as a defensible planning number. Replace with a measured
    /// value once the box exists and is profiled.
    public static let m3Ultra256 = SystemProfile(
        chip: "Apple M3 Ultra",
        totalRAMBytes: 256 * gib,
        wiredLimitBytes: Int(0.75 * Double(256 * gib))
    )

    /// ⚠️ ASSUMPTION: same caveat as `.m3Ultra256` — no measured wired limit for a 512GB M3
    /// Ultra; 75% of RAM used as the planning default per spec §3's "~66–75%" system default.
    public static let m3Ultra512 = SystemProfile(
        chip: "Apple M3 Ultra",
        totalRAMBytes: 512 * gib,
        wiredLimitBytes: Int(0.75 * Double(512 * gib))
    )
}
