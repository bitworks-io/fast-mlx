import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Durable host-use provenance for capacity policy. The default/automatic policy is conservative:
/// absent or ambiguous host tenancy is treated as shared. Dedicated serving is only created by an
/// explicit operator assertion (the factory, or a decoded configuration whose source says exactly
/// that), so callers cannot infer it from RAM, chip, or wired-limit shape.
///
/// This is provenance, not authorization: decoding an evidence artifact never grants permission to
/// mutate the host. The later operator-facing admission boundary must require its own explicit input.
public struct HostUseClassification: Sendable, Equatable, Codable {
    public enum Use: String, Sendable, Codable {
        case shared
        case dedicatedServing = "dedicated-serving"
    }

    public enum Source: String, Sendable, Codable {
        case `default` = "default"
        case automatic = "automatic"
        case operatorAssertion = "operator-assertion"
    }

    public static let currentPolicyVersion = "host-use/v1"

    public let use: Use
    public let source: Source
    public let policyVersion: String

    public var rawValue: String { use.rawValue }

    private enum CodingKeys: String, CodingKey {
        case use
        case source
        case policyVersion
    }

    private init(use: Use, source: Source, policyVersion: String = Self.currentPolicyVersion) {
        self.use = use
        self.source = source
        self.policyVersion = policyVersion
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let use = try container.decode(Use.self, forKey: .use)
        let source = try container.decode(Source.self, forKey: .source)
        let policyVersion = try container.decode(String.self, forKey: .policyVersion)
        guard policyVersion == Self.currentPolicyVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .policyVersion,
                in: container,
                debugDescription: "unsupported host-use policy version")
        }
        guard use == .shared || source == .operatorAssertion else {
            throw DecodingError.dataCorruptedError(
                forKey: .source,
                in: container,
                debugDescription: "dedicated-serving host use requires operator-assertion source")
        }
        self.init(use: use, source: source, policyVersion: policyVersion)
    }

    public static let defaultShared = HostUseClassification(use: .shared, source: .default)
    public static let automaticShared = HostUseClassification(use: .shared, source: .automatic)

    public static func operatorAssertedShared() -> HostUseClassification {
        HostUseClassification(
            use: .shared,
            source: .operatorAssertion,
            policyVersion: Self.currentPolicyVersion)
    }

    public static func operatorAssertedDedicatedServing() -> HostUseClassification {
        HostUseClassification(
            use: .dedicatedServing,
            source: .operatorAssertion,
            policyVersion: Self.currentPolicyVersion)
    }
}

/// The conservative process-planning envelope selected from the host observations that apply to
/// the current host-use policy. The source is retained separately from the wired-limit measurement
/// flag: a Metal recommendation can bind a shared host without changing where the wired input came
/// from.
public struct EffectiveMemoryCeiling: Sendable, Equatable {
    public enum Source: String, Sendable, Equatable {
        case physicalRAM = "physical-ram"
        case sharedPolicy = "shared-policy"
        case wiredLimit = "wired-limit"
        case recommendedWorkingSet = "metal-recommended-working-set"
        case operatorBudget = "operator-budget"
    }

    public let bytes: Int
    public let source: Source
}

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
    /// The resolved GPU wired-memory observation, in bytes (`iogpu.wired_limit_mb` × 1024² on the
    /// real box; `0` from the sysctl means "system default", which callers must resolve to a
    /// concrete byte count before constructing this profile). Shared capacity math consumes
    /// `effectiveMemoryCeiling`, which may be lower than this observation.
    public let wiredLimitBytes: Int
    /// `true` when `wiredLimitBytes` came from an actual `iogpu.wired_limit_mb` reading (or a
    /// hand-measured preset); `false` when it was synthesized/estimated from a fraction of RAM
    /// (`detectHost()`, or the unmeasured `.m3Ultra256`/`.m3Ultra512` presets). Callers (the model
    /// sizer, the CLI) use this to flag headroom numbers as approximate rather than presenting an
    /// estimate as a measured fact. Defaults to `true` to preserve existing call sites that
    /// construct a `SystemProfile` from a genuinely measured/assumed-authoritative value.
    public let wiredLimitIsMeasured: Bool
    /// Metal's approximate good-performance working-set threshold. This is an advisory observation,
    /// not a hard allocation guarantee. A positive value conservatively bounds shared-host planning;
    /// dedicated-serving qualification remains a separate operator-controlled gate.
    public let recommendedWorkingSetBytes: Int?
    public let hostUse: HostUseClassification
    /// An operator-asserted planning budget, in bytes. Applied as a FURTHER reduction of the
    /// `effectiveMemoryCeiling` that `hostUse` already computed — never an increase, and never a
    /// substitute for an already-tighter measured/advisory bound. `nil` (the default) preserves
    /// every existing call site's behavior exactly. Declared last, with a default, so existing
    /// positional/keyword call sites keep compiling unchanged.
    public let operatorMemoryBudgetBytes: Int?

    public init(
        chip: String,
        totalRAMBytes: Int,
        wiredLimitBytes: Int,
        wiredLimitIsMeasured: Bool = true,
        recommendedWorkingSetBytes: Int? = nil,
        hostUse: HostUseClassification = .defaultShared,
        operatorMemoryBudgetBytes: Int? = nil
    ) {
        self.chip = chip
        self.totalRAMBytes = totalRAMBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.wiredLimitIsMeasured = wiredLimitIsMeasured
        self.recommendedWorkingSetBytes = recommendedWorkingSetBytes
        self.hostUse = hostUse
        self.operatorMemoryBudgetBytes = operatorMemoryBudgetBytes
    }

    /// The host-use-driven envelope BEFORE any operator budget is applied: exactly what
    /// `hostUse`'s shared/dedicated-serving policy computes from the host's own observations,
    /// including the `totalRAMBytes <= 0` early return. This is the single place that branch
    /// logic is written; `effectiveMemoryCeiling` is defined in terms of this property plus the
    /// operator budget. Its `.source` is never `.operatorBudget` — an operator budget cannot be
    /// "applied" here because it hasn't been considered yet.
    ///
    /// Shared hosts are capped at exact integer `floor(75% * physical RAM)`, then further bounded
    /// by a lower positive wired-limit observation and a lower positive Metal recommended working
    /// set. Missing/non-positive Metal observations are unavailable rather than zero-byte limits.
    /// Dedicated-serving retains the pre-existing `min(wired limit, physical RAM)` behavior here;
    /// its separate qualification path owns any future Metal/OS/service-reserve policy.
    public var hostCeilingBeforeOperatorBudget: EffectiveMemoryCeiling {
        guard totalRAMBytes > 0 else {
            return EffectiveMemoryCeiling(bytes: 0, source: .physicalRAM)
        }

        switch hostUse.use {
        case .shared:
            var shared = EffectiveMemoryCeiling(
                bytes: Self.estimatedWiredLimitBytes(totalRAMBytes: totalRAMBytes),
                source: .sharedPolicy)
            if wiredLimitBytes > 0, wiredLimitBytes < shared.bytes {
                shared = EffectiveMemoryCeiling(bytes: wiredLimitBytes, source: .wiredLimit)
            }
            if let recommendedWorkingSetBytes,
                recommendedWorkingSetBytes > 0,
                recommendedWorkingSetBytes < shared.bytes
            {
                shared = EffectiveMemoryCeiling(
                    bytes: recommendedWorkingSetBytes,
                    source: .recommendedWorkingSet)
            }
            return shared

        case .dedicatedServing:
            if wiredLimitBytes <= 0 {
                return EffectiveMemoryCeiling(bytes: 0, source: .wiredLimit)
            } else if wiredLimitBytes < totalRAMBytes {
                return EffectiveMemoryCeiling(bytes: wiredLimitBytes, source: .wiredLimit)
            } else {
                return EffectiveMemoryCeiling(bytes: totalRAMBytes, source: .physicalRAM)
            }
        }
    }

    /// The envelope all capacity, admission, and allocator/cache budgeting must consume: the
    /// `hostCeilingBeforeOperatorBudget` policy result, further bounded by an operator-asserted
    /// budget when one binds.
    public var effectiveMemoryCeiling: EffectiveMemoryCeiling {
        Self.applyOperatorBudget(hostCeilingBeforeOperatorBudget, operatorMemoryBudgetBytes)
    }

    /// The single place the operator budget rule is written: it binds only when it is a positive,
    /// strictly-tighter bound than whatever the hostUse-driven policy already computed. A budget
    /// can only ever reduce the envelope, never raise it or stand in for an already-tighter bound
    /// (equal is not "binding" — ties keep the pre-existing source).
    private static func applyOperatorBudget(
        _ result: EffectiveMemoryCeiling,
        _ operatorMemoryBudgetBytes: Int?
    ) -> EffectiveMemoryCeiling {
        guard let operatorMemoryBudgetBytes,
            operatorMemoryBudgetBytes > 0,
            operatorMemoryBudgetBytes < result.bytes
        else {
            return result
        }
        return EffectiveMemoryCeiling(bytes: operatorMemoryBudgetBytes, source: .operatorBudget)
    }

    /// Whether the observation that selected `hostCeilingBeforeOperatorBudget` — the host's own
    /// policy result, BEFORE any operator budget is applied — is measured rather than synthesized
    /// or advisory. Physical-RAM provenance is not represented by this type yet, and Metal
    /// documents its recommendation as an approximation, so both remain conservative.
    ///
    /// This deliberately switches on `hostCeilingBeforeOperatorBudget.source`, not
    /// `effectiveMemoryCeiling.source`: this field describes the provenance of the *host
    /// observation*, and an operator budget does not degrade that provenance merely by binding
    /// tighter than it. An operator-supplied budget is an exact figure the operator chose — it
    /// erases which host observation would otherwise have bound, but it does not make that
    /// observation's own provenance any less measured. (Review finding: making this switch on
    /// `effectiveMemoryCeiling.source` with a `.operatorBudget -> false` arm would flip
    /// `fit_estimate_measured` from `true` to `false` on a fully-measured host purely because an
    /// operator supplied a budget, implying headroom became approximate when it did not.)
    public var effectiveMemoryCeilingIsMeasured: Bool {
        switch hostCeilingBeforeOperatorBudget.source {
        case .wiredLimit:
            return wiredLimitIsMeasured
        case .physicalRAM, .sharedPolicy, .recommendedWorkingSet, .operatorBudget:
            // `.operatorBudget` is unreachable here by construction: `hostCeilingBeforeOperatorBudget`
            // is computed before any operator budget is considered and never carries this source.
            // Kept explicit (grouped, not a `default:`) so the switch stays exhaustive and visibly
            // covers every `EffectiveMemoryCeiling.Source` case.
            return false
        }
    }

    /// The bytes actually available to hold model weights + KV + transient overhead once the
    /// smaller of {wired limit, physical RAM} is reduced by the weights already resident and an
    /// OS/other-processes reserve. This is the `hardwareHolds` function from spec §4/§5 — the
    /// single headroom computation both the context ceiling and the capacity advisor share.
    public func hardwareHoldsBytes(weightsBytes: Int, osReserveBytes: Int) -> Int {
        effectiveMemoryCeiling.bytes - weightsBytes - osReserveBytes
    }

    /// Returns a copy of this profile with `operatorMemoryBudgetBytes` replaced by `bytes` and
    /// every other field preserved. Pass `nil` to clear a previously-set budget.
    public func withOperatorMemoryBudget(_ bytes: Int?) -> SystemProfile {
        SystemProfile(
            chip: chip,
            totalRAMBytes: totalRAMBytes,
            wiredLimitBytes: wiredLimitBytes,
            wiredLimitIsMeasured: wiredLimitIsMeasured,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            hostUse: hostUse,
            operatorMemoryBudgetBytes: bytes)
    }

    private static let gib = 1024 * 1024 * 1024

    /// Qualified 128 GiB hardware profile with a measured 115 GiB wired observation. Host use still
    /// defaults to shared: the hardware preset is evidence, not an operator dedication assertion.
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

    /// Estimated `wiredLimitBytes` for a shared/default auto host whose `iogpu.wired_limit_mb`
    /// cannot be read directly. NOT a measurement: spec §3 describes the system default as
    /// "~66-75% RAM"; capacity planning uses the upper bound exactly and floors fractional bytes.
    public static func estimatedWiredLimitBytes(totalRAMBytes: Int) -> Int {
        (totalRAMBytes / 4) * 3 + ((totalRAMBytes % 4) * 3) / 4
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
            wiredLimitIsMeasured: false, hostUse: .automaticShared)
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
