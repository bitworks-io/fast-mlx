import Foundation

/// Detects when the OS's `iogpu.wired_limit_mb` ceiling has drifted ABOVE the budget fast-mlx
/// actually applies (`SystemProfile.effectiveMemoryCeiling`). `hostCeilingBeforeOperatorBudget`'s
/// shared-mode policy only consults the wired limit when it is LOWER than the exact-75%-RAM cap
/// (`SystemProfile.swift`, `if wiredLimitBytes > 0, wiredLimitBytes < shared.bytes`) — a wired
/// limit raised above that budget is silently ignored there, and fast-mlx has no other place that
/// would notice.
///
/// This is EXTERNAL-CEILING DRIFT DETECTION, not a proven cause of any incident: a 128 GiB
/// Apple-silicon production host kernel-panicked with a watchdog timeout while serving, and an external agent had raised
/// `iogpu.wired_limit_mb` to ~115 GiB (~19 GiB above the 96 GiB / 75%-shared budget fast-mlx
/// applied) before that panic — but `MLXScalarServing.swift` pins `Memory.memoryLimit` to the
/// applied budget before any weight load, so the raised OS ceiling was never observed to bind. This
/// guard reports the divergence between the OS ceiling and the applied budget as a correlated
/// precondition worth operator attention, WITHOUT asserting it caused that panic.
///
/// This guard is advisory ONLY: it never changes any budget, ceiling, or allocation, and it never
/// refuses to start (the box runs under launchd `KeepAlive`; refusing here would trade a rare panic
/// for a guaranteed crash-loop, which is worse).
public struct WiredCeilingOvercommit: Equatable, Sendable {
    /// `SystemProfile.wiredLimitBytes` — the OS ceiling (`iogpu.wired_limit_mb`), in bytes.
    public let systemWiredLimitBytes: Int
    /// `SystemProfile.effectiveMemoryCeiling.bytes` — the budget fast-mlx actually applies.
    public let appliedBudgetBytes: Int
    /// `systemWiredLimitBytes - appliedBudgetBytes`.
    public let excessBytes: Int

    public init(systemWiredLimitBytes: Int, appliedBudgetBytes: Int, excessBytes: Int) {
        self.systemWiredLimitBytes = systemWiredLimitBytes
        self.appliedBudgetBytes = appliedBudgetBytes
        self.excessBytes = excessBytes
    }
}

/// Three-state result of comparing the OS wired ceiling against the applied budget. Distinguishes
/// "checked and found no drift" from "never checked" — collapsing both into a bare `nil` would let
/// an unconfigured host (whose `wiredLimitBytes` is SYNTHESIZED as exactly 75% of RAM, algebraically
/// identical to the shared-mode budget — see `SystemProfiler.probe()`) look identical in the logs
/// to a host that was actually measured and found healthy, when in fact nothing was measured at all.
public enum WiredCeilingAssessment: Equatable, Sendable {
    /// `profile.wiredLimitIsMeasured == false` (or the "measured" observation was degenerate,
    /// e.g. a non-positive reading) — there is no real OS-ceiling reading to compare against the
    /// applied budget. Unknown is not zero and is not evidence of health.
    case unmeasured
    /// The OS ceiling was measured and does not exceed the applied budget by more than the margin.
    case withinBudget(systemWiredLimitBytes: Int, appliedBudgetBytes: Int)
    /// The OS ceiling was measured and exceeds the applied budget by more than the margin.
    case overcommitted(WiredCeilingOvercommit)
}

public enum WiredCeilingOvercommitGuard {
    /// Fraction of the applied budget the system wired limit must exceed before this fires.
    ///
    /// PREDECLARED at 2%, chosen against two real observed geometries rather than tuned after the
    /// fact: the live production-host ceiling (`iogpu.wired_limit_mb` = 102400 MiB = 100 GiB
    /// against a 96 GiB shared budget, +4.17%) and the panic-incident ceiling (~115 GiB against the
    /// same 96 GiB budget, +19.8%). A 5% margin would be SILENT on the first geometry — the exact
    /// host this guard exists for — which is a check that passes for free. 2% still absorbs
    /// rounding / MiB-vs-GiB conversion noise (a wired limit reported in whole MiB against a budget
    /// computed in bytes) while firing on both real geometries.
    public static let defaultMarginFraction: Double = 0.02

    /// Compares the OS wired ceiling against the applied budget and returns which of the three
    /// states holds. This is the PRIMARY API — `evaluate(profile:marginFraction:)` is a thin
    /// wrapper over this that collapses the healthy/unmeasured distinction back to `nil` for
    /// callers that only care about the overcommit case.
    ///
    /// `.overcommitted` fires ONLY when ALL of the following hold:
    /// - `profile.wiredLimitIsMeasured == true` — a SYNTHESIZED/estimated wired limit must never
    ///   raise this warning.
    /// - `profile.wiredLimitBytes > 0`
    /// - the applied budget (`profile.effectiveMemoryCeiling.bytes`) is positive
    /// - `profile.wiredLimitBytes > budget + floor(budget * marginFraction)`
    ///
    /// Mode-aware by construction: on `.dedicatedServing`, `effectiveMemoryCeiling` is already
    /// `min(wiredLimit, RAM)`, so a wired limit at or below RAM never exceeds its own budget and
    /// this never fires there. It fires on `.shared` hosts where the wired limit was raised above
    /// the 75%-RAM policy cap that `hostCeilingBeforeOperatorBudget` silently ignores.
    public static func assess(
        profile: SystemProfile,
        marginFraction: Double = defaultMarginFraction
    ) -> WiredCeilingAssessment {
        guard profile.wiredLimitIsMeasured else { return .unmeasured }
        guard profile.wiredLimitBytes > 0 else { return .unmeasured }

        let budget = profile.effectiveMemoryCeiling.bytes
        guard budget > 0 else { return .unmeasured }

        let margin = Int((Double(budget) * marginFraction).rounded(.down))
        let threshold = budget + margin
        guard profile.wiredLimitBytes > threshold else {
            return .withinBudget(systemWiredLimitBytes: profile.wiredLimitBytes, appliedBudgetBytes: budget)
        }

        return .overcommitted(
            WiredCeilingOvercommit(
                systemWiredLimitBytes: profile.wiredLimitBytes,
                appliedBudgetBytes: budget,
                excessBytes: profile.wiredLimitBytes - budget))
    }

    /// Thin wrapper over `assess(profile:marginFraction:)` for callers that only need the
    /// overcommit payload (or `nil` for everything else). Prefer `assess` at new call sites so
    /// "healthy" and "never measured" stay distinguishable.
    public static func evaluate(
        profile: SystemProfile,
        marginFraction: Double = defaultMarginFraction
    ) -> WiredCeilingOvercommit? {
        guard case .overcommitted(let overcommit) = assess(profile: profile, marginFraction: marginFraction) else {
            return nil
        }
        return overcommit
    }

    private static func gib(_ bytes: Int) -> String {
        String(format: "%.2f GiB", Double(bytes) / 1_073_741_824.0)
    }

    /// Operator-facing advisory lines for every assessment state — same "NOTE:"-prefixed,
    /// GiB-plus-raw-bytes style as `ModelSizer.provenanceNotes` / `ServingFitDecision
    /// .summaryLines()`. Every case carries exactly ONE stable machine-readable
    /// `wired_ceiling_state=<value>` token (`overcommit` / `within_budget` / `unmeasured`) so gate
    /// scripts and log scrapers can key on it without parsing prose. Emitting a line even in the
    /// healthy `within_budget` case is deliberate: it proves in production logs that the check
    /// actually ran, rather than leaving "checked and fine" indistinguishable from "never checked".
    public static func advisoryLines(for assessment: WiredCeilingAssessment) -> [String] {
        switch assessment {
        case .unmeasured:
            return [
                "NOTE: wired_ceiling_state=unmeasured — the OS wired-memory ceiling "
                    + "(iogpu.wired_limit_mb) was not directly read on this host (sysctl absent, "
                    + "zero, or otherwise unmeasured); the applied budget was synthesized rather "
                    + "than compared against a real reading, so no ceiling-drift check could run.",
            ]
        case .withinBudget(let systemWiredLimitBytes, let appliedBudgetBytes):
            return [
                "NOTE: wired_ceiling_state=within_budget — measured OS wired ceiling "
                    + "\(systemWiredLimitBytes) B (\(gib(systemWiredLimitBytes))) is within the "
                    + "tolerated margin of the \(appliedBudgetBytes) B (\(gib(appliedBudgetBytes))) "
                    + "budget fast-mlx applies.",
            ]
        case .overcommitted(let overcommit):
            return [
                "NOTE: wired_ceiling_state=overcommit — the OS wired-memory ceiling "
                    + "(iogpu.wired_limit_mb) has drifted above the budget fast-mlx applies.",
                "  system_wired_limit=\(overcommit.systemWiredLimitBytes) B "
                    + "(\(gib(overcommit.systemWiredLimitBytes)))",
                "  applied_budget=\(overcommit.appliedBudgetBytes) B (\(gib(overcommit.appliedBudgetBytes)))",
                "  excess=\(overcommit.excessBytes) B (\(gib(overcommit.excessBytes)))",
                "  fast-mlx does not raise or enforce the OS wired ceiling, and pins its own memory "
                    + "limit to the applied budget before any weight load, so this excess was never "
                    + "observed to bind; this is external-ceiling drift detection (the OS ceiling and "
                    + "the budget fast-mlx applies have diverged, which warrants operator attention), "
                    + "NOT a proven cause of any incident. The same drift was present ahead of a "
                    + "watchdog panic on a production host.",
            ]
        }
    }
}
