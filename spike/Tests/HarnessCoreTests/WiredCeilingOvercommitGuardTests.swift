import Testing

@testable import HarnessCore

/// `WiredCeilingOvercommitGuard` is the advisory-only detector for external-ceiling drift: an
/// external agent (or launchd/operator action) raises the OS `iogpu.wired_limit_mb` sysctl ABOVE
/// the budget fast-mlx actually applies, and `SystemProfile.hostCeilingBeforeOperatorBudget`
/// silently ignores a wired limit that is HIGHER than the shared 75%-RAM policy cap (it only ever
/// tightens, never widens, via that observation). This guard never mutates any budget, ceiling, or
/// allocation and never refuses to start — it only produces an advisory payload, wired into the
/// serve startup path via `emitFitCheck` in `FastMLXServe.swift`.
struct WiredCeilingOvercommitGuardTests {
    private static let gib = 1024 * 1024 * 1024

    // MARK: - 1. POSITIVE: the live production ceiling (verified on the production host, 2026-09-10)

    /// `iogpu.wired_limit_mb` = 102400 MiB = 100 GiB against the 128 GiB host's 96 GiB (75%
    /// shared) budget: +4.17% over budget. This is the geometry that motivates the 2% margin — the
    /// prior 5% margin was SILENT here, a check that passed for free on the exact host it exists
    /// to cover.
    @Test func firesOnTheLiveProductionCeiling() {
        let measuredWiredLimitMB = 102_400
        let measuredWiredLimitBytes = measuredWiredLimitMB * 1024 * 1024
        // Exact-integer sanity: 102400 MiB really is 100 GiB, with no rounding introduced here.
        #expect(measuredWiredLimitBytes == 100 * Self.gib)

        let profile = SystemProfile(
            chip: "Apple M5",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: measuredWiredLimitBytes,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)

        #expect(profile.effectiveMemoryCeiling.bytes == 96 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .sharedPolicy)

        let overcommit = WiredCeilingOvercommitGuard.evaluate(profile: profile)
        #expect(overcommit != nil)
        #expect(overcommit?.systemWiredLimitBytes == measuredWiredLimitBytes)
        #expect(overcommit?.appliedBudgetBytes == 96 * Self.gib)
        #expect(overcommit?.excessBytes == (100 * Self.gib) - (96 * Self.gib))

        guard case .overcommitted(let assessed) = WiredCeilingOvercommitGuard.assess(profile: profile) else {
            Issue.record("expected the live production ceiling to assess as overcommitted")
            return
        }
        #expect(assessed == overcommit)
    }

    // MARK: - 2. POSITIVE: reproduce the panic geometry

    @Test func firesOnThePanicGeometry() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)

        // Precondition check: the shared policy applies exact floor(75% * 128 GiB) = 96 GiB, and
        // the 115 GiB wired observation is ABOVE that (so `hostCeilingBeforeOperatorBudget` takes
        // the `sharedPolicy` branch, never even looking at the wired figure as a tightening bound).
        #expect(profile.effectiveMemoryCeiling.bytes == 96 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .sharedPolicy)

        let overcommit = WiredCeilingOvercommitGuard.evaluate(profile: profile)
        #expect(overcommit != nil)
        #expect(overcommit?.systemWiredLimitBytes == 115 * Self.gib)
        #expect(overcommit?.appliedBudgetBytes == 96 * Self.gib)
        #expect(overcommit?.excessBytes == 19 * Self.gib)
    }

    // MARK: - 3. NEGATIVE CONTROL: wired limit below the budget must not fire

    /// Identical to the panic geometry except the wired limit (90 GiB) sits BELOW the 96 GiB
    /// shared-policy figure — the healthy case the positive test's own budget math must be
    /// distinguished from. If this returned non-nil, the positive test above would be free (any
    /// wired limit at all would "fire"). Because 90 GiB is now the LOWER of the two, it itself
    /// becomes the applied budget here (`hostCeilingBeforeOperatorBudget`'s tightening branch), so
    /// the wired limit trivially equals the budget it would be compared against — no overcommit,
    /// and `assess` reports `.withinBudget` (checked, not merely unmeasured).
    @Test func doesNotFireWhenWiredLimitIsBelowBudget() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 90 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)

        #expect(profile.effectiveMemoryCeiling.bytes == 90 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .wiredLimit)
        #expect(WiredCeilingOvercommitGuard.evaluate(profile: profile) == nil)
        #expect(
            WiredCeilingOvercommitGuard.assess(profile: profile)
                == .withinBudget(systemWiredLimitBytes: 90 * Self.gib, appliedBudgetBytes: 90 * Self.gib))
    }

    // MARK: - 4. MARGIN: just inside the default 2% margin must not fire

    @Test func doesNotFireWhenOneMarginBelowThreshold() {
        // 4 GiB RAM -> exact floor(75%) = 3 GiB budget, so the arithmetic below is exact integer
        // math with no rounding ambiguity. wiredLimitBytes is set to budget + 1% (well inside the
        // default 2% margin), and stays above budget itself so it never becomes the tightening
        // bound inside `hostCeilingBeforeOperatorBudget` (budget would only change if wiredLimit
        // were LOWER than the shared policy figure).
        let budget = 3 * Self.gib
        let onePercentAbove = budget + budget / 100
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 4 * Self.gib,
            wiredLimitBytes: onePercentAbove,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)

        #expect(profile.effectiveMemoryCeiling.bytes == budget)
        #expect(WiredCeilingOvercommitGuard.evaluate(profile: profile) == nil)
    }

    // MARK: - 5. MARGIN: the exact boundary

    /// Chosen inclusivity: the guard fires on a STRICT `>` comparison against `budget +
    /// floor(budget * marginFraction)` — a wired limit exactly AT that threshold is still
    /// considered within tolerance and must NOT fire; one byte above it must fire. Both sides are
    /// asserted here so the boundary is pinned precisely rather than merely exercised.
    @Test func boundaryIsExclusiveAtThresholdAndInclusiveOneByteAbove() {
        let budget = 3 * Self.gib // 3_221_225_472
        let margin = Int((Double(budget) * WiredCeilingOvercommitGuard.defaultMarginFraction).rounded(.down))
        let threshold = budget + margin

        let atThreshold = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 4 * Self.gib,
            wiredLimitBytes: threshold,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)
        #expect(atThreshold.effectiveMemoryCeiling.bytes == budget)
        #expect(WiredCeilingOvercommitGuard.evaluate(profile: atThreshold) == nil)

        let oneByteAboveThreshold = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 4 * Self.gib,
            wiredLimitBytes: threshold + 1,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)
        #expect(oneByteAboveThreshold.effectiveMemoryCeiling.bytes == budget)
        let overcommit = WiredCeilingOvercommitGuard.evaluate(profile: oneByteAboveThreshold)
        #expect(overcommit != nil)
        #expect(overcommit?.excessBytes == (threshold + 1) - budget)
    }

    // MARK: - 6. UNKNOWN IS NOT ZERO, and is distinguishable from "checked and healthy"

    /// Same geometry as the panic reproduction, but the wired-limit observation is SYNTHESIZED
    /// rather than measured. A synthesized/estimated number must never trigger an operator-facing
    /// overcommit alarm — "we don't actually know the OS ceiling" is not evidence of overcommit.
    /// `assess` must report `.unmeasured` here, distinct from `.withinBudget`, so "never checked"
    /// cannot be mistaken for "checked and fine" downstream.
    @Test func doesNotFireWhenWiredLimitIsUnmeasured() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: false,
            hostUse: .defaultShared)

        #expect(WiredCeilingOvercommitGuard.evaluate(profile: profile) == nil)
        #expect(WiredCeilingOvercommitGuard.assess(profile: profile) == .unmeasured)
    }

    /// `SystemProfiler.probe()` SYNTHESIZES `wiredLimitBytes` as exactly 75% of RAM whenever the
    /// sysctl is absent — algebraically IDENTICAL to the shared-mode budget, so on an unconfigured
    /// host "no warning" was previously guaranteed by identity rather than by measurement. Proves
    /// that geometry now assesses as `.unmeasured` (not `.withinBudget`), and that the two states
    /// render textually distinguishable advisory lines rather than collapsing to the same prose.
    @Test func synthesizedWiredLimitIsUnmeasuredNotWithinBudget() {
        let ramBytes = 128 * Self.gib
        let syntheticSharedWiredLimit = SystemProfile.estimatedWiredLimitBytes(totalRAMBytes: ramBytes)
        let synthesizedProfile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: ramBytes,
            wiredLimitBytes: syntheticSharedWiredLimit,
            wiredLimitIsMeasured: false,
            hostUse: .defaultShared)

        // The synthesized figure and the shared budget really are identical here — the exact
        // condition that made the old nil-returning API unable to tell "healthy" from "unchecked".
        #expect(synthesizedProfile.effectiveMemoryCeiling.bytes == syntheticSharedWiredLimit)

        let synthesizedAssessment = WiredCeilingOvercommitGuard.assess(profile: synthesizedProfile)
        #expect(synthesizedAssessment == .unmeasured)

        let measuredWithinBudgetProfile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: ramBytes,
            wiredLimitBytes: 90 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)
        let measuredAssessment = WiredCeilingOvercommitGuard.assess(profile: measuredWithinBudgetProfile)
        #expect(measuredAssessment == .withinBudget(systemWiredLimitBytes: 90 * Self.gib, appliedBudgetBytes: 90 * Self.gib))

        let unmeasuredText = WiredCeilingOvercommitGuard.advisoryLines(for: synthesizedAssessment).joined(separator: "\n")
        let withinBudgetText = WiredCeilingOvercommitGuard.advisoryLines(for: measuredAssessment).joined(separator: "\n")
        #expect(unmeasuredText != withinBudgetText)
        #expect(unmeasuredText.contains("wired_ceiling_state=unmeasured"))
        #expect(withinBudgetText.contains("wired_ceiling_state=within_budget"))
    }

    // MARK: - 7. ADVISORY TEXT CARRIES BOTH NUMBERS, THE MACHINE TOKEN, AND NO CAUSAL OVERCLAIM

    @Test func advisoryLinesCarrySystemCeilingBudgetExcessAndMachineToken() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared)
        guard case .overcommitted(let overcommit) = WiredCeilingOvercommitGuard.assess(profile: profile) else {
            Issue.record("expected the panic geometry to assess as overcommitted")
            return
        }

        let text = WiredCeilingOvercommitGuard.advisoryLines(for: .overcommitted(overcommit)).joined(separator: "\n")

        #expect(text.contains("\(115 * Self.gib)"), "must name the system wired ceiling in raw bytes")
        #expect(text.contains("\(96 * Self.gib)"), "must name the applied budget in raw bytes")
        #expect(text.contains("\(19 * Self.gib)"), "must name the excess in raw bytes")
        #expect(text.contains("wired_ceiling_state=overcommit"))
        #expect(!text.contains("wired_ceiling_overcommit=true"), "the old token must be fully replaced")
        // Defect-4 wording discipline: this must read as drift detection, never as proven causation.
        #expect(!text.contains("cannot reclaim"), "must not repeat the retracted causal claim")
        #expect(text.contains("NOT a proven cause"))
    }

    // MARK: - 8. DEDICATED-SERVING does not false-positive

    /// On `.dedicatedServing`, `hostCeilingBeforeOperatorBudget` derives the budget as
    /// `min(wiredLimit, RAM)` — the budget already equals the wired limit whenever the wired limit
    /// is the tighter bound, so there is no overcommit to report. Proves the guard is mode-aware
    /// rather than firing on every host that merely has a large, measured wired limit; `assess`
    /// reports `.withinBudget` here (checked, not merely unmeasured) since the wired limit really
    /// was measured.
    @Test func doesNotFireOnDedicatedServing() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .operatorAssertedDedicatedServing())

        #expect(profile.effectiveMemoryCeiling.bytes == 115 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .wiredLimit)
        #expect(WiredCeilingOvercommitGuard.evaluate(profile: profile) == nil)
        #expect(
            WiredCeilingOvercommitGuard.assess(profile: profile)
                == .withinBudget(systemWiredLimitBytes: 115 * Self.gib, appliedBudgetBytes: 115 * Self.gib))
    }
}
