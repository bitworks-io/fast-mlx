import Testing

@testable import HarnessCore

/// `SystemProfile.operatorMemoryBudgetBytes` threads an operator-asserted planning budget into the
/// single `effectiveMemoryCeiling` envelope that the quant auto-pick, the serving fit planner, and
/// the MLX memory/cache limits all consume. The budget can only ever REDUCE the envelope that the
/// existing hostUse-driven policy already computed — never raise it, never substitute for a
/// measured/advisory bound that is already tighter. See
/// `docs/task-inbox/2026-09-06-operator-memory-limits-silently-discarded-DECISION.md`.
struct SystemProfileOperatorBudgetTests {
    private static let gib = 1024 * 1024 * 1024

    private func dedicatedFixture(
        operatorMemoryBudgetBytes: Int? = nil,
        recommendedWorkingSetBytes: Int? = nil
    ) -> SystemProfile {
        SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            hostUse: .operatorAssertedDedicatedServing(),
            operatorMemoryBudgetBytes: operatorMemoryBudgetBytes)
    }

    private func sharedFixture(
        operatorMemoryBudgetBytes: Int? = nil,
        recommendedWorkingSetBytes: Int? = nil
    ) -> SystemProfile {
        SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            hostUse: .defaultShared,
            operatorMemoryBudgetBytes: operatorMemoryBudgetBytes)
    }

    // MARK: - Regression lock: absent budget changes nothing

    @Test func dedicatedWithNilBudgetIsUnchanged() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: nil)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 115 * Self.gib)
        #expect(ceiling.source == .wiredLimit)
    }

    @Test func sharedWithNilBudgetIsUnchanged() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: nil)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 96 * Self.gib) // floor(0.75 * 128 GiB)
        #expect(ceiling.source == .sharedPolicy)
    }

    // MARK: - A binding budget reduces the envelope and reports its own source

    @Test func dedicatedWithBindingBudgetBinds() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 100 * Self.gib)
        #expect(ceiling.source == .operatorBudget)
    }

    @Test func sharedWithBindingBudgetBinds() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: 48 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 48 * Self.gib)
        #expect(ceiling.source == .operatorBudget)
    }

    // MARK: - A budget can only reduce, never raise, the ceiling

    @Test func budgetAboveCeilingDoesNotBind() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 200 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 115 * Self.gib)
        #expect(ceiling.source == .wiredLimit)
    }

    @Test func budgetExactlyEqualToCeilingDoesNotBind() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 115 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 115 * Self.gib)
        #expect(ceiling.source == .wiredLimit)
    }

    // MARK: - Non-positive budgets are ignored, identically to nil

    @Test func zeroBudgetIsIgnored() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 0)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 115 * Self.gib)
        #expect(ceiling.source == .wiredLimit)
    }

    @Test func negativeBudgetIsIgnored() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: -1)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 115 * Self.gib)
        #expect(ceiling.source == .wiredLimit)
    }

    // MARK: - The totalRAMBytes <= 0 early return is untouched by the budget

    @Test func zeroTotalRAMIgnoresPositiveBudget() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 0,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            hostUse: .defaultShared,
            operatorMemoryBudgetBytes: 1 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 0)
        #expect(ceiling.source == .physicalRAM)
    }

    // MARK: - The budget composes with, and does not bypass, the existing bounds

    @Test func budgetComposesWithRecommendedWorkingSet_smallerBudgetBinds() {
        let profile = sharedFixture(
            operatorMemoryBudgetBytes: 40 * Self.gib,
            recommendedWorkingSetBytes: 60 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 40 * Self.gib)
        #expect(ceiling.source == .operatorBudget)
    }

    @Test func budgetComposesWithRecommendedWorkingSet_largerBudgetDoesNotBind() {
        let profile = sharedFixture(
            operatorMemoryBudgetBytes: 80 * Self.gib,
            recommendedWorkingSetBytes: 60 * Self.gib)
        let ceiling = profile.effectiveMemoryCeiling
        #expect(ceiling.bytes == 60 * Self.gib)
        #expect(ceiling.source == .recommendedWorkingSet)
    }

    // MARK: - Provenance: an operator budget does not degrade the HOST observation's provenance

    /// Review finding (2026-09-06): the original implementation switched
    /// `effectiveMemoryCeilingIsMeasured` on `effectiveMemoryCeiling.source` with a
    /// `.operatorBudget -> false` arm. On this fixture (measured `iogpu.wired_limit_mb`), merely
    /// supplying a budget flipped the field from `true` to `false`, implying headroom numbers had
    /// become approximate. They had not: an operator budget is an exact figure. What the budget
    /// erases is only *which host observation would otherwise have bound* — that observation's own
    /// provenance (measured wired-limit) is unchanged by the operator's choice. This test is the
    /// inverted assertion the review required.
    @Test func operatorBudgetDoesNotDegradeHostProvenance() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)
        #expect(profile.effectiveMemoryCeilingIsMeasured == true)
    }

    @Test func sameHostWithNilBudgetStaysMeasured() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: nil)
        #expect(profile.effectiveMemoryCeilingIsMeasured == true)
    }

    /// The mirror case: on a host whose wired limit is SYNTHESIZED (not measured), a binding
    /// operator budget still reports `false`. Proves the field tracks the underlying host
    /// observation in both directions — it is not a constant `true` that merely stopped reading
    /// `.operatorBudget`.
    @Test func operatorBudgetBindingOnUnmeasuredHostStaysUnmeasured() {
        let profile = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: false,
            hostUse: .operatorAssertedDedicatedServing(),
            operatorMemoryBudgetBytes: 100 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .operatorBudget)
        #expect(profile.effectiveMemoryCeilingIsMeasured == false)
    }

    // MARK: - hostCeilingBeforeOperatorBudget: the un-budgeted envelope, alongside the budgeted one

    @Test func hostCeilingBeforeOperatorBudgetReportsTheUnbudgetedEnvelope() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)

        let before = profile.hostCeilingBeforeOperatorBudget
        #expect(before.bytes == 115 * Self.gib)
        #expect(before.source == .wiredLimit)

        let after = profile.effectiveMemoryCeiling
        #expect(after.bytes == 100 * Self.gib)
        #expect(after.source == .operatorBudget)
    }

    @Test func hostCeilingBeforeOperatorBudgetSourceIsNeverOperatorBudget() {
        let noBudget = dedicatedFixture(operatorMemoryBudgetBytes: nil)
        #expect(noBudget.hostCeilingBeforeOperatorBudget.source != .operatorBudget)

        let withBinding = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)
        #expect(withBinding.hostCeilingBeforeOperatorBudget.source != .operatorBudget)

        let sharedWithBinding = sharedFixture(operatorMemoryBudgetBytes: 1 * Self.gib)
        #expect(sharedWithBinding.hostCeilingBeforeOperatorBudget.source != .operatorBudget)
    }

    // MARK: - The budget reaches the downstream headroom math

    @Test func hardwareHoldsBytesReflectsBindingBudget() {
        let profile = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)
        let weights = 10 * Self.gib
        let reserve = 2 * Self.gib
        #expect(profile.hardwareHoldsBytes(weightsBytes: weights, osReserveBytes: reserve)
            == 100 * Self.gib - weights - reserve)
    }

    // MARK: - withOperatorMemoryBudget preserves every other field

    @Test func withOperatorMemoryBudgetPreservesOtherFieldsAndSetsBudget() {
        let base = SystemProfile(
            chip: "Apple M5 Max",
            totalRAMBytes: 128 * Self.gib,
            wiredLimitBytes: 115 * Self.gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 60 * Self.gib,
            hostUse: .operatorAssertedDedicatedServing())
        let updated = base.withOperatorMemoryBudget(42 * Self.gib)

        #expect(updated.chip == base.chip)
        #expect(updated.totalRAMBytes == base.totalRAMBytes)
        #expect(updated.wiredLimitBytes == base.wiredLimitBytes)
        #expect(updated.wiredLimitIsMeasured == base.wiredLimitIsMeasured)
        #expect(updated.recommendedWorkingSetBytes == base.recommendedWorkingSetBytes)
        #expect(updated.hostUse == base.hostUse)
        #expect(updated.operatorMemoryBudgetBytes == 42 * Self.gib)
    }

    @Test func withOperatorMemoryBudgetNilClearsBudget() {
        let base = dedicatedFixture(operatorMemoryBudgetBytes: 100 * Self.gib)
        let cleared = base.withOperatorMemoryBudget(nil)
        #expect(cleared.operatorMemoryBudgetBytes == nil)
        // clearing it restores the unbudgeted envelope
        #expect(cleared.effectiveMemoryCeiling.bytes == 115 * Self.gib)
        #expect(cleared.effectiveMemoryCeiling.source == .wiredLimit)
    }

    // MARK: - ModelSizer.provenanceNotes: a binding budget layers ON TOP OF the host's own note

    /// `sharedFixture` synthesizes its ceiling via shared policy (128 GiB RAM -> floor(75%) = 96
    /// GiB, tighter than the 115 GiB wired observation), so `hostCeilingBeforeOperatorBudget` is
    /// `.sharedPolicy` regardless of the operator budget. A 40 GiB budget binds tighter still. The
    /// operator must get BOTH facts: the budget is exact-but-not-hardware, AND the host ceiling it
    /// undercut was itself synthesized/advisory — not just the first one.
    @Test func provenanceNotesEmitsBothBudgetAndUnderlyingSharedPolicyNoteWhenBudgetBinds() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: 40 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .operatorBudget)
        #expect(profile.hostCeilingBeforeOperatorBudget.source == .sharedPolicy)

        let notes = ModelSizer.provenanceNotes(box: profile, kvQuant: .fp16)
        #expect(notes.count == 2)
        #expect(notes[0].contains("operator-supplied budget"))
        #expect(notes[1].contains("synthesized by shared policy"))
    }

    @Test func provenanceNotesEmitsOnlySharedPolicyNoteWithNoBudget() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: nil)
        #expect(profile.effectiveMemoryCeiling.source == .sharedPolicy)

        let notes = ModelSizer.provenanceNotes(box: profile, kvQuant: .fp16)
        #expect(notes.count == 1)
        #expect(notes[0].contains("synthesized by shared policy"))
    }

    /// Item 7's other half: a budget was SUPPLIED but did NOT bind (it is >= the host's own
    /// ceiling). This must not regress to a silent discard — the note must name BOTH the ceiling
    /// that won (96 GiB, shared policy) and the inert budget figure (200 GiB), so an operator who
    /// supplied a generous budget can see it had no effect rather than wondering why nothing changed.
    @Test func provenanceNotesEmitsNonBindingBudgetNoteWhenBudgetDoesNotBind() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: 200 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .sharedPolicy, "precondition: budget must not bind")
        #expect(profile.hostCeilingBeforeOperatorBudget.bytes == 96 * Self.gib)

        let notes = ModelSizer.provenanceNotes(box: profile, kvQuant: .fp16)
        #expect(notes.count == 2)
        #expect(notes[0].contains("did not bind"))
        #expect(notes[0].contains("\(200 * Self.gib)"), "must name the inert budget figure")
        #expect(notes[0].contains("\(96 * Self.gib)"), "must name the ceiling that won")
        #expect(notes[1].contains("synthesized by shared policy"), "the host's own note must still appear underneath")
    }

    /// The exactly-equal case (the production-skew safety property): a budget equal to the host
    /// ceiling is inert by the strict `<` rule, and must render as the NON-binding note, not the
    /// binding one — a regression that relaxed the comparison to `<=` would flip which note fires.
    @Test func provenanceNotesTreatsExactlyEqualBudgetAsNonBinding() {
        let profile = sharedFixture(operatorMemoryBudgetBytes: 96 * Self.gib)
        #expect(profile.effectiveMemoryCeiling.source == .sharedPolicy)

        let notes = ModelSizer.provenanceNotes(box: profile, kvQuant: .fp16)
        #expect(notes.count == 2)
        #expect(notes[0].contains("did not bind"))
        #expect(!notes[0].contains("the effective memory ceiling is an operator-supplied budget"))
    }
}
