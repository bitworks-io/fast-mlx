import XCTest
@testable import HarnessCore

/// Contract-lock tests for the two fit-check surfaces the next slices will edit — the `--tier`
/// dial (`docs/task-inbox/2026-08-18-serve-tier-dial.md`) and the human-gated concurrency-KV
/// advisory line (`…-fit-check-concurrency-kv-undercount.md`) both modify the announce path.
///
/// They pin two things the shipped fit-check relied on but did not lock:
///  1. the machine-readable startup line's key set + order + anchors (gate scripts parse it);
///  2. the `contextCeiling == 0 ⇒ red, fail-closed` invariant `decide` assumes (an unservable
///     model must be refused, never served under a fabricated fit — the moat's honesty spine).
///
/// Pure-Swift; no live model. Assertions are made against the rendered string / typed decision
/// fields directly — deliberately NOT a parse-back that re-derives semantics (a parser mirroring a
/// bug would pass green while the code is wrong).
final class ServingFitAnnounceContractTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    private func profile(_ id: String) -> ModelArchProfile {
        guard let m = ModelArchProfile.catalog.first(where: { $0.id == id }) else {
            fatalError("missing catalog entry \(id)")
        }
        return m
    }

    private func smallHost(ramGiB: Int, wiredGiB: Int) -> SystemProfile {
        SystemProfile(chip: "test", totalRAMBytes: ramGiB * gib, wiredLimitBytes: wiredGiB * gib, wiredLimitIsMeasured: true)
    }

    /// Ordered keys of a `key=value ...` line — used only to assert the line's SHAPE (which keys,
    /// in what order), never to recompute any value.
    private func keys(_ fields: String) -> [String] {
        fields.split(separator: " ").map { String($0.split(separator: "=").first ?? "") }
    }

    // MARK: - machine-readable line: frozen key set/order + anchors

    /// The green/unforced line is the frozen base contract gate scripts anchor on: `fit_check`
    /// first, this exact key set/order, no `fit_forced`, and never `listening=` (the serve path
    /// splices that AFTER, and it must remain the last token). Adding/removing/reordering a field
    /// breaks this test on purpose — that is a wire-format change reviewers must see.
    func testMachineLine_greenUnforced_frozenKeyContract() {
        let d = ServingFitPlanner.decide(profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128)
        let fields = d.machineReadableFields()
        XCTAssertEqual(keys(fields), [
            "fit_check", "fit_binding", "weights_measured", "wired_limit_measured",
            "fit_estimate_measured", "fit_quant_bits", "fit_served_context",
            "fit_context_ceiling", "fit_context_capped",
        ], "machine-line key set/order is a frozen contract")
        XCTAssertTrue(fields.hasPrefix("fit_check="), "fit_check must be the first token")
        XCTAssertFalse(fields.contains("fit_forced"), "an unforced verdict must not emit fit_forced")
        XCTAssertFalse(fields.contains("listening="), "listening= is appended by the serve path, never inside the fit fields")
    }

    /// `fit_forced` is the ONLY additional field a forced red may append, and it appends LAST so the
    /// base contract above is untouched (a gate script reading the fixed prefix keeps working).
    func testMachineLine_redUnderForce_appendsForcedLast_baseContractIntact() {
        let d = ServingFitPlanner.decide(
            profile: profile("GLM-4.5-Air"), weightsAreMeasured: true,
            host: smallHost(ramGiB: 8, wiredGiB: 6), force: true)
        let fields = d.machineReadableFields()
        XCTAssertTrue(fields.contains("fit_check=red"))
        XCTAssertEqual(keys(fields), [
            "fit_check", "fit_binding", "weights_measured", "wired_limit_measured",
            "fit_estimate_measured", "fit_quant_bits", "fit_served_context",
            "fit_context_ceiling", "fit_context_capped", "fit_forced",
        ], "forced line = the frozen base contract + fit_forced appended last")
        XCTAssertFalse(fields.contains("listening="))
    }

    /// The opt-in `--plan-concurrency` field extends the frozen line additively: when a planning
    /// concurrency >1 is supplied and the verdict proceeds, `fit_plan_concurrency` appends AFTER the
    /// frozen base contract and BEFORE `fit_forced` — so both the base prefix and the `fit_forced`-last
    /// invariant a gate script relies on stay intact. At the default concurrency the field is absent
    /// (locked by the frozen-contract test above, which runs at concurrency 1).
    func testMachineLine_planConcurrency_appendsAfterBaseContract() {
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128, concurrency: 4)
        XCTAssertTrue(d.shouldProceed, "precondition: 32B on 128 GiB proceeds at 4 streams")
        XCTAssertEqual(keys(d.machineReadableFields()), [
            "fit_check", "fit_binding", "weights_measured", "wired_limit_measured",
            "fit_estimate_measured", "fit_quant_bits", "fit_served_context",
            "fit_context_ceiling", "fit_context_capped", "fit_plan_concurrency",
        ], "fit_plan_concurrency appends after the frozen base contract")
    }

    /// Forced red + a planning concurrency: the order is base contract, then `fit_plan_concurrency`,
    /// then `fit_forced` LAST — the two additive fields compose without disturbing `fit_forced`'s
    /// tail position.
    func testMachineLine_planConcurrencyAndForce_forcedStaysLast() {
        let d = ServingFitPlanner.decide(
            profile: profile("GLM-4.5-Air"), weightsAreMeasured: true,
            host: smallHost(ramGiB: 8, wiredGiB: 6), concurrency: 4, force: true)
        XCTAssertEqual(keys(d.machineReadableFields()), [
            "fit_check", "fit_binding", "weights_measured", "wired_limit_measured",
            "fit_estimate_measured", "fit_quant_bits", "fit_served_context",
            "fit_context_ceiling", "fit_context_capped", "fit_plan_concurrency", "fit_forced",
        ], "fit_forced remains the last token even with fit_plan_concurrency present")
    }

    // MARK: - contextCeiling == 0 ⇒ red, fail-closed (invariant now pinned across every cause)

    /// Every distinct cause of `contextCeiling == 0` must drive `decide` to a red, fail-closed
    /// verdict. `contextCeiling` returns 0 exactly when `classify(predict@context=1)` is red, and
    /// `decide` then assesses the (≥1) requested context and *relies* on that also being red. The
    /// causes are all either context-independent (weights-don't-fit, KV-not-derivable) or
    /// monotonic-in-context (the memory-bound ratio path — see the monotonicity lock below), so the
    /// reliance holds — this test locks it so a future announce-path edit can't quietly serve an
    /// unservable model. If any case fails to go red, that is a real defect in shipped code.
    ///
    /// The four causes cover both context-independent reds (first three) AND the memory-bound ratio
    /// path (`Phi-4` weights fit with ~2 GiB headroom, but even context=1's non-weights peak —
    /// dominated by the 2 GiB allocator floor — exceeds `yellowMax`): that last is the ONLY cause
    /// where the "monotonic in context" argument does any work, so it is the one that matters most.
    func testCeilingZero_alwaysRedAndFailsClosed() {
        let cases: [(model: String, host: SystemProfile, why: String)] = [
            ("GLM-4.5-Air", smallHost(ramGiB: 8, wiredGiB: 6), "weights alone exceed the box (headroom ≤ 0)"),
            ("DeepSeek-V4-Flash", .m5Max128, "novel-compressed arch: KV not derivable even on a huge box"),
            ("Nemotron-3-Ultra", .m5Max128, "hybrid mamba2/MoE, nAttnLayers==0: KV not derivable"),
            ("Phi-4-14B", smallHost(ramGiB: 14, wiredGiB: 13), "memory-bound ratio: weights fit but context=1 peak > yellowMax"),
        ]
        for c in cases {
            let d = ServingFitPlanner.decide(profile: profile(c.model), weightsAreMeasured: true, host: c.host)
            XCTAssertEqual(d.contextCeiling, 0, "precondition: nothing fits at any context — \(c.why)")
            XCTAssertEqual(d.color, .red, "ceiling 0 must classify red — \(c.why)")
            XCTAssertFalse(d.shouldProceed, "ceiling 0 must fail closed without --force — \(c.why)")
        }
    }

    /// The memory-bound ratio path, asserted specifically: weights genuinely FIT (headroom > 0 — so
    /// this is NOT the weights-don't-fit branch) and KV IS derivable (NOT the kvNotDerivable branch),
    /// yet context=1 is still red, so the binding constraint is a RAM/wired-limit one. This is the
    /// case the parametrized test's "monotonic in context" reliance actually depends on.
    func testCeilingZero_memoryBoundRatioPath_weightsFitButContext1Red() {
        let host = smallHost(ramGiB: 14, wiredGiB: 13) // min(wired,RAM)=13 GiB; Phi-4 weights ≈ 7 GiB
        let headroom = host.hardwareHoldsBytes(weightsBytes: 7 * gib, osReserveBytes: 4 * gib)
        XCTAssertGreaterThan(headroom, 0, "precondition: weights fit — this must be the ratio path, not headroom ≤ 0")
        let d = ServingFitPlanner.decide(profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host)
        XCTAssertEqual(d.contextCeiling, 0)
        XCTAssertEqual(d.color, .red)
        XCTAssertFalse(d.shouldProceed)
        XCTAssertTrue([.wiredLimit, .physicalRAM].contains(d.bindingConstraint),
            "a memory-bound ratio red names a RAM/wired constraint, not kvNotDerivable")
    }

    /// Locks the assumption the ceiling-0 ratio-path reliance rests on: the predicted peak is
    /// non-decreasing in context, because the transient-prefill term is fixed (it scales with the
    /// prefill CHUNK, not the served context) while KV grows linearly. Read directly off
    /// `predictPeakBytes` so a future edit that made transient context-scaling — quietly breaking
    /// `red@1 ⇒ red@nativeMax` — fails here.
    func testPredictedPeak_isMonotonicInContext_transientFixed_kvGrows() {
        let m = profile("Phi-4-14B")
        let host = SystemProfile.m5Max128
        let lo = CapacityModel.predictPeakBytes(model: m, context: 1, concurrency: 1, kvQuant: .fp16, profile: host)
        let hi = CapacityModel.predictPeakBytes(model: m, context: 8192, concurrency: 1, kvQuant: .fp16, profile: host)
        XCTAssertEqual(lo.transientPrefillPeakBytes, hi.transientPrefillPeakBytes,
            "transient scales with the prefill chunk, not the served context — must be context-independent")
        XCTAssertGreaterThan(hi.kvBytes, lo.kvBytes, "KV grows with context")
        XCTAssertGreaterThan(hi.totalBytes, lo.totalBytes, "so peak is strictly increasing in context (red@1 ⇒ red above)")
    }

    /// The KV-not-derivable ceiling-0 case is refused with the honest binding constraint (spec
    /// §2.1/§8) — the fit-check says "I can't derive this" rather than fabricating a fit.
    func testCeilingZero_kvNotDerivable_bindingIsHonest() {
        let d = ServingFitPlanner.decide(profile: profile("DeepSeek-V4-Flash"), weightsAreMeasured: true, host: .m5Max128)
        XCTAssertEqual(d.bindingConstraint, .kvNotDerivable)
        XCTAssertFalse(d.shouldProceed)
    }

    /// `--force` is the only escape from a ceiling-0 red: it proceeds, but the verdict stays red and
    /// is flagged as forced (the announce never launders a forced serve into a clean one).
    func testCeilingZero_forceProceedsButStaysRedAndFlagged() {
        let d = ServingFitPlanner.decide(
            profile: profile("DeepSeek-V4-Flash"), weightsAreMeasured: true, host: .m5Max128, force: true)
        XCTAssertEqual(d.contextCeiling, 0)
        XCTAssertEqual(d.color, .red)
        XCTAssertTrue(d.shouldProceed, "--force proceeds past a ceiling-0 red")
        XCTAssertTrue(d.proceedingUnderForce)
        XCTAssertTrue(d.machineReadableFields().contains("fit_forced=true"))
    }

    // MARK: - concurrency-aware advisory KV line (…-fit-check-concurrency-kv-undercount.md, option 1)

    /// When a planning slot-count is supplied, the summary gains ONE advisory line reporting the
    /// modeled KV footprint at N concurrent slots — so the operator sees the concurrent-decode
    /// headroom the single-slot verdict omits, without the verdict itself failing closed on a
    /// modeled multiple (the exact regression `faf1f35` narrowed away). The line is stderr-summary
    /// only; it labels itself modeled+advisory so it is never read as a measured cap.
    func testConcurrencyAdvisory_present_rendersModeledSlotLine() {
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128, advisorySlotCount: 4)
        let advisory = d.summaryLines().first { $0.contains("x4 slots") }
        XCTAssertNotNil(advisory, "an advisory slot-count line must render when advisorySlotCount is set")
        XCTAssertTrue(advisory!.contains("kv@\(d.servedContext)"), "advisory names the served context it modeled")
        XCTAssertTrue(advisory!.lowercased().contains("modeled"), "advisory must label itself modeled")
        XCTAssertTrue(advisory!.lowercased().contains("advisory"), "advisory must label itself advisory (not a cap)")
        // The advisory KV at N>1 slots is strictly larger than the single-slot reserved-KV estimate.
        XCTAssertNotNil(d.kvAtSlotsBytes)
        XCTAssertGreaterThan(d.kvAtSlotsBytes!, d.maxReservedKVBytes,
            "concurrent KV must exceed the single-slot estimate for a model with nonzero KV")
    }

    /// Absent an advisory slot-count (the default), NO slot line renders — the shipped summary is
    /// byte-for-byte unchanged for existing callers.
    func testConcurrencyAdvisory_absent_noSlotLine() {
        let d = ServingFitPlanner.decide(profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128)
        XCTAssertNil(d.kvAtSlotsBytes)
        XCTAssertFalse(d.summaryLines().contains { $0.contains("slots") },
            "no slot advisory line without an explicit advisorySlotCount")
    }

    /// The invariant that makes this option 1 and NOT the forbidden hot-patch: supplying an advisory
    /// slot-count changes ONLY the added advisory line — every verdict/limit/context field, and the
    /// entire frozen machine-readable line, are identical with and without it. If a future edit lets
    /// the advisory feed the verdict math, this fails.
    func testConcurrencyAdvisory_doesNotAlterVerdictOrLimits() {
        let base = ServingFitPlanner.decide(profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128)
        let adv = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: .m5Max128, advisorySlotCount: 4)
        XCTAssertEqual(adv.color, base.color)
        XCTAssertEqual(adv.shouldProceed, base.shouldProceed)
        XCTAssertEqual(adv.servedContext, base.servedContext)
        XCTAssertEqual(adv.contextCeiling, base.contextCeiling)
        XCTAssertEqual(adv.memoryLimitBytes, base.memoryLimitBytes)
        XCTAssertEqual(adv.cacheLimitBytes, base.cacheLimitBytes)
        XCTAssertEqual(adv.maxReservedKVBytes, base.maxReservedKVBytes)
        XCTAssertEqual(adv.machineReadableFields(), base.machineReadableFields(),
            "the advisory is stderr-summary only — the frozen machine line must not change")
    }
}
