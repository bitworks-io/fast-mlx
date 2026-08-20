import XCTest
@testable import HarnessCore

/// TDD for the quant auto-picker (fit-checked-serve, differentiator #2 full shape): given several
/// quantizations of the same model + a host, pick the one that serves the most of the requested
/// context at the highest quality that still fits, or refuse when none does. Pure — every candidate
/// is fit-checked through `ServingFitPlanner.decide`, so the whole selection policy is exercised
/// off-box and the live serve only confirms the winner loads.
///
/// Comparison policy under test (resolved from the ambiguity the sizer left open — see
/// docs/task-inbox/2026-08-18-quant-auto-pick-policy.md): among non-red candidates, rank by
/// (servedContext desc, color green>yellow, quant bits desc [nil=unquantized ranks highest],
/// repoID asc). Context served is primary because it is the operator's explicit ask (default =
/// native max); silently shipping less context to gain quality bits is the worse surprise.
final class QuantAutoPickerTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    private func host(ramGiB: Int, wiredGiB: Int) -> SystemProfile {
        SystemProfile(chip: "test", totalRAMBytes: ramGiB * gib, wiredLimitBytes: wiredGiB * gib, wiredLimitIsMeasured: true)
    }

    /// A synthetic quant variant: uniform-GQA geometry fixed, only the declared bits + weight bytes
    /// vary (an 8-bit build is ~2× the 4-bit bytes). nLayers/heads chosen so KV@40960 ≈ 5.4 GiB, big
    /// enough that headroom decides context capping.
    private func candidate(_ repoID: String, bits: Int?, weightsGiB: Double) -> QuantServeCandidate {
        let profile = ModelArchProfile(
            id: repoID, modelType: .uniformGQA, nLayers: 32, nAttnLayers: 32,
            nKVHeads: 8, headDim: 128, fixedStateBytes: 0, nativeMaxContext: 40960,
            weightsBytes4bitEstimate: Int(weightsGiB * Double(gib)), license: "test")
        return QuantServeCandidate(
            repoID: repoID, parsed: ParsedModelArch(profile: profile, weightsAreMeasured: true, quantBits: bits))
    }

    // MARK: - 8bit red + 4bit green → pick 4bit

    func testEightBitRed_fourBitGreen_picksFourBit() {
        let h = host(ramGiB: 40, wiredGiB: 34)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-8bit", bits: 8, weightsGiB: 32),
                         candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        XCTAssertEqual(result.winnerRepoID, "repo-4bit")
        // the 8bit candidate was evaluated and found red (excluded), not silently dropped
        let eightBit = result.evaluations.first { $0.repoID == "repo-8bit" }
        XCTAssertEqual(eightBit?.decision?.color, .red)
        XCTAssertFalse(eightBit?.eligible ?? true)
    }

    // MARK: - both green at full context → pick 8bit (quality tiebreak)

    func testBothGreenFullContext_picksHigherBits() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16),
                         candidate("repo-8bit", bits: 8, weightsGiB: 32)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        XCTAssertEqual(result.winnerRepoID, "repo-8bit", "both serve full context → higher bits wins on quality")
        XCTAssertEqual(result.winnerDecision?.servedContext, 40960)
        XCTAssertFalse(result.winnerDecision?.contextWasCapped ?? true)
    }

    // MARK: - each candidate's decision carries its own quant bits (so winnerDecision self-labels)

    func testCandidateDecisionsCarryTheirQuantBits() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16),
                         candidate("repo-8bit", bits: 8, weightsGiB: 32)],
            host: h)
        // the picker must thread each candidate's bits into its ServingFitDecision, not leave nil —
        // otherwise any consumer of winnerDecision.summaryLines() silently loses the [N-bit] label.
        let fourBit = result.evaluations.first { $0.repoID == "repo-4bit" }
        let eightBit = result.evaluations.first { $0.repoID == "repo-8bit" }
        XCTAssertEqual(fourBit?.decision?.quantBits, 4)
        XCTAssertEqual(eightBit?.decision?.quantBits, 8)
        XCTAssertEqual(result.winnerDecision?.quantBits, 8)
        XCTAssertTrue(
            result.winnerDecision?.summaryLines().first?.contains("[8-bit]") ?? false,
            "winnerDecision header should self-label its quant bits")
    }

    // MARK: - equal bits, both green full context → DWQ build outranks the plain build

    func testEqualBits_dwqOutranksPlain() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        // Same 4-bit width, same weight bytes, both serve full native context: every ranking axis ties
        // until the DWQ tiebreak. A DWQ build recovers accuracy the naive quant loses (mlx-lm
        // LEARNED_QUANTS.md), so at equal bits it must win — not lose to the plain build on repoID asc
        // ("repo-4bit" < "repo-4bit-DWQ" would otherwise hand the tie to the lower-fidelity build).
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16),
                         candidate("repo-4bit-DWQ", bits: 4, weightsGiB: 16)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        XCTAssertEqual(result.winnerRepoID, "repo-4bit-DWQ",
                       "at equal bits + equal fit, the DWQ build outranks the plain build")
    }

    // MARK: - DWQ detection is scoped to the checkpoint name, not a parent directory in the path

    func testEqualBits_dwqTiebreakIgnoresParentPath() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        // The resolver identifies a candidate by its absolute directory path. A DWQ-looking PARENT dir
        // must NOT make a plain build read as DWQ — only the last path component (the checkpoint name)
        // decides. Both names are plain "repo-4bit", so the tie must fall to repoID asc, and
        // "/models/aaa/…" sorts before "/models/dwq/…" → the aaa candidate wins. If detection wrongly
        // matched the parent "dwq", the dwq-parent candidate would win the DWQ tiebreak — this pins it.
        let result = QuantAutoPicker.pick(
            candidates: [candidate("/models/dwq/repo-4bit", bits: 4, weightsGiB: 16),
                         candidate("/models/aaa/repo-4bit", bits: 4, weightsGiB: 16)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        XCTAssertEqual(result.winnerRepoID, "/models/aaa/repo-4bit",
                       "neither NAME is DWQ → repoID asc decides; a parent 'dwq' dir must not flip it")
    }

    // MARK: - a plain higher-bit build still beats a lower-bit DWQ (DWQ only breaks equal-bit ties)

    func testHigherBitsPlain_beatsLowerBitDWQ() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit-DWQ", bits: 4, weightsGiB: 16),
                         candidate("repo-8bit", bits: 8, weightsGiB: 32)],
            host: h)
        // Both serve full context; bits (8 > 4) decide before the DWQ tiebreak is ever reached — the
        // DWQ flag must not resurrect the lower-bit build. DWQ breaks ties only among EQUAL-bit builds.
        XCTAssertEqual(result.winnerRepoID, "repo-8bit")
    }

    // MARK: - 8bit only fits by capping, 4bit serves full → pick 4bit (more context)

    func testHigherBitsCapsContext_lowerBitsFull_picksMoreContext() {
        let h = host(ramGiB: 48, wiredGiB: 40)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-8bit", bits: 8, weightsGiB: 28),   // fits only at a capped context
                         candidate("repo-4bit", bits: 4, weightsGiB: 16)],  // fits at full native max
            host: h)
        XCTAssertTrue(result.shouldProceed)
        // 8bit is eligible but only by capping; 4bit serves the full ask → 4bit wins on served context
        let eight = result.evaluations.first { $0.repoID == "repo-8bit" }
        XCTAssertTrue(eight?.eligible ?? false, "8bit still fits (capped), just not at full context")
        XCTAssertTrue(eight?.decision?.contextWasCapped ?? false)
        XCTAssertEqual(result.winnerRepoID, "repo-4bit")
        XCTAssertEqual(result.winnerDecision?.servedContext, 40960)
        XCTAssertGreaterThan(result.winnerDecision!.servedContext, eight!.decision!.servedContext)
    }

    // MARK: - nothing fits → refuse with per-candidate reasons

    func testAllRed_refuses_withPerCandidateReasons() {
        let h = host(ramGiB: 8, wiredGiB: 6)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-8bit", bits: 8, weightsGiB: 18),
                         candidate("repo-4bit", bits: 4, weightsGiB: 9)],
            host: h)
        XCTAssertFalse(result.shouldProceed)
        XCTAssertNil(result.winnerRepoID)
        XCTAssertTrue(result.evaluations.allSatisfy { $0.decision?.color == .red })
        let lines = result.summaryLines().joined(separator: "\n")
        XCTAssertTrue(lines.contains("repo-8bit"))
        XCTAssertTrue(lines.contains("repo-4bit"))
        XCTAssertTrue(lines.lowercased().contains("no quant"))
    }

    // MARK: - an unmodeled-arch candidate is excluded with a reason, not fatal

    func testUnmodeledCandidateExcluded_notFatal() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let excluded = QuantServeCandidate(
            repoID: "repo-future", parsed: nil,
            exclusionReason: "model_type 'some_future_arch' is not fit-checkable yet")
        let result = QuantAutoPicker.pick(
            candidates: [excluded, candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        XCTAssertEqual(result.winnerRepoID, "repo-4bit")
        let ex = result.evaluations.first { $0.repoID == "repo-future" }
        XCTAssertNotNil(ex)
        XCTAssertNil(ex?.decision, "excluded before the fit-check ran")
        XCTAssertFalse(ex?.eligible ?? true)
        XCTAssertEqual(ex?.exclusionReason, "model_type 'some_future_arch' is not fit-checkable yet")
    }

    // MARK: - unquantized (nil bits) ranks as highest quality on an otherwise-equal tie

    func testUnquantizedRanksAboveEightBit_whenContextAndColorEqual() {
        let h = host(ramGiB: 256, wiredGiB: 192)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-8bit", bits: 8, weightsGiB: 32),
                         candidate("repo-bf16", bits: nil, weightsGiB: 60)],
            host: h)
        XCTAssertTrue(result.shouldProceed)
        // both green at full native context → nil (unquantized) is highest quality → wins
        XCTAssertEqual(result.winnerRepoID, "repo-bf16")
    }

    // MARK: - machine-readable winner line (--quant-pick-only scriptable primitive)

    /// The pick-only winner line is a machine contract (a script does `WINNER=$(… --quant-pick-only)`),
    /// so pin its exact key set/order: `quant_pick winner=… quant_bits=… fit_check=… fit_served_context=…`.
    /// Adding/reordering a field breaks this on purpose — that is a wire-format change to review.
    func testMachineReadableWinnerLine_greenWinner_frozenKeyContract() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16),
                         candidate("repo-8bit", bits: 8, weightsGiB: 32)],
            host: h)
        let line = result.machineReadableWinnerLine()
        XCTAssertEqual(line,
            "quant_pick winner=repo-8bit quant_bits=8 fit_check=green fit_served_context=40960",
            "winner line is a frozen machine contract")
    }

    /// An unquantized winner reports `quant_bits=none` (never a fabricated number).
    func testMachineReadableWinnerLine_unquantized_reportsNone() {
        let h = host(ramGiB: 256, wiredGiB: 192)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-bf16", bits: nil, weightsGiB: 60)],
            host: h)
        XCTAssertEqual(result.machineReadableWinnerLine(),
            "quant_pick winner=repo-bf16 quant_bits=none fit_check=green fit_served_context=40960")
    }

    // MARK: - joint weight-quant × KV-tier escalation (serve-dial full shape)

    /// The band host: fp16 KV cannot hold the full native context (caps below it), but the measured
    /// int8 tier halves KV and serves the full ask. Weights fit at both tiers. See the byte arithmetic
    /// in the test body's construction (H−W ≈ 9 GiB puts fp16@native red, int8@native yellow).
    private func escalationBandHost() -> SystemProfile { host(ramGiB: 40, wiredGiB: 29) }

    /// With only fp16 allowed, the picker caps the served context below the model's native max.
    func testFp16Only_capsBelowNativeMax() {
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: escalationBandHost(), allowedKVTiers: [.fp16])
        XCTAssertTrue(result.shouldProceed)
        let e = result.evaluations.first { $0.repoID == "repo-4bit" }
        XCTAssertEqual(e?.chosenKVTier, .fp16)
        XCTAssertTrue(e?.decision?.contextWasCapped ?? false, "fp16 can't hold full native context on this box")
        XCTAssertLessThan(e!.decision!.servedContext, 40960)
    }

    /// Allowing int8 lets the picker escalate to buy back the full context — chosenKVTier flips to
    /// int8 and servedContext rises to the native max the fp16-only run had to cap.
    func testEscalatesToInt8_toServeMoreContext() {
        let fp16Only = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: escalationBandHost(), allowedKVTiers: [.fp16])
        let escalated = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: escalationBandHost(), allowedKVTiers: [.fp16, .int8])
        let e = escalated.evaluations.first { $0.repoID == "repo-4bit" }
        XCTAssertEqual(e?.chosenKVTier, .int8, "escalated off fp16 to fit more context")
        XCTAssertEqual(e?.decision?.servedContext, 40960, "int8 serves the full native max here")
        XCTAssertGreaterThan(
            escalated.winnerDecision!.servedContext,
            fp16Only.winnerDecision!.servedContext,
            "escalation strictly increased served context")
    }

    /// The escalated winner names its tier in the machine line (additive field, after the frozen base).
    func testMachineLine_escalatedWinner_carriesKVTier() {
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: escalationBandHost(), allowedKVTiers: [.fp16, .int8])
        let line = result.machineReadableWinnerLine()
        XCTAssertNotNil(line)
        XCTAssertTrue(line!.hasPrefix("quant_pick winner=repo-4bit quant_bits=4 fit_check="),
            "frozen base contract prefix is preserved")
        XCTAssertTrue(line!.contains("kv_tier=int8"), "escalated tier is appended after the frozen base")
    }

    /// When fp16 already serves the full context, escalation keeps fp16 (higher fidelity wins the
    /// same-context tie) and appends NO kv_tier field — the pre-escalation line stays byte-identical.
    func testFp16KeptWhenItAlreadyServesFull_noKVTierField() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: h, allowedKVTiers: [.fp16, .int8])
        let e = result.evaluations.first { $0.repoID == "repo-4bit" }
        XCTAssertEqual(e?.chosenKVTier, .fp16, "fp16 already serves full context → not downgraded")
        XCTAssertEqual(result.machineReadableWinnerLine(),
            "quant_pick winner=repo-4bit quant_bits=4 fit_check=green fit_served_context=40960")
    }

    /// balanced tier (escalate KV, but no context cap): on the band host, fp16 would only fit by
    /// capping — forbidden — so the picker escalates to int8 which serves the FULL context, and
    /// proceeds. This is balanced's whole value: trade KV precision to avoid shortening context.
    func testBalancedStance_escalatesToInt8_toAvoidCapping() {
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: escalationBandHost(), allowedKVTiers: [.fp16, .int8], allowContextCapping: false)
        XCTAssertTrue(result.shouldProceed)
        let e = result.evaluations.first { $0.repoID == "repo-4bit" }
        XCTAssertEqual(e?.chosenKVTier, .int8, "escalated to hold full context without capping")
        XCTAssertEqual(e?.decision?.servedContext, 40960)
        XCTAssertFalse(e?.decision?.contextWasCapped ?? true)
    }

    /// When even int8 cannot hold the full context, the two stances diverge: balanced (no cap)
    /// REFUSES, while maxfit (cap) serves the int8-capped context. This is the behavioral distinction
    /// the dial promises.
    func testBalancedRefusesWhereMaxfitCaps_whenInt8AlsoOverflows() {
        // H−W ≈ 6 GiB: both fp16 and int8 overflow at full native context, so a fit needs capping.
        let tightHost = host(ramGiB: 40, wiredGiB: 26)
        let balanced = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: tightHost, allowedKVTiers: [.fp16, .int8], allowContextCapping: false)
        XCTAssertFalse(balanced.shouldProceed, "balanced refuses rather than shortening context")

        let maxfit = QuantAutoPicker.pick(
            candidates: [candidate("repo-4bit", bits: 4, weightsGiB: 16)],
            host: tightHost, allowedKVTiers: [.fp16, .int8], allowContextCapping: true)
        XCTAssertTrue(maxfit.shouldProceed, "maxfit caps context to fit")
        let e = maxfit.evaluations.first { $0.repoID == "repo-4bit" }
        XCTAssertTrue(e?.decision?.contextWasCapped ?? false)
        XCTAssertLessThan(e!.decision!.servedContext, 40960)
    }

    /// A red-only set has no winner → no winner line (the caller fails closed via the refusal path,
    /// exit 2, exactly like the serve path — never a bogus pick).
    func testMachineReadableWinnerLine_allRed_returnsNil() {
        let h = host(ramGiB: 8, wiredGiB: 6)
        let result = QuantAutoPicker.pick(
            candidates: [candidate("repo-8bit", bits: 8, weightsGiB: 18),
                         candidate("repo-4bit", bits: 4, weightsGiB: 9)],
            host: h)
        XCTAssertNil(result.machineReadableWinnerLine())
    }

    // MARK: - --prefer quality|context: which axis is primary in the cross-candidate ranking

    /// The SAME host+candidates as testHigherBitsCapsContext_lowerBitsFull_picksMoreContext (host
    /// 48/40, the 8bit build fits only by capping context, the 4bit serves the full native max). The
    /// default (.context) picks the 4bit for more context; `.quality` hoists quant bits to the primary
    /// key and picks the 8bit even though it serves LESS context — the operator asked for fidelity over
    /// length. Both stay eligible: this is a ranking-preference switch, not a fit-check change.
    func testPreferQuality_hoistsBitsOverContext_flipsWinner() {
        let h = host(ramGiB: 48, wiredGiB: 40)
        let cands = [candidate("repo-8bit", bits: 8, weightsGiB: 28),
                     candidate("repo-4bit", bits: 4, weightsGiB: 16)]
        let contextPick = QuantAutoPicker.pick(candidates: cands, host: h) // default .context
        XCTAssertEqual(contextPick.winnerRepoID, "repo-4bit")
        let qualityPick = QuantAutoPicker.pick(candidates: cands, host: h, preference: .quality)
        XCTAssertEqual(qualityPick.winnerRepoID, "repo-8bit", "quality preference picks the higher-bit build")
        XCTAssertTrue(qualityPick.winnerDecision?.contextWasCapped ?? false,
                      "the quality winner serves less context — the documented tradeoff")
        XCTAssertLessThan(qualityPick.winnerDecision!.servedContext, contextPick.winnerDecision!.servedContext)
    }

    /// `.quality` must not disturb the existing quality tiebreak at equal context: when both fit at the
    /// full native max, the 8bit already wins on the default's quality-bits tiebreak, and `.quality`
    /// agrees — the switch changes the primary axis, not the outcome when context is not contested.
    func testPreferQuality_atEqualContext_agreesWithDefault() {
        let h = host(ramGiB: 128, wiredGiB: 115)
        let cands = [candidate("repo-4bit", bits: 4, weightsGiB: 16),
                     candidate("repo-8bit", bits: 8, weightsGiB: 32)]
        XCTAssertEqual(QuantAutoPicker.pick(candidates: cands, host: h).winnerRepoID, "repo-8bit")
        XCTAssertEqual(
            QuantAutoPicker.pick(candidates: cands, host: h, preference: .quality).winnerRepoID, "repo-8bit")
    }

    /// The machine-readable winner line gains `prefer=quality` ONLY when the non-default preference is
    /// used — the frozen default line stays byte-identical (additive-field contract, mirroring kv_tier).
    func testMachineLine_preferMarkerOnlyWhenNonDefault() {
        let h = host(ramGiB: 48, wiredGiB: 40)
        let cands = [candidate("repo-8bit", bits: 8, weightsGiB: 28),
                     candidate("repo-4bit", bits: 4, weightsGiB: 16)]
        let dflt = QuantAutoPicker.pick(candidates: cands, host: h).machineReadableWinnerLine()
        XCTAssertNotNil(dflt)
        XCTAssertFalse(dflt!.contains("prefer="), "default preference adds no field: \(dflt!)")
        let quality = QuantAutoPicker.pick(
            candidates: cands, host: h, preference: .quality).machineReadableWinnerLine()
        XCTAssertTrue(quality!.contains("prefer=quality"), "non-default preference is surfaced: \(quality!)")
    }

    /// Preference parsing fails closed on an unknown value (mirrors ServeTier.validated), so a typo
    /// exits before any serve instead of silently defaulting to context-first.
    func testPreferenceValidation_failsClosedOnUnknown() {
        XCTAssertEqual(try? QuantPickPreference.validated("quality"), .quality)
        XCTAssertEqual(try? QuantPickPreference.validated("context"), .context)
        XCTAssertThrowsError(try QuantPickPreference.validated("speed"))
    }
}
