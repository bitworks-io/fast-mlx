import XCTest

@testable import ServingCore

/// The continuous-batch route rejects every non-`qwen3` family at proof time
/// (`DenseContinuousBatchModelProof.verifying`). For families the scalar serving
/// route has been LIVE-PROVEN to handle, the executable may fall back to scalar
/// serving instead of failing the operator's one-command `serve.sh --model <repo>`.
/// This allowlist is the sole gate for that fallback and must stay fail-closed:
/// only families with a recorded live serving proof are eligible; everything else
/// (including sizer-classified hybrid-linear families that were never serving-proven)
/// rethrows and keeps `verifying` the sole authority on continuous support.
final class ScalarHybridFallbackPolicyTests: XCTestCase {
    func testQwen3_5IsEligibleForScalarHybridFallback() {
        XCTAssertTrue(isScalarHybridServingFamily("qwen3_5"))
    }

    func testContinuousSupportedQwen3NeverFallsBack() {
        // qwen3 is served by the continuous route itself; if it ever reaches the
        // fallback gate something is wrong upstream, so it must fail closed.
        XCTAssertFalse(isScalarHybridServingFamily("qwen3"))
    }

    func testSizerHybridLinearButServingUnprovenFamiliesFailClosed() {
        // These are classified hybrid-linear by the sizer (ModelConfigDecoder) but
        // have NO live scalar-serving proof for the path this fallback takes;
        // qwen3_5_moe is additionally the human-gated weights>RAM MoE bet. None may
        // silently fall back.
        //
        // qwen4_exp is the subtle one, and it is listed here deliberately. It DOES have a
        // live serving proof — but a route-scoped one, covering only the offloaded n-gram
        // route (`markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan` in
        // SpikeServingAdapters/MLXScalarServing.swift). This predicate takes a family and
        // nothing else, so it cannot express that scoping. The fallback threads
        // `arguments.ngramOffloadPlanURL`, which the parser guarantees is nil under
        // continuous batching, so admitting qwen4_exp here would always land on the plain
        // fully-resident load — ~106 GiB of weights loaded AFTER the memory-limit mutation
        // and only THEN refused by the marker gate. Since the sizer classifies qwen4_exp as
        // hybrid-linear, its absence from the allowlist keeps looking like a missing entry;
        // this case is the lock that stops it being "fixed".
        // Reasoning and both reopen conditions:
        // docs/task-inbox/2026-09-08-qwen4exp-scalar-fallback-allowlist-NO-GO.md
        for family in ["qwen3_next", "qwen3_5_moe", "qwen4_exp"] {
            XCTAssertFalse(
                isScalarHybridServingFamily(family),
                "\(family) has no live scalar-serving proof and must fail closed")
        }
    }

    func testUnknownAndEmptyFamiliesFailClosed() {
        for family in ["", "mamba", "gemma3", "llama", "Qwen3_5", "QWEN3_5", " qwen3_5"] {
            XCTAssertFalse(
                isScalarHybridServingFamily(family),
                "\(family) is not the exact proven family tag and must fail closed")
        }
    }

    func testFallbackAnnounceLineFormatIsLocked() {
        XCTAssertEqual(
            scalarHybridFallbackAnnounceLine(modelType: "qwen3_5"),
            "fastmlx-serve continuous_fallback=scalar model_type=qwen3_5 "
                + "reason=unsupported_continuous_family")
    }

    /// The opt-in hybrid-continuous ADMISSION announce (the counterpart to the fallback line): emitted
    /// when --allow-hybrid-qwen35 admits qwen3_5 onto the continuous route, so the operator can tell a
    /// successful hybrid-continuous serve from a silent scalar fallback. Machine-readable, fixed key order.
    func testHybridContinuousAdmissionAnnounceLineFormatIsLocked() {
        XCTAssertEqual(
            hybridQwen35ContinuousAdmissionAnnounceLine(modelType: "qwen3_5"),
            "fastmlx-serve continuous_admitted=hybrid model_type=qwen3_5 "
                + "reason=allow_hybrid_qwen35")
    }
}
