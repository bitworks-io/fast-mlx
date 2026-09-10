import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// Drives `MTPSpeculativeDecoder` through a REAL `InferenceActor.generateBounded` run, using the
/// weight-free target/drafter fixtures extracted (unmodified, access-level-only) into
/// `InCheckpointMTPMockFixtures.swift`. Per
/// `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md` ("The discriminating test for an
/// inert bridge"): the load-bearing proof is that a genuinely speculating bridge costs FEWER
/// target forward calls than emitted tokens, not merely that it runs without crashing.
///
/// TWO KNOWN CAVEATS, written down rather than hidden (per the decision doc's own callout):
///   1. `InCheckpointMTPMockCountingKVCache.isTrimmable` is `true`, so every run below takes the
///      TRIMMABLE-cache path (`canTrimPromptCache(mainCache)` short-circuits true in
///      `MTPSpeculativeTokenIterator.init`). It does NOT exercise blocker 2's native-rewind
///      constraint (the production hybrid layer cache reports `isTrimmable == false`, with a
///      depth-2 native rewind) — that
///      needs a `SpeculativeCacheRewindModel`-conforming mock or a fleet run.
///   2. The mock target is not a `SpeculativeCacheRewindModel`, so `maximumNativeTargetCacheRewind`
///      is never consulted here either.
///   3. (Section 7b, sampled block decisions) None of this file's `MTPDrafterModel` mocks
///      (`InCheckpointMTPMockDrafter` et al.) ever call the `sampler: any LogitSampler` argument
///      `draftBlock` receives — they return a fixed proposal regardless. A real drafter calls it
///      once per proposed position, which is what feeds a sampled block-decision provider's
///      recorder its captured proposals. Against these fixtures a provider's `decide()` therefore
///      always sees zero captured proposals against `blockSize - 1` requested tokens and throws
///      `proposalCountMismatch` on every round — caught by
///      `MTPSpeculativeTokenIterator.speculateRound()` (never propagated) and degraded to sticky
///      passthrough with reason `"sampled MTP block decision failed: ..."`. Section 7b's tests
///      exploit this deliberately: reaching that specific, provider-only passthrough reason is
///      itself the proof that `sampledBlockDecisionsEnabled` really did construct a provider and
///      hand it to the iterator, not merely that the flag compiles. It does NOT prove a provider
///      ever ACCEPTS a proposal end to end — that needs a drafter mock that calls `sampler.sample`,
///      which is out of this increment's scope.
final class MTPSpeculativeDecoderTests: XCTestCase {

    // MARK: - Shared fixture builders

    /// A single-value-beyond-the-bonus planned continuation makes an accept-all run trivial to
    /// construct for any `blockSize`/round count: the mock target's one-hot logits are indexed
    /// purely by absolute cache position (see `InCheckpointMTPMockTargetModel.makeLogits`), so a
    /// drafter that always proposes the same fixed value matches the target's greedy continuation
    /// at every position from `bonusValue` onward, for as many rounds as the run needs. `count`
    /// must comfortably exceed anything a test below requests; the mock falls back to token `0`
    /// past the declared range rather than crashing, but staying inside it keeps every position's
    /// planned value explicit and intentional.
    private static func acceptAllPlannedTokens(
        bonusValue: Int32 = 5, tailValue: Int32 = 6, count: Int = 80
    ) -> [Int32] {
        [0, 0, bonusValue] + Array(repeating: tailValue, count: count)
    }

    /// Forces a REAL rejection inside an otherwise-accepting run — unlike `acceptAllPlannedTokens`
    /// paired with a drafter that always proposes the tail value (every draft matches by
    /// construction, so an "accepted < proposed" assertion built on it alone can never fail).
    /// Index 4 is the one planned value that does NOT equal `draftedTokenValue`; every other index
    /// from 3 onward does. With `blockSize == 3` (`numDraft == 2`) and a drafter that always
    /// proposes `draftedTokenValue` for both slots, this makes round 1's SECOND slot mismatch —
    /// `k == 1` accepted that round — while every later round (starting at the next planned index)
    /// accepts both slots — `k == 2`. See
    /// `testPerRequestDeltaMatchesIndependentlyComputedArithmeticForBlockSizeThree`'s doc comment
    /// for the full round-by-round derivation this fixture is built to support.
    private static func mismatchThenAcceptPlannedTokens(
        bonusValue: Int32 = 5, draftedTokenValue: Int32 = 6, divergentValue: Int32 = 7,
        count: Int = 80
    ) -> [Int32] {
        var tokens: [Int32] = [0, 0, bonusValue] + Array(repeating: draftedTokenValue, count: count)
        tokens[4] = divergentValue
        return tokens
    }

    // MARK: - 1. Anti-inertness (load-bearing)

    /// At `blockSize` 3, a genuinely speculating accept-all run of N emitted tokens costs roughly
    /// N/3 target forwards. A wired-but-inert bridge (sticky passthrough, or a plain `MLXDecoder`
    /// slipped in where this decoder belongs) costs exactly N. This is the single test the
    /// decision doc calls out as load-bearing; the mutation-check evidence for it is reported
    /// separately (see the cycle 77 verification log), not committed here as a permanent second
    /// test, per this increment's instructions.
    ///
    /// `draftBlockCallCount > 0` is asserted too, as the DECISION doc's own weaker companion: it
    /// proves the drafter ran, not that speculation reached the output, so it is paired with the
    /// forward-count assertion rather than substituted for it.
    func testTargetForwardCallCountIsFewerThanEmittedTokenCountOnAcceptAllRun() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        // Extract the Sendable counter boxes BEFORE `target`/`drafter` are captured into the
        // decoder and `sending`-transferred into the actor below. Reading `target`/`drafter`
        // themselves after that point is a compile-time data race under Swift 6 strict
        // concurrency (region isolation treats the whole object graph reachable from the sent
        // decoder as transferred) even though only a `Sendable` sub-property is wanted; holding
        // a direct reference to the boxes sidesteps that without touching the non-Sendable mocks.
        let forwardCallCounter = target.forwardCallCounter
        let draftBlockCallCounter = drafter.draftBlockCallCounter
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 9
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let forwardCalls = forwardCallCounter.value
        XCTAssertGreaterThan(draftBlockCallCounter.value, 0)
        XCTAssertLessThan(
            forwardCalls, summary.generatedTokenCount,
            "a genuinely speculating decoder must cost fewer target forwards than emitted "
                + "tokens; \(forwardCalls) forwards for \(summary.generatedTokenCount) tokens "
                + "reads like an inert (non-accelerating) bridge")
    }

    // MARK: - 2. Correctness companion

    /// A drafter whose fixed proposal never matches the target's real greedy continuation makes
    /// EVERY round's draft rejected, so this run is far closer to scalar throughput than the
    /// accept-all run above — but the target's own verify pass always resamples the correct next
    /// token(s) from the true distribution regardless of acceptance, so the OUTPUT must be
    /// byte-identical to the accept-all run's output. This is what proves the fast case (test 1)
    /// isn't bought with wrong answers. Compares length first, then scalar-by-scalar in a loop —
    /// never `XCTAssertEqual` on the two arrays directly (a failing array `==` triggers a Myers
    /// diff that dumps both arrays and can wedge the run).
    func testMismatchedDrafterProducesByteIdenticalTokensToTheAcceptAllRun() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let maxTokens = 9

        let acceptedTarget = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let acceptedDrafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        let acceptedDecoder = try MTPSpeculativeDecoder(
            target: acceptedTarget, drafter: acceptedDrafter,
            cacheFactory: { acceptedTarget.newCache(parameters: nil) })
        let acceptedActor = InferenceActor(decoder: acceptedDecoder)
        let acceptedRecorder = MTPSpeculativeDecoderTokenRecorder()
        let acceptedSummary = try await acceptedActor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { token in
            await acceptedRecorder.append(token)
            return .continueGeneration
        }

        let rejectedTarget = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        // 3 never matches the target's real continuation (5 for the bonus, 6 for everything
        // after), so every proposal in every round is rejected.
        let rejectedDrafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 3)
        // See the identical extraction in test 1 above for why this must happen before the
        // decoder/actor are built.
        let rejectedDraftBlockCallCounter = rejectedDrafter.draftBlockCallCounter
        let rejectedDecoder = try MTPSpeculativeDecoder(
            target: rejectedTarget, drafter: rejectedDrafter,
            cacheFactory: { rejectedTarget.newCache(parameters: nil) })
        let rejectedActor = InferenceActor(decoder: rejectedDecoder)
        let rejectedRecorder = MTPSpeculativeDecoderTokenRecorder()
        let rejectedSummary = try await rejectedActor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { token in
            await rejectedRecorder.append(token)
            return .continueGeneration
        }

        XCTAssertEqual(acceptedSummary.finishReason, .length)
        XCTAssertEqual(rejectedSummary.finishReason, .length)
        // The drafter genuinely ran and genuinely lost every round — the "weaker companion" from
        // the anti-inertness test, applied to the mismatched configuration.
        XCTAssertGreaterThan(rejectedDraftBlockCallCounter.value, 0)

        let accepted = await acceptedRecorder.values
        let rejected = await rejectedRecorder.values
        XCTAssertEqual(accepted.count, rejected.count)
        for (index, acceptedToken) in accepted.enumerated() {
            XCTAssertEqual(acceptedToken, rejected[index], "token index \(index)")
        }
    }

    // MARK: - 3. Budget: InferenceActor is the sole authority

    /// `MTPSpeculativeDecoder.buildParameters()` never sets `maxTokens` (always `nil`); this test
    /// proves that indirectly but concretely: requesting a generation far longer than any
    /// plausible hardcoded internal budget (the startup gate hardcodes 3-8 token budgets
    /// elsewhere in this codebase) still runs to completion with EXACTLY the actor-requested
    /// count. If `buildParameters()` regressed to hardcoding a small `maxTokens`, this run would
    /// either throw `.iteratorExhausted` early or stop short — either way this assertion fails.
    ///
    /// This test does NOT force `nextThrowing()` to return `nil` from a live iterator: reading
    /// `MTPSpeculativeTokenIterator.nextThrowing()`/`speculateRound()`/`passthroughStep()`
    /// (`MTPSpeculativeTokenIterator.swift:525-528`, `:899-900`, `:1010-1011`), the ONLY
    /// structural source of a budget-caused `nil` is `parameters.maxTokens` being reached — which
    /// this decoder guarantees never happens (parameters.maxTokens is always nil). So with this
    /// decoder's own invariant intact, `MTPSpeculativeDecoderError.iteratorExhausted` is
    /// unreachable through the real iterator in ordinary operation; forcing it organically would
    /// require a hostile custom drafter/target pair engineered around a currently-undocumented
    /// internal edge case, which is out of scope for this weight-free proof. The error path itself
    /// (never fabricating a token from `nil`) is proven by code inspection: `prefill`/`step` both
    /// route a `nil` from `nextThrowing()` straight to `throw .iteratorExhausted`, with no
    /// intervening branch that could substitute a token or a stop signal.
    func testActorMaxTokensIsTheSoleBudgetAuthorityForALongRun() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens(count: 200)
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 50
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
    }

    // MARK: - 4. step(last:) invariant

    /// `step(last:)` asserts `last == lastReturnedToken` on every call. `InferenceActor`'s own
    /// loops (`generateBounded` and `run`/`submit`) always feed back exactly the token they were
    /// last given, so a normal run exercises this assertion on every step without tripping it —
    /// `Swift.assert` traps the process on violation rather than throwing a catchable error, so
    /// the NEGATIVE case (an actual mismatch) cannot be exercised inside this test process without
    /// a subprocess harness; this test instead documents and exercises the positive case: the
    /// invariant holding across a full multi-round run, including the pending-token-drain steps
    /// where `last` is fed back mid-round rather than only at round boundaries.
    func testStepLastInvariantHoldsAcrossANormalMultiRoundRun() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 12, eos: 99
        ) { _ in .continueGeneration }

        // Reaching here at all (rather than a process trap from the assertion inside `step`) is
        // the proof: a mismatch between `last` and the decoder's own last-returned token would
        // have aborted the process before this line.
        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, 12)
    }

    // MARK: - 5. reset()

    /// A second `generateBounded` on the SAME actor (same decoder instance) starts fresh — this
    /// decoder's `prefill` always rebuilds a brand-new iterator over a brand-new
    /// `cacheFactory()`-built cache regardless of `reset()`, so the two runs' token sequences must
    /// be identical for an identically-shaped request — and telemetry ACCUMULATES across the reset
    /// rather than resetting to zero: the second run's snapshot must equal exactly twice the
    /// first's, since both runs are configured identically.
    func testResetStartsFreshAndTelemetryAccumulatesAcrossRuns() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let firstRecorder = MTPSpeculativeDecoderTokenRecorder()
        let firstSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 6, eos: 99
        ) { token in
            await firstRecorder.append(token)
            return .continueGeneration
        }
        let firstTelemetrySnapshot = await actor.speculativeTelemetry()
        let firstTelemetry = try XCTUnwrap(firstTelemetrySnapshot)

        let secondRecorder = MTPSpeculativeDecoderTokenRecorder()
        let secondSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 6, eos: 99
        ) { token in
            await secondRecorder.append(token)
            return .continueGeneration
        }
        let secondTelemetrySnapshot = await actor.speculativeTelemetry()
        let secondTelemetry = try XCTUnwrap(secondTelemetrySnapshot)

        XCTAssertEqual(firstSummary.finishReason, .length)
        XCTAssertEqual(secondSummary.finishReason, .length)

        let first = await firstRecorder.values
        let second = await secondRecorder.values
        XCTAssertEqual(first.count, second.count)
        for (index, firstToken) in first.enumerated() {
            XCTAssertEqual(firstToken, second[index], "token index \(index)")
        }

        XCTAssertGreaterThan(firstTelemetry.proposedCount, 0)
        XCTAssertGreaterThan(firstTelemetry.acceptedCount, 0)
        XCTAssertNil(firstTelemetry.passthroughReason)
        // Telemetry survives `reset()` (called in `generateBounded`'s `defer` after the first
        // run) AND accumulates: the second run's counters are exactly double the first's, since
        // both runs are configured identically and `reset()` snapshots into decoder-owned
        // accumulators rather than discarding them.
        XCTAssertEqual(secondTelemetry.proposedCount, firstTelemetry.proposedCount * 2)
        XCTAssertEqual(secondTelemetry.acceptedCount, firstTelemetry.acceptedCount * 2)
        XCTAssertNil(secondTelemetry.passthroughReason)
    }

    // MARK: - 6. blockSize rejection

    func testBlockSizeOtherThanServingBlockSizeIsRejectedAtConstruction() {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)

        XCTAssertThrowsError(
            try MTPSpeculativeDecoder(
                target: target, drafter: drafter,
                cacheFactory: { target.newCache(parameters: nil) },
                blockSize: 4)
        ) { error in
            XCTAssertEqual(error as? MTPSpeculativeDecoderError, .unsupportedBlockSize(4))
        }
    }

    func testBlockSizeThreeAndTheDefaultBothSucceedAtConstruction() throws {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)

        _ = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            blockSize: MTPSpeculativeDecoder.servingBlockSize)
        _ = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        XCTAssertEqual(MTPSpeculativeDecoder.servingBlockSize, 3)
    }

    // MARK: - 7. Sampled requests are not refused

    /// `MTPSpeculativeDecoderGreedyOnlyDrafter.requiresGreedySampling == true` mirrors the real
    /// in-checkpoint MTP drafter's constraint (Qwen MTP currently requires greedy). A `.sampled`
    /// request against it must still run to completion (never refused) and — because the
    /// iterator's own init gates exactly on `parameters.temperature != 0`
    /// (`MTPSpeculativeTokenIterator.swift:169-176`) — goes sticky passthrough, which only happens
    /// if `buildParameters()` actually threaded a nonzero temperature through. If this decoder
    /// silently hardcoded `temperature: 0` (the exact downgrade the decision doc forbids),
    /// `passthroughReason` would stay `nil` here.
    func testSampledRequestAgainstAGreedyOnlyDrafterIsNotRefusedAndGoesPassthrough() async throws {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99,
            sampling: .sampled(temperature: 0.8, topP: 1.0, topK: nil, minP: nil, seed: 42)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, 5)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        XCTAssertNotNil(telemetry.passthroughReason)
    }

    // MARK: - 7b. Sampled block decisions (opt-in provider wiring)
    //
    // See this file's header comment, caveat 3, for why these tests key on the specific
    // "sampled MTP block decision failed" passthrough reason rather than a genuine acceptance —
    // that reason string can ONLY appear if `sampledBlockDecisionsEnabled` actually constructed a
    // provider, the iterator's `providerIsEligible` gate actually admitted it (temperature/topP/
    // topK/minP/penalties all matched `sharedSampledMTPSupportsPredicate` and the stored
    // `SampledMTPSamplingTruncation`), and `decide()` was actually invoked. A decoder that
    // silently dropped the new parameter, or built the wrong provider variant, could not produce
    // this exact reason.

    /// Opt-in + a SEEDED sampled request (matches `SampledMTPSamplingTruncation.untruncated`:
    /// temperature 1, topP 1, topK 0, minP 0, no penalties) reaches
    /// `SeededSampledMTPBlockRuntimeProvider.decide()` and degrades gracefully (never throws to the
    /// caller) when it fails against this file's sampler-bypassing mocks — see caveat 3.
    func testSampledBlockDecisionsEnabledWithSeedReachesProviderDecideAndDegradesGracefully()
        async throws
    {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 1.0, topP: 1.0, topK: nil, minP: nil, seed: 7)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "a seeded, truncation-matched sampled request with sampledBlockDecisionsEnabled must "
                + "reach the provider seam, not silently stay on the ordinary sampler path")
        XCTAssertTrue(
            reason.contains("sampled MTP block decision failed"),
            "expected the provider-specific passthrough reason, got: \(reason)")
    }

    /// Same as above, but UNSEEDED — reaches
    /// `NondeterministicSampledMTPBlockRuntimeProvider.decide()` instead (the seed-vs-no-seed
    /// branch in `MTPSpeculativeDecoder.prefill`), and degrades identically.
    func testSampledBlockDecisionsEnabledWithoutSeedReachesProviderDecideAndDegradesGracefully()
        async throws
    {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 1.0, topP: 1.0, topK: nil, minP: nil, seed: nil)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "an unseeded, truncation-matched sampled request with sampledBlockDecisionsEnabled "
                + "must reach the provider seam, not silently stay on the ordinary sampler path")
        XCTAssertTrue(
            reason.contains("sampled MTP block decision failed"),
            "expected the provider-specific passthrough reason, got: \(reason)")
    }

    /// Control: the SAME truncation-matched sampled request, but with `sampledBlockDecisionsEnabled`
    /// left at its default (`false`) — the decoder default this whole flag must preserve. No
    /// provider is ever constructed (`MTPSpeculativeDecoder.prefill` passes `nil`), so
    /// `providerIsEligible` is unconditionally false regardless of how well the request's
    /// truncation matches, and the run takes today's unmodified genuinely-speculating path: no
    /// provider-specific passthrough (in fact no passthrough at all, since this drafter's fixed
    /// proposal matches every planned tail token).
    func testSampledBlockDecisionsDisabledByDefaultLeavesMatchedTruncationRequestGenuinelySpeculating()
        async throws
    {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        // Extracted BEFORE the decoder/actor are built, for the same reason as test 1 above: the
        // mocks are not `Sendable`, so touching `drafter` after it has been sent into the actor is
        // a Swift 6 data-race error. A direct reference to the counter box sidesteps that.
        let draftBlockCallCounter = drafter.draftBlockCallCounter
        // Deliberately NOT passing `sampledBlockDecisionsEnabled:` — this is the default-parameter
        // construction path every pre-existing call site (all ~15 of them) already uses.
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 1.0, topP: 1.0, topK: nil, minP: nil, seed: 7)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        XCTAssertNil(
            telemetry.passthroughReason,
            "sampledBlockDecisionsEnabled defaulting to false must not change this request's "
                + "already-genuinely-speculating outcome")
        XCTAssertGreaterThan(draftBlockCallCounter.value, 0)
    }

    // MARK: - 7c. Sampled block decisions against a REQUIRES-GREEDY drafter (the flag's actual
    // reason to exist)
    //
    // Every test in section 7b above uses `MTPSpeculativeDecoderCountingDrafter`, which inherits
    // `MTPDrafterModel`'s protocol-default `requiresGreedySampling == false`
    // (`MTPDrafterModel.swift:101`). Against that mock, `MTPSpeculativeTokenIterator`'s greedy
    // passthrough gate (`drafter.requiresGreedySampling, parameters.temperature != 0,
    // !providerIsEligible`) never even reaches its third term, since the FIRST term is already
    // false — so section 7b's tests cannot show the flag doing the one thing it exists to do:
    // release a `requiresGreedySampling == true` drafter (the real in-checkpoint MTP drafter's
    // actual constraint, per section 7's own `MTPSpeculativeDecoderGreedyOnlyDrafter`) from that
    // gate for a truncation-matched sampled request. These three tests use
    // `MTPSpeculativeDecoderGreedyOnlyDrafter` instead, exactly as section 7 already does.

    /// Flag ON + a `requiresGreedySampling == true` drafter + the deployed thinking preset
    /// (temperature 1, topP 0.95, topK 20, minP 0 — `sharedSampledMTPSupportsPredicate` accepts
    /// this truncated range, not only the untruncated `topP == 1, topK == 0` shape) must NOT take
    /// the greedy passthrough: `providerIsEligible` becomes true (temperature != 0, no processor,
    /// and the provider's `supports(parameters:)` matches both the shared predicate and its own
    /// stored truncation — see `MTPSpeculativeDecoder.prefill`, which derives that stored
    /// truncation from this SAME request), so the gate's `!providerIsEligible` term is false and
    /// the whole conjunction never fires, regardless of `requiresGreedySampling`.
    ///
    /// This file's own header comment (caveat 3) already establishes that none of its
    /// `MTPDrafterModel` mocks call the `sampler` argument `draftBlock` receives, so a provider
    /// constructed against these fixtures always sees zero captured proposals against
    /// `blockSize - 1` requested tokens and throws `proposalCountMismatch` — caught by
    /// `speculateRound()` and degraded to sticky passthrough with reason "sampled MTP block
    /// decision failed: ...". Reaching THAT specific, provider-only reason (never the greedy-
    /// requirement one) is exactly the proof this test needs: it can only appear if
    /// `providerIsEligible` really did flip true and the iterator really did hand control to the
    /// provider, not merely that the flag compiles.
    func testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndDeployedPresetEscapesGreedyPassthrough()
        async throws
    {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: 42)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "a truncation-matched sampled request against a requires-greedy drafter with "
                + "sampledBlockDecisionsEnabled must reach the provider seam, not silently stay "
                + "on the ordinary sampler path")
        XCTAssertFalse(
            reason.contains("requires temperature == 0"),
            "the greedy-requirement passthrough must not fire once providerIsEligible is true; "
                + "got: \(reason)")
        XCTAssertTrue(
            reason.contains("sampled MTP block decision failed"),
            "expected the provider-specific degradation reason (proof providerIsEligible really "
                + "flipped true and the iterator handed control to the provider), got: \(reason)")
    }

    /// The discriminating control for the test above, and the honest "production default" check
    /// section 7b's own control (`testSampledBlockDecisionsDisabledByDefaultLeavesMatchedTruncationRequestGenuinelySpeculating`)
    /// does not actually provide: THE SAME preset, THE SAME requires-greedy drafter, but with
    /// `sampledBlockDecisionsEnabled` left at its default `false`. No provider is ever constructed
    /// (`MTPSpeculativeDecoder.prefill` passes `nil`), so `providerIsEligible` is unconditionally
    /// false regardless of how well the request's truncation matches, and the real production
    /// drafter's greedy-requirement passthrough still fires.
    func testSampledBlockDecisionsDisabledWithGreedyOnlyDrafterAndDeployedPresetStillTakesGreedyPassthrough()
        async throws
    {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        // Deliberately NOT passing `sampledBlockDecisionsEnabled:` — the default every pre-existing
        // call site uses, and the production default this control protects.
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: 42)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "against a requires-greedy drafter with the flag off, this sampled request must still "
                + "take today's unmodified greedy-requirement passthrough")
        XCTAssertTrue(
            reason.contains("requires temperature == 0"),
            "expected the greedy-requirement passthrough reason (the production default this "
                + "flag must preserve when off), got: \(reason)")
    }

    /// Flag ON + the requires-greedy drafter, an INSTRUCT-style request (temperature 0.7, topP
    /// 1.0), and a nonzero presence penalty, which makes `parameters.processor()` non-nil.
    ///
    /// Before the step-(i) relaxation of `sharedSampledMTPSupportsPredicate` (temperature `== 1`
    /// -> `> 0 && .isFinite`), this request was refused on TWO independent grounds: temperature
    /// 0.7 failed the predicate's old `temperature == 1` clause, AND the presence penalty made
    /// `parameters.processor()` non-nil (the OTHER, independent way `providerIsEligible` is
    /// false). After that relaxation, temperature 0.7 is a finite positive temperature the
    /// predicate now ADMITS, so only the penalty ground survives here — the temperature ground
    /// was deliberately removed by the relaxation, not accidentally lost. This test alone can no
    /// longer distinguish "refused for the penalty" from "refused for anything at all"; see
    /// `testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndInstructPresetWithoutPenaltyEscapesGreedyPassthrough`
    /// immediately below for the companion that supplies that missing discrimination, by removing
    /// the penalty and asserting the SAME temperature/topP shape is now admitted.
    func testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndInstructPresetStillTakesGreedyPassthrough()
        async throws
    {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 0.7, topP: 1.0, topK: nil, minP: nil, seed: 42),
            penalties: DecoderPenalties(presencePenalty: 1.5)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "a penalized sampled request against a requires-greedy drafter must still take the "
                + "greedy-requirement passthrough even with the flag on")
        XCTAssertTrue(
            reason.contains("requires temperature == 0"),
            "the relaxed predicate must not over-admit a penalized request "
                + "sharedSampledMTPSupportsPredicate/providerIsEligible genuinely refuses on the "
                + "surviving (penalty) ground; got: \(reason)")
    }

    /// The discriminating companion to the test above: the IDENTICAL request shape (temperature
    /// 0.7, topP 1.0, requires-greedy drafter, flag on) with the ONE variable that test's own
    /// refusal now hinges on -- the presence penalty -- removed. Without this test, the assertion
    /// above cannot tell "refused for the penalty" apart from "refused for anything at all"; a
    /// regression that silently reintroduced a `temperature == 1` requirement would leave that
    /// test just as green. This test requires the opposite outcome: temperature 0.7 alone, once
    /// the step-(i) relaxation lands, must be enough to escape the greedy-requirement passthrough
    /// -- `providerIsEligible` must flip true on this shape exactly as it already does for
    /// temperature 1 (`testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndDeployedPresetEscapesGreedyPassthrough`
    /// above), landing on the SAME provider-specific degradation reason for the same documented
    /// cause (this file's mocks never call the `sampler` argument `draftBlock` receives).
    func testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndInstructPresetWithoutPenaltyEscapesGreedyPassthrough()
        async throws
    {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(temperature: 0.7, topP: 1.0, topK: nil, minP: nil, seed: 42)
            // Deliberately no `penalties:` argument -- the ONE variable changed from the test
            // above, isolating the temperature ground this companion exists to re-check.
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "a truncation-matched, unpenalized temperature-0.7 sampled request against a "
                + "requires-greedy drafter with sampledBlockDecisionsEnabled must reach the "
                + "provider seam, not silently stay on the ordinary sampler path")
        XCTAssertFalse(
            reason.contains("requires temperature == 0"),
            "the greedy-requirement passthrough must not fire once the step-(i) relaxation makes "
                + "providerIsEligible true at temperature 0.7; got: \(reason)")
        XCTAssertTrue(
            reason.contains("sampled MTP block decision failed"),
            "expected the provider-specific degradation reason (proof providerIsEligible really "
                + "flipped true at a non-1 finite temperature), got: \(reason)")
    }

    // MARK: - 7d. Tools do not enter sampled MTP admission (the tools x sampled-MTP intersection)
    //
    // `ToolCallFormat.infer` returns `.xmlFunction` for this family (`ToolCallFormat.swift:219-222`),
    // and tools affect how the prompt is RENDERED (`Qwen3TemplateRenderTests.swift`) and how output
    // text is PARSED into a tool call after generation — but never enter the sampling distribution
    // itself. This section pins that at the two seams that actually decide whether a sampled request
    // gets speculated: the provider's `supports(parameters:)` gate, and the iterator's
    // greedy-requirement passthrough. Both take a `GenerateParameters` value that structurally has
    // no `tools` field at all (`Evaluate.swift:54-135`), so a tool-bearing request's OWN sampling
    // knobs are exactly what reaches these gates. To make the "tool-bearing" framing more than a
    // comment, these tests derive those sampling knobs from a genuine, JSON-decoded
    // `OpenAIChatCompletionRequest` that carries a real `tools` array, through
    // `ServingSamplingPolicy.resolve(from:)` — the same production seam every serving backend
    // (`ContinuousServingBackend`/`ExactQwen35MTPServingBackend`) uses to turn a request into a
    // sampling decision — rather than writing `.sampled(temperature: 1.0, ...)` by hand.

    private static let toolBearingThinkingPresetRequestJSON = """
        {"model":"qwen3","messages":[
          {"role":"user","content":"do you have the RTX 6000 Ada in stock?"}
        ],"tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"seed":42,"max_tokens":384}
        """

    private static let toolFreeThinkingPresetRequestJSON = """
        {"model":"qwen3","messages":[
          {"role":"user","content":"do you have the RTX 6000 Ada in stock?"}
        ],"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"seed":42,"max_tokens":384}
        """

    /// Tools present, but a preset outside `sharedSampledMTPSupportsPredicate` — the anti-vacuity
    /// companion: a request that SHOULD be refused must still be refused with tools present.
    ///
    /// The refusal ground here is `min_p: 0.05` (the predicate requires `minP == 0`), NOT
    /// temperature: after the step-(i) relaxation of `sharedSampledMTPSupportsPredicate`
    /// (temperature `== 1` -> `> 0 && .isFinite`), this fixture's `temperature: 0.7` is itself
    /// admitted by the predicate, so it can no longer carry this test's refusal on its own. Keep
    /// `min_p` as the ground genuinely unrelated to tools — do not revert to relying on
    /// temperature here.
    private static let toolBearingInstructPresetRequestJSON = """
        {"model":"qwen3","messages":[
          {"role":"user","content":"do you have the RTX 6000 Ada in stock?"}
        ],"tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"temperature":0.7,"top_p":1.0,"min_p":0.05,"max_tokens":384}
        """

    private func requireSampledPolicy(
        _ policy: ServingSamplingPolicy, file: StaticString = #filePath, line: UInt = #line
    ) throws -> (temperature: Double, topP: Double, topK: Int?, minP: Double?, seed: Int64?) {
        guard case let .sampled(temperature, topP, topK, minP, seed) = policy else {
            XCTFail("expected a .sampled policy, got \(policy)", file: file, line: line)
            throw XCTSkip("unreachable")
        }
        return (temperature, topP, topK, minP, seed)
    }

    /// Item 1: admission (`supports(parameters:)`) is unaffected by tools. Resolving the
    /// tool-bearing and tool-free requests through `ServingSamplingPolicy.resolve(from:)` first —
    /// and asserting the two resolved policies are EQUAL — is the real, non-vacuous proof: that
    /// function's own signature reads only `temperature`/`topP`/`topK`/`minP`/`seed` off the
    /// request, never `tools`, so if a future change threaded `tools` into resolution this
    /// assertion would catch the divergence directly, rather than merely restating "these two
    /// hand-built `GenerateParameters` values happen to match."
    func testSampledProviderSupportsVerdictIsUnaffectedByToolsPresenceAtTheDeployedThinkingPreset()
        throws
    {
        let toolBearingRequest = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(Self.toolBearingThinkingPresetRequestJSON.utf8))
        let toolFreeRequest = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(Self.toolFreeThinkingPresetRequestJSON.utf8))
        XCTAssertFalse(
            toolBearingRequest.tools.isEmpty, "fixture sanity: this request must actually carry a tool")
        XCTAssertTrue(toolFreeRequest.tools.isEmpty, "fixture sanity: this request must carry none")

        let toolBearingPolicy = try ServingSamplingPolicy.resolve(from: toolBearingRequest)
        let toolFreePolicy = try ServingSamplingPolicy.resolve(from: toolFreeRequest)
        XCTAssertEqual(
            toolBearingPolicy, toolFreePolicy,
            "tools must not change what a request resolves to at the real serving boundary")

        let resolved = try requireSampledPolicy(toolBearingPolicy)
        let parameters = GenerateParameters(
            temperature: Float(resolved.temperature),
            topP: Float(resolved.topP),
            topK: resolved.topK ?? 0,
            minP: Float(resolved.minP ?? 0),
            presencePenalty: Float(toolBearingRequest.presencePenalty ?? 0))
        let provider = SeededSampledMTPBlockRuntimeProvider(
            seed: 1,
            truncation: SampledMTPSamplingTruncation(parameters: parameters))

        // Both requests resolve to the identical policy (asserted above), so both produce the
        // identical `GenerateParameters` and therefore the identical verdict — evaluated twice
        // below under the two labels the intersection actually cares about.
        XCTAssertTrue(
            provider.supports(parameters: parameters),
            "the deployed thinking preset, tools present, must be admitted")
        XCTAssertTrue(
            provider.supports(parameters: parameters),
            "the deployed thinking preset, tools absent (identical resolved parameters), must be "
                + "admitted identically")

        // Anti-vacuity companion: tools present, but a preset that SHOULD be refused. Without this
        // arm, a `supports()` that returned `true` unconditionally would have passed both
        // assertions above too.
        let instructPresetRequest = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(Self.toolBearingInstructPresetRequestJSON.utf8))
        XCTAssertFalse(instructPresetRequest.tools.isEmpty)
        let instructResolved = try requireSampledPolicy(
            try ServingSamplingPolicy.resolve(from: instructPresetRequest))
        let instructParameters = GenerateParameters(
            temperature: Float(instructResolved.temperature),
            topP: Float(instructResolved.topP),
            topK: instructResolved.topK ?? 0,
            minP: Float(instructResolved.minP ?? 0),
            presencePenalty: Float(instructPresetRequest.presencePenalty ?? 0))
        XCTAssertFalse(
            provider.supports(parameters: instructParameters),
            "an unsupported (min_p != 0) preset must still be refused, tools notwithstanding")
    }

    /// Item 2 (eligible arm): the SAME tool-bearing thinking-preset request, run all the way
    /// through a real `MTPSpeculativeDecoder` against a requires-greedy drafter with
    /// `sampledBlockDecisionsEnabled: true`. `providerIsEligible` becomes true, so the
    /// greedy-requirement passthrough's `!providerIsEligible` conjunct is false and that gate never
    /// fires — speculation engages (`initialPassthroughReason == nil` at
    /// `MTPSpeculativeTokenIterator.swift:169-183`). This file's own mocks never call the
    /// `sampler` argument `draftBlock` receives (see the file header, caveat 3), so the provider's
    /// `decide()` still degrades to a DIFFERENT, provider-only sticky passthrough — reaching that
    /// exact reason is itself the proof `providerIsEligible` really flipped true, mirroring
    /// `testSampledBlockDecisionsEnabledWithGreedyOnlyDrafterAndDeployedPresetEscapesGreedyPassthrough`
    /// above, but with the sampling knobs pulled from a genuine tool-bearing request instead of
    /// written by hand.
    func testToolBearingSampledRequestWithEligibleProviderEscapesGreedyPassthrough() async throws {
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(Self.toolBearingThinkingPresetRequestJSON.utf8))
        XCTAssertFalse(request.tools.isEmpty, "fixture sanity: this request must actually carry a tool")
        let resolved = try requireSampledPolicy(try ServingSamplingPolicy.resolve(from: request))

        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) },
            sampledBlockDecisionsEnabled: true)
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(
                temperature: resolved.temperature, topP: resolved.topP, topK: resolved.topK,
                minP: resolved.minP, seed: resolved.seed)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "a tool-bearing, truncation-matched sampled request against a requires-greedy drafter "
                + "with sampledBlockDecisionsEnabled must reach the provider seam, not silently stay "
                + "on the ordinary sampler path")
        XCTAssertFalse(
            reason.contains("requires temperature == 0"),
            "the greedy-requirement passthrough -- the ONE gate this test exists to prove a "
                + "tool-bearing request does not trip differently -- must not fire once "
                + "providerIsEligible is true; got: \(reason)")
        XCTAssertTrue(
            reason.contains("sampled MTP block decision failed"),
            "expected the provider-specific degradation reason, got: \(reason)")
    }

    /// Item 2 (ineligible arm, the decisive control): the identical tool-bearing request, with
    /// exactly ONE variable changed from the test above -- `sampledBlockDecisionsEnabled` left at
    /// its default `false` -- so `providerIsEligible` is unconditionally false and the real
    /// production drafter's greedy-requirement passthrough fires, with its EXACT reason string.
    func testToolBearingSampledRequestWithIneligibleProviderStillTakesGreedyPassthrough() async throws
    {
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(Self.toolBearingThinkingPresetRequestJSON.utf8))
        XCTAssertFalse(request.tools.isEmpty, "fixture sanity: this request must actually carry a tool")
        let resolved = try requireSampledPolicy(try ServingSamplingPolicy.resolve(from: request))

        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        // Deliberately NOT passing `sampledBlockDecisionsEnabled:` -- the default every
        // pre-existing call site uses, and the production default this control protects.
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 5
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99,
            sampling: .sampled(
                temperature: resolved.temperature, topP: resolved.topP, topK: resolved.topK,
                minP: resolved.minP, seed: resolved.seed)
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        let reason = try XCTUnwrap(
            telemetry.passthroughReason,
            "against a requires-greedy drafter with the flag off, this tool-bearing sampled "
                + "request must still take today's unmodified greedy-requirement passthrough")
        XCTAssertEqual(
            reason,
            "Qwen MTP currently requires temperature == 0; generating without speculation",
            "expected the EXACT greedy-requirement passthrough reason string, got: \(reason)")
    }

    // MARK: - 8. Per-request telemetry delta (observability)

    /// Independently-derived expected value, NOT the observed output pinned after the fact (see
    /// this file's own doc comment on `mismatchThenAcceptPlannedTokens` and the repeated project
    /// lesson that a pinned-without-derivation number is not a real check).
    ///
    /// Round-by-round arithmetic for `mismatchThenAcceptPlannedTokens()` (bonus=5, drafted=6,
    /// divergent=7 at planned index 4), prompt `[1, 2, 3]` (length 3), `blockSize == 3`
    /// (`numDraft == blockSize - 1 == 2`):
    ///
    /// - Prefill returns the prepare-time bonus token (planned index 2 == 5). This does NOT go
    ///   through `speculateRound()` — it is appended to `pendingTokens` directly during `prepare()`
    ///   — so it contributes nothing to `proposedCount`/`acceptedCount`/`verifyRoundCount`. The mock
    ///   target's cache offset is now 3 (the 3 prompt positions consumed).
    /// - Round 1 starts at cache offset (global planned index) 3. Draft slot 0 vs planned[3] == 6:
    ///   match. Draft slot 1 vs planned[4] == 7 (the injected divergence): drafted value is 6, so
    ///   this MISMATCHES — the accept loop stops with `accepted == 1` (`k == 1`), and the
    ///   mismatching target token (planned[4] == 7) is appended as the round's correction instead
    ///   of a bonus row. Round 1 emits `accepted + 1 == 2` tokens (values 6, 7).
    ///   `proposedCount += numDraft (2)`, `acceptedCount += accepted (1)`. The (real, non-mock) KV
    ///   cache trim advances the next round's start offset by exactly `accepted + 1 == 2`, so round
    ///   2 starts at offset `3 + 2 == 5`.
    /// - Round 2 starts at offset 5. Draft slot 0 vs planned[5] == 6: match. Draft slot 1 vs
    ///   planned[6] == 6: match — `accepted == numDraft == 2` (`k == 2`), so a bonus row is sampled
    ///   from planned[7] == 6. Round 2 emits 3 tokens (6, 6, 6). `proposedCount += 2` (running total
    ///   4), `acceptedCount += 2` (running total 3). Next round starts at offset `5 + 3 == 8`.
    /// - Round 3 starts at offset 8. Draft slot 0 vs planned[8] == 6: match. Draft slot 1 vs
    ///   planned[9] == 6: match — `k == 2` again, bonus from planned[10] == 6. Round 3 emits 3
    ///   tokens (6, 6, 6). `proposedCount += 2` (running total 6), `acceptedCount += 2` (running
    ///   total 5).
    ///
    /// Total emitted tokens: 1 (prefill bonus) + 2 (round 1) + 3 (round 2) + 3 (round 3) == 9,
    /// which is exactly `maxTokens` below — the run ends precisely at round 3's last token with no
    /// partial 4th round, so the accumulated counters ARE this request's whole delta (a fresh actor
    /// starts every counter at 0). Independently-computed expected values: `proposedDraftTokens ==
    /// 6`, `acceptedDraftTokens == 5` (strictly less than proposed, from round 1's real rejection —
    /// not a constant), `verifyRoundCount == 3`.
    func testPerRequestDeltaMatchesIndependentlyComputedArithmeticForBlockSizeThree() async throws {
        let plannedTokens = Self.mismatchThenAcceptPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 9
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        XCTAssertEqual(summary.generatedTokenCount, maxTokens)
        let delta = try XCTUnwrap(summary.speculativeDelta)
        XCTAssertEqual(delta.proposedDraftTokens, 6)
        XCTAssertEqual(delta.acceptedDraftTokens, 5)
        XCTAssertLessThan(delta.acceptedDraftTokens, delta.proposedDraftTokens)
        XCTAssertNil(delta.passthroughReason)

        // Independent second axis (round count), not derivable from proposed/accepted alone: three
        // `speculateRound()` calls (one per round above), reachable only through the actor's own
        // cumulative-snapshot API since it is not threaded onto `InferenceRunSummary`.
        let telemetrySnapshot = await actor.speculativeTelemetry()
        let telemetry = try XCTUnwrap(telemetrySnapshot)
        XCTAssertEqual(telemetry.verifyRoundCount, 3)
    }

    /// Companion to the arithmetic test above: a drafter whose fixed proposal never matches ANY
    /// planned continuation (every round rejects immediately, `k == 0`) still yields
    /// `acceptedDraftTokens` strictly less than `proposedDraftTokens` on the per-request delta —
    /// proving the reported number tracks real acceptance rather than being pinned to a constant
    /// (e.g. always reporting 0, or always reporting the proposed count unchanged).
    func testMismatchedDrafterYieldsPerRequestDeltaAcceptedStrictlyLessThanProposed() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        // 3 never matches the target's real continuation (5 for the bonus, 6 for everything
        // after) — every proposal in every round is rejected, so accepted stays 0 for the whole run
        // while proposed keeps growing.
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 3)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 9
        let summary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .length)
        let delta = try XCTUnwrap(summary.speculativeDelta)
        XCTAssertEqual(delta.acceptedDraftTokens, 0)
        XCTAssertGreaterThan(delta.proposedDraftTokens, 0)
        XCTAssertLessThan(delta.acceptedDraftTokens, delta.proposedDraftTokens)
    }

    /// THE load-bearing test for this increment: proves the reported number is a PER-REQUEST rate,
    /// not the decoder's lifetime cumulative counter. `MTPSpeculativeDecoder`'s counters are
    /// cumulative and deliberately survive `reset()` (see `testResetStartsFreshAndTelemetryAccumulatesAcrossRuns`
    /// above) — so a caller reading `InferenceRunSummary.speculativeDelta` on a SECOND request must
    /// see that request's OWN contribution, not the running total across both requests. Two
    /// identically-configured accept-all runs of `maxTokens == 6` each drive exactly 2 speculation
    /// rounds apiece (both full-accept, `k == 2`): `proposedCount += 2` per round (total 4),
    /// `acceptedCount += 2` per round (total 4) — matching
    /// `testResetStartsFreshAndTelemetryAccumulatesAcrossRuns`'s already-passing cumulative
    /// assertion that the SECOND run's cumulative snapshot equals exactly double the first's. If
    /// the second request's delta were computed wrong (e.g. reporting the cumulative snapshot
    /// instead of a delta), it would read `(8, 8)` here, not `(4, 4)`.
    func testSuccessiveRequestsReportPerRequestDeltasNotCumulativeTotals() async throws {
        let plannedTokens = Self.acceptAllPlannedTokens()
        let target = MTPSpeculativeDecoderCountingTargetModel(plannedTokens: plannedTokens)
        let drafter = MTPSpeculativeDecoderCountingDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let maxTokens = 6
        let firstSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }
        let secondSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: maxTokens, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(firstSummary.finishReason, .length)
        XCTAssertEqual(secondSummary.finishReason, .length)

        let firstDelta = try XCTUnwrap(firstSummary.speculativeDelta)
        let secondDelta = try XCTUnwrap(secondSummary.speculativeDelta)

        XCTAssertEqual(firstDelta.proposedDraftTokens, 4)
        XCTAssertEqual(firstDelta.acceptedDraftTokens, 4)
        // The load-bearing assertions: the SECOND request's own delta, not the running total.
        // A decoder that reported the cumulative snapshot instead of a delta would read (8, 8).
        XCTAssertEqual(secondDelta.proposedDraftTokens, 4)
        XCTAssertEqual(secondDelta.acceptedDraftTokens, 4)
    }

    /// THE regression this increment fixes:
    /// `docs/task-inbox/2026-09-08-mtp-passthrough-reason-sticky-leak.md`. Unlike every test above
    /// (each of which uses a FRESH decoder per case), this drives ONE decoder/actor through a
    /// passed-through request followed by a genuinely speculating one — exactly the serving
    /// shape (one decoder bound at load, reused for every request). Before the fix,
    /// `InferenceActor.runSummary` read `passthroughReason` from the decoder's cumulative
    /// snapshot, whose `stickyPassthroughReason` never clears
    /// (`MTPSpeculativeDecoder.speculativeTelemetrySnapshot`) — so the second request's reason
    /// would still read the first request's, even though the second demonstrably speculated
    /// (nonzero `acceptedDraftTokens`, asserted below so this cannot pass vacuously on a request
    /// that simply did not speculate).
    func testPassthroughReasonDoesNotLeakIntoALaterSpeculatingRequestOnTheSameDecoder() async throws {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        // `requiresGreedySampling == true`, but still functions as an ordinary fixed-proposal
        // drafter on a GREEDY request — see the second request below.
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        // First request: sampled against a drafter that requires greedy sampling -> sticky
        // passthrough, exactly like `testSampledRequestAgainstAGreedyOnlyDrafterIsNotRefusedAndGoesPassthrough`.
        let firstSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99,
            sampling: .sampled(temperature: 0.8, topP: 1.0, topK: nil, minP: nil, seed: 42)
        ) { _ in .continueGeneration }
        XCTAssertEqual(firstSummary.finishReason, .length)
        let firstDelta = try XCTUnwrap(firstSummary.speculativeDelta)
        XCTAssertNotNil(
            firstDelta.passthroughReason,
            "the passed-through request itself must still report its own reason — the fix must "
                + "not degenerate into 'never report a reason'")

        // Second request: an ORDINARY GREEDY request against the same decoder/drafter. The
        // drafter's fixed proposal (6) matches the target's accept-all continuation, so this
        // request genuinely speculates.
        let secondSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 6, eos: 99
        ) { _ in .continueGeneration }
        XCTAssertEqual(secondSummary.finishReason, .length)
        let secondDelta = try XCTUnwrap(secondSummary.speculativeDelta)
        XCTAssertGreaterThan(
            secondDelta.acceptedDraftTokens, 0,
            "the second request must have genuinely accepted draft tokens, or a nil "
                + "passthroughReason assertion below would pass vacuously on a request that "
                + "simply did not speculate")
        XCTAssertNil(
            secondDelta.passthroughReason,
            "a later, genuinely speculating request must not inherit an earlier request's sticky "
                + "passthrough reason")

        // Per-request counters (the passthrough gauge is a one-way latch; see
        // docs/task-inbox/2026-09-09-mtp-passthrough-rate-telemetry.md):
        // request 1 passed through, request 2 genuinely speculated (proven above by
        // `secondDelta.acceptedDraftTokens > 0`, the anti-vacuity control this assertion pair
        // depends on — without it, a decoder that never speculated at all could still produce a
        // `(1, 1)` reading here). These must read as a real per-request tally, not a sticky latch.
        let telemetryAfterBothRequests = await actor.speculativeTelemetry()
        let finalTelemetry = try XCTUnwrap(telemetryAfterBothRequests)
        XCTAssertEqual(
            finalTelemetry.passthroughRequestCount, 1,
            "exactly one of the two requests passed through")
        XCTAssertEqual(
            finalTelemetry.speculativeRequestCount, 1,
            "exactly one of the two requests genuinely speculated")
    }

    /// The trap this fix must avoid (see the task-inbox doc's "fix is not a plain string delta"
    /// section): comparing the cumulative snapshot's reason before/after a request looks like a
    /// delta but is wrong, because TWO consecutive requests that both legitimately pass through
    /// for the SAME reason would compare equal and the second would be falsely reported as
    /// `nil` (speculating). Both requests here must report a non-nil reason.
    func testConsecutivePassthroughRequestsBothReportTheirOwnReason() async throws {
        let target = MTPSpeculativeDecoderCountingTargetModel(
            plannedTokens: Self.acceptAllPlannedTokens())
        let drafter = MTPSpeculativeDecoderGreedyOnlyDrafter(draftedTokenValue: 6)
        let decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })
        let actor = InferenceActor(decoder: decoder)

        let sampling: DecoderSampling = .sampled(
            temperature: 0.8, topP: 1.0, topK: nil, minP: nil, seed: 42)

        let firstSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99, sampling: sampling
        ) { _ in .continueGeneration }
        let secondSummary = try await actor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99, sampling: sampling
        ) { _ in .continueGeneration }

        XCTAssertEqual(firstSummary.finishReason, .length)
        XCTAssertEqual(secondSummary.finishReason, .length)
        let firstDelta = try XCTUnwrap(firstSummary.speculativeDelta)
        let secondDelta = try XCTUnwrap(secondSummary.speculativeDelta)

        XCTAssertNotNil(
            firstDelta.passthroughReason,
            "guards against the fix degenerating into 'never report a reason'")
        XCTAssertNotNil(
            secondDelta.passthroughReason,
            "a naive 'reason changed since the previous cumulative snapshot' delta would report "
                + "nil here, since both requests pass through for the identical reason")
    }

    /// `ScriptedDecoder` does not conform to `SpeculativeTelemetryProviding`, so
    /// `InferenceRunSummary.speculativeDelta` must be `nil` — ABSENT, not a zero-valued delta.
    /// Zero is a legitimate speculative outcome (see the mismatched-drafter test above); `nil` is
    /// reserved for "this decoder has no speculative counters at all". Collapsing the two would
    /// make a non-speculative route indistinguishable from a speculative route that never accepts.
    func testNonSpeculativeDecoderReportsSpeculativeDeltaAsAbsentNotZero() async throws {
        let decoder = ScriptedDecoder(script: [1, 2, 3, 99], eos: 99)
        let actor = InferenceActor(decoder: decoder)

        let summary = try await actor.generateBounded(
            promptTokens: [7], maxTokens: 10, eos: 99
        ) { _ in .continueGeneration }

        XCTAssertEqual(summary.finishReason, .endOfSequence)
        XCTAssertNil(summary.speculativeDelta)
    }

    // MARK: - 9. Boot attestation: ScalarServingInCheckpointMTPStartupVerdict.machineReadableFields()
    //
    // Mirrors `InCheckpointMTPStartupGateEndToEndTests.swift`'s own
    // `testMachineReadableFieldsRendersEveryFieldWithDistinctValues` shape exactly (construct the
    // verdict directly with every OTHER field pinned, vary only the field under test, assert on
    // individual `contains("key=value")` scalars) -- that file is not in this increment's write
    // set, so these two directional assertions live here instead.

    /// `in_checkpoint_mtp_sampled_block_decisions=true` when the load enabled sampled block
    /// decisions.
    func testMachineReadableFieldsRendersSampledBlockDecisionsTrueWhenEnabled() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .converted,
            revision: "abc123",
            sourceKeyCount: 11,
            promptTokenCount: 22,
            generatedTokenCount: 33,
            proposedDraftTokens: 44,
            acceptedDraftTokens: 55,
            drafterServing: true,
            drafterActiveBytesDelta: 66,
            drafterCacheBytesDelta: 77,
            sampledBlockDecisionsEnabled: true)

        let fields = verdict.machineReadableFields()

        XCTAssertTrue(fields.contains("in_checkpoint_mtp_sampled_block_decisions=true"))
    }

    /// `in_checkpoint_mtp_sampled_block_decisions=false` when the load did NOT enable sampled block
    /// decisions -- both directions asserted, per this increment's own instructions, rather than
    /// only the positive case.
    func testMachineReadableFieldsRendersSampledBlockDecisionsFalseWhenDisabled() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .converted,
            revision: "abc123",
            sourceKeyCount: 11,
            promptTokenCount: 22,
            generatedTokenCount: 33,
            proposedDraftTokens: 44,
            acceptedDraftTokens: 55,
            drafterServing: true,
            drafterActiveBytesDelta: 66,
            drafterCacheBytesDelta: 77,
            sampledBlockDecisionsEnabled: false)

        let fields = verdict.machineReadableFields()

        XCTAssertTrue(fields.contains("in_checkpoint_mtp_sampled_block_decisions=false"))
    }

    /// The default-parameter construction path (no explicit `sampledBlockDecisionsEnabled:`) also
    /// renders `false` -- every existing call site into this initializer that predates this field
    /// (e.g. `InCheckpointMTPStartupGateEndToEndTests.swift`'s own verdict constructions) keeps
    /// reporting the conservative, unaffected value rather than an unset/garbage one.
    func testMachineReadableFieldsRendersSampledBlockDecisionsFalseByDefault() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .converted,
            revision: "abc123",
            sourceKeyCount: 11,
            promptTokenCount: 22,
            generatedTokenCount: 33,
            proposedDraftTokens: 44,
            acceptedDraftTokens: 55,
            drafterServing: true,
            drafterActiveBytesDelta: 66,
            drafterCacheBytesDelta: 77)

        let fields = verdict.machineReadableFields()

        XCTAssertTrue(fields.contains("in_checkpoint_mtp_sampled_block_decisions=false"))
    }
}

// MARK: - Test-local mocks (see this file's header for the shared, extracted fixtures)

/// Sendable, lock-protected counter box. A bare mutable field on the (non-Sendable)
/// `InCheckpointMTPMockTargetModel`/`MTPSpeculativeDecoderCountingTargetModel` cannot be read from
/// the test after the owning decoder crosses the actor boundary — reading it directly would be a
/// data race under Swift 6 strict concurrency (and the whole reason the decision doc calls this
/// out explicitly). Every increment and every read goes through this box's lock instead.
final class MTPSpeculativeDecoderCallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        count += 1
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}

/// Wraps `InCheckpointMTPMockTargetModel`, delegating every `LanguageModel` method to it
/// unchanged, except that every entry point representing "one target forward pass" additionally
/// increments `forwardCallCounter` (a `Sendable` box, safe to read from the test thread after the
/// owning decoder has been transferred into an `InferenceActor`). This is the discriminating
/// signal for the anti-inertness test: `LanguageModel.evaluateThrowing`'s default implementation
/// forwards to the `LMInput.Text`-overload `callAsFunction`, which is protocol-witness-dispatched
/// to THIS type's override (not the wrapped `inner`'s), so every forward the iterator performs —
/// through `evaluateThrowing`, `prepare`'s own follow-up forward, or `speculateRound`'s batched
/// verify call — is counted exactly once, regardless of which call site reached it.
final class MTPSpeculativeDecoderCountingTargetModel: Module, LanguageModel {
    private let inner: InCheckpointMTPMockTargetModel
    let forwardCallCounter = MTPSpeculativeDecoderCallCounter()

    init(plannedTokens: [Int32]) {
        self.inner = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, windowSize: windowSize)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCallCounter.increment()
        return inner.callAsFunction(inputs, cache: cache)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardCallCounter.increment()
        return inner.callAsFunction(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        inner.newCache(parameters: parameters)
    }
}

extension MTPSpeculativeDecoderCountingTargetModel: MTPPromptHiddenStatePreparingModel {
    /// `inner.prepareForMTP` calls `inner`'s OWN `callAsFunction` override internally (protocol
    /// dispatch resolves `self` to `inner`, not this wrapper), so that inner forward would not
    /// otherwise increment `forwardCallCounter`. Counted here explicitly instead, so this stays a
    /// clean 1-count-per-forward signal regardless of which iterator branch (stateless vs.
    /// production-shaped stateful) reached it. None of this file's tests currently drive a
    /// `requiresPromptPrefill` drafter through this path, but the conformance and counting are
    /// kept consistent in case a future increment does.
    func prepareForMTP(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> MTPPromptPreparation? {
        forwardCallCounter.increment()
        return try inner.prepareForMTP(input, cache: cache, windowSize: windowSize)
    }
}

/// Wraps `InCheckpointMTPMockDrafter`, delegating every `MTPDrafterModel` method to it unchanged,
/// except that `draftBlock` additionally increments `draftBlockCallCounter` (a `Sendable` box) —
/// the same race-free-read requirement as `MTPSpeculativeDecoderCountingTargetModel` above, for the
/// same reason: `InCheckpointMTPMockDrafter.draftBlockCallCount` is a bare mutable field on a
/// non-Sendable `Module`, so reading it from the test thread after the owning decoder has been
/// `sending`-transferred into an `InferenceActor` is a compile-time data race under Swift 6 strict
/// concurrency (confirmed by the build failure this type's introduction fixes).
final class MTPSpeculativeDecoderCountingDrafter: Module, MTPDrafterModel {
    private let inner: InCheckpointMTPMockDrafter
    let draftBlockCallCounter = MTPSpeculativeDecoderCallCounter()

    init(draftedTokenValue: Int32) {
        self.inner = InCheckpointMTPMockDrafter(draftedTokenValue: draftedTokenValue)
        super.init()
    }

    func supportsSpeculation(for input: LMInput, target: any LanguageModel) -> Bool {
        inner.supportsSpeculation(for: input, target: target)
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCounter.increment()
        return inner.draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            sampler: sampler)
    }
}

/// Same fixed-proposal behavior as `InCheckpointMTPMockDrafter`, except
/// `requiresGreedySampling == true` — mirroring the real in-checkpoint MTP drafter's actual
/// constraint (Qwen MTP currently requires greedy sampling), which the extracted
/// `InCheckpointMTPMockDrafter`/`InCheckpointMTPMockStatefulDrafter` fixtures deliberately do NOT
/// set (their tests only exercise greedy requests). Needed here, and only here, for test 7.
final class MTPSpeculativeDecoderGreedyOnlyDrafter: Module, MTPDrafterModel {
    let draftedTokenValue: Int32
    let requiresGreedySampling = true
    private(set) var draftBlockCallCount = 0

    init(draftedTokenValue: Int32) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

/// `actor`, not a plain class: the `consume` closure passed to `generateBounded` is `@Sendable`,
/// so token capture must itself be safe to call from within that closure. Port of the identical
/// pattern in `spike/Tests/SpikeCoreTests/InferenceActorTests.swift`'s private `TokenRecorder`
/// (not reusable directly — that one is `private` to a different test target).
actor MTPSpeculativeDecoderTokenRecorder {
    private var tokens: [Int] = []

    var values: [Int] { tokens }

    func append(_ token: Int) {
        tokens.append(token)
    }
}
