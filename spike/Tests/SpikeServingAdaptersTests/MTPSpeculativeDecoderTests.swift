import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
import SpikeCore

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
