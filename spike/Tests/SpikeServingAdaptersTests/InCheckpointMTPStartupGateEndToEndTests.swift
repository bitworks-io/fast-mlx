import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
@testable import SpikeServingAdapters

final class InCheckpointMTPStartupGateEndToEndTests: XCTestCase {

    /// SUCCESS PATH (the whole point): a matched target+drafter pair where the drafter's fixed
    /// proposal equals the target's own greedy continuation, so the run's single speculative
    /// round fully accepts. Drives `verifyInCheckpointMTPStartupEquivalence` end to end through
    /// the REAL MLX `TokenIterator`/`MTPSpeculativeTokenIterator` machinery — the first test in
    /// this repository to do so (previously only the extracted, hand-fed decision function was
    /// covered). Asserts the call returns (does not throw) and that its reported telemetry
    /// reflects genuine speculation.
    func testVerifyStartupEquivalenceSucceedsWithGenuineAcceptedSpeculation() throws {
        // plannedTokens[2] is the "bonus" sampled from forwarding the 3-token prompt;
        // plannedTokens[3]/[4] are what the target itself greedily predicts next. Setting the
        // drafter's fixed proposal to that same value (7) makes round 1's two drafted tokens
        // both accept.
        let plannedTokens: [Int32] = [0, 0, 5, 7, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 7)

        let verdict = try verifyInCheckpointMTPStartupEquivalence(
            mainModel: target,
            drafter: drafter,
            promptTokens: [1, 2, 3],
            stopTokenIDs: [99],
            maxTokens: 4,
            blockSize: 3)

        XCTAssertEqual(verdict.promptTokenCount, 3)
        XCTAssertEqual(verdict.generatedTokenCount, 4)
        XCTAssertGreaterThan(verdict.proposedDraftTokens, 0)
        XCTAssertEqual(verdict.proposedDraftTokens, 2)
        XCTAssertEqual(verdict.acceptedDraftTokens, 2)
        XCTAssertEqual(drafter.draftBlockCallCount, 1)
        // ANTI-VACUITY, the other direction: this run's drafter has
        // `requiresPromptPrefill == false`, so none of the four `prepareForMTP`-reaching branches
        // in `MTPSpeculativeTokenIterator.prepare` can fire regardless of what the target
        // conforms to. Zero here — contrasted with the nonzero count on the same target type in
        // `testVerifyStartupEquivalenceSucceedsWithGenuineAcceptedSpeculationUsingTheProductionStatefulDrafter`
        // below — is what proves the stateless and production-shaped configurations actually
        // exercise different iterator code, rather than the stateful mock silently reproducing
        // this branch by accident.
        XCTAssertEqual(target.prepareForMTPCallCount, 0)
    }

    /// THE REGRESSION TEST FOR THE BUG FIXED IN cfcec033 — highest value in this increment.
    /// `draftedTokenValue` (3) never matches the target's own greedy continuation (7), so every
    /// proposal in the run's one full speculative round is rejected: `acceptedDraftTokens == 0`
    /// while `proposedDraftTokens > 0` (the round genuinely ran — `draftBlock` was actually
    /// called, once, with 1 proposed token). Before cfcec033,
    /// `inCheckpointMTPStartupEquivalenceDecision` gated on `acceptedDraftTokens > 0`, so this
    /// EXACT telemetry shape — a correct, genuinely-exercised drafter whose one proposal for the
    /// fixed startup prompt happens to diverge from the target's greedy path — would have
    /// PERMANENTLY refused to boot. This test reproduces that state end to end, with REAL
    /// telemetry from the real MLX iterator machinery (not a hand-written
    /// `InCheckpointMTPGreedyDecodeResult` literal), and proves the fixed gate boots it anyway.
    ///
    /// `verifyInCheckpointMTPStartupEquivalence`'s success return type does not expose
    /// `passthroughReason` directly, but `inCheckpointMTPStartupEquivalenceDecision` throws
    /// `.inCheckpointMTPStartupDidNotSpeculate` whenever `passthroughReason != nil` BEFORE ever
    /// reaching its return statement — so this call succeeding is itself the proof that
    /// `passthroughReason == nil` here, not merely an assumption.
    func testVerifyStartupEquivalenceSucceedsWhenEveryDraftProposalIsRejected() throws {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 3)

        let verdict = try verifyInCheckpointMTPStartupEquivalence(
            mainModel: target,
            drafter: drafter,
            promptTokens: [1, 2, 3],
            stopTokenIDs: [99],
            maxTokens: 3,
            blockSize: 2)

        XCTAssertGreaterThan(verdict.proposedDraftTokens, 0)
        XCTAssertEqual(verdict.proposedDraftTokens, 1)
        XCTAssertEqual(verdict.acceptedDraftTokens, 0)
        XCTAssertEqual(drafter.draftBlockCallCount, 1)
    }

    /// Passthrough refusal, end to end: a drafter that refuses this specific target
    /// (`supportsSpeculation(for:target:)` returns false) makes the iterator engage STICKY
    /// passthrough starting at `init`, so `draftBlock` is never called. Clause (i) still passes
    /// (passthrough forwards the identical greedy sequence scalar decode would produce from the
    /// same target), so only clause (ii)'s passthrough guard refuses this run.
    func testVerifyStartupEquivalenceThrowsDidNotSpeculateWhenDrafterRefusesTarget() {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 7)
        drafter.supportsTarget = false

        XCTAssertThrowsError(
            try verifyInCheckpointMTPStartupEquivalence(
                mainModel: target,
                drafter: drafter,
                promptTokens: [1, 2, 3],
                stopTokenIDs: [99],
                maxTokens: 3,
                blockSize: 2)
        ) { error in
            guard let loadError = error as? ScalarServingModelLoadError,
                case .inCheckpointMTPStartupDidNotSpeculate(let reason, _, _) = loadError
            else {
                XCTFail("expected inCheckpointMTPStartupDidNotSpeculate, got \(error)")
                return
            }
            XCTAssertNotNil(reason)
        }
        XCTAssertEqual(drafter.draftBlockCallCount, 0)
    }

    /// THE ZERO-ROUND CASE: the very first generated token (the prepare-time bonus) is itself a
    /// stop token, so the decode loop breaks immediately after draining it — `speculateRound`
    /// (and therefore `drafter.draftBlock`) never runs, `proposedDraftTokens == 0`. Pins the
    /// exact `inCheckpointMTPStartupDidNotSpeculate(reason: nil, proposedDraftTokens: 0,
    /// acceptedDraftTokens: 0)` shape this run must throw — the case
    /// `inCheckpointMTPStartupGateMaxTokens`'s own doc comment (corrected in cfcec033) warns a
    /// startup prompt whose first generated token is a stop token will always hit, regardless of
    /// how large the max-token budget is.
    func testVerifyStartupEquivalenceThrowsZeroProposedWhenFirstTokenIsAStopToken() {
        // The stop marker (5) must stay inside the mock target's fixed 20-token vocabulary — see
        // `InCheckpointMTPMockTargetModel.makeLogits`; a value outside it would index past the
        // one-hot logits buffer instead of exercising the gate.
        let plannedTokens: [Int32] = [0, 0, 5]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 7)

        XCTAssertThrowsError(
            try verifyInCheckpointMTPStartupEquivalence(
                mainModel: target,
                drafter: drafter,
                promptTokens: [1, 2, 3],
                stopTokenIDs: [5],
                maxTokens: 8,
                blockSize: 3)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .inCheckpointMTPStartupDidNotSpeculate(
                    reason: nil, proposedDraftTokens: 0, acceptedDraftTokens: 0))
        }
        XCTAssertEqual(drafter.draftBlockCallCount, 0)
    }

    // MARK: - Production-shaped pair: StatefulMTPDrafterModel + MTPPromptHiddenStatePreparingModel
    //
    // Every test above drives `InCheckpointMTPMockDrafter`, whose `requiresPromptPrefill` is
    // false — the branch production never takes (see this file's header and this increment's
    // task background: the real in-checkpoint MTP drafter sets `requiresPromptPrefill = true` and
    // pairs with a target conforming to `MTPPromptHiddenStatePreparingModel`). The three tests
    // below rerun the SAME plannedTokens/draftedTokenValue fixtures as their stateless
    // counterparts above through `InCheckpointMTPMockStatefulDrafter` (`requiresPromptPrefill ==
    // true`) against the SAME `InCheckpointMTPMockTargetModel` type, now also conforming to
    // `MTPPromptHiddenStatePreparingModel`. Reusing identical fixtures and asserting identical
    // verdict shapes is deliberate: it proves the production branch reproduces the already-proven
    // stateless outcome, while the `prepareForMTPCallCount`/`makeStateCallCount`/
    // `prepareDrafterStateCallCount` assertions prove it does so via genuinely different iterator
    // code, not by silently falling back to the stateless branch.

    /// Production-shaped success path — the stateful-drafter counterpart of
    /// `testVerifyStartupEquivalenceSucceedsWithGenuineAcceptedSpeculation`. Same
    /// `plannedTokens`/`draftedTokenValue`/`promptTokens`/`blockSize`/`maxTokens`, so an identical
    /// verdict here is not a coincidence: it is what "the production branch reproduces the
    /// stateless branch's proven behavior" looks like when true. The three nonzero call counts
    /// asserted at the end are this file's core anti-vacuity proof for the success path — each
    /// pins a DIFFERENT iterator call site the stateless configuration never reaches (see the
    /// mock class doc comments above for the exact line references).
    func testVerifyStartupEquivalenceSucceedsWithGenuineAcceptedSpeculationUsingTheProductionStatefulDrafter()
        throws
    {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockStatefulDrafter(draftedTokenValue: 7)

        let verdict = try verifyInCheckpointMTPStartupEquivalence(
            mainModel: target,
            drafter: drafter,
            promptTokens: [1, 2, 3],
            stopTokenIDs: [99],
            maxTokens: 4,
            blockSize: 3)

        XCTAssertEqual(verdict.promptTokenCount, 3)
        XCTAssertEqual(verdict.generatedTokenCount, 4)
        XCTAssertGreaterThan(verdict.proposedDraftTokens, 0)
        XCTAssertEqual(verdict.proposedDraftTokens, 2)
        XCTAssertEqual(verdict.acceptedDraftTokens, 2)
        XCTAssertEqual(drafter.draftBlockCallCount, 1)
        XCTAssertGreaterThan(target.prepareForMTPCallCount, 0)
        XCTAssertGreaterThan(drafter.makeStateCallCount, 0)
        XCTAssertGreaterThan(drafter.prepareDrafterStateCallCount, 0)
    }

    /// Production-shaped counterpart of
    /// `testVerifyStartupEquivalenceSucceedsWhenEveryDraftProposalIsRejected` — same regression
    /// coverage for the bug fixed in cfcec033, now proven on the branch production actually runs.
    /// Before this increment, the real drafter/target pair's every-
    /// proposal-rejected shape was untested end to end: the stateless test above proves the GATE
    /// LOGIC is correct, but not that the gate logic is reachable with the exact drafter/target
    /// capability combination production actually loads. This test closes that gap.
    func testVerifyStartupEquivalenceSucceedsWhenEveryDraftProposalIsRejectedUsingTheProductionStatefulDrafter()
        throws
    {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockStatefulDrafter(draftedTokenValue: 3)

        let verdict = try verifyInCheckpointMTPStartupEquivalence(
            mainModel: target,
            drafter: drafter,
            promptTokens: [1, 2, 3],
            stopTokenIDs: [99],
            maxTokens: 3,
            blockSize: 2)

        XCTAssertGreaterThan(verdict.proposedDraftTokens, 0)
        XCTAssertEqual(verdict.proposedDraftTokens, 1)
        XCTAssertEqual(verdict.acceptedDraftTokens, 0)
        XCTAssertEqual(drafter.draftBlockCallCount, 1)
        XCTAssertGreaterThan(target.prepareForMTPCallCount, 0)
        XCTAssertGreaterThan(drafter.makeStateCallCount, 0)
        XCTAssertGreaterThan(drafter.prepareDrafterStateCallCount, 0)
    }

    /// Production-shaped counterpart of
    /// `testVerifyStartupEquivalenceThrowsDidNotSpeculateWhenDrafterRefusesTarget`. A drafter
    /// refusal engages STICKY passthrough at `init`, BEFORE `prepare()` evaluates any of the
    /// `requiresPromptPrefill` branches (`MTPSpeculativeTokenIterator.swift:181-183` guards
    /// `makeState` on `initialPassthroughReason == nil`; `prepare()`'s four `prepareForMTP`-
    /// reaching branches are each additionally gated on `!passthrough`). So even though this
    /// drafter and target ARE production-shaped, none of the stateful call sites fire — asserting
    /// all three counts stay 0 here (contrasted with nonzero in the two tests above) is itself a
    /// distinct anti-vacuity proof: it shows the refusal path short-circuits the production
    /// machinery cleanly rather than partially engaging it.
    func testVerifyStartupEquivalenceThrowsDidNotSpeculateWhenDrafterRefusesTargetUsingTheProductionStatefulDrafter()
    {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockStatefulDrafter(draftedTokenValue: 7)
        drafter.supportsTarget = false

        XCTAssertThrowsError(
            try verifyInCheckpointMTPStartupEquivalence(
                mainModel: target,
                drafter: drafter,
                promptTokens: [1, 2, 3],
                stopTokenIDs: [99],
                maxTokens: 3,
                blockSize: 2)
        ) { error in
            guard let loadError = error as? ScalarServingModelLoadError,
                case .inCheckpointMTPStartupDidNotSpeculate(let reason, _, _) = loadError
            else {
                XCTFail("expected inCheckpointMTPStartupDidNotSpeculate, got \(error)")
                return
            }
            XCTAssertNotNil(reason)
        }
        XCTAssertEqual(drafter.draftBlockCallCount, 0)
        XCTAssertEqual(target.prepareForMTPCallCount, 0)
        XCTAssertEqual(drafter.makeStateCallCount, 0)
        XCTAssertEqual(drafter.prepareDrafterStateCallCount, 0)
    }

    /// TOKEN-SEQUENCE EQUALITY IS REAL, NOT ASSUMED: reruns the matched pair from
    /// `testVerifyStartupEquivalenceSucceedsWithGenuineAcceptedSpeculation` through test-local
    /// replays of the scalar (`TokenIterator`) and speculative (`MTPSpeculativeTokenIterator`)
    /// decode loops — mirroring, rather than calling, `runInCheckpointMTPScalarReference`/
    /// `runInCheckpointMTPSpeculativeDecode` (both `private`, so unreachable from a test target
    /// even with `@testable import`) — and asserts their raw token arrays actually match. This
    /// proves clause (i) of `inCheckpointMTPStartupEquivalenceDecision` compares two GENUINELY
    /// DIFFERENT call patterns against the same target (batched draft+verify calls vs sequential
    /// single-token steps), rather than being unreachable or vacuous: a position-tracking or
    /// batching bug in either loop would show up here as a mismatch. Compares by count and then
    /// scalar-by-scalar (never `Array ==`) per this increment's anti-Myers-diff constraint.
    func testScalarAndSpeculativeDecodeProduceIdenticalTokensForTheMatchedPair() throws {
        let plannedTokens: [Int32] = [0, 0, 5, 7, 7, 9]
        let target = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 7)
        let promptTokens = [1, 2, 3]
        let stopTokenIDs: Set<Int> = [99]
        let maxTokens = 4
        let blockSize = 3

        let scalarParameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        var scalarIterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens.map { Int32($0) })),
            model: target,
            cache: target.newCache(parameters: scalarParameters),
            parameters: scalarParameters)
        var scalarTokens: [Int] = []
        while let token = scalarIterator.next() {
            if stopTokenIDs.contains(token) {
                scalarIterator.discardGeneratedToken()
                break
            }
            scalarTokens.append(token)
            if scalarTokens.count >= maxTokens { break }
        }

        let speculativeParameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        var speculativeIterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens.map { Int32($0) })),
            mainModel: target,
            drafter: drafter,
            mainCache: target.newCache(parameters: speculativeParameters),
            parameters: speculativeParameters,
            blockSize: blockSize)
        var speculativeTokens: [Int] = []
        while let token = speculativeIterator.next() {
            if stopTokenIDs.contains(token) {
                speculativeIterator.discardGeneratedToken()
                break
            }
            speculativeTokens.append(token)
            if speculativeTokens.count >= maxTokens { break }
        }

        XCTAssertEqual(scalarTokens.count, speculativeTokens.count)
        for (index, scalarToken) in scalarTokens.enumerated() {
            XCTAssertEqual(scalarToken, speculativeTokens[index], "token index \(index)")
        }
        XCTAssertEqual(speculativeIterator.proposedDraftTokens, 2)
        XCTAssertEqual(speculativeIterator.acceptedDraftTokens, 2)
        XCTAssertNil(speculativeIterator.passthroughReason)
    }

    // MARK: - ScalarServingInCheckpointMTPStartupVerdict.machineReadableFields()
    //
    // The startup gate above proves the verdict's NUMBERS are correct. These tests prove those
    // numbers actually reach the scalar startup line (`FastMLXServe.swift`'s `startupLine`), which
    // is the only capturable evidence a real-weight `--qwen4exp-mtp` fleet serve produces. Without
    // this, a green startup proves nothing beyond "did not crash" -- see this increment's task
    // background.

    /// Every field present, every value distinct and non-equal to every other field's value, so a
    /// field-ordering or copy-paste swap (e.g. `proposed` vs `accepted`) would fail this test even
    /// though both fields are individually present. Asserts on individual `contains("key=value")`
    /// scalars, never on the whole rendered string, per this repository's anti-Myers-diff
    /// constraint for large-collection/string equality.
    func testMachineReadableFieldsRendersEveryFieldWithDistinctValues() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .converted,
            revision: "abc123",
            sourceKeyCount: 11,
            promptTokenCount: 22,
            generatedTokenCount: 33,
            proposedDraftTokens: 44,
            acceptedDraftTokens: 55,
            drafterActiveBytesDelta: 66,
            drafterCacheBytesDelta: 77)

        let fields = verdict.machineReadableFields()

        XCTAssertTrue(fields.contains("in_checkpoint_mtp=true"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_namespace=converted"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_revision=abc123"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_source_key_count=11"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_prompt_token_count=22"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_generated_token_count=33"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_proposed_draft_tokens=44"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_accepted_draft_tokens=55"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_drafter_active_bytes_delta=66"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_drafter_cache_bytes_delta=77"))
    }

    /// ANTI-VACUITY, THE CASE THAT MATTERS MOST: a run where every draft proposal was rejected
    /// (`acceptedDraftTokens == 0`) -- or where the run made zero proposals at all
    /// (`proposedDraftTokens == 0`) -- must still render both fields explicitly as `=0`, not omit
    /// them. Per `ScalarServingInCheckpointMTPStartupVerdict`'s own doc comment, `accepted == 0` is
    /// a legitimate, meaningful outcome the gate can genuinely produce (see
    /// `testVerifyStartupEquivalenceSucceedsWhenEveryDraftProposalIsRejected` above); if the
    /// startup line silently dropped zero-valued fields, that legitimate outcome would be
    /// indistinguishable from the field never having been wired up at all -- exactly the gap this
    /// increment closes.
    func testMachineReadableFieldsRendersZeroValuedDraftFieldsExplicitly() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .official,
            revision: "rev0",
            sourceKeyCount: 4,
            promptTokenCount: 3,
            generatedTokenCount: 1,
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            drafterActiveBytesDelta: 0,
            drafterCacheBytesDelta: 0)

        let fields = verdict.machineReadableFields()

        XCTAssertTrue(fields.contains("in_checkpoint_mtp_proposed_draft_tokens=0"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_accepted_draft_tokens=0"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_drafter_active_bytes_delta=0"))
        XCTAssertTrue(fields.contains("in_checkpoint_mtp_drafter_cache_bytes_delta=0"))
    }

    /// Namespace separation: every emitted key carries the `in_checkpoint_mtp` prefix, never the
    /// bare `fit_` or `exact_mtp` prefixes the same startup line already emits from
    /// `ServingFitDecision.machineReadableFields()` and
    /// `ExactQwen35MTPServeStartupReport.machineReadableFields()` -- a collision there would let
    /// one gate's evidence silently overwrite or be mistaken for another's when a human or script
    /// greps the line.
    func testMachineReadableFieldsKeysDoNotCollideWithOtherStartupLineNamespaces() {
        let verdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: .official,
            revision: "rev0",
            sourceKeyCount: 4,
            promptTokenCount: 3,
            generatedTokenCount: 1,
            proposedDraftTokens: 2,
            acceptedDraftTokens: 1,
            drafterActiveBytesDelta: 5,
            drafterCacheBytesDelta: 6)

        let fields = verdict.machineReadableFields()

        for key in fields.split(separator: " ") {
            XCTAssertTrue(
                key.hasPrefix("in_checkpoint_mtp"),
                "unexpected key outside the in_checkpoint_mtp namespace: \(key)")
            XCTAssertFalse(key.hasPrefix("fit_"), "collided with the fit_ namespace: \(key)")
            XCTAssertFalse(
                key.hasPrefix("exact_mtp"), "collided with the exact_mtp namespace: \(key)")
        }
    }
}
