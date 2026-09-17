import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// Tests for `RouteSwitchingDecoder`'s routing/latching/memory-invariant behavior, using two pure
/// fake decoders (no MLX model) rather than real `CompiledMLXDecoder`/`MLXDecoder` instances — this
/// file only exercises the composition logic `RouteSwitchingDecoder` itself owns (which side each
/// call goes to, when the route latches, when `releaseKVCaches()` fires), not either side's own
/// decode behavior (covered by `CompiledMLXDecoderMLACacheAdmissionTests`/`MLXDecoder*Tests`).
///
/// `RouteSwitchingDecoder` is a struct holding `fast`/`general` BY VALUE, so — mirroring
/// `RecordingDecoder`/`RecordingConstraintDecoder`'s existing idiom in this test target — each fake
/// records into a shared, reference-typed `CallLog` rather than its own stored properties, so calls
/// made on the copy living inside the composed decoder are observable back in the test.
final class RouteSwitchingDecoderTests: XCTestCase {

    // MARK: - 1. Greedy request routes entirely to fast

    func testGreedyRequestRoutesPrefillAndStepsToFastOnly() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        let baseline = log.events.count

        let first = try decoder.prefill([1, 2, 3])
        let second = try decoder.step(last: first)
        let third = try decoder.step(last: second)

        XCTAssertEqual(log.eventsSince(baseline), ["fast.prefill", "fast.step", "fast.step"], "greedy request must route only to fast")
        XCTAssertEqual([first, second, third], [100, 101, 102], "fast route must produce fast's tokens")
        XCTAssertEqual(log.releaseKVCachesCount, 0, "fast never handed off to general, so no release should fire")
    }

    // MARK: - 2. Sampled request routes entirely to general

    func testSampledRequestRoutesPrefillAndStepsToGeneralOnly() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.reset()
        decoder.setSampling(.sampled(temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 42))
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        let baseline = log.events.count

        let first = try decoder.prefill([1, 2, 3])
        let second = try decoder.step(last: first)

        XCTAssertEqual(log.eventsSince(baseline), ["general.prefill", "general.step"], "non-greedy sampling must route to general")
        XCTAssertEqual([first, second], [200, 201], "general route must produce general's tokens")
        XCTAssertEqual(log.fastCallCount, 0, "fast must never be touched (prefill/step) by a sampled request")
    }

    // MARK: - 3. Penalties: non-empty routes general, empty/neutral routes fast

    func testNonEmptyPenaltiesAloneRoutesToGeneral() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(DecoderPenalties(presencePenalty: 1.5))
        decoder.setResponseFormatConstraint(nil)
        let baseline = log.events.count

        _ = try decoder.prefill([1])

        XCTAssertEqual(log.eventsSince(baseline), ["general.prefill"], "non-empty penalties alone must route to general even with greedy sampling")
    }

    /// `DecoderPenalties.isEmpty` treats `.none` (the all-`nil` default) as empty/neutral — the only
    /// value guaranteed empty without relying on `repetitionPenalty`'s `1.0`-is-neutral special case,
    /// which this test does not need to exercise (that's `RepetitionPenaltyNeutralElementTests`'s
    /// job one layer down).
    func testEmptyPenaltiesAloneRoutesToFast() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))
        XCTAssertTrue(DecoderPenalties.none.isEmpty, "precondition: .none must be the neutral/empty penalty set this test relies on")

        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        let baseline = log.events.count

        _ = try decoder.prefill([1])

        XCTAssertEqual(log.eventsSince(baseline), ["fast.prefill"], "empty penalties must not force the general route")
    }

    // MARK: - 4. Response-format constraint alone routes general

    func testResponseFormatConstraintAloneRoutesToGeneral() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(NoOpLogitProcessor())
        let baseline = log.events.count

        _ = try decoder.prefill([1])

        XCTAssertEqual(log.eventsSince(baseline), ["general.prefill"], "a non-nil response-format constraint alone must route to general")
    }

    // MARK: - 5. Memory invariant: release fires exactly on a genuine fast -> general handoff

    func testReleaseFiresOnceOnFastToGeneralHandoffAndNotAgainOnRepeatedGeneral() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        // Request 1: greedy, fast allocates.
        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([1])
        _ = try decoder.step(last: 100)

        // Request 2: sampled, first handoff to general -- must release before general.prefill.
        decoder.reset()
        decoder.setSampling(.sampled(temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil))
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([2])

        XCTAssertEqual(log.releaseKVCachesCount, 1, "exactly one release must fire on the fast -> general handoff")
        guard let releaseIndex = log.events.firstIndex(of: "fast.releaseKVCaches"),
            let generalPrefillIndex = log.events.firstIndex(of: "general.prefill")
        else {
            XCTFail("expected both a release and a general.prefill event in the log")
            return
        }
        XCTAssertLessThan(
            releaseIndex, generalPrefillIndex,
            "release must happen BEFORE general.prefill, per the type's MEMORY INVARIANT doc comment")

        // Request 3: another sampled request -- fast never reallocated since the release, so no
        // additional release should fire.
        decoder.reset()
        decoder.setSampling(.sampled(temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil))
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([3])

        XCTAssertEqual(log.releaseKVCachesCount, 1, "a second consecutive general request must not trigger an additional release")

        // Request 4: greedy again -- fast reallocates.
        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([4])

        // Request 5: sampled again -- second genuine handoff, second release.
        decoder.reset()
        decoder.setSampling(.sampled(temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil))
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([5])

        XCTAssertEqual(log.releaseKVCachesCount, 2, "a second genuine fast -> general handoff must trigger a second release")
    }

    // MARK: - 6. General-first request never allocates fast, so no release

    func testGeneralRequestFirstNeverAllocatesFastAndTriggersNoRelease() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.reset()
        decoder.setSampling(.sampled(temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil))
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([1])
        _ = try decoder.step(last: 200)

        XCTAssertEqual(log.fastCallCount, 0, "fast must never be invoked when the very first request needs general")
        XCTAssertEqual(log.releaseKVCachesCount, 0, "fast never allocated, so releaseKVCaches must not fire")
    }

    // MARK: - 7. prefillWithLogprob/stepWithLogprob always route general

    func testPrefillWithLogprobAlwaysRoutesGeneralEvenWithGreedyPendingAndReleasesIfFastAllocated() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        // Fast allocates on an ordinary greedy request first.
        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        _ = try decoder.prefill([1])

        // Next request is a logprob request, still with greedy pending -- must still route general
        // and must release fast's KV storage first.
        decoder.reset()
        decoder.setSampling(.greedy)
        decoder.setPenalties(.none)
        decoder.setResponseFormatConstraint(nil)
        let (firstToken, _) = try decoder.prefillWithLogprob([2], topN: 3)
        let (secondToken, _) = try decoder.stepWithLogprob(last: firstToken, topN: 3)

        XCTAssertEqual(firstToken, 200, "prefillWithLogprob must always route to general")
        XCTAssertEqual(secondToken, 201, "stepWithLogprob following a general-latched prefillWithLogprob must route to general")
        XCTAssertEqual(log.releaseKVCachesCount, 1, "fast had allocated, so the logprob prefill must still trigger the release")
        guard let releaseIndex = log.events.firstIndex(of: "fast.releaseKVCaches"),
            let logprobPrefillIndex = log.events.firstIndex(of: "general.prefillWithLogprob")
        else {
            XCTFail("expected both a release and a general.prefillWithLogprob event in the log")
            return
        }
        XCTAssertLessThan(releaseIndex, logprobPrefillIndex, "release must happen before general.prefillWithLogprob")
    }

    // MARK: - 8. reset() clears pending state

    func testResetClearsPendingSamplingSoASubsequentPrefillRoutesFast() throws {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        decoder.setSampling(.sampled(temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil))
        decoder.reset()
        let token = try decoder.prefill([1])

        // `RouteSwitchingDecoder.reset()` calls `fast.reset()` before `general.reset()` -- see its
        // own body.
        XCTAssertEqual(log.events, ["fast.reset", "general.reset", "fast.prefill"], "reset() must clear pending sampling so the next prefill (with nothing re-set) routes fast")
        XCTAssertEqual(token, 100, "post-reset prefill with no pending state must use the fast route")
    }

    // MARK: - 9. Capability surface mirrors general

    func testCapabilitySurfaceMirrorsGeneral() {
        let trueLog = CallLog()
        let trueDecoder = RouteSwitchingDecoder(
            fast: FakeFast(log: trueLog), general: FakeGeneral(log: trueLog))
        XCTAssertTrue(trueDecoder.supportsSampling, "supportsSampling must mirror general's (true)")
        XCTAssertTrue(trueDecoder.supportsPenalties, "supportsPenalties must mirror general's (true)")
        XCTAssertTrue(
            trueDecoder.supportsResponseFormatConstraint,
            "supportsResponseFormatConstraint must mirror general's (true)")

        let falseLog = CallLog()
        let falseGeneral = FakeGeneral(log: falseLog)
        falseGeneral.capabilities = (false, false, false)
        let falseDecoder = RouteSwitchingDecoder(fast: FakeFast(log: falseLog), general: falseGeneral)
        XCTAssertFalse(falseDecoder.supportsSampling, "supportsSampling must mirror general's (false)")
        XCTAssertFalse(falseDecoder.supportsPenalties, "supportsPenalties must mirror general's (false)")
        XCTAssertFalse(
            falseDecoder.supportsResponseFormatConstraint,
            "supportsResponseFormatConstraint must mirror general's (false)")
    }

    // MARK: - 10. Setters forward to general regardless of pending route

    func testSettersForwardToGeneralRegardlessOfRoute() {
        let log = CallLog()
        var decoder = RouteSwitchingDecoder(fast: FakeFast(log: log), general: FakeGeneral(log: log))

        let sampling = DecoderSampling.sampled(temperature: 0.5, topP: 0.8, topK: 10, minP: nil, seed: 7)
        decoder.setSampling(sampling)
        XCTAssertEqual(log.lastSampling, sampling, "setSampling must forward the exact value to general")

        let penalties = DecoderPenalties(presencePenalty: 1.2, frequencyPenalty: nil, repetitionPenalty: 1.1)
        decoder.setPenalties(penalties)
        XCTAssertEqual(log.lastPenalties, penalties, "setPenalties must forward the exact value to general")

        decoder.setResponseFormatConstraint(NoOpLogitProcessor())
        XCTAssertTrue(log.lastConstraintWasNonNil == true, "setResponseFormatConstraint(non-nil) must forward a non-nil value to general")

        decoder.setResponseFormatConstraint(nil)
        XCTAssertTrue(log.lastConstraintWasNonNil == false, "setResponseFormatConstraint(nil) must forward nil to general")
    }
}

// MARK: - Test doubles

/// A no-op `LogitProcessor`, mirroring `InferenceActorResponseFormatConstraintTests`'s
/// `NoOpLogitProcessor` — never actually invoked by these tests (`RouteSwitchingDecoder` only
/// threads the constraint through to `general.setResponseFormatConstraint`/routing decisions, never
/// calls into the processor itself), only constructed to prove a non-nil value drives routing.
private struct NoOpLogitProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {}
}

/// Shared, reference-typed call log both fakes below record into. `RouteSwitchingDecoder` (and its
/// stored `fast`/`general`) are value types copied around by the test/actor boundary, so — mirroring
/// `RecordingDecoderState`/`ConstraintCallRecorder`'s existing idiom in this test target — a plain
/// stored property on either fake would not be observable back in the test after being copied into
/// the composed decoder; this class is what both sides mutate and the test reads.
private final class CallLog {
    private(set) var events: [String] = []
    private(set) var releaseKVCachesCount = 0
    private(set) var fastCallCount = 0
    private(set) var lastSampling: DecoderSampling?
    private(set) var lastPenalties: DecoderPenalties?
    private(set) var lastConstraintWasNonNil: Bool?

    private var fastTokenCounter = 100
    private var generalTokenCounter = 200

    func record(_ event: String) {
        events.append(event)
    }

    /// For `fast.prefill`/`fast.step` only (NOT `fast.reset`) -- `fastCallCount` is used by tests
    /// asserting "fast was never touched by this decode", and `reset()` unconditionally resets BOTH
    /// sides regardless of route, so counting it here would make that assertion fail for reasons
    /// unrelated to routing.
    func recordFastCall(_ event: String) {
        fastCallCount += 1
        events.append(event)
    }

    func recordRelease() {
        releaseKVCachesCount += 1
        events.append("fast.releaseKVCaches")
    }

    func recordSampling(_ sampling: DecoderSampling) {
        lastSampling = sampling
    }

    func recordPenalties(_ penalties: DecoderPenalties) {
        lastPenalties = penalties
    }

    func recordConstraint(nonNil: Bool) {
        lastConstraintWasNonNil = nonNil
    }

    /// Next distinct token for the fast route (100, 101, 102, ...).
    func nextFastToken() -> Int {
        defer { fastTokenCounter += 1 }
        return fastTokenCounter
    }

    /// Next distinct token for the general route (200, 201, 202, ...).
    func nextGeneralToken() -> Int {
        defer { generalTokenCounter += 1 }
        return generalTokenCounter
    }

    /// Events recorded strictly AFTER `baseline` (an earlier `events.count` snapshot) -- lets a test
    /// assert an exact call sequence for the routing calls under test while ignoring the
    /// `reset()`/`setSampling`/etc. setup calls that necessarily precede them (per the "reset ->
    /// setSampling -> ... -> prefill" shape every request in this file follows).
    func eventsSince(_ baseline: Int) -> [String] {
        Array(events.dropFirst(baseline))
    }
}

/// Pure fake `KVCacheReleasingDecoder`: no MLX model, just records calls into `log` and returns
/// distinct incrementing tokens (100, 101, 102, ...) so tests can tell fast-route output apart from
/// general-route output (200, 201, ...) without inspecting the call log.
private struct FakeFast: KVCacheReleasingDecoder {
    let log: CallLog

    mutating func prefill(_ promptTokens: [Int]) throws -> Int {
        log.recordFastCall("fast.prefill")
        return log.nextFastToken()
    }

    mutating func step(last: Int) throws -> Int {
        log.recordFastCall("fast.step")
        return log.nextFastToken()
    }

    mutating func reset() {
        log.record("fast.reset")
    }

    mutating func releaseKVCaches() {
        log.recordRelease()
    }
}

/// Pure fake `Decoder & LogprobDecoding`: no MLX model, records calls into `log`, returns distinct
/// incrementing tokens (200, 201, ...), and records the last sampling/penalties/constraint it was
/// configured with so setter-forwarding tests can assert on the exact value received.
/// `capabilities` defaults to all-`true` (this fake stands in for a fully-capable `MLXDecoder`); test
/// 9 flips it to all-`false` to prove `RouteSwitchingDecoder`'s capability surface actually reads
/// through rather than being hardcoded `true`.
private final class FakeGeneral: Decoder, LogprobDecoding {
    let log: CallLog
    var capabilities: (sampling: Bool, penalties: Bool, constraint: Bool) = (true, true, true)

    init(log: CallLog) {
        self.log = log
    }

    var supportsSampling: Bool { capabilities.sampling }
    var supportsPenalties: Bool { capabilities.penalties }
    var supportsResponseFormatConstraint: Bool { capabilities.constraint }

    func prefill(_ promptTokens: [Int]) throws -> Int {
        log.record("general.prefill")
        return log.nextGeneralToken()
    }

    func step(last: Int) throws -> Int {
        log.record("general.step")
        return log.nextGeneralToken()
    }

    func reset() {
        log.record("general.reset")
    }

    func setSampling(_ sampling: DecoderSampling) {
        log.recordSampling(sampling)
    }

    func setPenalties(_ penalties: DecoderPenalties) {
        log.recordPenalties(penalties)
    }

    func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        log.recordConstraint(nonNil: constraint != nil)
    }

    func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        log.record("general.prefillWithLogprob")
        let token = log.nextGeneralToken()
        return (token, DecodedTokenLogprob(tokenID: token, logprob: Float(-token)))
    }

    func stepWithLogprob(
        last: Int, topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        log.record("general.stepWithLogprob")
        let token = log.nextGeneralToken()
        return (token, DecodedTokenLogprob(tokenID: token, logprob: Float(-token)))
    }
}
