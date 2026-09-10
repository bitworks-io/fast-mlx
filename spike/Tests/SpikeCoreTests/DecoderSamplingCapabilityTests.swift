import XCTest
@testable import SpikeCore

/// Tests for the `Decoder.supportsSampling`/`supportsPenalties` capability guard added to
/// `InferenceActor.generateBounded`, per
/// `docs/task-inbox/2026-09-10-compiled-fp16-route-silently-downgrades-sampled-to-greedy-DEFECT.md`.
/// Before this guard, a `.sampled`/penalized request against a decoder that silently no-ops
/// `setSampling`/`setPenalties` (e.g. `ScriptedDecoder`, `CompiledMLXDecoder`) decoded greedy while
/// being reported as sampled — no error, no observable difference. These tests assert the SPECIFIC
/// refusal case (not merely "it threw") and that the guard does not over-refuse supported requests
/// or brick the actor for subsequent requests.
final class DecoderSamplingCapabilityTests: XCTestCase {

    // MARK: - Refusals

    /// R1: `.sampled` + a decoder reporting `supportsSampling == false` (`ScriptedDecoder`, taking
    /// the protocol-extension default) is refused with the specific error case.
    func testSampledRequestAgainstNonSupportingDecoderThrowsSamplingUnsupported() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                sampling: .sampled(
                    temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil)
            ) { _ in .continueGeneration }
            XCTFail("expected samplingUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .samplingUnsupportedByDecoder)
        }
    }

    /// R2: non-empty penalties + a decoder reporting `supportsPenalties == false` is refused with
    /// the specific error case.
    func testNonEmptyPenaltiesAgainstNonSupportingDecoderThrowsPenaltiesUnsupported() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                penalties: DecoderPenalties(presencePenalty: 1.5)
            ) { _ in .continueGeneration }
            XCTFail("expected penaltiesUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .penaltiesUnsupportedByDecoder)
        }
    }

    // MARK: - Happy-path controls

    /// C1: a decoder that reports `supportsSampling == true` AND genuinely records the sampling it
    /// receives admits a `.sampled` request and generates normally. A double that claims a
    /// capability and ignores the argument would be the same defect one layer down — this makes the
    /// `true` verifiable by asserting the recorded value equals what was passed.
    func testSupportingDecoderAdmitsSampledRequestAndReceivesIt() async throws {
        let recorder = RecordingDecoder(script: [5, 6, 2], eos: 2)
        // Capture the shared state reference BEFORE `recorder` is transferred into the actor via
        // `sending` below — reading `recorder` itself afterward would be a use-after-send.
        let state = recorder.state
        let actor = InferenceActor(decoder: recorder)
        let requestedSampling = DecoderSampling.sampled(
            temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 42)

        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            sampling: requestedSampling
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }

        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
        // `generateBounded`'s own `defer` calls `setSampling(.greedy)` on the way out (restoring
        // the decoder's rest state after every request — including a successful one), so the
        // FIRST recorded call is this request's own value; `lastSampling` would observe that
        // trailing reset instead. See `receivedSamplings`'s doc comment.
        XCTAssertEqual(state.receivedSamplings.first, requestedSampling)
    }

    /// C2: `.greedy` (the default) against a NON-supporting decoder still admits and generates.
    /// Catches a guard that degenerates into "refuse this decoder for everything" instead of only
    /// refusing what it cannot honor.
    func testGreedyRequestAgainstNonSupportingDecoderStillAdmits() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            sampling: .greedy
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }

        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    /// C3: the documented NEUTRAL penalty values both HF presets for the deployed model send
    /// (`presencePenalty: 0, repetitionPenalty: 1.0`) must ADMIT against a non-supporting decoder.
    /// `DecoderPenalties.isEmpty` treats `1` as `repetitionPenalty`'s neutral element (it is
    /// multiplicative) — refusing this would 400 production traffic that requests no penalty at
    /// all.
    func testNeutralPenaltyValuesAgainstNonSupportingDecoderStillAdmit() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            penalties: DecoderPenalties(presencePenalty: 0, repetitionPenalty: 1.0)
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }

        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    // MARK: - Regression control

    /// C4: after a REFUSED request (the R1 shape), the SAME `InferenceActor` must still
    /// successfully serve a subsequent `.greedy` request. This is what proves the new guards sit
    /// BEFORE `boundedGenerationActive = true` (and before `decoder.reset()`/`setSampling`/
    /// `setPenalties`, whose `defer` clearing that flag is registered after them) — a refusal must
    /// never leave the actor's reentrancy flag stuck `true`, which would brick every subsequent
    /// request on this same (process-lifetime, per `ScalarServingBackend`) actor with
    /// `.generationAlreadyActive`. Do not omit this test.
    func testActorRemainsUsableAfterARefusedSampledRequest() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                sampling: .sampled(
                    temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil)
            ) { _ in .continueGeneration }
            XCTFail("expected samplingUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .samplingUnsupportedByDecoder)
        }

        // The same actor, a subsequent ordinary greedy request, must succeed — not
        // `.generationAlreadyActive`.
        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }
        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    // MARK: - Mismatched-capability controls (closes the "which flag does each guard read?" gap)

    /// C5a: a decoder reporting `supportsSampling == true` but `supportsPenalties == false` admits
    /// a `.sampled` request and generates normally. Every decoder used above has the two flags
    /// EQUAL to each other (`ScriptedDecoder`: false/false, `RecordingDecoder`: true/true), so
    /// R1/R2/C1-C4 cannot tell which flag the SAMPLING guard actually reads: swapping it to read
    /// `supportsPenalties` instead of `supportsSampling` would leave every one of those tests
    /// green. This decoder's `supportsPenalties == false` makes that swap observable — if the
    /// sampling guard wrongly consulted `supportsPenalties`, this request would be refused instead
    /// of admitted. See `CapabilityMismatchedDecoder`'s doc comment.
    func testSampledRequestAgainstSamplingOnlyDecoderAdmits() async throws {
        let actor = InferenceActor(
            decoder: CapabilityMismatchedDecoder(
                script: [5, 6, 2], eos: 2, supportsSampling: true, supportsPenalties: false))

        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            sampling: .sampled(
                temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil)
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }

        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }

    /// C5b: the SAME sampling-only decoder (`supportsSampling == true`, `supportsPenalties ==
    /// false`) refuses non-empty penalties with the SPECIFIC `penaltiesUnsupportedByDecoder` case,
    /// not merely "it threw". If the PENALTIES guard wrongly consulted `supportsSampling` (`true`
    /// on this decoder) instead of `supportsPenalties`, it would admit instead of refuse — this is
    /// the mirror half of C5a's swap check, and together they pin each guard to its OWN flag
    /// rather than to "some capability flag on the decoder".
    func testPenaltiesAgainstSamplingOnlyDecoderThrowsPenaltiesUnsupported() async throws {
        let actor = InferenceActor(
            decoder: CapabilityMismatchedDecoder(
                script: [5, 6, 2], eos: 2, supportsSampling: true, supportsPenalties: false))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                penalties: DecoderPenalties(presencePenalty: 1.5)
            ) { _ in .continueGeneration }
            XCTFail("expected penaltiesUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .penaltiesUnsupportedByDecoder)
        }
    }

    /// C6a: the MIRRORED penalties-only decoder (`supportsSampling == false`, `supportsPenalties
    /// == true`). Not required to catch the single-field-swap mutations M6/M7 — C5a/C5b alone
    /// already turn red under those (see the task's mutation table). What this pair additionally
    /// closes: a guard that ANDs/ORs both flags together instead of reading only its own (e.g.
    /// "refuse sampling only if BOTH flags say unsupported") would pass every test above INCLUDING
    /// C5a/C5b, because C5a/C5b only ever probe the true/false diagonal, where such a
    /// composed-condition bug happens to coincide with correct behavior. Only a false/true decoder
    /// exposes that composition bug, and only by checking the OPPOSITE admit/refuse pairing from
    /// C5a/C5b: here sampling must be REFUSED (`supportsSampling` is false) and penalties must be
    /// ADMITTED (`supportsPenalties` is true).
    func testSampledRequestAgainstPenaltiesOnlyDecoderThrowsSamplingUnsupported() async throws {
        let actor = InferenceActor(
            decoder: CapabilityMismatchedDecoder(
                script: [5, 6, 2], eos: 2, supportsSampling: false, supportsPenalties: true))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                sampling: .sampled(
                    temperature: 0.7, topP: 1, topK: nil, minP: nil, seed: nil)
            ) { _ in .continueGeneration }
            XCTFail("expected samplingUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .samplingUnsupportedByDecoder)
        }
    }

    /// C6b: the penalties-only decoder admits non-empty penalties and generates normally — the
    /// other half of C6a's pairing.
    func testPenaltiesAgainstPenaltiesOnlyDecoderAdmits() async throws {
        let actor = InferenceActor(
            decoder: CapabilityMismatchedDecoder(
                script: [5, 6, 2], eos: 2, supportsSampling: false, supportsPenalties: true))

        let observed = TokenRecorder()
        let summary = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            penalties: DecoderPenalties(presencePenalty: 1.5)
        ) { token in
            await observed.append(token)
            return .continueGeneration
        }

        let got = await observed.values
        XCTAssertEqual(got, [5, 6])
        XCTAssertEqual(summary.finishReason, .endOfSequence)
    }
}

/// Mirrors `InferenceActorTests.TokenRecorder` (kept private to each test file since
/// `InferenceActorTests`'s is itself `private`, so it is not visible here). Needed because
/// `generateBounded`'s consume callback is `@Sendable`, and a captured `var` cannot be mutated
/// from within it directly.
private actor TokenRecorder {
    private var tokens: [Int] = []

    var values: [Int] {
        tokens
    }

    func append(_ token: Int) {
        tokens.append(token)
    }
}

/// Lock-protected recorder backing `RecordingDecoder`, so the last-received sampling/penalties can
/// be read from the test's own context after `RecordingDecoder` (a value type) has crossed the
/// actor boundary via `sending` into an `InferenceActor`. Plain stored properties on
/// `RecordingDecoder` itself would not be readable back out — the actor holds its own copy — so
/// this shared, reference-typed, lock-guarded box is what both sides mutate/read. `setSampling`/
/// `setPenalties` are only ever called from `InferenceActor`'s serialized isolation, and the test
/// only reads after `await`-ing the whole `generateBounded` call to completion, so the accesses
/// are strictly ordered in practice; `@unchecked Sendable` plus the lock is the narrowest way to
/// satisfy the compiler for this test-only cross-isolation read.
private final class RecordingDecoderState: @unchecked Sendable {
    private let lock = NSLock()
    /// Every `setSampling` call this decoder received, in order. `generateBounded` calls
    /// `setSampling` twice per request — once with the caller's requested value before generation,
    /// and once with `.greedy` from its own `defer` on the way out (see `Decoder.reset`/
    /// `InferenceActor.generateBounded`'s doc comments) — so a test asserting "the value this
    /// request was configured with" must read `.first`, not the trailing reset.
    private var _receivedSamplings: [DecoderSampling] = []
    private var _receivedPenalties: [DecoderPenalties] = []

    var receivedSamplings: [DecoderSampling] {
        lock.lock()
        defer { lock.unlock() }
        return _receivedSamplings
    }

    var receivedPenalties: [DecoderPenalties] {
        lock.lock()
        defer { lock.unlock() }
        return _receivedPenalties
    }

    func record(sampling: DecoderSampling) {
        lock.lock()
        defer { lock.unlock() }
        _receivedSamplings.append(sampling)
    }

    func record(penalties: DecoderPenalties) {
        lock.lock()
        defer { lock.unlock() }
        _receivedPenalties.append(penalties)
    }
}

/// Test double: replays a fixed script like `ScriptedDecoder`, but reports both capabilities
/// `true` AND records the last `DecoderSampling`/`DecoderPenalties` it actually received via
/// `setSampling`/`setPenalties`. A double that merely claims a capability without recording (and
/// having the test assert on) what it received would hide the same defect one layer down: an
/// opt-in that lies about applying the configuration it's handed.
private struct RecordingDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    /// Reference type shared with whatever copy of this struct ends up inside the `InferenceActor`
    /// (structs copy on assignment, but this stored class reference does not) — see
    /// `RecordingDecoderState`'s doc comment.
    let state = RecordingDecoderState()

    init(script: [Int], eos: Int) {
        self.script = script
        self.eos = eos
    }

    var supportsSampling: Bool { true }
    var supportsPenalties: Bool { true }

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func step(last: Int) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func reset() { i = 0 }

    mutating func setSampling(_ sampling: DecoderSampling) {
        state.record(sampling: sampling)
    }

    mutating func setPenalties(_ penalties: DecoderPenalties) {
        state.record(penalties: penalties)
    }
}

/// Test double with DELIBERATELY MISMATCHED capabilities — `supportsSampling` and
/// `supportsPenalties` differ from EACH OTHER, unlike every other double in this file
/// (`ScriptedDecoder`: false/false, `RecordingDecoder`: true/true). That equality is exactly what
/// hides the vacuity C5a/C5b/C6a/C6b close: swap the SAMPLING guard in
/// `InferenceActor.generateBounded` to read `supportsPenalties` (or the PENALTIES guard to read
/// `supportsSampling`) and every test that uses only `ScriptedDecoder`/`RecordingDecoder` stays
/// green, because in both of those doubles the two flags carry the same value — the swap is
/// unobservable. This double's two boolean parameters are independent, so a request whose
/// admit/refuse outcome differs depending on WHICH flag is consulted becomes distinguishable.
/// Deliberately does not record received sampling/penalties (unlike `RecordingDecoder`) — C1
/// already covers "the value passed through is honored"; this double exists only to pin which
/// flag gates which guard.
private struct CapabilityMismatchedDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    let supportsSampling: Bool
    let supportsPenalties: Bool

    init(script: [Int], eos: Int, supportsSampling: Bool, supportsPenalties: Bool) {
        self.script = script
        self.eos = eos
        self.supportsSampling = supportsSampling
        self.supportsPenalties = supportsPenalties
    }

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func step(last: Int) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func reset() { i = 0 }
}
