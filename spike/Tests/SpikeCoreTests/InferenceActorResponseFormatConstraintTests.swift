import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// A no-op `LogitProcessor`: never called in the refusal-path tests (the capability guard throws
/// BEFORE any processor method runs), only constructed to prove a non-nil value is what triggers
/// the guard.
private struct NoOpLogitProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {}
}

/// `ScriptedDecoder` predates `setResponseFormatConstraint`/`supportsResponseFormatConstraint` and
/// takes the `Decoder` protocol extension's default (no-op / `false`) — exactly the "unlisted
/// conformer" case `InferenceActor.generateBounded`'s guard exists to refuse.
final class InferenceActorResponseFormatConstraintCapabilityGuardTests: XCTestCase {
    func testNonNilConstraintAgainstUnsupportingDecoderIsRefused() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))

        do {
            _ = try await actor.generateBounded(
                promptTokens: [1],
                maxTokens: 10,
                eos: 2,
                responseFormatConstraint: NoOpLogitProcessor()
            ) { _ in .continueGeneration }
            XCTFail("expected responseFormatConstraintUnsupportedByDecoder")
        } catch let error as InferenceActorError {
            XCTAssertEqual(error, .responseFormatConstraintUnsupportedByDecoder)
        }
    }

    /// Control: the SAME decoder, SAME request, with `responseFormatConstraint: nil` (the default)
    /// succeeds — proves the guard fires on the constraint's PRESENCE, not on `ScriptedDecoder`
    /// itself being generally unusable with this API.
    func testNilConstraintAgainstUnsupportingDecoderSucceeds() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        let got = TokenRecorder()
        _ = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2
        ) { token in
            await got.append(token)
            return .continueGeneration
        }
        let values = await got.values
        XCTAssertEqual(values, [5, 6])
    }
}

private actor TokenRecorder {
    private var tokens: [Int] = []
    var values: [Int] { tokens }
    func append(_ token: Int) { tokens.append(token) }
}

/// Test double proving `InferenceActor.generateBounded` actually WIRES a supporting decoder: sets
/// the constraint before generation begins and clears it (back to `nil`) once the request ends —
/// mirroring `setSampling`/`setPenalties`'s existing reset-on-exit contract exactly.
private struct RecordingConstraintDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    /// Reference type so state written by a value-type `Decoder` mutation is observable from the
    /// test after the actor has copied/mutated its own private `var decoder`.
    let recorder: ConstraintCallRecorder

    var supportsResponseFormatConstraint: Bool { true }

    mutating func prefill(_ promptTokens: [Int]) -> Int { defer { i += 1 }; return script[i] }
    mutating func step(last: Int) -> Int { defer { i += 1 }; return script[i] }
    mutating func reset() { i = 0 }
    mutating func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        recorder.record(isSet: constraint != nil)
    }
}

private final class ConstraintCallRecorder: @unchecked Sendable {
    private(set) var calls: [Bool] = []
    func record(isSet: Bool) { calls.append(isSet) }
}

final class InferenceActorResponseFormatConstraintWiringTests: XCTestCase {
    func testSupportingDecoderReceivesConstraintThenClearsItOnExit() async throws {
        let recorder = ConstraintCallRecorder()
        let actor = InferenceActor(
            decoder: RecordingConstraintDecoder(script: [5, 6, 2], eos: 2, recorder: recorder))

        _ = try await actor.generateBounded(
            promptTokens: [1],
            maxTokens: 10,
            eos: 2,
            responseFormatConstraint: NoOpLogitProcessor()
        ) { _ in .continueGeneration }

        // `setResponseFormatConstraint` is called once with a non-nil value (admission) and once
        // with `nil` (the `defer` cleanup) — in that order, exactly like `setSampling`/
        // `setPenalties`'s existing reset-on-exit pattern.
        XCTAssertEqual(recorder.calls, [true, false])
    }
}
