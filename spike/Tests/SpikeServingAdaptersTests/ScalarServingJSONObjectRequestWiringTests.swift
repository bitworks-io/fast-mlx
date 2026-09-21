import Foundation
import XCTest

import MLX
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// End-to-end `ScalarServingBackend.start()` wiring tests for `response_format: json_object`
/// (should-fix #9), independent of `ScalarServingJSONObjectConstraintSupportTests`'s load-time
/// tests and `JSONObjectMaskingLogitProcessorTests`'s processor-level tests: this file drives the
/// backend's ADMISSION path itself, on a fake decoder, to prove the constraint reaches
/// `InferenceActor.generateBounded` and that `activeFromStart` is derived correctly per request.
///
/// MTP-route contract (recorded here rather than only in a doc comment, per should-fix #9's last
/// bullet): a request carrying `response_format` against a speculative-route backend is refused
/// with 400 by `supportsJSONObjectResponseFormat` being `false` — it is NEVER forced onto a
/// non-speculative decoder to make the constraint reachable (see that property's own doc comment
/// in `SpikeServingAdapters.swift`).
final class ScalarServingJSONObjectRequestWiringTests: XCTestCase {

    // MARK: - Fixtures

    /// A real `ScalarServingJSONObjectConstraintSupport` built from a minimal, deterministic
    /// classification: id 0 is `{`, id 1 is `}`, id 2 is EOS — the smallest grammar that still has
    /// a genuine "masked" vs. "unmasked" distinction (unlike an EOS-only table, whose initial state
    /// has NO allowed content token and so cannot distinguish "inactive" from "active but broken").
    private func fakeSupport(thinkEndTokenID: Int?) -> ScalarServingJSONObjectConstraintSupport {
        ScalarServingJSONObjectConstraintSupport(
            classifications: [.bytes([0x7B]), .bytes([0x7D]), .eos],
            thinkEndTokenID: thinkEndTokenID)
    }

    private func rawLogits() -> MLXArray {
        MLXArray([Float](repeating: 0, count: 3)).reshaped([1, 3])
    }

    private func request(
        maxTokens: Int = 4,
        enableThinking: Bool? = nil,
        responseFormat: ServingResponseFormat? = .jsonObject
    ) -> OpenAIChatCompletionRequest {
        OpenAIChatCompletionRequest(
            model: "fixture-model",
            messages: [OpenAIChatMessage(role: .user, text: "hi")],
            maxCompletionTokens: maxTokens,
            temperature: 0,
            choiceCount: 1,
            stream: true,
            stop: [],
            enableThinking: enableThinking,
            responseFormat: responseFormat)
    }

    private func collect(_ mailbox: BoundedDeltaMailbox) async throws -> [ServingResponseDelta] {
        var events: [ServingResponseDelta] = []
        while let event = try await mailbox.next() {
            events.append(event)
        }
        return events
    }

    private func makeBackend(
        decoder: sending any Decoder,
        jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport?,
        thinksByDefault: Bool = false
    ) -> ScalarServingBackend {
        ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: decoder),
            codec: FixtureCodec(),
            stopTokenIDs: [2],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
                thinksByDefault: thinksByDefault,
                jsonObjectConstraintSupport: jsonObjectConstraintSupport,
                isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true))
    }

    // MARK: - Constraint passthrough + activeFromStart derivation

    /// A non-thinking request (`thinksByDefault: false` — `separatesReasoning` is `false`) must
    /// build the constraint `activeFromStart: true`: the mask is already live on the very first
    /// token, observable as `process(logits:)` masking away everything but `{` (id 0) immediately,
    /// with no need to sample a think-end token first.
    func testStartPassesThroughAConstraintActiveFromStartForANonThinkingRequest() async throws {
        let recorder = ConstraintCapturingRecorder()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [2], recorder: recorder),
            jsonObjectConstraintSupport: fakeSupport(thinkEndTokenID: nil),
            thinksByDefault: false)

        let handle = try await backend.start(request())
        _ = try await collect(handle.mailbox)

        let processor = try XCTUnwrap(
            recorder.lastConstraint as? JSONObjectMaskingLogitProcessor,
            "expected the admitted request's constraint to be passed through to the decoder")
        XCTAssertTrue(recorder.clearedAfterLastConstraint, "expected the constraint slot to be cleared after the request")
        let masked = processor.process(logits: rawLogits()).asArray(Float.self)
        XCTAssertEqual(masked, [0, -Float.infinity, -Float.infinity], "expected only `{` to be allowed immediately")
    }

    /// A THINKING request (`thinksByDefault: true`, thinking not turned off — `separatesReasoning`
    /// is `true`) must build the constraint `activeFromStart: false`: the mask stays inactive (a
    /// pure passthrough) until `</think>` is sampled, so the reasoning block is never corrupted.
    func testStartPassesThroughAConstraintInactiveUntilThinkEndForAThinkingRequest() async throws {
        let recorder = ConstraintCapturingRecorder()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [2], recorder: recorder),
            jsonObjectConstraintSupport: fakeSupport(thinkEndTokenID: 999),
            thinksByDefault: true)

        let handle = try await backend.start(request(enableThinking: true))
        _ = try await collect(handle.mailbox)

        let processor = try XCTUnwrap(
            recorder.lastConstraint as? JSONObjectMaskingLogitProcessor,
            "expected the admitted request's constraint to be passed through to the decoder")
        XCTAssertTrue(recorder.clearedAfterLastConstraint, "expected the constraint slot to be cleared after the request")
        let raw = rawLogits()
        XCTAssertEqual(
            processor.process(logits: raw).asArray(Float.self), raw.asArray(Float.self),
            "expected the mask to be a no-op before the think-end token for a thinking request")
    }

    // MARK: - Unresolvable </think> refusal

    /// A thinking request against support with NO resolvable `</think>` id (`thinkEndTokenID: nil`)
    /// must be refused as an `invalidRequest` (400-class) rather than silently either never
    /// activating the mask or activating it from token 0 and corrupting the reasoning block.
    func testThinkingRequestWithUnresolvableThinkEndTokenIsRefused() async throws {
        let recorder = ConstraintCapturingRecorder()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [2], recorder: recorder),
            jsonObjectConstraintSupport: fakeSupport(thinkEndTokenID: nil),
            thinksByDefault: true)

        do {
            _ = try await backend.start(request(enableThinking: true))
            XCTFail("expected an invalidRequest refusal")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                return XCTFail("expected .invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    // MARK: - Mid-stream failure never surfaces as success

    /// A constraint that fails PARTWAY through generation (the decoder itself throws once its
    /// script is exhausted early — standing in for `MLXDecoder` throwing a real
    /// `JSONObjectConstraintError` mid-stream, see `MLXDecoderResponseFormatConstraintFailureTests`
    /// for that half of the contract) must reach the client as a failure: `collect` must throw,
    /// and the event stream up to that point must never contain a `.completion` delta.
    func testMidStreamFailureNeverReachesTheClientAsACompletion() async throws {
        let recorder = ConstraintCapturingRecorder()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(
                script: [0, 1], recorder: recorder, failAfterScriptExhausted: true),
            jsonObjectConstraintSupport: fakeSupport(thinkEndTokenID: nil),
            thinksByDefault: false)

        let handle = try await backend.start(request(maxTokens: 10))

        var sawCompletion = false
        do {
            for try await event in AsyncThrowingCollector(mailbox: handle.mailbox) {
                if case .completion = event { sawCompletion = true }
            }
            XCTFail("expected the mailbox to fail rather than drain cleanly")
        } catch {
            // Expected: a mid-stream failure surfaces as a thrown error from the mailbox.
        }
        XCTAssertFalse(sawCompletion, "a mid-stream failure must never be reported as a completion")
    }

    // MARK: - Control: a plain request (no response_format) is untouched by this feature

    /// A request with NO `response_format` must never even build a constraint, and must produce
    /// the SAME token/text stream a request predating this feature would have — proving the new
    /// code path is additive, not a behavior change for every existing caller.
    func testPlainRequestWithoutResponseFormatBuildsNoConstraintAndStreamsNormally() async throws {
        let recorder = ConstraintCapturingRecorder()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [0, 1, 2], recorder: recorder),
            jsonObjectConstraintSupport: fakeSupport(thinkEndTokenID: nil),
            thinksByDefault: false)

        let handle = try await backend.start(request(responseFormat: nil))
        let events = try await collect(handle.mailbox)

        XCTAssertNil(recorder.lastConstraint, "a plain request must never build a constraint")
        let text = events.compactMap { event -> String? in
            if case .text(let piece) = event { return piece }
            return nil
        }.joined()
        XCTAssertEqual(text, "{}")
        let finish = events.compactMap { event -> OpenAIChatFinishReason? in
            if case .completion(let completion) = event { return completion.finishReason }
            return nil
        }.first
        XCTAssertEqual(finish, .stop)
    }
}

/// Adapts `BoundedDeltaMailbox.next()` into an `AsyncSequence` so a mid-drain failure can be
/// observed with a `for try await` loop that also records everything seen BEFORE the throw.
private struct AsyncThrowingCollector: AsyncSequence {
    typealias Element = ServingResponseDelta
    let mailbox: BoundedDeltaMailbox

    struct AsyncIterator: AsyncIteratorProtocol {
        let mailbox: BoundedDeltaMailbox
        mutating func next() async throws -> ServingResponseDelta? {
            try await mailbox.next()
        }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator(mailbox: mailbox) }
}

private final class ConstraintCapturingRecorder: @unchecked Sendable {
    /// The last NON-nil constraint configured. `InferenceActor` clears the slot with `nil` once
    /// the request finishes, so the last call overall is always a clear.
    private(set) var lastConstraint: (any LogitProcessor)?
    /// Whether a clear (`nil`) arrived after the last non-nil constraint.
    private(set) var clearedAfterLastConstraint = false
    func record(_ constraint: (any LogitProcessor)?) {
        if let constraint {
            lastConstraint = constraint
            clearedAfterLastConstraint = false
        } else if lastConstraint != nil {
            clearedAfterLastConstraint = true
        }
    }
}

/// A `Decoder` fixture that supports `setResponseFormatConstraint` and records whatever it is
/// given, so a test can drive the CAPTURED processor directly (proving what `start()` built)
/// without needing a real MLX model. Optionally throws once its script is exhausted, standing in
/// for a constraint violation `MLXDecoder` would otherwise surface as a real thrown error mid-decode
/// (see `MLXDecoderResponseFormatConstraintFailureTests` for that half of the contract in SpikeCore).
private struct ConstraintCapturingDecoder: Decoder {
    struct SimulatedMidStreamFailure: Error {}

    let script: [Int]
    let recorder: ConstraintCapturingRecorder
    var failAfterScriptExhausted: Bool = false
    var i = 0

    var supportsResponseFormatConstraint: Bool { true }

    mutating func prefill(_ promptTokens: [Int]) throws -> Int { try nextScripted() }
    mutating func step(last: Int) throws -> Int { try nextScripted() }
    mutating func reset() { i = 0 }
    mutating func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        recorder.record(constraint)
    }

    private mutating func nextScripted() throws -> Int {
        guard i < script.count else {
            if failAfterScriptExhausted {
                throw SimulatedMidStreamFailure()
            }
            return script.last ?? 0
        }
        defer { i += 1 }
        return script[i]
    }
}

private struct FixtureCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?, addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureDetokenizer()
    }
}

private struct FixtureDetokenizer: ScalarServingDetokenizer {
    private static let pieces: [Int: String] = [0: "{", 1: "}"]
    private var pending: String?

    mutating func append(token: Int) {
        pending = Self.pieces[token]
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}
