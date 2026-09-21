import Foundation
import XCTest

import HarnessCore
import MLX
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// End-to-end `ScalarServingBackend.start()` wiring tests for `response_format: json_schema`
/// (response-format design stage 3b), mirroring `ScalarServingJSONObjectRequestWiringTests`
/// test-for-test wherever the two formats share a contract (constraint passthrough, the thinking-
/// phase gate, the unresolvable-`</think>` refusal, MTP-route non-forcing), plus capability tests
/// mirroring `ScalarServingBackendJSONObjectCapabilityTests` and a static/constructed check that
/// every OTHER `ServingGenerationBackend` conformer stays `false`.
final class ScalarServingJSONSchemaRequestWiringTests: XCTestCase {

    // MARK: - Fixtures

    /// A minimal, deterministic single-byte classification (every id 0...255 is its own raw byte,
    /// id 256 is EOS) shared by every test in this file — large enough that ANY compiled schema's
    /// literal bytes ("true", "false", digits, property-name characters, structural punctuation)
    /// are reachable, unlike `ScalarServingJSONObjectRequestWiringTests`' tighter 3-id fixture
    /// (json_object's own grammar only ever needs `{`/`}`/EOS to prove masked-vs-unmasked).
    private static func fullByteClassifications() -> [TokenByteClassification] {
        var classifications: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
        classifications.append(.eos) // id 256
        return classifications
    }

    /// A real `ScalarServingJSONSchemaConstraintSupport`, built the SAME way production does
    /// (`loadScalarServingJSONSchemaConstraintSupport(sharing:)`, from an already-built sibling
    /// `ScalarServingJSONObjectConstraintSupport`) — never hand-constructed, so this file also
    /// exercises the sharing wiring itself, not just the capability/masking behavior downstream of
    /// it.
    private func fakeSchemaSupport(thinkEndTokenID: Int?) -> ScalarServingJSONSchemaConstraintSupport {
        let objectSupport = ScalarServingJSONObjectConstraintSupport(
            classifications: Self.fullByteClassifications(), thinkEndTokenID: thinkEndTokenID)
        return loadScalarServingJSONSchemaConstraintSupport(sharing: objectSupport)!
    }

    private func compileSchema(
        _ json: String, name: String = "test_schema", strict: Bool = true
    ) throws -> JSONSchemaResponseFormat {
        let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(json.utf8))
        return try JSONSchemaSubsetCompiler.compile(name: name, schema: raw, strict: strict)
    }

    private func booleanFormat() throws -> JSONSchemaResponseFormat {
        try compileSchema(#"{"type":"boolean"}"#)
    }

    private func rawLogits() -> MLXArray {
        MLXArray([Float](repeating: 0, count: 257)).reshaped([1, 257])
    }

    private func request(
        maxTokens: Int = 8,
        enableThinking: Bool? = nil,
        responseFormat: ServingResponseFormat?
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
        jsonSchemaConstraintSupport: ScalarServingJSONSchemaConstraintSupport?,
        thinksByDefault: Bool = false
    ) -> ScalarServingBackend {
        ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: decoder),
            codec: FixtureSchemaCodec(),
            stopTokenIDs: [256],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
                thinksByDefault: thinksByDefault,
                jsonSchemaConstraintSupport: jsonSchemaConstraintSupport,
                isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true))
    }

    // MARK: - Constraint passthrough + activeFromStart derivation

    func testStartPassesThroughASchemaConstraintActiveFromStartForANonThinkingRequest() async throws {
        let recorder = ConstraintCapturingRecorder()
        let format = try booleanFormat()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [256], recorder: recorder),
            jsonSchemaConstraintSupport: fakeSchemaSupport(thinkEndTokenID: nil),
            thinksByDefault: false)

        let handle = try await backend.start(request(responseFormat: .jsonSchema(format)))
        _ = try await collect(handle.mailbox)

        let processor = try XCTUnwrap(
            recorder.lastConstraint as? JSONSchemaMaskingLogitProcessor,
            "expected the admitted request's constraint to be passed through to the decoder")
        let masked = processor.process(logits: rawLogits()).asArray(Float.self)
        XCTAssertEqual(masked[Int(UInt8(ascii: "t"))], 0, "`t` must be allowed immediately")
        XCTAssertEqual(masked[Int(UInt8(ascii: "x"))], -Float.infinity, "`x` must never be allowed")
    }

    func testStartPassesThroughASchemaConstraintInactiveUntilThinkEndForAThinkingRequest() async throws {
        let recorder = ConstraintCapturingRecorder()
        let format = try booleanFormat()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [256], recorder: recorder),
            jsonSchemaConstraintSupport: fakeSchemaSupport(thinkEndTokenID: 999),
            thinksByDefault: true)

        let handle = try await backend.start(request(enableThinking: true, responseFormat: .jsonSchema(format)))
        _ = try await collect(handle.mailbox)

        let processor = try XCTUnwrap(
            recorder.lastConstraint as? JSONSchemaMaskingLogitProcessor,
            "expected the admitted request's constraint to be passed through to the decoder")
        let raw = rawLogits()
        XCTAssertEqual(
            processor.process(logits: raw).asArray(Float.self), raw.asArray(Float.self),
            "expected the mask to be a no-op before the think-end token for a thinking request")
    }

    // MARK: - Unresolvable </think> refusal

    func testThinkingRequestWithUnresolvableThinkEndTokenIsRefused() async throws {
        let recorder = ConstraintCapturingRecorder()
        let format = try booleanFormat()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [256], recorder: recorder),
            jsonSchemaConstraintSupport: fakeSchemaSupport(thinkEndTokenID: nil),
            thinksByDefault: true)

        do {
            _ = try await backend.start(request(enableThinking: true, responseFormat: .jsonSchema(format)))
            XCTFail("expected an invalidRequest refusal")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                return XCTFail("expected .invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    // MARK: - Support absent (MTP-route contract: never forced onto a non-speculative decoder)

    /// A request carrying `response_format: json_schema` against a backend with NO schema support
    /// configured (e.g. this backend's route never attempted/succeeded at loading it — including
    /// every speculative-route configuration, whose `jsonSchemaConstraintSupport` is always `nil`)
    /// is refused with a clean 400, never silently served as free text.
    func testRequestIsRefusedWhenNoSchemaSupportIsConfigured() async throws {
        let recorder = ConstraintCapturingRecorder()
        let format = try booleanFormat()
        let backend = makeBackend(
            decoder: ConstraintCapturingDecoder(script: [256], recorder: recorder),
            jsonSchemaConstraintSupport: nil)

        do {
            _ = try await backend.start(request(responseFormat: .jsonSchema(format)))
            XCTFail("expected an invalidRequest refusal")
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                return XCTFail("expected .invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
        XCTAssertNil(recorder.lastConstraint)
    }
}

// MARK: - ScalarServingBackend capability composition

private func makeCapabilityTestBackend(
    jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport? = nil,
    jsonSchemaConstraintSupport: ScalarServingJSONSchemaConstraintSupport?,
    isNonSpeculativeScalarRoute: Bool,
    decoderSupportsResponseFormatConstraint: Bool
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: ScriptedDecoder(script: [99], eos: 99)),
        codec: FixtureSchemaCodec(),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            jsonObjectConstraintSupport: jsonObjectConstraintSupport,
            jsonSchemaConstraintSupport: jsonSchemaConstraintSupport,
            isNonSpeculativeScalarRoute: isNonSpeculativeScalarRoute,
            decoderSupportsResponseFormatConstraint: decoderSupportsResponseFormatConstraint))
}

final class ScalarServingBackendJSONSchemaCapabilityTests: XCTestCase {
    private func fakeSupport() -> ScalarServingJSONSchemaConstraintSupport {
        let objectSupport = ScalarServingJSONObjectConstraintSupport(
            classifications: [.eos], thinkEndTokenID: nil)
        return loadScalarServingJSONSchemaConstraintSupport(sharing: objectSupport)!
    }

    /// Mirrors `ScalarServingBackendJSONObjectCapabilityTests
    /// .testCapabilityTrueOnlyWhenSupportPresentAndRouteIsNonSpeculative`.
    func testCapabilityTrueOnlyWhenSupportPresentAndRouteIsNonSpeculative() {
        XCTAssertTrue(
            makeCapabilityTestBackend(
                jsonSchemaConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONSchemaResponseFormat)
    }

    func testCapabilityFalseWhenDecoderDoesNotSupportConstraintEvenWithRouteAndSupportPresent() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonSchemaConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: false
            ).supportsJSONSchemaResponseFormat)
    }

    /// Every SPECULATIVE/compiled route stays `false` even when the schema support was built — the
    /// response-format design's "MTP eligibility cannot see this processor" rule, applying to
    /// `json_schema` identically to `json_object`.
    func testCapabilityFalseWhenRouteIsSpeculativeEvenWithSupport() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonSchemaConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: false,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONSchemaResponseFormat)
    }

    func testCapabilityFalseWhenSupportIsNilEvenOnTheScalarRoute() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonSchemaConstraintSupport: nil, isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONSchemaResponseFormat)
    }

    /// Every EXISTING construction site (predating this feature) leaves the field at its default —
    /// must compute the SAME `false` capability as before this feature existed.
    func testDefaultConfigurationCapabilityIsFalse() {
        let backend = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: ScriptedDecoder(script: [99], eos: 99)),
            codec: FixtureSchemaCodec(),
            stopTokenIDs: [99],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))
        XCTAssertFalse(backend.supportsJSONSchemaResponseFormat)
    }

    /// The two format capabilities are INDEPENDENT booleans, not one flag driving both: a backend
    /// with ONLY `json_object` support configured must not also report `json_schema` support, and
    /// vice versa — proves `supportsJSONSchemaResponseFormat` reads its OWN configuration field
    /// (`jsonSchemaConstraintSupport`), not `jsonObjectConstraintSupport`.
    func testJSONObjectAndJSONSchemaCapabilitiesAreIndependent() {
        let objectOnlySupport = ScalarServingJSONObjectConstraintSupport(
            classifications: [.eos], thinkEndTokenID: nil)
        let objectOnlyBackend = makeCapabilityTestBackend(
            jsonObjectConstraintSupport: objectOnlySupport,
            jsonSchemaConstraintSupport: nil,
            isNonSpeculativeScalarRoute: true,
            decoderSupportsResponseFormatConstraint: true)
        XCTAssertTrue(objectOnlyBackend.supportsJSONObjectResponseFormat)
        XCTAssertFalse(objectOnlyBackend.supportsJSONSchemaResponseFormat)

        let schemaOnlyBackend = makeCapabilityTestBackend(
            jsonObjectConstraintSupport: nil,
            jsonSchemaConstraintSupport: fakeSupport(),
            isNonSpeculativeScalarRoute: true,
            decoderSupportsResponseFormatConstraint: true)
        XCTAssertFalse(schemaOnlyBackend.supportsJSONObjectResponseFormat)
        XCTAssertTrue(schemaOnlyBackend.supportsJSONSchemaResponseFormat)
    }
}

// MARK: - Every other backend stays false (response-format design "fail-open guard" finding)

/// A minimal `ExactQwen35MTPServingRunner` double that is never actually reached by these
/// capability-only tests (they read `supportsJSONSchemaResponseFormat` directly, without ever
/// calling `start`).
private struct UnreachableMTPRunner: ExactQwen35MTPServingRunner {
    struct Unreachable: Error {}
    var binding: QwenMTPArtifactBinding? { nil }
    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        throw Unreachable()
    }
}

final class ResponseFormatCapabilityDefaultsAcrossBackendsTests: XCTestCase {
    /// The MTP/speculative backend never overrides `ServingGenerationBackend
    /// .supportsJSONSchemaResponseFormat` (grep-verified: no override exists in
    /// `ExactQwen35MTPServingBackend.swift`), so it stays `false` via the protocol extension
    /// default — this constructs a REAL instance (lightweight fixture doubles: a scalar fallback
    /// backed by `ScriptedDecoder`, a runner that is never invoked) and reads the property directly,
    /// rather than trusting the grep alone.
    func testExactSpeculativeServingBackendReportsNoJSONSchemaOrJSONObjectSupport() throws {
        let scalarFallback = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: ScriptedDecoder(script: [99], eos: 99)),
            codec: FixtureSchemaCodec(),
            stopTokenIDs: [99],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))

        let backend = try ExactQwen35MTPServingBackend(
            launchedModel: "fixture-model",
            enabled: true,
            runner: UnreachableMTPRunner(),
            scalarFallback: scalarFallback,
            scalarFallbackIsolation: .strictlySeparateRawTarget,
            codec: FixtureSchemaCodec(),
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))

        XCTAssertFalse(backend.supportsJSONSchemaResponseFormat)
        XCTAssertFalse(backend.supportsJSONObjectResponseFormat)
    }

    /// `ContinuousServingBackend` is NOT constructed here (its `ContinuousBatchCoordinator`
    /// dependency needs a runtime fixture heavier than a capability-only test warrants) — verified
    /// instead by static inspection: `grep -n "supportsJSONObjectResponseFormat\|
    /// supportsJSONSchemaResponseFormat" ContinuousServingBackend.swift` returns no match, so it
    /// inherits `ServingGenerationBackend`'s `false` default identically to the MTP backend above.
    /// Recorded here (rather than silently omitted) per this stage's task spec: "if such backends
    /// are constructible in tests" — this one was judged not cheaply so, unlike the MTP backend.
    func testContinuousServingBackendCapabilityDefaultIsVerifiedStaticallyNotByConstruction() {
        // Intentionally empty: see the doc comment above for the static verification this records.
    }
}

private final class ConstraintCapturingRecorder: @unchecked Sendable {
    private(set) var lastConstraint: (any LogitProcessor)?
    func record(_ constraint: (any LogitProcessor)?) {
        if let constraint {
            lastConstraint = constraint
        }
    }
}

private struct ConstraintCapturingDecoder: Decoder {
    struct SimulatedMidStreamFailure: Error {}

    let script: [Int]
    let recorder: ConstraintCapturingRecorder
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
            return script.last ?? 0
        }
        defer { i += 1 }
        return script[i]
    }
}

private struct FixtureSchemaCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?, addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureSchemaDetokenizer()
    }
}

private struct FixtureSchemaDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) {}
    mutating func next() -> String? { nil }
}
