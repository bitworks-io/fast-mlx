import Foundation
import XCTest

import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// Increment 2 of `docs/task-inbox/2026-09-09-default-sampling-params-DECISION.md`: wires
/// `--default-sampling generation-config` into the SCALAR serve route only, fail-closed at load.
/// Increment 1 (the pure resolver, `ServingSamplingPolicy.resolve(from:defaults:)`) is already
/// covered by `ServingSamplingPolicyTests.swift`; this file covers the NEW load-time guard
/// (`scalarServingDefaultSamplingDecoderStrategyError`), the isolation invariant that keeps the
/// default confined to the scalar route, and that the resolved defaults are genuinely APPLIED at
/// admission, not merely accepted.
final class DefaultSamplingServeWiringTests: XCTestCase {

    // MARK: - Guard-unit tests, mirroring MTPDecoderBridgeSelectionTests

    /// The refusal: `.generationConfig` + `.compiledFP16` must refuse with the specific case. See
    /// `ScalarServingModelLoadError.defaultSamplingIncompatibleWithCompiledDecoderStrategy`'s doc
    /// comment for why this combination is the DOMINANT resolution for `--default-sampling
    /// generation-config`, not a rare edge case (unlike the MTP guard this mirrors).
    func testDefaultSamplingDecoderStrategyGuardRefusesGenerationConfigWithCompiledFP16() {
        let error = scalarServingDefaultSamplingDecoderStrategyError(
            defaultSampling: .generationConfig, decoderStrategy: .compiledFP16)
        XCTAssertEqual(error, .defaultSamplingIncompatibleWithCompiledDecoderStrategy)
    }

    /// Control A: `.off` (the flag not requested at all) paired with `.compiledFP16` must ADMIT. A
    /// guard reading only `decoderStrategy` (ignoring `defaultSampling` entirely) would refuse this
    /// too, which would break every existing `--compiled` serve that never asked for
    /// `--default-sampling generation-config`. Required alongside Control B: with only one control, a
    /// guard reading a single operand alone still passes.
    func testDefaultSamplingDecoderStrategyGuardAdmitsOffWithCompiledFP16() {
        XCTAssertNil(
            scalarServingDefaultSamplingDecoderStrategyError(
                defaultSampling: .off, decoderStrategy: .compiledFP16))
    }

    /// Control B: `.generationConfig` paired with `.nativeCaches(.fp16)` (NOT `.compiledFP16`) must
    /// ADMIT. A guard reading only `defaultSampling` (ignoring `decoderStrategy` entirely) would
    /// refuse this too, which would break every non-compiled route that opts into the flag. Required
    /// alongside Control A for the same single-operand-coverage reason.
    func testDefaultSamplingDecoderStrategyGuardAdmitsGenerationConfigWithNativeCaches() {
        XCTAssertNil(
            scalarServingDefaultSamplingDecoderStrategyError(
                defaultSampling: .generationConfig, decoderStrategy: .nativeCaches(.fp16)))
    }

    // MARK: - Isolation invariant, pinned BY CONSTRUCTION (not by grep)

    /// `ScalarServingBackendConfiguration.samplingDefaults` defaults to `nil` when omitted, matching
    /// every existing construction site's behavior byte-for-byte.
    func testScalarServingBackendConfigurationSamplingDefaultsDefaultsToNil() {
        let configuration = ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 1,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024))
        XCTAssertNil(configuration.samplingDefaults)
    }

    /// `ScalarServingModelLoadConfiguration.defaultSampling` defaults to `.off` when omitted,
    /// matching every existing construction site's behavior byte-for-byte.
    func testScalarServingModelLoadConfigurationDefaultSamplingDefaultsToOff() {
        let configuration = ScalarServingModelLoadConfiguration(
            launchedModel: "fixture-model",
            modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
            memoryLimitBytes: 64,
            cacheLimitBytes: 16,
            backendConfiguration: ScalarServingBackendConfiguration(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 1,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))
        XCTAssertEqual(configuration.defaultSampling, .off)
    }

    /// The isolation invariant `ScalarServingModelLoadConfiguration.defaultSampling`'s doc comment
    /// claims: `ExactQwen35MTPServeComposition.scalarConfiguration` does NOT forward
    /// `FastMLXServeArguments.defaultSampling`, so the exact-qwen35 route's scalar fallback always
    /// resolves `defaultSampling == .off` regardless of what the composition configuration carries.
    /// Pinned by constructing the REAL composition configuration and reading the REAL computed
    /// property, not by grepping the source for the absence of a forwarded argument.
    func testExactQwen35MTPServeCompositionScalarConfigurationKeepsDefaultSamplingOff() {
        let compositionConfiguration = ExactQwen35MTPServeCompositionConfiguration(
            launchedModel: "fixture-model",
            targetDirectory: URL(fileURLWithPath: "/models/qwen35-target", isDirectory: true),
            drafterDirectory: URL(fileURLWithPath: "/models/qwen35-drafter", isDirectory: true),
            memoryLimitBytes: 64,
            cacheLimitBytes: 16,
            scalarBackendConfiguration: ScalarServingBackendConfiguration(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 1,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))

        XCTAssertEqual(compositionConfiguration.scalarConfiguration.defaultSampling, .off)
    }

    // MARK: - Resolver-delivery: the defaults are actually APPLIED, not merely accepted

    /// A param-less request (every sampling field `nil`) against a backend configured with a
    /// non-`nil` `samplingDefaults` must resolve to `.sampled` carrying the CONFIGURED values, and
    /// the decoder must actually RECEIVE that resolved sampling — asserting only "the request
    /// admitted" would be vacuous, since a dropped default also admits (as `.greedy`). Uses
    /// `RecordingDecoder` (copied from `DecoderSamplingCapabilityTests.swift`), which reports
    /// `supportsSampling == true` AND records what `setSampling` actually received.
    func testParamLessRequestWithSamplingDefaultsResolvesSampledWithConfiguredValuesAtTheDecoder()
        async throws
    {
        let backend = makeDefaultSamplingBackend(
            samplingDefaults: ServingSamplingDefaults(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0))

        let handle = try await backend.start(paramLessRequest())
        _ = try await collectDeltas(handle.mailbox)

        XCTAssertEqual(
            backend.recordingState.receivedSamplings.first,
            .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: nil))
    }

    /// Anti-regression: an EXPLICIT `temperature: 0` must still resolve `.greedy` even against a
    /// backend configured with non-`nil` `samplingDefaults` — `temperature: 0` stays the client's
    /// escape hatch to greedy, per `ServingSamplingPolicy.resolve`'s own doc comment. Uses the SAME
    /// `samplingDefaults` as the positive case above, so this test would fail if defaults ever
    /// overrode an explicit `0` rather than only filling an absent value.
    func testExplicitTemperatureZeroWithSamplingDefaultsStillResolvesGreedyAtTheDecoder()
        async throws
    {
        let backend = makeDefaultSamplingBackend(
            samplingDefaults: ServingSamplingDefaults(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0))

        let handle = try await backend.start(paramLessRequest(temperature: 0))
        _ = try await collectDeltas(handle.mailbox)

        XCTAssertEqual(backend.recordingState.receivedSamplings.first, .greedy)
    }
}

// MARK: - Fixtures

/// A `ScalarServingBackend` wired to a `RecordingDecoder` (via `InferenceActor`), constructed
/// directly through the backend's public initializer — the same shape
/// `ScalarServingBackendTests.swift`'s `makeBackend` uses, with weight-free doubles standing in for
/// `context.model`/`context.tokenizer`. Bundles the shared `RecordingDecoderState` alongside the
/// backend so a test can read what the decoder actually received after driving a request to
/// completion.
private struct DefaultSamplingBackendFixture {
    let backend: ScalarServingBackend
    let recordingState: RecordingDecoderState

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        try await backend.start(request)
    }
}

private func makeDefaultSamplingBackend(
    samplingDefaults: ServingSamplingDefaults?
) -> DefaultSamplingBackendFixture {
    let recorder = RecordingDecoder(script: [5, 6, 2], eos: 2)
    let state = recorder.state
    let backend = ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: recorder),
        codec: DefaultSamplingFixtureTextCodec(),
        stopTokenIDs: [2],
        modelStopStrings: [],
        configuration: ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 1,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            samplingDefaults: samplingDefaults))
    return DefaultSamplingBackendFixture(backend: backend, recordingState: state)
}

private func paramLessRequest(temperature: Double? = nil) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [OpenAIChatMessage(role: .user, text: "prompt")],
        maxCompletionTokens: 4,
        temperature: temperature,
        choiceCount: 1,
        stream: true,
        stop: [])
}

private func collectDeltas(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

private struct DefaultSamplingFixtureTextCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        DefaultSamplingFixtureDetokenizer()
    }
}

private struct DefaultSamplingFixtureDetokenizer: ScalarServingDetokenizer {
    private var pending: String?

    mutating func append(token: Int) {
        pending = "x"
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

// MARK: - RecordingDecoder, copied from DecoderSamplingCapabilityTests.swift

/// Lock-protected recorder backing `RecordingDecoder`, so the last-received sampling can be read
/// from the test's own context after `RecordingDecoder` (a value type) has crossed the actor
/// boundary via `sending` into an `InferenceActor`. Copied from
/// `SpikeCoreTests/DecoderSamplingCapabilityTests.swift` — that type is `private` to its own file,
/// so it is not visible here, and this file's own copy is likewise kept `private` to avoid any cross-
/// file collision. See that file's doc comment for the full rationale.
private final class RecordingDecoderState: @unchecked Sendable {
    private let lock = NSLock()
    private var _receivedSamplings: [DecoderSampling] = []

    var receivedSamplings: [DecoderSampling] {
        lock.lock()
        defer { lock.unlock() }
        return _receivedSamplings
    }

    func record(sampling: DecoderSampling) {
        lock.lock()
        defer { lock.unlock() }
        _receivedSamplings.append(sampling)
    }
}

/// Test double: replays a fixed script, reports `supportsSampling == true`, and records the last
/// `DecoderSampling` it actually received via `setSampling` — so a test can assert what the
/// DECODER received, not merely that the request admitted (a dropped default also admits, as
/// `.greedy`). Copied from `SpikeCoreTests/DecoderSamplingCapabilityTests.swift`'s `RecordingDecoder`,
/// trimmed to the sampling half only (this file has no penalties assertion to make).
private struct RecordingDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
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

    mutating func setPenalties(_ penalties: DecoderPenalties) {}
}
