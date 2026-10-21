import Foundation
import XCTest

import MLX
import ServingCore
import HarnessCore
import SpikeCore
@testable import SpikeServingAdapters

/// Increment 2 of `docs/task-inbox/2026-09-09-default-sampling-params-DECISION.md`: wires
/// `--default-sampling generation-config` into the SCALAR serve route, fail-closed at load, and
/// (cycle 278, `docs/task-inbox/2026-10-21-PREDECLARATION-default-sampling-on-the-continuous-route.md`)
/// into the CONTINUOUS-batch route through the same resolver. `ExactQwen35MTP` is still deliberately
/// not wired. Increment 1 (the pure resolver, `ServingSamplingPolicy.resolve(from:defaults:)`) is
/// already covered by `ServingSamplingPolicyTests.swift`; this file covers that each wired route's
/// backend configuration defaults to no sampling defaults, and that the resolved defaults are
/// genuinely APPLIED at admission, not merely accepted. The production call-site wiring itself is
/// pinned by `DefaultSamplingContinuousWiringStructuralTests.swift`. It no longer covers a `.compiledFP16` load-time
/// refusal — `scalarServingDefaultSamplingDecoderStrategyError` was REMOVED once `.compiledFP16`
/// became a `RouteSwitchingDecoder` whose general side genuinely supports sampling (see
/// `RouteSwitchingDecoder`'s doc comment); that combination is now admitted, not refused.
final class DefaultSamplingServeWiringTests: XCTestCase {

    /// Snapshot the process-global MLX `Memory.memoryLimit` / `Memory.cacheLimit` and return a closure
    /// restoring them. The load tests below that get PAST the generation-config guard execute
    /// `Memory.memoryLimit = ...` / `Memory.cacheLimit = ...` before failing on the weight-less
    /// fixture; restoring in `defer` keeps that from leaking into whichever test runs next.
    private func captureProcessGlobalMemoryLimits() -> () -> Void {
        let savedMemoryLimit = Memory.memoryLimit
        let savedCacheLimit = Memory.cacheLimit
        return {
            Memory.memoryLimit = savedMemoryLimit
            Memory.cacheLimit = savedCacheLimit
        }
    }

    // The guard-unit tests that used to live here (`scalarServingDefaultSamplingDecoderStrategyError`,
    // refusing `.generationConfig` + `.compiledFP16`) were REMOVED along with that function: once
    // `.compiledFP16` became a `RouteSwitchingDecoder` whose general side genuinely supports
    // sampling (see `RouteSwitchingDecoder`'s doc comment), the combination that guard used to
    // refuse is now correctly admitted — there is no longer a load-time refusal to unit test here.
    // `ScalarServingIsNonSpeculativeRouteTests.swift` and `RouteSwitchingDecoderTests.swift` cover
    // the new behavior instead.

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

    // MARK: - Call-site tests for guard (A), `loadScalarServingModel`'s
    // `generation_config.json` load
    //
    // Unlike guard (B) (`scalarServingDefaultSamplingDecoderStrategyError`, covered above as a pure
    // function only -- reaching its call site needs a real fleet checkpoint), guard (A) runs BEFORE
    // any weight load or `Memory.*` mutation (see `loadScalarServingModel`'s comment immediately
    // above its call, ~MLXScalarServing.swift:1007), so its call site IS reachable from a unit test
    // against an EMPTY temp directory. These four tests close that half of the gap: two refusals
    // (1, 2) and two controls (3, 4) that prove the refusals are actually discriminating on guard
    // (A) itself, not merely on "the loader failed for some unrelated reason".

    /// Refusal 1: an empty model directory has no `generation_config.json` at all --
    /// `GenerationConfigSamplingDefaults.load` throws `.unreadable`, which guard (A) must convert to
    /// exactly `.defaultSamplingGenerationConfigUnavailable(.unreadable)`.
    func testLoadScalarServingModelRefusesMissingGenerationConfigArtifact() async throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        let configuration = makeDefaultSamplingLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadScalarServingModel(configuration: configuration)
            XCTFail("expected loadScalarServingModel to refuse a missing generation_config.json")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .defaultSamplingGenerationConfigUnavailable(.unreadable))
        }
    }

    /// Refusal 2: a `generation_config.json` present but with `do_sample: false` -- proves the
    /// refusal discriminates WHY the artifact is unusable, not merely THAT it is. Must throw exactly
    /// `.defaultSamplingGenerationConfigUnavailable(.samplingNotEnabled)`, a DIFFERENT associated
    /// value than refusal 1's.
    func testLoadScalarServingModelRefusesGenerationConfigWithSamplingDisabled() async throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        try """
            {"do_sample": false, "temperature": 1.0}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)

        let configuration = makeDefaultSamplingLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadScalarServingModel(configuration: configuration)
            XCTFail("expected loadScalarServingModel to refuse do_sample: false")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .defaultSamplingGenerationConfigUnavailable(.samplingNotEnabled))
        }
    }

    /// CONTROL for refusals 1-2: same empty temp directory as refusal 1, but `defaultSampling: .off`
    /// skips guard (A) entirely (the `if configuration.defaultSampling == .generationConfig` branch
    /// never runs), so this reaches PAST it into the real model load (`loadModel(from:using:)`),
    /// which fails on its OWN account since the directory has no `config.json`/weights. Without this
    /// control, refusals 1-2 cannot be distinguished from "the loader fails on this directory no
    /// matter what" -- this proves guard (A) itself is what fired above, by proving it does NOT fire
    /// here.
    ///
    /// HAZARD: reaching past guard (A) executes `Memory.memoryLimit = configuration.memoryLimitBytes`
    /// / `Memory.cacheLimit = configuration.cacheLimitBytes` / `Memory.clearCache()` --
    /// PROCESS-GLOBAL MLX allocator state shared by every other test in this bundle, and MLX
    /// peak-memory readings are order-dependent. This test deliberately uses realistic,
    /// non-constraining limits (8 GiB / 1 GiB, see `makeDefaultSamplingLoadConfiguration`) rather
    /// than tiny ones, so it does not throttle whatever test happens to run later in the same
    /// process.
    func testLoadScalarServingModelWithDefaultSamplingOffDoesNotHitGenerationConfigGuard()
        async throws
    {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let restoreMemoryLimits = captureProcessGlobalMemoryLimits()
        defer { restoreMemoryLimits() }

        let configuration = makeDefaultSamplingLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .off)

        do {
            _ = try await loadScalarServingModel(configuration: configuration)
            XCTFail("expected loadScalarServingModel to fail on the empty directory's real load")
        } catch let error as ScalarServingModelLoadError {
            if case .defaultSamplingGenerationConfigUnavailable = error {
                XCTFail("guard (A) must not fire when defaultSampling == .off; got \(error)")
            }
            // Any OTHER ScalarServingModelLoadError case is the expected outcome here -- this
            // control only proves guard (A) itself did not fire.
        } catch {
            // A non-ScalarServingModelLoadError thrown straight from `loadModel` is also an
            // acceptable pass for this control.
        }
    }

    /// CONTROL for refusal 2, in the other direction: a VALID `generation_config.json`
    /// (`do_sample: true` with a usable temperature and in-range `top_p`/`top_k`) paired with
    /// `defaultSampling: .generationConfig` must ALSO proceed PAST guard (A) -- proving the guard
    /// does not refuse every artifact indiscriminately, only the ones refusal 1-2 target. Shares the
    /// same process-global `Memory.*` mutation hazard documented on the control immediately above,
    /// for the same reason (guard (A) passes, so execution reaches the `Memory.*` lines).
    func testLoadScalarServingModelWithValidGenerationConfigProceedsPastTheGuard() async throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let restoreMemoryLimits = captureProcessGlobalMemoryLimits()
        defer { restoreMemoryLimits() }
        try """
            {"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)

        let configuration = makeDefaultSamplingLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadScalarServingModel(configuration: configuration)
            XCTFail("expected loadScalarServingModel to fail on the empty directory's real load")
        } catch let error as ScalarServingModelLoadError {
            if case .defaultSamplingGenerationConfigUnavailable = error {
                XCTFail("a VALID generation_config.json must pass guard (A); got \(error)")
            }
        } catch {
            // A non-ScalarServingModelLoadError thrown straight from `loadModel` is also an
            // acceptable pass for this control.
        }
    }

    // MARK: - Continuous route (cycle 278): `loadContinuousServingModel` resolves the same artifact

    /// D5(a): on the continuous route a missing `generation_config.json` is refused with the SAME
    /// `.defaultSamplingGenerationConfigUnavailable(.unreadable)` the scalar load throws, AFTER the
    /// architecture proof (the qwen3 fixture passes it) and BEFORE any weight load.
    func testLoadContinuousServingModelRefusesMissingGenerationConfigArtifact() async throws {
        let modelDirectory = try makeDenseContinuousModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        let configuration = try makeDefaultSamplingContinuousLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadContinuousServingModel(configuration: configuration)
            XCTFail("expected loadContinuousServingModel to refuse a missing generation_config.json")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .defaultSamplingGenerationConfigUnavailable(.unreadable))
        }
    }

    /// D5(b): `do_sample: false` is refused with `.samplingNotEnabled` (a different associated value
    /// than (a)'s), proving the continuous load discriminates WHY the artifact is unusable.
    func testLoadContinuousServingModelRefusesGenerationConfigWithSamplingDisabled() async throws {
        let modelDirectory = try makeDenseContinuousModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        try """
            {"do_sample": false, "temperature": 1.0}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)

        let configuration = try makeDefaultSamplingContinuousLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadContinuousServingModel(configuration: configuration)
            XCTFail("expected loadContinuousServingModel to refuse do_sample: false")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .defaultSamplingGenerationConfigUnavailable(.samplingNotEnabled))
        }
    }

    /// CONTROL for D5: the SAME fixture with `defaultSampling: .off` (no generation_config.json at
    /// all) passes the proof AND skips the new check, so it fails later on its OWN account (no
    /// weights) with something other than `.defaultSamplingGenerationConfigUnavailable`. Proves the
    /// fixture really reaches the check (the proof accepts it) and that the check, not the fixture,
    /// is what fired in D5. Uses realistic memory limits: reaching past the check mutates the
    /// process-global `Memory.*` limits (see the scalar controls above).
    func testLoadContinuousServingModelWithDefaultSamplingOffFailsLaterThanTheGuard() async throws {
        let modelDirectory = try makeDenseContinuousModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let restoreMemoryLimits = captureProcessGlobalMemoryLimits()
        defer { restoreMemoryLimits() }

        let configuration = try makeDefaultSamplingContinuousLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .off)

        do {
            _ = try await loadContinuousServingModel(configuration: configuration)
            XCTFail("expected the weight-less fixture to fail the real model load")
        } catch {
            assertFailedPastTheGenerationConfigGuard(error)
        }
    }

    /// CONTROL for D5, other direction: a VALID `generation_config.json` passes the guard and the load
    /// fails later (no weights), never with `.defaultSamplingGenerationConfigUnavailable`.
    func testLoadContinuousServingModelWithValidGenerationConfigProceedsPastTheGuard() async throws {
        let modelDirectory = try makeDenseContinuousModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let restoreMemoryLimits = captureProcessGlobalMemoryLimits()
        defer { restoreMemoryLimits() }
        try """
            {"do_sample": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)

        let configuration = try makeDefaultSamplingContinuousLoadConfiguration(
            modelDirectory: modelDirectory, defaultSampling: .generationConfig)

        do {
            _ = try await loadContinuousServingModel(configuration: configuration)
            XCTFail("expected the weight-less fixture to fail the real model load")
        } catch {
            assertFailedPastTheGenerationConfigGuard(error)
        }
    }

    /// The load configuration's `defaultSampling` defaults to `.off` and survives validation.
    func testContinuousServingModelLoadConfigurationDefaultSamplingDefaultsToOffAndSurvivesValidation()
        throws
    {
        let off = try makeDefaultSamplingContinuousLoadConfiguration(
            modelDirectory: URL(fileURLWithPath: "/tmp"), defaultSampling: nil)
        XCTAssertEqual(off.defaultSampling, .off)

        let on = try validateContinuousServingModelLoadConfiguration(
            try makeDefaultSamplingContinuousLoadConfiguration(
                modelDirectory: URL(fileURLWithPath: "/tmp"), defaultSampling: .generationConfig))
        XCTAssertEqual(on.defaultSampling, .generationConfig)
    }

    /// `ContinuousServingBackendConfiguration.samplingDefaults` defaults to `nil`.
    func testContinuousServingBackendConfigurationSamplingDefaultsDefaultsToNil() {
        let configuration = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096))
        XCTAssertNil(configuration.samplingDefaults)
    }

    // MARK: - The shared helper both routes call (`resolveServingDefaultSampling`)

    /// `.off` never reads the artifact: nil even when the directory has no `generation_config.json`.
    func testResolveServingDefaultSamplingOffReturnsNilWithoutReadingTheArtifact() throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        XCTAssertNil(try resolveServingDefaultSampling(.off, modelDirectory: modelDirectory))
    }

    /// `.generationConfig` returns the artifact's sampling subset.
    func testResolveServingDefaultSamplingGenerationConfigReturnsTheArtifactDefaults() throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        try """
            {"do_sample": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)
        XCTAssertEqual(
            try resolveServingDefaultSampling(.generationConfig, modelDirectory: modelDirectory),
            ServingSamplingDefaults(temperature: 0.6, topP: 0.95, topK: 20, minP: nil))
    }

    /// `.generationConfig` fails closed on every unusable artifact, wrapped as the ONE machine-readable
    /// load error both routes share.
    func testResolveServingDefaultSamplingGenerationConfigFailsClosed() throws {
        let modelDirectory = try makeEmptyDefaultSamplingModelDirectory()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        XCTAssertThrowsError(
            try resolveServingDefaultSampling(.generationConfig, modelDirectory: modelDirectory)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .defaultSamplingGenerationConfigUnavailable(.unreadable))
        }
        try """
            {"do_sample": false, "temperature": 1.0}
            """.write(
                to: modelDirectory.appendingPathComponent("generation_config.json"),
                atomically: true, encoding: .utf8)
        XCTAssertThrowsError(
            try resolveServingDefaultSampling(.generationConfig, modelDirectory: modelDirectory)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .defaultSamplingGenerationConfigUnavailable(.samplingNotEnabled))
        }
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

/// A fresh, empty per-test directory under `FileManager.default.temporaryDirectory`, for the guard
/// (A) call-site tests. The caller is responsible for removing it (each test does so via `defer`).
private func makeEmptyDefaultSamplingModelDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("default-sampling-guard-a-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

/// A model directory whose `config.json` passes `DenseContinuousBatchModelProof.verifying` (a dense
/// `qwen3` config) but holds NO weights or tokenizer, so a continuous load that gets past the
/// `generation_config.json` guard fails on its own account at the real model load.
private func makeDenseContinuousModelDirectory() throws -> URL {
    let directory = try makeEmptyDefaultSamplingModelDirectory()
    try Data(#"""
        {"model_type":"qwen3","max_position_embeddings":32768,"vocab_size":2048,
         "num_hidden_layers":2,"num_key_value_heads":2,"head_dim":16,"torch_dtype":"bfloat16"}
        """#.utf8)
        .write(to: directory.appendingPathComponent("config.json"))
    return directory
}

/// A control's failure must have come from PAST the guard: not the guard's own error, and not a
/// proof error either (a proof error would mean the fixture never reached the guard, which would
/// make the D5 refusals and these controls vacuous).
private func assertFailedPastTheGenerationConfigGuard(
    _ error: Error, file: StaticString = #filePath, line: UInt = #line
) {
    if case .defaultSamplingGenerationConfigUnavailable? = error as? ScalarServingModelLoadError {
        XCTFail("the generation_config guard fired unexpectedly: \(error)", file: file, line: line)
    }
    XCTAssertFalse(
        error is DenseContinuousBatchRuntimeError,
        "the fixture must pass the architecture proof to reach the guard; got \(error)",
        file: file, line: line)
}

/// Continuous-route load configuration with realistic (8 GiB / 1 GiB) limits for the same reason as
/// `makeDefaultSamplingLoadConfiguration`. `defaultSampling: nil` omits the argument entirely (to
/// exercise the init default).
private func makeDefaultSamplingContinuousLoadConfiguration(
    modelDirectory: URL,
    defaultSampling: FastMLXServeDefaultSampling?
) throws -> ContinuousServingModelLoadConfiguration {
    let coordinator = try ContinuousBatchConfiguration(
        maxActiveSlots: 1, maxPrefillSlots: 1, prefillChunkSize: 128, maxQueuedRequests: 1)
    let backend = ContinuousServingBackendConfiguration(
        defaultMaximumCompletionTokens: 8,
        queueRetryAfterSeconds: 1,
        mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096))
    guard let defaultSampling else {
        return ContinuousServingModelLoadConfiguration(
            launchedModel: "fixture-model",
            modelDirectory: modelDirectory,
            memoryLimitBytes: 8 * 1024 * 1024 * 1024,
            cacheLimitBytes: 1 * 1024 * 1024 * 1024,
            maxReservedKVBytes: 1 * 1024 * 1024 * 1024,
            coordinatorConfiguration: coordinator,
            publicationCapacity: 1,
            backendConfiguration: backend)
    }
    return ContinuousServingModelLoadConfiguration(
        launchedModel: "fixture-model",
        modelDirectory: modelDirectory,
        memoryLimitBytes: 8 * 1024 * 1024 * 1024,
        cacheLimitBytes: 1 * 1024 * 1024 * 1024,
        maxReservedKVBytes: 1 * 1024 * 1024 * 1024,
        coordinatorConfiguration: coordinator,
        publicationCapacity: 1,
        backendConfiguration: backend,
        defaultSampling: defaultSampling)
}

/// Builds a `ScalarServingModelLoadConfiguration` for the guard (A) call-site tests. Memory/cache
/// limits are realistic (8 GiB / 1 GiB), not tiny, so the process-global `Memory.*` mutation the two
/// control tests trigger by design (see their doc comments) does not throttle unrelated tests that
/// share this MLX allocator state.
private func makeDefaultSamplingLoadConfiguration(
    modelDirectory: URL,
    defaultSampling: FastMLXServeDefaultSampling
) -> ScalarServingModelLoadConfiguration {
    ScalarServingModelLoadConfiguration(
        launchedModel: "fixture-model",
        modelDirectory: modelDirectory,
        memoryLimitBytes: 8 * 1024 * 1024 * 1024,
        cacheLimitBytes: 1 * 1024 * 1024 * 1024,
        backendConfiguration: ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 1,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)),
        defaultSampling: defaultSampling)
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
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
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
