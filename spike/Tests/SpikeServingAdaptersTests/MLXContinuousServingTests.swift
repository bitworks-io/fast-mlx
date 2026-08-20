import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class MLXContinuousServingTests: XCTestCase {
    func testQwen3WidthOnePolicyPinsVerifiedExactShape() {
        let policy = ContinuousServingSoloPLDPolicy.qwen3WidthOne

        XCTAssertEqual(policy.ngram, 3)
        XCTAssertEqual(policy.maxDraft, 1)
        XCTAssertEqual(policy.lookback, 4_096)
        XCTAssertTrue(policy.compiledVerify)
    }

    func testStartupValidationExecutesRuntimeAndReleasesEveryReservation()
        async throws
    {
        let recorder = StartupValidationRuntimeRecorder()
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 2,
                maxQueuedRequests: 1),
            runtime: StartupValidationRuntime(recorder: recorder),
            automaticDrive: true,
            publicationCapacity: 1,
            traceLimit: 8)

        let generated = try await validateContinuousServingStartup(
            coordinator: coordinator,
            promptTokens: [10, 11, 12],
            stopTokenIDs: [99])

        XCTAssertEqual(generated, 1)
        XCTAssertEqual(recorder.prefillSlices, [0 ..< 2, 2 ..< 3])
        XCTAssertEqual(recorder.decodeCount, 1)
        XCTAssertEqual(recorder.removeCount, 1)
        let finalSlots = await coordinator.snapshots()
        let finalResources = await coordinator.runtimeResourceSnapshot()
        XCTAssertTrue(finalSlots.isEmpty)
        XCTAssertEqual(finalResources?.reservedKVBytes, 0)
    }

    func testLoaderRejectsUnsupportedArchitectureBeforeWeightLoading()
        async throws
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "fastmlx-continuous-unsupported-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: directory)
        }
        try Data(
            """
            {
              "model_type": "llama",
              "max_position_embeddings": 8192,
              "vocab_size": 128256
            }
            """.utf8
        ).write(to: directory.appendingPathComponent("config.json"))

        do {
            _ = try await loadContinuousServingModel(
                configuration: ContinuousServingModelLoadConfiguration(
                    launchedModel: "unsupported-source-locked",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 4_096,
                    coordinatorConfiguration: try ContinuousBatchConfiguration(
                        maxActiveSlots: 1,
                        maxPrefillSlots: 1,
                        prefillChunkSize: 128,
                        maxQueuedRequests: 1),
                    publicationCapacity: 1,
                    backendConfiguration:
                        ContinuousServingBackendConfiguration(
                            defaultMaximumCompletionTokens: 8,
                            queueRetryAfterSeconds: 1,
                            mailboxCapacity: .init(
                                maxDeltas: 2,
                                maxBytes: 4_096))))
            XCTFail("Unsupported architecture must fail before model loading")
        } catch let error as DenseContinuousBatchRuntimeError {
            XCTAssertEqual(error, .unsupportedModelFamily("llama"))
        }
    }

    // MARK: - qwen3_5 hybrid continuous-route admission (--allow-hybrid-qwen35, incr-3)
    //
    // Wiring proof for docs/task-inbox/2026-08-20-hybrid-continuous-serve-path-admission.md: the
    // config flag is threaded into DenseContinuousBatchModelProof.verifying(allowHybridQwen35:). All
    // three tests fail at the proof step (config.json only), BEFORE any weight load / global Memory
    // mutation — no safetensors, tokenizer, or network required.

    /// A VL-wrapped qwen3_5 config with full, valid hybrid geometry (mirrors HybridQwen35ProofArmTests).
    private func qwen35VLConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    /// Same qwen3_5 family tag, but with `linear_num_value_heads` dropped → the hybrid geometry
    /// derivation fails closed (invalidModelConfiguration) once the family gate is passed.
    private func qwen35BrokenGeometryConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","text_config":{"max_position_embeddings":262144,"vocab_size":248320,
          "num_hidden_layers":48,"full_attention_interval":4,"num_key_value_heads":8,"head_dim":128,
          "torch_dtype":"bfloat16","linear_num_key_heads":16,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    private func writeConfigDirectory(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "fastmlx-continuous-hybrid-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(json.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private func continuousLoadConfiguration(
        modelDirectory: URL, allowHybridQwen35: Bool
    ) throws -> ContinuousServingModelLoadConfiguration {
        ContinuousServingModelLoadConfiguration(
            launchedModel: "qwen3_5-source-locked",
            modelDirectory: modelDirectory,
            memoryLimitBytes: 8_192,
            cacheLimitBytes: 1_024,
            maxReservedKVBytes: 4_096,
            coordinatorConfiguration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 128,
                maxQueuedRequests: 1),
            publicationCapacity: 1,
            backendConfiguration: ContinuousServingBackendConfiguration(
                defaultMaximumCompletionTokens: 8,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096)),
            allowHybridQwen35: allowHybridQwen35)
    }

    /// Fallback invariant: WITHOUT the flag, a real qwen3_5 hybrid config is rejected at the proof with
    /// unsupportedModelFamily before any weight load — exactly the signal the executable's scalar
    /// fallback keys on, so the flag-OFF behavior is preserved byte-for-byte.
    func testContinuousLoaderRejectsQwen35HybridWithoutFlagBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(qwen35VLConfigJSON())
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await loadContinuousServingModel(
                configuration: try continuousLoadConfiguration(
                    modelDirectory: directory, allowHybridQwen35: false))
            XCTFail("qwen3_5 must be rejected on the continuous route without --allow-hybrid-qwen35")
        } catch let error as DenseContinuousBatchRuntimeError {
            XCTAssertEqual(error, .unsupportedModelFamily("qwen3_5"))
        }
    }

    /// Wiring proof: the SAME config's proof outcome flips with the flag — OFF rejects at the family
    /// gate (unsupportedModelFamily), ON passes the family gate and only then fails on the broken
    /// geometry (invalidModelConfiguration). Both occur at the proof, before weight load. This proves
    /// the flag is threaded (config → proof call) and is NOT hardcoded either direction.
    func testAllowHybridQwen35FlagFlipsProofOutcomeBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(qwen35BrokenGeometryConfigJSON())
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await loadContinuousServingModel(
                configuration: try continuousLoadConfiguration(
                    modelDirectory: directory, allowHybridQwen35: false))
            XCTFail("flag OFF: qwen3_5 must fail at the family gate")
        } catch let error as DenseContinuousBatchRuntimeError {
            XCTAssertEqual(error, .unsupportedModelFamily("qwen3_5"))
        }

        do {
            _ = try await loadContinuousServingModel(
                configuration: try continuousLoadConfiguration(
                    modelDirectory: directory, allowHybridQwen35: true))
            XCTFail("flag ON: broken hybrid geometry must fail closed at the proof")
        } catch let error as DenseContinuousBatchRuntimeError {
            // Passed the family gate (no longer unsupportedModelFamily), then rejected on the
            // admission-grade geometry derivation — proving the flag admitted qwen3_5 past the gate.
            XCTAssertEqual(error, .invalidModelConfiguration)
        }
    }

    /// The new configuration field round-trips through validation unchanged (both directions).
    func testAllowHybridQwen35SurvivesConfigurationValidation() throws {
        for admitted in [false, true] {
            let validated = try validateContinuousServingModelLoadConfiguration(
                try continuousLoadConfiguration(
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    allowHybridQwen35: admitted))
            XCTAssertEqual(validated.allowHybridQwen35, admitted)
        }
    }

    /// Same VL-wrapped qwen3_5 shape but linear_key_head_dim = 48 — a valid positive-integer geometry
    /// (the proof admits it) that the gated-delta Metal kernel cannot serve (Dk not divisible by 32).
    private func qwen35UnalignedDkConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    /// Real-kernel viability guard (incr-4): an admitted hybrid checkpoint whose Dk is not a multiple of
    /// 32 is refused on the real serving adapter — AFTER the proof admits the geometry, but BEFORE any
    /// weight load — so a bad checkpoint fails closed instead of truncating/faulting in the gated-delta
    /// Metal kernel. The guard lives here (not the proof) so the fp32 toy runtime tests (Dk=1) are
    /// unaffected.
    func testAllowHybridQwen35RejectsUnalignedKeyHeadDimBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(qwen35UnalignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await loadContinuousServingModel(
                configuration: try continuousLoadConfiguration(
                    modelDirectory: directory, allowHybridQwen35: true))
            XCTFail("Dk not divisible by 32 must fail closed before weight load")
        } catch let error as ContinuousServingModelLoadError {
            XCTAssertEqual(error, .hybridKernelKeyHeadDimUnaligned(48))
        }
    }

    func testLoaderConfigurationAcceptsBoundedDenseServingPolicy() throws {
        let configuration = ContinuousServingModelLoadConfiguration(
            launchedModel: "qwen3-source-locked",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 8_192,
            cacheLimitBytes: 1_024,
            maxReservedKVBytes: 4_096,
            coordinatorConfiguration: try ContinuousBatchConfiguration(
                maxActiveSlots: 4,
                maxPrefillSlots: 2,
                prefillChunkSize: 512,
                maxQueuedRequests: 8),
            publicationCapacity: 1,
            backendConfiguration: ContinuousServingBackendConfiguration(
                defaultMaximumCompletionTokens: 512,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(
                    maxDeltas: 8,
                    maxBytes: 32 * 1_024)))

        let validated = try validateContinuousServingModelLoadConfiguration(
            configuration)

        XCTAssertEqual(validated.launchedModel, "qwen3-source-locked")
        XCTAssertEqual(validated.modelDirectory.path, "/tmp")
        XCTAssertEqual(validated.memoryLimitBytes, 8_192)
        XCTAssertEqual(validated.cacheLimitBytes, 1_024)
        XCTAssertEqual(validated.maxReservedKVBytes, 4_096)
        XCTAssertEqual(validated.coordinatorConfiguration.maxActiveSlots, 4)
        XCTAssertEqual(validated.publicationCapacity, 1)
    }

    func testLoaderConfigurationRequiresExactSoloPLDPolicyAndDynamicAdmission()
        throws
    {
        let coordinator = try ContinuousBatchConfiguration(
            maxActiveSlots: 4,
            maxPrefillSlots: 2,
            prefillChunkSize: 512,
            maxQueuedRequests: 8)
        let dynamicBackend = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 512,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(
                maxDeltas: 8,
                maxBytes: 32 * 1_024),
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: true,
                    maximumBatchRequests: 4,
                    maximumQueuedRequests: 8),
                coalescing: .automatic(.milliseconds(5))))
        let policy = ContinuousServingSoloPLDPolicy.qwen3WidthOne
        let validated = try validateContinuousServingModelLoadConfiguration(
            ContinuousServingModelLoadConfiguration(
                launchedModel: "qwen3-source-locked",
                modelDirectory: URL(fileURLWithPath: "/tmp"),
                memoryLimitBytes: 8_192,
                cacheLimitBytes: 1_024,
                maxReservedKVBytes: 4_096,
                coordinatorConfiguration: coordinator,
                publicationCapacity: 1,
                backendConfiguration: dynamicBackend,
                soloPLDPolicy: policy))

        XCTAssertEqual(validated.soloPLDPolicy, policy)

        let unqualifiedWidePolicy = ContinuousServingSoloPLDPolicy(
            ngram: 3,
            maxDraft: 8,
            lookback: 4_096,
            compiledVerify: true)
        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 4_096,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: dynamicBackend,
                    soloPLDPolicy: unqualifiedWidePolicy))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .soloPLDPolicyMismatch)
        }

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 4_096,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: dynamicBackend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .soloPLDPolicyMismatch)
        }
    }

    func testLoaderRejectsDefaultCompletionBudgetAboveBatchableReserve() throws {
        let backend = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 129,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(
                maxDeltas: 8,
                maxBytes: 32 * 1_024))

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 4_096,
                    initialDecodeReserve: 128,
                    coordinatorConfiguration: try ContinuousBatchConfiguration(
                        maxActiveSlots: 4,
                        maxPrefillSlots: 2,
                        prefillChunkSize: 512,
                        maxQueuedRequests: 8),
                    publicationCapacity: 1,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .defaultCompletionBudgetExceedsInitialDecodeReserve(
                    defaultCompletionTokens: 129,
                    initialDecodeReserve: 128))
        }
    }

    func testLoaderConfigurationRejectsUnboundedOrUnsafeKVPolicy() throws {
        let coordinator = try ContinuousBatchConfiguration(
            maxActiveSlots: 2,
            maxPrefillSlots: 1,
            prefillChunkSize: 128,
            maxQueuedRequests: 2)
        let backend = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 32,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096))

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 0,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .invalidReservedKVLimit)
        }

        XCTAssertThrowsError(
            try validateContinuousServingModelLoadConfiguration(
                ContinuousServingModelLoadConfiguration(
                    launchedModel: "qwen3-source-locked",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    maxReservedKVBytes: 16_384,
                    coordinatorConfiguration: coordinator,
                    publicationCapacity: 1,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousServingModelLoadError,
                .reservedKVLimitExceedsMemoryLimit)
        }
    }
}

private final class StartupValidationRuntimeRecorder: Sendable {
    private struct State: Sendable {
        var prefillSlices: [Range<Int>] = []
        var decodeCount = 0
        var removeCount = 0
        var reservedKVBytes = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var prefillSlices: [Range<Int>] {
        state.withLock { $0.prefillSlices }
    }

    var decodeCount: Int {
        state.withLock { $0.decodeCount }
    }

    var removeCount: Int {
        state.withLock { $0.removeCount }
    }

    var reservedKVBytes: Int {
        state.withLock { $0.reservedKVBytes }
    }

    func admitted() {
        state.withLock { $0.reservedKVBytes = 1_024 }
    }

    func prefilling(_ range: Range<Int>) {
        state.withLock { $0.prefillSlices.append(range) }
    }

    func decoded() {
        state.withLock { $0.decodeCount += 1 }
    }

    func removed() {
        state.withLock {
            $0.removeCount += 1
            $0.reservedKVBytes = 0
        }
    }
}

private final class StartupValidationRuntime: ContinuousBatchRuntime {
    private let recorder: StartupValidationRuntimeRecorder

    init(recorder: StartupValidationRuntimeRecorder) {
        self.recorder = recorder
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        XCTAssertEqual(admissions.count, 1)
        recorder.admitted()
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: recorder.reservedKVBytes,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        recorder.prefilling(
            work.startToken ..< work.startToken + work.tokens.count)
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        guard case .solo(let id, speculationAllowed: false) = action else {
            XCTFail("Startup validation must use one non-speculative solo decode")
            return []
        }
        recorder.decoded()
        return [
            ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [42],
                finished: true,
                hasPendingSoloLookahead: false)
        ]
    }

    func remove(_ id: BatchRequestID) {
        recorder.removed()
    }
}
