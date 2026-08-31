import Foundation
import XCTest

import HarnessCore
import MLX
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import SpikeCore
@_spi(ProductionRouteEvidence) @testable import SpikeServingAdapters
import Tokenizers

/// Regression coverage for the production-route teardown defect: after the
/// continuous serving stack shuts down and every strong reference leaves scope,
/// active Metal memory must return to the pre-load baseline.
///
/// The Qwen3.8-27B dedicated-host observation was rejected because post-run
/// `activeMetalBytes` froze at loaded-weight scale (21.8 GB) across all 31
/// cleanup attempts. Root cause: the compiled decode steps captured the model
/// weights (and, under MLX_QWEN_FOUR_GDN=1, the fused GDN projections) as
/// compiled-trace CONSTANTS; mlx-swift 0.31.6's traced graphs contain
/// multi-output sibling reference cycles that its teardown heuristic can orphan
/// permanently (verified with `leaks` ROOT CYCLE reports over
/// `mlx::core::array::ArrayDesc`), and an orphaned trace pins every captured
/// constant's buffer. The fix passes the weights and fused projections as
/// compile STATE (tracer inputs) so an orphaned trace retains only
/// kilobyte-scale descriptors.
///
/// Residual bound: the upstream orphaned-cycle defect still leaks ~5 KB of
/// small constant buffers per load+serve cycle (measured 5,070-5,090 bytes on
/// Qwen3.5-9B; grows per cycle, not warmable). The 2 MB bound documents that
/// floor while failing loudly on any weight-scale (>= 85 MB observed for the
/// fused projections alone) regression. The production observation gate admits
/// the same upstream floor through a reviewed 4 MiB absolute tolerance (see
/// `qwen38ProductionRouteResidualToleranceBytes`) until the upstream leak is
/// fixed.
///
/// The upstream orphaning is per-trace nondeterministic, so a REGRESSION
/// (re-capturing weights as constants in every trace) is statistically loud
/// here, but a single pre-fix-style run can occasionally settle clean; treat a
/// pass as necessary, not sufficient, when bisecting. These assertions read
/// process-global Metal memory, so the suite must run serially (project
/// practice: `--no-parallel` for Metal-touching suites).
///
/// These tests drive the REAL load -> serve -> shutdown -> settle sequence and
/// need a local model directory via `FASTMLX_TEARDOWN_TEST_MODEL_PATH`
/// (a qwen3_5-family hybrid checkpoint exercises the production code path;
/// run with and without MLX_QWEN_FOUR_GDN=1).
final class ContinuousServingTeardownSettleTests: XCTestCase {
    /// Documented upstream per-cycle floor allowance -- see the type comment.
    private static let upstreamResidualAllowanceBytes = 2 << 20

    private static func modelDirectory() throws -> URL {
        guard
            let path = ProcessInfo.processInfo
                .environment["FASTMLX_TEARDOWN_TEST_MODEL_PATH"]
        else {
            throw XCTSkip(
                "Set FASTMLX_TEARDOWN_TEST_MODEL_PATH to a local model directory")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private static func settledActiveMemory(
        baseline: Int,
        attempts: Int = 20
    ) async throws -> Int {
        var active = Int.max
        for attempt in 0 ..< attempts {
            Memory.clearCache()
            active = Memory.snapshot().activeMemory
            if active <= baseline + upstreamResidualAllowanceBytes {
                return active
            }
            if attempt + 1 < attempts {
                try await Task.sleep(for: .milliseconds(200))
            }
        }
        return active
    }

    /// Real public-path regression: load through `loadContinuousServingModel`,
    /// run concurrent production-route evidence requests (the c2 batch shape
    /// from the rejected scorecard observation), shut the backend down, drop
    /// every reference, then require active Metal memory to settle back to the
    /// pre-load baseline (modulo the documented upstream floor).
    func testLoadedContinuousServingShutdownSettlesActiveMemoryToBaseline()
        async throws
    {
        let modelDirectory = try Self.modelDirectory()
        Memory.clearCache()
        let baseline = Memory.snapshot().activeMemory

        weak var weakBackend: ContinuousServingBackend?
        do {
            let loaded = try await loadContinuousServingModel(
                configuration: Self.loadConfiguration(
                    modelDirectory: modelDirectory))
            weakBackend = loaded.backend
            let authorization =
                try loaded.productionRouteEvidenceAuthorization()
            let tokenTrace = ContinuousServingOutputTokenTraceConfiguration
                .outputTokenIDs(maxCompletedRequests: 8, maxTokensPerRequest: 8)

            var handles: [ServingGenerationHandle] = []
            for index in 0 ..< 2 {
                let handle = try await loaded.backend
                    .startProductionRouteEvidence(
                        OpenAIChatCompletionRequest(
                            model: loaded.startupReport.launchedModel,
                            messages: [
                                OpenAIChatMessage(
                                    role: .user,
                                    text: "Report a stable one-line observation \(index).")
                            ],
                            maxCompletionTokens: 8,
                            temperature: 0,
                            choiceCount: 1,
                            stream: true,
                            stop: []),
                        authorization: authorization,
                        tokenTrace: tokenTrace)
                handles.append(handle)
            }
            for handle in handles {
                while let _ = try await handle.mailbox.next() {}
            }
            await loaded.backend.shutdown()
        }

        let active = try await Self.settledActiveMemory(baseline: baseline)
        XCTAssertNil(
            weakBackend,
            "ContinuousServingBackend must deinit once the loaded model leaves scope")
        XCTAssertLessThanOrEqual(
            active,
            baseline + Self.upstreamResidualAllowanceBytes,
            "active Metal memory must settle to the pre-load baseline after shutdown (weight-scale retention regressed)")
    }

    /// Runtime-level regression aimed at the fixed mechanism: drive the REAL
    /// `DenseContinuousBatchRuntime` prefill+decode API directly (compiled
    /// scalar step), remove the request, drop the stack, and require settle.
    func testDirectRuntimeSoloDecodeReleaseSettles() async throws {
        let modelDirectory = try Self.modelDirectory()
        Memory.clearCache()
        let baseline = Memory.snapshot().activeMemory

        func runOnce() async throws {
            let proof = try DenseContinuousBatchModelProof.verifying(
                modelDirectory: modelDirectory,
                allowHybridQwen35: true)
            let context = try await loadModel(
                from: modelDirectory,
                using: #huggingFaceTokenizerLoader())
            let stopTokenIDs = try resolveScalarServingStopTokenIDs(
                configuration: context.configuration,
                tokenizer: context.tokenizer)
            let runtime = try DenseContinuousBatchRuntime(
                model: context.model,
                verifiedBy: proof,
                allocationChunk: 256,
                maxContextTokens: 2_048,
                maxReservedContextTokens: 2_048,
                initialDecodeReserve: 8,
                maxReservedKVBytes: 1 << 30,
                kvCacheKind: .fp16,
                affineAttentionMode: .materialize,
                soloPLDConfiguration: nil)
            let id = BatchRequestID(1)
            let promptTokens = [1, 2, 3, 4, 5, 6, 7, 8]
            let submission = ContinuousBatchSubmission(
                promptTokens: promptTokens,
                maxOutputTokens: 4,
                stopTokenIDs: stopTokenIDs,
                architecture: .denseAttention,
                requestsSpeculation: false)
            let admission = ContinuousBatchRuntimeAdmission(
                id: id,
                submission: submission)
            _ = try runtime.decodeCohort(for: admission)
            try runtime.admit([admission])
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: id,
                    startToken: 0,
                    tokens: promptTokens,
                    isFinal: true,
                    totalPromptTokens: promptTokens.count,
                    maxOutputTokens: 4))
            for _ in 0 ..< 4 {
                let results = try runtime.decode(
                    .solo(id, speculationAllowed: false))
                if results.first?.finished ?? true { break }
            }
            runtime.remove(id)
        }
        try await runOnce()

        let active = try await Self.settledActiveMemory(baseline: baseline)
        XCTAssertLessThanOrEqual(
            active,
            baseline + Self.upstreamResidualAllowanceBytes,
            "direct runtime prefill+decode must not pin weight-scale memory after removal and drop")
    }

    private static func loadConfiguration(
        modelDirectory: URL
    ) throws -> ContinuousServingModelLoadConfiguration {
        // Mirrors qwen38ProductionRouteModelLoadConfiguration shape at test scale.
        ContinuousServingModelLoadConfiguration(
            launchedModel: "teardown-settle-test",
            modelDirectory: modelDirectory,
            memoryLimitBytes: 16 << 30,
            cacheLimitBytes: 2 << 30,
            maxReservedKVBytes: 1 << 30,
            maxContextTokens: 2_048,
            maxReservedContextTokens: 2_048,
            initialDecodeReserve: 8,
            coordinatorConfiguration: try ContinuousBatchConfiguration(
                maxActiveSlots: 4,
                maxPrefillSlots: 4,
                prefillChunkSize: 512,
                maxQueuedRequests: 4),
            publicationCapacity: 8,
            traceLimit: 4_096,
            backendConfiguration: ContinuousServingBackendConfiguration(
                defaultMaximumCompletionTokens: 8,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
                admission: .dynamic(
                    configuration: ServingAdmissionConfiguration(
                        soloPLDQualified: false,
                        maximumBatchRequests: 4,
                        maximumQueuedRequests: 4),
                    coalescing: .automatic(.milliseconds(5))),
                disableThinkingWhenToolsActive: false),
            soloPLDPolicy: nil,
            kvQuantTier: .fp16,
            allowHybridQwen35: true)
    }
}
