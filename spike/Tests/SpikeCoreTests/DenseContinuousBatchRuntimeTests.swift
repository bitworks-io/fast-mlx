import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

private final class TinyDenseLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = [1]
    private let vocabularySize = 2_048

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        guard let cache = cache?.first else {
            preconditionFailure("tiny cache-sensitive model requires one cache")
        }
        let (keys, _) = cache.update(keys: values, values: values)

        let historyTarget = keys.sum(axes: [1, 2, 3]).asType(.int32) + 1
        let target = broadcast(
            historyTarget.reshaped([inputs.dim(0), 1, 1]),
            to: [inputs.dim(0), inputs.dim(1), 1])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class DenseContinuousBatchRuntimeTests: XCTestCase {
    private func makeRuntime(
        allocationChunk: Int = 4,
        maxContextTokens: Int = 32_768,
        initialDecodeReserve: Int = 384
    ) throws
        -> DenseContinuousBatchRuntime
    {
        try DenseContinuousBatchRuntime(
            testing: TinyDenseLanguageModel(),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            initialDecodeReserve: initialDecodeReserve)
    }

    private func prefill(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        tokens: [Int],
        chunks: [Int],
        maxOutputTokens: Int = 8
    ) throws {
        var start = 0
        for count in chunks {
            let end = start + count
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: BatchRequestID(id),
                    startToken: start,
                    tokens: Array(tokens[start ..< end]),
                    isFinal: end == tokens.count,
                    totalPromptTokens: tokens.count,
                    maxOutputTokens: maxOutputTokens))
            start = end
        }
        XCTAssertEqual(start, tokens.count)
    }

    private func emitted(_ results: [ContinuousBatchRuntimeDecodeResult]) -> [UInt64: [Int]] {
        Dictionary(uniqueKeysWithValues: results.map { ($0.id.rawValue, $0.tokens) })
    }

    private func collect(_ stream: AsyncThrowingStream<Int, Error>) async throws -> [Int] {
        var tokens: [Int] = []
        for try await token in stream { tokens.append(token) }
        return tokens
    }

    func testChunkedPrefillStagesOnlyTheFinalGreedyToken() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10, 11, 12], chunks: [1, 2])

        let first = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        let second = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))

        XCTAssertEqual(first, [
            ContinuousBatchRuntimeDecodeResult(
                id: BatchRequestID(1), tokens: [34], finished: false,
                hasPendingSoloLookahead: true),
        ])
        XCTAssertEqual(second.map(\.tokens), [[68]])
    }

    func testSoloDrainThenSharedBatchMatchesUninterruptedScalarTokens() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])

        let solo = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        try prefill(runtime, id: 2, tokens: [50], chunks: [1])
        let drain = try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))
        let firstBatch = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        let secondBatch = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))

        XCTAssertEqual(solo.map(\.tokens), [[11]])
        XCTAssertEqual(drain, [
            ContinuousBatchRuntimeDecodeResult(
                id: BatchRequestID(1), tokens: [22], finished: false,
                hasPendingSoloLookahead: false),
        ])
        XCTAssertEqual(emitted(firstBatch), [1: [44], 2: [51]])
        XCTAssertEqual(emitted(secondBatch), [1: [88], 2: [102]])

        let reference = try makeRuntime()
        try prefill(reference, id: 1, tokens: [10], chunks: [1])
        let scalar = try (0 ..< 4).map { _ in
            try reference.decode(.solo(BatchRequestID(1), speculationAllowed: false))[0]
                .tokens[0]
        }
        XCTAssertEqual(scalar, [11, 22, 44, 88])
    }

    func testMiddleRemovalPreservesStableIDRowMapping() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        try prefill(runtime, id: 2, tokens: [20], chunks: [1])
        try prefill(runtime, id: 3, tokens: [30], chunks: [1])

        let first = try runtime.decode(
            .batch(
                [BatchRequestID(1), BatchRequestID(2), BatchRequestID(3)],
                speculationAllowed: false))
        runtime.remove(BatchRequestID(2))
        let second = try runtime.decode(
            .batch([BatchRequestID(3), BatchRequestID(1)], speculationAllowed: false))

        XCTAssertEqual(emitted(first), [1: [11], 2: [21], 3: [31]])
        XCTAssertEqual(emitted(second), [1: [22], 3: [62]])
    }

    func testBatchToSoloSurvivorContinuesWithoutDuplicateOrDrop() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        try prefill(runtime, id: 2, tokens: [20], chunks: [1])

        let batch = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        runtime.remove(BatchRequestID(2))
        let solo = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))

        XCTAssertEqual(emitted(batch), [1: [11], 2: [21]])
        XCTAssertEqual(solo.map(\.tokens), [[22]])
    }

    func testStableBatchMembershipCompilesOnceAndUsesBoundedInitialReserve() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 256,
            initialDecodeReserve: 2)
        try prefill(
            runtime, id: 1, tokens: [10, 11, 12], chunks: [1, 2],
            maxOutputTokens: 100)
        try prefill(
            runtime, id: 2, tokens: [20], chunks: [1],
            maxOutputTokens: 100)

        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))

        let diagnostics = runtime.diagnostics()
        XCTAssertEqual(diagnostics.batchTraceCount, 1)
        XCTAssertEqual(diagnostics.batchMembership, [BatchRequestID(1), BatchRequestID(2)])
        XCTAssertEqual(diagnostics.batchCapacity, 8)
    }

    func testInvalidTransitionsAndSpeculationFailClosed() throws {
        let runtime = try makeRuntime()

        XCTAssertThrowsError(
            try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .unknownRequest(BatchRequestID(1)))
            }

        XCTAssertThrowsError(
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: BatchRequestID(1), startToken: 1, tokens: [10], isFinal: true,
                    totalPromptTokens: 2, maxOutputTokens: 8))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .outOfOrderPrefill(BatchRequestID(1), expected: 0, actual: 1))
            }

        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        XCTAssertThrowsError(
            try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .drainWithoutPendingLookahead(BatchRequestID(1)))
            }
        XCTAssertThrowsError(
            try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: true))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .speculationUnsupported)
            }
        XCTAssertThrowsError(
            try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(1)],
                    speculationAllowed: false))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidBatchMembership([BatchRequestID(1), BatchRequestID(1)]))
            }
    }

    func testConfigDerivedProofRejectsUnsupportedModelBeforeRuntimeConstruction() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(
            #"{"model_type":"qwen3_moe","max_position_embeddings":32768,"vocab_size":2048}"#.utf8
        ).write(to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(
            try DenseContinuousBatchModelProof.verifying(modelDirectory: directory)) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .unsupportedModelFamily("qwen3_moe"))
            }
    }

    func testRuntimeCapabilityAndContextLimitsRejectAtAdmissionWithoutPoisoningCoordinator()
        async throws
    {
        let configuration = try ContinuousBatchConfiguration(
            maxActiveSlots: 1,
            maxPrefillSlots: 1,
            prefillChunkSize: 4)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration,
            runtime: try DenseContinuousBatchRuntime(
                testing: TinyDenseLanguageModel(),
                allocationChunk: 4,
                maxContextTokens: 16,
                initialDecodeReserve: 2),
            automaticDrive: false)

        do {
            _ = try await coordinator.submit(
                ContinuousBatchSubmission(
                    promptTokens: [10], maxOutputTokens: 4, eosToken: 2,
                    architecture: .denseAttention, requestsSpeculation: true))
            XCTFail("continuous runtime accepted unsupported solo speculation")
        } catch {
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .speculationUnsupported)
        }
        do {
            _ = try await coordinator.submit(
                ContinuousBatchSubmission(
                    promptTokens: Array(repeating: 10, count: 10),
                    maxOutputTokens: 7, eosToken: 2,
                    architecture: .denseAttention))
            XCTFail("continuous runtime accepted an over-limit context")
        } catch {
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .contextLimitExceeded(BatchRequestID(1), requested: 17, limit: 16))
        }
        let isShutDown = await coordinator.isShutDown()
        XCTAssertFalse(isShutDown)

        let accepted = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: [10], maxOutputTokens: 1, eosToken: 2,
                architecture: .denseAttention))
        while try await coordinator.runOneTick() {}
        let acceptedTokens = try await collect(accepted.tokens)
        XCTAssertEqual(acceptedTokens, [11])
    }

    func testInvalidTokenIDFailsBeforeInt32Conversion() throws {
        let runtime = try makeRuntime()
        XCTAssertThrowsError(
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: BatchRequestID(9), startToken: 0, tokens: [-1], isFinal: true,
                    totalPromptTokens: 1, maxOutputTokens: 1))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidTokenID(BatchRequestID(9), -1))
            }
        XCTAssertThrowsError(
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: BatchRequestID(10), startToken: 0, tokens: [2_048], isFinal: true,
                    totalPromptTokens: 1, maxOutputTokens: 1))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidTokenID(BatchRequestID(10), 2_048))
            }
    }

    func testAllocationChunkCannotExceedConfiguredContextLimit() {
        XCTAssertThrowsError(
            try makeRuntime(allocationChunk: 17, maxContextTokens: 16)) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidAllocationChunk(17))
            }
    }

    func testAggregateContextReservationRejectsBurstAtomicallyAndReleasesOnRemoval()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 2,
                maxPrefillSlots: 2,
                prefillChunkSize: 4),
            runtime: try DenseContinuousBatchRuntime(
                testing: TinyDenseLanguageModel(),
                allocationChunk: 4,
                maxContextTokens: 16,
                maxReservedContextTokens: 16,
                initialDecodeReserve: 2),
            automaticDrive: false)
        let nineTokens = ContinuousBatchSubmission(
            promptTokens: [10, 11, 12, 13, 14],
            maxOutputTokens: 4,
            eosToken: 2,
            architecture: .denseAttention)

        do {
            _ = try await coordinator.submitBatch([nineTokens, nineTokens])
            XCTFail("aggregate reservation admitted 18 tokens into a 16-token budget")
        } catch {
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .aggregateContextLimitExceeded(requested: 18, limit: 16))
        }
        let snapshotsAfterRejection = await coordinator.snapshots()
        XCTAssertTrue(snapshotsAfterRejection.isEmpty)

        let accepted = try await coordinator.submit(nineTokens)
        XCTAssertEqual(accepted.id, BatchRequestID(1))
        _ = await coordinator.cancel(accepted.id)
        let replacement = try await coordinator.submit(nineTokens)
        XCTAssertEqual(replacement.id, BatchRequestID(2))
        await coordinator.shutdown()
    }

    func testCoordinatorExecutesDecodeFirstDrainAndSharedBatchEndToEnd() async throws {
        let configuration = try ContinuousBatchConfiguration(
            maxActiveSlots: 2,
            maxPrefillSlots: 2,
            prefillChunkSize: 1)
        let coordinator = ContinuousBatchCoordinator(
            configuration: configuration,
            runtime: try DenseContinuousBatchRuntime(
                testing: TinyDenseLanguageModel(),
                allocationChunk: 4),
            automaticDrive: false,
            traceLimit: 32)
        let handles = try await coordinator.submitBatch([
            ContinuousBatchSubmission(
                promptTokens: [10, 11, 12], maxOutputTokens: 2, eosToken: 127,
                architecture: .denseAttention),
            ContinuousBatchSubmission(
                promptTokens: [50], maxOutputTokens: 4, eosToken: 127,
                architecture: .denseAttention),
        ])

        while try await coordinator.runOneTick() {}

        let long = try await collect(handles[0].tokens)
        let short = try await collect(handles[1].tokens)
        XCTAssertEqual(long, [34, 68])
        XCTAssertEqual(short, [51, 102, 204, 408])
        let operations = await coordinator.executionTrace().compactMap { event in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        XCTAssertTrue(
            operations.contains(.decode(.drainSoloPipeline(handles[1].id))))
        XCTAssertTrue(
            operations.contains(
                .decode(.batch([handles[0].id, handles[1].id], speculationAllowed: false))))
    }
}
