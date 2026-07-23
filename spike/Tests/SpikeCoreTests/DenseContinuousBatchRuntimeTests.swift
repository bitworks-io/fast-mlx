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

private final class TinyCompressedBatchLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [2]
    private let vocabularySize = 2_048
    private(set) var newCacheCallCount = 0

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        newCacheCallCount += 1
        return [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache = cache?.first else {
            preconditionFailure("tiny compressed model requires one cache")
        }
        let scalar = inputs.asType(.float16).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let keys = broadcast(
            scalar, to: [inputs.dim(0), 2, inputs.dim(1), 128])
        let values = keys
        let queries = broadcast(
            scalar, to: [inputs.dim(0), 4, inputs.dim(1), 128])
        let mask = cache.makeMask(
            n: inputs.dim(1), windowSize: nil, returnArray: true)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: Float(1 / sqrt(128.0)),
            mask: mask)
        let target = (attended.mean(axes: [1, 3], keepDims: false) + 1)
            .asType(.int32).expandedDimensions(axis: -1)
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class DenseContinuousBatchRuntimeTests: XCTestCase {
    private func makeRuntime(
        allocationChunk: Int = 4,
        maxContextTokens: Int = 32_768,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil
    ) throws
        -> DenseContinuousBatchRuntime
    {
        try DenseContinuousBatchRuntime(
            testing: TinyDenseLanguageModel(),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            initialDecodeReserve: initialDecodeReserve,
            maxReservedKVBytes: maxReservedKVBytes)
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

    private func makeCompressedRuntime() throws -> DenseContinuousBatchRuntime {
        try DenseContinuousBatchRuntime(
            testing: TinyCompressedBatchLanguageModel(),
            allocationChunk: 4,
            maxContextTokens: 64,
            initialDecodeReserve: 3,
            kvCacheKind: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM,
            compressedKVAttentionAdmission: try makeCompressedBatchAdmission(),
            keyValueHeadCount: 2,
            headDimension: 128,
            elementBytes: 2)
    }

    private func makePhi3MiniCompressedBatchAdmission()
        throws -> CompressedKVAttentionRuntimeAdmission
    {
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: phi3MiniSourceLockedConfig(),
                checkpointManifestHash: "0123456789abcdef",
                checkpointContentSHA256: String(repeating: "d", count: 64),
                tokenizerSHA256: String(repeating: "c", count: 64)))
    }

    private func makePhi3MiniCompressedBatchSourceSnapshot()
        throws -> CompressedKVAttentionRuntimeSourceSnapshot
    {
        try .load(
            exactModelConfigData: phi3MiniSourceLockedConfig(),
            checkpointManifestHash: "0123456789abcdef",
            checkpointContentSHA256: String(repeating: "d", count: 64),
            tokenizerSHA256: String(repeating: "c", count: 64))
    }

    private func phi3MiniSourceLockedConfig() -> Data {
        Data(#"""
        {
            "architectures": [
                "Phi3ForCausalLM"
            ],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_phi3.Phi3Config",
                "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
                "AutoTokenizer": "Xenova/gpt-4o"
            },
            "bos_token_id": 199999,
            "embd_pdrop": 0.0,
            "eos_token_id": 200020,
            "full_attn_mod": 1,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "interpolate_factor": 1,
            "lm_head_bias": false,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "phi3",
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "original_max_position_embeddings": 4096,
            "pad_token_id": 199999,
            "partial_rotary_factor": 0.75,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "quantization_config": {
                "group_size": 64,
                "bits": 4
            },
            "resid_pdrop": 0.0,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "long_factor": [
                    1,
                    1.118320672,
                    1.250641126,
                    1.398617824,
                    1.564103225,
                    1.74916897,
                    1.956131817,
                    2.187582649,
                    2.446418898,
                    2.735880826,
                    3.059592084,
                    3.421605075,
                    3.826451687,
                    4.279200023,
                    4.785517845,
                    5.351743533,
                    5.984965424,
                    6.693110555,
                    7.485043894,
                    8.370679318,
                    9.36110372,
                    10.4687158,
                    11.70738129,
                    13.09260651,
                    14.64173252,
                    16.37415215,
                    18.31155283,
                    20.47818807,
                    22.90118105,
                    25.61086418,
                    28.64115884,
                    32.03,
                    32.1,
                    32.13,
                    32.23,
                    32.6,
                    32.61,
                    32.64,
                    32.66,
                    32.7,
                    32.71,
                    32.93,
                    32.97,
                    33.28,
                    33.49,
                    33.5,
                    44.16,
                    47.77
                ],
                "short_factor": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                "type": "longrope"
            },
            "rope_theta": 10000.0,
            "sliding_window": 262144,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.45.0",
            "use_cache": true,
            "vocab_size": 200064
        }
        """#.utf8)
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

    func testCompressedLongestRowDepartureMatchesUninterruptedScalarSurvivors() throws {
        let shared = try makeCompressedRuntime()
        let prompts: [UInt64: [Int]] = [
            1: Array(1 ... 32),
            2: Array(100 ... 147),
            3: Array(50 ... 73),
        ]
        for id in [UInt64(1), 2, 3] {
            try prefill(
                shared,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: 8)
        }

        let first = try shared.decode(
            .batch(
                [BatchRequestID(1), BatchRequestID(2), BatchRequestID(3)],
                speculationAllowed: false))
        XCTAssertEqual(shared.diagnostics().batchPhysicalWrittenEnd, 49)
        shared.remove(BatchRequestID(2))
        var continuations: [UInt64: [Int]] = [
            1: emitted(first)[1]!,
            3: emitted(first)[3]!,
        ]
        for _ in 0 ..< 3 {
            let next = try shared.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(3)],
                    speculationAllowed: false))
            continuations[1]!.append(contentsOf: emitted(next)[1]!)
            continuations[3]!.append(contentsOf: emitted(next)[3]!)
        }

        for id in [UInt64(1), 3] {
            let scalar = try makeCompressedRuntime()
            try prefill(
                scalar,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: 8)
            let expected = try (0 ..< 4).map { _ in
                try scalar.decode(
                    .solo(BatchRequestID(id), speculationAllowed: false))[0]
                    .tokens[0]
            }
            XCTAssertEqual(continuations[id], expected, "survivor \(id)")
        }
    }

    func testContinuousBatchRejectsKVarNUntilItsPackedBatchTransformIsQualified() {
        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: TinyCompressedBatchLanguageModel(),
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .kvarn(.k4v2G128I8),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: try makeCompressedBatchAdmission(),
                keyValueHeadCount: 2,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .unsupportedCompressedBatchCache)
        }
    }

    func testCompressedBatchRequiresAdmissionBeforeTouchingModelCacheState() {
        let model = TinyCompressedBatchLanguageModel()

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: model,
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                keyValueHeadCount: 2,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionRequired)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
    }

    func testCompressedBatchRejectsAdmissionGeometryBeforeTouchingModelCacheState() throws {
        let model = TinyCompressedBatchLanguageModel()
        let wrongGeometry = try makeCompressedBatchAdmission(
            keyValueHeadCount: 1)

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: model,
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: wrongGeometry,
                keyValueHeadCount: 2,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionMismatch)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
    }

    func testCompressedBatchRejectsLlamaAdmissionForQwenOnlyContinuousProof() throws {
        let model = TinyCompressedBatchLanguageModel()
        let llama = try makeCompressedBatchAdmission(
            modelType: "llama",
            architecture: "LlamaForCausalLM")

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: model,
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: llama,
                keyValueHeadCount: 2,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionMismatch)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
    }

    func testCompressedBatchRejectsPhi3AdmissionForQwenOnlyContinuousProof() throws {
        let model = TinyCompressedBatchLanguageModel()
        let phi3 = try makePhi3MiniCompressedBatchAdmission()

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: model,
                allocationChunk: 4,
                maxContextTokens: 131_072,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: phi3,
                layerCount: 32,
                keyValueHeadCount: 8,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionMismatch)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
    }

    func testCompressedBatchRejectsDifferentExactModelConfigBeforeCacheCreation() throws {
        let model = TinyCompressedBatchLanguageModel()
        let admission = try makeCompressedBatchAdmission()
        let proof = DenseContinuousBatchModelProof.testing(
            maxPositionEmbeddings: 64,
            vocabularySize: 2_048,
            modelConfigHash: "fedcba9876543210",
            modelConfigSHA256: String(repeating: "e", count: 64),
            keyValueHeadCount: 2,
            headDimension: 128,
            elementBytes: 2)

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                model: model,
                verifiedBy: proof,
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: admission)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionMismatch)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
    }

    func testCompressedBatchRejectsDifferentCheckpointContentBeforeCacheCreation() throws {
        let model = TinyCompressedBatchLanguageModel()
        let admission = try makeCompressedBatchAdmission()
        let proof = DenseContinuousBatchModelProof.testing(
            maxPositionEmbeddings: 64,
            vocabularySize: 2_048,
            modelConfigHash: admission.modelConfigHash,
            modelConfigSHA256: admission.modelConfigSHA256,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: admission.tokenizerSHA256,
            keyValueHeadCount: 2,
            headDimension: 128,
            elementBytes: 2)

        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                model: model,
                verifiedBy: proof,
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: 3,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission: admission)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchAdmissionMismatch)
        }
        XCTAssertEqual(model.newCacheCallCount, 0)
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

    func testConfigDerivedProofRejectsPhi3BeforeRuntimeConstruction() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try makePhi3MiniCompressedBatchSourceSnapshot()
        try source.exactModelConfigData.write(
            to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(
            try DenseContinuousBatchModelProof.verifying(
                modelDirectory: directory,
                stableCompressedSource: source)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .unsupportedModelFamily("phi3"))
        }
    }

    func testConfigDerivedProofBindsStableCheckpointAndTokenizerSource() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try makeCompressedBatchSourceSnapshot()
        try source.exactModelConfigData.write(
            to: directory.appendingPathComponent("config.json"))
        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: source)
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: directory,
            stableCompressedSource: source)
        let model = TinyCompressedBatchLanguageModel()

        _ = try DenseContinuousBatchRuntime(
            model: model,
            verifiedBy: proof,
            allocationChunk: 4,
            maxContextTokens: 64,
            initialDecodeReserve: 3,
            kvCacheKind: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM,
            compressedKVAttentionAdmission: admission)

        XCTAssertEqual(model.newCacheCallCount, 1)
    }

    func testConfigDerivedProofRejectsSourceSnapshotForDifferentConfigBytes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try makeCompressedBatchSourceSnapshot()
        var differentConfig = source.exactModelConfigData
        differentConfig.append(0x0A)
        try differentConfig.write(
            to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(
            try DenseContinuousBatchModelProof.verifying(
                modelDirectory: directory,
                stableCompressedSource: source)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .compressedBatchSourceProofMismatch)
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

    func testKVByteReservationRejectsAtomicallyAndReleasesOnRemoval() throws {
        let submission = ContinuousBatchSubmission(
            promptTokens: [10],
            maxOutputTokens: 1,
            eosToken: 2,
            architecture: .denseAttention)
        let admission = ContinuousBatchRuntimeAdmission(
            id: BatchRequestID(1), submission: submission)
        let rejected = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 199)

        XCTAssertThrowsError(try rejected.admit([admission])) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError,
                .aggregateKVByteLimitExceeded(requested: 200, limit: 199))
        }
        XCTAssertEqual(rejected.diagnostics().reservedKVBytes, 0)

        let accepted = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 200)
        try accepted.admit([admission])
        XCTAssertEqual(accepted.diagnostics().reservedKVBytes, 200)
        XCTAssertEqual(accepted.diagnostics().kvBytesPerToken, 8)
        accepted.remove(BatchRequestID(1))
        XCTAssertEqual(accepted.diagnostics().reservedKVBytes, 0)
    }

    func testKVGeometryCalibrationRejectsBeforeAdmissionReservation() throws {
        let runtime = try DenseContinuousBatchRuntime(
            model: TinyDenseLanguageModel(),
            verifiedBy: .testing(
                maxPositionEmbeddings: 16,
                vocabularySize: 2_048,
                keyValueHeadCount: 2),
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 1_000)
        let admission = ContinuousBatchRuntimeAdmission(
            id: BatchRequestID(1),
            submission: ContinuousBatchSubmission(
                promptTokens: [10],
                maxOutputTokens: 1,
                eosToken: 2,
                architecture: .denseAttention))

        XCTAssertThrowsError(try runtime.admit([admission])) { error in
            guard let runtimeError = error as? DenseContinuousBatchRuntimeError,
                case .cacheGeometryMismatch(let expectedHeads, _, _) = runtimeError
            else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(expectedHeads, 2)
        }
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 0)
    }

    func testRemovedBatchRowRetainsByteReservationUntilMembershipRebuild() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 400)
        let submission = ContinuousBatchSubmission(
            promptTokens: [10],
            maxOutputTokens: 2,
            eosToken: 2,
            architecture: .denseAttention)
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(id: BatchRequestID(1), submission: submission),
            ContinuousBatchRuntimeAdmission(id: BatchRequestID(2), submission: submission),
        ])
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 2)
        try prefill(runtime, id: 2, tokens: [10], chunks: [1], maxOutputTokens: 2)
        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 400)

        runtime.remove(BatchRequestID(2))
        XCTAssertEqual(
            runtime.diagnostics().reservedKVBytes,
            400,
            "old B=2 arrays remain live until the next membership boundary")

        _ = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 200)
    }

    func testMixedCapacitySurvivorRetainsPaddedPhysicalReservation() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 720)
        let short = ContinuousBatchSubmission(
            promptTokens: [10],
            maxOutputTokens: 2,
            eosToken: 2,
            architecture: .denseAttention)
        let long = ContinuousBatchSubmission(
            promptTokens: [20, 21, 22, 23, 24],
            maxOutputTokens: 2,
            eosToken: 2,
            architecture: .denseAttention)
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(id: BatchRequestID(1), submission: short),
            ContinuousBatchRuntimeAdmission(id: BatchRequestID(2), submission: long),
        ])
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 720)
        try prefill(runtime, id: 1, tokens: short.promptTokens, chunks: [1], maxOutputTokens: 2)
        try prefill(runtime, id: 2, tokens: long.promptTokens, chunks: [5], maxOutputTokens: 2)
        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))

        runtime.remove(BatchRequestID(2))
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 720)
        _ = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(
            runtime.diagnostics().reservedKVBytes,
            360,
            "the short survivor still owns a scalar cache padded to the removed row's capacity")
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
