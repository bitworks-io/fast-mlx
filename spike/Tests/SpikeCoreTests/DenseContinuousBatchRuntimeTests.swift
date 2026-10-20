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

        let flatInputs = inputs.asType(.int32)
        let historyBeforeChunk =
            keys.sum(axes: [1, 2, 3]).asType(.int32)
            - flatInputs.sum(axis: 1)
        let target = (
            historyBeforeChunk.reshaped([inputs.dim(0), 1])
                + flatInputs.cumsum(axis: 1)
                + 1
        ).expandedDimensions(axis: -1)
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

private struct TinyDenseExactDrafter: SpecDrafter {
    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard maxDraft > 0 else { return [] }
        var working = context
        var result: [Int] = []
        result.reserveCapacity(maxDraft)
        for _ in 0 ..< maxDraft {
            let next = working.reduce(0, +) + 1
            result.append(next)
            working.append(next)
        }
        return result
    }
}

private struct TinyDenseSingleDraftDrafter: SpecDrafter {
    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard maxDraft > 0 else { return [] }
        return [context.reduce(0, +) + 1]
    }
}

private struct TinyDenseColdAfterTokenDrafter: SpecDrafter {
    let coldLastToken: Int

    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard context.last != coldLastToken else { return [] }
        return TinyDenseExactDrafter().propose(
            context: context,
            maxDraft: maxDraft)
    }
}

private struct TinyDenseInvalidDrafter: SpecDrafter {
    func propose(context _: [Int], maxDraft: Int) -> [Int] {
        maxDraft > 0 ? [2_048] : []
    }
}

private struct TinyNextTokenDrafter: SpecDrafter {
    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard let last = context.last, maxDraft > 0 else { return [] }
        return (1 ... maxDraft).map { last + $0 }
    }
}

private struct TinyEmptyDrafter: SpecDrafter {
    func propose(context _: [Int], maxDraft _: Int) -> [Int] { [] }
}

private struct TinyNextTokenColdAfterTokenDrafter: SpecDrafter {
    let coldLastToken: Int

    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard context.last != coldLastToken else { return [] }
        return TinyNextTokenDrafter().propose(
            context: context,
            maxDraft: maxDraft)
    }
}

private final class TinyNonFiniteLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        guard let cache = cache?.first else {
            preconditionFailure("tiny non-finite model requires one cache")
        }
        _ = cache.update(keys: values, values: values)
        return MLXArray.full(
            [inputs.dim(0), inputs.dim(1), 2_048],
            values: MLXArray(Float.nan))
    }
}

private final class TinyDelayedNonFiniteLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let invalidInput: Int32

    init(invalidInput: Int32) {
        self.invalidInput = invalidInput
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        guard let cache = cache?.first else {
            preconditionFailure("tiny delayed non-finite model requires one cache")
        }
        _ = cache.update(keys: values, values: values)

        let targets = (inputs.asType(.int32) + 1).expandedDimensions(axis: -1)
        let vocabulary = MLXArray(Int32(0) ..< Int32(2_048))
            .reshaped([1, 1, 2_048])
        let finite = (targets .== vocabulary).asType(.float32) * 100
        let invalid = (inputs.asType(.int32) .== invalidInput)
            .expandedDimensions(axis: -1)
        return which(
            invalid,
            MLXArray.full(
                [inputs.dim(0), inputs.dim(1), 2_048],
                values: MLXArray(Float.nan)),
            finite)
    }
}

/// Like `TinyDelayedNonFiniteLanguageModel`, but the row at `invalidInput` stays FINITE in the
/// column the next token wants (input + 1) and carries `poison` in ONE other column. A mask that
/// disallows that column leaves the masked row's maximum finite, so only a guard on the RAW
/// logits can see the poison (`-inf` and `+inf` are the cases the masked maximum cannot catch).
private final class TinyDelayedPoisonedColumnLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let invalidInput: Int32
    private let poison: Float
    private let poisonColumn: Int32

    init(invalidInput: Int32, poison: Float, poisonColumn: Int32 = 500) {
        self.invalidInput = invalidInput
        self.poison = poison
        self.poisonColumn = poisonColumn
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        guard let cache = cache?.first else {
            preconditionFailure("tiny poisoned-column model requires one cache")
        }
        _ = cache.update(keys: values, values: values)

        let targets = (inputs.asType(.int32) + 1).expandedDimensions(axis: -1)
        let vocabulary = MLXArray(Int32(0) ..< Int32(2_048))
            .reshaped([1, 1, 2_048])
        let finite = (targets .== vocabulary).asType(.float32) * 100
        // A full-shaped array, not a scalar: the compiled step would inline a scalar NaN/inf as
        // a bare `nan`/`inf` identifier in its JIT Metal source, which does not build.
        let poisonRow = MLXArray.full([1, 1, 2_048], values: MLXArray(poison))
        let poisoned = which(vocabulary .== MLXArray(poisonColumn), poisonRow, finite)
        let invalid = (inputs.asType(.int32) .== invalidInput)
            .expandedDimensions(axis: -1)
        return which(invalid, poisoned, finite)
    }
}

final class DenseContinuousBatchRuntimeTests: XCTestCase {
    private func makeRuntime(
        allocationChunk: Int = 4,
        maxContextTokens: Int = 32_768,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil,
        soloPLDConfiguration: SpecDecodeConfig? = nil
    ) throws
        -> DenseContinuousBatchRuntime
    {
        try DenseContinuousBatchRuntime(
            testing: TinyDenseLanguageModel(),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            initialDecodeReserve: initialDecodeReserve,
            maxReservedKVBytes: maxReservedKVBytes,
            soloPLDConfiguration: soloPLDConfiguration)
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
            allocationChunk: 64,
            maxContextTokens: 64,
            initialDecodeReserve: 8,
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

    private func collect(_ stream: ContinuousBatchTokenStream) async throws -> [Int] {
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

    func testIncrementalSoloPLDMatchesScalarAndDrainsWithoutDuplicateToken() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 3,
                gate: PLDGate(
                    window: 8,
                    minimumSamples: 4,
                    minAcceptPerStep: 0.5,
                    cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let speculative = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let drain = try runtime.decode(
            .drainSoloPipeline(BatchRequestID(1)))

        XCTAssertEqual(first.map(\.tokens), [[11]])
        XCTAssertEqual(first.map(\.soloPipelineState), [.pipelinedLookahead])
        XCTAssertEqual(speculative.map(\.tokens), [[22, 44, 88, 176]])
        XCTAssertEqual(speculative.map(\.soloPipelineState), [.speculative])
        XCTAssertEqual(
            drain,
            [
                ContinuousBatchRuntimeDecodeResult(
                    id: BatchRequestID(1),
                    tokens: [],
                    finished: false,
                    soloPipelineState: .canonical),
            ])
        let engagedResources = try XCTUnwrap(runtime.resourceSnapshot())
        XCTAssertEqual(
            engagedResources.speculation,
            ContinuousBatchRuntimeSpeculationSnapshot(
                requestedRequests: 1,
                activeSessions: 1,
                draftedTokens: 3,
                acceptedDraftTokens: 3,
                verificationRounds: 1,
                fallbackRounds: 0))

        let reference = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16)
        try prefill(
            reference,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)
        let expected = try (0 ..< 6).map { _ in
            try reference.decode(
                .solo(BatchRequestID(1), speculationAllowed: false))[0]
                .tokens[0]
        }
        let afterDrain = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(
            first[0].tokens + speculative[0].tokens + afterDrain[0].tokens,
            expected)

        runtime.remove(BatchRequestID(1))
        let releasedResources = try XCTUnwrap(runtime.resourceSnapshot())
        XCTAssertEqual(releasedResources.speculation?.activeSessions, 0)
        XCTAssertEqual(releasedResources.speculation?.draftedTokens, 3)
        XCTAssertEqual(releasedResources.speculation?.verificationRounds, 1)
    }

    func testIncrementalSoloPLDDrainThenSharedBatchMatchesScalarControls() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 2,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(2),
                submission: ContinuousBatchSubmission(
                    promptTokens: [20],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 16)
        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let speculative = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        try prefill(runtime, id: 2, tokens: [20], chunks: [1], maxOutputTokens: 16)
        let drain = try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))
        let shared = try runtime.decode(
            .batch(
                [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))

        XCTAssertTrue(drain[0].tokens.isEmpty)
        XCTAssertEqual(drain[0].soloPipelineState, .canonical)

        var actual: [UInt64: [Int]] = [
            1: first[0].tokens + speculative[0].tokens,
            2: [],
        ]
        for result in shared {
            actual[result.id.rawValue, default: []].append(
                contentsOf: result.tokens)
        }

        for (id, prompt) in [(UInt64(1), 10), (UInt64(2), 20)] {
            let control = try makeRuntime(
                allocationChunk: 8,
                maxContextTokens: 128,
                initialDecodeReserve: 16)
            try prefill(
                control,
                id: id,
                tokens: [prompt],
                chunks: [1],
                maxOutputTokens: 16)
            let expectedCount = actual[id]!.count
            let expected = try (0 ..< expectedCount).map { _ in
                try control.decode(
                    .solo(
                        BatchRequestID(id),
                        speculationAllowed: false))[0].tokens[0]
            }
            XCTAssertEqual(actual[id], expected, "request \(id)")
        }
    }

    func testIncrementalSoloPLDRejectsLossyKVAtRuntimeConstruction() throws {
        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                testing: TinyCompressedBatchLanguageModel(),
                allocationChunk: 64,
                maxContextTokens: 64,
                initialDecodeReserve: 8,
                kvCacheKind: .affine(.k4v2G64),
                affineAttentionMode: .splitQuantizedMM,
                compressedKVAttentionAdmission:
                    try makeCompressedBatchAdmission(),
                soloPLDConfiguration: SpecDecodeConfig(),
                keyValueHeadCount: 2,
                headDimension: 128,
                elementBytes: 2)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .speculationUnsupported)
        }
    }

    func testIncrementalSoloPLDCompiledShortDraftPadsWithoutChangingScalarOutput()
        throws
    {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseSingleDraftDrafter(),
                maxDraft: 3,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let padded = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertEqual(first[0].tokens + padded[0].tokens, [11, 22, 44])
        XCTAssertEqual(padded[0].soloPipelineState, .speculative)
    }

    func testIncrementalSoloPLDLastOutputTokenFallsBackWithinContextLimit() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 12,
            initialDecodeReserve: 2,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 3,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 2,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 2)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let last = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertEqual(first[0].tokens + last[0].tokens, [11, 22])
        XCTAssertEqual(last[0].soloPipelineState, .pipelinedLookahead)
    }

    func testIncrementalSoloPLDBatchThenSoloReentryUsesFreshContext() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 2,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(2),
                submission: ContinuousBatchSubmission(
                    promptTokens: [20],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 16)
        try prefill(runtime, id: 2, tokens: [20], chunks: [1], maxOutputTokens: 16)
        let shared = try runtime.decode(
            .batch(
                [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))
        runtime.remove(BatchRequestID(2))
        let solo = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertEqual(shared[0].tokens + solo[0].tokens, [11, 22, 44, 88])
        XCTAssertEqual(solo[0].soloPipelineState, .speculative)
    }

    func testIncrementalSoloPLDColdFallbackFromSpeculativeMatchesScalar() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseColdAfterTokenDrafter(coldLastToken: 88),
                maxDraft: 2,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let speculative = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let fallback = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertEqual(
            first[0].tokens + speculative[0].tokens + fallback[0].tokens,
            [11, 22, 44, 88, 176])
        XCTAssertEqual(fallback[0].soloPipelineState, .pipelinedLookahead)
    }

    func testIncrementalSoloPLDReportsLowYieldDisableWhenGateTrips() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyNextTokenDrafter(),
                maxDraft: 1,
                gate: PLDGate(
                    window: 8,
                    minimumSamples: 4,
                    minAcceptPerStep: 0.5,
                    cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        XCTAssertEqual(first[0].observation, .none)

        let firstMiss = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let secondMiss = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let thirdMiss = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        let disablingMiss = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertEqual(firstMiss[0].observation, .none)
        XCTAssertEqual(secondMiss[0].observation, .none)
        XCTAssertEqual(thirdMiss[0].observation, .none)
        XCTAssertEqual(disablingMiss[0].observation, .pldLowYieldDisabled)
        XCTAssertEqual(disablingMiss[0].soloPipelineState, .speculative)
    }

    func testIncrementalSoloPLDReportsLowYieldDisableAfterEmptyLookupRounds() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyEmptyDrafter(),
                maxDraft: 1,
                gate: PLDGate(
                    window: 8,
                    minimumSamples: 4,
                    minAcceptPerStep: 0.5,
                    cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)

        let first = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        XCTAssertEqual(first[0].observation, .none)

        for sample in 1 ... 4 {
            let result = try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: true))
            XCTAssertEqual(
                result[0].observation,
                sample == 4 ? .pldLowYieldDisabled : .none)
            XCTAssertEqual(
                result[0].soloPipelineState,
                .pipelinedLookahead)
        }
    }

    func testIncrementalSoloPLDRemovalIsSafeFromEveryCacheInvariant() throws {
        for decodeRounds in 0 ... 2 {
            let runtime = try makeRuntime(
                allocationChunk: 8,
                maxContextTokens: 128,
                initialDecodeReserve: 16,
                soloPLDConfiguration: SpecDecodeConfig(
                    drafter: TinyDenseExactDrafter(),
                    maxDraft: 2,
                    lookback: 32,
                    compiledVerify: true))
            let id = BatchRequestID(1)
            try runtime.admit([
                ContinuousBatchRuntimeAdmission(
                    id: id,
                    submission: ContinuousBatchSubmission(
                        promptTokens: [10],
                        maxOutputTokens: 16,
                        eosToken: 2,
                        architecture: .denseAttention,
                        requestsSpeculation: true)),
            ])
            try prefill(
                runtime,
                id: id.rawValue,
                tokens: [10],
                chunks: [1],
                maxOutputTokens: 16)
            for _ in 0 ..< decodeRounds {
                _ = try runtime.decode(
                    .solo(id, speculationAllowed: true))
            }

            runtime.remove(id)
            try runtime.admit([
                ContinuousBatchRuntimeAdmission(
                    id: id,
                    submission: ContinuousBatchSubmission(
                        promptTokens: [5],
                        maxOutputTokens: 4,
                        eosToken: 2,
                        architecture: .denseAttention)),
            ])
            try prefill(
                runtime,
                id: id.rawValue,
                tokens: [5],
                chunks: [1],
                maxOutputTokens: 4)
            XCTAssertEqual(
                try runtime.decode(
                    .solo(id, speculationAllowed: false))[0].tokens,
                [6],
                "decode rounds \(decodeRounds)")
        }
    }

    func testIncrementalSoloPLDRejectsInvalidDraftTokenBeforeVerification() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseInvalidDrafter(),
                maxDraft: 1,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)
        _ = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))

        XCTAssertThrowsError(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: true))
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .invalidTokenID(BatchRequestID(1), 2_048))
        }
    }

    func testIncrementalSoloPLDRollsBackKVAfterInvalidVerifyToken() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 12),
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyNextTokenDrafter(),
                maxDraft: 1,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)
        XCTAssertEqual(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: true))[0].tokens,
            [11])
        let before = runtime.diagnosticScalarLogicalOffset(
            for: BatchRequestID(1))

        for _ in 0 ..< 2 {
            XCTAssertThrowsError(
                try runtime.decode(
                    .solo(BatchRequestID(1), speculationAllowed: true))
            ) { error in
                XCTAssertEqual(
                    error as? DenseContinuousBatchRuntimeError,
                    .invalidTokenID(BatchRequestID(1), -1))
            }
            XCTAssertEqual(
                runtime.diagnosticScalarLogicalOffset(
                    for: BatchRequestID(1)),
                before)
        }
    }

    func testIncrementalSoloPLDRollsBackKVAfterInvalidSpeculativeFallbackToken()
        throws
    {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 13),
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyNextTokenColdAfterTokenDrafter(coldLastToken: 13),
                maxDraft: 1,
                lookback: 32,
                compiledVerify: true))
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: ContinuousBatchSubmission(
                    promptTokens: [10],
                    maxOutputTokens: 16,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: true)),
        ])
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 16)
        XCTAssertEqual(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: true))[0].tokens,
            [11])
        let speculative = try runtime.decode(
            .solo(BatchRequestID(1), speculationAllowed: true))
        XCTAssertEqual(speculative[0].tokens, [12, 13])
        XCTAssertEqual(speculative[0].soloPipelineState, .speculative)
        let before = runtime.diagnosticScalarLogicalOffset(
            for: BatchRequestID(1))

        for _ in 0 ..< 2 {
            XCTAssertThrowsError(
                try runtime.decode(
                    .solo(BatchRequestID(1), speculationAllowed: true))
            ) { error in
                XCTAssertEqual(
                    error as? DenseContinuousBatchRuntimeError,
                    .invalidTokenID(BatchRequestID(1), -1))
            }
            XCTAssertEqual(
                runtime.diagnosticScalarLogicalOffset(
                    for: BatchRequestID(1)),
                before)
        }
    }

    func testContinuousRuntimeRejectsNonFiniteGreedyToken() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyNonFiniteLanguageModel(),
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 4)

        XCTAssertThrowsError(
            try prefill(
                runtime,
                id: 1,
                tokens: [10],
                chunks: [1],
                maxOutputTokens: 4)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .invalidTokenID(BatchRequestID(1), -1))
        }
    }

    func testScalarRejectsStagedNonFiniteTokenBeforeItMutatesKV() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 11),
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 8)
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 4)

        XCTAssertEqual(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: false))[0].tokens,
            [11])
        let before = runtime.diagnosticScalarLogicalOffset(
            for: BatchRequestID(1))

        XCTAssertThrowsError(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: false))
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .invalidTokenID(BatchRequestID(1), -1))
        }
        XCTAssertEqual(
            runtime.diagnosticScalarLogicalOffset(for: BatchRequestID(1)),
            before)
    }

    func testBatchRejectsStagedNonFiniteTokensBeforeTheyMutateKV() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 11),
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 8)
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        try prefill(runtime, id: 2, tokens: [10], chunks: [1])

        XCTAssertEqual(
            emitted(
                try runtime.decode(
                    .batch(
                        [BatchRequestID(1), BatchRequestID(2)],
                        speculationAllowed: false))),
            [1: [11], 2: [11]])
        let before = runtime.diagnostics().batchPhysicalWrittenEnd

        XCTAssertThrowsError(
            try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(2)],
                    speculationAllowed: false))
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .invalidTokenID(BatchRequestID(1), -1))
        }
        XCTAssertEqual(
            runtime.diagnostics().batchPhysicalWrittenEnd,
            before)
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

    func testRemovedRequestIDReuseRebuildsPhysicalBatchRow() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        try prefill(runtime, id: 2, tokens: [20], chunks: [1])

        XCTAssertEqual(
            emitted(
                try runtime.decode(
                    .batch(
                        [BatchRequestID(1), BatchRequestID(2)],
                        speculationAllowed: false))),
            [1: [11], 2: [21]])
        runtime.remove(BatchRequestID(2))

        try prefill(runtime, id: 2, tokens: [100], chunks: [1])
        let replacement = try runtime.decode(
            .batch(
                [BatchRequestID(1), BatchRequestID(2)],
                speculationAllowed: false))

        XCTAssertEqual(
            emitted(replacement),
            [1: [22], 2: [101]],
            "a reused request ID must not inherit the removed request's staged token or KV row")
    }

    func testReusedRemovedRequestIDDoesNotBlockSurvivorScalarExtraction() throws {
        let runtime = try makeRuntime()
        try prefill(runtime, id: 1, tokens: [10], chunks: [1])
        try prefill(runtime, id: 2, tokens: [20], chunks: [1])
        XCTAssertEqual(
            emitted(
                try runtime.decode(
                    .batch(
                        [BatchRequestID(1), BatchRequestID(2)],
                        speculationAllowed: false))),
            [1: [11], 2: [21]])

        runtime.remove(BatchRequestID(2))
        try prefill(runtime, id: 2, tokens: [100], chunks: [1])

        XCTAssertEqual(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: false))[0].tokens,
            [22],
            "a replacement incarnation must not keep the removed physical row live")
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

    func testStableFixedCapacityBatchMembershipCompilesOnce() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 256,
            initialDecodeReserve: 2)
        try prefill(
            runtime, id: 1, tokens: [10, 11, 12], chunks: [1, 2],
            maxOutputTokens: 2)
        try prefill(
            runtime, id: 2, tokens: [20], chunks: [1],
            maxOutputTokens: 2)

        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        _ = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))

        let diagnostics = runtime.diagnostics()
        XCTAssertEqual(diagnostics.batchTraceCount, 1)
        XCTAssertEqual(diagnostics.batchMembership, [BatchRequestID(1), BatchRequestID(2)])
        XCTAssertEqual(diagnostics.batchCapacity, 8)
    }

    func testDecodeCohortUsesExactFixedCapacityAndIsolatesGrowthEligibleRequests()
        throws
    {
        let runtime = try makeRuntime(
            allocationChunk: 256,
            maxContextTokens: 4_096,
            initialDecodeReserve: 384)

        func admission(
            id: UInt64,
            promptCount: Int,
            outputCount: Int
        ) -> ContinuousBatchRuntimeAdmission {
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: Array(repeating: 10, count: promptCount),
                    maxOutputTokens: outputCount,
                    eosToken: 2,
                    architecture: .denseAttention))
        }

        XCTAssertEqual(
            try runtime.decodeCohort(
                for: admission(id: 1, promptCount: 40, outputCount: 128)),
            .fixedKVCapacity(256))
        XCTAssertEqual(
            try runtime.decodeCohort(
                for: admission(id: 2, promptCount: 100, outputCount: 128)),
            .fixedKVCapacity(256))
        XCTAssertEqual(
            try runtime.decodeCohort(
                for: admission(id: 3, promptCount: 200, outputCount: 128)),
            .fixedKVCapacity(512))
        XCTAssertEqual(
            try runtime.decodeCohort(
                for: admission(id: 4, promptCount: 40, outputCount: 385)),
            .isolated(BatchRequestID(4)))
    }

    func testDirectMixedCapacityBatchFailsClosedBeforeCacheMerge() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 64,
            initialDecodeReserve: 8)
        try prefill(
            runtime,
            id: 1,
            tokens: [10],
            chunks: [1],
            maxOutputTokens: 2)
        try prefill(
            runtime,
            id: 2,
            tokens: [20, 21, 22, 23, 24],
            chunks: [5],
            maxOutputTokens: 2)

        XCTAssertThrowsError(
            try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(2)],
                    speculationAllowed: false))
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .incompatibleDecodeCohort(
                    [BatchRequestID(1), BatchRequestID(2)]))
        }
        XCTAssertNil(runtime.diagnostics().batchCapacity)
    }

    func testSameCapacityHostileB3B2B3MatchesScalarControls() throws {
        let prompts: [UInt64: [Int]] = [
            1: [10],
            2: [30, 31, 32],
            3: [20, 21],
            4: [40, 41],
        ]
        let outputCount = 8

        func scalarControl(
            id: UInt64,
            count: Int
        ) throws -> [Int] {
            let runtime = try makeRuntime(
                allocationChunk: 16,
                maxContextTokens: 64,
                initialDecodeReserve: outputCount)
            try prefill(
                runtime,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: outputCount)
            return try (0 ..< count).flatMap { _ in
                try runtime.decode(
                    .solo(
                        BatchRequestID(id),
                        speculationAllowed: false)
                )[0].tokens
            }
        }

        let controls = try Dictionary(
            uniqueKeysWithValues: prompts.keys.map {
                ($0, try scalarControl(id: $0, count: 6))
            })
        let runtime = try makeRuntime(
            allocationChunk: 16,
            maxContextTokens: 64,
            initialDecodeReserve: outputCount)
        for id in [UInt64(1), 2, 3] {
            try prefill(
                runtime,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: outputCount)
        }

        var actual: [UInt64: [Int]] = [1: [], 2: [], 3: [], 4: []]
        for _ in 0 ..< 2 {
            for result in try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(2), BatchRequestID(3)],
                    speculationAllowed: false))
            {
                actual[result.id.rawValue, default: []].append(
                    contentsOf: result.tokens)
            }
        }

        runtime.remove(BatchRequestID(2))
        for _ in 0 ..< 2 {
            for result in try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(3)],
                    speculationAllowed: false))
            {
                actual[result.id.rawValue, default: []].append(
                    contentsOf: result.tokens)
            }
        }

        try prefill(
            runtime,
            id: 4,
            tokens: prompts[4]!,
            chunks: [prompts[4]!.count],
            maxOutputTokens: outputCount)
        for _ in 0 ..< 2 {
            for result in try runtime.decode(
                .batch(
                    [BatchRequestID(1), BatchRequestID(3), BatchRequestID(4)],
                    speculationAllowed: false))
            {
                actual[result.id.rawValue, default: []].append(
                    contentsOf: result.tokens)
            }
        }

        XCTAssertEqual(actual[1], controls[1])
        XCTAssertEqual(actual[2], Array(controls[2]!.prefix(2)))
        XCTAssertEqual(actual[3], controls[3])
        XCTAssertEqual(actual[4], Array(controls[4]!.prefix(2)))
        XCTAssertEqual(runtime.diagnostics().batchCapacity, 16)
    }

    func testTwoFixedCapacityCohortsCanAlternateSharedBatchesWithoutStateLoss()
        throws
    {
        let prompts: [UInt64: [Int]] = [
            1: [10],
            2: [20, 21],
            3: [30, 31, 32, 33, 34],
            4: [40, 41, 42, 43, 44, 45],
        ]
        let outputCount = 8

        func scalarControl(
            id: UInt64,
            count: Int
        ) throws -> [Int] {
            let runtime = try makeRuntime(
                allocationChunk: 4,
                maxContextTokens: 64,
                initialDecodeReserve: outputCount)
            try prefill(
                runtime,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: outputCount)
            return try (0 ..< count).flatMap { _ in
                try runtime.decode(
                    .solo(
                        BatchRequestID(id),
                        speculationAllowed: false)
                )[0].tokens
            }
        }

        let controls = try Dictionary(
            uniqueKeysWithValues: prompts.keys.map {
                ($0, try scalarControl(id: $0, count: 3))
            })
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 64,
            initialDecodeReserve: outputCount)
        for id in prompts.keys.sorted() {
            try prefill(
                runtime,
                id: id,
                tokens: prompts[id]!,
                chunks: [prompts[id]!.count],
                maxOutputTokens: outputCount)
        }

        let cohortA = [BatchRequestID(1), BatchRequestID(2)]
        let cohortB = [BatchRequestID(3), BatchRequestID(4)]
        var actual: [UInt64: [Int]] = [
            1: [], 2: [], 3: [], 4: [],
        ]
        for _ in 0 ..< 3 {
            for ids in [cohortA, cohortB] {
                for result in try runtime.decode(
                    .batch(ids, speculationAllowed: false))
                {
                    actual[result.id.rawValue, default: []].append(
                        contentsOf: result.tokens)
                }
            }
        }

        XCTAssertEqual(actual[1], controls[1])
        XCTAssertEqual(actual[2], controls[2])
        XCTAssertEqual(actual[3], controls[3])
        XCTAssertEqual(actual[4], controls[4])
        XCTAssertEqual(runtime.diagnostics().batchMembership, cohortB)
        XCTAssertEqual(runtime.diagnostics().batchCapacity, 16)
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

    func testConfigDerivedProofRejectsVLWrappedHybridAsUnsupportedFamilyNotInvalidConfig() throws {
        // qwen3_5 ships VL-wrapped: root model_type is "qwen3_5" but the geometry
        // (max_position_embeddings/vocab_size/num_hidden_layers) lives under text_config.
        // A valid config for an unsupported family must fail as unsupportedModelFamily —
        // NOT invalidModelConfiguration — so the continuous→scalar serving fallback, which
        // keys on unsupportedModelFamily, can engage. Regression for a044970: the strict
        // Configuration decode used to fail on the absent top-level fields first and mask
        // the family behind invalidModelConfiguration.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(
            #"""
            {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
             "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
             "vocab_size":248320,"num_hidden_layers":32}}
            """#.utf8
        ).write(to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(
            try DenseContinuousBatchModelProof.verifying(modelDirectory: directory)) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .unsupportedModelFamily("qwen3_5"))
        }
    }

    func testConfigDerivedProofStillRejectsMalformedQwen3AsInvalidConfig() throws {
        // The family gate must NOT mask a genuinely broken SUPPORTED config: qwen3 with a
        // missing required top-level field is invalidModelConfiguration, not a family miss.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(
            // model_type=qwen3 (supported) but max_position_embeddings absent.
            #"{"model_type":"qwen3","vocab_size":2048}"#.utf8
        ).write(to: directory.appendingPathComponent("config.json"))

        XCTAssertThrowsError(
            try DenseContinuousBatchModelProof.verifying(modelDirectory: directory)) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidModelConfiguration)
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

    func testPerRequestReservedContextLimitRejectsBeforeAggregateAccounting()
        throws
    {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDenseLanguageModel(),
            allocationChunk: 4,
            maxContextTokens: 32,
            maxReservedContextTokens: 8,
            initialDecodeReserve: 2)
        let admission = ContinuousBatchRuntimeAdmission(
            id: BatchRequestID(7),
            submission: ContinuousBatchSubmission(
                promptTokens: [10, 11, 12, 13, 14],
                maxOutputTokens: 4,
                eosToken: 2,
                architecture: .denseAttention))

        XCTAssertThrowsError(try runtime.admit([admission])) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError,
                .requestReservedContextLimitExceeded(
                    BatchRequestID(7),
                    requested: 9,
                    limit: 8))
        }
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 0)
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
                .requestReservedKVByteLimitExceeded(
                    BatchRequestID(1),
                    requested: 200,
                    limit: 199))
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

    func testAggregateKVByteReservationDistinguishesTransientPressure()
        throws
    {
        let submission = ContinuousBatchSubmission(
            promptTokens: [10],
            maxOutputTokens: 1,
            eosToken: 2,
            architecture: .denseAttention)
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 1,
            maxReservedKVBytes: 300)
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(1),
                submission: submission)
        ])

        XCTAssertThrowsError(
            try runtime.admit([
                ContinuousBatchRuntimeAdmission(
                    id: BatchRequestID(2),
                    submission: submission)
            ])
        ) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError,
                .aggregateKVByteLimitExceeded(
                    requested: 400,
                    limit: 300))
        }
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 200)
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
            initialDecodeReserve: 2,
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

    func testReplacementIDDoesNotKeepFullyRemovedPhysicalBatchReserved() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 2,
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

        runtime.remove(BatchRequestID(1))
        try prefill(runtime, id: 1, tokens: [100], chunks: [1], maxOutputTokens: 2)
        runtime.remove(BatchRequestID(2))

        XCTAssertTrue(runtime.diagnostics().batchMembership.isEmpty)
        XCTAssertNil(runtime.diagnostics().batchCapacity)
        XCTAssertEqual(
            runtime.diagnostics().reservedKVBytes,
            200,
            "a replacement incarnation must not keep a fully removed physical batch reserved")
        XCTAssertEqual(
            try runtime.decode(
                .solo(BatchRequestID(1), speculationAllowed: false))[0].tokens,
            [101])
    }

    func testDifferentCapacityRowsRemainIndependentAndReleaseOwnReservations() throws {
        let runtime = try makeRuntime(
            allocationChunk: 4,
            maxContextTokens: 16,
            initialDecodeReserve: 2,
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
            .solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertNil(runtime.diagnostics().batchCapacity)

        runtime.remove(BatchRequestID(2))
        XCTAssertEqual(
            runtime.diagnostics().reservedKVBytes,
            200,
            "removing the wider isolated cohort releases its reservation without padding the survivor")
        runtime.remove(BatchRequestID(1))
        XCTAssertEqual(runtime.diagnostics().reservedKVBytes, 0)
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

    func testCoordinatorTrimsDensePLDBatchAtEOSAndReusesReleasedRuntime()
        async throws
    {
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: 1,
                maxPrefillSlots: 1,
                prefillChunkSize: 4),
            runtime: try DenseContinuousBatchRuntime(
                testing: TinyDenseLanguageModel(),
                allocationChunk: 8,
                maxContextTokens: 128,
                initialDecodeReserve: 16,
                soloPLDConfiguration: SpecDecodeConfig(
                    drafter: TinyDenseExactDrafter(),
                    maxDraft: 2,
                    lookback: 32,
                    compiledVerify: true)),
            automaticDrive: false,
            traceLimit: 32)
        let speculative = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: [10],
                maxOutputTokens: 16,
                stopTokenIDs: [44],
                architecture: .denseAttention,
                requestsSpeculation: true))

        while try await coordinator.runOneTick() {}

        let speculativeTokens = try await collect(speculative.tokens)
        let speculativeSnapshots = await coordinator.snapshots()
        let speculativeResources = await coordinator.runtimeResourceSnapshot()
        XCTAssertEqual(speculativeTokens, [11, 22])
        XCTAssertTrue(speculativeSnapshots.isEmpty)
        XCTAssertEqual(speculativeResources?.reservedKVBytes, 0)
        let trace = await coordinator.executionTrace()
        XCTAssertTrue(
            trace.contains(
                .operation(
                    .decode(
                        .solo(
                            speculative.id,
                            speculationAllowed: true)))))
        XCTAssertTrue(trace.contains(.finished(speculative.id)))

        let replacement = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: [5],
                maxOutputTokens: 2,
                stopTokenIDs: [127],
                architecture: .denseAttention))
        while try await coordinator.runOneTick() {}

        let replacementTokens = try await collect(replacement.tokens)
        let replacementSnapshots = await coordinator.snapshots()
        let replacementResources = await coordinator.runtimeResourceSnapshot()
        XCTAssertEqual(replacementTokens, [6, 12])
        XCTAssertTrue(replacementSnapshots.isEmpty)
        XCTAssertEqual(replacementResources?.reservedKVBytes, 0)
    }

    // MARK: - Per-row additive logit_bias (cycle 269)
    //
    // The tiny model's logits are 100 at exactly one id (the greedy token), 0 elsewhere. Prompt
    // [10] decodes 11, 22, 44, 88 unbiased; prompt [50] decodes 51, 102, 204, 408.

    private func admitWithBias(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        tokens: [Int],
        bias: [Int: Float],
        requestsSpeculation: Bool = false,
        maxOutputTokens: Int = 8
    ) throws {
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: tokens,
                    maxOutputTokens: maxOutputTokens,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: requestsSpeculation,
                    logitBias: bias)),
        ])
    }

    private func soloBiasedTokens(
        prompt: [Int], bias: [Int: Float], count: Int = 4
    ) throws -> [Int] {
        let runtime = try makeRuntime()
        try admitWithBias(runtime, id: 1, tokens: prompt, bias: bias)
        try prefill(runtime, id: 1, tokens: prompt, chunks: [prompt.count])
        return try (0 ..< count).map { _ in
            try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))[0]
                .tokens[0]
        }
    }

    /// Two-row batch (ids 1 and 2); returns each id's tokens over `count` shared steps.
    private func batchTokens(
        prompts: [[Int]], biases: [[Int: Float]], count: Int = 4
    ) throws -> [UInt64: [Int]] {
        let runtime = try makeRuntime()
        for (index, prompt) in prompts.enumerated() {
            let id = UInt64(index + 1)
            try admitWithBias(runtime, id: id, tokens: prompt, bias: biases[index])
            try prefill(runtime, id: id, tokens: prompt, chunks: [prompt.count])
        }
        let ids = prompts.indices.map { BatchRequestID(UInt64($0 + 1)) }
        var tokens: [UInt64: [Int]] = [:]
        for _ in 0 ..< count {
            for (id, emittedTokens) in emitted(
                try runtime.decode(.batch(ids, speculationAllowed: false)))
            {
                tokens[id, default: []].append(contentsOf: emittedTokens)
            }
        }
        return tokens
    }

    func testLogitBiasMinusOnGreedyTokenChangesPrefillFinalAndDecodedTokens() throws {
        let unbiased = try soloBiasedTokens(prompt: [10], bias: [:])
        XCTAssertEqual(unbiased, [11, 22, 44, 88])

        let biased = try soloBiasedTokens(prompt: [10], bias: [11: -100])

        XCTAssertEqual(biased.count, unbiased.count)
        // Index 0 is the prefill-final token; later indices come from the scalar step.
        for (index, pair) in zip(biased, unbiased).enumerated() {
            XCTAssertNotEqual(pair.0, pair.1, "token \(index) ignored the -100 bias")
        }
    }

    func testLogitBiasPlusOnChosenIDForcesItSoloAndInABatch() throws {
        let solo = try soloBiasedTokens(prompt: [10], bias: [7: 200])
        XCTAssertEqual(solo, [7, 7, 7, 7])

        let batch = try batchTokens(
            prompts: [[10], [50]], biases: [[7: 200], [:]])
        XCTAssertEqual(batch[1], [7, 7, 7, 7])
        XCTAssertEqual(batch[2], [51, 102, 204, 408])

        // The biased row in the second physical position.
        let swapped = try batchTokens(
            prompts: [[50], [10]], biases: [[:], [9: 200]])
        XCTAssertEqual(swapped[1], [51, 102, 204, 408])
        XCTAssertEqual(swapped[2], [9, 9, 9, 9])
    }

    func testLogitBiasSurvivesBatchToSoloDeparture() throws {
        let runtime = try makeRuntime()
        for (id, prompt, bias) in [
            (UInt64(1), [10], [Int: Float]()),
            (UInt64(2), [50], [7: Float(200)]),
        ] {
            try admitWithBias(runtime, id: id, tokens: prompt, bias: bias)
            try prefill(runtime, id: id, tokens: prompt, chunks: [prompt.count])
        }
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        var biased: [Int] = []
        for _ in 0 ..< 2 {
            biased += emitted(try runtime.decode(.batch(ids, speculationAllowed: false)))[2] ?? []
        }
        runtime.remove(BatchRequestID(1))
        for _ in 0 ..< 2 {
            biased.append(
                try runtime.decode(.solo(BatchRequestID(2), speculationAllowed: false))[0]
                    .tokens[0])
        }
        XCTAssertEqual(biased, [7, 7, 7, 7])
    }

    func testUnbiasedRowIsBitIdenticalSoloInAllUnbiasedBatchAndBesideABiasedRow() throws {
        let solo = try soloBiasedTokens(prompt: [10], bias: [:])
        let allUnbiased = try batchTokens(
            prompts: [[10], [50]], biases: [[:], [:]])
        let besideBiased = try batchTokens(
            prompts: [[10], [50]], biases: [[:], [7: 200]])
        let biasedFirst = try batchTokens(
            prompts: [[50], [10]], biases: [[7: 200], [:]])

        XCTAssertEqual(solo, [11, 22, 44, 88])
        XCTAssertEqual(allUnbiased[1], solo)
        XCTAssertEqual(besideBiased[1], solo)
        XCTAssertEqual(biasedFirst[2], solo)
        XCTAssertEqual(besideBiased[2], [7, 7, 7, 7])
        XCTAssertEqual(biasedFirst[1], [7, 7, 7, 7])
    }

    func testAllZeroLogitBiasMapDecodesIdenticallyToNoMap() throws {
        let none = try soloBiasedTokens(prompt: [10], bias: [:])
        let zero = try soloBiasedTokens(prompt: [10], bias: [5: 0, 11: 0])
        XCTAssertEqual(zero, none)

        let batchNone = try batchTokens(prompts: [[10], [50]], biases: [[:], [:]])
        let batchZero = try batchTokens(
            prompts: [[10], [50]], biases: [[5: 0], [11: 0]])
        XCTAssertEqual(batchZero, batchNone)
    }

    func testLogitBiasWithSpeculationIsRefusedAndAdmitsNothing() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 3,
                gate: PLDGate(
                    window: 8,
                    minimumSamples: 4,
                    minAcceptPerStep: 0.5,
                    cooldown: 8),
                lookback: 32,
                compiledVerify: true))

        XCTAssertThrowsError(
            try admitWithBias(
                runtime, id: 1, tokens: [10], bias: [7: 5], requestsSpeculation: true)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .logitBiasWithSpeculation(BatchRequestID(1)))
        }
        let refused = runtime.resourceSnapshot()
        XCTAssertEqual(refused?.reservedKVBytes, 0)
        XCTAssertEqual(refused?.speculation?.requestedRequests, 0)
        XCTAssertEqual(refused?.speculation?.activeSessions, 0)

        // The refusal left nothing behind: the same id admits cleanly afterwards.
        try admitWithBias(runtime, id: 1, tokens: [10], bias: [:])
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
    }

    func testLogitBiasIDOutsideTheVocabularyIsRefusedAtAdmission() throws {
        let runtime = try makeRuntime()
        for bad in [2_048, 100_000, -1] {
            XCTAssertThrowsError(
                try admitWithBias(runtime, id: 1, tokens: [10], bias: [3: 1, bad: 1])
            ) { error in
                XCTAssertEqual(
                    error as? DenseContinuousBatchRuntimeError,
                    .logitBiasTokenIDOutOfRange(BatchRequestID(1), tokenID: bad))
            }
        }
        XCTAssertEqual(runtime.resourceSnapshot()?.reservedKVBytes, 0)

        // The last valid id is accepted.
        try admitWithBias(runtime, id: 1, tokens: [10], bias: [2_047: 1])
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
    }

    // MARK: - Per-row, per-step token mask (cycle 274)
    //
    // `ScriptedMask` allows exactly the script's ids for the row's next token and records every
    // `advance`. The step index is the number of advances so far, so the bitset a step sees
    // reflects every earlier emitted token. The tiny model's greedy token is irrelevant while the
    // script allows a single id: -inf everywhere else leaves that id as the only finite column.

    private func admitWithMask(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        tokens: [Int],
        mask: (any ContinuousBatchRowTokenMask)?,
        bias: [Int: Float] = [:],
        requestsSpeculation: Bool = false,
        maxOutputTokens: Int = 16
    ) throws {
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: tokens,
                    maxOutputTokens: maxOutputTokens,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: requestsSpeculation,
                    logitBias: bias,
                    tokenMask: mask)),
        ])
    }

    private func solo(_ runtime: DenseContinuousBatchRuntime, _ id: UInt64) throws -> Int {
        try runtime.decode(.solo(BatchRequestID(id), speculationAllowed: false))[0].tokens[0]
    }

    private func drain(_ runtime: DenseContinuousBatchRuntime, _ id: UInt64) throws -> Int {
        try runtime.decode(.drainSoloPipeline(BatchRequestID(id)))[0].tokens[0]
    }

    /// Solo decode of `count` tokens: pipelined steps with a synchronous drain every 4th token.
    private func soloMaskedTokens(
        prompt: [Int], mask: (any ContinuousBatchRowTokenMask)?,
        bias: [Int: Float] = [:], count: Int
    ) throws -> [Int] {
        let runtime = try makeRuntime()
        try admitWithMask(runtime, id: 1, tokens: prompt, mask: mask, bias: bias)
        try prefill(runtime, id: 1, tokens: prompt, chunks: [prompt.count], maxOutputTokens: 16)
        var tokens: [Int] = []
        for index in 0 ..< count {
            if index % 4 == 3 {
                tokens.append(try drain(runtime, 1))
            } else {
                tokens.append(try solo(runtime, 1))
            }
        }
        return tokens
    }

    /// Shared batch over ids 1...n; returns each id's tokens over `count` shared steps.
    private func maskedBatchTokens(
        prompts: [[Int]], masks: [(any ContinuousBatchRowTokenMask)?],
        biases: [[Int: Float]]? = nil, count: Int
    ) throws -> [UInt64: [Int]] {
        let runtime = try makeRuntime()
        for (index, prompt) in prompts.enumerated() {
            let id = UInt64(index + 1)
            try admitWithMask(
                runtime, id: id, tokens: prompt, mask: masks[index],
                bias: biases?[index] ?? [:])
            try prefill(runtime, id: id, tokens: prompt, chunks: [prompt.count], maxOutputTokens: 16)
        }
        let ids = prompts.indices.map { BatchRequestID(UInt64($0 + 1)) }
        var tokens: [UInt64: [Int]] = [:]
        for _ in 0 ..< count {
            for (id, emittedTokens) in emitted(
                try runtime.decode(.batch(ids, speculationAllowed: false)))
            {
                tokens[id, default: []].append(contentsOf: emittedTokens)
            }
        }
        return tokens
    }

    func testScriptedMaskForcesTheExactScriptSoloThroughPrefillFinalPipelineAndDrain() throws {
        let script = [7, 9, 5, 3, 7, 9, 5, 3, 13]
        let mask = ScriptedMask(allowed: script.map { [$0] })

        let tokens = try soloMaskedTokens(prompt: [10], mask: mask, count: script.count)

        XCTAssertEqual(tokens, script)
        // The staged-but-unemitted lookahead token is never advanced.
        XCTAssertEqual(mask.advanced, script)
    }

    func testScriptedMaskForcesTheScriptOnABatchRowBesideAnUnmaskedPartner() throws {
        let script = [7, 9, 5, 3, 13, 21]
        let masked = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [ScriptedMask(allowed: script.map { [$0] }), nil],
            count: script.count)
        // Same batch shape, an unmasked control in the masked row's place.
        let control = try maskedBatchTokens(
            prompts: [[10], [50]], masks: [nil, nil], count: script.count)
        XCTAssertEqual(masked[1], script)
        XCTAssertEqual(masked[2], control[2])
        XCTAssertEqual(masked[2], [51, 102, 204, 408, 816, 1_632])

        // The masked row in the second physical position.
        let swapped = try maskedBatchTokens(
            prompts: [[50], [10]],
            masks: [nil, ScriptedMask(allowed: script.map { [$0] })],
            count: script.count)
        XCTAssertEqual(swapped[2], script)
        XCTAssertEqual(swapped[1], control[2])
    }

    func testBothRowsMaskedDifferentlyGetTheirOwnPerStepRows() throws {
        let first = [7, 9, 5, 3]
        let second = [40, 41, 42, 43]
        let tokens = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [
                ScriptedMask(allowed: first.map { [$0] }),
                ScriptedMask(allowed: second.map { [$0] }),
            ],
            count: 4)
        XCTAssertEqual(tokens[1], first)
        XCTAssertEqual(tokens[2], second)
    }

    func testAdvanceReceivesExactlyTheEmittedTokensAcrossSoloBatchSoloTransitions() throws {
        let runtime = try makeRuntime()
        let scriptOne = [7, 9, 5, 3, 13, 21, 8, 6, 4, 2, 17, 19]
        let scriptTwo = [40, 41, 42, 43, 44, 45, 46, 47]
        let maskOne = ScriptedMask(allowed: scriptOne.map { [$0] })
        let maskTwo = ScriptedMask(allowed: scriptTwo.map { [$0] })
        try admitWithMask(runtime, id: 1, tokens: [10], mask: maskOne)
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 16)
        try admitWithMask(runtime, id: 2, tokens: [50], mask: maskTwo)
        try prefill(runtime, id: 2, tokens: [50], chunks: [1], maxOutputTokens: 16)

        var emittedOne: [Int] = []
        var emittedTwo: [Int] = []
        // Solo: id 1 alone through a pipelined step, a pipelined step and a drain.
        emittedOne.append(try solo(runtime, 1))
        emittedOne.append(try solo(runtime, 1))
        emittedOne.append(try drain(runtime, 1))
        XCTAssertEqual(maskOne.advanced, emittedOne)
        XCTAssertEqual(maskTwo.advanced, [])

        // The partner joins.
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        for _ in 0 ..< 3 {
            let step = emitted(try runtime.decode(.batch(ids, speculationAllowed: false)))
            emittedOne += step[1] ?? []
            emittedTwo += step[2] ?? []
            XCTAssertEqual(maskOne.advanced, emittedOne)
            XCTAssertEqual(maskTwo.advanced, emittedTwo)
        }

        // The partner leaves; id 1 is solo again.
        runtime.remove(BatchRequestID(2))
        emittedOne.append(try solo(runtime, 1))
        emittedOne.append(try solo(runtime, 1))
        emittedOne.append(try drain(runtime, 1))

        XCTAssertEqual(emittedOne, Array(scriptOne.prefix(emittedOne.count)))
        XCTAssertEqual(emittedTwo, Array(scriptTwo.prefix(emittedTwo.count)))
        XCTAssertEqual(maskOne.advanced, emittedOne)
        XCTAssertEqual(maskTwo.advanced, emittedTwo)
        XCTAssertEqual(emittedOne.count, 9)
        XCTAssertEqual(emittedTwo.count, 3)
    }

    func testMaskThatAlwaysReturnsNilDecodesIdenticallyToNoMaskSoloAndInABatch() throws {
        let none = try soloMaskedTokens(prompt: [10], mask: nil, count: 5)
        let nilMask = ScriptedMask(allowed: [nil])
        let masked = try soloMaskedTokens(prompt: [10], mask: nilMask, count: 5)
        XCTAssertEqual(none, [11, 22, 44, 88, 176])
        XCTAssertEqual(masked, none)
        XCTAssertEqual(nilMask.advanced, none)

        let batchNone = try maskedBatchTokens(
            prompts: [[10], [50]], masks: [nil, nil], count: 4)
        let batchMasks = [ScriptedMask(allowed: [nil]), ScriptedMask(allowed: [nil])]
        let batchMasked = try maskedBatchTokens(
            prompts: [[10], [50]], masks: batchMasks, count: 4)
        XCTAssertEqual(batchMasked, batchNone)
        XCTAssertEqual(batchMasks[0].advanced, batchNone[1])
        XCTAssertEqual(batchMasks[1].advanced, batchNone[2])
    }

    func testMaskComposesWithBiasDisallowedPlusHundredNeverWinsAndBiasOrdersAllowedIDs() throws {
        let allowed = [[20, 21, 22]]
        // +100 on a disallowed id never wins; +3 on 21 orders the allowed ids.
        let bias: [Int: Float] = [5: 100, 21: 3, 22: 2]
        let soloTokens = try soloMaskedTokens(
            prompt: [10], mask: ScriptedMask(allowed: allowed), bias: bias, count: 4)
        XCTAssertEqual(soloTokens, [21, 21, 21, 21])

        let reordered = try soloMaskedTokens(
            prompt: [10], mask: ScriptedMask(allowed: allowed),
            bias: [5: 100, 22: 3, 21: 2], count: 4)
        XCTAssertEqual(reordered, [22, 22, 22, 22])

        let batch = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [ScriptedMask(allowed: allowed), nil],
            biases: [bias, [:]], count: 4)
        XCTAssertEqual(batch[1], [21, 21, 21, 21])
        XCTAssertEqual(batch[2], [51, 102, 204, 408])

        // Bias on a masked row beside a biased-only partner.
        let biasedPartner = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [ScriptedMask(allowed: allowed), nil],
            biases: [bias, [7: 200]], count: 4)
        XCTAssertEqual(biasedPartner[1], [21, 21, 21, 21])
        XCTAssertEqual(biasedPartner[2], [7, 7, 7, 7])
    }

    func testTokenMaskWithSpeculationIsRefusedAndAdmitsNothing() throws {
        let runtime = try makeRuntime(
            allocationChunk: 8,
            maxContextTokens: 128,
            initialDecodeReserve: 16,
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: TinyDenseExactDrafter(),
                maxDraft: 3,
                gate: PLDGate(
                    window: 8,
                    minimumSamples: 4,
                    minAcceptPerStep: 0.5,
                    cooldown: 8),
                lookback: 32,
                compiledVerify: true))

        XCTAssertThrowsError(
            try admitWithMask(
                runtime, id: 1, tokens: [10], mask: ScriptedMask(allowed: [[7]]),
                requestsSpeculation: true)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .tokenMaskWithSpeculation(BatchRequestID(1)))
        }
        let refused = runtime.resourceSnapshot()
        XCTAssertEqual(refused?.reservedKVBytes, 0)
        XCTAssertEqual(refused?.speculation?.requestedRequests, 0)
        XCTAssertEqual(refused?.speculation?.activeSessions, 0)

        // The refusal left nothing behind: the same id admits cleanly afterwards.
        try admitWithMask(runtime, id: 1, tokens: [10], mask: nil)
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
    }

    func testMaskIDsBeyondTheLogitsWidthAreIgnoredAndColumnsBeyondTheBitsetNeverWin() throws {
        // The tiny model's logits are 2048 wide. Ids 2048, 3000 and 4095 sit beyond the logits;
        // only 13 is a real column.
        let wide = ScriptedMask.bitset(of: [13, 2_048, 3_000, 4_095])
        XCTAssertEqual(wide.count, 64)
        let tokens = try soloMaskedTokens(
            prompt: [10], mask: ScriptedMask(bitsets: [wide]), count: 4)
        XCTAssertEqual(tokens, [13, 13, 13, 13])
        let batch = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [ScriptedMask(bitsets: [wide]), nil], count: 3)
        XCTAssertEqual(batch[1], [13, 13, 13])
        XCTAssertEqual(batch[2], [51, 102, 204])

        // A one-word bitset (ids 0..<64) leaves every logits column from 64 up at -inf, even the
        // model's own 100-valued greedy column (11 first, then 22, 44).
        let narrow = ScriptedMask.bitset(of: [60, 61])
        XCTAssertEqual(narrow.count, 1)
        let narrowTokens = try soloMaskedTokens(
            prompt: [10], mask: ScriptedMask(bitsets: [narrow]), count: 5)
        XCTAssertEqual(Set(narrowTokens).subtracting([60, 61]), [])
        let narrowBatch = try maskedBatchTokens(
            prompts: [[10], [50]],
            masks: [ScriptedMask(bitsets: [narrow]), ScriptedMask(bitsets: [narrow])],
            count: 4)
        for tokens in narrowBatch.values {
            XCTAssertEqual(Set(tokens).subtracting([60, 61]), [])
            XCTAssertTrue(tokens.allSatisfy { $0 < 64 })
        }
    }

    // MARK: - Hardening: a mask with no allowed column inside the logits width (cycle 274 follow-up)

    func testMaskWithNoColumnInsideTheLogitsWidthIsUnconstrainedAndNeverFailsTheStep() throws {
        // The tiny model's logits are 2048 wide. Every set bit sits at or beyond it (or there is
        // no set bit at all), so the row builder has no allowed column. Such a row must decode
        // exactly like an unmasked row instead of reading back the `-1` sentinel and throwing
        // `invalidTokenID`, which the coordinator would turn into `failAll(terminal: true)`.
        let bitsets: [[UInt64]] = [
            ScriptedMask.bitset(of: [2_048, 3_000, 4_095]),
            ScriptedMask.bitset(of: [2_048]),
            [0],
            [],
        ]
        let soloControl = try soloMaskedTokens(prompt: [10], mask: nil, count: 6)
        XCTAssertEqual(soloControl, [11, 22, 44, 88, 176, 352])
        let batchControl = try maskedBatchTokens(
            prompts: [[10], [50]], masks: [nil, nil], count: 4)

        for bitset in bitsets {
            // (a) Solo: prefill-final, pipelined steps and a drain do not throw.
            let soloMask = ScriptedMask(bitsets: [bitset])
            let solo = try soloMaskedTokens(prompt: [10], mask: soloMask, count: 6)
            XCTAssertEqual(solo, soloControl)
            // (c) `advance` still receives every emitted token.
            XCTAssertEqual(soloMask.advanced, solo)

            // (b) In a batch beside an unmasked partner, in either physical position.
            let first = ScriptedMask(bitsets: [bitset])
            let masked = try maskedBatchTokens(
                prompts: [[10], [50]], masks: [first, nil], count: 4)
            XCTAssertEqual(masked[1], batchControl[1])
            XCTAssertEqual(masked[2], batchControl[2])
            XCTAssertEqual(first.advanced, masked[1])

            let second = ScriptedMask(bitsets: [bitset])
            let swapped = try maskedBatchTokens(
                prompts: [[10], [50]], masks: [nil, second], count: 4)
            XCTAssertEqual(swapped[1], batchControl[1])
            XCTAssertEqual(swapped[2], batchControl[2])
            XCTAssertEqual(second.advanced, swapped[2])
        }
    }

    func testMaskThatLeavesTheLogitsWidthMidStreamIsUnconstrainedForThatStepOnly() throws {
        // Step 0 allows 7, step 1 has only out-of-width ids (unconstrained: the model's own
        // greedy token follows 10, 7), step 2 allows 9 again.
        let beyond = ScriptedMask.bitset(of: [3_000])
        let script: [[UInt64]] = [
            ScriptedMask.bitset(of: [7]), beyond, ScriptedMask.bitset(of: [9]),
        ]
        let solo = ScriptedMask(bitsets: script)
        let soloTokens = try soloMaskedTokens(prompt: [10], mask: solo, count: 3)
        XCTAssertEqual(soloTokens.first, 7)
        XCTAssertEqual(soloTokens.last, 9)
        XCTAssertEqual(solo.advanced, soloTokens)

        let batchMask = ScriptedMask(bitsets: script)
        let batch = try maskedBatchTokens(
            prompts: [[10], [50]], masks: [batchMask, nil], count: 3)
        XCTAssertEqual(batch[1], soloTokens)
        XCTAssertEqual(batch[2], [51, 102, 204])
        XCTAssertEqual(batchMask.advanced, soloTokens)
    }

    // MARK: - Hardening: the raw-logit finite guard on masked rows (cycle 274 follow-up)

    /// Solo decode of two tokens on `model` with `mask`: the first decode must emit `[11]`, the
    /// second reads back the `-1` sentinel. Returns the second decode's error.
    private func soloSentinelError(
        model: some LanguageModel & KVCacheDimensionProvider & Module,
        mask: (any ContinuousBatchRowTokenMask)?
    ) throws -> DenseContinuousBatchRuntimeError? {
        let runtime = try DenseContinuousBatchRuntime(
            testing: model, allocationChunk: 4, maxContextTokens: 16, initialDecodeReserve: 8)
        try admitWithMask(runtime, id: 1, tokens: [10], mask: mask, maxOutputTokens: 4)
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 4)
        XCTAssertEqual(try solo(runtime, 1), 11)
        do {
            _ = try solo(runtime, 1)
            return nil
        } catch {
            return error as? DenseContinuousBatchRuntimeError
        }
    }

    /// Shared batch of ids 1 and 2 (both prompt `[10]`, id 1 carries `mask`): the first step must
    /// emit `[11]` for both, the second reads back the `-1` sentinel. Returns the second error.
    private func batchSentinelError(
        model: some LanguageModel & KVCacheDimensionProvider & Module,
        mask: (any ContinuousBatchRowTokenMask)?
    ) throws -> DenseContinuousBatchRuntimeError? {
        let runtime = try DenseContinuousBatchRuntime(
            testing: model, allocationChunk: 4, maxContextTokens: 16, initialDecodeReserve: 8)
        try admitWithMask(runtime, id: 1, tokens: [10], mask: mask, maxOutputTokens: 4)
        try prefill(runtime, id: 1, tokens: [10], chunks: [1], maxOutputTokens: 4)
        try admitWithMask(runtime, id: 2, tokens: [10], mask: nil, maxOutputTokens: 4)
        try prefill(runtime, id: 2, tokens: [10], chunks: [1], maxOutputTokens: 4)
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        XCTAssertEqual(
            emitted(try runtime.decode(.batch(ids, speculationAllowed: false))),
            [1: [11], 2: [11]])
        do {
            _ = try runtime.decode(.batch(ids, speculationAllowed: false))
            return nil
        } catch {
            return error as? DenseContinuousBatchRuntimeError
        }
    }

    func testMaskedRowsFailClosedOnNonFiniteRawLogitsExactlyLikeUnmaskedRows() throws {
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        // Step 0 allows 11 (the model's own token), step 1 allows 12, which stays the finite
        // column of the poisoned row; the poison sits in column 500, which the mask disallows.
        func script() -> ScriptedMask { ScriptedMask(allowed: [[11], [12]]) }
        let fixtures: [(name: String, model: () -> any (LanguageModel & KVCacheDimensionProvider & Module))] = [
            ("NaN row", { TinyDelayedNonFiniteLanguageModel(invalidInput: 11) }),
            (
                "NaN column",
                { TinyDelayedPoisonedColumnLanguageModel(invalidInput: 11, poison: .nan) }
            ),
            (
                "+inf column",
                { TinyDelayedPoisonedColumnLanguageModel(invalidInput: 11, poison: .infinity) }
            ),
            (
                "-inf column",
                { TinyDelayedPoisonedColumnLanguageModel(invalidInput: 11, poison: -.infinity) }
            ),
        ]
        for fixture in fixtures {
            // Unmasked controls pin the parity: the sentinel and the error are the same.
            XCTAssertEqual(
                try soloSentinelError(model: fixture.model(), mask: nil), sentinel,
                "unmasked solo control: \(fixture.name)")
            XCTAssertEqual(
                try batchSentinelError(model: fixture.model(), mask: nil), sentinel,
                "unmasked batch control: \(fixture.name)")
            // Masked solo row and masked row in a batch.
            XCTAssertEqual(
                try soloSentinelError(model: fixture.model(), mask: script()), sentinel,
                "masked solo: \(fixture.name)")
            XCTAssertEqual(
                try batchSentinelError(model: fixture.model(), mask: script()), sentinel,
                "masked batch: \(fixture.name)")
        }
    }
}

/// Test double for `ContinuousBatchRowTokenMask`: the bitset for the row's NEXT token is
/// `bitsets[advanced.count]` (the last entry repeats), and every `advance` is recorded.
private final class ScriptedMask: ContinuousBatchRowTokenMask, @unchecked Sendable {
    private let lock = NSLock()
    private let bitsets: [[UInt64]?]
    private var recorded: [Int] = []

    init(bitsets: [[UInt64]?]) { self.bitsets = bitsets }

    /// One entry per step: the allowed ids, or `nil` for unconstrained.
    convenience init(allowed: [[Int]?]) {
        self.init(bitsets: allowed.map { $0.map(ScriptedMask.bitset(of:)) })
    }

    static func bitset(of ids: [Int]) -> [UInt64] {
        var words = [UInt64](repeating: 0, count: (ids.max() ?? 0) / 64 + 1)
        for id in ids { words[id >> 6] |= UInt64(1) << UInt64(id & 63) }
        return words
    }

    var advanced: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func allowedTokenBitset() -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        return bitsets[min(recorded.count, bitsets.count - 1)]
    }

    func advance(token: Int) {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(token)
    }
}

// MARK: - Non-final prefill slices do not compute the lm_head

/// Two-layer random-weight decoder with a REAL lm_head matmul, RoPE through the cache's
/// `ropeOffset`, and causal attention over the cache. Unlike the integer toy models above, its
/// forward is fully lazy and value-rich, so a prefill that evaluated the wrong arrays (or fewer
/// than the next slice reads) would change values. Equal seeds give equal weights.
private final class RandomWeightDenseProbeModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    static let vocab = 2_048
    static let dim = 16
    static let heads = 2
    static let headDim = 8
    static let layers = 2

    let kvHeads = Array(repeating: RandomWeightDenseProbeModel.heads, count: layers)
    private let rope = RoPE(dimensions: headDim, traditional: false, base: 10_000)
    private let embedding: MLXArray
    private let wq: [MLXArray]
    private let wk: [MLXArray]
    private let wv: [MLXArray]
    private let wo: [MLXArray]
    private let head: MLXArray

    init(seed: UInt64) {
        MLXRandom.seed(seed)
        func weight(_ shape: [Int]) -> MLXArray {
            let array = MLXRandom.normal(shape) * 0.4
            eval(array)
            return array
        }
        let d = Self.dim
        embedding = weight([Self.vocab, d])
        wq = (0 ..< Self.layers).map { _ in weight([d, d]) }
        wk = (0 ..< Self.layers).map { _ in weight([d, d]) }
        wv = (0 ..< Self.layers).map { _ in weight([d, d]) }
        wo = (0 ..< Self.layers).map { _ in weight([d, d]) }
        head = weight([d, Self.vocab])
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.count == Self.layers else {
            preconditionFailure("probe model requires one cache per layer")
        }
        let t = inputs.dim(1)
        var h = embedding[inputs]  // [1, T, D]
        for layer in 0 ..< Self.layers {
            func split(_ x: MLXArray) -> MLXArray {
                x.reshaped([1, t, Self.heads, Self.headDim]).transposed(0, 2, 1, 3)
            }
            let layerCache = cache[layer]
            let q = applyRotaryPosition(
                rope, to: split(matmul(h, wq[layer])), cache: layerCache)
            let k = applyRotaryPosition(
                rope, to: split(matmul(h, wk[layer])), cache: layerCache)
            let v = split(matmul(h, wv[layer]))
            let mask = layerCache.makeMask(n: t, windowSize: nil, returnArray: true)
            let attended = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: layerCache,
                scale: 1 / Float(Self.headDim).squareRoot(), mask: mask)
            let merged = attended.transposed(0, 2, 1, 3).reshaped([1, t, Self.dim])
            h = h + matmul(merged, wo[layer])
        }
        return matmul(h, head)  // [1, T, V]
    }
}

/// Compact exact-equality report for two arrays (never `XCTAssertEqual` on large collections).
private func mismatch(_ label: String, _ actual: MLXArray, _ expected: MLXArray) -> String? {
    guard actual.shape == expected.shape else {
        return "\(label): shape \(actual.shape) != \(expected.shape)"
    }
    guard !arrayEqual(actual, expected).item(Bool.self) else { return nil }
    let wrong = (actual .!= expected).asType(.int32).sum().item(Int.self)
    return "\(label): \(wrong)/\(actual.size) elements differ"
}

final class DenseContinuousBatchRuntimePrefillSliceTests: XCTestCase {
    private static let continuation = 6
    private static let prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9]  // 13 = 5 + 5 + 3

    private func slice(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, tokens: [Int], from start: Int,
        count: Int, maxOutputTokens: Int = 8
    ) throws {
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(id),
                startToken: start,
                tokens: Array(tokens[start ..< start + count]),
                isFinal: start + count == tokens.count,
                totalPromptTokens: tokens.count,
                maxOutputTokens: maxOutputTokens))
    }

    private func makeProbeRuntime(seed: UInt64) throws -> DenseContinuousBatchRuntime {
        try DenseContinuousBatchRuntime(
            testing: RandomWeightDenseProbeModel(seed: seed),
            allocationChunk: 4,
            maxContextTokens: 64,
            initialDecodeReserve: 384,
            layerCount: RandomWeightDenseProbeModel.layers,
            keyValueHeadCount: RandomWeightDenseProbeModel.heads,
            headDimension: RandomWeightDenseProbeModel.headDim,
            elementBytes: 4)
    }

    private func decodeSolo(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, count: Int
    ) throws -> [Int] {
        try (0 ..< count).map { _ in
            try runtime.decode(.solo(BatchRequestID(id), speculationAllowed: false))[0].tokens[0]
        }
    }

    // MARK: T1 — non-final slice logits are never consumed

    /// `TinyDelayedNonFiniteLanguageModel` emits NaN logits at every position whose INPUT token is
    /// 11 while its KV writes stay finite. Token 11 closes the first (non-final) slice, so before
    /// the change the staged-token read of that slice's last row hit the non-finite guard and the
    /// prefill threw. A non-final slice's logits feed nothing, so the prefill must succeed and the
    /// FINAL slice must stage the token from its own (finite) last row.
    func testNonFiniteLogitsOnANonFinalSliceAreNeverRead() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 11),
            allocationChunk: 4, maxContextTokens: 32, initialDecodeReserve: 4)
        let tokens = [10, 11, 12, 13]

        try slice(runtime, id: 1, tokens: tokens, from: 0, count: 2)  // ends on 11: NaN row
        try slice(runtime, id: 1, tokens: tokens, from: 2, count: 1)
        try slice(runtime, id: 1, tokens: tokens, from: 3, count: 1)

        XCTAssertEqual(try decodeSolo(runtime, id: 1, count: 2), [14, 15])
    }

    /// The accepted trade is only on NON-final slices: the guard still runs on the final slice.
    func testNonFiniteLogitsOnTheFinalSliceStillFailClosed() throws {
        let runtime = try DenseContinuousBatchRuntime(
            testing: TinyDelayedNonFiniteLanguageModel(invalidInput: 13),
            allocationChunk: 4, maxContextTokens: 32, initialDecodeReserve: 4)
        let tokens = [10, 11, 12, 13]
        try slice(runtime, id: 1, tokens: tokens, from: 0, count: 2)
        try slice(runtime, id: 1, tokens: tokens, from: 2, count: 1)

        XCTAssertThrowsError(try slice(runtime, id: 1, tokens: tokens, from: 3, count: 1)) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError,
                .invalidTokenID(BatchRequestID(1), -1))
        }
        XCTAssertEqual(runtime.diagnosticPrefillProgress(for: BatchRequestID(1))?.processedTokens, 3)
        XCTAssertEqual(
            runtime.diagnosticPrefillProgress(for: BatchRequestID(1))?.prefillComplete, false)
    }

    // MARK: T2 — exactness against the evaluate-every-slice oracle and the scalar control

    func testRaggedSlicedPrefillIsExactAgainstEvaluateEverySliceOracleAndScalarControl() throws {
        let seed: UInt64 = 20_261_011
        let prompt = Self.prompt
        let slices = [5, 5, 3]
        XCTAssertEqual(slices.reduce(0, +), prompt.count)

        // Production path: sliced prefill, then solo decode.
        let runtime = try makeProbeRuntime(seed: seed)
        var start = 0
        for count in slices {
            try slice(runtime, id: 1, tokens: prompt, from: start, count: count)
            start += count
        }
        let prefillState = try XCTUnwrap(runtime.diagnosticScalarCacheState(for: BatchRequestID(1)))
        eval(prefillState.flatMap { $0 })
        let runtimeTokens = try decodeSolo(runtime, id: 1, count: Self.continuation)

        // Oracle (a): the PRE-CHANGE behaviour — same model weights, same cache class and capacity,
        // same slicing, but EVERY slice's logits are evaluated (the removed work).
        let oracleModel = RandomWeightDenseProbeModel(seed: seed)
        let capacity = prefillState[0][0].dim(2)
        let oracleCaches = (0 ..< RandomWeightDenseProbeModel.layers).map { _ in
            CompiledKVCache(capacity: capacity)
        }
        var oracleStaged: MLXArray!
        start = 0
        for count in slices {
            let ids = MLXArray(prompt[start ..< start + count].map(Int32.init))
                .reshaped([1, count])
            let logits = oracleModel(ids, cache: oracleCaches)
            eval(logits)
            oracleStaged = CompiledMLXDecoder.greedyTokenOrInvalidSentinel(logits[0..., -1, 0...])
            eval(oracleStaged)
            start += count
        }
        var problems: [String] = []
        XCTAssertEqual(prefillState.count, oracleCaches.count)
        for (layer, cache) in oracleCaches.enumerated() {
            let expected = cache.innerState()
            XCTAssertEqual(prefillState[layer].count, expected.count)
            for (index, array) in expected.enumerated() where index < prefillState[layer].count {
                if let message = mismatch(
                    "layer \(layer) innerState[\(index)] vs evaluate-every-slice oracle",
                    prefillState[layer][index], array)
                {
                    problems.append(message)
                }
            }
        }
        var oracleTokens = [oracleStaged.item(Int.self)]
        while oracleTokens.count < Self.continuation {
            let y = MLXArray([Int32(oracleTokens.last!)]).reshaped([1, 1])
            let logits = oracleModel(y, cache: oracleCaches)
            oracleTokens.append(argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
        }

        // Control (b): scalar route — one unsliced forward over a plain `KVCacheSimple` stack.
        let controlModel = RandomWeightDenseProbeModel(seed: seed)
        let controlCaches: [KVCache] = (0 ..< RandomWeightDenseProbeModel.layers).map { _ in
            KVCacheSimple()
        }
        let controlIDs = MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count])
        let controlLogits = controlModel(controlIDs, cache: controlCaches)
        var controlTokens = [argMax(controlLogits[0..., -1, 0...], axis: -1).item(Int.self)]
        while controlTokens.count < Self.continuation {
            let y = MLXArray([Int32(controlTokens.last!)]).reshaped([1, 1])
            let logits = controlModel(y, cache: controlCaches)
            controlTokens.append(argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
        }

        if runtimeTokens != oracleTokens {
            problems.append("tokens \(runtimeTokens) != evaluate-every-slice oracle \(oracleTokens)")
        }
        if runtimeTokens != controlTokens {
            problems.append("tokens \(runtimeTokens) != scalar control \(controlTokens)")
        }
        XCTAssertEqual(Set(runtimeTokens).count > 1, true, "degenerate fixture: constant tokens")
        XCTAssertTrue(problems.isEmpty, problems.joined(separator: "; "))
    }

    // MARK: T3 — bookkeeping and error paths are unchanged

    func testNonFinalSlicesAdvanceBookkeepingAndLeaveNoStagedToken() throws {
        let runtime = try makeRuntime()
        let id = BatchRequestID(1)
        let tokens = [10, 11, 12, 13, 14]

        try slice(runtime, id: 1, tokens: tokens, from: 0, count: 2)
        XCTAssertEqual(
            runtime.diagnosticPrefillProgress(for: id),
            .init(processedTokens: 2, cachedTokens: 2, prefillComplete: false, hasStagedToken: false))
        XCTAssertEqual(runtime.diagnosticScalarLogicalOffset(for: id), 2)
        XCTAssertThrowsError(try runtime.decode(.solo(id, speculationAllowed: false))) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError, .decodeBeforeFinalPrefill(id))
        }

        // Out-of-order slice: rejected, bookkeeping untouched.
        XCTAssertThrowsError(try slice(runtime, id: 1, tokens: tokens, from: 3, count: 2)) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError,
                .outOfOrderPrefill(id, expected: 2, actual: 3))
        }
        XCTAssertEqual(runtime.diagnosticPrefillProgress(for: id)?.processedTokens, 2)

        try slice(runtime, id: 1, tokens: tokens, from: 2, count: 2)
        XCTAssertEqual(
            runtime.diagnosticPrefillProgress(for: id),
            .init(processedTokens: 4, cachedTokens: 4, prefillComplete: false, hasStagedToken: false))
        XCTAssertEqual(runtime.diagnosticScalarLogicalOffset(for: id), 4)

        try slice(runtime, id: 1, tokens: tokens, from: 4, count: 1)
        XCTAssertEqual(
            runtime.diagnosticPrefillProgress(for: id),
            .init(processedTokens: 5, cachedTokens: 5, prefillComplete: true, hasStagedToken: true))
        XCTAssertEqual(runtime.diagnosticScalarLogicalOffset(for: id), 5)

        XCTAssertThrowsError(try slice(runtime, id: 1, tokens: tokens, from: 4, count: 1)) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError, .prefillAfterFinalization(id))
        }
        // The staged first token is the final slice's own: 10+11+12+13+14 + 1.
        XCTAssertEqual(
            try runtime.decode(.solo(id, speculationAllowed: false)).map(\.tokens), [[61]])
    }

    private func makeRuntime() throws -> DenseContinuousBatchRuntime {
        try DenseContinuousBatchRuntime(
            testing: TinyDenseLanguageModel(),
            allocationChunk: 4, maxContextTokens: 32_768, initialDecodeReserve: 384)
    }
}
