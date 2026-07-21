import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import HarnessCore
@testable import SpikeCore
@testable import fastmlx_harness

private final class TinyScoringRouterModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let vocabularySize = 32
    private let kvDType: DType

    init(kvDType: DType = .float16) {
        self.kvDType = kvDType
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        guard let layerCache = cache?.first, cache?.count == 1 else {
            preconditionFailure("tiny scoring model requires one cache")
        }
        let scalar = inputs.asType(kvDType).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let keys = broadcast(
            scalar, to: [inputs.dim(0), 1, inputs.dim(1), 128])
        let queries = broadcast(
            scalar, to: [inputs.dim(0), 2, inputs.dim(1), 128])
        let mask = layerCache.makeMask(
            n: inputs.dim(1), windowSize: nil, returnArray: true)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: keys,
            cache: layerCache,
            scale: Float(1 / sqrt(128.0)),
            mask: mask)
        let signal = attended.mean(axes: [1, 3], keepDims: false)
            .expandedDimensions(axis: -1)
        let target = inputs.asType(.int32).reshaped([
            inputs.dim(0), inputs.dim(1), 1,
        ])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
            + signal.asType(.float32)
    }
}

final class CompressedKVScoringChunkTests: XCTestCase {
    func testDirectKVarNScoringCapacityReservesOnePostSinkToken() {
        XCTAssertEqual(
            HarnessEngineActor.scoringCacheCapacity(
                requested: 4,
                kind: .kvarn(.k4v2G128I8),
                kvarnAttentionMode: .splitQuantizedMM),
            KVarNKVTier.k4v2G128.sinkTokens + 1)
        XCTAssertEqual(
            HarnessEngineActor.scoringCacheCapacity(
                requested: KVarNKVTier.k4v2G128.sinkTokens,
                kind: .kvarn(.k4v2G128I8),
                kvarnAttentionMode: .splitQuantizedMM),
            KVarNKVTier.k4v2G128.sinkTokens + 1)
        XCTAssertEqual(
            HarnessEngineActor.scoringCacheCapacity(
                requested: KVarNKVTier.k4v2G128.sinkTokens + 1,
                kind: .kvarn(.k4v2G128I8),
                kvarnAttentionMode: .splitQuantizedMM),
            KVarNKVTier.k4v2G128.sinkTokens + 1)
        XCTAssertEqual(
            HarnessEngineActor.scoringCacheCapacity(
                requested: 4,
                kind: .kvarn(.k4v2G128I8),
                kvarnAttentionMode: .materialize),
            4)
        XCTAssertEqual(
            HarnessEngineActor.scoringCacheCapacity(
                requested: 4,
                kind: .affine(.k4v2G64),
                kvarnAttentionMode: .splitQuantizedMM),
            4)
    }

    func testKVarNScoringTelemetryIsRetainedPerAttentionMode() async throws {
        let actor = HarnessEngineActor(model: TinyScoringRouterModel())
        let materializedPrompt = (0..<256).map { ($0 % 31) + 1 }
        let directPrompt = (0..<129).map { ($0 % 31) + 1 }

        _ = await actor.logprobs(
            prompt: materializedPrompt,
            maxTokens: 1,
            eos: -1,
            kvCache: .kvarn(.k4v2G128I8),
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .materialize)
        _ = await actor.logprobs(
            prompt: directPrompt,
            maxTokens: 1,
            eos: -1,
            kvCache: .kvarn(.k4v2G128I8),
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .splitQuantizedMM)

        let materialized = await actor.kvarnScoringTelemetry(
            for: .k4v2G128I8,
            attentionMode: .materialize)
        let direct = await actor.kvarnScoringTelemetry(
            for: .k4v2G128I8,
            attentionMode: .splitQuantizedMM)
        XCTAssertEqual(materialized?.capacityTokens, 257)
        XCTAssertEqual(materialized?.attentionOperation, .materializedKV)
        XCTAssertEqual(direct?.capacityTokens, 130)
        XCTAssertEqual(direct?.attentionOperation, .splitQuantizedMM)
    }

    func testKVarNScoringCarriesAuthenticatedBFloat16IngressReceipt()
        async throws
    {
        let actor = HarnessEngineActor(
            model: TinyScoringRouterModel(kvDType: .float32))
        let prompt = (0..<256).map { ($0 % 31) + 1 }

        let task = await actor.taskChoiceLogits(
            prompt: prompt,
            kvCache: .kvarn(.k4v2G128I8),
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .splitQuantizedMM,
            kvarnStorageDType: .bfloat16)

        XCTAssertEqual(
            task.engagement.counts["scoring_kvarn_source_key_float32"], 1)
        XCTAssertEqual(
            task.engagement.counts["scoring_kvarn_source_value_float32"], 1)
        XCTAssertEqual(
            task.engagement.counts["scoring_kvarn_storage_key_bfloat16"], 1)
        XCTAssertEqual(
            task.engagement.counts["scoring_kvarn_storage_value_bfloat16"], 1)
        XCTAssertEqual(
            task.engagement.counts["scoring_kvarn_ingress_normalized"], 1)
        XCTAssertGreaterThan(
            task.engagement.counts[
                "scoring_kvarn_normalization_workspace_bytes"] ?? 0,
            0)
    }

    func testKVarNGenerateCarriesAuthenticatedBFloat16IngressReceipt()
        async throws
    {
        let actor = HarnessEngineActor(
            model: TinyScoringRouterModel(kvDType: .float32))
        let admission = try makeKVarNAdmission(modelNativeDType: "bfloat16")
        let driver = SwiftEngineDriver(
            engine: actor,
            eos: -1,
            compressedKVAttentionAdmission: admission)
        let prompt = (0..<256).map { ($0 % 31) + 1 }

        let result = try await driver.generate(
            prompt: prompt,
            config: RunConfig(
                maxTokens: 2,
                kvQuant: "kvarn-k4v2-g128",
                compressedKVAttention: .splitKVarNQuantizedMM,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    admission.checkpointContentSHA256))

        XCTAssertEqual(result.tokens.count, 2)
        XCTAssertEqual(result.engagement.counts["decode"], 2)
        XCTAssertEqual(result.engagement.counts["kvarn_attention_split"], 1)
        XCTAssertEqual(
            result.engagement.counts["kvarn_attention_materialized"], 0)
        XCTAssertEqual(
            result.engagement.counts["kvarn_uncompiled_correctness"], 0)
        XCTAssertEqual(result.engagement.counts["kvarn_compiled"], 1)
        XCTAssertEqual(result.engagement.counts["kvarn_source_key_float32"], 1)
        XCTAssertEqual(
            result.engagement.counts["kvarn_source_value_float32"], 1)
        assertKVarNDTypeOneHot(
            result.engagement.counts,
            prefix: "kvarn",
            role: "source_key",
            selected: "float32")
        assertKVarNDTypeOneHot(
            result.engagement.counts,
            prefix: "kvarn",
            role: "source_value",
            selected: "float32")
        assertKVarNDTypeOneHot(
            result.engagement.counts,
            prefix: "kvarn",
            role: "storage_key",
            selected: "bfloat16")
        assertKVarNDTypeOneHot(
            result.engagement.counts,
            prefix: "kvarn",
            role: "storage_value",
            selected: "bfloat16")
        XCTAssertEqual(
            result.engagement.counts["kvarn_ingress_normalized"], 1)
        XCTAssertGreaterThan(
            result.engagement.counts[
                "kvarn_normalization_workspace_bytes"] ?? 0,
            0)
    }

    func testSplitTaskAndFreeRunningScoringUseBoundedPrefill() async throws {
        let actor = HarnessEngineActor(model: TinyScoringRouterModel())
        let prompt = (0..<513).map { ($0 % 31) + 1 }
        let expectedToken = try XCTUnwrap(prompt.last)

        let task = await actor.taskChoiceLogits(
            prompt: prompt,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)
        let taskForwardWidths = await actor.scoringPrefillTokenCounts()
        XCTAssertEqual(taskForwardWidths, [512, 1])
        XCTAssertEqual(argmax(task.logits), expectedToken)

        let rows = await actor.logprobs(
            prompt: prompt,
            maxTokens: 1,
            eos: -1,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)
        let freeRunningForwardWidths =
            await actor.scoringPrefillTokenCounts()
        XCTAssertEqual(freeRunningForwardWidths, [512, 1])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(argmax(try XCTUnwrap(rows.first)), expectedToken)
    }

    func testMaterializedTaskScoringPreservesSingleForward() async throws {
        let actor = HarnessEngineActor(model: TinyScoringRouterModel())
        let prompt = (0..<513).map { ($0 % 31) + 1 }

        let task = await actor.taskChoiceLogits(
            prompt: prompt,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .materialize)

        let forwardWidths = await actor.scoringPrefillTokenCounts()
        XCTAssertEqual(forwardWidths, [513])
        XCTAssertEqual(argmax(task.logits), prompt.last)
    }

    private func argmax(_ values: [Float]) -> Int? {
        values.enumerated().max { $0.element < $1.element }?.offset
    }

    private func assertKVarNDTypeOneHot(
        _ counts: [String: Int],
        prefix: String,
        role: String,
        selected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for dtype in ["float16", "bfloat16", "float32"] {
            XCTAssertEqual(
                counts["\(prefix)_\(role)_\(dtype)"],
                dtype == selected ? 1 : 0,
                "\(prefix)_\(role)_\(dtype)",
                file: file,
                line: line)
        }
    }

    private func makeKVarNAdmission(
        modelNativeDType: String
    ) throws -> CompressedKVAttentionRuntimeAdmission {
        let config = Data(
            """
            {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":5120,"num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false,"torch_dtype":"\(modelNativeDType)"}
            """.utf8)
        let snapshot = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: config,
            checkpointManifestHash: "0123456789abcdef",
            checkpointContentSHA256: String(repeating: "d", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64))
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: snapshot)
    }
}
