import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore
@testable import fastmlx_harness

private final class TinyScoringRouterModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let vocabularySize = 32

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
        let scalar = inputs.asType(.float16).reshaped([
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
}
