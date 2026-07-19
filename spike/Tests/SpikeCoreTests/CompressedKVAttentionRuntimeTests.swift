import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

private final class TinySharedRouterAttentionModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let vocabularySize = 32
    private(set) var forwardTokenCounts: [Int] = []

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        forwardTokenCounts.append(inputs.dim(1))
        guard let layerCache = cache?.first, cache?.count == 1 else {
            preconditionFailure("tiny shared-router model requires one cache")
        }
        let scalar = inputs.asType(.float16).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let keys = broadcast(
            scalar, to: [inputs.dim(0), 1, inputs.dim(1), 128])
        let values = keys
        let queries = broadcast(
            scalar, to: [inputs.dim(0), 2, inputs.dim(1), 128])
        let mask = layerCache.makeMask(
            n: inputs.dim(1), windowSize: nil, returnArray: true)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
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

final class CompressedKVAttentionRuntimeTests: XCTestCase {
    func testUniformFactoryKeepsMaterializationByDefaultAndAllowsExplicitSplit() throws {
        let kind = KVCacheKind.affine(.k4v2G64)
        let defaultCache = try XCTUnwrap(
            kind.makeCache(capacity: 8) as? AffineKVCache)
        let splitCache = try XCTUnwrap(
            kind.makeCache(
                capacity: 8,
                affineAttentionMode: .splitQuantizedMM)
                as? AffineKVCache)

        XCTAssertEqual(defaultCache.attentionMode, .materialize)
        XCTAssertEqual(splitCache.attentionMode, .splitQuantizedMM)
    }

    func testCompiledDecoderExplicitSplitEngagesSharedRouterWithoutMaterialization() throws {
        var decoder = CompiledMLXDecoder(
            model: TinySharedRouterAttentionModel(),
            reserve: 8,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)

        _ = decoder.prefill([1, 2])
        _ = decoder.step(last: 2)
        let telemetry = try XCTUnwrap(decoder.affineKVTelemetry())

        XCTAssertEqual(telemetry.attentionOperation, .splitQuantizedMM)
        XCTAssertEqual(telemetry.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(telemetry.attentionWorkspaceBytes, 0)
        XCTAssertEqual(telemetry.workspaceBytes, telemetry.attentionWorkspaceBytes)
        XCTAssertGreaterThanOrEqual(telemetry.cachedTokens, 3)
    }

    func testCompiledDecoderDefaultRouteRemainsMaterialized() throws {
        var decoder = CompiledMLXDecoder(
            model: TinySharedRouterAttentionModel(),
            reserve: 8,
            kvCache: .affine(.k4v2G64))

        _ = decoder.prefill([1, 2])
        let telemetry = try XCTUnwrap(decoder.affineKVTelemetry())

        XCTAssertEqual(telemetry.attentionOperation, .materializedKV)
        XCTAssertGreaterThan(telemetry.materializationWorkspaceBytes, 0)
        XCTAssertEqual(telemetry.attentionWorkspaceBytes, 0)
        XCTAssertEqual(
            telemetry.workspaceBytes,
            telemetry.materializationWorkspaceBytes)
    }

    func testSplitRouteChunksPrefillToBoundTheExplicitScoreTensor() throws {
        let model = TinySharedRouterAttentionModel()
        var decoder = CompiledMLXDecoder(
            model: model,
            reserve: 8,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)
        let prompt = (0..<513).map { ($0 % 31) + 1 }

        _ = decoder.prefill(prompt)

        XCTAssertEqual(
            Array(model.forwardTokenCounts.prefix(2)),
            [512, 1])
    }

    func testResetStartsANewSplitAttentionWorkspaceHighWater() throws {
        var decoder = CompiledMLXDecoder(
            model: TinySharedRouterAttentionModel(),
            reserve: 8,
            kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)

        _ = decoder.prefill([1, 2, 3, 4])
        let longRun = try XCTUnwrap(decoder.affineKVTelemetry())

        decoder.reset()
        _ = decoder.prefill([1])
        let shortRun = try XCTUnwrap(decoder.affineKVTelemetry())

        XCTAssertGreaterThan(
            longRun.attentionWorkspaceBytes,
            shortRun.attentionWorkspaceBytes)
        XCTAssertEqual(shortRun.materializationWorkspaceBytes, 0)
        XCTAssertEqual(
            shortRun.workspaceBytes,
            shortRun.attentionWorkspaceBytes)
    }
}
