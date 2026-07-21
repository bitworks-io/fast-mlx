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
    private let attentionDrivenLogits: Bool
    private(set) var forwardTokenCounts: [Int] = []

    init(attentionDrivenLogits: Bool = false) {
        self.attentionDrivenLogits = attentionDrivenLogits
    }

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
        let keys: MLXArray
        let values: MLXArray
        let queries: MLXArray
        if attentionDrivenLogits {
            let dimensions = MLXArray((0 ..< 128).map {
                Float16($0 % vocabularySize) / Float16(vocabularySize)
            }).reshaped([1, 1, 1, 128])
            let normalizedToken = scalar / Float16(vocabularySize)
            let delta = dimensions - normalizedToken
            let features = MLXArray(Float16(1)) - delta * delta
            keys = features
            values = features
            queries = broadcast(
                features, to: [inputs.dim(0), 2, inputs.dim(1), 128])
        } else {
            keys = broadcast(
                scalar, to: [inputs.dim(0), 1, inputs.dim(1), 128])
            values = keys
            queries = broadcast(
                scalar, to: [inputs.dim(0), 2, inputs.dim(1), 128])
        }
        let mask = layerCache.makeMask(
            n: inputs.dim(1), windowSize: nil, returnArray: true)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: layerCache,
            scale: Float(1 / sqrt(128.0)),
            mask: mask)
        if attentionDrivenLogits {
            return attended.mean(axes: [1], keepDims: false)[
                0..., 0..., 0 ..< vocabularySize
            ].asType(.float32)
        }
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
    func testGreedyTokenGuardUsesAnInvalidSentinelForNonFiniteLogits() {
        let finite = MLXArray([Float32(0.1), 0.9, 0.2])
            .reshaped([1, 3])
        let nonFinite = MLXArray([Float32(0.1), .nan, 0.2])
            .reshaped([1, 3])

        XCTAssertEqual(
            CompiledMLXDecoder.greedyTokenOrInvalidSentinel(finite)
                .item(Int.self),
            1)
        XCTAssertEqual(
            CompiledMLXDecoder.greedyTokenOrInvalidSentinel(nonFinite)
                .item(Int.self),
            -1)
    }

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

    func testKVarNFactoryKeepsMaterializedDefaultAndAllowsExplicitCompiledDirectRoute()
        throws
    {
        let kind = KVCacheKind.kvarn(.k4v2G128I8)
        let defaultCache = try XCTUnwrap(
            kind.makeCache(capacity: 256) as? KVarNKVCache)
        let directCache = try XCTUnwrap(
            kind.makeCache(
                capacity: 256,
                kvarnAttentionMode: .splitQuantizedMM)
                as? KVarNKVCache)

        XCTAssertEqual(defaultCache.attentionMode, .materialize)
        XCTAssertEqual(directCache.attentionMode, .splitQuantizedMM)
        XCTAssertEqual(
            kind.executionMode(
                requestingCompilation: true,
                kvarnAttentionMode: .materialize),
            .uncompiledCorrectness)
        XCTAssertEqual(
            kind.executionMode(
                requestingCompilation: true,
                kvarnAttentionMode: .splitQuantizedMM),
            .compiled)
    }

    func testBareDirectKVarNCacheDoesNotClaimCompiledExecution() throws {
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let keys = MLXArray.ones([1, 1, 1, 128], dtype: .float16)
        let values = MLXArray.ones([1, 1, 1, 128], dtype: .float16)
        let queries = MLXArray.ones([1, 2, 1, 128], dtype: .float16)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: Float(1 / sqrt(128.0)),
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(output)
        let telemetry = KVarNKVCacheTelemetry.capture(caches: [cache])

        XCTAssertEqual(telemetry.attentionOperation, .splitQuantizedMM)
        XCTAssertEqual(telemetry.executionMode, .uncompiledCorrectness)
    }

    func testCompiledDecoderDirectKVarNCrossesTileBoundaryAndResetsWithoutMaterialization()
        throws
    {
        let directModel = TinySharedRouterAttentionModel(
            attentionDrivenLogits: true)
        let materializedModel = TinySharedRouterAttentionModel(
            attentionDrivenLogits: true)
        var decoder = CompiledMLXDecoder(
            model: directModel,
            reserve: 8,
            kvCache: .kvarn(.k4v2G128I8),
            kvarnAttentionMode: .splitQuantizedMM)
        var materializedControl = CompiledMLXDecoder(
            model: materializedModel,
            reserve: 8,
            kvCache: .kvarn(.k4v2G128I8))
        XCTAssertEqual(decoder.executionMode, .compiled)
        XCTAssertEqual(
            materializedControl.executionMode,
            .uncompiledCorrectness)

        let prompt = (0 ..< 254).map { ($0 % 31) + 1 }
        let first = decoder.prefill(prompt)
        let controlFirst = materializedControl.prefill(prompt)
        XCTAssertEqual(first, controlFirst)
        XCTAssertNotEqual(
            controlFirst, 0,
            "the oracle must produce an attention-derived token, not a flat-zero tie")
        let beforeBoundary = try XCTUnwrap(decoder.kvarnKVTelemetry())
        XCTAssertEqual(beforeBoundary.cachedTokens, 255)
        XCTAssertEqual(beforeBoundary.completedTileCount, 0)
        XCTAssertEqual(beforeBoundary.materializationWorkspaceBytes, 0)
        XCTAssertEqual(beforeBoundary.attentionOperation, .splitQuantizedMM)

        let second = decoder.step(last: first)
        let controlSecond = materializedControl.step(last: controlFirst)
        XCTAssertEqual(second, controlSecond)
        let atBoundary = try XCTUnwrap(decoder.kvarnKVTelemetry())
        XCTAssertEqual(atBoundary.cachedTokens, 256)
        XCTAssertEqual(atBoundary.completedTileCount, 1)
        XCTAssertEqual(atBoundary.compressedTokens, 128)
        XCTAssertEqual(atBoundary.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(atBoundary.attentionWorkspaceBytes, 0)

        let third = decoder.step(last: second)
        let controlThird = materializedControl.step(last: controlSecond)
        XCTAssertEqual(third, controlThird)
        let afterBoundary = try XCTUnwrap(decoder.kvarnKVTelemetry())
        XCTAssertEqual(afterBoundary.cachedTokens, 257)
        XCTAssertEqual(afterBoundary.completedTileCount, 1)
        XCTAssertEqual(afterBoundary.compressedTokens, 128)
        XCTAssertEqual(afterBoundary.materializationWorkspaceBytes, 0)
        XCTAssertEqual(afterBoundary.executionMode, .compiled)

        decoder.reset()
        materializedControl.reset()
        let resetPrompt = (0 ..< 126).map { ($0 % 31) + 1 }
        let reusedFirst = decoder.prefill(resetPrompt)
        let controlReusedFirst = materializedControl.prefill(resetPrompt)
        XCTAssertEqual(reusedFirst, controlReusedFirst)
        let reusedSecond = decoder.step(last: reusedFirst)
        let controlReusedSecond = materializedControl.step(
            last: controlReusedFirst)
        XCTAssertEqual(reusedSecond, controlReusedSecond)
        let reusedThird = decoder.step(last: reusedSecond)
        let controlReusedThird = materializedControl.step(
            last: controlReusedSecond)
        XCTAssertEqual(reusedThird, controlReusedThird)
        let reused = try XCTUnwrap(decoder.kvarnKVTelemetry())
        XCTAssertEqual(reused.cachedTokens, 129)
        XCTAssertEqual(reused.completedTileCount, 0)
        XCTAssertEqual(reused.compressedTokens, 0)
        XCTAssertEqual(reused.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(reused.attentionWorkspaceBytes, 0)
        XCTAssertEqual(reused.attentionOperation, .splitQuantizedMM)
        XCTAssertEqual(reused.executionMode, .compiled)
        let controlTelemetry = try XCTUnwrap(
            materializedControl.kvarnKVTelemetry())
        XCTAssertEqual(controlTelemetry.attentionOperation, .materializedKV)
        XCTAssertGreaterThan(controlTelemetry.materializationWorkspaceBytes, 0)
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

    func testDirectKVarNRouteChunksPrefillToBoundTheExplicitScoreTensor() throws {
        let model = TinySharedRouterAttentionModel()
        var decoder = CompiledMLXDecoder(
            model: model,
            reserve: 8,
            kvCache: .kvarn(.k4v2G128I8),
            kvarnAttentionMode: .splitQuantizedMM)
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
