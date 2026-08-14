import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

private final class TinyKVarNTelemetryModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let vocabularySize = 512

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache = cache?.first else {
            preconditionFailure("tiny KVarN model requires one cache")
        }
        let scalar = inputs.asType(.float16).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let kv = broadcast(
            scalar, to: [inputs.dim(0), 1, inputs.dim(1), 128])
        let (keys, _) = cache.update(keys: kv, values: kv)
        let target = keys.sum(axes: [1, 2, 3]).asType(.int32)
            .reshaped([inputs.dim(0), 1, 1])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class KVarNKVCacheTests: XCTestCase {
    func testCacheAcceptsOnlyTwoByteFloatingDTypesForExactSinkAndTail() {
        XCTAssertTrue(
            KVarNKVCache.supportsExactSinkAndTail(
                keyDType: .float16, valueDType: .float16))
        XCTAssertTrue(
            KVarNKVCache.supportsExactSinkAndTail(
                keyDType: .bfloat16, valueDType: .float16))
        XCTAssertTrue(
            KVarNKVCache.supportsExactSinkAndTail(
                keyDType: .float16, valueDType: .bfloat16))
        XCTAssertTrue(
            KVarNKVCache.supportsExactSinkAndTail(
                keyDType: .bfloat16, valueDType: .bfloat16))
        XCTAssertFalse(
            KVarNKVCache.supportsExactSinkAndTail(
                keyDType: .float32, valueDType: .float32))
    }

    func testCacheRequiresStablePerStreamDTypesForExactSinkAndTail() {
        XCTAssertTrue(
            KVarNKVCache.matchesEstablishedSinkAndTailDTypes(
                keyDType: .bfloat16, valueDType: .float16,
                establishedKeyDType: nil, establishedValueDType: nil))
        XCTAssertTrue(
            KVarNKVCache.matchesEstablishedSinkAndTailDTypes(
                keyDType: .bfloat16, valueDType: .float16,
                establishedKeyDType: .bfloat16,
                establishedValueDType: .float16))
        XCTAssertFalse(
            KVarNKVCache.matchesEstablishedSinkAndTailDTypes(
                keyDType: .float16, valueDType: .float16,
                establishedKeyDType: .bfloat16,
                establishedValueDType: .float16))
        XCTAssertFalse(
            KVarNKVCache.matchesEstablishedSinkAndTailDTypes(
                keyDType: .bfloat16, valueDType: .bfloat16,
                establishedKeyDType: .bfloat16,
                establishedValueDType: .float16))
    }

    func testCachePreservesBFloat16SinkAndTailWithoutChangingStorageBytes() throws {
        let cache = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8)
        let elementCount = 129 * 128
        let keys = MLXArray(
            (0 ..< elementCount).map { Float(($0 % 31) - 15) / 8 }
        ).reshaped([1, 1, 129, 128]).asType(.bfloat16)
        let values = MLXArray(
            (0 ..< elementCount).map { Float(($0 % 29) - 14) / 4 }
        ).reshaped([1, 1, 129, 128]).asType(.bfloat16)

        let (materializedKeys, materializedValues) = cache.update(
            keys: keys, values: values)
        eval([materializedKeys, materializedValues])

        XCTAssertEqual(cache.sinkKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.sinkValues?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailValues?.dtype, .bfloat16)
        XCTAssertEqual(materializedKeys.dtype, .bfloat16)
        XCTAssertEqual(materializedValues.dtype, .bfloat16)
        XCTAssertTrue(
            (materializedKeys[0..., 0..., 0 ..< 129, 0...] .== keys)
                .all().item(Bool.self))
        XCTAssertTrue(
            (materializedValues[0..., 0..., 0 ..< 129, 0...] .== values)
                .all().item(Bool.self))

        let continuationKeys = MLXArray(
            (0 ..< 128 * 128).map { Float(($0 % 23) - 11) / 16 }
        ).reshaped([1, 1, 128, 128]).asType(.bfloat16)
        let continuationValues = MLXArray(
            (0 ..< 128 * 128).map { Float(($0 % 19) - 9) / 8 }
        ).reshaped([1, 1, 128, 128]).asType(.bfloat16)
        let (completedKeys, completedValues) = cache.update(
            keys: continuationKeys, values: continuationValues)
        eval([completedKeys, completedValues])

        XCTAssertEqual(cache.completedTileCount, 1)
        XCTAssertEqual(cache.tailKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailValues?.dtype, .bfloat16)
        XCTAssertEqual(completedKeys.dtype, .bfloat16)
        XCTAssertEqual(completedValues.dtype, .bfloat16)
        XCTAssertTrue(
            (completedKeys[0..., 0..., 256 ..< 257, 0...]
                .== continuationKeys[0..., 0..., 127 ..< 128, 0...])
                .all().item(Bool.self))
        XCTAssertTrue(
            (completedValues[0..., 0..., 256 ..< 257, 0...]
                .== continuationValues[0..., 0..., 127 ..< 128, 0...])
                .all().item(Bool.self))

        let snapshot = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(snapshot.fp16SinkBytes, 65_536)
        XCTAssertEqual(snapshot.fp16TailBytes, 65_536)
        XCTAssertEqual(snapshot.materializationWorkspaceBytes, 131_584)
    }

    func testMLXCodecMatchesPinnedOfficialFixtureByteForByte() throws {
        let keys: [Float] = [
            -1.5, -0.5, 0.5, 1.5,
            0.25, -1.25, 1.75, -0.75,
            2.0, 0.5, -1.0, -0.25,
            -0.5, 1.25, 0.25, -2.0,
        ]
        let values: [Float] = [
            1.25, -0.25, -1.5, 0.5,
            -1.75, 0.75, 0.25, 1.0,
            0.5, 1.5, -0.75, -1.25,
            2.25, -1.0, 0.75, -0.5,
        ]
        let configuration = try KVarNMLXConfiguration(
            headDimension: 4, groupSize: 4,
            keyBits: 4, valueBits: 2, iterations: 16)

        let record = try KVarNMLXCodec.quantize(
            keys: MLXArray(keys).reshaped([1, 1, 4, 4]),
            values: MLXArray(values).reshaped([1, 1, 4, 4]),
            configuration: configuration)

        XCTAssertEqual(
            view(record.keyPayload, dtype: .uint8).asArray(UInt8.self),
            [85, 15, 240, 121, 64, 175, 87, 15])
        XCTAssertEqual(
            record.keyAbsorbedScale.asArray(Float16.self).map(\.bitPattern),
            [11560, 12665, 13550, 12891])
        XCTAssertEqual(
            record.keyAbsorbedBias.asArray(Float16.self).map(\.bitPattern),
            [46552, 48174, 49198, 48600])
        XCTAssertEqual(
            record.keyTokenScale.asArray(Float16.self).map(\.bitPattern),
            [15273, 15683, 14829, 15738])
        XCTAssertEqual(
            view(record.valuePayload, dtype: .uint8).asArray(UInt8.self),
            [224, 67, 53, 79])
        XCTAssertEqual(
            record.valueChannelScale.asArray(Float16.self).map(\.bitPattern),
            [14829, 16476, 16139, 16134])
        XCTAssertEqual(
            record.valueAbsorbedScale.asArray(Float16.self).map(\.bitPattern),
            [13806, 13536, 14379, 13307])
        XCTAssertEqual(
            record.valueAbsorbedBias.asArray(Float16.self).map(\.bitPattern),
            [44887, 47607, 46806, 13451])

        let reconstruction = try KVarNMLXCodec.dequantize(record)
        XCTAssertEqual(reconstruction.keys.shape, [1, 1, 4, 4])
        XCTAssertEqual(reconstruction.values.shape, [1, 1, 4, 4])
        let expectedKeys: [Float] = [
            -1.5166375637054443, -0.4484281837940216,
            0.5521049499511719, 1.484961748123169,
            0.15302783250808716, -1.2314488887786865,
            1.8963897228240967, -0.7190544605255127,
            1.9953035116195679, 0.5046354532241821,
            -1.0045689344406128, -0.24612390995025635,
            -0.4673830270767212, 1.3244329690933228,
            0.17573869228363037, -2.0329031944274902,
        ]
        let expectedValues: [Float] = [
            1.259522795677185, -0.24132204055786133,
            -1.5944502353668213, 0.406349778175354,
            -1.7937079668045044, 0.6056689023971558,
            0.293300986289978, 1.1442979574203491,
            0.7616767883300781, 1.3075151443481445,
            -0.4878883361816406, -1.442418098449707,
            2.2252750396728516, -0.9608343839645386,
            0.7888935804367065, -0.5243276357650757,
        ]
        for (actual, expected) in zip(
            reconstruction.keys.asType(.float32).asArray(Float.self), expectedKeys)
        {
            // Packed values and fp16 metadata above are exact. The inverse Hadamard runs
            // through MLX's Metal matmul rather than the fixture's scalar CPU reduction.
            XCTAssertEqual(actual, expected, accuracy: 1.5e-3)
        }
        for (actual, expected) in zip(
            reconstruction.values.asType(.float32).asArray(Float.self), expectedValues)
        {
            XCTAssertEqual(actual, expected, accuracy: 1.5e-3)
        }
    }

    func testDenseD128MatchesPureOracleAtEightAndSixteenIterations() throws {
        let elementCount = 128 * 128
        let keys = (0 ..< elementCount).map { index in
            Float(Float16(
                sin(Double(index) * 0.017) + 0.35 * cos(Double(index) * 0.031)))
        }
        let values = (0 ..< elementCount).map { index in
            Float(Float16(
                cos(Double(index) * 0.013) - 0.2 * sin(Double(index) * 0.029)))
        }

        for iterations in [8, 16] {
            let referenceConfiguration = KVarNReferenceConfig(
                headDimension: 128, groupSize: 128,
                keyBits: 4, valueBits: 2, iterations: iterations)
            let reference = try KVarNReference.quantize(
                keysTokenMajor: keys, valuesTokenMajor: values,
                config: referenceConfiguration)
            let mlxConfiguration = try KVarNMLXConfiguration(
                headDimension: 128, groupSize: 128,
                keyBits: 4, valueBits: 2, iterations: iterations)
            let mlx = try KVarNMLXCodec.quantize(
                keys: MLXArray(keys).reshaped([1, 1, 128, 128]).asType(.float16),
                values: MLXArray(values).reshaped([1, 1, 128, 128]).asType(.float16),
                configuration: mlxConfiguration)

            let mlxKeyCodes = unpack(
                mlx.keyPayload.asArray(UInt8.self), bits: 4)
            let referenceKeyCodes = unpack(reference.keyPacked, bits: 4)
            XCTAssertLessThanOrEqual(
                mismatchCount(mlxKeyCodes, referenceKeyCodes), 8,
                "K code drift exceeds 8/16,384 at iteration count \(iterations)")
            XCTAssertLessThanOrEqual(
                maximumDistance(mlxKeyCodes, referenceKeyCodes), 1,
                "K code drift exceeds one quantization level")
            assertFP16NearExact(
                mlx.keyAbsorbedScale.asArray(Float16.self).map(\.bitPattern),
                reference.keyAbsorbedScale.map(\.bitPattern),
                label: "K absorbed scales", iterations: iterations)
            assertFP16NearExact(
                mlx.keyAbsorbedBias.asArray(Float16.self).map(\.bitPattern),
                reference.keyAbsorbedBias.map(\.bitPattern),
                label: "K absorbed biases", iterations: iterations)
            XCTAssertEqual(mismatchCount(
                mlx.keyTokenScale.asArray(Float16.self).map(\.bitPattern),
                reference.keyTokenScale.map(\.bitPattern)), 0,
                "K token scales differ at iteration count \(iterations)")
            XCTAssertEqual(mismatchCount(
                mlx.valuePayload.asArray(UInt8.self), reference.valuePacked), 0,
                "V payload differs at iteration count \(iterations)")
            XCTAssertEqual(mismatchCount(
                mlx.valueChannelScale.asArray(Float16.self).map(\.bitPattern),
                reference.valueChannelScale.map(\.bitPattern)), 0,
                "V channel scales differ at iteration count \(iterations)")
            XCTAssertEqual(mismatchCount(
                mlx.valueAbsorbedScale.asArray(Float16.self).map(\.bitPattern),
                reference.valueAbsorbedScale.map(\.bitPattern)), 0,
                "V absorbed scales differ at iteration count \(iterations)")
            XCTAssertEqual(mismatchCount(
                mlx.valueAbsorbedBias.asArray(Float16.self).map(\.bitPattern),
                reference.valueAbsorbedBias.map(\.bitPattern)), 0,
                "V absorbed biases differ at iteration count \(iterations)")

            XCTAssertEqual(mlx.keyPayload.nbytes, 8_192)
            XCTAssertEqual(mlx.valuePayload.nbytes, 4_096)
            XCTAssertEqual([
                mlx.keyAbsorbedScale, mlx.keyAbsorbedBias, mlx.keyTokenScale,
                mlx.valueChannelScale, mlx.valueAbsorbedScale, mlx.valueAbsorbedBias,
            ].reduce(0) { $0 + $1.nbytes }, 1_536)
            let reconstructed = try KVarNMLXCodec.dequantize(mlx)
            XCTAssertEqual(reconstructed.keys.shape, [1, 1, 128, 128])
            XCTAssertEqual(reconstructed.values.shape, [1, 1, 128, 128])
            XCTAssertEqual(reconstructed.keys.dtype, .float16)
            XCTAssertEqual(reconstructed.values.dtype, .float16)
            XCTAssertTrue(isFinite(reconstructed.keys).all().item(Bool.self))
            XCTAssertTrue(isFinite(reconstructed.values).all().item(Bool.self))
        }
    }

    func testDirectPackedPrimitivesMatchMaterializedAlgebraAndConsumeExactBytes()
        throws
    {
        let dimension = 128
        let kvHeads = 2
        let queryHeads = 4
        let queryLength = 3
        let elementCount = kvHeads * dimension * dimension
        let keys = MLXArray((0 ..< elementCount).map { index in
            Float16(
                sin(Double(index) * 0.017)
                    + 0.35 * cos(Double(index) * 0.031))
        }).reshaped([1, kvHeads, dimension, dimension])
        let values = MLXArray((0 ..< elementCount).map { index in
            Float16(
                cos(Double(index) * 0.013)
                    - 0.2 * sin(Double(index) * 0.029))
        }).reshaped([1, kvHeads, dimension, dimension])
        let configuration = try KVarNMLXConfiguration(
            headDimension: dimension,
            groupSize: dimension,
            keyBits: 4,
            valueBits: 2,
            iterations: 8)
        let record = try KVarNMLXCodec.detachedStorageCopy(of:
            KVarNMLXCodec.quantize(
                keys: keys,
                values: values,
                configuration: configuration))
        let reconstruction = try KVarNMLXCodec.dequantize(record)
        let queries = MLXArray((0 ..< queryHeads * queryLength * dimension).map { index in
            Float16(
                0.5 * sin(Double(index) * 0.041)
                    - 0.25 * cos(Double(index) * 0.023))
        }).reshaped([1, queryHeads, queryLength, dimension])
        let valueWeights = softmax(
            MLXArray((0 ..< queryHeads * queryLength * dimension).map { index in
                Float16(sin(Double(index) * 0.007))
            }).reshaped([1, queryHeads, queryLength, dimension]),
            axis: -1,
            precise: true)

        let directScores = try KVarNMLXCodec.directKeyScores(
            queries: queries,
            key: record.keyOperand)
        let repeats = queryHeads / kvHeads
        let controlScores = matmul(
            queries.reshaped([
                1, kvHeads, repeats, queryLength, dimension,
            ]),
            reconstruction.keys
                .expandedDimensions(axis: 2)
                .transposed(0, 1, 2, 4, 3)
        ).reshaped([1, queryHeads, queryLength, dimension])
        let directValues = try KVarNMLXCodec.directValueProduct(
            weights: valueWeights,
            value: record.valueOperand)
        let controlValues = matmul(
            valueWeights.reshaped([
                1, kvHeads, repeats, queryLength, dimension,
            ]),
            reconstruction.values.expandedDimensions(axis: 2)
        ).reshaped([1, queryHeads, queryLength, dimension])
        eval([directScores, controlScores, directValues, controlValues])

        XCTAssertEqual(directScores.shape, controlScores.shape)
        XCTAssertEqual(directValues.shape, controlValues.shape)
        XCTAssertEqual(directScores.dtype, controlScores.dtype)
        XCTAssertEqual(directValues.dtype, controlValues.dtype)
        XCTAssertEqual(
            argMax(directScores, axis: -1).asArray(Int32.self),
            argMax(controlScores, axis: -1).asArray(Int32.self))
        for (actual, expected) in zip(
            directScores.asType(.float32).asArray(Float.self),
            controlScores.asType(.float32).asArray(Float.self))
        {
            XCTAssertEqual(actual, expected, accuracy: 8e-2)
        }
        for (actual, expected) in zip(
            directValues.asType(.float32).asArray(Float.self),
            controlValues.asType(.float32).asArray(Float.self))
        {
            XCTAssertEqual(actual, expected, accuracy: 8e-3)
        }

        // Match the production attention composition. Softmax over unscaled dot products
        // exaggerates small quantized-MM rounding differences that the real decoder divides by
        // sqrt(headDimension) before normalization.
        let attentionScale = MLXArray(
            Float(1 / sqrt(Double(dimension)))
        ).asType(directScores.dtype)
        let directAttention = try KVarNMLXCodec.directValueProduct(
            weights: softmax(
                directScores * attentionScale, axis: -1, precise: true),
            value: record.valueOperand)
        let controlAttention = matmul(
            softmax(
                controlScores * attentionScale, axis: -1, precise: true
            ).reshaped([
                1, kvHeads, repeats, queryLength, dimension,
            ]),
            reconstruction.values.expandedDimensions(axis: 2)
        ).reshaped([1, queryHeads, queryLength, dimension])
        eval([directAttention, controlAttention])

        XCTAssertEqual(directAttention.shape, controlAttention.shape)
        XCTAssertEqual(directAttention.dtype, controlAttention.dtype)
        XCTAssertEqual(
            argMax(directAttention, axis: -1).asArray(Int32.self),
            argMax(controlAttention, axis: -1).asArray(Int32.self))
        for (actual, expected) in zip(
            directAttention.asType(.float32).asArray(Float.self),
            controlAttention.asType(.float32).asArray(Float.self))
        {
            XCTAssertLessThanOrEqual(
                abs(actual - expected),
                2e-3 + 2e-3 * abs(expected))
        }

        let key = record.keyOperand
        var changedKeyBytes = key.payload.asArray(UInt8.self)
        changedKeyBytes[0] ^= 0x0F
        let changedKeys: [(String, KVarNMLXPackedKeyOperand)] = [
            (
                "payload",
                KVarNMLXPackedKeyOperand(
                    configuration: key.configuration,
                    batchSize: key.batchSize,
                    headCount: key.headCount,
                    outputDType: key.outputDType,
                    payload: MLXArray(changedKeyBytes).reshaped(key.payload.shape),
                    absorbedScale: key.absorbedScale,
                    absorbedBias: key.absorbedBias,
                    tokenScale: key.tokenScale)
            ),
            (
                "absorbed scale",
                KVarNMLXPackedKeyOperand(
                    configuration: key.configuration,
                    batchSize: key.batchSize,
                    headCount: key.headCount,
                    outputDType: key.outputDType,
                    payload: key.payload,
                    absorbedScale: key.absorbedScale + Float16(0.125),
                    absorbedBias: key.absorbedBias,
                    tokenScale: key.tokenScale)
            ),
            (
                "absorbed bias",
                KVarNMLXPackedKeyOperand(
                    configuration: key.configuration,
                    batchSize: key.batchSize,
                    headCount: key.headCount,
                    outputDType: key.outputDType,
                    payload: key.payload,
                    absorbedScale: key.absorbedScale,
                    absorbedBias: key.absorbedBias + Float16(0.125),
                    tokenScale: key.tokenScale)
            ),
            (
                "token scale",
                KVarNMLXPackedKeyOperand(
                    configuration: key.configuration,
                    batchSize: key.batchSize,
                    headCount: key.headCount,
                    outputDType: key.outputDType,
                    payload: key.payload,
                    absorbedScale: key.absorbedScale,
                    absorbedBias: key.absorbedBias,
                    tokenScale: key.tokenScale * Float16(1.25))
            ),
        ]

        let value = record.valueOperand
        var changedValueBytes = value.payload.asArray(UInt8.self)
        changedValueBytes[0] ^= 0x03
        let changedValues: [(String, KVarNMLXPackedValueOperand)] = [
            (
                "payload",
                KVarNMLXPackedValueOperand(
                    configuration: value.configuration,
                    batchSize: value.batchSize,
                    headCount: value.headCount,
                    outputDType: value.outputDType,
                    payload: MLXArray(changedValueBytes).reshaped(value.payload.shape),
                    channelScale: value.channelScale,
                    absorbedScale: value.absorbedScale,
                    absorbedBias: value.absorbedBias)
            ),
            (
                "channel scale",
                KVarNMLXPackedValueOperand(
                    configuration: value.configuration,
                    batchSize: value.batchSize,
                    headCount: value.headCount,
                    outputDType: value.outputDType,
                    payload: value.payload,
                    channelScale: value.channelScale * Float16(1.25),
                    absorbedScale: value.absorbedScale,
                    absorbedBias: value.absorbedBias)
            ),
            (
                "absorbed scale",
                KVarNMLXPackedValueOperand(
                    configuration: value.configuration,
                    batchSize: value.batchSize,
                    headCount: value.headCount,
                    outputDType: value.outputDType,
                    payload: value.payload,
                    channelScale: value.channelScale,
                    absorbedScale: value.absorbedScale + Float16(0.125),
                    absorbedBias: value.absorbedBias)
            ),
            (
                "absorbed bias",
                KVarNMLXPackedValueOperand(
                    configuration: value.configuration,
                    batchSize: value.batchSize,
                    headCount: value.headCount,
                    outputDType: value.outputDType,
                    payload: value.payload,
                    channelScale: value.channelScale,
                    absorbedScale: value.absorbedScale,
                    absorbedBias: value.absorbedBias + Float16(0.125))
            ),
        ]

        for (label, changedKey) in changedKeys {
            let changedScores = try KVarNMLXCodec.directKeyScores(
                queries: queries, key: changedKey)
            eval(changedScores)
            XCTAssertGreaterThan(
                (changedScores - directScores).abs().max().item(Float.self),
                1e-6,
                "direct key scores must consume the exact \(label) bytes")
        }
        for (label, changedValue) in changedValues {
            let changedProduct = try KVarNMLXCodec.directValueProduct(
                weights: valueWeights, value: changedValue)
            eval(changedProduct)
            XCTAssertGreaterThan(
                (changedProduct - directValues).abs().max().item(Float.self),
                1e-6,
                "direct value products must consume the exact \(label) bytes")
        }
    }

    func testDirectPackedPrimitivesPreserveBFloat16AndFloat32OutputDTypes() throws {
        let dimension = 128
        let configuration = try KVarNMLXConfiguration(
            headDimension: dimension,
            groupSize: dimension,
            keyBits: 4,
            valueBits: 2,
            iterations: 8)

        for dtype: DType in [.bfloat16, .float32] {
            let source = MLXArray((0 ..< dimension * dimension).map { index in
                Float(sin(Double(index) * 0.019))
            }).reshaped([1, 1, dimension, dimension]).asType(dtype)
            let record = try KVarNMLXCodec.detachedStorageCopy(of:
                KVarNMLXCodec.quantize(
                    keys: source,
                    values: source * Float(0.75),
                    configuration: configuration))
            let reconstruction = try KVarNMLXCodec.dequantize(record)
            let queries = MLXArray((0 ..< 2 * dimension).map { index in
                Float(cos(Double(index) * 0.023))
            }).reshaped([1, 1, 2, dimension]).asType(dtype)
            let weights = softmax(queries, axis: -1, precise: true)

            let scores = try KVarNMLXCodec.directKeyScores(
                queries: queries, key: record.keyOperand)
            let products = try KVarNMLXCodec.directValueProduct(
                weights: weights, value: record.valueOperand)
            let controlScores = matmul(
                queries, reconstruction.keys.transposed(0, 1, 3, 2))
            let controlProducts = matmul(weights, reconstruction.values)
            eval([scores, products, controlScores, controlProducts])

            XCTAssertEqual(scores.dtype, dtype)
            XCTAssertEqual(products.dtype, dtype)
            XCTAssertTrue(
                scores.allClose(controlScores, rtol: 3e-2, atol: 3e-2)
                    .item(Bool.self))
            XCTAssertTrue(
                products.allClose(controlProducts, rtol: 3e-2, atol: 3e-2)
                    .item(Bool.self))
        }
    }

    func testDirectPackedPrimitivesKeepGroupAndHeadDimensionsDistinct() throws {
        let dimension = 256
        let groupSize = 128
        let configuration = try KVarNMLXConfiguration(
            headDimension: dimension,
            groupSize: groupSize,
            keyBits: 4,
            valueBits: 2,
            iterations: 8)
        let keys = MLXArray((0 ..< groupSize * dimension).map { index in
            Float16(
                0.5 * sin(Double(index) * 0.013)
                    + 0.2 * cos(Double(index) * 0.019))
        }).reshaped([1, 1, groupSize, dimension])
        let values = MLXArray((0 ..< groupSize * dimension).map { index in
            Float16(
                0.4 * cos(Double(index) * 0.017)
                    - 0.1 * sin(Double(index) * 0.023))
        }).reshaped([1, 1, groupSize, dimension])
        let record = try KVarNMLXCodec.detachedStorageCopy(of:
            KVarNMLXCodec.quantize(
                keys: keys,
                values: values,
                configuration: configuration))
        let reconstruction = try KVarNMLXCodec.dequantize(record)
        let queries = MLXArray((0 ..< 2 * dimension).map { index in
            Float16(cos(Double(index) * 0.029))
        }).reshaped([1, 1, 2, dimension])
        let weights = softmax(
            MLXArray((0 ..< 2 * groupSize).map { index in
                Float16(sin(Double(index) * 0.031))
            }).reshaped([1, 1, 2, groupSize]),
            axis: -1,
            precise: true)

        let scores = try KVarNMLXCodec.directKeyScores(
            queries: queries,
            key: record.keyOperand)
        let products = try KVarNMLXCodec.directValueProduct(
            weights: weights,
            value: record.valueOperand)
        let controlScores = matmul(
            queries,
            reconstruction.keys.transposed(0, 1, 3, 2))
        let controlProducts = matmul(weights, reconstruction.values)
        eval([scores, products, controlScores, controlProducts])

        XCTAssertEqual(scores.shape, [1, 1, 2, groupSize])
        XCTAssertEqual(products.shape, [1, 1, 2, dimension])
        XCTAssertEqual(scores.dtype, .float16)
        XCTAssertEqual(products.dtype, .float16)
        XCTAssertTrue(
            scores.allClose(controlScores, rtol: 3e-2, atol: 3e-2)
                .item(Bool.self))
        XCTAssertTrue(
            products.allClose(controlProducts, rtol: 3e-2, atol: 3e-2)
                .item(Bool.self))
    }

    func testSharedAttentionRouterDirectKVarNUsesPackedGQAWithoutMaterialization()
        throws
    {
        let dimension = 128
        let tokens = 256
        let cache = KVarNKVCache(
            capacity: tokens,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let oracleCache = KVarNKVCache(
            capacity: tokens,
            tier: .k4v2G128,
            iterations: 8)
        let kvHeads = 2
        let queryHeads = 4
        let keys = MLXArray((0 ..< kvHeads * tokens * dimension).map { index in
            Float16(
                0.5 * sin(Double(index) * 0.011)
                    + 0.2 * cos(Double(index) * 0.017))
        }).reshaped([1, kvHeads, tokens, dimension])
        let values = MLXArray((0 ..< kvHeads * tokens * dimension).map { index in
            Float16(
                0.4 * cos(Double(index) * 0.013)
                    - 0.3 * sin(Double(index) * 0.019))
        }).reshaped([1, kvHeads, tokens, dimension])
        let queries = MLXArray((0 ..< queryHeads * tokens * dimension).map { index in
            Float16(
                0.25 * sin(Double(index) * 0.023)
                    + 0.1 * cos(Double(index) * 0.029))
        }).reshaped([1, queryHeads, tokens, dimension])
        let mask = cache.makeMask(
            n: tokens, windowSize: nil, returnArray: true)
        let scale = Float(1 / sqrt(Double(dimension)))

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask)
        let oracle = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: oracleCache,
            scale: scale,
            mask: mask)
        eval(output, oracle)

        XCTAssertTrue(cache is any AttentionKVCacheProtocol)
        XCTAssertEqual(output.shape, oracle.shape)
        XCTAssertEqual(output.dtype, oracle.dtype)
        XCTAssertTrue(
            output.allClose(oracle, rtol: 2e-3, atol: 2e-3)
                .item(Bool.self))
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(
            storage.materializationWorkspaceBytes, 0,
            "the shared router must not reconstruct the packed KVarN tile")
        XCTAssertGreaterThan(
            storage.attentionWorkspaceBytes, 0,
            "the direct route must report its score/weight workspace")
        XCTAssertEqual(
            storage.workspaceBytes, storage.attentionWorkspaceBytes)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
        XCTAssertEqual(cache.completedTileCount, 1)
    }

    func testDirectKVarNRouterPreservesGrowRollbackResetAndReuseLifecycle()
        throws
    {
        let dimension = 128
        let kvHeads = 2
        let queryHeads = 4
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 257,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let oracleCache = KVarNKVCache(
            capacity: 257,
            tier: .k4v2G128,
            iterations: 8)

        func tensor(
            seed: Int, heads: Int, tokens: Int, phase: Double
        ) -> MLXArray {
            MLXArray((0 ..< heads * tokens * dimension).map { index in
                Float16(
                    0.4 * sin(Double(index + seed) * phase)
                        + 0.2 * cos(Double(index + seed) * (phase + 0.011)))
            }).reshaped([1, heads, tokens, dimension])
        }

        func runStep(seed: Int, tokens: Int) -> (MLXArray, MLXArray) {
            let queries = tensor(
                seed: seed, heads: queryHeads, tokens: tokens, phase: 0.017)
            let keys = tensor(
                seed: seed + 1, heads: kvHeads, tokens: tokens, phase: 0.019)
            let values = tensor(
                seed: seed + 2, heads: kvHeads, tokens: tokens, phase: 0.023)
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: cache.makeMask(
                    n: tokens, windowSize: nil, returnArray: true))
            let oracle = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: oracleCache,
                scale: scale,
                mask: oracleCache.makeMask(
                    n: tokens, windowSize: nil, returnArray: true))
            eval(output, oracle)
            return (output, oracle)
        }

        func assertParity(
            _ pair: (MLXArray, MLXArray),
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            XCTAssertEqual(pair.0.shape, pair.1.shape, file: file, line: line)
            XCTAssertEqual(pair.0.dtype, pair.1.dtype, file: file, line: line)
            XCTAssertTrue(
                pair.0.allClose(pair.1, rtol: 2e-3, atol: 2e-3)
                    .item(Bool.self),
                file: file,
                line: line)
        }

        assertParity(runStep(seed: 10, tokens: 129))
        XCTAssertEqual(cache.offset, 129)
        XCTAssertEqual(cache.completedTileCount, 0)

        cache.grow(by: 128)
        oracleCache.grow(by: 128)
        XCTAssertEqual(cache.capacity, 385)
        XCTAssertEqual(oracleCache.capacity, 385)
        cache.truncate(to: 128)
        oracleCache.truncate(to: 128)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 128)

        assertParity(runStep(seed: 20, tokens: 128))
        XCTAssertEqual(cache.offset, 256)
        XCTAssertEqual(cache.completedTileCount, 1)
        let beforeReset = cache.innerState()

        cache.resetInPlace()
        oracleCache.resetInPlace()
        XCTAssertTrue(zip(beforeReset, cache.innerState()).allSatisfy { $0 === $1 })
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        XCTAssertEqual(cache.completedTileCount, 0)
        let resetStorage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(resetStorage.materializationWorkspaceBytes, 0)
        XCTAssertEqual(resetStorage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(resetStorage.workspaceBytes, 0)
        XCTAssertEqual(resetStorage.attentionOperation, .splitQuantizedMM)

        assertParity(runStep(seed: 30, tokens: 256))
        XCTAssertEqual(cache.offset, 256)
        XCTAssertEqual(cache.completedTileCount, 1)
        let reusedStorage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(reusedStorage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(reusedStorage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(
            reusedStorage.workspaceBytes,
            reusedStorage.attentionWorkspaceBytes)
        XCTAssertEqual(reusedStorage.attentionOperation, .splitQuantizedMM)
    }

    func testDirectKVarNRouterConsumesMultiplePackedTilesAndTheLiveTail() throws {
        let dimension = 128
        let tokens = 385
        let kvHeads = 1
        let queryHeads = 2
        let cache = KVarNKVCache(
            capacity: tokens,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let oracleCache = KVarNKVCache(
            capacity: tokens,
            tier: .k4v2G128,
            iterations: 8)

        func tensor(heads: Int, phase: Double, offset: Double) -> MLXArray {
            MLXArray((0 ..< heads * tokens * dimension).map { index in
                Float16(
                    0.45 * sin((Double(index) + offset) * phase)
                        + 0.15 * cos((Double(index) + offset) * (phase + 0.007)))
            }).reshaped([1, heads, tokens, dimension])
        }

        let queries = tensor(heads: queryHeads, phase: 0.011, offset: 3)
        let keys = tensor(heads: kvHeads, phase: 0.017, offset: 5)
        let values = tensor(heads: kvHeads, phase: 0.023, offset: 7)
        let scale = Float(1 / sqrt(Double(dimension)))
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: tokens, windowSize: nil, returnArray: true))
        let oracle = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: oracleCache,
            scale: scale,
            mask: oracleCache.makeMask(
                n: tokens, windowSize: nil, returnArray: true))
        eval(output, oracle)

        XCTAssertEqual(cache.completedTileCount, 2)
        XCTAssertEqual(cache.offset, tokens)
        XCTAssertEqual(output.shape, oracle.shape)
        XCTAssertEqual(output.dtype, oracle.dtype)
        XCTAssertTrue(
            output.allClose(oracle, rtol: 2e-3, atol: 2e-3)
                .item(Bool.self))
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testDirectKVarNRouterMasksUnusedPackedSlotsWhenLiveTailScoresAreNegative()
        throws
    {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 512,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let oracleCache = KVarNKVCache(
            capacity: 512,
            tier: .k4v2G128,
            iterations: 8)

        func tensor(
            heads: Int, tokens: Int, transform: (Int) -> Float16
        ) -> MLXArray {
            MLXArray((0 ..< heads * tokens * dimension).map(transform))
                .reshaped([1, heads, tokens, dimension])
        }

        func update(
            queries: MLXArray, keys: MLXArray, values: MLXArray
        ) -> (MLXArray, MLXArray) {
            let directMask = cache.makeMask(
                n: queries.dim(2), windowSize: nil, returnArray: true)
            let oracleMask = oracleCache.makeMask(
                n: queries.dim(2), windowSize: nil, returnArray: true)
            let direct = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: directMask)
            let oracle = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: oracleCache,
                scale: scale,
                mask: oracleMask)
            eval(direct, oracle)
            return (direct, oracle)
        }

        let prefill = update(
            queries: tensor(heads: 2, tokens: 256) { _ in 1 },
            keys: tensor(heads: 1, tokens: 256) { index in
                Float16(-0.45 - 0.05 * sin(Double(index) * 0.017))
            },
            values: tensor(heads: 1, tokens: 256) { index in
                Float16(0.4 + 0.1 * cos(Double(index) * 0.013))
            })
        XCTAssertTrue(
            prefill.0.allClose(prefill.1, rtol: 2e-3, atol: 2e-3)
                .item(Bool.self))

        let liveTail = update(
            queries: tensor(heads: 2, tokens: 1) { _ in 1 },
            keys: tensor(heads: 1, tokens: 1) { index in
                Float16(-0.7 - 0.05 * cos(Double(index) * 0.019))
            },
            values: tensor(heads: 1, tokens: 1) { index in
                Float16(0.8 + 0.05 * sin(Double(index) * 0.023))
            })

        XCTAssertEqual(cache.offset, 257)
        XCTAssertEqual(cache.completedTileCount, 1)
        XCTAssertTrue(
            liveTail.0.allClose(liveTail.1, rtol: 2e-3, atol: 2e-3)
                .item(Bool.self),
            "zero-filled inactive packed slots must not compete with the live tail")
        XCTAssertGreaterThan(
            liveTail.1.abs().max().item(Float.self), 0.1,
            "the oracle must expose any softmax mass stolen by inactive zero slots")
        XCTAssertTrue(cache is any AttentionKVCacheProtocol)
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.workspaceBytes, storage.attentionWorkspaceBytes)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testCompiledDirectKVarNReplayAdvancesPackedStateWithoutMaterialization()
        throws
    {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 384,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let oracleCache = KVarNKVCache(
            capacity: 384,
            tier: .k4v2G128,
            iterations: 8)

        func inputs(seed: Int, tokens: Int) -> [MLXArray] {
            func tensor(heads: Int, phase: Double) -> MLXArray {
                MLXArray((0 ..< heads * tokens * dimension).map { index in
                    Float16(
                        0.4 * sin(Double(index + seed) * phase)
                            + 0.2 * cos(
                                Double(index + seed) * (phase + 0.013)))
                }).reshaped([1, heads, tokens, dimension])
            }
            return [
                tensor(heads: 2, phase: 0.017),
                tensor(heads: 1, phase: 0.019),
                tensor(heads: 1, phase: 0.023),
            ]
        }

        let prefill = inputs(seed: 100, tokens: 255)
        let prefillOutput = attentionWithCacheUpdate(
            queries: prefill[0],
            keys: prefill[1],
            values: prefill[2],
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 255, windowSize: nil, returnArray: true))
        let oraclePrefillMask = oracleCache.makeMask(
            n: 255, windowSize: nil, returnArray: true)
        let (oraclePrefillKeys, oraclePrefillValues) = oracleCache.update(
            keys: prefill[1], values: prefill[2])
        let oraclePrefill = MLXFast.scaledDotProductAttention(
            queries: prefill[0],
            keys: oraclePrefillKeys,
            values: oraclePrefillValues,
            scale: scale,
            mask: oraclePrefillMask)
        eval(prefillOutput, oraclePrefill)
        XCTAssertTrue(
            prefillOutput.allClose(
                oraclePrefill, rtol: 2e-3, atol: 2e-3
            ).item(Bool.self))
        XCTAssertEqual(cache.completedTileCount, 0)
        let stateBeforeBoundary = cache.innerState()

        let step = compile(inputs: [cache], outputs: [cache]) { arguments in
            [attentionWithCacheUpdate(
                queries: arguments[0],
                keys: arguments[1],
                values: arguments[2],
                cache: cache,
                scale: scale,
                mask: cache.makeMask(
                    n: 1, windowSize: nil, returnArray: true))]
        }

        var firstTailKey: MLXArray?
        var firstTailValue: MLXArray?
        for (replayIndex, seed) in [200, 300, 400].enumerated() {
            let next = inputs(seed: seed, tokens: 1)
            let output = step(next)[0]
            let oracleMask = oracleCache.makeMask(
                n: 1, windowSize: nil, returnArray: true)
            let (oracleKeys, oracleValues) = oracleCache.update(
                keys: next[1], values: next[2])
            let oracle = MLXFast.scaledDotProductAttention(
                queries: next[0],
                keys: oracleKeys,
                values: oracleValues,
                scale: scale,
                mask: oracleMask)
            eval(output, oracle)
            XCTAssertTrue(isFinite(output).all().item(Bool.self))
            XCTAssertTrue(
                output.allClose(oracle, rtol: 2e-3, atol: 2e-3)
                    .item(Bool.self))

            XCTAssertEqual(
                cache.offsetArr.item(Int32.self), Int32(256 + replayIndex))
            if replayIndex == 0 {
                XCTAssertEqual(cache.completedTileCount, 1)
                XCTAssertTrue(
                    zip(stateBeforeBoundary, cache.innerState()).allSatisfy {
                        $0 === $1
                    },
                    "tile finalization must preserve every compiled-state handle")
                let clearedTailKey = cache.tailKeys![0, 0, 0, 0...]
                let clearedTailValue = cache.tailValues![0, 0, 0, 0...]
                eval(clearedTailKey, clearedTailValue)
                XCTAssertTrue(
                    (clearedTailKey .== MLXArray(Float16(0)))
                        .all().item(Bool.self))
                XCTAssertTrue(
                    (clearedTailValue .== MLXArray(Float16(0)))
                        .all().item(Bool.self))
            } else if replayIndex == 1 {
                firstTailKey = cache.tailKeys![0, 0, 0, 0...] + 0
                firstTailValue = cache.tailValues![0, 0, 0, 0...] + 0
                eval(firstTailKey!, firstTailValue!)
            } else {
                let retainedKey = cache.tailKeys![0, 0, 0, 0...]
                let retainedValue = cache.tailValues![0, 0, 0, 0...]
                let secondKey = cache.tailKeys![0, 0, 1, 0...]
                let secondValue = cache.tailValues![0, 0, 1, 0...]
                eval(retainedKey, retainedValue, secondKey, secondValue)
                XCTAssertTrue(
                    retainedKey.allClose(firstTailKey!, rtol: 0, atol: 0)
                        .item(Bool.self),
                    "compiled replay must preserve the first key row")
                XCTAssertTrue(
                    retainedValue.allClose(firstTailValue!, rtol: 0, atol: 0)
                        .item(Bool.self),
                    "compiled replay must preserve the first value row")
                XCTAssertGreaterThan(
                    (retainedKey - secondKey).abs().max().item(Float.self),
                    1e-3,
                    "compiled replay must write keys to distinct tail rows")
                XCTAssertGreaterThan(
                    (retainedValue - secondValue).abs().max().item(Float.self),
                    1e-3,
                    "compiled replay must write values to distinct tail rows")
            }
        }

        XCTAssertTrue(cache is any AttentionKVCacheProtocol)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 258)
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.workspaceBytes, storage.attentionWorkspaceBytes)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testCompiledDirectKVarNReplaySupportsLoadedQwenEightKCapacity() throws {
        let capacity = 8_448
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: capacity,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)

        func inputs() -> [MLXArray] {
            let keys = MLXArray.zeros(
                [1, 8, 1, dimension], dtype: .float16)
            return [
                MLXArray.zeros(
                    [1, 64, 1, dimension], dtype: .float16),
                keys,
                keys,
            ]
        }

        let prefill = inputs()
        let prefillOutput = attentionWithCacheUpdate(
            queries: prefill[0],
            keys: prefill[1],
            values: prefill[2],
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(prefillOutput, cache.innerState())
        let stateBeforeReplay = cache.innerState()

        let step = compile(inputs: [cache], outputs: [cache]) { arguments in
            [attentionWithCacheUpdate(
                queries: arguments[0],
                keys: arguments[1],
                values: arguments[2],
                cache: cache,
                scale: scale,
                mask: cache.makeMask(
                    n: 1, windowSize: nil, returnArray: true))]
        }
        let output = step(inputs())[0]
        eval(output, cache.innerState())

        XCTAssertTrue(isFinite(output).all().item(Bool.self))
        XCTAssertTrue(
            (output .== MLXArray(Float16(0))).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 2)
        XCTAssertTrue(
            zip(stateBeforeReplay, cache.innerState()).allSatisfy { $0 === $1 },
            "loaded-capacity replay must preserve every compiled state handle")
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.capacityTokens, capacity)
        XCTAssertEqual(storage.packedTileSlots, 65)
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testLoadedQwenFloat32PrefillUsesAuthenticatedBFloat16KVarNStorage()
        throws
    {
        let capacity = 8_448
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: capacity,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM,
            storageDType: .bfloat16)
        let keys = deterministicFloat32KVTensor(
            heads: 8, tokens: 512, dimension: dimension, seed: 10)
        let values = deterministicFloat32KVTensor(
            heads: 8, tokens: 512, dimension: dimension, seed: 20)
        let queries = deterministicFloat32KVTensor(
            heads: 64, tokens: 512, dimension: dimension, seed: 30)
        let expectedKeys = keys.asType(.bfloat16)
        let expectedValues = values.asType(.bfloat16)

        let prefillOutput = try cache.checkedUpdateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: cache.makeMask(
                n: 512, windowSize: nil, returnArray: true))
        eval(
            prefillOutput, cache.sinkKeys!, cache.sinkValues!,
            expectedKeys, expectedValues)

        XCTAssertEqual(prefillOutput.dtype, .float32)
        XCTAssertTrue(isFinite(prefillOutput).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 512)
        XCTAssertEqual(cache.sinkKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.sinkValues?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailValues?.dtype, .bfloat16)
        assertStorageBytes(
            cache.sinkKeys!,
            equal: expectedKeys[0..., 0..., 0 ..< 128, 0...],
            "authenticated float32 K ingress must persist exact bf16 sink bytes")
        assertStorageBytes(
            cache.sinkValues!,
            equal: expectedValues[0..., 0..., 0 ..< 128, 0...],
            "authenticated float32 V ingress must persist exact bf16 sink bytes")

        let prefillStorage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(prefillStorage.sourceKeyDType, .float32)
        XCTAssertEqual(prefillStorage.sourceValueDType, .float32)
        XCTAssertEqual(prefillStorage.storageKeyDType, .bfloat16)
        XCTAssertEqual(prefillStorage.storageValueDType, .bfloat16)
        XCTAssertTrue(prefillStorage.ingressNormalizationApplied)
        XCTAssertEqual(
            prefillStorage.normalizationWorkspaceBytes,
            2 * 8 * 512 * dimension * MemoryLayout<UInt16>.size)
        XCTAssertEqual(prefillStorage.fp16SinkBytes, 524_288)
        XCTAssertEqual(prefillStorage.fp16TailBytes, 524_288)
        XCTAssertEqual(
            prefillStorage.workspaceBytes,
            prefillStorage.normalizationWorkspaceBytes
                + prefillStorage.attentionWorkspaceBytes)

        let stateBeforeReplay = cache.innerState()
        let step = compile(inputs: [cache], outputs: [cache]) { arguments in
            [attentionWithCacheUpdate(
                queries: arguments[0],
                keys: arguments[1],
                values: arguments[2],
                cache: cache,
                scale: scale,
                mask: cache.makeMask(
                    n: 1, windowSize: nil, returnArray: true))]
        }
        let replayKeys = deterministicFloat32KVTensor(
            heads: 8, tokens: 1, dimension: dimension, seed: 40)
        let replayValues = deterministicFloat32KVTensor(
            heads: 8, tokens: 1, dimension: dimension, seed: 50)
        let replayQueries = deterministicFloat32KVTensor(
            heads: 64, tokens: 1, dimension: dimension, seed: 60)
        let expectedReplayKeys = replayKeys.asType(.bfloat16)
        let expectedReplayValues = replayValues.asType(.bfloat16)
        let replayOutput = step([
            replayQueries, replayKeys, replayValues,
        ])[0]
        eval(
            replayOutput, cache.tailKeys!, cache.tailValues!,
            expectedReplayKeys, expectedReplayValues)

        XCTAssertEqual(replayOutput.dtype, .float32)
        XCTAssertTrue(isFinite(replayOutput).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 513)
        XCTAssertTrue(
            zip(stateBeforeReplay, cache.innerState()).allSatisfy { $0 === $1 },
            "model-native ingress normalization must preserve compiled cache handles")
        assertStorageBytes(
            cache.tailKeys![0..., 0..., 0 ..< 1, 0...],
            equal: expectedReplayKeys,
            "authenticated float32 K replay must persist exact bf16 live-tail bytes")
        assertStorageBytes(
            cache.tailValues![0..., 0..., 0 ..< 1, 0...],
            equal: expectedReplayValues,
            "authenticated float32 V replay must persist exact bf16 live-tail bytes")
        let replayStorage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(replayStorage.sourceKeyDType, .float32)
        XCTAssertEqual(replayStorage.sourceValueDType, .float32)
        XCTAssertEqual(replayStorage.storageKeyDType, .bfloat16)
        XCTAssertEqual(replayStorage.storageValueDType, .bfloat16)
        XCTAssertTrue(replayStorage.ingressNormalizationApplied)
    }

    func testAuthenticatedFloat32IngressStoresExactFloat16SinkAndLiveTailBytes()
        throws
    {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 257,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM,
            storageDType: .float16)
        let keys = deterministicFloat32KVTensor(
            heads: 1, tokens: 129, dimension: dimension, seed: 70)
        let values = deterministicFloat32KVTensor(
            heads: 1, tokens: 129, dimension: dimension, seed: 80)
        let queries = deterministicFloat32KVTensor(
            heads: 2, tokens: 129, dimension: dimension, seed: 90)
        let expectedKeys = keys.asType(.float16)
        let expectedValues = values.asType(.float16)

        let output = try cache.checkedUpdateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: cache.makeMask(
                n: 129, windowSize: nil, returnArray: true))
        eval(
            output, cache.sinkKeys!, cache.sinkValues!,
            cache.tailKeys!, cache.tailValues!, expectedKeys, expectedValues)

        XCTAssertEqual(output.dtype, .float32)
        XCTAssertTrue(isFinite(output).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 129)
        XCTAssertEqual(cache.completedTileCount, 0)
        XCTAssertEqual(cache.sinkKeys?.dtype, .float16)
        XCTAssertEqual(cache.sinkValues?.dtype, .float16)
        XCTAssertEqual(cache.tailKeys?.dtype, .float16)
        XCTAssertEqual(cache.tailValues?.dtype, .float16)
        assertStorageBytes(
            cache.sinkKeys!,
            equal: expectedKeys[0..., 0..., 0 ..< 128, 0...],
            "authenticated float32 K ingress must persist exact fp16 sink bytes")
        assertStorageBytes(
            cache.sinkValues!,
            equal: expectedValues[0..., 0..., 0 ..< 128, 0...],
            "authenticated float32 V ingress must persist exact fp16 sink bytes")
        assertStorageBytes(
            cache.tailKeys![0..., 0..., 0 ..< 1, 0...],
            equal: expectedKeys[0..., 0..., 128 ..< 129, 0...],
            "authenticated float32 K ingress must persist exact fp16 live-tail bytes")
        assertStorageBytes(
            cache.tailValues![0..., 0..., 0 ..< 1, 0...],
            equal: expectedValues[0..., 0..., 128 ..< 129, 0...],
            "authenticated float32 V ingress must persist exact fp16 live-tail bytes")

        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.sourceKeyDType, .float32)
        XCTAssertEqual(storage.sourceValueDType, .float32)
        XCTAssertEqual(storage.storageKeyDType, .float16)
        XCTAssertEqual(storage.storageValueDType, .float16)
        XCTAssertTrue(storage.ingressNormalizationApplied)
        XCTAssertEqual(
            storage.normalizationWorkspaceBytes,
            2 * 129 * dimension * MemoryLayout<UInt16>.size)
        XCTAssertEqual(storage.fp16SinkBytes, 65_536)
        XCTAssertEqual(storage.fp16TailBytes, 65_536)
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testMaterializedAuthenticatedBFloat16IngressPreservesStorageBytes()
        throws
    {
        let dimension = 128
        let cache = KVarNKVCache(
            capacity: 257,
            tier: .k4v2G128,
            iterations: 8,
            storageDType: .bfloat16)
        let keys = deterministicFloat32KVTensor(
            heads: 1, tokens: 129, dimension: dimension, seed: 100)
        let values = deterministicFloat32KVTensor(
            heads: 1, tokens: 129, dimension: dimension, seed: 110)
        let expectedKeys = keys.asType(.bfloat16)
        let expectedValues = values.asType(.bfloat16)

        let (materializedKeys, materializedValues) = cache.update(
            keys: keys, values: values)
        eval(
            materializedKeys, materializedValues,
            cache.sinkKeys!, cache.sinkValues!, cache.tailKeys!, cache.tailValues!,
            expectedKeys, expectedValues)

        XCTAssertEqual(materializedKeys.dtype, .bfloat16)
        XCTAssertEqual(materializedValues.dtype, .bfloat16)
        XCTAssertEqual(cache.sinkKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.sinkValues?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailKeys?.dtype, .bfloat16)
        XCTAssertEqual(cache.tailValues?.dtype, .bfloat16)
        assertStorageBytes(
            cache.sinkKeys!,
            equal: expectedKeys[0..., 0..., 0 ..< 128, 0...],
            "materialized authenticated float32 K ingress must persist exact bf16 sink bytes")
        assertStorageBytes(
            cache.sinkValues!,
            equal: expectedValues[0..., 0..., 0 ..< 128, 0...],
            "materialized authenticated float32 V ingress must persist exact bf16 sink bytes")
        assertStorageBytes(
            cache.tailKeys![0..., 0..., 0 ..< 1, 0...],
            equal: expectedKeys[0..., 0..., 128 ..< 129, 0...],
            "materialized authenticated float32 K ingress must persist exact bf16 live-tail bytes")
        assertStorageBytes(
            cache.tailValues![0..., 0..., 0 ..< 1, 0...],
            equal: expectedValues[0..., 0..., 128 ..< 129, 0...],
            "materialized authenticated float32 V ingress must persist exact bf16 live-tail bytes")
        assertStorageBytes(
            materializedKeys[0..., 0..., 0 ..< 129, 0...],
            equal: expectedKeys,
            "materialized authenticated float32 K ingress must expose bf16 materialized bytes")
        assertStorageBytes(
            materializedValues[0..., 0..., 0 ..< 129, 0...],
            equal: expectedValues,
            "materialized authenticated float32 V ingress must expose bf16 materialized bytes")

        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.sourceKeyDType, .float32)
        XCTAssertEqual(storage.sourceValueDType, .float32)
        XCTAssertEqual(storage.storageKeyDType, .bfloat16)
        XCTAssertEqual(storage.storageValueDType, .bfloat16)
        XCTAssertTrue(storage.ingressNormalizationApplied)
        XCTAssertEqual(
            storage.normalizationWorkspaceBytes,
            2 * 129 * dimension * MemoryLayout<UInt16>.size)
        XCTAssertEqual(storage.fp16SinkBytes, 65_536)
        XCTAssertEqual(storage.fp16TailBytes, 65_536)
        XCTAssertEqual(storage.materializationWorkspaceBytes, 131_584)
        XCTAssertEqual(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.attentionOperation, .materializedKV)
    }

    func testAuthenticatedKVarNIngressDTypeFailuresPrecedeCacheMutation()
        throws
    {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let query = deterministicFloat32KVTensor(
            heads: 2, tokens: 1, dimension: dimension, seed: 120)

        func assertInitialFailure(
            cache: KVarNKVCache,
            keys: MLXArray,
            values: MLXArray? = nil,
            expected: KVarNMLXError,
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            let initialState = cache.innerState()
            XCTAssertThrowsError(try cache.checkedUpdateAndAttend(
                queries: query,
                keys: keys,
                values: values ?? keys,
                scale: scale,
                mask: cache.makeMask(
                    n: 1, windowSize: nil, returnArray: true)
            ), file: file, line: line) {
                XCTAssertEqual($0 as? KVarNMLXError, expected)
            }
            XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
            XCTAssertNil(cache.storageSnapshot())
            XCTAssertTrue(zip(initialState, cache.innerState()).allSatisfy {
                $0 === $1
            })
        }

        assertInitialFailure(
            cache: KVarNKVCache(
                capacity: 256,
                tier: .k4v2G128,
                iterations: 8,
                attentionMode: .splitQuantizedMM,
                storageDType: .bfloat16),
            keys: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float16),
            expected: .inputDTypeMismatch)
        assertInitialFailure(
            cache: KVarNKVCache(
                capacity: 256,
                tier: .k4v2G128,
                iterations: 8,
                attentionMode: .splitQuantizedMM,
                storageDType: .bfloat16),
            keys: deterministicFloat32KVTensor(
                heads: 1, tokens: 1, dimension: dimension, seed: 130),
            values: deterministicFloat32KVTensor(
                heads: 1, tokens: 1, dimension: dimension, seed: 140)
                .asType(.bfloat16),
            expected: .inputDTypeMismatch)
        assertInitialFailure(
            cache: KVarNKVCache(
                capacity: 256,
                tier: .k4v2G128,
                iterations: 8,
                attentionMode: .splitQuantizedMM),
            keys: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float32),
            expected: .unsupportedInputDType)
        assertInitialFailure(
            cache: KVarNKVCache(
                capacity: 256,
                tier: .k4v2G128,
                iterations: 8,
                attentionMode: .splitQuantizedMM,
                storageDType: .bfloat16),
            keys: MLXArray.full(
                [1, 1, 1, dimension], values: MLXArray(Float.nan)),
            expected: .nonFiniteInput)
        assertInitialFailure(
            cache: KVarNKVCache(
                capacity: 256,
                tier: .k4v2G128,
                iterations: 8,
                attentionMode: .splitQuantizedMM,
                storageDType: .float16),
            keys: MLXArray.full(
                [1, 1, 1, dimension], values: MLXArray(Float(70_000))),
            expected: .nonFiniteInput)

        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM,
            storageDType: .bfloat16)
        let initialFloat32 = deterministicFloat32KVTensor(
            heads: 1, tokens: 1, dimension: dimension, seed: 150)
        let output = try cache.checkedUpdateAndAttend(
            queries: query,
            keys: initialFloat32,
            values: initialFloat32,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(output, cache.innerState())
        let offsetBefore = cache.offsetArr.item(Int32.self)
        let bytesBefore = cache.innerState().map {
            $0.asData(access: .copy).data
        }
        let storageBefore = try XCTUnwrap(cache.storageSnapshot())
        let changed = deterministicFloat32KVTensor(
            heads: 1, tokens: 1, dimension: dimension, seed: 160)
            .asType(.bfloat16)

        XCTAssertThrowsError(try cache.checkedUpdateAndAttend(
            queries: query,
            keys: changed,
            values: changed,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true)
        )) {
            XCTAssertEqual($0 as? KVarNMLXError, .inputDTypeChanged)
        }
        XCTAssertEqual(cache.offsetArr.item(Int32.self), offsetBefore)
        XCTAssertEqual(
            cache.innerState().map { $0.asData(access: .copy).data },
            bytesBefore)
        XCTAssertEqual(cache.storageSnapshot(), storageBefore)
    }

    func testAuthenticatedFloat32ToFloat16OverflowFailsBeforeGraphMutation() {
        let dimension = 128
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM,
            storageDType: .float16)
        let initialState = cache.innerState()
        let queries = MLXArray.zeros(
            [1, 2, 1, dimension], dtype: .float32)
        let overflowingKV = MLXArray.full(
            [1, 1, 1, dimension], values: MLXArray(Float(70_000)))

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: overflowingKV,
            values: overflowingKV,
            cache: cache,
            scale: Float(1 / sqrt(Double(dimension))),
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(output, cache.innerState())

        XCTAssertFalse(isFinite(output).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        XCTAssertNil(cache.storageSnapshot())
        XCTAssertTrue(zip(initialState, cache.innerState()).allSatisfy {
            $0 === $1
        })
    }

    func testDirectRouterRejectsInvalidRequestBeforeCacheMutation() throws {
        let dimension = 128
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let initialState = cache.innerState()
        let validKeys = MLXArray.zeros(
            [1, 2, 1, dimension], dtype: .float16)
        let validValues = MLXArray.zeros(
            [1, 2, 1, dimension], dtype: .float16)

        func assertRejectedWithoutMutation(
            queries: MLXArray,
            keys: MLXArray? = nil,
            values: MLXArray? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
            expectedError: KVarNMLXError,
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            XCTAssertThrowsError(try cache.checkedUpdateAndAttend(
                queries: queries,
                keys: keys ?? validKeys,
                values: values ?? validValues,
                scale: Float(1 / sqrt(Double(dimension))),
                mask: mask
            ), file: file, line: line) {
                XCTAssertEqual(
                    $0 as? KVarNMLXError,
                    expectedError,
                    file: file,
                    line: line)
            }
            XCTAssertEqual(cache.offset, 0, file: file, line: line)
            XCTAssertEqual(
                cache.offsetArr.item(Int32.self), 0, file: file, line: line)
            XCTAssertNil(cache.storageSnapshot(), file: file, line: line)
            XCTAssertTrue(
                zip(initialState, cache.innerState()).allSatisfy { $0 === $1 },
                file: file,
                line: line)
        }

        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 3, 1, dimension], dtype: .float16),
            expectedError: .invalidTileShape)
        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 4, 1, dimension], dtype: .float16),
            mask: .array(MLXArray.zeros(
                [1, 1, 256], dtype: .float16)),
            expectedError: .invalidTileShape)
        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 4, 1, dimension], dtype: .float16),
            mask: .array(MLXArray.zeros(
                [2, 256], dtype: .float16)),
            expectedError: .invalidTileShape)
        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 4, 1, dimension], dtype: .float16),
            mask: .array(MLXArray.zeros(
                [2, 1, 1, 256], dtype: .float16)),
            expectedError: .invalidTileShape)
        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 4, 1, dimension], dtype: .float16),
            mask: .array(MLXArray.zeros(
                [1, 3, 1, 256], dtype: .float16)),
            expectedError: .invalidTileShape)
        assertRejectedWithoutMutation(
            queries: MLXArray.full(
                [1, 4, 1, dimension],
                values: MLXArray(Float16.nan)),
            expectedError: .nonFiniteInput)
        assertRejectedWithoutMutation(
            queries: MLXArray.zeros(
                [1, 4, 1, dimension], dtype: .float16),
            keys: MLXArray.zeros(
                [2, 2, 1, dimension], dtype: .float16),
            values: MLXArray.zeros(
                [2, 2, 1, dimension], dtype: .float16),
            expectedError: .invalidTileShape)
    }

    func testCompiledDirectRouterRejectsNonFiniteReplayWithoutMutatingArrayState() {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let prefill = MLXArray.zeros(
            [1, 1, 128, dimension], dtype: .float16)
        let prefillQueries = broadcast(
            prefill, to: [1, 2, 128, dimension])
        let prefillOutput = attentionWithCacheUpdate(
            queries: prefillQueries,
            keys: prefill,
            values: prefill,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 128, windowSize: nil, returnArray: true))
        eval(prefillOutput, cache.innerState())

        let stateBefore = cache.innerState().map {
            view($0, dtype: .uint8).asArray(UInt8.self)
        }
        let offsetBefore = cache.offsetArr.item(Int32.self)
        let step = compile(inputs: [cache], outputs: [cache]) { arguments in
            [attentionWithCacheUpdate(
                queries: arguments[0],
                keys: arguments[1],
                values: arguments[2],
                cache: cache,
                scale: scale,
                mask: cache.makeMask(
                    n: 1, windowSize: nil, returnArray: true))]
        }
        let nonFiniteQueries = MLXArray.full(
            [1, 2, 1, dimension],
            values: MLXArray(Float16.nan))
        let nonFiniteKV = MLXArray.full(
            [1, 1, 1, dimension],
            values: MLXArray(Float16.nan))

        let output = step([
            nonFiniteQueries, nonFiniteKV, nonFiniteKV,
        ])[0]
        eval(output, cache.innerState())

        XCTAssertFalse(isFinite(output).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), offsetBefore)
        XCTAssertEqual(
            cache.offset, Int(offsetBefore),
            "compiled tracing and a rejected replay cannot advance the host mirror")
        XCTAssertEqual(
            cache.innerState().map {
                view($0, dtype: .uint8).asArray(UInt8.self)
            },
            stateBefore,
            "non-finite replay must leave every compiled cache-state byte unchanged")
    }

    func testDirectRouterRejectAfterResetKeepsWorkspaceReceiptClear() throws {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let prefill = MLXArray.zeros(
            [1, 1, 128, dimension], dtype: .float16)
        let prefillOutput = attentionWithCacheUpdate(
            queries: broadcast(prefill, to: [1, 2, 128, dimension]),
            keys: prefill,
            values: prefill,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 128, windowSize: nil, returnArray: true))
        eval(prefillOutput, cache.innerState())

        cache.resetInPlace()
        let nonFiniteQueries = MLXArray.full(
            [1, 2, 1, dimension],
            values: MLXArray(Float16.nan))
        let nonFiniteKV = MLXArray.full(
            [1, 1, 1, dimension],
            values: MLXArray(Float16.nan))
        let output = attentionWithCacheUpdate(
            queries: nonFiniteQueries,
            keys: nonFiniteKV,
            values: nonFiniteKV,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(output, cache.innerState())

        XCTAssertFalse(isFinite(output).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.materializationWorkspaceBytes, 0)
        XCTAssertEqual(storage.attentionWorkspaceBytes, 0)
        XCTAssertEqual(storage.workspaceBytes, 0)
        XCTAssertEqual(storage.attentionOperation, .splitQuantizedMM)
    }

    func testDirectSingleTokenOverflowFailsInGraphWithoutMutatingState() {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 256,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let prefill = MLXArray.zeros(
            [1, 1, 255, dimension], dtype: .float16)
        let prefillOutput = attentionWithCacheUpdate(
            queries: broadcast(prefill, to: [1, 2, 255, dimension]),
            keys: prefill,
            values: prefill,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 255, windowSize: nil, returnArray: true))
        eval(prefillOutput, cache.innerState())

        let token = MLXArray.zeros(
            [1, 1, 1, dimension], dtype: .float16)
        let query = broadcast(token, to: [1, 2, 1, dimension])
        let admitted = attentionWithCacheUpdate(
            queries: query,
            keys: token,
            values: token,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(admitted, cache.innerState())
        XCTAssertTrue(isFinite(admitted).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 256)

        let stateBeforeOverflow = cache.innerState().map {
            view($0, dtype: .uint8).asArray(UInt8.self)
        }
        let rejected = attentionWithCacheUpdate(
            queries: query,
            keys: token,
            values: token,
            cache: cache,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true))
        eval(rejected, cache.innerState())

        XCTAssertFalse(isFinite(rejected).all().item(Bool.self))
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 256)
        XCTAssertEqual(cache.offset, 256)
        XCTAssertEqual(
            cache.innerState().map {
                view($0, dtype: .uint8).asArray(UInt8.self)
            },
            stateBeforeOverflow,
            "a full direct cache must reject the next token without changing any state byte")
    }

    func testDirectSinkOnlyCapacityFailsPreflightWithoutAllocation() {
        let dimension = 128
        let scale = Float(1 / sqrt(Double(dimension)))
        let cache = KVarNKVCache(
            capacity: 128,
            tier: .k4v2G128,
            iterations: 8,
            attentionMode: .splitQuantizedMM)
        let token = MLXArray.zeros(
            [1, 1, 1, dimension], dtype: .float16)
        XCTAssertThrowsError(try cache.checkedUpdateAndAttend(
            queries: broadcast(token, to: [1, 2, 1, dimension]),
            keys: token,
            values: token,
            scale: scale,
            mask: cache.makeMask(
                n: 1, windowSize: nil, returnArray: true)
        )) { error in
            XCTAssertEqual(error as? KVarNMLXError, .invalidTileShape)
        }
        XCTAssertEqual(cache.offset, 0)
        XCTAssertNil(cache.storageSnapshot())
        XCTAssertEqual(cache.innerState().count, 1)
    }

    func testDirectPackedPrimitivesRejectInvalidGeometryRecordsAndNonFiniteInput()
        throws
    {
        let dimension = 128
        let configuration = try KVarNMLXConfiguration(
            headDimension: dimension,
            groupSize: dimension,
            keyBits: 4,
            valueBits: 2,
            iterations: 8)
        let source = MLXArray((0 ..< dimension * dimension).map { index in
            Float16(sin(Double(index) * 0.019))
        }).reshaped([1, 1, dimension, dimension])
        let record = try KVarNMLXCodec.detachedStorageCopy(of:
            KVarNMLXCodec.quantize(
                keys: source,
                values: source * Float16(-0.5),
                configuration: configuration))

        let twoHeadSource = concatenated([
            source,
            source * Float16(0.75),
        ], axis: 1)
        let twoHeadRecord = try KVarNMLXCodec.detachedStorageCopy(of:
            KVarNMLXCodec.quantize(
                keys: twoHeadSource,
                values: twoHeadSource * Float16(-0.5),
                configuration: configuration))
        let wrongQueryGeometry = MLXArray.zeros(
            [1, 3, 1, dimension], dtype: .float16)
        XCTAssertThrowsError(try KVarNMLXCodec.directKeyScores(
            queries: wrongQueryGeometry,
            key: twoHeadRecord.keyOperand)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidTileShape)
        }

        let nonFiniteQueries = MLXArray.full(
            [1, 1, 1, dimension],
            values: MLXArray(Float16.nan))
        XCTAssertThrowsError(try KVarNMLXCodec.directKeyScores(
            queries: nonFiniteQueries,
            key: record.keyOperand)) {
            XCTAssertEqual($0 as? KVarNMLXError, .nonFiniteInput)
        }

        let key = record.keyOperand
        let wrongPayload = KVarNMLXPackedKeyOperand(
            configuration: key.configuration,
            batchSize: key.batchSize,
            headCount: key.headCount,
            outputDType: key.outputDType,
            payload: key.payload.asType(.uint32),
            absorbedScale: key.absorbedScale,
            absorbedBias: key.absorbedBias,
            tokenScale: key.tokenScale)
        XCTAssertThrowsError(try KVarNMLXCodec.directKeyScores(
            queries: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float16),
            key: wrongPayload)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }

        let nonFiniteMetadata = KVarNMLXPackedKeyOperand(
            configuration: key.configuration,
            batchSize: key.batchSize,
            headCount: key.headCount,
            outputDType: key.outputDType,
            payload: key.payload,
            absorbedScale: MLXArray.full(
                key.absorbedScale.shape,
                values: MLXArray(Float16.nan)),
            absorbedBias: key.absorbedBias,
            tokenScale: key.tokenScale)
        XCTAssertThrowsError(try KVarNMLXCodec.directKeyScores(
            queries: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float16),
            key: nonFiniteMetadata)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }

        XCTAssertThrowsError(try KVarNMLXCodec.directValueProduct(
            weights: MLXArray.zeros(
                [1, 1, 1, dimension - 1], dtype: .float16),
            value: record.valueOperand)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidTileShape)
        }

        XCTAssertThrowsError(try KVarNMLXCodec.directValueProduct(
            weights: MLXArray.full(
                [1, 1, 1, dimension],
                values: MLXArray(Float16.infinity)),
            value: record.valueOperand)) {
            XCTAssertEqual($0 as? KVarNMLXError, .nonFiniteInput)
        }

        let value = record.valueOperand
        let wrongValuePayload = KVarNMLXPackedValueOperand(
            configuration: value.configuration,
            batchSize: value.batchSize,
            headCount: value.headCount,
            outputDType: value.outputDType,
            payload: value.payload.asType(.uint32),
            channelScale: value.channelScale,
            absorbedScale: value.absorbedScale,
            absorbedBias: value.absorbedBias)
        XCTAssertThrowsError(try KVarNMLXCodec.directValueProduct(
            weights: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float16),
            value: wrongValuePayload)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }

        let nonFiniteValueMetadata = KVarNMLXPackedValueOperand(
            configuration: value.configuration,
            batchSize: value.batchSize,
            headCount: value.headCount,
            outputDType: value.outputDType,
            payload: value.payload,
            channelScale: MLXArray.full(
                value.channelScale.shape,
                values: MLXArray(Float16.nan)),
            absorbedScale: value.absorbedScale,
            absorbedBias: value.absorbedBias)
        XCTAssertThrowsError(try KVarNMLXCodec.directValueProduct(
            weights: MLXArray.zeros(
                [1, 1, 1, dimension], dtype: .float16),
            value: nonFiniteValueMetadata)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }
    }

    func testDetachedStorageCopyPreservesEveryRecordByteAndReconstruction() throws {
        let configuration = try KVarNMLXConfiguration(
            headDimension: 4, groupSize: 4,
            keyBits: 4, valueBits: 2, iterations: 8)
        let source = MLXArray((0 ..< 32).map { Float16(sin(Double($0) * 0.17)) })
            .reshaped([1, 2, 4, 4])
        let record = try KVarNMLXCodec.quantize(
            keys: source, values: source * Float16(-0.5),
            configuration: configuration)

        let detached = try KVarNMLXCodec.detachedStorageCopy(of: record)

        let originalArrays = recordArrays(record)
        let detachedArrays = recordArrays(detached)
        XCTAssertEqual(originalArrays.count, detachedArrays.count)
        for (original, copy) in zip(originalArrays, detachedArrays) {
            XCTAssertEqual(original.shape, copy.shape)
            XCTAssertEqual(original.dtype, copy.dtype)
            if original.dtype == .uint8 {
                XCTAssertEqual(original.asArray(UInt8.self), copy.asArray(UInt8.self))
            } else {
                XCTAssertEqual(
                    original.asArray(Float16.self).map(\.bitPattern),
                    copy.asArray(Float16.self).map(\.bitPattern))
            }
        }

        let originalReconstruction = try KVarNMLXCodec.dequantize(record)
        let detachedReconstruction = try KVarNMLXCodec.dequantize(detached)
        XCTAssertTrue((originalReconstruction.keys .== detachedReconstruction.keys)
            .all().item(Bool.self))
        XCTAssertTrue((originalReconstruction.values .== detachedReconstruction.values)
            .all().item(Bool.self))
    }

    func testCodecKeepsDistinctKVHeadsIsolated() throws {
        let dimension = 4
        let elementCount = dimension * dimension
        var keys: [Float16] = []
        var values: [Float16] = []
        for index in 0 ..< (2 * elementCount) {
            let head = Float(index / elementCount)
            keys.append(Float16(Float(sin(Double(index) * 0.31)) + head))
            values.append(Float16(Float(cos(Double(index) * 0.27)) - 0.5 * head))
        }
        let configuration = try KVarNMLXConfiguration(
            headDimension: dimension, groupSize: dimension,
            keyBits: 4, valueBits: 2, iterations: 8)
        let combined = try KVarNMLXCodec.quantize(
            keys: MLXArray(keys).reshaped([1, 2, dimension, dimension]),
            values: MLXArray(values).reshaped([1, 2, dimension, dimension]),
            configuration: configuration)

        for head in 0 ..< 2 {
            let range = (head * elementCount) ..< ((head + 1) * elementCount)
            let single = try KVarNMLXCodec.quantize(
                keys: MLXArray(Array(keys[range])).reshaped([1, 1, dimension, dimension]),
                values: MLXArray(Array(values[range])).reshaped([1, 1, dimension, dimension]),
                configuration: configuration)
            assertRecordRow(combined, row: head, equals: single)
        }
    }

    func testConfigurationRejectsOverflowingHadamardGeometry() {
        XCTAssertThrowsError(try KVarNMLXConfiguration(
            headDimension: 1 << 32, groupSize: 4,
            keyBits: 4, valueBits: 2, iterations: 8)
        ) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidConfiguration)
        }
    }

    func testDequantizeRejectsWrongPayloadAndMetadataDTypes() throws {
        let configuration = try KVarNMLXConfiguration(
            headDimension: 4, groupSize: 4,
            keyBits: 4, valueBits: 2, iterations: 8)
        let source = MLXArray((0 ..< 16).map(Float.init)).reshaped([1, 1, 4, 4])
        let record = try KVarNMLXCodec.quantize(
            keys: source, values: source, configuration: configuration)
        let wrongPayload = KVarNMLXRecord(
            configuration: record.configuration,
            batchSize: record.batchSize, headCount: record.headCount,
            keyDType: record.keyDType, valueDType: record.valueDType,
            keyPayload: record.keyPayload.asType(.uint32),
            keyAbsorbedScale: record.keyAbsorbedScale,
            keyAbsorbedBias: record.keyAbsorbedBias,
            keyTokenScale: record.keyTokenScale,
            valuePayload: record.valuePayload,
            valueChannelScale: record.valueChannelScale,
            valueAbsorbedScale: record.valueAbsorbedScale,
            valueAbsorbedBias: record.valueAbsorbedBias)
        XCTAssertThrowsError(try KVarNMLXCodec.dequantize(wrongPayload)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }
        let wrongMetadata = KVarNMLXRecord(
            configuration: record.configuration,
            batchSize: record.batchSize, headCount: record.headCount,
            keyDType: record.keyDType, valueDType: record.valueDType,
            keyPayload: record.keyPayload,
            keyAbsorbedScale: record.keyAbsorbedScale.asType(.float32),
            keyAbsorbedBias: record.keyAbsorbedBias,
            keyTokenScale: record.keyTokenScale,
            valuePayload: record.valuePayload,
            valueChannelScale: record.valueChannelScale,
            valueAbsorbedScale: record.valueAbsorbedScale,
            valueAbsorbedBias: record.valueAbsorbedBias)
        XCTAssertThrowsError(try KVarNMLXCodec.dequantize(wrongMetadata)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }

        let overflowingRows = KVarNMLXRecord(
            configuration: record.configuration,
            batchSize: Int.max, headCount: 2,
            keyDType: record.keyDType, valueDType: record.valueDType,
            keyPayload: record.keyPayload,
            keyAbsorbedScale: record.keyAbsorbedScale,
            keyAbsorbedBias: record.keyAbsorbedBias,
            keyTokenScale: record.keyTokenScale,
            valuePayload: record.valuePayload,
            valueChannelScale: record.valueChannelScale,
            valueAbsorbedScale: record.valueAbsorbedScale,
            valueAbsorbedBias: record.valueAbsorbedBias)
        XCTAssertThrowsError(try KVarNMLXCodec.dequantize(overflowingRows)) {
            XCTAssertEqual($0 as? KVarNMLXError, .invalidRecord)
        }
    }

    func testCacheKeepsSinkAndTailExactAndPacksOnlyACompletedTile() throws {
        let capacity = 257
        let dimension = 128
        let count = capacity * dimension
        let keys = (0 ..< count).map { index in
            Float16(sin(Double(index) * 0.011) + 0.25 * cos(Double(index) * 0.037))
        }
        let values = (0 ..< count).map { index in
            Float16(cos(Double(index) * 0.019) - 0.3 * sin(Double(index) * 0.023))
        }
        let keyInput = MLXArray(keys).reshaped([1, 1, capacity, dimension])
        let valueInput = MLXArray(values).reshaped([1, 1, capacity, dimension])
        let oneShot = KVarNKVCache(
            capacity: capacity, tier: .k4v2G128, iterations: 8)
        let chunked = KVarNKVCache(
            capacity: capacity, tier: .k4v2G128, iterations: 8)

        let (oneShotKeys, oneShotValues) = oneShot.update(
            keys: keyInput, values: valueInput)
        _ = chunked.update(
            keys: keyInput[0..., 0..., 0 ..< 128, 0...],
            values: valueInput[0..., 0..., 0 ..< 128, 0...])
        _ = chunked.update(
            keys: keyInput[0..., 0..., 128 ..< 255, 0...],
            values: valueInput[0..., 0..., 128 ..< 255, 0...])

        XCTAssertEqual(chunked.completedTileCount, 0)
        XCTAssertEqual(
            chunked.kPayload?.asType(.int32).max().item(Int32.self), 0,
            "an incomplete post-sink tail must not create a packed record")
        XCTAssertEqual(
            chunked.vPayload?.asType(.int32).max().item(Int32.self), 0,
            "an incomplete post-sink tail must not create a packed record")

        let (completedKeys, completedValues) = chunked.update(
            keys: keyInput[0..., 0..., 255 ..< 256, 0...],
            values: valueInput[0..., 0..., 255 ..< 256, 0...])
        XCTAssertEqual(chunked.completedTileCount, 1)
        XCTAssertEqual(
            completedKeys[0, 0, 256, 0...].abs().max().item(Float.self), 0,
            "unused materialized capacity must remain zero")
        XCTAssertEqual(
            completedValues[0, 0, 256, 0...].abs().max().item(Float.self), 0,
            "unused materialized capacity must remain zero")

        let tileConfiguration = try KVarNMLXConfiguration(
            headDimension: 128, groupSize: 128,
            keyBits: 4, valueBits: 2, iterations: 8)
        let tileRecord = try KVarNMLXCodec.quantize(
            keys: keyInput[0..., 0..., 128 ..< 256, 0...],
            values: valueInput[0..., 0..., 128 ..< 256, 0...],
            configuration: tileConfiguration)
        XCTAssertTrue((
            chunked.kPayload![0, 0, 0, 0...].reshaped(tileRecord.keyPayload.shape)
                .== tileRecord.keyPayload
        ).all().item(Bool.self))
        XCTAssertTrue((
            chunked.vPayload![0, 0, 0, 0...].reshaped(tileRecord.valuePayload.shape)
                .== tileRecord.valuePayload
        ).all().item(Bool.self))
        let tileReconstruction = try KVarNMLXCodec.dequantize(tileRecord)
        XCTAssertEqual(
            (completedKeys[0..., 0..., 128 ..< 256, 0...]
                - tileReconstruction.keys).abs().max().item(Float.self),
            0, accuracy: 1e-4)
        XCTAssertEqual(
            (completedValues[0..., 0..., 128 ..< 256, 0...]
                - tileReconstruction.values).abs().max().item(Float.self),
            0, accuracy: 1e-4)

        let (chunkedKeys, chunkedValues) = chunked.update(
            keys: keyInput[0..., 0..., 256 ..< 257, 0...],
            values: valueInput[0..., 0..., 256 ..< 257, 0...])
        XCTAssertEqual(chunked.offsetArr.item(Int32.self), 257)
        XCTAssertEqual(
            chunkedKeys[0..., 0..., 0 ..< 128, 0...].asArray(Float16.self),
            keyInput[0..., 0..., 0 ..< 128, 0...].asArray(Float16.self))
        XCTAssertEqual(
            chunkedValues[0..., 0..., 0 ..< 128, 0...].asArray(Float16.self),
            valueInput[0..., 0..., 0 ..< 128, 0...].asArray(Float16.self))
        XCTAssertEqual(
            chunkedKeys[0, 0, 256, 0...].asArray(Float16.self),
            keyInput[0, 0, 256, 0...].asArray(Float16.self))
        XCTAssertEqual(
            chunkedValues[0, 0, 256, 0...].asArray(Float16.self),
            valueInput[0, 0, 256, 0...].asArray(Float16.self))

        XCTAssertEqual(oneShot.innerState().count, chunked.innerState().count)
        for (lhs, rhs) in zip(oneShot.innerState(), chunked.innerState()) {
            XCTAssertEqual(lhs.shape, rhs.shape)
            XCTAssertEqual(lhs.dtype, rhs.dtype)
            XCTAssertTrue((lhs .== rhs).all().item(Bool.self))
        }
        XCTAssertTrue((oneShotKeys .== chunkedKeys).all().item(Bool.self))
        XCTAssertTrue((oneShotValues .== chunkedValues).all().item(Bool.self))
    }

    func testCacheStorageSnapshotReconcilesEveryNativeArrayByte() throws {
        let cache = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8)
        XCTAssertNil(cache.storageSnapshot())
        let keys = MLXArray.zeros([1, 1, 1, 128], dtype: .float16)
        let values = MLXArray.zeros([1, 1, 1, 128], dtype: .float16)
        _ = cache.update(keys: keys, values: values)

        let snapshot = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(snapshot.packedTileSlots, 2)
        XCTAssertEqual(snapshot.payloadBytes, 24_576)
        XCTAssertEqual(snapshot.metadataBytes, 3_072)
        XCTAssertEqual(snapshot.alignmentPaddingBytes, 0)
        XCTAssertEqual(snapshot.fp16SinkBytes, 65_536)
        XCTAssertEqual(snapshot.fp16TailBytes, 65_536)
        XCTAssertEqual(snapshot.formatPersistentBytes, 158_720)
        XCTAssertEqual(snapshot.materializationWorkspaceBytes, 131_584)
        XCTAssertEqual(snapshot.attentionWorkspaceBytes, 0)
        XCTAssertEqual(snapshot.workspaceBytes, 131_584)
        XCTAssertEqual(snapshot.attentionOperation, .materializedKV)
        XCTAssertEqual(snapshot.storageAndMaterializationBytes, 290_304)
        XCTAssertEqual(snapshot.storageAndWorkspaceBytes, 290_304)
        XCTAssertEqual(snapshot.controlBytes, 4)
        XCTAssertEqual(snapshot.totalPersistentBytes, 158_724)

        let allocation = try KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8
        ).allocation(
            geometry: KVStorageGeometry(
                layerCount: 1, kvHeadCount: 1, headDimension: 128),
            capacityTokens: 257, sequences: 1,
            workspaceBytes: snapshot.materializationWorkspaceBytes)
        XCTAssertEqual(snapshot.packedTileSlots, allocation.packedTileSlotsPerSequence)
        XCTAssertEqual(snapshot.payloadBytes, allocation.payloadBytes)
        XCTAssertEqual(snapshot.metadataBytes, allocation.metadataBytes)
        XCTAssertEqual(snapshot.alignmentPaddingBytes, allocation.alignmentPaddingBytes)
        XCTAssertEqual(snapshot.fp16SinkBytes, allocation.fp16SinkBytes)
        XCTAssertEqual(snapshot.fp16TailBytes, allocation.fp16TailBytes)
        XCTAssertEqual(snapshot.materializationWorkspaceBytes, allocation.workspaceBytes)
        XCTAssertEqual(snapshot.storageAndMaterializationBytes, allocation.totalBytes)
    }

    func testCacheGrowTailLocalTruncateAndResetPreserveStateContract() {
        let cache = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8)
        let initialKeys = MLXArray.zeros([1, 1, 129, 128], dtype: .float16)
        let initialValues = MLXArray.zeros([1, 1, 129, 128], dtype: .float16)
        _ = cache.update(keys: initialKeys, values: initialValues)
        XCTAssertEqual(cache.innerState().count, 13)

        let sinkObject = cache.sinkKeys!
        let tailObject = cache.tailKeys!
        let offsetObject = cache.offsetArr
        cache.grow(by: 128)
        XCTAssertEqual(cache.capacity, 385)
        XCTAssertEqual(cache.kPayload?.shape, [1, 1, 3, 8_192])
        XCTAssertEqual(cache.vPayload?.shape, [1, 1, 3, 4_096])
        XCTAssertTrue(cache.sinkKeys! === sinkObject)
        XCTAssertTrue(cache.tailKeys! === tailObject)
        XCTAssertTrue(cache.offsetArr === offsetObject)

        let objectsBeforeTruncate = cache.innerState()
        cache.truncate(to: 128)
        XCTAssertTrue(zip(objectsBeforeTruncate, cache.innerState()).allSatisfy { $0 === $1 })
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 128)
        let replacementKeys = MLXArray.full(
            [1, 1, 1, 128], values: MLXArray(Float16(9)))
        let replacementValues = MLXArray.full(
            [1, 1, 1, 128], values: MLXArray(Float16(-3)))
        let (materializedKeys, materializedValues) = cache.update(
            keys: replacementKeys, values: replacementValues)
        XCTAssertEqual(materializedKeys[0, 0, 128, 0].item(Float.self), 9)
        XCTAssertEqual(materializedValues[0, 0, 128, 0].item(Float.self), -3)

        let objectsBeforeReset = cache.innerState()
        cache.resetInPlace()
        XCTAssertTrue(zip(objectsBeforeReset, cache.innerState()).allSatisfy { $0 === $1 })
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        for array in cache.innerState() {
            XCTAssertEqual(array.asType(.float32).abs().max().item(Float.self), 0)
        }
    }

    func testCanonicalFactoryTelemetryAndExecutionModeFailClosed() throws {
        XCTAssertEqual(KVarNKVTier.k4v2G128.rawValue, "kvarn-k4v2-g128")
        XCTAssertEqual(
            KVarNKVRuntimeCell.k4v2G128I8.rawValue,
            "kvarn-k4v2-g128")
        XCTAssertEqual(
            KVarNKVRuntimeCell.k4v2G128I16.rawValue,
            "kvarn-k4v2-g128-i16")
        let kind = try XCTUnwrap(KVCacheKind(kvQuant: "kvarn-k4v2-g128"))
        XCTAssertEqual(kind, .kvarn(.k4v2G128I8))
        let factoryCache = try XCTUnwrap(
            kind.makeCache(capacity: 257) as? KVarNKVCache)
        XCTAssertEqual(factoryCache.tier, .k4v2G128)
        XCTAssertEqual(factoryCache.iterations, 8)
        let i16Kind = try XCTUnwrap(
            KVCacheKind(kvQuant: "kvarn-k4v2-g128-i16"))
        XCTAssertEqual(i16Kind, .kvarn(.k4v2G128I16))
        let i16Cache = try XCTUnwrap(
            i16Kind.makeCache(capacity: 257) as? KVarNKVCache)
        XCTAssertEqual(i16Cache.tier, .k4v2G128)
        XCTAssertEqual(i16Cache.iterations, 16)
        for spelling in [
            "k4v2-g128", "kvarn-k4v2-g128-i8", "kvarn-k4v2-g64",
            "kvarn-k4v4-g128", "kvarn-k4v2-g96",
        ] {
            XCTAssertNil(KVCacheKind(kvQuant: spelling), "unexpected alias: \(spelling)")
        }
        XCTAssertEqual(
            kind.executionMode(requestingCompilation: true),
            .uncompiledCorrectness)
        XCTAssertEqual(
            kind.executionMode(requestingCompilation: false),
            .uncompiledCorrectness)
        XCTAssertEqual(
            KVCacheKind.fp16.executionMode(requestingCompilation: true),
            .compiled)
        XCTAssertEqual(
            KVCacheKind.fp16.executionMode(requestingCompilation: false),
            .uncompiledCorrectness)
        XCTAssertTrue(KVCacheKind.fp16.supportsSpecDecode)
        XCTAssertFalse(kind.supportsSpecDecode)
        XCTAssertFalse(
            KVCacheKind.affine(.k4v2G128).supportsSpecDecode)
        XCTAssertFalse(
            KVCacheKind.turboQuant(.tqB3).supportsSpecDecode)

        let caches = (0 ..< 2).map { _ in
            KVarNKVCache(capacity: 257, tier: .k4v2G128, iterations: 8)
        }
        for cache in caches {
            _ = cache.update(
                keys: MLXArray.zeros([1, 1, 1, 128], dtype: .float16),
                values: MLXArray.zeros([1, 1, 1, 128], dtype: .float16))
        }
        let telemetry = KVarNKVCacheTelemetry.capture(caches: caches)
        XCTAssertEqual(telemetry.tier, .k4v2G128)
        XCTAssertEqual(telemetry.iterations, 8)
        XCTAssertEqual(telemetry.executionMode, .uncompiledCorrectness)
        XCTAssertEqual(telemetry.cachedTokens, 1)
        XCTAssertEqual(telemetry.completedTileCount, 0)
        XCTAssertEqual(telemetry.compressedTokens, 0)
        XCTAssertEqual(telemetry.layerCount, 2)
        XCTAssertEqual(telemetry.capacityTokens, 257)
        XCTAssertEqual(telemetry.packedTileSlots, 2)
        XCTAssertEqual(telemetry.sequences, 1)
        XCTAssertEqual(telemetry.kvHeadCount, 1)
        XCTAssertEqual(telemetry.headDimension, 128)
        XCTAssertEqual(telemetry.payloadBytes, 49_152)
        XCTAssertEqual(telemetry.metadataBytes, 6_144)
        XCTAssertEqual(telemetry.alignmentPaddingBytes, 0)
        XCTAssertEqual(telemetry.fp16SinkBytes, 131_072)
        XCTAssertEqual(telemetry.fp16TailBytes, 131_072)
        XCTAssertEqual(telemetry.controlBytes, 8)
        XCTAssertEqual(telemetry.materializationWorkspaceBytes, 131_584)
        XCTAssertEqual(telemetry.attentionWorkspaceBytes, 0)
        XCTAssertEqual(telemetry.workspaceBytes, 131_584)
        XCTAssertEqual(telemetry.attentionOperation, .materializedKV)
        XCTAssertEqual(telemetry.formatPersistentBytes, 317_440)
        XCTAssertEqual(telemetry.totalPersistentBytes, 317_448)
        XCTAssertEqual(telemetry.storageAndMaterializationBytes, 449_024)
        XCTAssertEqual(telemetry.storageAndWorkspaceBytes, 449_024)

        for cache in caches {
            _ = cache.update(
                keys: MLXArray.zeros([1, 1, 255, 128], dtype: .float16),
                values: MLXArray.zeros([1, 1, 255, 128], dtype: .float16))
        }
        let boundaryTelemetry = KVarNKVCacheTelemetry.capture(caches: caches)
        XCTAssertEqual(boundaryTelemetry.cachedTokens, 256)
        XCTAssertEqual(boundaryTelemetry.completedTileCount, 1)
        XCTAssertEqual(boundaryTelemetry.compressedTokens, 128)

        for cache in caches { cache.resetInPlace() }
        let resetTelemetry = KVarNKVCacheTelemetry.capture(caches: caches)
        XCTAssertEqual(resetTelemetry.cachedTokens, 0)
        XCTAssertEqual(resetTelemetry.completedTileCount, 0)
        XCTAssertEqual(resetTelemetry.compressedTokens, 0)
    }

    func testTelemetryAggregatesMixedLayerIngressAgainstOneNativeStorageDType()
        throws
    {
        let native = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8,
            storageDType: .bfloat16)
        let promoted = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8,
            storageDType: .bfloat16)
        _ = native.update(
            keys: MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16),
            values: MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16))
        _ = promoted.update(
            keys: MLXArray.zeros([1, 1, 1, 128], dtype: .float32),
            values: MLXArray.zeros([1, 1, 1, 128], dtype: .float32))

        let nativeSnapshot = try XCTUnwrap(native.storageSnapshot())
        let promotedSnapshot = try XCTUnwrap(promoted.storageSnapshot())
        let telemetry = KVarNKVCacheTelemetry.capture(
            caches: [native, promoted])

        XCTAssertEqual(
            telemetry.sourceKeyDTypes, [.bfloat16, .float32])
        XCTAssertEqual(
            telemetry.sourceValueDTypes, [.bfloat16, .float32])
        XCTAssertEqual(telemetry.storageKeyDType, .bfloat16)
        XCTAssertEqual(telemetry.storageValueDType, .bfloat16)
        XCTAssertTrue(telemetry.ingressNormalizationApplied)
        XCTAssertEqual(
            telemetry.payloadBytes,
            nativeSnapshot.payloadBytes + promotedSnapshot.payloadBytes)
        XCTAssertEqual(
            telemetry.totalPersistentBytes,
            nativeSnapshot.totalPersistentBytes
                + promotedSnapshot.totalPersistentBytes)

        let highWater = promotedSnapshot.workspaceBytes
            > nativeSnapshot.workspaceBytes
            ? promotedSnapshot : nativeSnapshot
        XCTAssertEqual(
            telemetry.materializationWorkspaceBytes,
            highWater.materializationWorkspaceBytes)
        XCTAssertEqual(
            telemetry.normalizationWorkspaceBytes,
            highWater.normalizationWorkspaceBytes)
        XCTAssertEqual(
            telemetry.attentionWorkspaceBytes,
            highWater.attentionWorkspaceBytes)
        XCTAssertEqual(telemetry.workspaceBytes, highWater.workspaceBytes)
        XCTAssertEqual(
            telemetry.workspaceBytes,
            telemetry.materializationWorkspaceBytes
                + telemetry.normalizationWorkspaceBytes
                + telemetry.attentionWorkspaceBytes)
    }

    func testTelemetryAggregationRejectsMixedPersistentStorageDTypes() throws {
        let float16 = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8,
            storageDType: .float16)
        let bfloat16 = KVarNKVCache(
            capacity: 257, tier: .k4v2G128, iterations: 8,
            storageDType: .bfloat16)
        _ = float16.update(
            keys: MLXArray.zeros([1, 1, 1, 128], dtype: .float16),
            values: MLXArray.zeros([1, 1, 1, 128], dtype: .float16))
        _ = bfloat16.update(
            keys: MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16),
            values: MLXArray.zeros([1, 1, 1, 128], dtype: .bfloat16))
        let snapshots = try [float16, bfloat16].map {
            try XCTUnwrap($0.storageSnapshot())
        }

        XCTAssertThrowsError(try KVarNKVCacheTelemetry.aggregate(
            snapshots: snapshots, cachedTokens: 1,
            completedTileCount: 0)) { error in
                XCTAssertEqual(
                    error as? KVarNKVCacheTelemetryError,
                    .inconsistentStorageDType(layerIndex: 1))
            }
    }

    func testDecoderExportsOnlyScalarTelemetryForSelectedKVarNRuntimeCell() throws {
        var decoder = CompiledMLXDecoder(
            model: TinyKVarNTelemetryModel(), reserve: 1,
            kvCache: .kvarn(.k4v2G128I16), compileStep: true)
        XCTAssertEqual(decoder.executionMode, .uncompiledCorrectness)

        _ = decoder.prefill([1, 2])
        let telemetry = try XCTUnwrap(decoder.kvarnKVTelemetry())
        XCTAssertEqual(telemetry.tier, .k4v2G128)
        XCTAssertEqual(telemetry.iterations, 16)
        XCTAssertEqual(telemetry.executionMode, .uncompiledCorrectness)
        XCTAssertEqual(telemetry.cachedTokens, 3)
        XCTAssertEqual(telemetry.completedTileCount, 0)
        XCTAssertEqual(telemetry.compressedTokens, 0)
        XCTAssertEqual(telemetry.layerCount, 1)
        XCTAssertEqual(telemetry.capacityTokens, 256)
        XCTAssertEqual(telemetry.kvHeadCount, 1)
        XCTAssertEqual(telemetry.headDimension, 128)
        XCTAssertEqual(telemetry.attentionWorkspaceBytes, 0)
        XCTAssertEqual(telemetry.attentionOperation, .materializedKV)
        XCTAssertGreaterThan(telemetry.storageAndWorkspaceBytes, 0)
    }

    private func mismatchCount<T: Equatable>(_ lhs: [T], _ rhs: [T]) -> Int {
        XCTAssertEqual(lhs.count, rhs.count)
        return zip(lhs, rhs).reduce(into: 0) { count, pair in
            if pair.0 != pair.1 { count += 1 }
        }
    }

    private func recordArrays(_ record: KVarNMLXRecord) -> [MLXArray] {
        [
            record.keyPayload, record.keyAbsorbedScale, record.keyAbsorbedBias,
            record.keyTokenScale, record.valuePayload, record.valueChannelScale,
            record.valueAbsorbedScale, record.valueAbsorbedBias,
        ]
    }

    private func assertRecordRow(
        _ combined: KVarNMLXRecord, row: Int, equals single: KVarNMLXRecord,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let pairs: [(MLXArray, MLXArray)] = [
            (combined.keyPayload[row ..< (row + 1), 0..., 0...], single.keyPayload),
            (combined.keyAbsorbedScale[row ..< (row + 1), 0...], single.keyAbsorbedScale),
            (combined.keyAbsorbedBias[row ..< (row + 1), 0...], single.keyAbsorbedBias),
            (combined.keyTokenScale[row ..< (row + 1), 0...], single.keyTokenScale),
            (combined.valuePayload[row ..< (row + 1), 0..., 0...], single.valuePayload),
            (combined.valueChannelScale[row ..< (row + 1), 0...], single.valueChannelScale),
            (combined.valueAbsorbedScale[row ..< (row + 1), 0...], single.valueAbsorbedScale),
            (combined.valueAbsorbedBias[row ..< (row + 1), 0...], single.valueAbsorbedBias),
        ]
        for (actual, expected) in pairs {
            XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
            XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
            XCTAssertTrue((actual .== expected).all().item(Bool.self), file: file, line: line)
        }
    }

    private func deterministicFloat32KVTensor(
        heads: Int, tokens: Int, dimension: Int, seed: Int
    ) -> MLXArray {
        MLXArray((0 ..< heads * tokens * dimension).map { index in
            Float(
                0.45 * sin(Double(index + seed) * 0.017)
                    + 0.20 * cos(Double(index + seed) * 0.031)
                    + 0.01 * Double(((index + seed) % 7) - 3))
        }).reshaped([1, heads, tokens, dimension]).asType(.float32)
    }

    private func assertStorageBytes(
        _ actual: MLXArray,
        equal expected: MLXArray,
        _ message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data,
            expected.asData(access: .copy).data,
            message,
            file: file,
            line: line)
    }

    private func maximumDistance(_ lhs: [Int], _ rhs: [Int]) -> Int {
        XCTAssertEqual(lhs.count, rhs.count)
        return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }

    private func unpack(_ bytes: [UInt8], bits: Int) -> [Int] {
        let valuesPerByte = 8 / bits
        let mask = UInt8((1 << bits) - 1)
        return bytes.flatMap { byte in
            (0 ..< valuesPerByte).map { index in
                Int((byte >> UInt8(index * bits)) & mask)
            }
        }
    }

    private func assertFP16NearExact(
        _ actual: [UInt16], _ expected: [UInt16],
        label: String, iterations: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertLessThanOrEqual(
            mismatchCount(actual, expected), 8,
            "\(label) drift exceeds 8 values at iteration count \(iterations)",
            file: file, line: line)
        let distances = zip(actual, expected).map {
            abs(Int($0) - Int($1))
        }
        XCTAssertLessThanOrEqual(
            distances.max() ?? 0, 1,
            "\(label) drift exceeds one fp16 representation step",
            file: file, line: line)
    }
}
