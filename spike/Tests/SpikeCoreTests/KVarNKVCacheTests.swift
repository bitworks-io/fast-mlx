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
        XCTAssertEqual(snapshot.storageAndMaterializationBytes, 290_304)
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
        XCTAssertEqual(telemetry.formatPersistentBytes, 317_440)
        XCTAssertEqual(telemetry.totalPersistentBytes, 317_448)
        XCTAssertEqual(telemetry.storageAndMaterializationBytes, 449_024)

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
        XCTAssertGreaterThan(telemetry.storageAndMaterializationBytes, 0)
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
