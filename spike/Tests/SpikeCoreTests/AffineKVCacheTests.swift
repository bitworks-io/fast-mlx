import MLX
import MLXRandom
import XCTest

@testable import SpikeCore

final class AffineKVCacheTests: XCTestCase {
    private func randomKV(
        batch: Int, heads: Int, tokens: Int, dimension: Int, seed: UInt64
    ) -> MLXArray {
        MLXRandom.normal(
            [batch, heads, tokens, dimension],
            key: MLXRandom.key(seed)).asType(.float16)
    }

    private func nativeRoundTrip(
        _ input: MLXArray, bits: Int, groupSize: Int
    ) -> MLXArray {
        let dimension = input.dim(3)
        let flat = input.reshaped([-1, dimension])
        let code = quantized(
            flat, groupSize: groupSize, bits: bits, mode: .affine)
        return dequantized(
            code.wq, scales: code.scales, biases: code.biases,
            groupSize: groupSize, bits: bits, mode: .affine,
            dtype: input.dtype).reshaped(input.shape)
    }

    private func configuration(
        keyBits: Int = 4, valueBits: Int = 2,
        keyGroupSize: Int = 128, valueGroupSize: Int = 128
    ) throws -> AffineKVCacheConfiguration {
        try AffineKVCacheConfiguration(
            keyBits: keyBits, valueBits: valueBits,
            keyGroupSize: keyGroupSize, valueGroupSize: valueGroupSize)
    }

    func testK4V2G128MaterializesNativeRoundTripAndReportsExactArrays() throws {
        let configuration = try AffineKVCacheConfiguration(
            keyBits: 4, valueBits: 2,
            keyGroupSize: 128, valueGroupSize: 128)
        let cache = AffineKVCache(capacity: 8, configuration: configuration)
        let reference = CompiledKVCache(capacity: 8)

        let k1 = randomKV(batch: 1, heads: 2, tokens: 3, dimension: 128, seed: 1)
        let v1 = randomKV(batch: 1, heads: 2, tokens: 3, dimension: 128, seed: 2)
        let k2 = randomKV(batch: 1, heads: 2, tokens: 1, dimension: 128, seed: 3)
        let v2 = randomKV(batch: 1, heads: 2, tokens: 1, dimension: 128, seed: 4)

        _ = cache.update(keys: k1, values: v1)
        _ = reference.update(keys: k1, values: v1)
        let (materializedKeys, materializedValues) = cache.update(keys: k2, values: v2)
        let (referenceKeys, referenceValues) = reference.update(keys: k2, values: v2)

        XCTAssertEqual(materializedKeys.shape, referenceKeys.shape)
        XCTAssertEqual(materializedValues.shape, referenceValues.shape)
        XCTAssertEqual(cache.offset, reference.offset)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), reference.offsetArr.item(Int32.self))
        guard case .array(let mask) = cache.makeMask(
            n: 1, windowSize: nil, returnArray: true),
            case .array(let referenceMask) = reference.makeMask(
                n: 1, windowSize: nil, returnArray: true)
        else { return XCTFail("expected fixed-capacity masks") }
        XCTAssertTrue((mask .== referenceMask).all().item(Bool.self))

        let allKeys = concatenated([k1, k2], axis: 2)
        let allValues = concatenated([v1, v2], axis: 2)
        let keyError = (
            materializedKeys[0..., 0..., 0 ..< 4, 0...]
                - nativeRoundTrip(allKeys, bits: 4, groupSize: 128)
        ).abs().max().item(Float.self)
        let valueError = (
            materializedValues[0..., 0..., 0 ..< 4, 0...]
                - nativeRoundTrip(allValues, bits: 2, groupSize: 128)
        ).abs().max().item(Float.self)
        XCTAssertEqual(keyError, 0, accuracy: 1e-4)
        XCTAssertEqual(valueError, 0, accuracy: 1e-4)
        XCTAssertGreaterThan(
            (materializedKeys[0..., 0..., 0 ..< 4, 0...] - allKeys)
                .abs().max().item(Float.self),
            1e-3)
        XCTAssertEqual(materializedKeys[0..., 0..., 4..., 0...].abs().max().item(Float.self), 0)
        XCTAssertEqual(materializedValues[0..., 0..., 4..., 0...].abs().max().item(Float.self), 0)

        XCTAssertEqual(cache.kPayload?.shape, [1, 2, 8, 16])
        XCTAssertEqual(cache.vPayload?.shape, [1, 2, 8, 8])
        XCTAssertEqual(cache.kScales?.shape, [1, 2, 8, 1])
        XCTAssertEqual(cache.kBiases?.shape, [1, 2, 8, 1])
        XCTAssertEqual(cache.vScales?.shape, [1, 2, 8, 1])
        XCTAssertEqual(cache.vBiases?.shape, [1, 2, 8, 1])
        XCTAssertEqual(cache.kPayload?.dtype, .uint32)
        XCTAssertEqual(cache.vPayload?.dtype, .uint32)
        XCTAssertEqual(cache.kScales?.itemSize, 2)
        XCTAssertEqual(cache.kBiases?.itemSize, 2)
        XCTAssertEqual(cache.vScales?.itemSize, 2)
        XCTAssertEqual(cache.vBiases?.itemSize, 2)

        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.payloadBytes, 1_536)
        XCTAssertEqual(storage.metadataBytes, 128)
        XCTAssertEqual(storage.controlBytes, 4)
        XCTAssertEqual(storage.materializationWorkspaceBytes, 8_192)
        XCTAssertEqual(storage.dataArrayBytes, 1_664)
        XCTAssertEqual(storage.totalPersistentBytes, 1_668)
        XCTAssertEqual(cache.innerState().count, 7)
    }

    func testConfigurationRejectsUnsupportedBitsAndGroupsWithoutFallback() {
        XCTAssertThrowsError(try configuration(keyBits: 3)) {
            XCTAssertEqual(
                $0 as? AffineKVCacheConfigurationError,
                .unsupportedBitWidth(3))
        }
        XCTAssertThrowsError(try configuration(valueBits: 6)) {
            XCTAssertEqual(
                $0 as? AffineKVCacheConfigurationError,
                .unsupportedBitWidth(6))
        }
        XCTAssertThrowsError(try configuration(keyGroupSize: 96)) {
            XCTAssertEqual(
                $0 as? AffineKVCacheConfigurationError,
                .unsupportedGroupSize(96))
        }
        XCTAssertThrowsError(try configuration(valueGroupSize: 16)) {
            XCTAssertEqual(
                $0 as? AffineKVCacheConfigurationError,
                .unsupportedGroupSize(16))
        }
    }

    func testAffineTierNamesMapFailClosedAndBuildTheRequestedCache() {
        let expected: [(String, AffineKVTier)] = [
            ("affine-k4v2-g64", .k4v2G64),
            ("affine-k4v2-g128", .k4v2G128),
            ("affine-k8v2-g64", .k8v2G64),
            ("affine-k8v2-g128", .k8v2G128),
            ("affine-k4v4-g128", .k4v4G128),
        ]

        for (name, tier) in expected {
            XCTAssertEqual(KVCacheKind(kvQuant: name), .affine(tier))
            guard let cache = KVCacheKind.affine(tier).makeCache(capacity: 7)
                as? AffineKVCache
            else { return XCTFail("expected AffineKVCache for \(name)") }
            XCTAssertEqual(cache.capacity, 7)
            XCTAssertEqual(cache.configuration.keyBits, tier.keyBits)
            XCTAssertEqual(cache.configuration.valueBits, tier.valueBits)
            XCTAssertEqual(cache.configuration.keyGroupSize, tier.groupSize)
            XCTAssertEqual(cache.configuration.valueGroupSize, tier.groupSize)
        }

        XCTAssertNil(KVCacheKind(kvQuant: "affine-k4v2-g96"))
        XCTAssertNil(KVCacheKind(kvQuant: "k4v2-g128"))
        XCTAssertNil(KVCacheKind(kvQuant: "affine-k3v2-g128"))
    }

    func testDeclaredAffineCellsReportTheirActualPackedShapesAndBytes() throws {
        struct Cell {
            let keyBits: Int
            let valueBits: Int
            let groupSize: Int
            let keyPayloadWidth: Int
            let valuePayloadWidth: Int
            let metadataWidth: Int
            let payloadBytes: Int
            let metadataBytes: Int
        }
        let cells = [
            Cell(
                keyBits: 4, valueBits: 2, groupSize: 64,
                keyPayloadWidth: 16, valuePayloadWidth: 8, metadataWidth: 2,
                payloadBytes: 480, metadataBytes: 80),
            Cell(
                keyBits: 8, valueBits: 2, groupSize: 64,
                keyPayloadWidth: 32, valuePayloadWidth: 8, metadataWidth: 2,
                payloadBytes: 800, metadataBytes: 80),
            Cell(
                keyBits: 8, valueBits: 2, groupSize: 128,
                keyPayloadWidth: 32, valuePayloadWidth: 8, metadataWidth: 1,
                payloadBytes: 800, metadataBytes: 40),
            Cell(
                keyBits: 4, valueBits: 4, groupSize: 128,
                keyPayloadWidth: 16, valuePayloadWidth: 16, metadataWidth: 1,
                payloadBytes: 640, metadataBytes: 40),
        ]

        for cell in cells {
            let cache = AffineKVCache(
                capacity: 5,
                configuration: try configuration(
                    keyBits: cell.keyBits, valueBits: cell.valueBits,
                    keyGroupSize: cell.groupSize, valueGroupSize: cell.groupSize))
            XCTAssertNil(cache.storageSnapshot())
            _ = cache.update(
                keys: randomKV(batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 20),
                values: randomKV(batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 21))

            XCTAssertEqual(cache.kPayload?.shape, [1, 1, 5, cell.keyPayloadWidth])
            XCTAssertEqual(cache.vPayload?.shape, [1, 1, 5, cell.valuePayloadWidth])
            XCTAssertEqual(cache.kScales?.shape, [1, 1, 5, cell.metadataWidth])
            XCTAssertEqual(cache.kBiases?.shape, [1, 1, 5, cell.metadataWidth])
            XCTAssertEqual(cache.vScales?.shape, [1, 1, 5, cell.metadataWidth])
            XCTAssertEqual(cache.vBiases?.shape, [1, 1, 5, cell.metadataWidth])

            let storage = try XCTUnwrap(cache.storageSnapshot())
            XCTAssertEqual(storage.payloadBytes, cell.payloadBytes)
            XCTAssertEqual(storage.metadataBytes, cell.metadataBytes)
            XCTAssertEqual(storage.dataArrayBytes, cell.payloadBytes + cell.metadataBytes)
            XCTAssertEqual(storage.controlBytes, 4)
        }
    }

    func testKeyAndValueGroupSizesRemainIndependent() throws {
        let cache = AffineKVCache(
            capacity: 3,
            configuration: try configuration(
                keyGroupSize: 64, valueGroupSize: 128))
        _ = cache.update(
            keys: randomKV(batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 25),
            values: randomKV(batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 26))

        XCTAssertEqual(cache.configuration.keyGroupSize, 64)
        XCTAssertEqual(cache.configuration.valueGroupSize, 128)
        XCTAssertEqual(cache.kScales?.shape, [1, 1, 3, 2])
        XCTAssertEqual(cache.kBiases?.shape, [1, 1, 3, 2])
        XCTAssertEqual(cache.vScales?.shape, [1, 1, 3, 1])
        XCTAssertEqual(cache.vBiases?.shape, [1, 1, 3, 1])
        let storage = try XCTUnwrap(cache.storageSnapshot())
        XCTAssertEqual(storage.payloadBytes, 288)
        XCTAssertEqual(storage.metadataBytes, 36)
        XCTAssertEqual(storage.totalPersistentBytes, 328)
    }

    func testTelemetryAggregatesAllLayersAndKeepsControlBytesExplicit() throws {
        let caches: [AffineKVCache] = (0 ..< 3).map { _ in
            AffineKVCache(
                capacity: 8,
                configuration: AffineKVTier.k4v2G128.configuration)
        }
        let keys = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 27)
        let values = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 28)
        for cache in caches { _ = cache.update(keys: keys, values: values) }

        let telemetry = AffineKVCacheTelemetry.capture(
            tier: .k4v2G128,
            caches: caches)

        XCTAssertEqual(telemetry.tier, .k4v2G128)
        XCTAssertEqual(telemetry.cachedTokens, 2)
        XCTAssertEqual(telemetry.layerCount, 3)
        XCTAssertEqual(telemetry.capacityTokens, 8)
        XCTAssertEqual(telemetry.sequences, 1)
        XCTAssertEqual(telemetry.kvHeadCount, 2)
        XCTAssertEqual(telemetry.headDimension, 128)
        XCTAssertEqual(telemetry.metadataScalarBytes, 2)
        XCTAssertEqual(telemetry.payloadBytes, 4_608)
        XCTAssertEqual(telemetry.metadataBytes, 384)
        XCTAssertEqual(telemetry.controlBytes, 12)
        // Layers execute sequentially, so the logical materialization workspace is one
        // full K/V pair, not the sum of all three layer-local pairs.
        XCTAssertEqual(telemetry.materializationWorkspaceBytes, 8_192)
        XCTAssertEqual(telemetry.dataArrayBytes, 4_992)
        XCTAssertEqual(telemetry.totalPersistentBytes, 5_004)
    }

    func testGrowTruncateAndResetPreserveTheCompiledStateContract() throws {
        let cache = AffineKVCache(capacity: 4, configuration: try configuration())
        let initialKeys = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 30)
        let initialValues = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 31)
        _ = cache.update(keys: initialKeys, values: initialValues)

        cache.grow(by: 4)
        XCTAssertEqual(cache.capacity, 8)
        XCTAssertEqual(cache.kPayload?.shape, [1, 2, 8, 16])
        let nextKeys = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 32)
        let nextValues = randomKV(
            batch: 1, heads: 2, tokens: 2, dimension: 128, seed: 33)
        _ = cache.update(keys: nextKeys, values: nextValues)

        let stateObjects = cache.innerState()
        cache.truncate(to: 3)
        XCTAssertTrue(
            zip(stateObjects, cache.innerState()).allSatisfy { $0 === $1 },
            "truncate must preserve every compiled-state object")
        let replacementKeys = randomKV(
            batch: 1, heads: 2, tokens: 1, dimension: 128, seed: 34)
        let replacementValues = randomKV(
            batch: 1, heads: 2, tokens: 1, dimension: 128, seed: 35)
        let (materializedKeys, _) = cache.update(
            keys: replacementKeys, values: replacementValues)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 4)
        XCTAssertEqual(
            (materializedKeys[0..., 0..., 3 ..< 4, 0...]
                - nativeRoundTrip(replacementKeys, bits: 4, groupSize: 128))
                .abs().max().item(Float.self),
            0, accuracy: 1e-4)

        let objectsBeforeReset = cache.innerState()
        cache.resetInPlace()
        XCTAssertTrue(
            zip(objectsBeforeReset, cache.innerState()).allSatisfy { $0 === $1 },
            "reset must preserve every compiled-state object")
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        for array in cache.innerState().dropLast() {
            XCTAssertEqual(array.abs().max().item(Float.self), 0)
        }
    }

    func testCompiledSingleTokenReplayAdvancesOffsetAndWritesDistinctRows() throws {
        let cache = AffineKVCache(capacity: 6, configuration: try configuration())
        let prefillKeys = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 40)
        let prefillValues = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 41)
        _ = cache.update(keys: prefillKeys, values: prefillValues)

        let step = compile(inputs: [cache], outputs: [cache]) { inputs in
            let (keys, values) = cache.update(keys: inputs[0], values: inputs[1])
            return [keys, values]
        }
        let decodeKeysA = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 42)
        let decodeValuesA = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 43)
        eval(step([decodeKeysA, decodeValuesA]))
        let decodeKeysB = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 44)
        let decodeValuesB = randomKV(
            batch: 1, heads: 1, tokens: 1, dimension: 128, seed: 45)
        let replay = step([decodeKeysB, decodeValuesB])
        eval(replay)

        XCTAssertEqual(cache.offsetArr.item(Int32.self), 3)
        let expected = nativeRoundTrip(
            concatenated([prefillKeys, decodeKeysA, decodeKeysB], axis: 2),
            bits: 4, groupSize: 128)
        XCTAssertEqual(
            (replay[0][0..., 0..., 0 ..< 3, 0...] - expected)
                .abs().max().item(Float.self),
            0, accuracy: 1e-4)
        XCTAssertGreaterThan(
            (replay[0][0, 0, 1, 0...] - replay[0][0, 0, 2, 0...])
                .abs().max().item(Float.self),
            1e-3,
            "compiled replay must not overwrite the first decode row")
    }
}
