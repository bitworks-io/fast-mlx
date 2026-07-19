import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// The compressed-batch poison fixture is deliberately end-aligned. Removing the row that
/// established the shared physical boundary must not move the next packed write backward to a
/// surviving row's shorter logical RoPE offset.
final class BatchedAffineKVCacheTests: XCTestCase {
    private let configuration = AffineKVTier.k4v2G64.configuration

    private func tensor(
        batch: Int = 1,
        heads: Int = 2,
        tokens: Int,
        dimension: Int = 128,
        seed: Int
    ) -> MLXArray {
        let count = batch * heads * tokens * dimension
        let values = (0 ..< count).map { index -> Float in
            var mixed = UInt64(index) &+ UInt64(seed) &* 0x9E37_79B9_7F4A_7C15
            mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
            mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
            mixed ^= mixed >> 31
            return Float(Int(mixed % 2_001) - 1_000) / 1_000
        }
        return MLXArray(values).asType(.float16)
            .reshaped([batch, heads, tokens, dimension])
    }

    private func makeScalar(
        length: Int,
        seed: Int,
        capacity: Int = 48,
        configuration: AffineKVCacheConfiguration? = nil,
        attentionMode: AffineKVAttentionMode = .splitQuantizedMM
    ) -> (cache: AffineKVCache, keys: MLXArray, values: MLXArray) {
        let cache = AffineKVCache(
            capacity: capacity,
            configuration: configuration ?? self.configuration,
            attentionMode: attentionMode)
        let keys = tensor(tokens: length, seed: seed)
        let values = tensor(tokens: length, seed: seed + 1_000)
        _ = cache.update(keys: keys, values: values)
        return (cache, keys, values)
    }

    private func scalarArrays(_ cache: AffineKVCache) -> [MLXArray] {
        [
            cache.kPayload!, cache.kScales!, cache.kBiases!,
            cache.vPayload!, cache.vScales!, cache.vBiases!,
        ]
    }

    private func batchArrays(_ cache: BatchedAffineKVCache) -> [MLXArray] {
        [
            cache.kPayload, cache.kScales, cache.kBiases,
            cache.vPayload, cache.vScales, cache.vBiases,
        ]
    }

    private func bytes(_ array: MLXArray) -> [UInt8] {
        view(array, dtype: .uint8).asArray(UInt8.self)
    }

    private func assertExact(
        _ actual: MLXArray,
        _ expected: MLXArray,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(bytes(actual), bytes(expected), file: file, line: line)
    }

    private func assertZero(
        _ array: MLXArray,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(bytes(array).allSatisfy { $0 == 0 }, file: file, line: line)
    }

    func testRaggedMergeMapsAllSixPackedArraysExactly() throws {
        let rows = [
            makeScalar(length: 32, seed: 10),
            makeScalar(length: 48, seed: 20),
            makeScalar(length: 24, seed: 30),
        ]

        let batch = try BatchedAffineKVCache.merging(rows.map(\.cache))

        XCTAssertEqual(batch.capacity, 48)
        XCTAssertEqual(batch.physicalWrittenEnd, 48)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [32, 48, 24])
        XCTAssertEqual(batch.leftPadding, [16, 0, 24])
        XCTAssertTrue(batch.innerState().contains { $0 === batch.physicalEndArr })

        let starts = [16, 0, 24]
        for arrayIndex in 0 ..< 6 {
            let merged = batchArrays(batch)[arrayIndex]
            for row in rows.indices {
                let source = scalarArrays(rows[row].cache)[arrayIndex]
                assertExact(
                    merged[row ..< (row + 1), 0..., starts[row] ..< 48, 0...],
                    source[0..., 0..., 0 ..< rows[row].keys.dim(2), 0...])
                if starts[row] > 0 {
                    assertZero(merged[row ..< (row + 1), 0..., 0 ..< starts[row], 0...])
                }
            }
        }
    }

    func testLongestRowRemovalRetainsPhysicalEndMaskAndPackedAppend() throws {
        let rows = [
            makeScalar(length: 32, seed: 100),
            makeScalar(length: 48, seed: 200),
            makeScalar(length: 24, seed: 300),
        ]
        let batch = try BatchedAffineKVCache.merging(rows.map(\.cache))
        try batch.filter(keeping: [0, 2])
        let survivorBytes = batchArrays(batch).map(bytes)

        XCTAssertEqual(batch.physicalWrittenEnd, 48)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [32, 24])
        XCTAssertEqual(batch.leftPadding, [16, 24])

        batch.grow(by: 3)
        guard case .array(let mask) = batch.makeMask(
            n: 3, windowSize: nil, returnArray: true)
        else { return XCTFail("expected an explicit batch mask") }
        eval(mask)
        XCTAssertEqual(mask.shape, [2, 1, 3, 51])
        for row in 0 ..< 2 {
            let start = [16, 24][row]
            for query in 0 ..< 3 {
                for column in 0 ..< 51 {
                    let expected = column >= start && column <= 48 + query
                    XCTAssertEqual(
                        mask[row, 0, query, column].item(Bool.self), expected,
                        "row \(row), query \(query), column \(column)")
                }
            }
        }

        let appendKeys = tensor(batch: 2, tokens: 3, seed: 400)
        let appendValues = tensor(batch: 2, tokens: 3, seed: 1_400)
        try batch.validateAppend(keys: appendKeys, values: appendValues)
        let queries = tensor(batch: 2, heads: 4, tokens: 3, seed: 2_400)
        let output = batch.updateAndAttend(
            queries: queries,
            keys: appendKeys,
            values: appendValues,
            scale: Float(1 / sqrt(128.0)),
            mask: .array(mask))
        eval(output)

        XCTAssertEqual(batch.physicalWrittenEnd, 51)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [35, 27])
        XCTAssertEqual(batch.leftPadding, [16, 24])
        XCTAssertTrue(isFinite(output).all().item(Bool.self))

        let appendedControls = (0 ..< 2).map { row -> AffineKVCache in
            let control = AffineKVCache(
                capacity: 3,
                configuration: configuration,
                attentionMode: .splitQuantizedMM)
            _ = control.update(
                keys: appendKeys[row ..< (row + 1), 0..., 0..., 0...],
                values: appendValues[row ..< (row + 1), 0..., 0..., 0...])
            return control
        }
        for arrayIndex in 0 ..< 6 {
            let merged = batchArrays(batch)[arrayIndex]
            let before = survivorBytes[arrayIndex]
            XCTAssertEqual(
                bytes(merged[0..., 0..., 0 ..< 48, 0...]),
                Array(before.prefix(merged[0..., 0..., 0 ..< 48, 0...].nbytes)))
            for row in 0 ..< 2 {
                assertExact(
                    merged[row ..< (row + 1), 0..., 48 ..< 51, 0...],
                    scalarArrays(appendedControls[row])[arrayIndex][
                        0..., 0..., 0 ..< 3, 0...])
            }
        }
    }

    func testPoisonTransitionSplitMatchesSamePackedMaterializedControlAndExtract() throws {
        let rows = [
            makeScalar(length: 32, seed: 500),
            makeScalar(length: 48, seed: 600),
            makeScalar(length: 24, seed: 700),
        ]
        let batch = try BatchedAffineKVCache.merging(rows.map(\.cache))
        try batch.filter(keeping: [0, 2])
        batch.grow(by: 3)
        guard case .array(let mask) = batch.makeMask(
            n: 3, windowSize: nil, returnArray: true)
        else { return XCTFail("expected an explicit batch mask") }

        let keys = tensor(batch: 2, tokens: 3, seed: 800)
        let values = tensor(batch: 2, tokens: 3, seed: 1_800)
        let queries = tensor(batch: 2, heads: 4, tokens: 3, seed: 2_800)
        let scale = Float(1 / sqrt(128.0))
        let scalarControls = [rows[0], rows[2]].map { original -> AffineKVCache in
            let cache = AffineKVCache(
                capacity: 51,
                configuration: configuration,
                attentionMode: .splitQuantizedMM)
            _ = cache.update(keys: original.keys, values: original.values)
            return cache
        }
        let scalarSplit = concatenated(
            (0 ..< 2).map { row in
                let scalarMask = scalarControls[row].makeMask(
                    n: 3, windowSize: nil, returnArray: true)
                return scalarControls[row].updateAndAttend(
                    queries: queries[row ..< (row + 1), 0..., 0..., 0...],
                    keys: keys[row ..< (row + 1), 0..., 0..., 0...],
                    values: values[row ..< (row + 1), 0..., 0..., 0...],
                    scale: scale,
                    mask: scalarMask)
            },
            axis: 0)
        let split = batch.updateAndAttend(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: .array(mask))
        let (materializedKeys, materializedValues) = batch.materializedStoredKV()
        let control = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: materializedKeys,
            values: materializedValues,
            scale: scale,
            mask: .array(mask))
        eval(split, scalarSplit, control)

        XCTAssertTrue(isFinite(split).all().item(Bool.self))
        XCTAssertTrue(
            split.allClose(scalarSplit, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "batch compaction introduced a delta beyond the scalar compressed path; "
                + "same-row=\((0 ..< 2).map { (split[$0] - scalarSplit[$0]).abs().max().item(Float.self) }), "
                + "cross-row=\((0 ..< 2).map { (split[$0] - scalarSplit[1 - $0]).abs().max().item(Float.self) })")
        let maximumDelta = (split - control).abs().max().item(Float.self)
        let rowDeltas = (0 ..< 2).map {
            (split[$0] - control[$0]).abs().max().item(Float.self)
        }
        XCTAssertTrue(
            split.allClose(control, rtol: 5e-2, atol: 5e-2).item(Bool.self),
            "same-packed control delta max=\(maximumDelta), rows=\(rowDeltas)")
        let projection = tensor(batch: 1, heads: 1, tokens: 8, seed: 3_800)
            .reshaped([128, 8]).asType(.float32)
        let splitLogits = matmul(
            split.mean(axes: [1, 2], keepDims: false).asType(.float32),
            projection)
        let controlLogits = matmul(
            control.mean(axes: [1, 2], keepDims: false).asType(.float32),
            projection)
        XCTAssertEqual(
            argMax(splitLogits, axis: -1).asArray(Int32.self),
            argMax(controlLogits, axis: -1).asArray(Int32.self))

        for row in 0 ..< 2 {
            let extracted = try batch.extract(slot: row)
            let expected = scalarControls[row]
            XCTAssertEqual(
                extracted.offsetArr.item(Int32.self),
                expected.offsetArr.item(Int32.self))
            for arrayIndex in 0 ..< 6 {
                assertExact(
                    scalarArrays(extracted)[arrayIndex],
                    scalarArrays(expected)[arrayIndex])
            }
        }
    }

    func testCompiledReplayCapturesAndAdvancesPhysicalEndInGraphState() throws {
        let rows = [
            makeScalar(length: 32, seed: 900),
            makeScalar(length: 48, seed: 1_000),
            makeScalar(length: 24, seed: 1_100),
        ]
        let batch = try BatchedAffineKVCache.merging(rows.map(\.cache))
        try batch.filter(keeping: [0, 2])
        batch.grow(by: 3)
        let scale = Float(1 / sqrt(128.0))

        let step = compile(inputs: [batch], outputs: [batch]) { arguments in
            let mask = batch.makeMask(n: 1, windowSize: nil, returnArray: true)
            return [batch.updateAndAttend(
                queries: arguments[0],
                keys: arguments[1],
                values: arguments[2],
                scale: scale,
                mask: mask)]
        }

        for replay in 0 ..< 3 {
            let output = step([
                tensor(batch: 2, heads: 4, tokens: 1, seed: 1_200 + replay * 10),
                tensor(batch: 2, tokens: 1, seed: 1_201 + replay * 10),
                tensor(batch: 2, tokens: 1, seed: 1_202 + replay * 10),
            ])[0]
            eval(output)
            XCTAssertTrue(isFinite(output).all().item(Bool.self))
            XCTAssertEqual(batch.physicalEndArr.item(Int32.self), Int32(49 + replay))
            XCTAssertEqual(
                batch.batchOffset.asArray(Int32.self),
                [Int32(33 + replay), Int32(25 + replay)])
        }
    }

    func testMergeAndMembershipFailuresAreRecoverableAndNonMutating() throws {
        XCTAssertThrowsError(try BatchedAffineKVCache.merging([]))
        XCTAssertThrowsError(
            try BatchedAffineKVCache.merging([
                AffineKVCache(
                    capacity: 48,
                    configuration: configuration,
                    attentionMode: .splitQuantizedMM),
            ]))

        let first = makeScalar(length: 8, seed: 1_500)
        let differentConfiguration = makeScalar(
            length: 8,
            seed: 1_600,
            configuration: AffineKVTier.k8v2G64.configuration)
        XCTAssertThrowsError(
            try BatchedAffineKVCache.merging([
                first.cache, differentConfiguration.cache,
            ]))
        XCTAssertThrowsError(
            try BatchedAffineKVCache.merging(
                [first.cache, makeScalar(length: 8, seed: 1_700).cache],
                lengths: [7, 8]))

        let batch = try BatchedAffineKVCache.merging([
            first.cache, makeScalar(length: 8, seed: 1_800).cache,
        ])
        let beforeArrays = batchArrays(batch).map(bytes)
        let beforeOffsets = batch.batchOffset.asArray(Int32.self)
        let beforePhysicalEnd = batch.physicalWrittenEnd
        XCTAssertThrowsError(try batch.filter(keeping: []))
        XCTAssertThrowsError(try batch.filter(keeping: [0, 0]))
        XCTAssertThrowsError(try batch.requireCapacity(for: 41))

        let nonFinite = MLXArray.full(
            [2, 2, 1, 128], values: MLXArray(Float.nan), dtype: .float16)
        XCTAssertThrowsError(try batch.validateAppend(keys: nonFinite, values: nonFinite))

        XCTAssertEqual(batchArrays(batch).map(bytes), beforeArrays)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), beforeOffsets)
        XCTAssertEqual(batch.physicalWrittenEnd, beforePhysicalEnd)
    }
}
