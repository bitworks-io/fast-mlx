import MLX
import XCTest

@testable import SpikeCore

/// Dense continuous-batching cache semantics. Every row retains the scalar cache's
/// right-padded layout and uses its own logical write column and RoPE offset.
final class BatchedCompiledKVCacheTests: XCTestCase {

    private func makeScalar(
        _ values: [Float], capacity: Int = 8, heads: Int = 1
    ) -> CompiledKVCache {
        let cache = CompiledKVCache(capacity: capacity)
        let key = MLXArray(
            values.flatMap { Array(repeating: $0, count: heads) }
        ).reshaped([1, heads, values.count, 1]).transposed(0, 1, 2, 3)
        let value = key + 100
        _ = cache.update(keys: key, values: value)
        return cache
    }

    func testMergeKeepsScalarHistoryPositionsAndLogicalOffsets() throws {
        let long = makeScalar([10, 11, 12])
        let short = makeScalar([20])

        let batch = try BatchedCompiledKVCache.merging([long, short])

        XCTAssertEqual(batch.batchSize, 2)
        XCTAssertEqual(batch.capacity, 8)
        XCTAssertEqual(batch.offset, 3)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [3, 1])
        XCTAssertEqual(batch.keysBuf!.shape, [2, 1, 8, 1])

        XCTAssertEqual(batch.keysBuf![0, 0, 0, 0].item(Float.self), 10)
        XCTAssertEqual(batch.keysBuf![0, 0, 2, 0].item(Float.self), 12)
        XCTAssertEqual(batch.keysBuf![1, 0, 0, 0].item(Float.self), 20)
        XCTAssertEqual(batch.keysBuf![1, 0, 1, 0].item(Float.self), 0)
    }

    func testMaskMatchesScalarPrefixesAndRejectsUnusedTail() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12]),
            makeScalar([20]),
        ])

        guard case .array(let mask) = batch.makeMask(n: 1, windowSize: nil, returnArray: true)
        else {
            return XCTFail("expected an array mask")
        }

        XCTAssertEqual(mask.shape, [2, 1, 1, 8])
        let expected = [
            [true, true, true, true, false, false, false, false],
            [true, true, false, false, false, false, false, false],
        ]
        for row in expected.indices {
            for column in expected[row].indices {
                XCTAssertEqual(
                    mask[row, 0, 0, column].item(Bool.self), expected[row][column],
                    "row \(row), key position \(column)")
            }
        }
    }

    func testDecodeUpdateUsesPerRowWriteColumnsAndAdvancesLogicalOffsets() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12]),
            makeScalar([20]),
        ])
        let keys = MLXArray([Float(13), 21]).reshaped([2, 1, 1, 1])
        let values = keys + 100

        let (updatedKeys, updatedValues) = batch.update(keys: keys, values: values)

        XCTAssertEqual(batch.offset, 4)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [4, 2])
        XCTAssertEqual(updatedKeys[0, 0, 3, 0].item(Float.self), 13)
        XCTAssertEqual(updatedKeys[1, 0, 1, 0].item(Float.self), 21)
        XCTAssertEqual(updatedKeys[1, 0, 2, 0].item(Float.self), 0)
        XCTAssertEqual(updatedValues[1, 0, 1, 0].item(Float.self), 121)
    }

    func testMultiTokenUpdateScattersAContiguousRangePerRow() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12]),
            makeScalar([20]),
        ])
        let keys = MLXArray([Float(13), 14, 21, 22]).reshaped([2, 1, 2, 1])

        let (updatedKeys, _) = batch.update(keys: keys, values: keys + 100)

        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [5, 3])
        XCTAssertEqual(updatedKeys[0, 0, 3, 0].item(Float.self), 13)
        XCTAssertEqual(updatedKeys[0, 0, 4, 0].item(Float.self), 14)
        XCTAssertEqual(updatedKeys[1, 0, 1, 0].item(Float.self), 21)
        XCTAssertEqual(updatedKeys[1, 0, 2, 0].item(Float.self), 22)
        XCTAssertEqual(updatedKeys[1, 0, 3, 0].item(Float.self), 0)
    }

    func testExtractRestoresScalarHistories() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12]),
            makeScalar([20]),
        ])
        let keys = MLXArray([Float(13), 21]).reshaped([2, 1, 1, 1])
        _ = batch.update(keys: keys, values: keys + 100)

        let first = try batch.extract(slot: 0)
        let second = try batch.extract(slot: 1)

        XCTAssertEqual(first.offsetArr.item(Int32.self), 4)
        XCTAssertEqual(second.offsetArr.item(Int32.self), 2)
        XCTAssertEqual(first.keysBuf![0, 0, 0, 0].item(Float.self), 10)
        XCTAssertEqual(first.keysBuf![0, 0, 3, 0].item(Float.self), 13)
        XCTAssertEqual(second.keysBuf![0, 0, 0, 0].item(Float.self), 20)
        XCTAssertEqual(second.keysBuf![0, 0, 1, 0].item(Float.self), 21)
        XCTAssertEqual(second.keysBuf![0, 0, 2, 0].item(Float.self), 0)
        XCTAssertEqual(second.valuesBuf![0, 0, 1, 0].item(Float.self), 121)
    }

    func testFilterPreservesRequestedOrderAndRowMetadata() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11]),
            makeScalar([20]),
            makeScalar([30, 31, 32]),
        ])

        try batch.filter(keeping: [2, 0])

        XCTAssertEqual(batch.batchSize, 2)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [3, 2])
        XCTAssertEqual(batch.keysBuf![0, 0, 0, 0].item(Float.self), 30)
        XCTAssertEqual(batch.keysBuf![0, 0, 2, 0].item(Float.self), 32)
        XCTAssertEqual(batch.keysBuf![1, 0, 0, 0].item(Float.self), 10)
        XCTAssertEqual(batch.keysBuf![1, 0, 1, 0].item(Float.self), 11)
        XCTAssertEqual(batch.keysBuf![1, 0, 2, 0].item(Float.self), 0)
    }

    func testGrowPreservesBuffersAndBatchMetadata() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12], capacity: 4),
            makeScalar([20], capacity: 4),
        ])

        batch.grow(by: 4)

        XCTAssertEqual(batch.capacity, 8)
        XCTAssertEqual(batch.keysBuf!.shape, [2, 1, 8, 1])
        XCTAssertEqual(batch.keysBuf![1, 0, 0, 0].item(Float.self), 20)
        XCTAssertEqual(batch.keysBuf![0, 0, 6, 0].item(Float.self), 0)
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [3, 1])
    }

    func testCapacityValidationReadsAuthoritativePerRowOffsets() throws {
        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11, 12], capacity: 4),
            makeScalar([20], capacity: 4),
        ])

        XCTAssertNoThrow(try batch.requireCapacity(for: 1))
        XCTAssertThrowsError(try batch.requireCapacity(for: 2))
        XCTAssertThrowsError(try batch.requireCapacity(for: -1))
    }

    func testMergeAndMembershipChangesFailClosedOnInvalidState() throws {
        XCTAssertThrowsError(try BatchedCompiledKVCache.merging([]))
        XCTAssertThrowsError(
            try BatchedCompiledKVCache.merging([CompiledKVCache(capacity: 8)]))
        XCTAssertThrowsError(
            try BatchedCompiledKVCache.merging([
                makeScalar([1], heads: 1),
                makeScalar([2], heads: 2),
            ]))
        XCTAssertThrowsError(
            try BatchedCompiledKVCache.merging(
                [makeScalar([1]), makeScalar([2])], lengths: [1]))
        XCTAssertThrowsError(
            try BatchedCompiledKVCache.merging(
                [makeScalar([1]), makeScalar([2])], lengths: [2, 1]))

        let batch = try BatchedCompiledKVCache.merging([
            makeScalar([10, 11]),
            makeScalar([20]),
        ])
        XCTAssertThrowsError(try batch.extract(slot: 2))
        XCTAssertThrowsError(try batch.filter(keeping: []))
        XCTAssertThrowsError(try batch.filter(keeping: [0, 0]))
        XCTAssertThrowsError(try batch.filter(keeping: [2]))
    }
}
