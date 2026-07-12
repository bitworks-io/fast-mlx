import MLX
import XCTest

@testable import SpikeCore

/// Semantics of the compile-legal KV cache (runs on Metal-capable hosts; these are
/// the invariants the compiled decode step depends on).
final class CompiledKVCacheTests: XCTestCase {

    private func makeKV(_ value: Float, n: Int = 1) -> (MLXArray, MLXArray) {
        // [B=1, kvHeads=2, n, headDim=4]
        let k = MLXArray.full([1, 2, n, 4], values: MLXArray(value))
        let v = MLXArray.full([1, 2, n, 4], values: MLXArray(value * 10))
        return (k, v)
    }

    func testUpdateWritesAtOffsetAndAdvancesInGraph() {
        let cache = CompiledKVCache(capacity: 8)

        let (k1, v1) = makeKV(1, n: 3) // prefill: 3 tokens at positions 0..2
        let (keysA, _) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(keysA.shape, [1, 2, 8, 4]) // full fixed-size buffer
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 3)

        let (k2, v2) = makeKV(2, n: 1) // decode: 1 token at position 3
        let (keysB, valuesB) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 4)

        // row 2 still from prefill, row 3 from the decode write, row 4 untouched (0)
        XCTAssertEqual(keysB[0, 0, 2, 0].item(Float.self), 1)
        XCTAssertEqual(keysB[0, 0, 3, 0].item(Float.self), 2)
        XCTAssertEqual(valuesB[0, 1, 3, 3].item(Float.self), 20)
        XCTAssertEqual(keysB[0, 0, 4, 0].item(Float.self), 0)
    }

    func testMaskCoversExactlyThePastAndSelf() {
        let cache = CompiledKVCache(capacity: 6)
        let (k, v) = makeKV(1, n: 2)
        _ = cache.update(keys: k, values: v) // offset -> 2

        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: nil, returnArray: true)
        else {
            return XCTFail("expected an array mask")
        }
        XCTAssertEqual(mask.shape, [1, 1, 1, 6])
        // query at position 2 attends to keys 0,1,2 only
        let expected: [Bool] = [true, true, true, false, false, false]
        for (j, e) in expected.enumerated() {
            XCTAssertEqual(mask[0, 0, 0, j].item(Bool.self), e, "key position \(j)")
        }
    }

    func testResetInPlacePreservesArrayIdentity() {
        let cache = CompiledKVCache(capacity: 4)
        let (k, v) = makeKV(3, n: 2)
        _ = cache.update(keys: k, values: v)

        // the compiled step is bound to these exact MLXArray objects
        let keysObject = cache.keysBuf!
        let offsetObject = cache.offsetArr

        cache.resetInPlace()

        XCTAssertTrue(cache.keysBuf! === keysObject)
        XCTAssertTrue(cache.offsetArr === offsetObject)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
        XCTAssertEqual(cache.keysBuf![0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(cache.offset, 0)
    }

    /// Speculative-decoding rollback: a verify forward writes K+1 rows, the accept-walk
    /// rejects a suffix, and `truncate(to:)` must make the cache behave as if only the
    /// kept prefix was ever appended — same offset, same mask, next write lands at the
    /// kept length — while preserving the MLXArray identities a compiled step is bound to.
    func testTruncateRollsBackToLengthAndNextUpdateOverwrites() {
        let cache = CompiledKVCache(capacity: 8)
        let (k1, v1) = makeKV(1, n: 3) // "prompt": positions 0..2
        _ = cache.update(keys: k1, values: v1)
        let (k2, v2) = makeKV(2, n: 3) // "verify forward": positions 3..5
        _ = cache.update(keys: k2, values: v2)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 6)

        let keysObject = cache.keysBuf!
        let valuesObject = cache.valuesBuf!
        let offsetObject = cache.offsetArr

        cache.truncate(to: 4) // reject the writes at positions 4..5

        // identity preserved (the compiled step stays bound), offset rolled back in-graph
        XCTAssertTrue(cache.keysBuf! === keysObject)
        XCTAssertTrue(cache.valuesBuf! === valuesObject)
        XCTAssertTrue(cache.offsetArr === offsetObject)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 4)
        XCTAssertEqual(cache.offset, 4)

        // the mask treats positions past 4 as unwritten: the next query (position 4)
        // may attend to 0...4 only — the stale rows at 5.. are excluded
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: nil, returnArray: true)
        else {
            return XCTFail("expected an array mask")
        }
        let expected: [Bool] = [true, true, true, true, true, false, false, false]
        for (j, e) in expected.enumerated() {
            XCTAssertEqual(mask[0, 0, 0, j].item(Bool.self), e, "key position \(j)")
        }

        // the next update behaves as if only 4 rows were ever appended: it writes AT 4,
        // overwriting the stale rejected row
        let (k3, v3) = makeKV(9, n: 1)
        let (keys, values) = cache.update(keys: k3, values: v3)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 5)
        XCTAssertEqual(keys[0, 0, 3, 0].item(Float.self), 2) // kept verify row survives
        XCTAssertEqual(keys[0, 0, 4, 0].item(Float.self), 9) // rejected row overwritten
        XCTAssertEqual(values[0, 1, 4, 3].item(Float.self), 90)
    }

    func testGrowExtendsCapacityAndKeepsContents() {
        let cache = CompiledKVCache(capacity: 4)
        let (k, v) = makeKV(5, n: 2)
        _ = cache.update(keys: k, values: v)

        cache.grow(by: 4)
        XCTAssertEqual(cache.capacity, 8)
        XCTAssertEqual(cache.keysBuf!.shape, [1, 2, 8, 4])
        XCTAssertEqual(cache.keysBuf![0, 0, 1, 0].item(Float.self), 5)
        XCTAssertEqual(cache.keysBuf![0, 0, 6, 0].item(Float.self), 0)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 2) // growth does not move offset
    }
}
