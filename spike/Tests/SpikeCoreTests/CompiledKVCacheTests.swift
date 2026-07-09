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
