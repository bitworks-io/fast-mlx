import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// S2→S3 batched-side super-protocol seam. Mirrors the scalar-side `ContinuousScalarRowCache` split
/// (c69e866): a kind-agnostic `ContinuousBatchedRowCache` seam that BOTH the dense growing-KV batched
/// cache and the recurrent fixed-state batched cache join, so the hybrid runtime can hold one
/// `[any ContinuousBatchedRowCache]` per-layer array and key dense-only capacity checks off
/// `firstDenseLayerIndex` instead of assuming layer 0 is dense.
///
/// This is a retype seam with ZERO behavior change to the dense path — the existing dense batched
/// tests remain the regression gate. These tests are the tripwire proving both kinds are reachable
/// through the seam and that `extractContinuousRow` returns the correct scalar-row kind per cache.
final class ContinuousBatchedRowCacheSeamTests: XCTestCase {

    private func makeDenseScalar(_ values: [Float], capacity: Int = 8) -> CompiledKVCache {
        let cache = CompiledKVCache(capacity: capacity)
        let key = MLXArray(values).reshaped([1, 1, values.count, 1])
        let value = key + 100
        _ = cache.update(keys: key, values: value)
        return cache
    }

    private func makeRecurrentScalar(seed: Float, offset: Int) -> RecurrentScalarRowCache {
        let conv = MLXArray((0 ..< 12).map { seed + Float($0) })
            .reshaped([1, 3, 4]).asType(.float16)
        let ssm = MLXArray((0 ..< 12).map { seed + 1000 + Float($0) })
            .reshaped([1, 2, 3, 2]).asType(.float32)
        let row = MambaCache()
        row[0] = conv
        row[1] = ssm
        row.offset = offset
        return row
    }

    func testDenseBatchedReachableThroughRowSeam() throws {
        let batched = try BatchedCompiledKVCache.merging([
            makeDenseScalar([10, 11, 12]),
            makeDenseScalar([20]),
        ])
        let seam: any ContinuousBatchedRowCache = batched

        XCTAssertEqual(seam.continuousLogicalOffsets, [3, 1])
        XCTAssertEqual(seam.continuousPhysicalWrittenEnd, 3)

        let row = try seam.extractContinuousRow(slot: 0)
        XCTAssertTrue(row is CompiledKVCache, "dense seam must yield a dense scalar row")
        XCTAssertEqual(row.continuousLogicalOffset, 3)
    }

    func testRecurrentBatchedReachableThroughRowSeam() throws {
        let rows = [
            makeRecurrentScalar(seed: 0, offset: 5),
            makeRecurrentScalar(seed: 100, offset: 7),
        ]
        let batch = try BatchedRecurrentStateCache.merging(rows)
        let seam: any ContinuousBatchedRowCache = batch

        XCTAssertEqual(seam.continuousLogicalOffsets, [5, 7])
        XCTAssertEqual(seam.continuousPhysicalWrittenEnd, 7)

        let row = try seam.extractContinuousRow(slot: 1)
        XCTAssertTrue(row is MambaCache, "recurrent seam must yield a recurrent scalar row")
        XCTAssertEqual(row.continuousLogicalOffset, 7)
    }

    func testGrowIsReachableAndNoOpForRecurrentThroughSeam() throws {
        let batch = try BatchedRecurrentStateCache.merging([
            makeRecurrentScalar(seed: 0, offset: 1),
            makeRecurrentScalar(seed: 100, offset: 2),
        ])
        let seam: any ContinuousBatchedRowCache = batch
        let offsetsBefore = seam.continuousLogicalOffsets
        seam.grow(by: 64)
        XCTAssertEqual(seam.continuousLogicalOffsets, offsetsBefore)
    }

    func testModelCacheSeamYieldsCorrectKindPerLayer() throws {
        // S3 step 2: the runtime builds the model's [any KVCache] array from one
        // [any ContinuousBatchedRowCache] per layer via the kind-aware `modelCache` accessor. A dense
        // layer's model cache is the batched KV cache itself (a KVCache, NOT a MambaCache); a recurrent
        // layer's model cache is the owned inner MambaCache the model downcasts to.
        let dense = try BatchedCompiledKVCache.merging([makeDenseScalar([1, 2])])
        let recurrent = try BatchedRecurrentStateCache.merging([makeRecurrentScalar(seed: 0, offset: 2)])
        let layers: [any ContinuousBatchedRowCache] = [recurrent, dense]

        let modelCaches: [any KVCache] = layers.map(\.modelCache)
        XCTAssertTrue(modelCaches[0] is MambaCache, "recurrent layer's model cache must BE a MambaCache")
        XCTAssertFalse(modelCaches[1] is MambaCache, "dense layer's model cache must not be a MambaCache")
        // The recurrent layer's model cache is the wrapper's authoritative inner object (stable identity).
        XCTAssertTrue((modelCaches[0] as AnyObject) === (recurrent.modelCache as AnyObject))
    }

    func testHeterogeneousArrayHoldsBothKindsThroughSeam() throws {
        // The runtime holds one [any ContinuousBatchedRowCache] per layer; prove a mixed array typechecks
        // and each element reports the correct kind on extract.
        let dense = try BatchedCompiledKVCache.merging([makeDenseScalar([1, 2])])
        let recurrent = try BatchedRecurrentStateCache.merging([makeRecurrentScalar(seed: 0, offset: 2)])
        let layers: [any ContinuousBatchedRowCache] = [recurrent, dense]

        XCTAssertTrue(try layers[0].extractContinuousRow(slot: 0) is MambaCache)
        XCTAssertTrue(try layers[1].extractContinuousRow(slot: 0) is CompiledKVCache)
    }
}
