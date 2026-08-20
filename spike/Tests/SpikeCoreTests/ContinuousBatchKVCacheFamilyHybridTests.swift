import HarnessCore
import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// S2 `.hybridFP16` family-case dispatch (design of record: the continuous-batching
/// heterogeneous-cache design, §2.3). The
/// heterogeneous cache family builds a dense `CompiledKVCache` at attention layers and a recurrent
/// `MambaCache` at linear layers, and merges each layer BY ITS KIND — never a cross-kind merge. The
/// existing fp16/affine dispatch stays byte-stable; hybrid rides the kind-agnostic row seam.
final class ContinuousBatchKVCacheFamilyHybridTests: XCTestCase {

    // Asymmetric toy geometry: recurrent at layer 0 (the landmine index), dense at layer 1.
    private func makeGeometry() -> HybridCacheGeometry {
        let map = HybridLayerKindMap(kinds: [.recurrentState, .denseAttention])
        let dense = DenseKVGeometry(kvHeads: 2, headDim: 4, elementBytes: 2)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: 4, convDim: 4, valueHeads: 2, valueHeadDim: 3, keyHeadDim: 2,
            convElementBytes: 2)
        return HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent)!
    }

    private func makeDenseScalar(_ values: [Float], capacity: Int = 8) -> CompiledKVCache {
        let cache = CompiledKVCache(capacity: capacity)
        let key = MLXArray(values).reshaped([1, 1, values.count, 1])
        _ = cache.update(keys: key, values: key + 100)
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

    func testHybridInitAndAccessors() {
        let geometry = makeGeometry()
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: geometry)
        XCTAssertEqual(family.hybridGeometry, geometry)
        XCTAssertNil(family.affineConfigurations)
    }

    func testMakeScalarRowsBuildsRecurrentAndDensePerLayer() {
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: makeGeometry())
        let rows = family.makeScalarRows(layerCount: 2, capacity: 8)
        XCTAssertEqual(rows.count, 2)
        XCTAssertTrue(rows[0] is MambaCache, "layer 0 (recurrent) must be a MambaCache")
        XCTAssertTrue(rows[1] is CompiledKVCache, "layer 1 (dense) must be a CompiledKVCache")
    }

    func testMergeRowDenseLayerYieldsDenseBatchedRow() throws {
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: makeGeometry())
        let batched = try family.mergeRow(
            layer: 1,
            rows: [makeDenseScalar([10, 11, 12]), makeDenseScalar([20])],
            lengths: [3, 1])
        XCTAssertEqual(batched.continuousLogicalOffsets, [3, 1])
        XCTAssertTrue(try batched.extractContinuousRow(slot: 0) is CompiledKVCache)
    }

    func testMergeRowRecurrentLayerYieldsRecurrentBatchedRow() throws {
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: makeGeometry())
        let batched = try family.mergeRow(
            layer: 0,
            rows: [makeRecurrentScalar(seed: 0, offset: 5), makeRecurrentScalar(seed: 100, offset: 7)],
            lengths: [5, 7])
        XCTAssertEqual(batched.continuousLogicalOffsets, [5, 7])
        let row = try batched.extractContinuousRow(slot: 1)
        XCTAssertTrue(row is MambaCache)
        XCTAssertEqual(row.continuousLogicalOffset, 7)
    }

    func testMergeRowRejectsWrongKindAtDenseLayer() {
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: makeGeometry())
        // Recurrent rows offered at the dense layer 1 → fail closed, no silent cross-kind merge.
        XCTAssertThrowsError(
            try family.mergeRow(
                layer: 1,
                rows: [makeRecurrentScalar(seed: 0, offset: 1), makeRecurrentScalar(seed: 1, offset: 1)],
                lengths: [1, 1])
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchKVCacheFamilyError, .incompatibleScalarCache(layer: 1))
        }
    }

    func testMergeRowRejectsWrongKindAtRecurrentLayer() {
        let family = ContinuousBatchKVCacheFamily(hybridGeometry: makeGeometry())
        // Dense rows offered at the recurrent layer 0 → fail closed.
        XCTAssertThrowsError(
            try family.mergeRow(
                layer: 0,
                rows: [makeDenseScalar([1]), makeDenseScalar([2])],
                lengths: [1, 1])
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchKVCacheFamilyError, .incompatibleScalarCache(layer: 0))
        }
    }

    func testFp16ArmUnchangedThroughRowSeam() throws {
        let family = ContinuousBatchKVCacheFamily.fp16
        let rows = family.makeScalarRows(layerCount: 3, capacity: 8)
        XCTAssertEqual(rows.count, 3)
        XCTAssertTrue(rows.allSatisfy { $0 is CompiledKVCache }, "fp16 rows are all dense")

        let batched = try family.mergeRow(
            layer: 0,
            rows: [makeDenseScalar([10, 11]), makeDenseScalar([20])],
            lengths: [2, 1])
        XCTAssertEqual(batched.continuousLogicalOffsets, [2, 1])
        XCTAssertTrue(batched is BatchedCompiledKVCache)
    }
}
