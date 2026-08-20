import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

/// S2 recurrent-kind cache semantics (design of record: the continuous-batching
/// heterogeneous-cache design, §2.3, §5).
///
/// The non-negotiable S2 gate: `merge(rows).extract(i)` is BIT-IDENTICAL to `rows[i]` for both the
/// conv tail and the fp32 SSM state, across N ∈ {2,3,4}, including after a simulated decode advance;
/// `grow` is provably a no-op. Recurrent state is fixed-size, so this round-trip has no padding to
/// tolerate — equality is exact, not "close enough" (project parity identity).
final class BatchedRecurrentStateCacheTests: XCTestCase {

    // qwen3_5-shaped toy geometry: conv [1, k-1, convDim], ssm [1, Hv, Dv, Dk] fp32.
    private let convShape = [1, 3, 4]   // convKernelSize 4 → k-1 = 3
    private let ssmShape = [1, 2, 3, 2] // Hv=2, Dv=3, Dk=2

    /// Build a recurrent row whose every element is a distinct, row-identifying value so a cross-row
    /// swap or a value-corrupting merge cannot pass. Conv tail is model dtype (fp16); SSM state is fp32.
    private func makeRow(seed: Float, offset: Int) -> RecurrentScalarRowCache {
        let convCount = convShape.reduce(1, *)
        let ssmCount = ssmShape.reduce(1, *)
        let conv = MLXArray((0 ..< convCount).map { seed + Float($0) })
            .reshaped(convShape).asType(.float16)
        let ssm = MLXArray((0 ..< ssmCount).map { seed + 1000 + Float($0) })
            .reshaped(ssmShape).asType(.float32)
        let row = MambaCache()
        row[0] = conv
        row[1] = ssm
        row.offset = offset
        return row
    }

    private func assertBitIdentical(
        _ extracted: RecurrentScalarRowCache,
        _ original: RecurrentScalarRowCache,
        _ message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let ec = extracted[0], let es = extracted[1],
            let oc = original[0], let os = original[1]
        else {
            return XCTFail("row missing conv/ssm state: \(message)", file: file, line: line)
        }
        MLX.eval(ec, es, oc, os)
        XCTAssertEqual(ec.shape, oc.shape, "conv shape: \(message)", file: file, line: line)
        XCTAssertEqual(es.shape, os.shape, "ssm shape: \(message)", file: file, line: line)
        XCTAssertEqual(ec.dtype, oc.dtype, "conv dtype: \(message)", file: file, line: line)
        XCTAssertEqual(es.dtype, os.dtype, "ssm dtype: \(message)", file: file, line: line)
        XCTAssertTrue(
            MLX.all(ec .== oc).item(Bool.self), "conv values: \(message)", file: file, line: line)
        XCTAssertTrue(
            MLX.all(es .== os).item(Bool.self), "ssm values: \(message)", file: file, line: line)
        XCTAssertEqual(
            extracted.continuousLogicalOffset, original.continuousLogicalOffset,
            "logical offset: \(message)", file: file, line: line)
    }

    func testMergeExtractRoundTripBitIdenticalAcrossCohortSizes() throws {
        for n in [2, 3, 4] {
            let rows = (0 ..< n).map { makeRow(seed: Float($0) * 100, offset: $0 + 1) }
            let batch = try BatchedRecurrentStateCache.merging(rows)

            XCTAssertEqual(batch.batchSize, n)
            XCTAssertEqual(batch.continuousLogicalOffsets, (0 ..< n).map { $0 + 1 })

            for slot in 0 ..< n {
                let extracted = try batch.extract(slot: slot)
                assertBitIdentical(extracted, rows[slot], "N=\(n) slot=\(slot)")
            }
        }
    }

    func testRoundTripBitIdenticalAfterSimulatedAdvance() throws {
        // Rows as they would look after several decode ticks: fresh state values, advanced offsets.
        var rows = (0 ..< 3).map { makeRow(seed: Float($0) * 100, offset: $0 + 1) }
        let batch0 = try BatchedRecurrentStateCache.merging(rows)
        for slot in 0 ..< 3 {
            assertBitIdentical(try batch0.extract(slot: slot), rows[slot], "pre-advance slot=\(slot)")
        }

        // Simulate one lockstep decode tick: each row's recurrent state is overwritten in place and its
        // offset advances by one. Re-merge/extract must still round-trip exactly.
        rows = (0 ..< 3).map { makeRow(seed: Float($0) * 100 + 7, offset: $0 + 2) }
        let batch1 = try BatchedRecurrentStateCache.merging(rows)
        XCTAssertEqual(batch1.continuousLogicalOffsets, [2, 3, 4])
        for slot in 0 ..< 3 {
            assertBitIdentical(try batch1.extract(slot: slot), rows[slot], "post-advance slot=\(slot)")
        }
    }

    func testGrowIsProvablyANoOp() throws {
        let rows = (0 ..< 3).map { makeRow(seed: Float($0) * 100, offset: $0 + 1) }
        let batch = try BatchedRecurrentStateCache.merging(rows)

        let convBefore = batch.convState
        let ssmBefore = batch.ssmState
        let offsetsBefore = batch.continuousLogicalOffsets
        MLX.eval(convBefore, ssmBefore)

        batch.grow(by: 128)

        XCTAssertEqual(batch.continuousLogicalOffsets, offsetsBefore)
        MLX.eval(batch.convState, batch.ssmState)
        XCTAssertEqual(batch.convState.shape, convBefore.shape)
        XCTAssertEqual(batch.ssmState.shape, ssmBefore.shape)
        XCTAssertTrue(MLX.all(batch.convState .== convBefore).item(Bool.self))
        XCTAssertTrue(MLX.all(batch.ssmState .== ssmBefore).item(Bool.self))
    }

    // MARK: - S3 step 1: model-visible cache identity (compose-on-inner-MambaCache)

    /// The S3 silent-stateless trap guard: the model consumes recurrent state via a CONCRETE
    /// `cache as? MambaCache` downcast (Qwen35.swift:487,537) and writes updated state back through the
    /// SAME object each decode tick (:254,:289). So the object the runtime hands the model MUST literally
    /// be a `MambaCache`, and a model-side write-back MUST mutate this wrapper's authoritative state — not
    /// a detached copy. This pins both: `modelCache` is a `MambaCache`, its identity is stable, and a
    /// write through it is observable on `convState`/`ssmState`.
    func testModelCacheIsTheAuthoritativeInnerMambaCache() throws {
        let rows = (0 ..< 3).map { makeRow(seed: Float($0) * 100, offset: $0 + 1) }
        let batch = try BatchedRecurrentStateCache.merging(rows)

        // The model-visible cache is a MambaCache (so `cache as? MambaCache` succeeds), and its identity
        // is stable across accesses (no per-call copy that would strand the model's write-backs).
        let modelCache = try XCTUnwrap(batch.modelCache as? MambaCache, "model cache must BE a MambaCache")
        XCTAssertTrue(
            modelCache === (batch.modelCache as? MambaCache), "model cache identity must be stable")

        // It exposes the SAME batched arrays the wrapper reports (cohort on dim 0).
        MLX.eval(batch.convState, batch.ssmState)
        XCTAssertEqual(modelCache[0]?.shape, batch.convState.shape)
        XCTAssertEqual(modelCache[1]?.shape, batch.ssmState.shape)

        // A model-side write-back (as at Qwen35.swift:254/:289) mutates the wrapper's authoritative state.
        let newConv = MLXArray((0 ..< convShape.reduce(1, *) * 3).map { Float($0) + 42 })
            .reshaped([3] + Array(convShape.dropFirst())).asType(.float16)
        modelCache[0] = newConv
        MLX.eval(batch.convState, newConv)
        XCTAssertTrue(
            MLX.all(batch.convState .== newConv).item(Bool.self),
            "write through modelCache must be visible on the wrapper's convState (single authoritative object)")

        // And the mutated state still extracts row-wise bit-identically to the written slabs.
        let extracted0 = try batch.extract(slot: 0)
        MLX.eval(extracted0[0]!)
        XCTAssertTrue(MLX.all(extracted0[0]! .== newConv[0 ..< 1]).item(Bool.self))
    }

    func testMergeRejectsEmptyBatch() {
        XCTAssertThrowsError(try BatchedRecurrentStateCache.merging([])) { error in
            XCTAssertEqual(error as? BatchedRecurrentStateCacheError, .emptyBatch)
        }
    }

    func testMergeRejectsLengthCountMismatch() {
        let rows = [makeRow(seed: 0, offset: 1), makeRow(seed: 100, offset: 2)]
        XCTAssertThrowsError(try BatchedRecurrentStateCache.merging(rows, lengths: [1])) { error in
            XCTAssertEqual(
                error as? BatchedRecurrentStateCacheError, .lengthCount(expected: 2, actual: 1))
        }
    }

    func testMergeRejectsLengthMismatchAgainstAuthoritativeOffset() {
        let rows = [makeRow(seed: 0, offset: 1), makeRow(seed: 100, offset: 2)]
        XCTAssertThrowsError(try BatchedRecurrentStateCache.merging(rows, lengths: [1, 5])) { error in
            XCTAssertEqual(
                error as? BatchedRecurrentStateCacheError,
                .lengthMismatch(slot: 1, expected: 2, actual: 5))
        }
    }

    func testMergeRejectsShapeMismatch() {
        let ok = makeRow(seed: 0, offset: 1)
        let badRow = MambaCache()
        badRow[0] = MLXArray((0 ..< 6).map(Float.init)).reshaped([1, 2, 3]).asType(.float16)  // wrong conv suffix
        badRow[1] = MLXArray((0 ..< 12).map(Float.init)).reshaped(ssmShape).asType(.float32)
        badRow.offset = 1
        XCTAssertThrowsError(try BatchedRecurrentStateCache.merging([ok, badRow])) { error in
            guard case .incompatibleShape(let slot, _, _) = error as? BatchedRecurrentStateCacheError
            else { return XCTFail("expected incompatibleShape, got \(error)") }
            XCTAssertEqual(slot, 1)
        }
    }

    func testMergeRejectsDTypeMismatch() {
        let ok = makeRow(seed: 0, offset: 1)
        let badRow = MambaCache()
        badRow[0] = MLXArray((0 ..< convShape.reduce(1, *)).map(Float.init))
            .reshaped(convShape).asType(.float32)  // conv should be fp16 here
        badRow[1] = MLXArray((0 ..< ssmShape.reduce(1, *)).map(Float.init))
            .reshaped(ssmShape).asType(.float32)
        badRow.offset = 1
        XCTAssertThrowsError(try BatchedRecurrentStateCache.merging([ok, badRow])) { error in
            XCTAssertEqual(
                error as? BatchedRecurrentStateCacheError, .incompatibleDType(slot: 1))
        }
    }

    func testExtractRejectsOutOfRangeSlot() throws {
        let batch = try BatchedRecurrentStateCache.merging(
            [makeRow(seed: 0, offset: 1), makeRow(seed: 100, offset: 2)])
        XCTAssertThrowsError(try batch.extract(slot: 2)) { error in
            XCTAssertEqual(
                error as? BatchedRecurrentStateCacheError, .invalidSlot(index: 2, batchSize: 2))
        }
        XCTAssertThrowsError(try batch.extract(slot: -1)) { error in
            XCTAssertEqual(
                error as? BatchedRecurrentStateCacheError, .invalidSlot(index: -1, batchSize: 2))
        }
    }
}
