import MLX
import XCTest

@testable import SpikeCore

/// Semantics of the compile-legal KV cache (runs on Metal-capable hosts; these are
/// the invariants the compiled decode step depends on).
final class CompiledKVCacheTests: XCTestCase {

    private func makeKV(
        _ value: Float, n: Int = 1, dtype: DType = .float32
    ) -> (MLXArray, MLXArray) {
        // [B=1, kvHeads=2, n, headDim=4]
        let k = MLXArray.full([1, 2, n, 4], values: MLXArray(value)).asType(dtype)
        let v = MLXArray.full([1, 2, n, 4], values: MLXArray(value * 10)).asType(dtype)
        return (k, v)
    }

    private func bytes(_ array: MLXArray) -> Data {
        array.asData(access: .copy).data
    }

    private func assertSnapshotError(
        _ expected: CompiledKVCacheSnapshotError,
        _ body: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) {
            XCTAssertEqual(
                $0 as? CompiledKVCacheSnapshotError, expected,
                file: file, line: line)
        }
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

    func testSnapshotCapturesDetachedLogicalPrefixAndExactMetadata() throws {
        let cache = CompiledKVCache(capacity: 8)
        let (keys, values) = makeKV(3, n: 3, dtype: .float16)
        _ = cache.update(keys: keys, values: values)

        let snapshot = try cache.captureSnapshot(logicalTokenCount: 3)

        XCTAssertEqual(snapshot.rank, 4)
        XCTAssertEqual(snapshot.batchSize, 1)
        XCTAssertEqual(snapshot.kvHeadCount, 2)
        XCTAssertEqual(snapshot.tokenCount, 3)
        XCTAssertEqual(snapshot.headDimension, 4)
        XCTAssertEqual(snapshot.keyDType, .float16)
        XCTAssertEqual(snapshot.valueDType, .float16)
        XCTAssertEqual(snapshot.keyNBytes, 48)
        XCTAssertEqual(snapshot.valueNBytes, 48)
        XCTAssertEqual(snapshot.totalNBytes, 96)
        XCTAssertEqual(snapshot.keys?.shape, [1, 2, 3, 4])
        XCTAssertEqual(snapshot.values?.shape, [1, 2, 3, 4])

        let snapshotKeys = try XCTUnwrap(snapshot.keys)
        let snapshotValues = try XCTUnwrap(snapshot.values)
        let capturedKeyBytes = bytes(snapshotKeys)
        let capturedValueBytes = bytes(snapshotValues)

        cache.resetInPlace()
        let (replacementKeys, replacementValues) = makeKV(9, n: 3, dtype: .float16)
        _ = cache.update(keys: replacementKeys, values: replacementValues)

        XCTAssertEqual(bytes(snapshotKeys), capturedKeyBytes)
        XCTAssertEqual(bytes(snapshotValues), capturedValueBytes)
        XCTAssertEqual(snapshotKeys[0, 0, 2, 0].item(Float.self), 3)
        XCTAssertEqual(snapshotValues[0, 1, 2, 3].item(Float.self), 30)
    }

    func testRestorePreservesIdentitiesUpdatesBothOffsetsAndZerosTail() throws {
        let source = CompiledKVCache(capacity: 8)
        let (sourceKeys, sourceValues) = makeKV(2, n: 3, dtype: .float16)
        _ = source.update(keys: sourceKeys, values: sourceValues)
        let snapshot = try source.captureSnapshot(logicalTokenCount: 3)

        let target = CompiledKVCache(capacity: 8)
        let (staleKeys, staleValues) = makeKV(7, n: 6, dtype: .float16)
        _ = target.update(keys: staleKeys, values: staleValues)
        let keysObject = try XCTUnwrap(target.keysBuf)
        let valuesObject = try XCTUnwrap(target.valuesBuf)
        let offsetObject = target.offsetArr

        try target.restoreInPlace(from: snapshot)

        XCTAssertTrue(target.keysBuf! === keysObject)
        XCTAssertTrue(target.valuesBuf! === valuesObject)
        XCTAssertTrue(target.offsetArr === offsetObject)
        XCTAssertEqual(target.offsetArr.item(Int32.self), 3)
        XCTAssertEqual(target.offset, 3)
        XCTAssertEqual(target.keysBuf![0, 0, 2, 0].item(Float.self), 2)
        XCTAssertEqual(target.valuesBuf![0, 1, 2, 3].item(Float.self), 20)
        for token in 3 ..< target.capacity {
            XCTAssertEqual(
                target.keysBuf![0, 0, token, 0].item(Float.self), 0,
                "key tail token \(token)")
            XCTAssertEqual(
                target.valuesBuf![0, 1, token, 3].item(Float.self), 0,
                "value tail token \(token)")
        }
    }

    func testRestoreThenAppendMatchesUninterruptedControl() throws {
        let control = CompiledKVCache(capacity: 8)
        let restored = CompiledKVCache(capacity: 8)
        let (prefixKeys, prefixValues) = makeKV(1, n: 2, dtype: .float16)
        _ = control.update(keys: prefixKeys, values: prefixValues)
        _ = restored.update(keys: prefixKeys, values: prefixValues)
        let snapshot = try restored.captureSnapshot(logicalTokenCount: 2)

        let (tailKeys, tailValues) = makeKV(6, n: 3, dtype: .float16)
        _ = control.update(keys: tailKeys, values: tailValues)

        restored.resetInPlace()
        try restored.restoreInPlace(from: snapshot)
        _ = restored.update(keys: tailKeys, values: tailValues)

        XCTAssertEqual(restored.offsetArr.item(Int32.self), 5)
        XCTAssertEqual(restored.offset, 5)
        XCTAssertEqual(bytes(restored.keysBuf!), bytes(control.keysBuf!))
        XCTAssertEqual(bytes(restored.valuesBuf!), bytes(control.valuesBuf!))
    }

    func testPreparedGrowthDoesNotMutateLiveCapacityUntilAtomicApply() throws {
        let source = CompiledKVCache(capacity: 4)
        let (sourceKeys, sourceValues) = makeKV(2, n: 2, dtype: .float16)
        _ = source.update(keys: sourceKeys, values: sourceValues)
        let snapshot = try source.captureSnapshot(logicalTokenCount: 2)

        let target = CompiledKVCache(capacity: 4)
        let (targetKeys, targetValues) = makeKV(7, n: 2, dtype: .float16)
        _ = target.update(keys: targetKeys, values: targetValues)
        let originalKeys = try XCTUnwrap(target.keysBuf)
        let originalValues = try XCTUnwrap(target.valuesBuf)
        let originalKeyBytes = bytes(originalKeys)
        let originalValueBytes = bytes(originalValues)

        let plan = try target.prepareRestore(
            from: snapshot, targetCapacity: 260)

        XCTAssertEqual(target.capacity, 4)
        XCTAssertTrue(target.keysBuf === originalKeys)
        XCTAssertTrue(target.valuesBuf === originalValues)
        XCTAssertEqual(bytes(target.keysBuf!), originalKeyBytes)
        XCTAssertEqual(bytes(target.valuesBuf!), originalValueBytes)
        XCTAssertEqual(target.offsetArr.item(Int32.self), 2)

        target.applyPreparedRestore(plan)

        XCTAssertEqual(target.capacity, 260)
        XCTAssertEqual(target.keysBuf?.shape, [1, 2, 260, 4])
        XCTAssertEqual(target.valuesBuf?.shape, [1, 2, 260, 4])
        XCTAssertEqual(target.offsetArr.item(Int32.self), 2)
        XCTAssertEqual(target.offset, 2)
        XCTAssertEqual(target.keysBuf![0, 0, 1, 0].item(Float.self), 2)
        XCTAssertEqual(target.keysBuf![0, 0, 259, 0].item(Float.self), 0)
    }

    func testCaptureRejectsUninitializedInvalidLengthOffsetMismatchAndNonFinite() {
        let uninitialized = CompiledKVCache(capacity: 8)
        assertSnapshotError(.uninitializedCache) {
            _ = try uninitialized.captureSnapshot(logicalTokenCount: 1)
        }

        let cache = CompiledKVCache(capacity: 8)
        let (keys, values) = makeKV(1, n: 2)
        _ = cache.update(keys: keys, values: values)
        assertSnapshotError(.invalidTokenCount) {
            _ = try cache.captureSnapshot(logicalTokenCount: 0)
        }
        assertSnapshotError(.invalidTokenCount) {
            _ = try cache.captureSnapshot(logicalTokenCount: 9)
        }
        assertSnapshotError(.offsetMismatch) {
            _ = try cache.captureSnapshot(logicalTokenCount: 1)
        }
        assertSnapshotError(.unsupportedDType) {
            _ = try cache.captureSnapshot(logicalTokenCount: 2)
        }

        let nonFinite = CompiledKVCache(capacity: 4)
        let badKeys = MLXArray.full(
            [1, 2, 1, 4], values: MLXArray(Float16.nan), dtype: .float16)
        let goodValues = MLXArray.zeros([1, 2, 1, 4], dtype: .float16)
        _ = nonFinite.update(keys: badKeys, values: goodValues)
        assertSnapshotError(.nonFiniteValues) {
            _ = try nonFinite.captureSnapshot(logicalTokenCount: 1)
        }

        let invalidControl = CompiledKVCache(capacity: 4)
        let (validKeys, validValues) = makeKV(1, dtype: .float16)
        _ = invalidControl.update(keys: validKeys, values: validValues)
        invalidControl.offsetArr = MLXArray([Float32(1)])
        assertSnapshotError(.invalidControlState) {
            _ = try invalidControl.captureSnapshot(logicalTokenCount: 1)
        }
    }

    func testRestoreRejectsCapacityDTypeShapeBytesPartialAndNonFiniteWithoutMutation() throws {
        let source = CompiledKVCache(capacity: 8)
        let (keys, values) = makeKV(4, n: 3, dtype: .float16)
        _ = source.update(keys: keys, values: values)
        let valid = try source.captureSnapshot(logicalTokenCount: 3)

        let tooSmall = CompiledKVCache(capacity: 2)
        let (smallKeys, smallValues) = makeKV(8, n: 1, dtype: .float16)
        _ = tooSmall.update(keys: smallKeys, values: smallValues)
        assertSnapshotError(.insufficientCapacity) {
            try tooSmall.restoreInPlace(from: valid)
        }

        let wrongDType = CompiledKVCache(capacity: 8)
        let (floatKeys, floatValues) = makeKV(8, n: 1, dtype: .float32)
        _ = wrongDType.update(keys: floatKeys, values: floatValues)
        assertSnapshotError(.dtypeMismatch) {
            try wrongDType.restoreInPlace(from: valid)
        }

        let target = CompiledKVCache(capacity: 8)
        let (targetKeys, targetValues) = makeKV(8, n: 2, dtype: .float16)
        _ = target.update(keys: targetKeys, values: targetValues)
        let originalKeyBytes = bytes(target.keysBuf!)
        let originalValueBytes = bytes(target.valuesBuf!)
        let originalOffset = target.offsetArr.item(Int32.self)

        let malformedShape = CompiledKVCacheSnapshot(
            rank: 4,
            batchSize: 1,
            kvHeadCount: 2,
            tokenCount: 3,
            headDimension: 4,
            keyDType: .float16,
            valueDType: .float16,
            keyNBytes: valid.keyNBytes,
            valueNBytes: valid.valueNBytes,
            keys: MLXArray.zeros([1, 1, 3, 4], dtype: .float16),
            values: valid.values)
        assertSnapshotError(.shapeMismatch) {
            try target.restoreInPlace(from: malformedShape)
        }

        let malformedBytes = CompiledKVCacheSnapshot(
            rank: valid.rank,
            batchSize: valid.batchSize,
            kvHeadCount: valid.kvHeadCount,
            tokenCount: valid.tokenCount,
            headDimension: valid.headDimension,
            keyDType: valid.keyDType,
            valueDType: valid.valueDType,
            keyNBytes: valid.keyNBytes + 1,
            valueNBytes: valid.valueNBytes,
            keys: valid.keys,
            values: valid.values)
        assertSnapshotError(.byteCountMismatch) {
            try target.restoreInPlace(from: malformedBytes)
        }

        let partial = CompiledKVCacheSnapshot(
            rank: valid.rank,
            batchSize: valid.batchSize,
            kvHeadCount: valid.kvHeadCount,
            tokenCount: valid.tokenCount,
            headDimension: valid.headDimension,
            keyDType: valid.keyDType,
            valueDType: valid.valueDType,
            keyNBytes: valid.keyNBytes,
            valueNBytes: valid.valueNBytes,
            keys: valid.keys,
            values: nil)
        assertSnapshotError(.partialSnapshot) {
            try target.restoreInPlace(from: partial)
        }

        let nonFiniteValues = MLXArray.full(
            [1, 2, 3, 4], values: MLXArray(Float16.nan), dtype: .float16)
        let nonFinite = CompiledKVCacheSnapshot(
            rank: valid.rank,
            batchSize: valid.batchSize,
            kvHeadCount: valid.kvHeadCount,
            tokenCount: valid.tokenCount,
            headDimension: valid.headDimension,
            keyDType: valid.keyDType,
            valueDType: valid.valueDType,
            keyNBytes: valid.keyNBytes,
            valueNBytes: valid.valueNBytes,
            keys: valid.keys,
            values: nonFiniteValues)
        assertSnapshotError(.nonFiniteValues) {
            try target.restoreInPlace(from: nonFinite)
        }

        XCTAssertEqual(bytes(target.keysBuf!), originalKeyBytes)
        XCTAssertEqual(bytes(target.valuesBuf!), originalValueBytes)
        XCTAssertEqual(target.offsetArr.item(Int32.self), originalOffset)
        XCTAssertEqual(target.offset, Int(originalOffset))
    }
}
