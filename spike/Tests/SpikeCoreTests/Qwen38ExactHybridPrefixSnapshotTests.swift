import MLX
import MLXLMCommon
import XCTest

@testable import SpikeCore

final class Qwen38ExactHybridPrefixSnapshotTests: XCTestCase {
    private let prefix = [151_644, 872, 198]

    private func digest(_ byte: String) -> String {
        String(repeating: byte, count: 64)
    }

    private func identity(
        model: String = "a",
        source: String = "b",
        artifact: String = "c",
        tokenizer: String = "d",
        template: String = "e",
        layout: String = "f",
        namespace: String = "0"
    ) throws -> Qwen38ExactHybridPrefixIdentity {
        try Qwen38ExactHybridPrefixIdentity(
            modelSHA256: digest(model),
            sourceSHA256: digest(source),
            artifactSHA256: digest(artifact),
            tokenizerSHA256: digest(tokenizer),
            templateSHA256: digest(template),
            cacheLayoutSHA256: digest(layout),
            requestNamespaceSHA256: digest(namespace))
    }

    private func denseCache(seed: Float = 1, tokens: Int = 3) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let elementCount = 1 * 2 * tokens * 4
        let keys = MLXArray((0 ..< elementCount).map { seed + Float($0) })
            .reshaped([1, 2, tokens, 4])
            .asType(.float16)
        let values = MLXArray((0 ..< elementCount).map { seed + 100 + Float($0) })
            .reshaped([1, 2, tokens, 4])
            .asType(.float16)
        _ = cache.update(keys: keys, values: values)
        return cache
    }

    private func recurrentCache(seed: Float = 10, offset: Int = 0) -> MambaCache {
        let cache = MambaCache()
        cache[0] = MLXArray((0 ..< 12).map { seed + Float($0) })
            .reshaped([1, 3, 4])
            .asType(.float16)
        cache[1] = MLXArray((0 ..< 12).map { seed + 1000 + Float($0) })
            .reshaped([1, 2, 3, 2])
            .asType(.float32)
        cache.offset = offset
        return cache
    }

    private func hybridCaches() -> [any KVCache] {
        [recurrentCache(), denseCache()]
    }

    func testCaptureBindsIdentityPrefixCacheFingerprintsAndDigest() throws {
        let semanticIdentity = try identity()
        let snapshot = try Qwen38ExactHybridPrefixSnapshot.capture(
            identity: semanticIdentity,
            prefixTokenIDs: prefix,
            caches: hybridCaches())

        XCTAssertEqual(snapshot.identity, semanticIdentity)
        XCTAssertEqual(snapshot.prefixTokenIDs, prefix)
        XCTAssertEqual(snapshot.prefixTokenCount, prefix.count)
        XCTAssertEqual(snapshot.cacheFingerprints.map(\.dynamicTypeName), [
            "MLXLMCommon.MambaCache",
            "MLXLMCommon.KVCacheSimple",
        ])
        XCTAssertEqual(snapshot.cacheFingerprints[0].offset, 0)
        XCTAssertEqual(snapshot.cacheFingerprints[0].metaState, ["2", "0,1"])
        XCTAssertEqual(snapshot.cacheFingerprints[0].stateArrays.map(\.shape), [
            [1, 3, 4],
            [1, 2, 3, 2],
        ])
        XCTAssertEqual(snapshot.cacheFingerprints[1].offset, 3)
        XCTAssertEqual(snapshot.cacheFingerprints[1].stateArrays.map(\.shape), [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        XCTAssertEqual(snapshot.cacheFingerprints[1].stateArrays.map(\.dtypeName), [
            "float16",
            "float16",
        ])
        XCTAssertEqual(snapshot.canonicalDigest.count, 64)
    }

    func testRestoreVerifiesExpectedIdentityPrefixAndDigestThenReturnsIndependentCopies() throws {
        let semanticIdentity = try identity()
        let source = hybridCaches()
        let snapshot = try Qwen38ExactHybridPrefixSnapshot.capture(
            identity: semanticIdentity,
            prefixTokenIDs: prefix,
            caches: source)
        let sealedDigest = snapshot.canonicalDigest

        let sourceRecurrent = try XCTUnwrap(source[0] as? MambaCache)
        sourceRecurrent[0] = MLXArray.zeros([1, 3, 4], dtype: .float16)
        sourceRecurrent.offset = 1
        let sourceDense = try XCTUnwrap(source[1] as? KVCacheSimple)
        let extraKeys = MLXArray.full([1, 2, 1, 4], values: MLXArray(Float(99))).asType(.float16)
        let extraValues = MLXArray.full([1, 2, 1, 4], values: MLXArray(Float(199))).asType(.float16)
        _ = sourceDense.update(keys: extraKeys, values: extraValues)

        let firstRestore = try snapshot.restore(
            expectedIdentity: semanticIdentity,
            expectedPrefixTokenIDs: prefix,
            expectedDigest: sealedDigest)
        XCTAssertEqual(
            try Qwen38ExactHybridPrefixSnapshot.canonicalDigest(
                identity: semanticIdentity,
                prefixTokenIDs: prefix,
                caches: firstRestore),
            sealedDigest)

        let restoredRecurrent = try XCTUnwrap(firstRestore[0] as? MambaCache)
        restoredRecurrent[1] = MLXArray.zeros([1, 2, 3, 2], dtype: .float32)
        restoredRecurrent.offset = 0

        let secondRestore = try snapshot.restore(
            expectedIdentity: semanticIdentity,
            expectedPrefixTokenIDs: prefix,
            expectedDigest: sealedDigest)
        XCTAssertEqual(
            try Qwen38ExactHybridPrefixSnapshot.canonicalDigest(
                identity: semanticIdentity,
                prefixTokenIDs: prefix,
                caches: secondRestore),
            sealedDigest)
        XCTAssertNotEqual(
            try Qwen38ExactHybridPrefixSnapshot.canonicalDigest(
                identity: semanticIdentity,
                prefixTokenIDs: prefix,
                caches: firstRestore),
            sealedDigest)
    }

    func testRestoreRejectsWrongIdentityPrefixAndDigest() throws {
        let semanticIdentity = try identity()
        let snapshot = try Qwen38ExactHybridPrefixSnapshot.capture(
            identity: semanticIdentity,
            prefixTokenIDs: prefix,
            caches: hybridCaches())

        XCTAssertThrowsError(
            try snapshot.restore(
                expectedIdentity: try identity(namespace: "1"),
                expectedPrefixTokenIDs: prefix,
                expectedDigest: snapshot.canonicalDigest)
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .identityMismatch)
        }

        XCTAssertThrowsError(
            try snapshot.restore(
                expectedIdentity: semanticIdentity,
                expectedPrefixTokenIDs: [151_644, 872],
                expectedDigest: snapshot.canonicalDigest)
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .prefixMismatch)
        }

        XCTAssertThrowsError(
            try snapshot.restore(
                expectedIdentity: semanticIdentity,
                expectedPrefixTokenIDs: prefix,
                expectedDigest: digest("9"))
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .digestMismatch)
        }
    }

    func testCaptureRejectsEmptyPartialUnsupportedAndPreparedMetadata() throws {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [])
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .emptyCacheList)
        }

        let partial = MambaCache()
        partial[0] = MLXArray.zeros([1, 3, 4], dtype: .float16)
        partial.offset = 0
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [partial])
        ) { error in
            guard case .partialCacheState(let index, let typeName, _) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected partialCacheState, got \(error)")
            }
            XCTAssertEqual(index, 0)
            XCTAssertEqual(typeName, "MLXLMCommon.MambaCache")
        }

        let prepared = recurrentCache()
        prepared.prepare(lengths: [3])
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [prepared])
        ) { error in
            guard case .invalidCacheMetadata(let index, let typeName, _) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected invalidCacheMetadata, got \(error)")
            }
            XCTAssertEqual(index, 0)
            XCTAssertEqual(typeName, "MLXLMCommon.MambaCache")
        }

        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [UnsupportedCopyableCache()])
        ) { error in
            guard case .unsupportedCacheType(let index, let typeName) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected unsupportedCacheType, got \(error)")
            }
            XCTAssertEqual(index, 0)
            XCTAssertTrue(
                typeName.hasSuffix(".UnsupportedCopyableCache"),
                "unexpected unsupported cache type: \(typeName)")
        }
    }

    func testCaptureRejectsMambaOffsetBatchAndRankMismatches() throws {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [recurrentCache(offset: 3), denseCache()])
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .invalidCacheOffset(
                    index: 0,
                    typeName: "MLXLMCommon.MambaCache",
                    offset: 3))
        }

        let batched = MambaCache()
        batched[0] = MLXArray.zeros([2, 3, 4], dtype: .float16)
        batched[1] = MLXArray.zeros([2, 2, 3, 2], dtype: .float32)
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [batched, denseCache()])
        ) { error in
            guard case .partialCacheState(let index, let typeName, let reason) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected partialCacheState, got \(error)")
            }
            XCTAssertEqual(index, 0)
            XCTAssertEqual(typeName, "MLXLMCommon.MambaCache")
            XCTAssertEqual(reason, "MambaCache scalar snapshot requires batch size 1")
        }

        let badConvRank = MambaCache()
        badConvRank[0] = MLXArray.zeros([1, 1, 3, 4], dtype: .float16)
        badConvRank[1] = MLXArray.zeros([1, 2, 3, 2], dtype: .float32)
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [badConvRank, denseCache()])
        ) { error in
            guard case .partialCacheState(_, _, let reason) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected partialCacheState, got \(error)")
            }
            XCTAssertEqual(reason, "MambaCache conv state must be rank 3")
        }

        let badRecurrentRank = MambaCache()
        badRecurrentRank[0] = MLXArray.zeros([1, 3, 4], dtype: .float16)
        badRecurrentRank[1] = MLXArray.zeros([1, 6, 4], dtype: .float32)
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [badRecurrentRank, denseCache()])
        ) { error in
            guard case .partialCacheState(_, _, let reason) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected partialCacheState, got \(error)")
            }
            XCTAssertEqual(reason, "MambaCache recurrent state must be rank 4")
        }
    }

    func testCaptureRejectsMambaUnsupportedDType() throws {
        let badDType = MambaCache()
        badDType[0] = MLXArray(Int32(0) ..< Int32(12))
            .reshaped([1, 3, 4])
        badDType[1] = MLXArray.zeros([1, 2, 3, 2], dtype: .float32)

        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [badDType, denseCache()])
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .invalidStateArray(
                    index: 0,
                    typeName: "MLXLMCommon.MambaCache",
                    arrayIndex: 0,
                    reason: "MambaCache state arrays require supported floating dtypes"))
        }
    }

    func testSnapshotPolicyRejectsActiveMambaSpeculativeCheckpoint() {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.validateResolvedSpeculativeCheckpoint(
                true,
                index: 4,
                typeName: "MLXLMCommon.MambaCache")
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .invalidCacheMetadata(
                    index: 4,
                    typeName: "MLXLMCommon.MambaCache",
                    reason: "MambaCache speculative checkpoint must be resolved before snapshot"))
        }
    }

    func testCaptureRejectsMalformedDensePrefixWitnesses() throws {
        let batched = KVCacheSimple()
        batched.state = [
            MLXArray.zeros([2, 2, 3, 4], dtype: .float16),
            MLXArray.zeros([2, 2, 3, 4], dtype: .float16),
        ]
        assertDenseCaptureRejected(
            batched,
            expectedReason: "KVCacheSimple scalar snapshot requires batch size 1")

        let wrongRank = KVCacheSimple()
        wrongRank.state = [
            MLXArray.zeros([1, 2, 3], dtype: .float16),
            MLXArray.zeros([1, 2, 3], dtype: .float16),
        ]
        assertDenseCaptureRejected(
            wrongRank,
            expectedReason: "KVCacheSimple state arrays must be rank 4")

        let integerState = KVCacheSimple()
        integerState.state = [
            MLXArray.zeros([1, 2, 3, 4], dtype: .int32),
            MLXArray.zeros([1, 2, 3, 4], dtype: .int32),
        ]
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [recurrentCache(), integerState])
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .invalidStateArray(
                    index: 1,
                    typeName: "MLXLMCommon.KVCacheSimple",
                    arrayIndex: 0,
                    reason: "KVCacheSimple state arrays require supported floating dtypes"))
        }

        let shortPrefix = denseCache(tokens: 2)
        assertDenseCaptureRejected(
            shortPrefix,
            expectedReason: "KVCacheSimple state must cover the complete prefix")
    }

    func testCaptureRequiresDensePrefixLengthWitness() throws {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [recurrentCache()])
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .missingDensePrefixWitness)
        }
    }

    func testCaptureRejectsBadIdentityAndPrefixTokens() {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixIdentity(
                modelSHA256: "not-a-digest",
                sourceSHA256: digest("b"),
                artifactSHA256: digest("c"),
                tokenizerSHA256: digest("d"),
                templateSHA256: digest("e"),
                cacheLayoutSHA256: digest("f"),
                requestNamespaceSHA256: digest("0"))
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .invalidIdentityField("modelSHA256"))
        }

        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: [],
                caches: hybridCaches())
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .emptyPrefix)
        }

        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: [1, -2],
                caches: hybridCaches())
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ExactHybridPrefixSnapshotError,
                .negativeTokenID(position: 1))
        }
    }

    private func assertDenseCaptureRejected(
        _ dense: KVCacheSimple,
        expectedReason: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try Qwen38ExactHybridPrefixSnapshot.capture(
                identity: identity(),
                prefixTokenIDs: prefix,
                caches: [recurrentCache(), dense]),
            file: file,
            line: line
        ) { error in
            guard case .partialCacheState(let index, let typeName, let reason) =
                error as? Qwen38ExactHybridPrefixSnapshotError
            else {
                return XCTFail("expected partialCacheState, got \(error)", file: file, line: line)
            }
            XCTAssertEqual(index, 1, file: file, line: line)
            XCTAssertEqual(typeName, "MLXLMCommon.KVCacheSimple", file: file, line: line)
            XCTAssertEqual(reason, expectedReason, file: file, line: line)
        }
    }
}

private final class UnsupportedCopyableCache: KVCache {
    private let payload = MLXArray([Float(1)])
    var offset = 1
    var maxSize: Int? { nil }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] {
        get { [payload] }
        set {}
    }

    var metaState: [String] {
        get { [""] }
        set {}
    }

    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        UnsupportedCopyableCache()
    }
}
