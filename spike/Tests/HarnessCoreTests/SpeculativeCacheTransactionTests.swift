import XCTest
@testable import HarnessCore

final class SpeculativeCacheTransactionTests: XCTestCase {
    func testDensePartialAcceptRollsBackOnlyRejectedDraftSuffix() throws {
        var backend = TestCacheBackend.dense(layerCount: 3, length: 8, prefix: [1, 2, 3])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 3),
            committedInputToken: 40,
            draftTokens: [41, 42, 43, 44],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 40, draftTokens: [41, 42, 43, 44])
        let scalar = preDraft.appending(tokens: [40, 41, 42], under: .dense(layerCount: 3))

        let result = try tx.finalize(nConfirmed: 2, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(result.acceptedDraftCount, 2)
        XCTAssertEqual(result.committedInputTokens, [40, 41, 42])
        XCTAssertEqual(result.rejectedDraftCount, 2)
        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .rollbackDenseSuffix(2), .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testDenseFullAcceptDoesNotRollbackOrReplay() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 4, prefix: [9])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 10,
            draftTokens: [11, 12],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 10, draftTokens: [11, 12])
        let scalar = preDraft.appending(tokens: [10, 11, 12], under: .dense(layerCount: 2))

        _ = try tx.finalize(nConfirmed: 2, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testDenseZeroDraftLeavesOnlyCommittedInputToken() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 3, prefix: [7])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 8,
            draftTokens: [],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 8, draftTokens: [])
        let scalar = preDraft.appending(tokens: [8], under: .dense(layerCount: 2))

        let result = try tx.finalize(nConfirmed: 0, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(result.committedInputTokens, [8])
        XCTAssertEqual(result.rejectedDraftCount, 0)
        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testDenseZeroAcceptanceRollsBackAllDraftsAndKeepsCommittedInputToken() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 3, prefix: [7])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 8,
            draftTokens: [9, 10],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 8, draftTokens: [9, 10])
        let scalar = preDraft.appending(tokens: [8], under: .dense(layerCount: 2))

        let result = try tx.finalize(nConfirmed: 0, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(result.committedInputTokens, [8])
        XCTAssertEqual(result.rejectedDraftCount, 2)
        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .rollbackDenseSuffix(2), .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testHybridPartialAcceptRestoresPreDraftAndReplaysCommittedPrefix() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 2))
        var backend = TestCacheBackend.hybrid(map: map, length: 6, prefix: [1, 2])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .hybrid(map),
            committedInputToken: 50,
            draftTokens: [51, 52, 53],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 50, draftTokens: [51, 52, 53])
        let scalar = preDraft.appending(tokens: [50, 51], under: .hybrid(map))

        let result = try tx.finalize(nConfirmed: 1, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(result.committedInputTokens, [50, 51])
        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .restore, .replay([50, 51]), .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testHybridZeroAcceptanceRestoresAndReplaysOnlyCommittedInputToken() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 2))
        var backend = TestCacheBackend.hybrid(map: map, length: 6, prefix: [1, 2])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .hybrid(map),
            committedInputToken: 50,
            draftTokens: [51, 52, 53],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 50, draftTokens: [51, 52, 53])
        let scalar = preDraft.appending(tokens: [50], under: .hybrid(map))

        let result = try tx.finalize(nConfirmed: 0, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(result.committedInputTokens, [50])
        XCTAssertEqual(result.rejectedDraftCount, 3)
        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .restore, .replay([50]), .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testHybridFullAcceptDoesNotRestoreOrReplay() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 2))
        var backend = TestCacheBackend.hybrid(map: map, length: 6, prefix: [1, 2])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .hybrid(map),
            committedInputToken: 60,
            draftTokens: [61, 62],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 60, draftTokens: [61, 62])
        let scalar = preDraft.appending(tokens: [60, 61, 62], under: .hybrid(map))

        _ = try tx.finalize(nConfirmed: 2, scalarEquivalentSnapshot: scalar, backend: &backend)

        XCTAssertEqual(backend.operations, [.snapshot, .snapshot, .snapshot, .snapshot])
        XCTAssertEqual(try backend.currentSnapshot(), scalar)
    }

    func testCancellationFailsClosedAndRestoresPreDraftSnapshot() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 5, prefix: [3])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 20,
            draftTokens: [21],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 20, draftTokens: [21])

        XCTAssertThrowsError(
            try tx.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: preDraft.appending(tokens: [20, 21], under: .dense(layerCount: 2)),
                backend: &backend,
                isCancelled: { true })
        ) { error in
            XCTAssertEqual(error as? SpeculativeCacheTransactionError, .cancelled(cleanupFailure: nil))
        }
        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }

    func testUnsupportedHybridLayoutIsRejectedAtBeginBeforeVerifyMutation() throws {
        let map = HybridLayerKindMap(kinds: [.recurrentState, .recurrentState])
        var backend = TestCacheBackend.hybrid(map: map, length: 5, prefix: [1])
        let preDraft = try backend.currentSnapshot()

        XCTAssertThrowsError(
            try SpeculativeCacheTransaction.begin(
                layout: .hybrid(map),
                committedInputToken: 2,
                draftTokens: [3],
                backend: &backend)
        ) { error in
            guard case .unsupportedLayout = error as? SpeculativeCacheTransactionError else {
                return XCTFail("expected unsupportedLayout, got \(error)")
            }
        }
        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }

    func testCacheSkewFailsClosedBeforeMutation() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 5, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 2,
            draftTokens: [3],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 2, draftTokens: [3])
        backend.skewLayer(1, by: 1)

        XCTAssertThrowsError(
            try tx.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: preDraft.appending(tokens: [2, 3], under: .dense(layerCount: 2)),
                backend: &backend)
        ) { error in
            guard case .cacheSkew = error as? SpeculativeCacheTransactionError else {
                return XCTFail("expected cacheSkew, got \(error)")
            }
        }
        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }

    func testOverflowAndInvalidConfirmedCountFailClosed() throws {
        var overflowBackend = TestCacheBackend.dense(layerCount: 1, length: Int.max, prefix: [])
        let overflowPreDraft = try overflowBackend.currentSnapshot()
        let overflowTx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 1,
            draftTokens: [],
            backend: &overflowBackend)

        XCTAssertThrowsError(
            try overflowTx.finalize(
                nConfirmed: 0,
                scalarEquivalentSnapshot: overflowPreDraft,
                backend: &overflowBackend)
        ) { error in
            guard case .spanOverflow = error as? SpeculativeCacheTransactionError else {
                return XCTFail("expected spanOverflow, got \(error)")
            }
        }
        XCTAssertEqual(try overflowBackend.currentSnapshot(), overflowPreDraft)

        var invalidBackend = TestCacheBackend.dense(layerCount: 1, length: 2, prefix: [])
        let invalidPreDraft = try invalidBackend.currentSnapshot()
        let invalidTx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 1,
            draftTokens: [2],
            backend: &invalidBackend)
        invalidBackend.appendVerifySpan(committedInputToken: 1, draftTokens: [2])

        XCTAssertThrowsError(
            try invalidTx.finalize(
                nConfirmed: 2,
                scalarEquivalentSnapshot: invalidPreDraft.appending(tokens: [1, 2], under: .dense(layerCount: 1)),
                backend: &invalidBackend)
        ) { error in
            XCTAssertEqual(
                error as? SpeculativeCacheTransactionError,
                .invalidConfirmedCount(confirmed: 2, draftCount: 1, cleanupFailure: nil))
        }
        XCTAssertEqual(try invalidBackend.currentSnapshot(), invalidPreDraft)
    }

    func testBackendReplayFailureFailsClosedAndReportsCleanup() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 2))
        var backend = TestCacheBackend.hybrid(map: map, length: 4, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .hybrid(map),
            committedInputToken: 30,
            draftTokens: [31, 32],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 30, draftTokens: [31, 32])
        backend.failOperations.insert(.replay([30, 31]))

        XCTAssertThrowsError(
            try tx.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: preDraft.appending(tokens: [30, 31], under: .hybrid(map)),
                backend: &backend)
        ) { error in
            XCTAssertEqual(
                error as? SpeculativeCacheTransactionError,
                .backendMutationFailed(operation: .replayCommittedInputTokens, message: "forced replay failure", cleanupFailure: nil))
        }
        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }

    func testDenseTrimFailureFailsClosedAndReportsCleanupFailure() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 3, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 2,
            draftTokens: [3, 4],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 2, draftTokens: [3, 4])
        backend.failOperations.insert(.rollbackDenseSuffix(1))
        backend.failOperations.insert(.restore)

        XCTAssertThrowsError(
            try tx.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: preDraft.appending(tokens: [2, 3], under: .dense(layerCount: 2)),
                backend: &backend)
        ) { error in
            XCTAssertEqual(
                error as? SpeculativeCacheTransactionError,
                .backendMutationFailed(
                    operation: .rollbackDenseSuffix,
                    message: "forced rollback failure",
                    cleanupFailure: .init(operation: .restorePreDraft, message: "forced restore failure")))
        }
    }

    func testPostCommitParityMismatchFailsClosed() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 3, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 2,
            draftTokens: [3],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 2, draftTokens: [3])
        let wrongScalar = preDraft.appending(tokens: [2, 99], under: .dense(layerCount: 2))

        XCTAssertThrowsError(
            try tx.finalize(nConfirmed: 1, scalarEquivalentSnapshot: wrongScalar, backend: &backend)
        ) { error in
            guard case .parityMismatch(let cleanupFailure) = error as? SpeculativeCacheTransactionError else {
                return XCTFail("expected parityMismatch, got \(error)")
            }
            XCTAssertNil(cleanupFailure)
        }
        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }

    func testSilentBadCleanupRestoreIsReported() throws {
        var backend = TestCacheBackend.dense(layerCount: 2, length: 3, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 2),
            committedInputToken: 2,
            draftTokens: [3],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 2, draftTokens: [3])
        backend.silentlyBadRestore = true
        let wrongScalar = preDraft.appending(tokens: [2, 99], under: .dense(layerCount: 2))

        XCTAssertThrowsError(
            try tx.finalize(nConfirmed: 1, scalarEquivalentSnapshot: wrongScalar, backend: &backend)
        ) { error in
            XCTAssertEqual(
                error as? SpeculativeCacheTransactionError,
                .parityMismatch(cleanupFailure: .init(
                    operation: .restorePreDraft,
                    message: "restore returned but snapshot did not match pre-draft snapshot")))
        }
    }

    func testTransactionSessionIsTerminalAfterFinalizeAbortAndFailure() throws {
        var successBackend = TestCacheBackend.dense(layerCount: 1, length: 1, prefix: [1])
        let successPre = try successBackend.currentSnapshot()
        let success = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 2,
            draftTokens: [],
            backend: &successBackend)
        let successCopy = success
        successBackend.appendVerifySpan(committedInputToken: 2, draftTokens: [])
        _ = try success.finalize(
            nConfirmed: 0,
            scalarEquivalentSnapshot: successPre.appending(tokens: [2], under: .dense(layerCount: 1)),
            backend: &successBackend)
        XCTAssertThrowsError(
            try successCopy.finalize(
                nConfirmed: 0,
                scalarEquivalentSnapshot: successPre.appending(tokens: [2], under: .dense(layerCount: 1)),
                backend: &successBackend)
        ) { error in
            XCTAssertEqual(error as? SpeculativeCacheTransactionError, .transactionAlreadyFinalized)
        }

        var abortBackend = TestCacheBackend.dense(layerCount: 1, length: 1, prefix: [1])
        let abortPre = try abortBackend.currentSnapshot()
        let aborted = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 2,
            draftTokens: [],
            backend: &abortBackend)
        abortBackend.appendVerifySpan(committedInputToken: 2, draftTokens: [])
        try aborted.abort(backend: &abortBackend)
        XCTAssertThrowsError(
            try aborted.finalize(
                nConfirmed: 0,
                scalarEquivalentSnapshot: abortPre.appending(tokens: [2], under: .dense(layerCount: 1)),
                backend: &abortBackend)
        ) { error in
            XCTAssertEqual(error as? SpeculativeCacheTransactionError, .transactionAlreadyFinalized)
        }

        var failedBackend = TestCacheBackend.dense(layerCount: 1, length: 1, prefix: [1])
        let failedPre = try failedBackend.currentSnapshot()
        let failed = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 2,
            draftTokens: [3],
            backend: &failedBackend)
        failedBackend.appendVerifySpan(committedInputToken: 2, draftTokens: [3])
        XCTAssertThrowsError(
            try failed.finalize(
                nConfirmed: 2,
                scalarEquivalentSnapshot: failedPre.appending(tokens: [2, 3], under: .dense(layerCount: 1)),
                backend: &failedBackend))
        XCTAssertThrowsError(
            try failed.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: failedPre.appending(tokens: [2, 3], under: .dense(layerCount: 1)),
                backend: &failedBackend)
        ) { error in
            XCTAssertEqual(error as? SpeculativeCacheTransactionError, .transactionAlreadyFinalized)
        }
    }

    func testAbortRestoresPreDraftSnapshot() throws {
        var backend = TestCacheBackend.dense(layerCount: 1, length: 3, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let tx = try SpeculativeCacheTransaction.begin(
            layout: .dense(layerCount: 1),
            committedInputToken: 2,
            draftTokens: [3],
            backend: &backend)

        backend.appendVerifySpan(committedInputToken: 2, draftTokens: [3])

        try tx.abort(backend: &backend)

        XCTAssertEqual(try backend.currentSnapshot(), preDraft)
    }
}

private struct TestCacheSnapshot: Equatable, Sendable, CustomStringConvertible, SpeculativeCacheSnapshotProtocol {
    let kinds: [LayerCacheKind]
    var tokenLengthsByLayer: [Int]
    var storedTokensByLayer: [[Int]]
    var recurrentStateByLayer: [[Int]]

    var description: String {
        "lengths=\(tokenLengthsByLayer), tokens=\(storedTokensByLayer), recurrent=\(recurrentStateByLayer)"
    }

    func appending(tokens: [Int], under layout: SpeculativeCacheLayout) -> TestCacheSnapshot {
        var copy = self
        copy.append(tokens: tokens, under: layout)
        return copy
    }

    private mutating func append(tokens: [Int], under layout: SpeculativeCacheLayout) {
        for layer in kinds.indices {
            tokenLengthsByLayer[layer] += tokens.count
            switch kinds[layer] {
            case .denseAttention:
                storedTokensByLayer[layer].append(contentsOf: tokens)
            case .recurrentState:
                recurrentStateByLayer[layer].append(contentsOf: tokens.map { $0 + layer })
            }
        }
        _ = layout
    }
}

private enum TestBackendError: Error, CustomStringConvertible {
    case forced(String)

    var description: String {
        switch self {
        case .forced(let message): message
        }
    }
}

private struct TestCacheBackend: SpeculativeCacheTransactionBackend {
    var snapshot: TestCacheSnapshot
    var operations: [TestBackendOperation] = []
    var failOperations: Set<TestBackendOperation> = []
    var silentlyBadRestore = false

    static func dense(layerCount: Int, length: Int, prefix: [Int]) -> TestCacheBackend {
        let kinds = Array(repeating: LayerCacheKind.denseAttention, count: layerCount)
        return TestCacheBackend(snapshot: TestCacheSnapshot(
            kinds: kinds,
            tokenLengthsByLayer: Array(repeating: length, count: layerCount),
            storedTokensByLayer: Array(repeating: prefix, count: layerCount),
            recurrentStateByLayer: Array(repeating: [], count: layerCount)))
    }

    static func hybrid(map: HybridLayerKindMap, length: Int, prefix: [Int]) -> TestCacheBackend {
        TestCacheBackend(snapshot: TestCacheSnapshot(
            kinds: map.kinds,
            tokenLengthsByLayer: Array(repeating: length, count: map.layerCount),
            storedTokensByLayer: map.kinds.map { $0 == .denseAttention ? prefix : [] },
            recurrentStateByLayer: map.kinds.enumerated().map { index, kind in
                kind == .recurrentState ? prefix.map { $0 + index } : []
            }))
    }

    mutating func currentSnapshot() throws -> TestCacheSnapshot {
        operations.append(.snapshot)
        return snapshot
    }

    mutating func restore(_ snapshot: TestCacheSnapshot) throws {
        operations.append(.restore)
        if failOperations.contains(.restore) {
            throw TestBackendError.forced("forced restore failure")
        }
        if silentlyBadRestore {
            self.snapshot = snapshot.appending(tokens: [10_000], under: .dense(layerCount: snapshot.kinds.count))
            return
        }
        self.snapshot = snapshot
    }

    mutating func rollbackDenseSuffix(tokenCount: Int) throws {
        operations.append(.rollbackDenseSuffix(tokenCount))
        if failOperations.contains(.rollbackDenseSuffix(tokenCount)) {
            throw TestBackendError.forced("forced rollback failure")
        }
        for layer in snapshot.kinds.indices where snapshot.kinds[layer] == .denseAttention {
            snapshot.tokenLengthsByLayer[layer] -= tokenCount
            snapshot.storedTokensByLayer[layer].removeLast(tokenCount)
        }
    }

    mutating func replayCommittedInputTokens(_ tokens: [Int]) throws {
        operations.append(.replay(tokens))
        if failOperations.contains(.replay(tokens)) {
            throw TestBackendError.forced("forced replay failure")
        }
        append(tokens: tokens)
    }

    mutating func appendVerifySpan(committedInputToken: Int, draftTokens: [Int]) {
        append(tokens: [committedInputToken] + draftTokens)
    }

    mutating func skewLayer(_ layer: Int, by delta: Int) {
        snapshot.tokenLengthsByLayer[layer] += delta
    }

    private mutating func append(tokens: [Int]) {
        for layer in snapshot.kinds.indices {
            snapshot.tokenLengthsByLayer[layer] += tokens.count
            switch snapshot.kinds[layer] {
            case .denseAttention:
                snapshot.storedTokensByLayer[layer].append(contentsOf: tokens)
            case .recurrentState:
                snapshot.recurrentStateByLayer[layer].append(contentsOf: tokens.map { $0 + layer })
            }
        }
    }
}

private enum TestBackendOperation: Equatable, Hashable, Sendable {
    case snapshot
    case restore
    case rollbackDenseSuffix(Int)
    case replay([Int])
}
