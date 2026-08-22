import Foundation

/// A cache snapshot that can be compared against an independently computed scalar path.
///
/// `tokenLengthsByLayer` is the logical sequence length per decoder layer. Recurrent layers may be
/// physically fixed-size, but the adapter must still report the logical token frontier so this contract
/// can detect skew and verify the committed prefix length without knowing the backend's tensor layout.
public protocol SpeculativeCacheSnapshotProtocol: Equatable, Sendable {
    var tokenLengthsByLayer: [Int] { get }
}

/// The cache layout supported by speculative cache finalization.
public enum SpeculativeCacheLayout: Equatable, Sendable {
    /// Uniform dense-attention KV cache. Partial acceptance can trim only the rejected draft suffix.
    case dense(layerCount: Int)
    /// Mixed dense-attention plus recurrent-state layout, such as Qwen-style hybrid linear attention.
    case hybrid(HybridLayerKindMap)
}

/// Backend operation names surfaced in typed transaction errors.
public enum SpeculativeCacheBackendOperation: Equatable, Sendable {
    case currentSnapshot
    case restorePreDraft
    case rollbackDenseSuffix
    case replayCommittedInputTokens
}

/// A restore failure observed while failing closed.
public struct SpeculativeCacheCleanupFailure: Error, Equatable, Sendable {
    public let operation: SpeculativeCacheBackendOperation
    public let message: String

    public init(operation: SpeculativeCacheBackendOperation, message: String) {
        self.operation = operation
        self.message = message
    }
}

/// Fail-closed outcomes for speculative cache finalization.
public enum SpeculativeCacheTransactionError: Error, Equatable, Sendable {
    case unsupportedLayout(String, cleanupFailure: SpeculativeCacheCleanupFailure?)
    case cacheSkew(lengths: [Int], cleanupFailure: SpeculativeCacheCleanupFailure?)
    case invalidVerifySpan(expectedLength: Int, actualLengths: [Int], cleanupFailure: SpeculativeCacheCleanupFailure?)
    case postCommitLengthMismatch(expectedLength: Int, actualLengths: [Int], cleanupFailure: SpeculativeCacheCleanupFailure?)
    case scalarLengthMismatch(expectedLength: Int, scalarLengths: [Int], cleanupFailure: SpeculativeCacheCleanupFailure?)
    case spanOverflow(preDraftLength: Int, span: Int, cleanupFailure: SpeculativeCacheCleanupFailure?)
    case invalidConfirmedCount(confirmed: Int, draftCount: Int, cleanupFailure: SpeculativeCacheCleanupFailure?)
    case transactionAlreadyFinalized
    case cancelled(cleanupFailure: SpeculativeCacheCleanupFailure?)
    case backendMutationFailed(
        operation: SpeculativeCacheBackendOperation,
        message: String,
        cleanupFailure: SpeculativeCacheCleanupFailure?)
    case parityMismatch(cleanupFailure: SpeculativeCacheCleanupFailure?)
}

/// Minimal backend adapter needed to make cache finalization testable without MLX.
public protocol SpeculativeCacheTransactionBackend {
    associatedtype Snapshot: SpeculativeCacheSnapshotProtocol

    mutating func currentSnapshot() throws -> Snapshot
    mutating func restore(_ snapshot: Snapshot) throws
    mutating func rollbackDenseSuffix(tokenCount: Int) throws
    mutating func replayCommittedInputTokens(_ tokens: [Int]) throws
}

/// Result of a successful speculative cache finalization.
public struct SpeculativeCacheTransactionResult<Snapshot: SpeculativeCacheSnapshotProtocol>: Equatable, Sendable {
    /// Number of drafted tokens accepted from the proposed draft prefix.
    public let acceptedDraftCount: Int
    /// The cache-committed input tokens: the already selected input token plus the accepted draft prefix.
    ///
    /// This deliberately does not include the newly sampled correction/trailing bonus output from the
    /// verify logits. That output is still uncommitted and belongs to the next decode step.
    public let committedInputTokens: [Int]
    /// Number of drafted suffix tokens removed or discarded.
    public let rejectedDraftCount: Int
    public let finalSnapshot: Snapshot
}

public enum SpeculativeCacheTransaction {
    /// Capture the cache state before the target verify pass appends the committed input token and
    /// draft span.
    public static func begin<Backend: SpeculativeCacheTransactionBackend>(
        layout: SpeculativeCacheLayout,
        committedInputToken: Int,
        draftTokens: [Int],
        backend: inout Backend
    ) throws -> SpeculativeCacheTransactionSession<Backend> {
        _ = try supportedLayerCount(for: layout)
        let preDraftSnapshot = try backend.currentSnapshot()
        return SpeculativeCacheTransactionSession(
            layout: layout,
            committedInputToken: committedInputToken,
            draftTokens: draftTokens,
            preDraftSnapshot: preDraftSnapshot,
            state: SpeculativeCacheTransactionState())
    }
}

public struct SpeculativeCacheTransactionSession<Backend: SpeculativeCacheTransactionBackend> {
    public let layout: SpeculativeCacheLayout
    public let committedInputToken: Int
    public let draftTokens: [Int]
    public let preDraftSnapshot: Backend.Snapshot
    private let state: SpeculativeCacheTransactionState

    fileprivate init(
        layout: SpeculativeCacheLayout,
        committedInputToken: Int,
        draftTokens: [Int],
        preDraftSnapshot: Backend.Snapshot,
        state: SpeculativeCacheTransactionState
    ) {
        self.layout = layout
        self.committedInputToken = committedInputToken
        self.draftTokens = draftTokens
        self.preDraftSnapshot = preDraftSnapshot
        self.state = state
    }

    /// Restore the captured pre-draft snapshot without accepting any draft tokens.
    public func abort(backend: inout Backend) throws {
        guard !state.isTerminal else {
            throw SpeculativeCacheTransactionError.transactionAlreadyFinalized
        }
        state.isTerminal = true
        try backend.restore(preDraftSnapshot)
        let restored = try backend.currentSnapshot()
        guard restored == preDraftSnapshot else {
            throw SpeculativeCacheTransactionError.backendMutationFailed(
                operation: .restorePreDraft,
                message: "restore returned but snapshot did not match pre-draft snapshot",
                cleanupFailure: nil)
        }
    }

    /// Finalize cache state after the target verify pass has appended `committedInputToken + draftTokens`.
    ///
    /// `nConfirmed` counts accepted draft tokens only. On success, the cache contains the committed
    /// input token and the accepted draft prefix. The sampled correction/trailing bonus output from the
    /// verify logits is intentionally not appended here; it remains the uncommitted input for the next
    /// decode step.
    public func finalize(
        nConfirmed: Int,
        scalarEquivalentSnapshot: Backend.Snapshot,
        backend: inout Backend,
        isCancelled: () -> Bool = { false }
    ) throws -> SpeculativeCacheTransactionResult<Backend.Snapshot> {
        guard !state.isTerminal else {
            throw SpeculativeCacheTransactionError.transactionAlreadyFinalized
        }
        state.isTerminal = true

        if isCancelled() {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.cancelled(cleanupFailure: cleanup)
        }

        guard (0...draftTokens.count).contains(nConfirmed) else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.invalidConfirmedCount(
                confirmed: nConfirmed,
                draftCount: draftTokens.count,
                cleanupFailure: cleanup)
        }

        let layerCount: Int
        do {
            layerCount = try supportedLayerCount(for: layout)
        } catch let error as SpeculativeCacheTransactionError {
            let cleanup = restorePreDraft(backend: &backend)
            throw error.withCleanup(cleanup)
        }

        let preDraftLength: Int
        do {
            preDraftLength = try uniformLength(preDraftSnapshot, expectedLayerCount: layerCount)
        } catch let error as SpeculativeCacheTransactionError {
            let cleanup = restorePreDraft(backend: &backend)
            throw error.withCleanup(cleanup)
        }

        let verifySpan = 1 + draftTokens.count
        let (verifyLength, verifyOverflow) = preDraftLength.addingReportingOverflow(verifySpan)
        guard !verifyOverflow else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.spanOverflow(
                preDraftLength: preDraftLength,
                span: verifySpan,
                cleanupFailure: cleanup)
        }

        let acceptedSpan = 1 + nConfirmed
        let (finalLength, finalOverflow) = preDraftLength.addingReportingOverflow(acceptedSpan)
        guard !finalOverflow else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.spanOverflow(
                preDraftLength: preDraftLength,
                span: acceptedSpan,
                cleanupFailure: cleanup)
        }

        let currentSnapshot: Backend.Snapshot
        do {
            currentSnapshot = try backend.currentSnapshot()
        } catch {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.backendMutationFailed(
                operation: .currentSnapshot,
                message: String(describing: error),
                cleanupFailure: cleanup)
        }

        let currentLength: Int
        do {
            currentLength = try uniformLength(currentSnapshot, expectedLayerCount: layerCount)
        } catch let error as SpeculativeCacheTransactionError {
            let cleanup = restorePreDraft(backend: &backend)
            throw error.withCleanup(cleanup)
        }
        guard currentLength == verifyLength else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.invalidVerifySpan(
                expectedLength: verifyLength,
                actualLengths: currentSnapshot.tokenLengthsByLayer,
                cleanupFailure: cleanup)
        }

        let scalarLength: Int
        do {
            scalarLength = try uniformLength(scalarEquivalentSnapshot, expectedLayerCount: layerCount)
        } catch let error as SpeculativeCacheTransactionError {
            let cleanup = restorePreDraft(backend: &backend)
            throw error.withCleanup(cleanup)
        }
        guard scalarLength == finalLength else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.scalarLengthMismatch(
                expectedLength: finalLength,
                scalarLengths: scalarEquivalentSnapshot.tokenLengthsByLayer,
                cleanupFailure: cleanup)
        }

        let committedInputTokens = [committedInputToken] + Array(draftTokens.prefix(nConfirmed))
        let rejectedDraftCount = draftTokens.count - nConfirmed
        if rejectedDraftCount > 0 {
            switch layout {
            case .dense:
                do {
                    try backend.rollbackDenseSuffix(tokenCount: rejectedDraftCount)
                } catch {
                    let cleanup = restorePreDraft(backend: &backend)
                    throw SpeculativeCacheTransactionError.backendMutationFailed(
                        operation: .rollbackDenseSuffix,
                        message: String(describing: error),
                        cleanupFailure: cleanup)
                }
            case .hybrid:
                do {
                    try backend.restore(preDraftSnapshot)
                } catch {
                    let cleanup = restorePreDraft(backend: &backend)
                    throw SpeculativeCacheTransactionError.backendMutationFailed(
                        operation: .restorePreDraft,
                        message: String(describing: error),
                        cleanupFailure: cleanup)
                }
                do {
                    try backend.replayCommittedInputTokens(committedInputTokens)
                } catch {
                    let cleanup = restorePreDraft(backend: &backend)
                    throw SpeculativeCacheTransactionError.backendMutationFailed(
                        operation: .replayCommittedInputTokens,
                        message: String(describing: error),
                        cleanupFailure: cleanup)
                }
            }
        }

        let finalSnapshot: Backend.Snapshot
        do {
            finalSnapshot = try backend.currentSnapshot()
        } catch {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.backendMutationFailed(
                operation: .currentSnapshot,
                message: String(describing: error),
                cleanupFailure: cleanup)
        }

        let committedLength: Int
        do {
            committedLength = try uniformLength(finalSnapshot, expectedLayerCount: layerCount)
        } catch let error as SpeculativeCacheTransactionError {
            let cleanup = restorePreDraft(backend: &backend)
            throw error.withCleanup(cleanup)
        }
        guard committedLength == finalLength else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.postCommitLengthMismatch(
                expectedLength: finalLength,
                actualLengths: finalSnapshot.tokenLengthsByLayer,
                cleanupFailure: cleanup)
        }

        guard finalSnapshot == scalarEquivalentSnapshot else {
            let cleanup = restorePreDraft(backend: &backend)
            throw SpeculativeCacheTransactionError.parityMismatch(cleanupFailure: cleanup)
        }

        return SpeculativeCacheTransactionResult(
            acceptedDraftCount: nConfirmed,
            committedInputTokens: committedInputTokens,
            rejectedDraftCount: rejectedDraftCount,
            finalSnapshot: finalSnapshot)
    }

    private func uniformLength(
        _ snapshot: Backend.Snapshot,
        expectedLayerCount: Int
    ) throws -> Int {
        let lengths = snapshot.tokenLengthsByLayer
        guard lengths.count == expectedLayerCount, let first = lengths.first, first >= 0 else {
            throw SpeculativeCacheTransactionError.cacheSkew(lengths: lengths, cleanupFailure: nil)
        }
        guard lengths.allSatisfy({ $0 == first && $0 >= 0 }) else {
            throw SpeculativeCacheTransactionError.cacheSkew(lengths: lengths, cleanupFailure: nil)
        }
        return first
    }

    private func restorePreDraft(backend: inout Backend) -> SpeculativeCacheCleanupFailure? {
        do {
            try backend.restore(preDraftSnapshot)
        } catch {
            return SpeculativeCacheCleanupFailure(
                operation: .restorePreDraft,
                message: String(describing: error))
        }
        do {
            let restored = try backend.currentSnapshot()
            guard restored == preDraftSnapshot else {
                return SpeculativeCacheCleanupFailure(
                    operation: .restorePreDraft,
                    message: "restore returned but snapshot did not match pre-draft snapshot")
            }
            return nil
        } catch {
            return SpeculativeCacheCleanupFailure(
                operation: .currentSnapshot,
                message: String(describing: error))
        }
    }
}

private final class SpeculativeCacheTransactionState {
    var isTerminal = false
}

private func supportedLayerCount(for layout: SpeculativeCacheLayout) throws -> Int {
    switch layout {
    case .dense(let layerCount):
        guard layerCount > 0 else {
            throw SpeculativeCacheTransactionError.unsupportedLayout(
                "dense layout requires at least one layer",
                cleanupFailure: nil)
        }
        return layerCount
    case .hybrid(let map):
        guard map.layerCount > 0, map.isHeterogeneous, map.firstDenseLayerIndex != nil else {
            throw SpeculativeCacheTransactionError.unsupportedLayout(
                "hybrid layout requires at least one dense and one recurrent layer",
                cleanupFailure: nil)
        }
        return map.layerCount
    }
}

private extension SpeculativeCacheTransactionError {
    func withCleanup(_ cleanupFailure: SpeculativeCacheCleanupFailure?) -> SpeculativeCacheTransactionError {
        switch self {
        case .unsupportedLayout(let message, _):
            return .unsupportedLayout(message, cleanupFailure: cleanupFailure)
        case .cacheSkew(let lengths, _):
            return .cacheSkew(lengths: lengths, cleanupFailure: cleanupFailure)
        case .invalidVerifySpan(let expectedLength, let actualLengths, _):
            return .invalidVerifySpan(
                expectedLength: expectedLength,
                actualLengths: actualLengths,
                cleanupFailure: cleanupFailure)
        case .postCommitLengthMismatch(let expectedLength, let actualLengths, _):
            return .postCommitLengthMismatch(
                expectedLength: expectedLength,
                actualLengths: actualLengths,
                cleanupFailure: cleanupFailure)
        case .scalarLengthMismatch(let expectedLength, let scalarLengths, _):
            return .scalarLengthMismatch(
                expectedLength: expectedLength,
                scalarLengths: scalarLengths,
                cleanupFailure: cleanupFailure)
        case .spanOverflow(let preDraftLength, let span, _):
            return .spanOverflow(preDraftLength: preDraftLength, span: span, cleanupFailure: cleanupFailure)
        case .invalidConfirmedCount(let confirmed, let draftCount, _):
            return .invalidConfirmedCount(
                confirmed: confirmed,
                draftCount: draftCount,
                cleanupFailure: cleanupFailure)
        case .transactionAlreadyFinalized:
            return .transactionAlreadyFinalized
        case .cancelled:
            return .cancelled(cleanupFailure: cleanupFailure)
        case .backendMutationFailed(let operation, let message, _):
            return .backendMutationFailed(operation: operation, message: message, cleanupFailure: cleanupFailure)
        case .parityMismatch:
            return .parityMismatch(cleanupFailure: cleanupFailure)
        }
    }
}
