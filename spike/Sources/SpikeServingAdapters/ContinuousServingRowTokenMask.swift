import Foundation

import HarnessCore
import ServingCore

/// A `ContinuousBatchRowTokenMask` that also reports a recorded constraint failure. Non-generic so
/// the serving backend can hold any constraint's mask (object, schema, tool union) in one field.
protocol ContinuousServingFailureRecordingMask: ContinuousBatchRowTokenMask {
    /// The first failure the mask recorded (a constraint that rejected an emitted token, threw while
    /// building its allowed set, or had no allowed token left); `nil` while healthy.
    var recordedFailure: (any Error)? { get }
}

/// Why a mask recorded a failure without the constraint throwing one.
enum ContinuousServingRowTokenMaskError: Error, Equatable, Sendable {
    /// The constraint's allowed-token set was empty (asserted unreachable in the constraints).
    case noAllowedToken
}

/// Adapts a `TokenConstraint` to the continuous-batch runtime's per-row mask seam. It carries the
/// SAME thinking-phase gate and failure recording as the scalar route's
/// `ConstraintMaskingLogitProcessor`:
/// - `active == false` (a thinking request): the row is unconstrained (`nil` bitset) and `advance`
///   only watches for `thinkEndTokenID`, activating on it (the think-end token itself is not fed to
///   the constraint);
/// - `active == true`: `allowedTokenBitset()` is the constraint's bitset, and `advance` feeds each
///   emitted token to the constraint;
/// - a constraint that throws, or an all-zero bitset, records a failure, answers `nil`, and from then
///   on every call is inert. The runtime protocol is non-throwing on purpose (one row's failure must
///   never fail a batch step other rows share), so the BACKEND reads `recordedFailure` and fails
///   only that request.
///
/// `@unchecked Sendable`: the runtime calls `allowedTokenBitset()`/`advance(token:)` from the
/// coordinator's executor while the backend actor reads `recordedFailure`; every access to the mutable
/// state is serialized by `lock`.
final class ContinuousServingRowTokenMask<C: TokenConstraint>: ContinuousServingFailureRecordingMask,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var constraint: C
    private var active: Bool
    private let thinkEndTokenID: Int?
    private var failure: (any Error)?

    /// - Parameters:
    ///   - activeFromStart: `true` for a request that does not separate reasoning (the mask applies
    ///     from the first token); `false` for a thinking request (inactive until `thinkEndTokenID`).
    ///     The backend refuses `false` with a `nil` `thinkEndTokenID` before building a mask.
    init(constraint: C, activeFromStart: Bool, thinkEndTokenID: Int?) {
        self.constraint = constraint
        self.active = activeFromStart
        self.thinkEndTokenID = thinkEndTokenID
    }

    var recordedFailure: (any Error)? {
        lock.lock()
        defer { lock.unlock() }
        return failure
    }

    func allowedTokenBitset() -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        guard active, failure == nil else { return nil }
        let bitset: [UInt64]
        do {
            bitset = try constraint.allowedTokenBitset()
        } catch {
            failure = error
            return nil
        }
        guard bitset.contains(where: { $0 != 0 }) else {
            failure = ContinuousServingRowTokenMaskError.noAllowedToken
            return nil
        }
        return bitset
    }

    func advance(token: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard failure == nil else { return }
        guard active else {
            if let thinkEndTokenID, token == thinkEndTokenID {
                active = true
            }
            return
        }
        do {
            try constraint.advance(token: token)
        } catch {
            failure = error
        }
    }
}
