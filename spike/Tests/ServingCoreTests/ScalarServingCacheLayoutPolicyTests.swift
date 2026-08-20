import XCTest

@testable import ServingCore

final class ScalarServingCacheLayoutPolicyTests: XCTestCase {
    func testDenseAttentionLayoutIsScalarCompatible() throws {
        XCTAssertNoThrow(
            try validateScalarServingCacheLayout([
                .denseAttention,
                .denseAttention,
            ]))
    }

    func testEmptyRotatingRecurrentCompositeAndUnknownLayoutsFailClosed() {
        XCTAssertThrowsError(
            try validateScalarServingCacheLayout([])
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingCacheLayoutError,
                .emptyCacheLayout)
        }

        let rejected: [ScalarServingNativeCacheKind] = [
            .rotatingAttention,
            .recurrentState,
            .composite,
            .unknown,
        ]
        for kind in rejected {
            XCTAssertThrowsError(
                try validateScalarServingCacheLayout([
                    .denseAttention,
                    kind,
                ])
            ) { error in
                XCTAssertEqual(
                    error as? ScalarServingCacheLayoutError,
                    .unsupportedCacheLayout(index: 1, kind: kind))
            }
        }
    }

    // MARK: - continuous serve-route cache-layout validation (hybrid admission, incr-2)
    //
    // The continuous-batch route is STRICTER than the scalar route: it flat-rejects `.recurrentState`
    // unless the operator has opted qwen3_5 hybrid in (`--allow-hybrid-qwen35` → proof-certified
    // `hybridAdmitted: true`). Rotating/composite/unknown stay unsupported on both branches (no audited
    // continuous-route correctness). See docs/task-inbox/2026-08-20-hybrid-continuous-serve-path-admission.md.

    func testContinuousValidator_denseLayout_acceptedRegardlessOfHybridFlag() {
        XCTAssertNoThrow(
            try validateContinuousServingCacheLayout(
                [.denseAttention, .denseAttention], hybridAdmitted: false))
        XCTAssertNoThrow(
            try validateContinuousServingCacheLayout(
                [.denseAttention, .denseAttention], hybridAdmitted: true))
    }

    func testContinuousValidator_recurrentState_acceptedOnlyWhenHybridAdmitted() {
        // Not admitted → the recurrent layer fails closed exactly like the scalar validator.
        XCTAssertThrowsError(
            try validateContinuousServingCacheLayout(
                [.denseAttention, .recurrentState], hybridAdmitted: false)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingCacheLayoutError,
                .unsupportedCacheLayout(index: 1, kind: .recurrentState))
        }
        // Admitted → the hybrid layout (dense + recurrent) is accepted.
        XCTAssertNoThrow(
            try validateContinuousServingCacheLayout(
                [.denseAttention, .recurrentState], hybridAdmitted: true))
        XCTAssertNoThrow(
            try validateContinuousServingCacheLayout(
                [.recurrentState], hybridAdmitted: true))
    }

    func testContinuousValidator_rotatingCompositeUnknown_failClosedEvenWhenHybridAdmitted() {
        let rejected: [ScalarServingNativeCacheKind] = [.rotatingAttention, .composite, .unknown]
        for kind in rejected {
            XCTAssertThrowsError(
                try validateContinuousServingCacheLayout(
                    [.denseAttention, kind], hybridAdmitted: true)
            ) { error in
                XCTAssertEqual(
                    error as? ScalarServingCacheLayoutError,
                    .unsupportedCacheLayout(index: 1, kind: kind))
            }
        }
    }

    func testContinuousValidator_emptyLayout_failsClosedEitherWay() {
        for admitted in [false, true] {
            XCTAssertThrowsError(
                try validateContinuousServingCacheLayout([], hybridAdmitted: admitted)
            ) { error in
                XCTAssertEqual(
                    error as? ScalarServingCacheLayoutError, .emptyCacheLayout)
            }
        }
    }

    // Pin: the continuous validator with `hybridAdmitted: false` is byte-identical to the scalar
    // validator, so the flag-OFF continuous route keeps today's fail-closed behavior exactly.
    func testContinuousValidator_notAdmitted_matchesScalarValidator() {
        let layouts: [[ScalarServingNativeCacheKind]] = [
            [.denseAttention],
            [.denseAttention, .rotatingAttention],
            [.denseAttention, .recurrentState],
            [.composite],
            [],
        ]
        for layout in layouts {
            let scalar = Result { try validateScalarServingCacheLayout(layout) }
            let continuous = Result {
                try validateContinuousServingCacheLayout(layout, hybridAdmitted: false)
            }
            switch (scalar, continuous) {
            case (.success, .success):
                break
            case (.failure(let s), .failure(let c)):
                XCTAssertEqual(
                    s as? ScalarServingCacheLayoutError,
                    c as? ScalarServingCacheLayoutError,
                    "layout \(layout) must fail identically on both validators")
            default:
                XCTFail("scalar and not-admitted continuous validators disagreed on \(layout)")
            }
        }
    }

    // Pin: the scalar DECODER-ROUTE classifier is untouched by the continuous-admission work. The
    // scalar route still admits `.recurrentState` (→ nativeHeterogeneous) and routes all-dense to the
    // compiled fast path; rotating/composite/unknown still fail closed. Guards the spec's "DO NOT change
    // classifyScalarServingDecoderRoute" invariant.
    func testScalarDecoderRouteClassificationUnchanged() throws {
        XCTAssertEqual(
            try classifyScalarServingDecoderRoute([.denseAttention, .denseAttention]),
            .compiled)
        XCTAssertEqual(
            try classifyScalarServingDecoderRoute([.denseAttention, .recurrentState]),
            .nativeHeterogeneous)
        XCTAssertEqual(
            try classifyScalarServingDecoderRoute([.recurrentState]),
            .nativeHeterogeneous)
        XCTAssertThrowsError(
            try classifyScalarServingDecoderRoute([.denseAttention, .rotatingAttention])
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingCacheLayoutError,
                .unsupportedCacheLayout(index: 1, kind: .rotatingAttention))
        }
    }

    // MARK: - snapshot-reuse granularity (cold-prefix reuse rule per cache layout, #3a connect)

    /// All-dense layout → block-aligned reuse (dense blocks are independently restorable).
    func testSnapshotReuse_allDense_isBlockAligned() {
        XCTAssertEqual(
            snapshotReuseGranularity(for: [.denseAttention, .denseAttention]),
            .blockAligned)
    }

    /// A layout containing recurrent state (e.g. Qwen3.5 hybrid) → whole-snapshot-only: the
    /// recurrent state can't be rewound to a mid-snapshot boundary.
    func testSnapshotReuse_recurrentStateLayout_isWholeSnapshotOnly() {
        XCTAssertEqual(
            snapshotReuseGranularity(for: [.denseAttention, .recurrentState]),
            .wholeSnapshotOnly)
        XCTAssertEqual(
            snapshotReuseGranularity(for: [.recurrentState]),
            .wholeSnapshotOnly)
    }

    /// Rotating (sliding-window), composite, unknown, and empty layouts have no audited reuse rule
    /// → `.unsupported` (fail closed as a value, not a throw).
    func testSnapshotReuse_unauditedOrEmptyLayouts_areUnsupported() {
        XCTAssertEqual(snapshotReuseGranularity(for: [.denseAttention, .rotatingAttention]), .unsupported)
        XCTAssertEqual(snapshotReuseGranularity(for: [.composite]), .unsupported)
        XCTAssertEqual(snapshotReuseGranularity(for: [.unknown]), .unsupported)
        XCTAssertEqual(snapshotReuseGranularity(for: []), .unsupported)
    }

    /// `.unsupported` dominates: a layout mixing recurrent AND rotating state is unsupported, not
    /// promoted to whole-snapshot-only.
    func testSnapshotReuse_unsupportedDominatesRecurrent() {
        XCTAssertEqual(
            snapshotReuseGranularity(for: [.recurrentState, .rotatingAttention]),
            .unsupported)
    }
}
