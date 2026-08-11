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
}
