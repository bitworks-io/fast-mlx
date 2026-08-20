import XCTest

@testable import ServingCore

final class ScalarServingDecoderRouteTests: XCTestCase {
    func testAllDenseAttentionRoutesToCompiled() throws {
        let route = try classifyScalarServingDecoderRoute([
            .denseAttention,
            .denseAttention,
        ])
        XCTAssertEqual(route, .compiled)
    }

    func testMixedDenseAndRecurrentStateRoutesToNativeHeterogeneous() throws {
        let route = try classifyScalarServingDecoderRoute([
            .denseAttention,
            .recurrentState,
            .denseAttention,
        ])
        XCTAssertEqual(route, .nativeHeterogeneous)
    }

    func testAllRecurrentStateRoutesToNativeHeterogeneous() throws {
        let route = try classifyScalarServingDecoderRoute([
            .recurrentState,
            .recurrentState,
        ])
        XCTAssertEqual(route, .nativeHeterogeneous)
    }

    func testEmptyLayoutThrowsEmptyCacheLayout() {
        XCTAssertThrowsError(
            try classifyScalarServingDecoderRoute([])
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingCacheLayoutError,
                .emptyCacheLayout)
        }
    }

    func testRotatingCompositeUnknownStillFailClosed() {
        let rejected: [ScalarServingNativeCacheKind] = [
            .rotatingAttention,
            .composite,
            .unknown,
        ]
        for kind in rejected {
            XCTAssertThrowsError(
                try classifyScalarServingDecoderRoute([
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
