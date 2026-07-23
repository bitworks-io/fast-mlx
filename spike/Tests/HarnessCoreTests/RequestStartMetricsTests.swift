import XCTest
@testable import HarnessCore

final class RequestStartMetricsTests: XCTestCase {
    func testPartialHitSeparatesLogicalAndPhysicalPrefillRates() throws {
        let metrics = try RequestStartMetrics(
            promptTokenCount: 1_000,
            cacheReadTokenCount: 900,
            physicalPrefillTokenCount: 100,
            prefixCacheOutcome: .partialHit,
            templateTokenCacheHit: true,
            templateSeconds: 0.001,
            tokenizeSeconds: 0.002,
            lookupSeconds: 0.003,
            restoreSeconds: 0.004,
            prefillSeconds: 0.5,
            retainedBytes: 1_024,
            entryCount: 2,
            evictionCount: 1,
            eagerWarmupSeconds: 0.25)

        XCTAssertEqual(metrics.apparentPrefillTokensPerSecond, 2_000)
        XCTAssertEqual(metrics.physicalPrefillTokensPerSecond, 200)
        XCTAssertEqual(metrics.cacheReadTokenCount, 900)
        XCTAssertEqual(metrics.physicalPrefillTokenCount, 100)
    }

    func testColdMissAndExactHitHaveExplicitTokenConservation() throws {
        let miss = try RequestStartMetrics(
            promptTokenCount: 100,
            cacheReadTokenCount: 0,
            physicalPrefillTokenCount: 100,
            prefixCacheOutcome: .miss,
            templateTokenCacheHit: false,
            templateSeconds: 0.1,
            tokenizeSeconds: 0.1,
            lookupSeconds: 0.01,
            restoreSeconds: 0,
            prefillSeconds: 0.5,
            retainedBytes: 0,
            entryCount: 0,
            evictionCount: 0)
        XCTAssertEqual(miss.apparentPrefillTokensPerSecond, 200)
        XCTAssertEqual(miss.physicalPrefillTokensPerSecond, 200)

        let exact = try RequestStartMetrics(
            promptTokenCount: 100,
            cacheReadTokenCount: 100,
            physicalPrefillTokenCount: 0,
            prefixCacheOutcome: .exactHit,
            templateTokenCacheHit: true,
            templateSeconds: 0,
            tokenizeSeconds: 0,
            lookupSeconds: 0.01,
            restoreSeconds: 0.02,
            prefillSeconds: 0.02,
            retainedBytes: 1_024,
            entryCount: 1,
            evictionCount: 0)
        XCTAssertEqual(exact.apparentPrefillTokensPerSecond, 5_000)
        XCTAssertEqual(exact.physicalPrefillTokensPerSecond, 0)
    }

    func testImpossibleOrPartialEvidenceFailsClosed() throws {
        XCTAssertThrowsError(
            try RequestStartMetrics(
                promptTokenCount: 100,
                cacheReadTokenCount: 80,
                physicalPrefillTokenCount: 30,
                prefixCacheOutcome: .partialHit,
                templateTokenCacheHit: false,
                templateSeconds: 0,
                tokenizeSeconds: 0,
                lookupSeconds: 0,
                restoreSeconds: 0,
                prefillSeconds: 1,
                retainedBytes: 0,
                entryCount: 0,
                evictionCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? RequestStartMetricsError,
                .promptTokenConservation)
        }

        let valid = try RequestStartMetrics(
            promptTokenCount: 100,
            cacheReadTokenCount: 0,
            physicalPrefillTokenCount: 100,
            prefixCacheOutcome: .miss,
            templateTokenCacheHit: false,
            templateSeconds: 0,
            tokenizeSeconds: 0,
            lookupSeconds: 0,
            restoreSeconds: 0,
            prefillSeconds: 1,
            retainedBytes: 0,
            entryCount: 0,
            evictionCount: 0)
        let encoded = try JSONEncoder().encode(valid)
        var decodedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded)
                as? [String: Any])
        decodedObject["physicalPrefillTokenCount"] = 1
        let invalidEncoded = try JSONSerialization.data(
            withJSONObject: decodedObject)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                RequestStartMetrics.self,
                from: invalidEncoded))

        XCTAssertThrowsError(
            try RequestStartMetrics(
                promptTokenCount: 100,
                cacheReadTokenCount: 0,
                physicalPrefillTokenCount: 100,
                prefixCacheOutcome: .exactHit,
                templateTokenCacheHit: false,
                templateSeconds: 0,
                tokenizeSeconds: 0,
                lookupSeconds: 0,
                restoreSeconds: 0,
                prefillSeconds: 1,
                retainedBytes: 0,
                entryCount: 0,
                evictionCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? RequestStartMetricsError,
                .outcomeTokenMismatch)
        }

        XCTAssertThrowsError(
            try RequestStartMetrics(
                promptTokenCount: 1,
                cacheReadTokenCount: 0,
                physicalPrefillTokenCount: 1,
                prefixCacheOutcome: .miss,
                templateTokenCacheHit: false,
                templateSeconds: -.infinity,
                tokenizeSeconds: 0,
                lookupSeconds: 0,
                restoreSeconds: 0,
                prefillSeconds: 1,
                retainedBytes: 0,
                entryCount: 0,
                evictionCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? RequestStartMetricsError,
                .invalidDuration("templateSeconds"))
        }

        XCTAssertThrowsError(
            try RequestStartMetrics(
                promptTokenCount: Int.max,
                cacheReadTokenCount: 0,
                physicalPrefillTokenCount: Int.max,
                prefixCacheOutcome: .miss,
                templateTokenCacheHit: false,
                templateSeconds: 0,
                tokenizeSeconds: 0,
                lookupSeconds: 0,
                restoreSeconds: 0,
                prefillSeconds: .leastNonzeroMagnitude,
                retainedBytes: 0,
                entryCount: 0,
                evictionCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? RequestStartMetricsError,
                .nonFiniteDerivedRate(
                    "apparentPrefillTokensPerSecond"))
        }

        XCTAssertThrowsError(
            try RequestStartMetrics(
                promptTokenCount: Int.max,
                cacheReadTokenCount: Int.max,
                physicalPrefillTokenCount: 1,
                prefixCacheOutcome: .partialHit,
                templateTokenCacheHit: false,
                templateSeconds: 0,
                tokenizeSeconds: 0,
                lookupSeconds: 0,
                restoreSeconds: 0,
                prefillSeconds: 1,
                retainedBytes: 0,
                entryCount: 0,
                evictionCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? RequestStartMetricsError,
                .promptTokenConservation)
        }
    }
}
