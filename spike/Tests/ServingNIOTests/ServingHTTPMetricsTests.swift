import XCTest

@testable import ServingNIO

/// Unit coverage for the always-on `/metrics` HTTP dependability primitives in
/// `ServingHTTPMetrics.swift`: the two pure label-mapping helpers, the boundary formatter, the
/// per-route histogram, and the thread-safe recorder. Rendering the Prometheus text itself
/// (`prometheusMetrics`/`appendHTTPRequestMetrics`) is `private` to
/// `OpenAIChatCompletionsHTTPHandler.swift` and therefore not reachable from this file even under
/// `@testable import` (Swift's `private` is file-scoped, not module-scoped) -- those render-shape
/// assertions live as end-to-end `/metrics` scrape tests in
/// `OpenAIChatCompletionsHTTPHandlerTests.swift` instead.
final class ServingHTTPMetricsTests: XCTestCase {

    // MARK: - Status class mapping

    func testStatusClassMapsKnownHundredsGroups() {
        XCTAssertEqual(servingHTTPMetricsStatusClass(200), "2xx")
        XCTAssertEqual(servingHTTPMetricsStatusClass(404), "4xx")
        XCTAssertEqual(servingHTTPMetricsStatusClass(503), "5xx")
    }

    func testStatusClassFallsBackToOtherOutsideOneToFiveHundreds() {
        XCTAssertEqual(servingHTTPMetricsStatusClass(99), "other")
        XCTAssertEqual(servingHTTPMetricsStatusClass(600), "other")
    }

    // MARK: - Route mapping

    func testRouteMapsUnmatchedSentinelToOther() {
        XCTAssertEqual(servingHTTPMetricsRoute("<unmatched>"), "other")
    }

    func testRouteLeavesKnownTemplatedRoutesUnchanged() {
        XCTAssertEqual(servingHTTPMetricsRoute("/v1/chat/completions"), "/v1/chat/completions")
    }

    // MARK: - Boundary formatting

    func testFormatBoundaryRendersWholeNumbersWithoutTrailingZero() {
        XCTAssertEqual(servingHTTPMetricsFormatBoundary(1), "1")
        XCTAssertEqual(servingHTTPMetricsFormatBoundary(300), "300")
    }

    func testFormatBoundaryRendersFractionsViaShortestDescription() {
        XCTAssertEqual(servingHTTPMetricsFormatBoundary(0.005), "0.005")
        XCTAssertEqual(servingHTTPMetricsFormatBoundary(2.5), "2.5")
    }

    // MARK: - Histogram

    func testHistogramObserveAtExactBoundaryLandsInThatBucket() throws {
        var histogram = ServingHTTPMetricsHistogram()
        histogram.observe(1)

        let index = try XCTUnwrap(servingHTTPMetricsHistogramBuckets.firstIndex(of: 1))
        XCTAssertEqual(histogram.bucketCounts[index], 1)
        for otherIndex in histogram.bucketCounts.indices where otherIndex != index {
            XCTAssertEqual(histogram.bucketCounts[otherIndex], 0, "bucket \(otherIndex) must stay empty")
        }
        XCTAssertEqual(histogram.count, 1)
        XCTAssertEqual(histogram.sum, 1)
    }

    func testHistogramObserveAboveHighestBoundaryIncrementsNoRegularBucketButStillCountsAndSums() {
        var histogram = ServingHTTPMetricsHistogram()
        histogram.observe(301)

        XCTAssertEqual(histogram.bucketCounts, Array(repeating: 0, count: servingHTTPMetricsHistogramBuckets.count))
        XCTAssertEqual(histogram.count, 1)
        XCTAssertEqual(histogram.sum, 301)
        // The synthetic `+Inf` bucket is rendered as `count` directly (see the histogram's own doc
        // comment) -- it still equals 1 even though every REGULAR bucket stayed at 0.
        XCTAssertEqual(histogram.count, 1)
    }

    func testHistogramCumulativeBucketCountsAreMonotoneNonDecreasingAndBoundedByCount() {
        var histogram = ServingHTTPMetricsHistogram()
        // 0.001 -> the 0.005 bucket, 0.2 -> 0.25, 2 -> 2.5, 50 -> 60, 301 -> no regular bucket.
        for value in [0.001, 0.2, 2, 50, 301] {
            histogram.observe(value)
        }

        let cumulative = histogram.cumulativeBucketCounts
        for (previous, current) in zip(cumulative, cumulative.dropFirst()) {
            XCTAssertLessThanOrEqual(previous, current, "cumulative bucket counts must never decrease")
        }
        XCTAssertEqual(histogram.count, 5)
        XCTAssertLessThanOrEqual(
            cumulative.last ?? 0, histogram.count,
            "the last cumulative regular bucket can be strictly less than count (an above-range value)")
        XCTAssertEqual(cumulative.last, 4, "the 301 observation falls outside every regular bucket")
    }

    func testHistogramPlusInfinityBucketEqualsCountWhenEveryValueIsWithinRange() {
        var histogram = ServingHTTPMetricsHistogram()
        histogram.observe(0.001)
        histogram.observe(2)

        XCTAssertEqual(histogram.count, 2)
        XCTAssertEqual(
            histogram.cumulativeBucketCounts.last, histogram.count,
            "with no above-range observation, the last regular bucket already equals +Inf/count")
    }

    // MARK: - Recorder

    func testRecorderCountersAreKeyedByRouteStatusClassAndOutcome() {
        let recorder = ServingHTTPMetricsRecorder()
        recorder.record(
            route: "/v1/chat/completions", status: 200, outcome: "completed",
            durationSeconds: 0.01, ttftSeconds: nil)
        recorder.record(
            route: "/v1/chat/completions", status: 200, outcome: "completed",
            durationSeconds: 0.02, ttftSeconds: nil)
        recorder.record(
            route: "/v1/chat/completions", status: 500, outcome: "error",
            durationSeconds: 0.01, ttftSeconds: nil)
        recorder.record(
            route: "<unmatched>", status: 404, outcome: "error",
            durationSeconds: 0.01, ttftSeconds: nil)

        let counters = recorder.snapshot().counters
        XCTAssertEqual(counters.count, 3, "exactly three distinct (route, statusClass, outcome) keys")
        XCTAssertEqual(
            counters.first {
                $0.route == "/v1/chat/completions" && $0.statusClass == "2xx" && $0.outcome == "completed"
            }?.count,
            2)
        XCTAssertEqual(
            counters.first {
                $0.route == "/v1/chat/completions" && $0.statusClass == "5xx" && $0.outcome == "error"
            }?.count,
            1)
        XCTAssertEqual(
            counters.first { $0.route == "other" && $0.statusClass == "4xx" && $0.outcome == "error" }?.count,
            1,
            "the raw `<unmatched>` route collapses to `other`, same as `servingHTTPMetricsRoute`")
    }

    func testRecorderDoesNotCreateTTFTHistogramWhenTTFTIsNil() {
        let recorder = ServingHTTPMetricsRecorder()
        recorder.record(
            route: "/v1/chat/completions", status: 200, outcome: "completed",
            durationSeconds: 0.01, ttftSeconds: nil)
        XCTAssertTrue(recorder.snapshot().ttftHistograms.isEmpty)
        XCTAssertEqual(recorder.snapshot().durationHistograms.count, 1, "duration is still always observed")

        recorder.record(
            route: "/v1/chat/completions", status: 200, outcome: "completed",
            durationSeconds: 0.5, ttftSeconds: 0.05)
        let ttftHistograms = recorder.snapshot().ttftHistograms
        XCTAssertEqual(ttftHistograms.count, 1)
        XCTAssertEqual(ttftHistograms.first?.route, "/v1/chat/completions")
        XCTAssertEqual(ttftHistograms.first?.histogram.count, 1)
    }

    func testRecorderSnapshotOrderingIsDeterministicByRouteThenStatusClassThenOutcome() {
        func populate(_ recorder: ServingHTTPMetricsRecorder, insertionOrder: [(String, Int, String)]) {
            for (route, status, outcome) in insertionOrder {
                recorder.record(
                    route: route, status: status, outcome: outcome,
                    durationSeconds: 0.01, ttftSeconds: nil)
            }
        }

        let entries: [(String, Int, String)] = [
            ("/v1/models/{id}", 200, "completed"),
            ("/v1/chat/completions", 500, "error"),
            ("/v1/chat/completions", 200, "completed"),
            ("other", 404, "error"),
        ]
        let expectedOrder = [
            "/v1/chat/completions|2xx|completed",
            "/v1/chat/completions|5xx|error",
            "/v1/models/{id}|2xx|completed",
            "other|4xx|error",
        ]

        let recorder = ServingHTTPMetricsRecorder()
        populate(recorder, insertionOrder: entries)
        XCTAssertEqual(
            recorder.snapshot().counters.map { "\($0.route)|\($0.statusClass)|\($0.outcome)" },
            expectedOrder)

        // A second, independently-populated recorder fed the SAME entries in a different insertion
        // order renders the identical ordering -- proving it's a sort, not an insertion-order echo.
        let reordered = ServingHTTPMetricsRecorder()
        populate(reordered, insertionOrder: entries.reversed())
        XCTAssertEqual(
            reordered.snapshot().counters.map { "\($0.route)|\($0.statusClass)|\($0.outcome)" },
            expectedOrder)
    }

    func testRecorderConcurrentRecordFromManyTasksTotalsExactly() async {
        let recorder = ServingHTTPMetricsRecorder()
        let taskCount = 1_000
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<taskCount {
                group.addTask {
                    recorder.record(
                        route: "/v1/chat/completions",
                        status: 200,
                        outcome: "completed",
                        durationSeconds: 0.002,
                        ttftSeconds: 0.001)
                }
            }
        }

        let snapshot = recorder.snapshot()
        XCTAssertEqual(snapshot.counters.count, 1)
        XCTAssertEqual(snapshot.counters.first?.count, taskCount, "every one of the 1000 records must land")
        XCTAssertEqual(snapshot.durationHistograms.first?.histogram.count, taskCount)
        XCTAssertEqual(snapshot.ttftHistograms.first?.histogram.count, taskCount)
        XCTAssertEqual(
            snapshot.durationHistograms.first?.histogram.sum ?? 0,
            Double(taskCount) * 0.002,
            accuracy: 1e-9)
        XCTAssertEqual(
            snapshot.ttftHistograms.first?.histogram.sum ?? 0,
            Double(taskCount) * 0.001,
            accuracy: 1e-9)
    }
}
