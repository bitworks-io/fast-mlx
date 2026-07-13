import XCTest

@testable import HarnessCore

final class ServiceBenchMetricsTests: XCTestCase {
    private func timeline(
        _ id: UInt64,
        submittedAt: Double,
        tokenTimes: [Double],
        completedAt: Double
    ) -> ServiceRequestTimeline {
        ServiceRequestTimeline(
            requestID: BatchRequestID(id),
            promptTokenCount: 4,
            submittedAt: submittedAt,
            tokenTimes: tokenTimes,
            completedAt: completedAt)
    }

    func testMeasuresAggregateWallTimeAndPerRequestLatencyDistributions() throws {
        let metrics = try measureServiceRun([
            timeline(1, submittedAt: 0, tokenTimes: [2, 4, 6], completedAt: 6),
            timeline(2, submittedAt: 0, tokenTimes: [1, 3], completedAt: 4),
        ])

        XCTAssertEqual(metrics.totalOutputTokens, 5)
        XCTAssertEqual(metrics.aggregateTokensPerSecond, 5.0 / 6.0, accuracy: 1e-12)
        XCTAssertEqual(metrics.ttft.p50Seconds, 2, accuracy: 1e-12)
        XCTAssertEqual(metrics.ttft.p95Seconds, 2, accuracy: 1e-12)
        let tpot = try XCTUnwrap(metrics.tpot)
        XCTAssertEqual(tpot.p50Seconds, 2, accuracy: 1e-12)
        XCTAssertEqual(tpot.p95Seconds, 2, accuracy: 1e-12)
        XCTAssertEqual(metrics.completion.p50Seconds, 6, accuracy: 1e-12)
        XCTAssertEqual(metrics.completion.p95Seconds, 6, accuracy: 1e-12)
        XCTAssertEqual(metrics.jainCompletionRate, 1, accuracy: 1e-12)
        XCTAssertEqual(metrics.requests.map(\.outputTokenCount), [3, 2])
        XCTAssertEqual(metrics.requests.map(\.completionTokensPerSecond), [0.5, 0.5])
    }

    func testSingleTokenRequestHasNoTPOTAndSkewLowersJainFairness() throws {
        let metrics = try measureServiceRun([
            timeline(1, submittedAt: 0, tokenTimes: [1], completedAt: 1),
            timeline(2, submittedAt: 0, tokenTimes: [4], completedAt: 4),
        ])

        XCTAssertNil(metrics.tpot)
        XCTAssertTrue(metrics.requests.allSatisfy { $0.tpotSeconds == nil })
        XCTAssertEqual(metrics.jainCompletionRate, 1.5625 / 2.125, accuracy: 1e-12)
    }

    func testAggregatesTimedRunsWithoutDroppingRawRequestDistributions() throws {
        let first = try measureServiceRun([
            timeline(1, submittedAt: 0, tokenTimes: [1, 2], completedAt: 2),
            timeline(2, submittedAt: 0, tokenTimes: [2, 4], completedAt: 4),
        ])
        let second = try measureServiceRun([
            timeline(3, submittedAt: 10, tokenTimes: [11, 13], completedAt: 13),
        ])

        let aggregate = try aggregateServiceRuns([first, second])

        XCTAssertEqual(aggregate.runCount, 2)
        XCTAssertEqual(aggregate.requestCount, 3)
        XCTAssertEqual(
            aggregate.meanAggregateTokensPerSecond,
            (1 + (2.0 / 3.0)) / 2,
            accuracy: 1e-12)
        XCTAssertEqual(aggregate.ttft.p50Seconds, 1, accuracy: 1e-12)
        XCTAssertEqual(aggregate.ttft.p95Seconds, 2, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(aggregate.tpot).p95Seconds, 2, accuracy: 1e-12)
        XCTAssertEqual(
            aggregate.minJainCompletionRate,
            first.jainCompletionRate,
            accuracy: 1e-12)
    }

    func testRejectsEmptyAndMalformedTimelinesRatherThanReportingVacuousMetrics() {
        XCTAssertThrowsError(try measureServiceRun([])) {
            XCTAssertEqual($0 as? ServiceBenchMetricsError, .emptyRun)
        }
        XCTAssertThrowsError(
            try measureServiceRun([
                timeline(1, submittedAt: 0, tokenTimes: [], completedAt: 1)
            ])
        ) {
            XCTAssertEqual(
                $0 as? ServiceBenchMetricsError,
                .emptyTokenStream(BatchRequestID(1)))
        }
        XCTAssertThrowsError(
            try measureServiceRun([
                timeline(2, submittedAt: 0, tokenTimes: [2, 1], completedAt: 3)
            ])
        ) {
            XCTAssertEqual(
                $0 as? ServiceBenchMetricsError,
                .nonMonotonicTokenTimes(BatchRequestID(2)))
        }
        XCTAssertThrowsError(
            try measureServiceRun([
                timeline(3, submittedAt: 2, tokenTimes: [1], completedAt: 3)
            ])
        ) {
            XCTAssertEqual(
                $0 as? ServiceBenchMetricsError,
                .tokenBeforeSubmission(BatchRequestID(3)))
        }
    }

    func testSummarizesBatchOccupancyPromptChunksAndActiveSlots() {
        let one = BatchRequestID(1)
        let two = BatchRequestID(2)
        let three = BatchRequestID(3)
        let summary = summarizeServiceOperations([
            ServiceTickObservation(
                activeSlots: 1,
                queuedSlots: 2,
                operations: [
                    .prefill(BatchPrefillSlice(id: one, startToken: 0, count: 16))
                ]),
            ServiceTickObservation(
                activeSlots: 3,
                queuedSlots: 0,
                operations: [
                    .decode(.solo(one, speculationAllowed: false)),
                    .decode(.batch([one, two, three], speculationAllowed: false)),
                ]),
            ServiceTickObservation(
                activeSlots: 2,
                queuedSlots: 0,
                operations: [
                    .decode(.drainSoloPipeline(one))
                ]),
        ])

        XCTAssertEqual(summary.tickCount, 3)
        XCTAssertEqual(summary.maxActiveSlots, 3)
        XCTAssertEqual(summary.meanActiveSlots, 2, accuracy: 1e-12)
        XCTAssertEqual(summary.maxQueuedSlots, 2)
        XCTAssertEqual(summary.decodeBatchSizeHistogram, [1: 1, 3: 1])
        XCTAssertEqual(summary.promptChunkCount, 1)
        XCTAssertEqual(summary.promptTokensProcessed, 16)
        XCTAssertEqual(summary.drainCount, 1)
        XCTAssertFalse(summary.speculationEngaged)
    }

    func testSummarizesCancellationLatencyAndMemoryDriftSeparately() throws {
        let cancellation = try summarizeCancellationLatencies([
            ServiceCancellationTimeline(requestedAt: 1, removedAt: 1.1),
            ServiceCancellationTimeline(requestedAt: 2, removedAt: 2.3),
        ])
        XCTAssertEqual(cancellation.p50Seconds, 0.3, accuracy: 1e-12)
        XCTAssertEqual(cancellation.p95Seconds, 0.3, accuracy: 1e-12)
        XCTAssertEqual(cancellation.maxSeconds, 0.3, accuracy: 1e-12)

        let memory = try summarizeServiceMemory([
            ServiceMemorySample(
                timestamp: 0, physicalFootprintBytes: 100,
                mlxActiveBytes: 60, mlxCacheBytes: 20, mlxPeakBytes: 80),
            ServiceMemorySample(
                timestamp: 1, physicalFootprintBytes: 160,
                mlxActiveBytes: 80, mlxCacheBytes: 30, mlxPeakBytes: 120),
            ServiceMemorySample(
                timestamp: 2, physicalFootprintBytes: 104,
                mlxActiveBytes: 62, mlxCacheBytes: 22, mlxPeakBytes: 120),
        ])
        XCTAssertEqual(memory.startFootprintBytes, 100)
        XCTAssertEqual(memory.endFootprintBytes, 104)
        XCTAssertEqual(memory.maxSampledFootprintBytes, 160)
        XCTAssertEqual(memory.endFootprintDriftPercent, 4, accuracy: 1e-12)
        XCTAssertEqual(memory.maxFootprintDriftPercent, 60, accuracy: 1e-12)
        XCTAssertEqual(memory.maxMLXPeakBytes, 120)
    }
}
