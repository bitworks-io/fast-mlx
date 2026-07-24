import XCTest

@testable import HarnessCore

final class ServiceBenchMetricsTests: XCTestCase {
    private func speculationSnapshot(
        requestedRequests: Int = 0,
        activeSessions: Int = 0,
        draftedTokens: Int = 0,
        acceptedDraftTokens: Int = 0,
        verificationRounds: Int = 0,
        fallbackRounds: Int = 0
    ) -> ContinuousBatchRuntimeSpeculationSnapshot {
        ContinuousBatchRuntimeSpeculationSnapshot(
            requestedRequests: requestedRequests,
            activeSessions: activeSessions,
            draftedTokens: draftedTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            verificationRounds: verificationRounds,
            fallbackRounds: fallbackRounds)
    }

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
        XCTAssertFalse(summary.speculationAllowed)
        XCTAssertFalse(summary.speculationEngaged)
    }

    func testSeparatesSpeculationPermissionFromActualRuntimeEngagement() throws {
        let request = BatchRequestID(1)
        let allowedOnly = summarizeServiceOperations([
            ServiceTickObservation(
                activeSlots: 1,
                queuedSlots: 0,
                operations: [
                    .decode(.solo(request, speculationAllowed: true))
                ])
        ])

        XCTAssertTrue(allowedOnly.speculationAllowed)
        XCTAssertFalse(allowedOnly.speculationEngaged)

        let engaged = summarizeServiceOperations(
            [
                ServiceTickObservation(
                    activeSlots: 1,
                    queuedSlots: 0,
                    operations: [
                        .decode(.solo(request, speculationAllowed: true))
                    ])
            ],
            speculationStartSnapshot: speculationSnapshot(),
            speculationEndSnapshot: ContinuousBatchRuntimeSpeculationSnapshot(
                requestedRequests: 1,
                activeSessions: 0,
                draftedTokens: 3,
                acceptedDraftTokens: 2,
                verificationRounds: 1,
                fallbackRounds: 0))

        XCTAssertTrue(engaged.speculationAllowed)
        XCTAssertTrue(engaged.speculationEngaged)
    }

    func testDoesNotTreatHistoricalSpeculationCountersAsCurrentRunEngagement() {
        let summary = summarizeServiceOperations(
            [],
            speculationStartSnapshot: speculationSnapshot(
                requestedRequests: 4,
                draftedTokens: 12,
                acceptedDraftTokens: 8,
                verificationRounds: 3),
            speculationEndSnapshot: speculationSnapshot(
                requestedRequests: 4,
                draftedTokens: 12,
                acceptedDraftTokens: 8,
                verificationRounds: 3))

        XCTAssertFalse(summary.speculationEngaged)
    }

    func testSpeculationIntervalFailsClosedForCounterRegressionAndPresenceMismatch() {
        XCTAssertFalse(
            speculationEngagedDuringInterval(
                from: nil,
                to: speculationSnapshot(
                    requestedRequests: 1,
                    draftedTokens: 4,
                    verificationRounds: 1)))
        XCTAssertFalse(
            speculationEngagedDuringInterval(
                from: speculationSnapshot(
                    requestedRequests: 1,
                    draftedTokens: 4,
                    verificationRounds: 1),
                to: nil))
        XCTAssertFalse(
            speculationEngagedDuringInterval(
                from: speculationSnapshot(
                    requestedRequests: 2,
                    draftedTokens: 6,
                    acceptedDraftTokens: 4,
                    verificationRounds: 2),
                to: speculationSnapshot(
                    requestedRequests: 3,
                    draftedTokens: 5,
                    acceptedDraftTokens: 4,
                    verificationRounds: 3)))
    }

    func testSpeculationIntervalTreatsActiveSessionsAsGaugeOnly() {
        XCTAssertTrue(
            speculationEngagedDuringInterval(
                from: speculationSnapshot(
                    requestedRequests: 4,
                    activeSessions: 3,
                    draftedTokens: 10,
                    acceptedDraftTokens: 7,
                    verificationRounds: 2),
                to: speculationSnapshot(
                    requestedRequests: 5,
                    activeSessions: 0,
                    draftedTokens: 12,
                    acceptedDraftTokens: 8,
                    verificationRounds: 3)))
    }

    func testDecodesHistoricalSpeculationFieldAsPermissionNotEngagement() throws {
        let historical = """
            {
              "tickCount": 1,
              "maxActiveSlots": 1,
              "meanActiveSlots": 1,
              "maxQueuedSlots": 0,
              "decodeBatchSizeHistogram": {"1": 1},
              "promptChunkCount": 1,
              "promptTokensProcessed": 4,
              "drainCount": 0,
              "speculationEngaged": true
            }
            """

        let decoded = try JSONDecoder().decode(
            ServiceOperationSummary.self,
            from: Data(historical.utf8))

        XCTAssertTrue(decoded.speculationAllowed)
        XCTAssertFalse(decoded.speculationEngaged)
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

    func testCancellationGateUsesWorstRemovalLatencyAgainstKeepalive() throws {
        let passing = try evaluateCancellationGate(
            [
                ServiceCancellationTimeline(requestedAt: 1, removedAt: 1.05),
                ServiceCancellationTimeline(requestedAt: 2, removedAt: 2.2),
            ],
            keepaliveSeconds: 0.21)
        XCTAssertTrue(passing.withinKeepalive)
        XCTAssertEqual(passing.summary.maxSeconds, 0.2, accuracy: 1e-12)

        let failing = try evaluateCancellationGate(
            [ServiceCancellationTimeline(requestedAt: 4, removedAt: 4.21)],
            keepaliveSeconds: 0.2)
        XCTAssertFalse(failing.withinKeepalive)

        XCTAssertThrowsError(
            try evaluateCancellationGate(
                [ServiceCancellationTimeline(requestedAt: 1, removedAt: 1.1)],
                keepaliveSeconds: 0)
        ) {
            XCTAssertEqual($0 as? ServiceBenchMetricsError, .invalidKeepaliveSeconds(0))
        }
    }

    func testStatePoisonGateRequiresNonEmptyByteIdenticalRecoveryOutputs() {
        let before = [Array("alpha".utf8), Array("beta".utf8)]
        let passing = evaluateServiceStatePoisonRecovery(before: before, after: before)
        XCTAssertTrue(passing.requestCountsMatch)
        XCTAssertTrue(passing.outputsNonEmpty)
        XCTAssertEqual(passing.perRequestByteMatch, [true, true])
        XCTAssertTrue(passing.byteIdentical)
        XCTAssertTrue(passing.passed)
        XCTAssertEqual(passing.beforeOutputHashes, passing.afterOutputHashes)

        let drifted = evaluateServiceStatePoisonRecovery(
            before: before,
            after: [Array("alpha".utf8), Array("changed".utf8)])
        XCTAssertEqual(drifted.perRequestByteMatch, [true, false])
        XCTAssertFalse(drifted.byteIdentical)
        XCTAssertFalse(drifted.passed)

        let missing = evaluateServiceStatePoisonRecovery(
            before: before,
            after: [Array("alpha".utf8)])
        XCTAssertFalse(missing.requestCountsMatch)
        XCTAssertEqual(missing.perRequestByteMatch, [true, false])
        XCTAssertFalse(missing.passed)

        let empty = evaluateServiceStatePoisonRecovery(
            before: [Array("alpha".utf8), []],
            after: [Array("alpha".utf8), []])
        XCTAssertFalse(empty.outputsNonEmpty)
        XCTAssertFalse(empty.passed)
    }

    func testSoakGateUsesPostWarmupRSSAndSeparateResponsiveness() throws {
        func sample(_ timestamp: Double, _ rss: UInt64) -> ServiceMemorySample {
            ServiceMemorySample(
                timestamp: timestamp,
                physicalFootprintBytes: rss,
                mlxActiveBytes: 10,
                mlxCacheBytes: 5,
                mlxPeakBytes: 15)
        }
        let passing = try evaluateServiceSoakGate(
            memorySamples: [sample(0, 100), sample(1, 120), sample(2, 124), sample(3, 123)],
            cyclePassed: [true, true, true],
            responsivenessSeconds: [0.5, 0.7, 0.6],
            maxRSSDriftPercent: 5,
            responsivenessLimitSeconds: 1)
        XCTAssertEqual(passing.baselineSampleIndex, 1)
        XCTAssertEqual(passing.measuredCycleCount, 2)
        XCTAssertEqual(passing.maxRSSDriftPercent, 100.0 / 30.0, accuracy: 1e-12)
        XCTAssertTrue(passing.rssWithinLimit)
        XCTAssertTrue(passing.responsive)
        XCTAssertTrue(passing.allCyclesPassed)
        XCTAssertTrue(passing.passed)

        let bounded = try evaluateServiceSoakSummary(
            cycleCount: 4,
            allCyclesPassed: true,
            baselineRSSBytes: 120,
            endRSSBytes: 123,
            maxRSSBytes: 130,
            maxResponsivenessSeconds: 0.7,
            maxRSSDriftPercent: 5,
            responsivenessLimitSeconds: 1)
        XCTAssertEqual(bounded.measuredCycleCount, 3)
        XCTAssertEqual(bounded.maxRSSBytes, 130)
        XCTAssertFalse(bounded.rssWithinLimit)
        XCTAssertFalse(bounded.passed)

        let stalePeak = try evaluateServiceSoakSummary(
            cycleCount: 3,
            allCyclesPassed: true,
            baselineRSSBytes: 100,
            endRSSBytes: 120,
            maxRSSBytes: 102,
            maxResponsivenessSeconds: 0.5,
            maxRSSDriftPercent: 5,
            responsivenessLimitSeconds: 1)
        XCTAssertEqual(stalePeak.maxRSSBytes, 120)
        XCTAssertEqual(stalePeak.maxRSSDriftPercent, 20, accuracy: 1e-12)
        XCTAssertFalse(stalePeak.rssWithinLimit)
        XCTAssertFalse(stalePeak.passed)

        let failing = try evaluateServiceSoakGate(
            memorySamples: [sample(0, 100), sample(1, 120), sample(2, 130)],
            cyclePassed: [true, false],
            responsivenessSeconds: [0.5, 1.1],
            maxRSSDriftPercent: 5,
            responsivenessLimitSeconds: 1)
        XCTAssertFalse(failing.rssWithinLimit)
        XCTAssertFalse(failing.responsive)
        XCTAssertFalse(failing.allCyclesPassed)
        XCTAssertFalse(failing.passed)

        XCTAssertThrowsError(
            try evaluateServiceSoakGate(
                memorySamples: [sample(0, 100), sample(1, 120)],
                cyclePassed: [true],
                responsivenessSeconds: [],
                maxRSSDriftPercent: 5,
                responsivenessLimitSeconds: 1))
    }

    func testNormalizesTerminalEOSWithoutCountingItAsVisibleServiceOutput() throws {
        let normalized = try normalizeVisibleServiceTokens(
            tokens: [10, 11, 2],
            tokenTimes: [1, 2, 3],
            eosToken: 2)
        XCTAssertEqual(normalized.tokens, [10, 11])
        XCTAssertEqual(normalized.tokenTimes, [1, 2])

        XCTAssertThrowsError(
            try normalizeVisibleServiceTokens(
                tokens: [10, 11], tokenTimes: [1], eosToken: 2)
        ) {
            XCTAssertEqual(
                $0 as? ServiceBenchMetricsError,
                .tokenTimestampCountMismatch(tokens: 2, timestamps: 1))
        }
    }
}
