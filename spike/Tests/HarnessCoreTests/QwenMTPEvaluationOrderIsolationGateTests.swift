import XCTest
@testable import HarnessCore

final class QwenMTPEvaluationOrderIsolationGateTests: XCTestCase {
    func testGatePredeclaresLongRetrievalPairPlanAndThresholds() {
        XCTAssertEqual(QwenMTPEvaluationOrderIsolationGate.schemaVersion, 1)
        XCTAssertEqual(QwenMTPEvaluationOrderIsolationGate.droppedWarmupPairs, 2)
        XCTAssertEqual(QwenMTPEvaluationOrderIsolationGate.measuredPairs, 5)
        XCTAssertEqual(QwenMTPEvaluationOrderIsolationGate.requiredPromptTokenCount, 1_353)
        XCTAssertEqual(
            QwenMTPEvaluationOrderIsolationGate.pairOrders,
            [
                .cacheFirstThenHiddenFirst,
                .hiddenFirstThenCacheFirst,
                .cacheFirstThenHiddenFirst,
                .hiddenFirstThenCacheFirst,
                .cacheFirstThenHiddenFirst,
                .hiddenFirstThenCacheFirst,
                .cacheFirstThenHiddenFirst,
            ])
        XCTAssertEqual(
            QwenMTPEvaluationOrderIsolationGate.requiredAggregatePromptImprovementSeconds,
            2.5)
        XCTAssertEqual(
            QwenMTPEvaluationOrderIsolationGate.requiredMedianPromptImprovementSeconds,
            0.40)
    }

    func testValidateQualifiesOnlyWhenBothFrozenImprovementThresholdsPass() throws {
        let verdict = try QwenMTPEvaluationOrderIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.41, 0.45, 0.52, 0.58, 0.61]))

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(verdict.aggregatePromptImprovementSeconds, 2.57, accuracy: 1e-12)
        XCTAssertEqual(verdict.medianPromptImprovementSeconds, 0.52, accuracy: 1e-12)

        let aggregateMiss = try QwenMTPEvaluationOrderIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.40, 0.40, 0.40, 0.40, 0.40]))
        XCTAssertFalse(aggregateMiss.qualified)
        XCTAssertEqual(aggregateMiss.aggregatePromptImprovementSeconds, 2.0, accuracy: 1e-12)

        let medianMiss = try QwenMTPEvaluationOrderIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.10, 0.10, 0.10, 1.20, 1.20]))
        XCTAssertFalse(medianMiss.qualified)
        XCTAssertEqual(medianMiss.aggregatePromptImprovementSeconds, 2.70, accuracy: 1e-12)
        XCTAssertEqual(medianMiss.medianPromptImprovementSeconds, 0.10, accuracy: 1e-12)
    }

    func testValidateRejectsWrongPairShapeOrderAndReleaseMode() {
        var missingPair = makePayload()
        missingPair.pairs.removeLast()
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(missingPair))

        var wrongOrder = makePayload()
        wrongOrder.pairs[0] = wrongOrder.pairs[0].copy(
            runOrder: .hiddenFirstThenCacheFirst)
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(wrongOrder))

        var nonRelease = makePayload()
        nonRelease.releaseBuildObserved = false
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(nonRelease))
    }

    func testValidateRejectsNonExactOrIncompleteUserOutcome() {
        var tokenMismatch = makePayload()
        tokenMismatch.pairs[2] = tokenMismatch.pairs[2].copy(
            hiddenFirst: tokenMismatch.pairs[2].hiddenFirst.copy(
                exactness: exactness(mtpTokenDigest: hex("different"))))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(tokenMismatch))

        var wrongStop = makePayload()
        wrongStop.pairs[2] = wrongStop.pairs[2].copy(
            cacheFirst: wrongStop.pairs[2].cacheFirst.copy(
                exactness: exactness(mtpStop: .stop)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(wrongStop))

        var shortCache = makePayload()
        shortCache.pairs[2] = shortCache.pairs[2].copy(
            hiddenFirst: shortCache.pairs[2].hiddenFirst.copy(
                exactness: exactness(cacheEntryCount: 31)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(shortCache))
    }

    func testValidateRejectsPassthroughOrIncoherentDraftCounters() {
        var passthrough = makePayload()
        passthrough.pairs[2] = passthrough.pairs[2].copy(
            hiddenFirst: passthrough.pairs[2].hiddenFirst.copy(
                passthroughReason: "fallback"))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(passthrough))

        var counters = makePayload()
        counters.pairs[2] = counters.pairs[2].copy(
            cacheFirst: counters.pairs[2].cacheFirst.copy(
                telemetry: telemetry(proposed: 4, accepted: 5)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(counters))
    }

    func testValidateRejectsMissingWrongOrderOrForgedPromptAttribution() {
        var missing = makePayload()
        missing.pairs[2] = missing.pairs[2].copy(
            hiddenFirst: missing.pairs[2].hiddenFirst.copy(
                phaseAttribution: phaseAttribution(includePreparation: false)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(missing))

        var wrongOrder = makePayload()
        wrongOrder.pairs[2] = wrongOrder.pairs[2].copy(
            hiddenFirst: wrongOrder.pairs[2].hiddenFirst.copy(
                evaluationOrder: .cacheFirst))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(wrongOrder))

        var forgedEnvelope = makePayload()
        forgedEnvelope.pairs[2] = forgedEnvelope.pairs[2].copy(
            hiddenFirst: forgedEnvelope.pairs[2].hiddenFirst.copy(
                phaseAttribution: phaseAttribution(targetPrefillResidualSeconds: 0.8)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(forgedEnvelope))

        var wrongPrompt = makePayload()
        wrongPrompt.pairs[2] = wrongPrompt.pairs[2].copy(
            hiddenFirst: wrongPrompt.pairs[2].hiddenFirst.copy(
                phaseAttribution: phaseAttribution(promptTokenCount: 1_352)))
        XCTAssertThrowsError(try QwenMTPEvaluationOrderIsolationGate.validate(wrongPrompt))

        var malformedHiddenBytes = makePayload()
        malformedHiddenBytes.pairs[2] = malformedHiddenBytes.pairs[2].copy(
            hiddenFirst: malformedHiddenBytes.pairs[2].hiddenFirst.copy(
                phaseAttribution: phaseAttribution(hiddenByteCount: 10_825)))
        XCTAssertThrowsError(
            try QwenMTPEvaluationOrderIsolationGate.validate(malformedHiddenBytes))
    }

    private func makePayload(
        improvements: [Double] = [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
    ) -> QwenMTPEvaluationOrderIsolationPayload {
        QwenMTPEvaluationOrderIsolationPayload(
            schemaVersion: QwenMTPEvaluationOrderIsolationGate.schemaVersion,
            corpusID: QwenMTPEvaluationOrderIsolationGate.corpusID,
            corpusContentHash: QwenMTPEvaluationOrderIsolationGate.corpusContentHash,
            binding: QwenMTPCorpusGate.requiredBinding,
            host: .init(chip: "Apple M5", ramBytes: 24 * 1024 * 1024 * 1024, os: "macOS 26.0"),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            pairs: QwenMTPEvaluationOrderIsolationGate.pairOrders.enumerated().map {
                index, order in
                let cachePrompt = 4.0
                let hiddenPrompt = cachePrompt - improvements[index]
                return .init(
                    pairIndex: index,
                    warmup: index < QwenMTPEvaluationOrderIsolationGate.droppedWarmupPairs,
                    runOrder: order,
                    cacheFirst: run(
                        order: .cacheFirst,
                        promptSeconds: cachePrompt),
                    hiddenFirst: run(
                        order: .hiddenFirst,
                        promptSeconds: hiddenPrompt))
            })
    }

    private func run(
        order: QwenMTPPromptEvaluationOrderEvidence,
        promptSeconds: Double
    ) -> QwenMTPEvaluationOrderRunEvidence {
        .init(
            evaluationOrder: order,
            timing: .init(
                promptSeconds: promptSeconds,
                generationSeconds: 2,
                wallSeconds: promptSeconds + 2,
                e2eSeconds: promptSeconds + 2),
            exactness: exactness(),
            telemetry: telemetry(),
            phaseAttribution: phaseAttribution(promptSeconds: promptSeconds),
            passthroughReason: nil)
    }

    private func exactness(
        mtpTokenDigest: String? = nil,
        mtpStop: QwenMTPCorpusStopOutcome = .length,
        cacheEntryCount: Int = 32
    ) -> QwenMTPCorpusExactnessEvidence {
        let token = hex("token")
        let decoded = hex("decoded")
        let cache = QwenMTPCorpusCacheFingerprint(
            digest: hex("cache"),
            entries: (0..<cacheEntryCount).map { index in
                .init(
                    layerIndex: index,
                    cacheType: "MambaCache",
                    offset: 1_480,
                    metaStateSHA256: hex("meta-\(index)"),
                    stateCount: 1,
                    states: [
                        .init(
                            stateIndex: 0,
                            shape: [1, 1],
                            dtype: "float16",
                            byteCount: 2,
                            sha256: hex("state-\(index)")),
                    ])
            })
        return .init(
            scalarTokenCount: 128,
            mtpTokenCount: 128,
            scalarTokenIDsSHA256: token,
            mtpTokenIDsSHA256: mtpTokenDigest ?? token,
            scalarDecodedBytesSHA256: decoded,
            mtpDecodedBytesSHA256: decoded,
            scalarStopOutcome: .length,
            mtpStopOutcome: mtpStop,
            scalarCacheFingerprint: cache,
            mtpCacheFingerprint: cache,
            firstCacheMismatch: nil)
    }

    private func telemetry(
        proposed: Int = 20,
        accepted: Int = 10
    ) -> QwenMTPCorpusMTPTelemetry {
        .init(
            proposedDraftTokens: proposed,
            acceptedDraftTokens: accepted,
            rejectedDraftTokens: max(0, proposed - accepted),
            roundCount: 10,
            targetModelCallCount: 10,
            draftModelCallCount: 10,
            targetVerifiedTokenCount: 30,
            emittedTokenCount: 128)
    }

    private func phaseAttribution(
        includePreparation: Bool = true,
        promptSeconds: Double = 4,
        targetPrefillResidualSeconds: Double? = nil,
        promptTokenCount: Int = 1_353,
        hiddenByteCount: Int = 10_824
    ) -> QwenMTPCorpusMTPPhaseAttribution {
        let targetPrefillSeconds = promptSeconds - 1
        let residual = targetPrefillResidualSeconds
            ?? (targetPrefillSeconds - 1.05)
        return .init(
            targetPrefillSeconds: targetPrefillSeconds,
            drafterPromptPrimingSeconds: 1,
            draftBlockSeconds: 0.5,
            targetVerificationSeconds: 1,
            targetTailSeconds: 0,
            hybridRewindReplaySeconds: 0.25,
            finalizationSeconds: 0.25,
            cacheFingerprintSeconds: 0.1,
            targetPrefillCount: 1,
            drafterPromptPrimingCount: 1,
            draftBlockCount: 10,
            targetVerificationCount: 10,
            targetTailCount: 0,
            hybridRewindReplayCount: 10,
            finalizationCount: 1,
            cacheFingerprintCount: 1,
            targetPromptPreparation: includePreparation
                ? .init(
                    promptTokenCount: promptTokenCount,
                    hiddenShape: [1, promptTokenCount, 4],
                    hiddenByteCount: hiddenByteCount,
                    chunks: [
                        .init(
                            tokenOffset: 0,
                            tokenCount: promptTokenCount,
                            targetForwardSchedulingSeconds: 0.1),
                    ],
                    cacheEvaluationSeconds: 0.5,
                    hiddenEvaluationSeconds: 0.25,
                    concatenatedHiddenEvaluationSeconds: 0.1,
                    preparedCacheHandoffSeconds: 0.05,
                    phaseBoundarySynchronizationSeconds: 0.05,
                    targetPrefillResidualSeconds: residual)
                : nil)
    }

    private func hex(_ seed: String) -> String {
        let bytes = Array(seed.utf8)
        return (0..<64).map { index in
            String(format: "%x", bytes[index % bytes.count] & 0x0f)
        }.joined()
    }
}
