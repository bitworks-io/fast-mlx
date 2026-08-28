import XCTest
@testable import HarnessCore

final class QwenMTPCombinedEvaluationIsolationGateTests: XCTestCase {
    func testGateFreezesAlternatingLongRetrievalPlanAndThresholds() {
        XCTAssertEqual(QwenMTPCombinedEvaluationIsolationGate.schemaVersion, 1)
        XCTAssertEqual(QwenMTPCombinedEvaluationIsolationGate.droppedWarmupPairs, 2)
        XCTAssertEqual(QwenMTPCombinedEvaluationIsolationGate.measuredPairs, 5)
        XCTAssertEqual(QwenMTPCombinedEvaluationIsolationGate.requiredPromptTokenCount, 1_353)
        XCTAssertEqual(
            QwenMTPCombinedEvaluationIsolationGate.pairOrders,
            [
                .cacheFirstThenCombined,
                .combinedThenCacheFirst,
                .cacheFirstThenCombined,
                .combinedThenCacheFirst,
                .cacheFirstThenCombined,
                .combinedThenCacheFirst,
                .cacheFirstThenCombined,
            ])
        XCTAssertEqual(
            QwenMTPCombinedEvaluationIsolationGate
                .requiredAggregatePromptImprovementSeconds,
            2.5)
        XCTAssertEqual(
            QwenMTPCombinedEvaluationIsolationGate
                .requiredMedianPromptImprovementSeconds,
            0.40)
    }

    func testValidateQualifiesOnlyWhenBothFrozenThresholdsPass() throws {
        let verdict = try QwenMTPCombinedEvaluationIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.41, 0.45, 0.52, 0.58, 0.61]))
        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(verdict.aggregatePromptImprovementSeconds, 2.57, accuracy: 1e-12)
        XCTAssertEqual(verdict.medianPromptImprovementSeconds, 0.52, accuracy: 1e-12)

        let aggregateMiss = try QwenMTPCombinedEvaluationIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.40, 0.40, 0.40, 0.40, 0.40]))
        XCTAssertFalse(aggregateMiss.qualified)

        let medianMiss = try QwenMTPCombinedEvaluationIsolationGate.validate(
            makePayload(improvements: [0, 0, 0.10, 0.10, 0.10, 1.20, 1.20]))
        XCTAssertFalse(medianMiss.qualified)
    }

    func testValidateRejectsWrongShapeOrderOrReleaseMode() {
        var missing = makePayload()
        missing.pairs.removeLast()
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(missing))

        var wrongOrder = makePayload()
        wrongOrder.pairs[0] = wrongOrder.pairs[0].copy(
            runOrder: .combinedThenCacheFirst)
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(wrongOrder))

        var nonRelease = makePayload()
        nonRelease.releaseBuildObserved = false
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(nonRelease))
    }

    func testValidateRejectsWrongModeExactnessPassthroughOrPromptIdentity() {
        var wrongMode = makePayload()
        wrongMode.pairs[2] = wrongMode.pairs[2].copy(
            combined: wrongMode.pairs[2].combined.copy(evaluationOrder: .hiddenFirst))
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(wrongMode))

        var mismatch = makePayload()
        mismatch.pairs[2] = mismatch.pairs[2].copy(
            combined: mismatch.pairs[2].combined.copy(
                exactness: exactness(mtpTokenDigest: hex("mismatch"))))
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(mismatch))

        var passthrough = makePayload()
        passthrough.pairs[2] = passthrough.pairs[2].copy(
            combined: passthrough.pairs[2].combined.copy(
                passthroughReason: "fallback"))
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(passthrough))

        var wrongPrompt = makePayload()
        wrongPrompt.pairs[2] = wrongPrompt.pairs[2].copy(
            combined: wrongPrompt.pairs[2].combined.copy(
                phaseAttribution: phaseAttribution(promptTokenCount: 1_352)))
        XCTAssertThrowsError(try QwenMTPCombinedEvaluationIsolationGate.validate(wrongPrompt))

        var relabeledSeparateWaits = makePayload()
        relabeledSeparateWaits.pairs[2] = relabeledSeparateWaits.pairs[2].copy(
            combined: relabeledSeparateWaits.pairs[2].combined.copy(
                phaseAttribution: phaseAttribution(
                    promptSeconds:
                        relabeledSeparateWaits.pairs[2].combined.timing.promptSeconds,
                    hiddenEvaluationSeconds: 0.10)))
        XCTAssertThrowsError(
            try QwenMTPCombinedEvaluationIsolationGate.validate(
                relabeledSeparateWaits))
    }

    private func makePayload(
        improvements: [Double] = [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
    ) -> QwenMTPCombinedEvaluationIsolationPayload {
        QwenMTPCombinedEvaluationIsolationPayload(
            schemaVersion: QwenMTPCombinedEvaluationIsolationGate.schemaVersion,
            corpusID: QwenMTPCombinedEvaluationIsolationGate.corpusID,
            corpusContentHash: QwenMTPCombinedEvaluationIsolationGate.corpusContentHash,
            binding: QwenMTPCorpusGate.requiredBinding,
            host: .init(
                chip: "Apple M5",
                ramBytes: 24 * 1024 * 1024 * 1024,
                os: "macOS 26.0"),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            pairs: QwenMTPCombinedEvaluationIsolationGate.pairOrders.enumerated().map {
                index, order in
                let cachePrompt = 4.0
                return .init(
                    pairIndex: index,
                    warmup: index < QwenMTPCombinedEvaluationIsolationGate.droppedWarmupPairs,
                    runOrder: order,
                    cacheFirst: run(order: .cacheFirst, promptSeconds: cachePrompt),
                    combined: run(
                        order: .combined,
                        promptSeconds: cachePrompt - improvements[index]))
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
            telemetry: .init(
                proposedDraftTokens: 20,
                acceptedDraftTokens: 10,
                rejectedDraftTokens: 10,
                roundCount: 10,
                targetModelCallCount: 10,
                draftModelCallCount: 10,
                targetVerifiedTokenCount: 30,
                emittedTokenCount: 128),
            phaseAttribution: phaseAttribution(promptSeconds: promptSeconds),
            passthroughReason: nil)
    }

    private func exactness(
        mtpTokenDigest: String? = nil
    ) -> QwenMTPCorpusExactnessEvidence {
        let token = hex("token")
        let decoded = hex("decoded")
        let cache = QwenMTPCorpusCacheFingerprint(
            digest: hex("cache"),
            entries: (0..<32).map { index in
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
            mtpStopOutcome: .length,
            scalarCacheFingerprint: cache,
            mtpCacheFingerprint: cache,
            firstCacheMismatch: nil)
    }

    private func phaseAttribution(
        promptSeconds: Double = 4,
        promptTokenCount: Int = 1_353,
        hiddenEvaluationSeconds: Double = 0
    ) -> QwenMTPCorpusMTPPhaseAttribution {
        let targetPrefillSeconds = promptSeconds - 1
        let attributedSeconds = 1.05 + hiddenEvaluationSeconds
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
            targetPromptPreparation: .init(
                promptTokenCount: promptTokenCount,
                hiddenShape: [1, promptTokenCount, 4],
                hiddenByteCount: promptTokenCount * 4 * 2,
                chunks: [
                    .init(
                        tokenOffset: 0,
                        tokenCount: promptTokenCount,
                        targetForwardSchedulingSeconds: 0.1),
                ],
                cacheEvaluationSeconds: 0.75,
                hiddenEvaluationSeconds: hiddenEvaluationSeconds,
                concatenatedHiddenEvaluationSeconds: 0.1,
                preparedCacheHandoffSeconds: 0.05,
                phaseBoundarySynchronizationSeconds: 0.05,
                targetPrefillResidualSeconds:
                    targetPrefillSeconds - attributedSeconds))
    }

    private func hex(_ seed: String) -> String {
        let bytes = Array(seed.utf8)
        return (0..<64).map { index in
            String(format: "%x", bytes[index % bytes.count] & 0x0f)
        }.joined()
    }
}
