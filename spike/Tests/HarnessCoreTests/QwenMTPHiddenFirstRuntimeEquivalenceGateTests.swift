import XCTest
@testable import HarnessCore

final class QwenMTPHiddenFirstRuntimeEquivalenceGateTests: XCTestCase {
    func testGateFreezesOptInRuntimePlanAndExistingThresholds() {
        XCTAssertEqual(QwenMTPHiddenFirstRuntimeEquivalenceGate.schemaVersion, 1)
        XCTAssertEqual(QwenMTPHiddenFirstRuntimeEquivalenceGate.droppedWarmupPairs, 2)
        XCTAssertEqual(QwenMTPHiddenFirstRuntimeEquivalenceGate.measuredPairs, 5)
        XCTAssertEqual(QwenMTPHiddenFirstRuntimeEquivalenceGate.requiredPromptTokenCount, 1_353)
        XCTAssertEqual(
            QwenMTPHiddenFirstRuntimeEquivalenceGate.pairOrders,
            [
                .defaultThenHiddenFirst,
                .hiddenFirstThenDefault,
                .defaultThenHiddenFirst,
                .hiddenFirstThenDefault,
                .defaultThenHiddenFirst,
                .hiddenFirstThenDefault,
                .defaultThenHiddenFirst,
            ])
        XCTAssertEqual(
            QwenMTPHiddenFirstRuntimeEquivalenceGate
                .requiredAggregatePromptImprovementSeconds,
            QwenMTPEvaluationOrderIsolationGate
                .requiredAggregatePromptImprovementSeconds)
        XCTAssertEqual(
            QwenMTPHiddenFirstRuntimeEquivalenceGate
                .requiredMedianPromptImprovementSeconds,
            QwenMTPEvaluationOrderIsolationGate
                .requiredMedianPromptImprovementSeconds)
    }

    func testValidateQualifiesOnlyWhenParityAndBothPerformanceThresholdsPass() throws {
        let verdict = try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(
            makePayload(improvements: [0, 0, 0.41, 0.45, 0.52, 0.58, 0.61]))

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(verdict.aggregatePromptImprovementSeconds, 2.57, accuracy: 1e-12)
        XCTAssertEqual(verdict.medianPromptImprovementSeconds, 0.52, accuracy: 1e-12)

        let aggregateMiss = try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(
            makePayload(improvements: [0, 0, 0.40, 0.40, 0.40, 0.40, 0.40]))
        XCTAssertFalse(aggregateMiss.qualified)

        let medianMiss = try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(
            makePayload(improvements: [0, 0, 0.10, 0.10, 0.10, 1.20, 1.20]))
        XCTAssertFalse(medianMiss.qualified)
    }

    func testValidateRejectsWrongBindingReleasePairShapeAndOrder() {
        var wrongBinding = makePayload()
        wrongBinding.binding = .init(
            targetModelID: "wrong",
            drafterModelID: wrongBinding.binding.drafterModelID,
            targetRevision: wrongBinding.binding.targetRevision,
            drafterRevision: wrongBinding.binding.drafterRevision,
            sourceRevision: wrongBinding.binding.sourceRevision,
            blockSize: wrongBinding.binding.blockSize,
            maxAcceptedDrafts: wrongBinding.binding.maxAcceptedDrafts)
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(wrongBinding))

        var nonRelease = makePayload()
        nonRelease.releaseBuildObserved = false
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(nonRelease))

        var missingPair = makePayload()
        missingPair.pairs.removeLast()
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(missingPair))

        var wrongOrder = makePayload()
        wrongOrder.pairs[0] = wrongOrder.pairs[0].copy(
            runOrder: .hiddenFirstThenDefault)
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(wrongOrder))
    }

    func testValidateRejectsRelabeledDefaultHiddenFirstOrCombinedRuns() {
        var relabeledDefault = makePayload()
        relabeledDefault.pairs[2] = relabeledDefault.pairs[2].copy(
            defaultRuntime: relabeledDefault.pairs[2].defaultRuntime.copy(
                evaluationOrder: .hiddenFirst))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(relabeledDefault))

        var relabeledCandidate = makePayload()
        relabeledCandidate.pairs[2] = relabeledCandidate.pairs[2].copy(
            hiddenFirstRuntime: relabeledCandidate.pairs[2].hiddenFirstRuntime.copy(
                evaluationOrder: .cacheFirst))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(relabeledCandidate))

        var combinedCandidate = makePayload()
        combinedCandidate.pairs[2] = combinedCandidate.pairs[2].copy(
            hiddenFirstRuntime: combinedCandidate.pairs[2].hiddenFirstRuntime.copy(
                evaluationOrder: .combined))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(combinedCandidate))
    }

    func testValidateRejectsTokenDecodedStopAndCacheDrift() {
        var tokenMismatch = makePayload()
        tokenMismatch.pairs[2] = tokenMismatch.pairs[2].copy(
            hiddenFirstRuntime: tokenMismatch.pairs[2].hiddenFirstRuntime.copy(
                exactness: exactness(mtpTokenDigest: hex("different"))))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(tokenMismatch))

        var decodedMismatch = makePayload()
        decodedMismatch.pairs[2] = decodedMismatch.pairs[2].copy(
            hiddenFirstRuntime: decodedMismatch.pairs[2].hiddenFirstRuntime.copy(
                exactness: exactness(mtpDecodedDigest: hex("different"))))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(decodedMismatch))

        var stopMismatch = makePayload()
        stopMismatch.pairs[2] = stopMismatch.pairs[2].copy(
            hiddenFirstRuntime: stopMismatch.pairs[2].hiddenFirstRuntime.copy(
                exactness: exactness(mtpStop: .stop)))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(stopMismatch))

        var cacheMismatch = makePayload()
        cacheMismatch.pairs[2] = cacheMismatch.pairs[2].copy(
            hiddenFirstRuntime: cacheMismatch.pairs[2].hiddenFirstRuntime.copy(
                exactness: exactness(cacheEntryCount: 31)))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(cacheMismatch))
    }

    func testValidateRejectsPassthroughCountersAndTimingEnvelopeDrift() {
        var passthrough = makePayload()
        passthrough.pairs[2] = passthrough.pairs[2].copy(
            hiddenFirstRuntime: passthrough.pairs[2].hiddenFirstRuntime.copy(
                passthroughReason: "fallback"))
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(passthrough))

        var counters = makePayload()
        counters.pairs[2] = counters.pairs[2].copy(
            hiddenFirstRuntime: counters.pairs[2].hiddenFirstRuntime.copy(
                telemetry: telemetry(proposed: 4, accepted: 5)))
        XCTAssertThrowsError(try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(counters))

        var missingPreparation = makePayload()
        missingPreparation.pairs[2] = missingPreparation.pairs[2].copy(
            hiddenFirstRuntime: missingPreparation.pairs[2].hiddenFirstRuntime.copy(
                phaseAttribution: phaseAttribution(includePreparation: false)))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(missingPreparation))

        var forgedEnvelope = makePayload()
        forgedEnvelope.pairs[2] = forgedEnvelope.pairs[2].copy(
            hiddenFirstRuntime: forgedEnvelope.pairs[2].hiddenFirstRuntime.copy(
                phaseAttribution: phaseAttribution(targetPrefillResidualSeconds: 0.8)))
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(forgedEnvelope))
    }

    func testJSONLValidatorBindsOneQualifiedRecordSubcommandAndProvenance() throws {
        let payload = makePayload()
        let data = try recordData(
            payload: payload,
            subcommand: QwenMTPHiddenFirstRuntimeEquivalenceGate.subcommand)

        let verdict = try XCTUnwrap(
            QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(data).first)
        XCTAssertTrue(verdict.qualified)

        let wrongSubcommand = try recordData(payload: payload, subcommand: "qwen-mtp-eval-order")
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(wrongSubcommand))

        let twoRows = data + data
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(twoRows))

        let wrongProvenance = try recordData(
            payload: payload,
            subcommand: QwenMTPHiddenFirstRuntimeEquivalenceGate.subcommand,
            modelPath: "wrong")
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(wrongProvenance)
        ) { error in
            XCTAssertEqual(
                error as? QwenMTPHiddenFirstRuntimeEquivalenceGateError,
                .invalidProvenance("model path"))
        }
    }

    func testQualifiedAndRejectedJSONLValidatorsCannotBeInterchanged() throws {
        let unqualified = makePayload(
            improvements: [0, 0, 0.10, 0.10, 0.10, 1.20, 1.20])
        let rejectedData = try recordData(
            payload: unqualified,
            subcommand: QwenMTPHiddenFirstRuntimeEquivalenceGate.rejectedSubcommand)

        let rejectedVerdict = try XCTUnwrap(
            QwenMTPHiddenFirstRuntimeEquivalenceGate
                .validateRejectedJSONL(rejectedData).first)
        XCTAssertFalse(rejectedVerdict.qualified)
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(rejectedData))

        let qualified = makePayload()
        let qualifiedData = try recordData(
            payload: qualified,
            subcommand: QwenMTPHiddenFirstRuntimeEquivalenceGate.subcommand)
        XCTAssertThrowsError(
            try QwenMTPHiddenFirstRuntimeEquivalenceGate
                .validateRejectedJSONL(qualifiedData))
    }

    private func recordData(
        payload: QwenMTPHiddenFirstRuntimeEquivalencePayload,
        subcommand: String,
        modelPath: String? = nil
    ) throws -> Data {
        let record = ResultRecord(
            subcommand: subcommand,
            provenance: Provenance(
                date: "2026-08-26T00:00:00Z",
                hardwareChip: payload.host.chip,
                hardwareRAMBytes: payload.host.ramBytes,
                hardwareOS: payload.host.os,
                harnessGitSHA: String(repeating: "a", count: 40),
                mlxSwiftVersion: "0.31.6",
                referenceMLXVersion: nil,
                referenceMLXLMVersion:
                    "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
                modelPath: modelPath ?? payload.binding.targetModelID,
                modelConfigHash: "5a99be4477ebdac8",
                modelCheckpointManifestHash: "db2b2480a8525194",
                modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
                corpusId: payload.corpusID,
                corpusContentHash: payload.corpusContentHash,
                nonce: "test-nonce"),
            payload: payload)
        return Data((try record.jsonLine() + "\n").utf8)
    }

    private func makePayload(
        improvements: [Double] = [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
    ) -> QwenMTPHiddenFirstRuntimeEquivalencePayload {
        .init(
            schemaVersion: QwenMTPHiddenFirstRuntimeEquivalenceGate.schemaVersion,
            corpusID: QwenMTPHiddenFirstRuntimeEquivalenceGate.corpusID,
            corpusContentHash: QwenMTPHiddenFirstRuntimeEquivalenceGate.corpusContentHash,
            binding: QwenMTPCorpusGate.requiredBinding,
            host: .init(
                chip: "Apple M5",
                ramBytes: 24 * 1024 * 1024 * 1024,
                os: "macOS 26.0"),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            pairs: QwenMTPHiddenFirstRuntimeEquivalenceGate.pairOrders.enumerated().map {
                index, order in
                let defaultPrompt = 4.0
                return .init(
                    pairIndex: index,
                    warmup: index
                        < QwenMTPHiddenFirstRuntimeEquivalenceGate.droppedWarmupPairs,
                    runOrder: order,
                    defaultRuntime: run(order: .cacheFirst, promptSeconds: defaultPrompt),
                    hiddenFirstRuntime: run(
                        order: .hiddenFirst,
                        promptSeconds: defaultPrompt - improvements[index]))
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
        mtpDecodedDigest: String? = nil,
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
            mtpDecodedBytesSHA256: mtpDecodedDigest ?? decoded,
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
        targetPrefillResidualSeconds: Double? = nil
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
                    promptTokenCount: 1_353,
                    hiddenShape: [1, 1_353, 4],
                    hiddenByteCount: 10_824,
                    chunks: [
                        .init(
                            tokenOffset: 0,
                            tokenCount: 1_353,
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
