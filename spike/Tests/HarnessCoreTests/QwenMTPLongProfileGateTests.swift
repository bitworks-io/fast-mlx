import XCTest

@testable import HarnessCore

final class QwenMTPLongProfileGateTests: XCTestCase {
    func testCorpusIdentityIsFrozenAndSelfDescribing() {
        XCTAssertEqual(QwenMTPLongProfileGate.schemaVersion, 1)
        XCTAssertEqual(QwenMTPLongProfileGate.corpusID, "qwen38-mtp-long-decode-profile-v1")
        XCTAssertEqual(
            QwenMTPLongProfileGate.corpusContentHash,
            QwenMTPLongProfileGate.computedCorpusContentHash())
        XCTAssertEqual(
            QwenMTPLongProfileGate.measurementClass,
            "engineering-precheck-non-authoritative")
    }

    func testCasesAreLongDecisionLengthShapes() {
        let cases = QwenMTPLongProfileGate.cases
        XCTAssertEqual(
            cases.map(\.id),
            ["long-report-prose", "long-swift-code", "long-tool-json"])
        for spec in cases {
            XCTAssertEqual(spec.kind, .fullGreedy, spec.id)
            XCTAssertEqual(spec.maxTokens, 4096, spec.id)
            XCTAssertGreaterThanOrEqual(
                spec.prompt.utf8.count, 10_000,
                "\(spec.id) prompt must be long-context scale")
        }
    }

    func testProfilePlanUsesWarmupAndCounterbalancedMeasuredPairs() {
        let plan = QwenMTPLongProfileGate.profilePlan
        XCTAssertEqual(plan.caseIDs, QwenMTPLongProfileGate.cases.map(\.id))
        XCTAssertEqual(plan.droppedWarmupPairs, 1)
        XCTAssertEqual(plan.measuredPairs, 3)
        XCTAssertEqual(
            plan.orders,
            [.scalarThenMTP, .mtpThenScalar, .scalarThenMTP, .mtpThenScalar])
    }

    func testAllowedBindingsAreDerivedFromKnownArtifactLocks() {
        let bindings = QwenMTPLongProfileGate.allowedBindings
        XCTAssertTrue(bindings.contains(makeBinding(.qwen35)))
        XCTAssertTrue(bindings.contains(makeBinding(.qwen38MXFP8)))
        XCTAssertTrue(bindings.contains(makeBinding(.qwen384Bit)))
        for binding in bindings {
            XCTAssertEqual(binding.blockSize, 3)
            XCTAssertEqual(binding.maxAcceptedDrafts, 2)
        }
    }

    func testFourBitLockRowBindsSameDrafterAndTokenizer() {
        let fourBit = QwenMTPKnownArtifactLocks.qwen38_27B4BitDepth1
        let mxfp8 = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        XCTAssertEqual(fourBit.targetIdentity.modelID, "mlx-community/Qwen3.8-27B-4bit")
        XCTAssertEqual(
            fourBit.targetIdentity.revision,
            "3e6447f082e89cc7f0bc6e5441afd38dfce760ff")
        XCTAssertEqual(fourBit.drafterIdentity, mxfp8.drafterIdentity)
        XCTAssertEqual(fourBit.architecture, mxfp8.architecture)
        XCTAssertEqual(
            fourBit.targetIdentity.tokenizerSHA256,
            mxfp8.targetIdentity.tokenizerSHA256)
        XCTAssertEqual(fourBit.targetQuantization.bits, 4)
        XCTAssertEqual(fourBit.targetQuantization.groupSize, 64)
        XCTAssertEqual(fourBit.targetQuantization.mode, "affine")
    }

    func testValidPayloadYieldsDescriptiveSummary() throws {
        let payload = makePayload(
            ratioByCase: [
                "long-report-prose": 1.2,
                "long-swift-code": 1.3,
                "long-tool-json": 1.4,
            ])
        let summary = try QwenMTPLongProfileGate.validate(payload)
        XCTAssertEqual(summary.perCase.count, 3)
        XCTAssertEqual(
            summary.perCase.first { $0.caseID == "long-report-prose" }?
                .medianDecodeOnlyRatio ?? 0, 1.2, accuracy: 1e-9)
        XCTAssertEqual(
            summary.perCase.first { $0.caseID == "long-tool-json" }?
                .medianDecodeOnlyRatio ?? 0, 1.4, accuracy: 1e-9)
        XCTAssertEqual(summary.aggregateMedianDecodeOnlyRatio, 1.3, accuracy: 1e-9)
        let acceptance = summary.perCase.first { $0.caseID == "long-swift-code" }?
            .meanDraftAcceptanceRate ?? 0
        XCTAssertEqual(acceptance, 2496.0 / 3200.0, accuracy: 1e-9)
    }

    func testValidateRejectsIdentityDrift() {
        assertRejected { $0.schemaVersion = 99 }
        assertRejected { $0.corpusID = "some-other-corpus" }
        assertRejected { $0.corpusContentHash = "0000000000000000" }
        assertRejected { $0.measurementClass = "authoritative" }
    }

    func testValidateRejectsUnknownBinding() {
        assertRejected { payload in
            payload.binding = QwenMTPCorpusRuntimeBinding(
                targetModelID: "mlx-community/Some-Other-Model",
                drafterModelID: payload.binding.drafterModelID,
                targetRevision: payload.binding.targetRevision,
                drafterRevision: payload.binding.drafterRevision,
                sourceRevision: payload.binding.sourceRevision,
                blockSize: 3,
                maxAcceptedDrafts: 2)
        }
    }

    func testValidateRejectsGDNModeDrift() {
        assertRejected { $0.gdnMode = .gdnOff }
        assertRejected { $0.gdnObservedEnv = .disabled }
    }

    func testValidateRejectsDebugBuildEvidence() {
        assertRejected { $0.releaseBuildObserved = false }
        assertRejected { $0.releaseBuildRequired = false }
    }

    func testValidateRejectsCaseProfileDrift() {
        assertRejected { $0.caseProfiles.removeLast() }
        assertRejected { $0.caseProfiles[0].promptTokenCount = 128 }
        assertRejected { $0.caseProfiles.swapAt(0, 1) }
    }

    func testValidateRejectsSampleStructureDrift() {
        assertRejected { $0.samples.removeLast() }
        assertRejected { payload in
            payload.samples[0] = replacingSample(payload.samples[0], warmup: false)
        }
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1], order: .scalarThenMTP)
        }
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1], passthroughReason: .some("fallback"))
        }
        assertRejected { payload in
            let telemetry = payload.samples[1].mtpTelemetry
            payload.samples[1] = replacingSample(
                payload.samples[1],
                mtpTelemetry: .init(
                    proposedDraftTokens: 0,
                    acceptedDraftTokens: 0,
                    rejectedDraftTokens: 0,
                    roundCount: telemetry.roundCount,
                    targetModelCallCount: telemetry.targetModelCallCount,
                    draftModelCallCount: telemetry.draftModelCallCount,
                    targetVerifiedTokenCount: telemetry.targetVerifiedTokenCount,
                    emittedTokenCount: telemetry.emittedTokenCount))
        }
    }

    func testValidateRejectsExactnessViolations() {
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1],
                exactness: makeExactness(
                    tokens: 4096,
                    seed: "sample-1",
                    mtpTokenSeedOverride: "divergent"))
        }
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1],
                exactness: makeExactness(
                    tokens: 4096,
                    seed: "sample-1",
                    mtpCacheSeedOverride: "divergent-cache"))
        }
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1],
                exactness: makeExactness(
                    tokens: 4096,
                    seed: "sample-1",
                    mtpStopOutcomeOverride: .stop))
        }
    }

    func testValidateRejectsPerformanceRecomputeMismatch() {
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1],
                decodeOnlyRatio: payload.samples[1].decodeOnlyRatio * 1.5)
        }
        assertRejected { payload in
            payload.samples[1] = replacingSample(
                payload.samples[1],
                scalarTokensPerSecond: payload.samples[1].scalarTokensPerSecond * 2)
        }
    }

    func testValidateRejectsDegenerateGenerationLength() {
        assertRejected { payload in
            let sample = payload.samples[1]
            let short = 8
            let telemetry = sample.mtpTelemetry
            payload.samples[1] = replacingSample(
                sample,
                exactness: makeExactness(tokens: short, seed: "sample-1"),
                scalarTokensPerSecond: Double(short) / sample.scalarTiming.e2eSeconds,
                mtpTokensPerSecond: Double(short) / sample.mtpTiming.e2eSeconds,
                e2eRatio: (Double(short) / sample.mtpTiming.e2eSeconds)
                    / (Double(short) / sample.scalarTiming.e2eSeconds),
                mtpTelemetry: .init(
                    proposedDraftTokens: telemetry.proposedDraftTokens,
                    acceptedDraftTokens: telemetry.acceptedDraftTokens,
                    rejectedDraftTokens: telemetry.rejectedDraftTokens,
                    roundCount: telemetry.roundCount,
                    targetModelCallCount: telemetry.targetModelCallCount,
                    draftModelCallCount: telemetry.draftModelCallCount,
                    targetVerifiedTokenCount: telemetry.targetVerifiedTokenCount,
                    emittedTokenCount: short))
        }
    }

    func testValidateJSONLRoundTripAndRejections() throws {
        let payload = makePayload(ratioByCase: [:])
        let record = ResultRecord(
            subcommand: QwenMTPLongProfileGate.subcommand,
            provenance: makeProvenance(),
            payload: payload)
        let valid = Data((try record.jsonLine() + "\n").utf8)
        let summaries = try QwenMTPLongProfileGate.validateJSONL(valid)
        XCTAssertEqual(summaries.count, 1)

        let wrongSubcommand = ResultRecord(
            subcommand: "qwen-mtp-corpus",
            provenance: makeProvenance(),
            payload: payload)
        let wrongData = Data((try wrongSubcommand.jsonLine() + "\n").utf8)
        XCTAssertThrowsError(try QwenMTPLongProfileGate.validateJSONL(wrongData))

        let mismatchedModel = ResultRecord(
            subcommand: QwenMTPLongProfileGate.subcommand,
            provenance: makeProvenance(modelPath: "/local/dir"),
            payload: payload)
        let mismatchedData = Data((try mismatchedModel.jsonLine() + "\n").utf8)
        XCTAssertThrowsError(try QwenMTPLongProfileGate.validateJSONL(mismatchedData))

        XCTAssertThrowsError(
            try QwenMTPLongProfileGate.validateJSONL(Data("{}".utf8)))
    }

    // MARK: - Fixtures

    private enum BindingKind {
        case qwen35
        case qwen38MXFP8
        case qwen384Bit
    }

    private func makeBinding(_ kind: BindingKind) -> QwenMTPCorpusRuntimeBinding {
        let lock: QwenMTPArtifactLock
        switch kind {
        case .qwen35:
            lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38MXFP8:
            lock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        case .qwen384Bit:
            lock = QwenMTPKnownArtifactLocks.qwen38_27B4BitDepth1
        }
        return QwenMTPCorpusRuntimeBinding(
            targetModelID: lock.targetIdentity.modelID,
            drafterModelID: lock.drafterIdentity.modelID,
            targetRevision: lock.targetIdentity.revision,
            drafterRevision: lock.drafterIdentity.revision,
            sourceRevision: lock.sourceRevision,
            blockSize: 3,
            maxAcceptedDrafts: 2)
    }

    private func assertRejected(
        file: StaticString = #filePath,
        line: UInt = #line,
        _ mutate: (inout QwenMTPLongProfileEvidencePayload) -> Void
    ) {
        var payload = makePayload(ratioByCase: [:])
        mutate(&payload)
        XCTAssertThrowsError(
            try QwenMTPLongProfileGate.validate(payload), file: file, line: line)
    }

    private func makePayload(
        ratioByCase: [String: Double]
    ) -> QwenMTPLongProfileEvidencePayload {
        let promptTokenCount = 4100
        var samples: [QwenMTPCorpusProfileSample] = []
        for caseID in QwenMTPLongProfileGate.profilePlan.caseIDs {
            for pairIndex in 0..<QwenMTPLongProfileGate.profilePlan.totalPairsPerCase {
                let warmup =
                    pairIndex < QwenMTPLongProfileGate.profilePlan.droppedWarmupPairs
                let ratio = warmup ? 1.05 : ratioByCase[caseID, default: 1.25]
                samples.append(makeSample(
                    caseID: caseID,
                    pairIndex: pairIndex,
                    warmup: warmup,
                    order: QwenMTPLongProfileGate.profilePlan.orders[pairIndex],
                    promptTokenCount: promptTokenCount,
                    decodeRatio: ratio))
            }
        }
        return QwenMTPLongProfileEvidencePayload(
            schemaVersion: QwenMTPLongProfileGate.schemaVersion,
            corpusID: QwenMTPLongProfileGate.corpusID,
            corpusContentHash: QwenMTPLongProfileGate.corpusContentHash,
            measurementClass: QwenMTPLongProfileGate.measurementClass,
            binding: makeBinding(.qwen38MXFP8),
            host: .init(
                chip: "Apple M3 Ultra",
                ramBytes: 256 * 1024 * 1024 * 1024,
                os: "macOS 26.0.0"),
            gdnMode: .gdnOn,
            gdnObservedEnv: .enabled,
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            caseProfiles: QwenMTPLongProfileGate.profilePlan.caseIDs.map {
                .init(caseID: $0, promptTokenCount: promptTokenCount)
            },
            samples: samples)
    }

    private func makeSample(
        caseID: String,
        pairIndex: Int,
        warmup: Bool,
        order: QwenMTPCorpusRunOrder,
        promptTokenCount: Int,
        decodeRatio: Double
    ) -> QwenMTPCorpusProfileSample {
        let tokens = 4096
        let promptSeconds = 8.0
        let mtpGenerationSeconds = 204.8 / decodeRatio
        let scalarGenerationSeconds = 204.8
        let scalarE2E = promptSeconds + scalarGenerationSeconds
        let mtpE2E = promptSeconds + mtpGenerationSeconds
        let scalarTPS = Double(tokens) / scalarE2E
        let mtpTPS = Double(tokens) / mtpE2E
        return QwenMTPCorpusProfileSample(
            caseID: caseID,
            pairIndex: pairIndex,
            warmup: warmup,
            order: order,
            exactness: makeExactness(
                tokens: tokens, seed: "\(caseID)-\(pairIndex)"),
            scalarTiming: .init(
                promptSeconds: promptSeconds,
                generationSeconds: scalarGenerationSeconds,
                wallSeconds: scalarE2E,
                e2eSeconds: scalarE2E),
            mtpTiming: .init(
                promptSeconds: promptSeconds,
                generationSeconds: mtpGenerationSeconds,
                wallSeconds: mtpE2E,
                e2eSeconds: mtpE2E),
            scalarTokensPerSecond: scalarTPS,
            mtpTokensPerSecond: mtpTPS,
            decodeOnlyRatio: scalarGenerationSeconds / mtpGenerationSeconds,
            e2eRatio: mtpTPS / scalarTPS,
            mtpTelemetry: .init(
                proposedDraftTokens: 3200,
                acceptedDraftTokens: 2496,
                rejectedDraftTokens: 704,
                roundCount: 1600,
                targetModelCallCount: 1600,
                draftModelCallCount: 1600,
                targetVerifiedTokenCount: 4800,
                emittedTokenCount: tokens),
            mtpPhaseAttribution: makePhaseAttribution(
                promptTokenCount: promptTokenCount),
            passthroughReason: nil)
    }

    private func replacingSample(
        _ sample: QwenMTPCorpusProfileSample,
        order: QwenMTPCorpusRunOrder? = nil,
        warmup: Bool? = nil,
        passthroughReason: String?? = nil,
        exactness: QwenMTPCorpusExactnessEvidence? = nil,
        scalarTokensPerSecond: Double? = nil,
        mtpTokensPerSecond: Double? = nil,
        decodeOnlyRatio: Double? = nil,
        e2eRatio: Double? = nil,
        mtpTelemetry: QwenMTPCorpusMTPTelemetry? = nil
    ) -> QwenMTPCorpusProfileSample {
        QwenMTPCorpusProfileSample(
            caseID: sample.caseID,
            pairIndex: sample.pairIndex,
            warmup: warmup ?? sample.warmup,
            order: order ?? sample.order,
            exactness: exactness ?? sample.exactness,
            scalarTiming: sample.scalarTiming,
            mtpTiming: sample.mtpTiming,
            scalarTokensPerSecond: scalarTokensPerSecond ?? sample.scalarTokensPerSecond,
            mtpTokensPerSecond: mtpTokensPerSecond ?? sample.mtpTokensPerSecond,
            decodeOnlyRatio: decodeOnlyRatio ?? sample.decodeOnlyRatio,
            e2eRatio: e2eRatio ?? sample.e2eRatio,
            mtpTelemetry: mtpTelemetry ?? sample.mtpTelemetry,
            mtpPhaseAttribution: sample.mtpPhaseAttribution,
            passthroughReason: passthroughReason ?? sample.passthroughReason)
    }

    private func makeExactness(
        tokens: Int,
        seed: String,
        mtpTokenSeedOverride: String? = nil,
        mtpCacheSeedOverride: String? = nil,
        mtpStopOutcomeOverride: QwenMTPCorpusStopOutcome? = nil
    ) -> QwenMTPCorpusExactnessEvidence {
        let mtpCacheSeed = mtpCacheSeedOverride ?? "cache-\(seed)"
        return QwenMTPCorpusExactnessEvidence(
            scalarTokenCount: tokens,
            mtpTokenCount: tokens,
            scalarTokenIDsSHA256: hex("token-\(seed)"),
            mtpTokenIDsSHA256: hex("token-\(mtpTokenSeedOverride ?? seed)"),
            scalarDecodedBytesSHA256: hex("decoded-\(seed)"),
            mtpDecodedBytesSHA256: hex("decoded-\(seed)"),
            scalarStopOutcome: .length,
            mtpStopOutcome: mtpStopOutcomeOverride ?? .length,
            scalarCacheFingerprint: cacheFingerprint("cache-\(seed)"),
            mtpCacheFingerprint: cacheFingerprint(mtpCacheSeed),
            firstCacheMismatch: mtpCacheSeedOverride == nil ? nil : "digest")
    }

    private func makePhaseAttribution(
        promptTokenCount: Int
    ) -> QwenMTPCorpusMTPPhaseAttribution {
        let half = promptTokenCount / 2
        let preparation = QwenMTPPromptPreparationAttribution(
            promptTokenCount: promptTokenCount,
            hiddenShape: [1, promptTokenCount, 64],
            hiddenByteCount: promptTokenCount * 64 * 2,
            chunks: [
                .init(
                    tokenOffset: 0, tokenCount: half,
                    targetForwardSchedulingSeconds: 0.0002),
                .init(
                    tokenOffset: half, tokenCount: promptTokenCount - half,
                    targetForwardSchedulingSeconds: 0.0002),
            ],
            cacheEvaluationSeconds: 0.0003,
            hiddenEvaluationSeconds: 0.0002,
            concatenatedHiddenEvaluationSeconds: 0.0002,
            preparedCacheHandoffSeconds: 0.0004,
            phaseBoundarySynchronizationSeconds: 0.0001,
            targetPrefillResidualSeconds: 0.0004)
        return QwenMTPCorpusMTPPhaseAttribution(
            targetPrefillSeconds: 0.002,
            drafterPromptPrimingSeconds: 0.002,
            draftBlockSeconds: 0.004,
            targetVerificationSeconds: 0.006,
            targetTailSeconds: 0,
            hybridRewindReplaySeconds: 0.001,
            finalizationSeconds: 0.001,
            cacheFingerprintSeconds: 0.001,
            targetPrefillCount: 1,
            drafterPromptPrimingCount: 1,
            draftBlockCount: 1600,
            targetVerificationCount: 1600,
            targetTailCount: 0,
            hybridRewindReplayCount: 1600,
            finalizationCount: 1,
            cacheFingerprintCount: 1,
            targetPromptPreparation: preparation)
    }

    private func makeProvenance(
        modelPath: String =
            QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1.targetIdentity.modelID
    ) -> Provenance {
        Provenance(
            date: "2026-09-01T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1024 * 1024 * 1024,
            hardwareOS: "macOS 26.0.0",
            harnessGitSHA: "abc",
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: modelPath,
            modelConfigHash: "hash",
            modelQuant: .init(bits: 8, groupSize: 32),
            corpusId: QwenMTPLongProfileGate.corpusID,
            corpusContentHash: QwenMTPLongProfileGate.corpusContentHash,
            nonce: "nonce")
    }
}

private func hex(_ seed: String) -> String {
    let alphabet = Array("0123456789abcdef")
    var result = ""
    for scalar in seed.unicodeScalars {
        result.append(alphabet[Int(scalar.value) % alphabet.count])
    }
    while result.count < 64 {
        result.append(alphabet[result.count % alphabet.count])
    }
    return String(result.prefix(64))
}

private func cacheFingerprint(_ seed: String) -> QwenMTPCorpusCacheFingerprint {
    QwenMTPCorpusCacheFingerprint(
        digest: hex("cache-digest-\(seed)"),
        entries: [
            .init(
                layerIndex: 0,
                cacheType: "KVCacheSimple",
                offset: 8,
                metaStateSHA256: hex("meta-\(seed)"),
                stateCount: 1,
                states: [
                    .init(
                        stateIndex: 0,
                        shape: [1, 4, 8, 64],
                        dtype: "float16",
                        byteCount: 4096,
                        sha256: hex("state-\(seed)"))
                ])
        ])
}
