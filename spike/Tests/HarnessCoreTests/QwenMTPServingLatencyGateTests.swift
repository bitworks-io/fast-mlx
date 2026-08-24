import Foundation
import XCTest
@testable import HarnessCore

final class QwenMTPServingLatencyGateTests: XCTestCase {
    func testValidServingProfileQualifiesAndCarriesFrozenBindingPlanAndVerdict() throws {
        let evidence = makeEvidence()

        let verdict = try QwenMTPServingLatencyGate.validate(evidence)

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(evidence.schemaVersion, QwenMTPServingLatencyGate.schemaVersion)
        XCTAssertEqual(evidence.corpusID, QwenMTPCorpusGate.corpusID)
        XCTAssertEqual(evidence.corpusContentHash, QwenMTPCorpusGate.corpusContentHash)
        XCTAssertEqual(evidence.binding, QwenMTPCorpusGate.requiredBinding)
        XCTAssertEqual(evidence.measurementClass, QwenMTPServingLatencyGate.measurementClass)
        XCTAssertEqual(evidence.host.ramBytes, 25_769_803_776)
        XCTAssertEqual(evidence.profilePlan, QwenMTPCorpusGate.profilePlan)
        XCTAssertEqual(evidence.samples.count, 42)
        XCTAssertEqual(evidence.samples[0].route.kind, .exactQwen35MTP)
        XCTAssertEqual(evidence.samples[0].fallback.scalarFallbackStartCount, 0)
        XCTAssertEqual(evidence.request.temperature, 0)
        XCTAssertTrue(evidence.request.greedy)
        XCTAssertTrue(evidence.request.toolsEmpty)
        XCTAssertTrue(evidence.request.penaltiesDisabled)
        XCTAssertTrue(evidence.request.samplingKnobsUnset)
        XCTAssertEqual(evidence.verdict, verdict)

        let roundTrip = try JSONDecoder().decode(
            QwenMTPServingLatencyEvidence.self,
            from: JSONEncoder().encode(evidence))
        XCTAssertEqual(roundTrip, evidence)
    }

    func testServingProfileFailsClosedBeforePerformanceWhenRouteFallbackOrRequestDrifts() {
        var wrongRoute = makeEvidence()
        wrongRoute.samples[0].route = .init(kind: .scalarFallback)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(wrongRoute)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidRoute(index: 0, kind: "scalarFallback"))
        }

        var fallbackUsed = makeEvidence()
        fallbackUsed.samples[1].fallback = .init(scalarFallbackStartCount: 1)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(fallbackUsed)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .scalarFallbackUsed(index: 1, startCount: 1))
        }

        var sampled = makeEvidence()
        sampled.request = .init(
            temperature: 0.7,
            topP: nil,
            topK: nil,
            minP: nil,
            seed: nil,
            toolsEmpty: true,
            penaltiesDisabled: true)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(sampled)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidRequest("greedy"))
        }
    }

    func testServingProfileFailsClosedOnAnySampleExactnessTelemetryOrPassthroughMismatch() {
        var weakDraft = makeEvidence()
        weakDraft.samples[0].mtpTelemetry = .init(
            proposedDraftTokens: 4,
            acceptedDraftTokens: 0,
            rejectedDraftTokens: 4,
            roundCount: 2,
            targetModelCallCount: 2,
            draftModelCallCount: 2,
            targetVerifiedTokenCount: 6,
            emittedTokenCount: 128)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(weakDraft))

        var passthrough = makeEvidence()
        passthrough.samples[1].passthroughReason = "fallback"
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(passthrough))

        var mismatchedCache = makeEvidence()
        let exactness = mismatchedCache.samples[2].exactness
        mismatchedCache.samples[2].exactness = .init(
            tokenObservationMode: exactness.tokenObservationMode,
            scalarDirectTokenCount: exactness.scalarDirectTokenCount,
            mtpUsageCompletionTokenCount: exactness.mtpUsageCompletionTokenCount,
            scalarDirectTokenIDsSHA256: exactness.scalarDirectTokenIDsSHA256,
            mtpDecodedRoundTripTokenIDsSHA256:
                exactness.mtpDecodedRoundTripTokenIDsSHA256,
            scalarDecodedBytesSHA256: exactness.scalarDecodedBytesSHA256,
            mtpDecodedBytesSHA256: exactness.mtpDecodedBytesSHA256,
            scalarStopOutcome: exactness.scalarStopOutcome,
            mtpStopOutcome: exactness.mtpStopOutcome,
            scalarCacheFingerprint: exactness.scalarCacheFingerprint,
            mtpCacheFingerprint: cacheFingerprint("different"),
            firstCacheMismatch: "layer0")
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(mismatchedCache))

        var cancelled = makeEvidence()
        let completed = cancelled.samples[3].exactness
        cancelled.samples[3].exactness = .init(
            tokenObservationMode: completed.tokenObservationMode,
            scalarDirectTokenCount: completed.scalarDirectTokenCount,
            mtpUsageCompletionTokenCount: completed.mtpUsageCompletionTokenCount,
            scalarDirectTokenIDsSHA256: completed.scalarDirectTokenIDsSHA256,
            mtpDecodedRoundTripTokenIDsSHA256:
                completed.mtpDecodedRoundTripTokenIDsSHA256,
            scalarDecodedBytesSHA256: completed.scalarDecodedBytesSHA256,
            mtpDecodedBytesSHA256: completed.mtpDecodedBytesSHA256,
            scalarStopOutcome: .cancelled,
            mtpStopOutcome: .cancelled,
            scalarCacheFingerprint: completed.scalarCacheFingerprint,
            mtpCacheFingerprint: completed.mtpCacheFingerprint,
            firstCacheMismatch: nil)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(cancelled))
    }

    func testServingProfileFailsClosedOnLatencyThresholdsAndReleaseObservation() {
        var debug = makeEvidence()
        debug.releaseBuildObserved = false
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(debug)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .releaseBuildRequired)
        }

        let weakAggregate = makeEvidence(defaultRatio: 1.07)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(weakAggregate)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .unqualifiedPerformance)
        }

        let weakHalves = makeEvidence(
            measuredRatioByPair: [0: 1.09, 1: 1.09, 2: 1.04, 3: 1.04, 4: 1.04])
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(weakHalves)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .unqualifiedPerformance)
        }

        let weakPrompts = makeEvidence(
            ratioByCase: ["prime-sequence": 0.96, "swift-code": 0.96])
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(weakPrompts)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .unqualifiedPerformance)
        }
    }

    func testServingProfileFailsClosedOnMeasurementClassAndHostDrift() {
        let proof = QwenMTPServingLatencyGate.requiredLowerLevelProof
        let wrongProof = makeEvidence(lowerLevelProof: .init(
            acceptedCorpusSubcommand: proof.acceptedCorpusSubcommand,
            acceptedCorpusSchemaVersion: proof.acceptedCorpusSchemaVersion,
            acceptedCorpusID: proof.acceptedCorpusID,
            acceptedCorpusContentHash: proof.acceptedCorpusContentHash,
            acceptedCorpusBinding: proof.acceptedCorpusBinding,
            acceptedCorpusJSONLSHA256: String(repeating: "0", count: 64),
            acceptedCorpusHarnessGitSHA: proof.acceptedCorpusHarnessGitSHA))
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(wrongProof)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .invalidLowerLevelProof)
        }

        let wrongCheckpoint = makeEvidence(checkpointIdentity: .init(
            targetCheckpointContentSHA256: String(repeating: "0", count: 64),
            drafterCheckpointContentSHA256:
                QwenMTPServingLatencyGate.requiredCheckpointIdentity
                    .drafterCheckpointContentSHA256))
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(wrongCheckpoint)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidCheckpointIdentity)
        }

        let wrongClass = makeEvidence(measurementClass: "consumer-128gib")
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(wrongClass)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidMeasurementClass("consumer-128gib"))
        }

        let wrongRAM = makeEvidence(host: .init(
            chip: "Apple M5",
            ramBytes: 137_438_953_472,
            os: "macOS 16.0.0"))
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(wrongRAM)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .invalidHost)
        }

        let missingAppleChip = makeEvidence(host: .init(
            chip: "Virtual CPU",
            ramBytes: QwenMTPServingLatencyGate.requiredRAMBytes,
            os: "macOS 16.0.0"))
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(missingAppleChip)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .invalidHost)
        }
    }

    func testValidateRequiresStoredVerdictButCandidateEvaluationComputesIt() throws {
        let candidate = makeEvidence(verdict: nil)

        let verdict = try QwenMTPServingLatencyGate.evaluateCandidate(candidate)

        XCTAssertTrue(verdict.qualified)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(candidate)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .verdictMismatch)
        }

        let staleVerdict = makeEvidence(verdict: QwenMTPServingLatencyVerdict(
                qualified: true,
                aggregatePairedMedian: 1.09,
                chronologicalFirstHalfMedian: 1.09,
                chronologicalSecondHalfMedian: 1.09,
                perPromptMedians: Dictionary(
                    uniqueKeysWithValues: QwenMTPCorpusGate.profilePlan.caseIDs.map {
                        ($0, 1.09)
                    }),
                perPromptMedianBelowFloorCount: 0,
                aggregateThreshold: 1.08,
                chronologicalHalfThreshold: 1.05,
                perPromptFloor: 0.97))
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validate(staleVerdict)) { error in
            XCTAssertEqual(error as? QwenMTPServingLatencyGateError, .verdictMismatch)
        }
    }

    func testJSONLValidationAcceptsOnlyServingLatencySubcommand() throws {
        let record = ResultRecord(
            subcommand: QwenMTPServingLatencyGate.subcommand,
            provenance: makeProvenance(),
            payload: makeEvidence())
        let valid = Data((try record.jsonLine() + "\n").utf8)

        let verdicts = try QwenMTPServingLatencyGate.validateJSONL(valid)

        XCTAssertEqual(verdicts.count, 1)
        XCTAssertTrue(verdicts[0].qualified)

        let rejected = ResultRecord(
            subcommand: QwenMTPServingLatencyGate.rejectedSubcommand,
            provenance: makeProvenance(),
            payload: makeEvidence())
        XCTAssertThrowsError(
            try QwenMTPServingLatencyGate.validateJSONL(
                Data((try rejected.jsonLine() + "\n").utf8))) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .wrongSubcommand(QwenMTPServingLatencyGate.rejectedSubcommand))
        }

        let duplicate = Data((try record.jsonLine() + "\n" + record.jsonLine() + "\n").utf8)
        XCTAssertThrowsError(try QwenMTPServingLatencyGate.validateJSONL(duplicate)) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidRecordCardinality(2))
        }

        let unsealed = ResultRecord(
            subcommand: QwenMTPServingLatencyGate.subcommand,
            provenance: makeProvenance(harnessGitSHA: "unknown"),
            payload: makeEvidence())
        XCTAssertThrowsError(
            try QwenMTPServingLatencyGate.validateJSONL(
                Data((try unsealed.jsonLine() + "\n").utf8))) { error in
            XCTAssertEqual(
                error as? QwenMTPServingLatencyGateError,
                .invalidProvenance("harnessGitSHA"))
        }
    }
}

private func makeEvidence(
    defaultRatio: Double = 1.12,
    ratioByCase: [String: Double] = [:],
    measuredRatioByPair: [Int: Double] = [:],
    lowerLevelProof: QwenMTPServingLatencyLowerLevelProof =
        QwenMTPServingLatencyGate.requiredLowerLevelProof,
    checkpointIdentity: QwenMTPServingLatencyCheckpointIdentity =
        QwenMTPServingLatencyGate.requiredCheckpointIdentity,
    measurementClass: String = QwenMTPServingLatencyGate.measurementClass,
    host: QwenMTPCorpusHostEvidence = .init(
        chip: "Apple M5",
        ramBytes: 25_769_803_776,
        os: "macOS 16.0.0"),
    verdict: QwenMTPServingLatencyVerdict? = QwenMTPServingLatencyVerdict(
        qualified: true,
        aggregatePairedMedian: 1.12,
        chronologicalFirstHalfMedian: 1.12,
        chronologicalSecondHalfMedian: 1.12,
        perPromptMedians: Dictionary(
            uniqueKeysWithValues: QwenMTPCorpusGate.profilePlan.caseIDs.map {
                ($0, 1.12)
            }),
        perPromptMedianBelowFloorCount: 0,
        aggregateThreshold: 1.08,
        chronologicalHalfThreshold: 1.05,
        perPromptFloor: 0.97)
) -> QwenMTPServingLatencyEvidence {
    let samples = makeSamples(
        defaultRatio: defaultRatio,
        ratioByCase: ratioByCase,
        measuredRatioByPair: measuredRatioByPair)
    return QwenMTPServingLatencyEvidence(
        schemaVersion: QwenMTPServingLatencyGate.schemaVersion,
        corpusID: QwenMTPCorpusGate.corpusID,
        corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
        binding: QwenMTPCorpusGate.requiredBinding,
        lowerLevelProof: lowerLevelProof,
        checkpointIdentity: checkpointIdentity,
        measurementClass: measurementClass,
        host: host,
        profilePlan: QwenMTPCorpusGate.profilePlan,
        releaseBuildRequired: true,
        releaseBuildObserved: true,
        request: .init(
            temperature: 0,
            topP: nil,
            topK: nil,
            minP: nil,
            seed: nil,
            toolsEmpty: true,
            penaltiesDisabled: true),
        samples: samples,
        verdict: verdict)
}

private func makeSamples(
    defaultRatio: Double,
    ratioByCase: [String: Double],
    measuredRatioByPair: [Int: Double]
) -> [QwenMTPServingLatencySample] {
    var samples: [QwenMTPServingLatencySample] = []
    for caseID in QwenMTPCorpusGate.profilePlan.caseIDs {
        for pairIndex in 0..<QwenMTPCorpusGate.profilePlan.totalPairsPerCase {
            let measuredPairIndex = pairIndex - QwenMTPCorpusGate.profilePlan.droppedWarmupPairs
            let isWarmup = pairIndex < QwenMTPCorpusGate.profilePlan.droppedWarmupPairs
            let ratio = isWarmup
                ? defaultRatio
                : measuredRatioByPair[
                    measuredPairIndex,
                    default: ratioByCase[caseID, default: defaultRatio]]
            let mtpSeconds = 1.0
            let scalarSeconds = mtpSeconds * ratio
            samples.append(.init(
                caseID: caseID,
                pairIndex: pairIndex,
                warmup: isWarmup,
                order: QwenMTPCorpusGate.profilePlan.orders[pairIndex],
                route: .init(kind: .exactQwen35MTP),
                fallback: .init(scalarFallbackStartCount: 0),
                exactness: .init(
                    tokenObservationMode: .decodedRoundTrip,
                    scalarDirectTokenCount: 128,
                    mtpUsageCompletionTokenCount: 128,
                    scalarDirectTokenIDsSHA256:
                        hex("profile-token-\(caseID)-\(pairIndex)"),
                    mtpDecodedRoundTripTokenIDsSHA256:
                        hex("profile-token-\(caseID)-\(pairIndex)"),
                    scalarDecodedBytesSHA256: hex("profile-decoded-\(caseID)-\(pairIndex)"),
                    mtpDecodedBytesSHA256: hex("profile-decoded-\(caseID)-\(pairIndex)"),
                    scalarStopOutcome: .length,
                    mtpStopOutcome: .length,
                    scalarCacheFingerprint: cacheFingerprint("profile-cache-\(caseID)-\(pairIndex)"),
                    mtpCacheFingerprint: cacheFingerprint("profile-cache-\(caseID)-\(pairIndex)"),
                    firstCacheMismatch: nil),
                scalarUsage: .init(
                    promptTokens: 16,
                    completionTokens: 128,
                    totalTokens: 144),
                mtpUsage: .init(
                    promptTokens: 16,
                    completionTokens: 128,
                    totalTokens: 144),
                scalarE2ESeconds: scalarSeconds,
                mtpE2ESeconds: mtpSeconds,
                scalarTokensPerSecond: 128 / scalarSeconds,
                mtpTokensPerSecond: 128 / mtpSeconds,
                e2eRatio: ratio,
                mtpTelemetry: .init(
                    proposedDraftTokens: 10,
                    acceptedDraftTokens: 8,
                    rejectedDraftTokens: 2,
                    roundCount: 5,
                    targetModelCallCount: 5,
                    draftModelCallCount: 5,
                    targetVerifiedTokenCount: 15,
                    emittedTokenCount: 128),
                passthroughReason: nil))
        }
    }
    return samples
}

private func makeProvenance(
    harnessGitSHA: String = "0123456789abcdef0123456789abcdef01234567"
) -> Provenance {
    Provenance(
        date: "2026-08-24T00:00:00Z",
        hardwareChip: "Apple M5",
        hardwareRAMBytes: 25_769_803_776,
        hardwareOS: "macOS 16.0.0",
        harnessGitSHA: harnessGitSHA,
        mlxSwiftVersion: "0.31.6",
        referenceMLXVersion: nil,
        referenceMLXLMVersion: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
        modelPath: QwenMTPCorpusGate.requiredBinding.targetModelID,
        modelConfigHash: "5a99be4477ebdac8",
        modelCheckpointManifestHash: "db2b2480a8525194",
        modelQuant: .init(bits: 4, groupSize: 64),
        corpusId: QwenMTPCorpusGate.corpusID,
        corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
        nonce: "nonce")
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
                stateCount: 2,
                states: [
                    .init(
                        stateIndex: 0,
                        shape: [1, 4, 8, 64],
                        dtype: "float16",
                        byteCount: 4096,
                        sha256: hex("state-k-\(seed)")),
                    .init(
                        stateIndex: 1,
                        shape: [1, 4, 8, 64],
                        dtype: "float16",
                        byteCount: 4096,
                        sha256: hex("state-v-\(seed)")),
                ]),
        ])
}
