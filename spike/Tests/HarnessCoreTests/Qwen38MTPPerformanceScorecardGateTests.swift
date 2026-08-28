import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38MTPPerformanceScorecardGateTests: XCTestCase {
    private typealias Gate = Qwen38MTPPerformanceScorecardGate
    private typealias GateError = Qwen38MTPPerformanceScorecardGateError

    func testFrozenRunBudgetPinsEngineAndRequestCardinality() {
        let budget = Gate.runPlan.budget

        XCTAssertEqual(budget.pairRecords, 126)
        XCTAssertEqual(budget.engineMeasurements, 252)
        XCTAssertEqual(budget.requestMeasurementsIncludingWarmups, 588)
        XCTAssertEqual(budget.warmupRequestMeasurements, 28)
        XCTAssertEqual(budget.measuredRequestMeasurements, 560)
        XCTAssertEqual(
            budget.perConcurrency,
            [
                Qwen38MTPPerformanceScorecardConcurrencyBudget(
                    concurrency: 1,
                    pairRecords: 42,
                    warmupPairs: 2,
                    measuredPairs: 40,
                    engineMeasurements: 84,
                    requestMeasurementsIncludingWarmups: 84,
                    warmupRequestMeasurements: 4,
                    measuredRequestMeasurements: 80),
                Qwen38MTPPerformanceScorecardConcurrencyBudget(
                    concurrency: 2,
                    pairRecords: 42,
                    warmupPairs: 2,
                    measuredPairs: 40,
                    engineMeasurements: 84,
                    requestMeasurementsIncludingWarmups: 168,
                    warmupRequestMeasurements: 8,
                    measuredRequestMeasurements: 160),
                Qwen38MTPPerformanceScorecardConcurrencyBudget(
                    concurrency: 4,
                    pairRecords: 42,
                    warmupPairs: 2,
                    measuredPairs: 40,
                    engineMeasurements: 84,
                    requestMeasurementsIncludingWarmups: 336,
                    warmupRequestMeasurements: 16,
                    measuredRequestMeasurements: 320),
            ])
    }

    func testPublicValidationFailsClosedUntilExactnessEngineAndRunIdentitiesArePromoted() throws {
        let evidence = makeEvidence()

        XCTAssertNil(Gate.requiredAcceptedLiveExactnessProof)
        XCTAssertNil(Gate.requiredTrustedEngineIdentities)
        XCTAssertNil(Gate.requiredTrustedRunIdentity)
        XCTAssertThrowsError(try Gate.validate(evidence)) { error in
            XCTAssertEqual(error as? GateError, .liveExactnessNotPromoted)
        }

        let record = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: evidence)
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try record.jsonLine() + "\n").utf8))) { error in
            XCTAssertEqual(error as? GateError, .liveExactnessNotPromoted)
        }

        XCTAssertThrowsError(
            try Gate.validate(evidence, trustedLiveExactnessProof: trustedProof)) { error in
            XCTAssertEqual(error as? GateError, .performanceIdentityNotPromoted)
        }

        XCTAssertThrowsError(
            try Gate.validate(
                evidence,
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities)) { error in
            XCTAssertEqual(error as? GateError, .runIdentityNotPromoted)
        }
    }

    func testTrustedSyntheticScorecardQualifiesAndPinsFrozen27BContract() throws {
        let evidence = makeEvidence()

        let verdict = try validateTrusted(evidence)

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(Gate.requiredArtifactLock, QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1)
        XCTAssertEqual(evidence.schemaVersion, Gate.schemaVersion)
        XCTAssertEqual(evidence.artifact, Gate.requiredArtifact)
        XCTAssertEqual(evidence.artifact.depth, 1)
        XCTAssertEqual(evidence.artifact.blockSize, 3)
        XCTAssertEqual(evidence.artifact.maxAcceptedDrafts, 2)
        XCTAssertEqual(evidence.artifact.targetQuantizationGroupSize, 32)
        XCTAssertEqual(evidence.artifact.drafterQuantizationGroupSize, 32)
        XCTAssertEqual(evidence.artifact.targetQuantizationMode, "mxfp8")
        XCTAssertEqual(evidence.artifact.drafterQuantizationMode, "mxfp8")
        XCTAssertEqual(evidence.candidate, trustedEngineIdentities.candidate)
        XCTAssertEqual(evidence.reference, trustedEngineIdentities.reference)
        XCTAssertEqual(evidence.reference.artifact, evidence.candidate.artifact)
        XCTAssertNotEqual(evidence.candidate.executionDigest, evidence.reference.executionDigest)
        XCTAssertNotEqual(evidence.candidate.sourceDigest, evidence.reference.sourceDigest)
        XCTAssertEqual(evidence.liveExactnessProof, trustedProof)
        XCTAssertEqual(evidence.measurementClass, "dedicated-heavy-256gib")
        XCTAssertEqual(evidence.hardware.chip, trustedRunIdentity.hardwareChip)
        XCTAssertEqual(evidence.hardware.ramBytes, 274_877_906_944)
        XCTAssertEqual(evidence.hardware.osBuild, trustedRunIdentity.hardwareOSBuild)
        XCTAssertEqual(evidence.hardware.hostIdentityDigest, trustedRunIdentity.hostIdentityDigest)
        XCTAssertEqual(evidence.workload, Gate.requiredWorkload)
        XCTAssertEqual(
            evidence.workload.chatTemplateSHA256,
            "b426d0bb02412efa9e44777312cc7df1bf95ea332dc0d2e46376c801f273599d")
        XCTAssertEqual(evidence.workload.contextTokenLimit, 40_960)
        XCTAssertEqual(evidence.workload.cases.count, 6)
        XCTAssertEqual(
            evidence.workload.cases.map(\.promptSHA256),
            evidence.workload.cases.map { Gate.promptSHA256($0.prompt) })
        XCTAssertEqual(
            evidence.workload.contentSHA256,
            Gate.canonicalWorkloadContentSHA256(evidence.workload.cases))
        XCTAssertEqual(evidence.runPlan.concurrencies, [1, 2, 4])
        XCTAssertEqual(evidence.runPlan.droppedWarmupPairs, 2)
        XCTAssertEqual(evidence.runPlan.measuredPairs, 40)
        XCTAssertEqual(evidence.runPlan.schedules.count, 126)
        assertMeasuredHalvesAreBalanced(evidence.runPlan)
        XCTAssertEqual(evidence.metrics, try computeTrustedMetrics(evidence))
        XCTAssertEqual(verdict, evidence.verdict)
        XCTAssertGreaterThanOrEqual(verdict.chronologicalFirstHalf.minimum, 0.98)
        XCTAssertGreaterThanOrEqual(verdict.chronologicalSecondHalf.minimum, 0.98)

        let json = try XCTUnwrap(String(data: try JSONEncoder().encode(evidence), encoding: .utf8))
        XCTAssertFalse(json.contains("mlx-community"))
        XCTAssertFalse(json.contains("http"))
        XCTAssertFalse(json.contains("/" + "Users/"))
        XCTAssertFalse(json.contains("192" + ".168."))
    }

    func testTrustedValidationRejectsAnyProofOrEngineIdentityOtherThanInjectedExactObjects() {
        var missing = makeEvidence()
        missing.liveExactnessProof = nil
        XCTAssertThrowsError(try validateTrusted(missing)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var selfAttested = makeEvidence()
        selfAttested.liveExactnessProof?.evidenceID = hex("a")
        XCTAssertThrowsError(try validateTrusted(selfAttested)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var rejected = makeEvidence()
        rejected.liveExactnessProof?.accepted = false
        XCTAssertThrowsError(try validateTrusted(rejected)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var driftedTrustedIdentities = trustedEngineIdentities
        driftedTrustedIdentities.candidate.executionDigest = Gate.promptSHA256(
            "different generic candidate execution")
        XCTAssertThrowsError(
            try Gate.validate(
                makeEvidence(),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: driftedTrustedIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var driftedRunIdentity = trustedRunIdentity
        driftedRunIdentity.hostIdentityDigest = Gate.promptSHA256("different generic heavy host")
        XCTAssertThrowsError(
            try Gate.validate(
                makeEvidence(),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: driftedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .invalidRunIdentity)
        }
    }

    func testPublicAuthorityBundleValidatesScorecardAndJSONLWithoutPromotedDefaults() throws {
        let evidence = makeEvidence()
        let authority = Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)

        XCTAssertNil(Gate.requiredAcceptedLiveExactnessProof)
        XCTAssertNil(Gate.requiredTrustedEngineIdentities)
        XCTAssertNil(Gate.requiredTrustedRunIdentity)
        XCTAssertEqual(try Gate.validate(evidence, authority: authority), evidence.verdict)

        let record = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: evidence)
        XCTAssertEqual(
            try Gate.validateJSONL(
                Data((try record.jsonLine() + "\n").utf8),
                authority: authority),
            [evidence.verdict])

        var driftedAuthority = authority
        driftedAuthority.trustedRunIdentity.hostIdentityDigest =
            Gate.promptSHA256("different generic heavy host identity")
        XCTAssertThrowsError(try Gate.validate(evidence, authority: driftedAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidRunIdentity)
        }
    }

    func testWorkloadCorpusTemplateAndContextAreFrozen() {
        var corpusDrift = makeEvidence()
        corpusDrift.workload.cases[0].prompt = "Describe a drifted generic prompt."
        corpusDrift.workload.cases[0].promptSHA256 = Gate.promptSHA256(
            corpusDrift.workload.cases[0].prompt)
        corpusDrift.workload.contentSHA256 = Gate.canonicalWorkloadContentSHA256(
            corpusDrift.workload.cases)
        XCTAssertThrowsError(try validateTrusted(corpusDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidWorkload)
        }

        var arbitraryContentHash = makeEvidence()
        arbitraryContentHash.workload.contentSHA256 = hex("d")
        XCTAssertThrowsError(try validateTrusted(arbitraryContentHash)) { error in
            XCTAssertEqual(error as? GateError, .invalidWorkload)
        }

        var templateDrift = makeEvidence()
        templateDrift.workload.chatTemplateSHA256 = hex("e")
        XCTAssertThrowsError(try validateTrusted(templateDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidWorkload)
        }

        var contextDrift = makeEvidence()
        contextDrift.workload.contextTokenLimit = 16_384
        XCTAssertThrowsError(try validateTrusted(contextDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidWorkload)
        }
    }

    func testRejectsWrongArtifactProjectionAndEngineIdentityDrift() {
        var wrongArtifact = makeEvidence()
        wrongArtifact.artifact.maxAcceptedDrafts = 1
        XCTAssertThrowsError(try validateTrusted(wrongArtifact)) { error in
            XCTAssertEqual(error as? GateError, .invalidArtifactBinding)
        }

        var identityDrift = makeEvidence()
        identityDrift.reference.executionDigest = identityDrift.candidate.executionDigest
        XCTAssertThrowsError(try validateTrusted(identityDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }
    }

    func testRejectsScheduleRequestCountOutputCacheAndDecodeTokenForgery() {
        let c2Pair = firstMeasuredPairIndex(concurrency: 2)
        let nextC2Pair = c2Pair + 1
        let thirdC2Pair = c2Pair + 2

        var requestCount = makeEvidence()
        requestCount.pairs[c2Pair].candidate.requests.removeLast()
        XCTAssertThrowsError(try validateTrusted(requestCount)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: c2Pair, reason: "request count"))
        }

        var scheduleDrift = makeEvidence()
        let soloPair = firstMeasuredPairIndex(concurrency: 1)
        scheduleDrift.pairs[soloPair].candidate.requests[0].caseID = "case-06"
        XCTAssertThrowsError(try validateTrusted(scheduleDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "request schedule"))
        }

        var outputDrift = makeEvidence()
        outputDrift.pairs[c2Pair].reference.requests[1].outputDigest = hex("b")
        XCTAssertThrowsError(try validateTrusted(outputDrift)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: c2Pair, reason: "output/cache provenance"))
        }

        var tokenParityDrift = makeEvidence()
        tokenParityDrift.pairs[nextC2Pair].candidate.requests[0].decodeTokenCount = 101
        XCTAssertThrowsError(try validateTrusted(tokenParityDrift)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: nextC2Pair, reason: "decode token parity"))
        }

        var forgedInflation = makeEvidence()
        forgedInflation.pairs[thirdC2Pair].candidate.requests[0].decodeTokenCount = 10_000
        forgedInflation.pairs[thirdC2Pair].reference.requests[0].decodeTokenCount = 10_000
        XCTAssertThrowsError(try validateTrusted(forgedInflation)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: thirdC2Pair, reason: "max completion tokens"))
        }
    }

    func testRejectsRawMeasurementMathAndThermalFailures() {
        let soloPair = firstMeasuredPairIndex(concurrency: 1)
        let c2Pair = firstMeasuredPairIndex(concurrency: 2)

        var acceptedOverflow = makeEvidence()
        acceptedOverflow.pairs[0].candidate.acceptedCount =
            acceptedOverflow.pairs[0].candidate.proposalCount + 1
        XCTAssertThrowsError(try validateTrusted(acceptedOverflow)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: 0, reason: "draft activity"))
        }

        var aggregateWallMismatch = makeEvidence()
        aggregateWallMismatch.pairs[soloPair].candidate.wallSeconds = 11
        XCTAssertThrowsError(try validateTrusted(aggregateWallMismatch)) { error in
            XCTAssertEqual(error as? GateError, .metricsMismatch)
        }

        var wallTooShort = makeEvidence()
        wallTooShort.pairs[soloPair].candidate.wallSeconds = 1
        XCTAssertThrowsError(try validateTrusted(wallTooShort)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "wall/request timing"))
        }

        var impossibleDecodeWindow = makeEvidence()
        impossibleDecodeWindow.pairs[soloPair].candidate.requests[0].ttftSeconds = 8.5
        XCTAssertThrowsError(try validateTrusted(impossibleDecodeWindow)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "request timing"))
        }

        var overflow = makeEvidence()
        overflow.pairs[c2Pair].candidate.requests[0].decodeTokenCount = Int.max
        overflow.pairs[c2Pair].candidate.requests[1].decodeTokenCount = Int.max
        XCTAssertThrowsError(try validateTrusted(overflow)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: c2Pair, reason: "token overflow"))
        }

        var seriousThermal = makeEvidence()
        seriousThermal.pairs[soloPair].candidate.thermalAfter = "serious"
        XCTAssertThrowsError(try validateTrusted(seriousThermal)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "thermal"))
        }
    }

    func testRejectsFallbackPassthroughSamplingToolsPenaltiesAndStreaming() {
        let soloPair = firstMeasuredPairIndex(concurrency: 1)

        var candidateFallback = makeEvidence()
        candidateFallback.pairs[soloPair].candidate.fallbackUsed = true
        XCTAssertThrowsError(try validateTrusted(candidateFallback)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: soloPair, reason: "fallback/passthrough"))
        }

        var candidatePassthrough = makeEvidence()
        candidatePassthrough.pairs[soloPair].candidate.passthroughUsed = true
        XCTAssertThrowsError(try validateTrusted(candidatePassthrough)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: soloPair, reason: "fallback/passthrough"))
        }

        var referenceFallback = makeEvidence()
        referenceFallback.pairs[soloPair].reference.fallbackUsed = true
        XCTAssertThrowsError(try validateTrusted(referenceFallback)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: soloPair, reason: "fallback/passthrough"))
        }

        var referencePassthrough = makeEvidence()
        referencePassthrough.pairs[soloPair].reference.passthroughUsed = true
        XCTAssertThrowsError(try validateTrusted(referencePassthrough)) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: soloPair, reason: "fallback/passthrough"))
        }

        var nonGreedyTemperature = makeEvidence()
        nonGreedyTemperature.settings.temperature = 0.1
        XCTAssertThrowsError(try validateTrusted(nonGreedyTemperature)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("greedy"))
        }

        var topP = makeEvidence()
        topP.settings.topP = 0.95
        XCTAssertThrowsError(try validateTrusted(topP)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("sampling"))
        }

        var topK = makeEvidence()
        topK.settings.topK = 40
        XCTAssertThrowsError(try validateTrusted(topK)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("sampling"))
        }

        var minP = makeEvidence()
        minP.settings.minP = 0.05
        XCTAssertThrowsError(try validateTrusted(minP)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("sampling"))
        }

        var seed = makeEvidence()
        seed.settings.seed = 7
        XCTAssertThrowsError(try validateTrusted(seed)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("sampling"))
        }

        var tools = makeEvidence()
        tools.settings.toolsEmpty = false
        XCTAssertThrowsError(try validateTrusted(tools)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("tools"))
        }

        var penalties = makeEvidence()
        penalties.settings.penaltiesDisabled = false
        XCTAssertThrowsError(try validateTrusted(penalties)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("penalties"))
        }

        var streaming = makeEvidence()
        streaming.settings.streaming = true
        XCTAssertThrowsError(try validateTrusted(streaming)) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("streaming"))
        }
    }

    func testPerRequestP95CannotBeHiddenByGoodRunAverage() {
        var evidence = makeEvidence()
        for pairIndex in measuredPairIndices(concurrency: 4).prefix(3) {
            for requestIndex in evidence.pairs[pairIndex].candidate.requests.indices {
                evidence.pairs[pairIndex].candidate.requests[requestIndex].e2eSeconds = 30
                evidence.pairs[pairIndex].candidate.requests[requestIndex].ttftSeconds = 1
            }
            evidence.pairs[pairIndex].candidate.wallSeconds = 30
        }
        evidence = recomputed(evidence)

        XCTAssertThrowsError(try validateTrusted(evidence)) { error in
            XCTAssertEqual(error as? GateError, .unqualifiedPerformance)
        }
    }

    func testSecondHalfC2AggregateThroughputUsesRawHalfSumsNotMedianPairRatios() {
        var evidence = makeEvidence()
        let c2Measured = measuredPairIndices(concurrency: 2)
        let secondHalf = Array(c2Measured.dropFirst(c2Measured.count / 2))
        evidence.pairs[secondHalf[0]].candidate.wallSeconds = 35.0
        evidence = recomputed(evidence)

        let perPairMedian = median(secondHalf.map {
            10.0 / evidence.pairs[$0].candidate.wallSeconds
        })
        XCTAssertEqual(secondHalf.count, 20)
        XCTAssertGreaterThanOrEqual(perPairMedian, 0.98)
        XCTAssertLessThan(evidence.verdict.chronologicalSecondHalf.c2AggregateThroughput, 0.98)
        XCTAssertThrowsError(try validateTrusted(evidence)) { error in
            XCTAssertEqual(error as? GateError, .unqualifiedPerformance)
        }
    }

    func testStoredAbsoluteHeadlineMetricsAreRecomputedAndSealed() throws {
        var evidence = makeEvidence()
        let solo = try XCTUnwrap(evidence.metrics.perConcurrency.first { $0.concurrency == 1 })

        XCTAssertEqual(solo.candidate.prompt.p50, 0.25)
        XCTAssertEqual(solo.candidate.prefill.p95, 0.80)
        XCTAssertEqual(solo.candidate.ttft.p95, 1.00)
        XCTAssertEqual(solo.candidate.decodeTokensPerSecond.p50, 100 / 0.90, accuracy: 1e-12)
        XCTAssertEqual(solo.candidate.e2e.p95, 9.0)
        XCTAssertEqual(
            solo.candidate.aggregateThroughputTokensPerSecond,
            4_000 / 360.0,
            accuracy: 1e-12)
        XCTAssertEqual(solo.candidate.peakRSSBytes, 210_000_000_000)
        XCTAssertEqual(solo.candidate.peakMetalBytes, 190_000_000_000)
        XCTAssertEqual(solo.reference.aggregateThroughputTokensPerSecond, 4_000 / 400.0, accuracy: 1e-12)

        evidence.metrics.perConcurrency[0].candidate.e2e.p50 = 42
        XCTAssertThrowsError(try validateTrusted(evidence)) { error in
            XCTAssertEqual(error as? GateError, .metricsMismatch)
        }
    }

    func testJSONLTrustedValidationPreservesSingleRecordAndSubcommandRules() throws {
        let record = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: makeEvidence())
        let valid = Data((try record.jsonLine() + "\n").utf8)

        let verdicts = try Gate.validateJSONL(
            valid,
            trustedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)

        XCTAssertEqual(verdicts, [makeEvidence().verdict])

        let rejected = ResultRecord(
            subcommand: Gate.rejectedSubcommand,
            provenance: makeProvenance(),
            payload: makeEvidence())
        XCTAssertThrowsError(
            try Gate.validateJSONL(
                Data((try rejected.jsonLine() + "\n").utf8),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .wrongSubcommand(Gate.rejectedSubcommand))
        }

        XCTAssertThrowsError(
            try Gate.validateJSONL(
                Data("{bad json}\n".utf8),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .malformedJSONL(line: 1))
        }

        XCTAssertThrowsError(
            try Gate.validateJSONL(
                Data((try record.jsonLine()).utf8),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }

        let extra = Data((try record.jsonLine() + "\n" + record.jsonLine() + "\n").utf8)
        XCTAssertThrowsError(
            try Gate.validateJSONL(
                extra,
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .invalidRecordCardinality(2))
        }

        let wrongRunProvenance = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(harnessGitSHA: String(repeating: "2", count: 40)),
            payload: makeEvidence())
        XCTAssertThrowsError(
            try Gate.validateJSONL(
                Data((try wrongRunProvenance.jsonLine() + "\n").utf8),
                trustedLiveExactnessProof: trustedProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity)) { error in
            XCTAssertEqual(error as? GateError, .invalidProvenance("record"))
        }
    }

    private var trustedProof: Qwen38MTPPerformanceScorecardLiveExactnessProof {
        Qwen38MTPPerformanceScorecardLiveExactnessProof(
            artifact: Gate.requiredArtifact,
            artifactID: hex("a"),
            sourceID: hex("b"),
            evidenceID: hex("c"),
            accepted: true)
    }

    private var trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities {
        Qwen38MTPPerformanceScorecardTrustedEngineIdentities(
            candidate: Qwen38MTPPerformanceScorecardModel(
                label: "candidate",
                artifact: Gate.requiredArtifact,
                executionDigest: Gate.promptSHA256("generic candidate execution identity"),
                sourceDigest: Gate.promptSHA256("generic candidate source identity")),
            reference: Qwen38MTPPerformanceScorecardModel(
                label: "reference",
                artifact: Gate.requiredArtifact,
                executionDigest: Gate.promptSHA256("generic reference execution identity"),
                sourceDigest: Gate.promptSHA256("generic reference source identity")))
    }

    private var trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: Gate.measurementClass,
            hardwareChip: "generic-heavy-chip",
            hardwareRAMBytes: Gate.requiredRAMBytes,
            hardwareOSBuild: "generic-os-build-2026-08-24",
            hostIdentityDigest: Gate.promptSHA256("generic dedicated heavy host identity"),
            harnessGitSHA: String(repeating: "1", count: 40),
            candidateMLXSwiftVersion: "generic-mlx-swift-framework-1",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: Gate.modelArtifactLabel,
            modelConfigHash: Gate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: Gate.requiredWorkload.id,
            corpusContentHash: Gate.requiredWorkload.contentSHA256)
    }

    private func validateTrusted(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        try Gate.validate(
            evidence,
            trustedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    private func computeTrustedMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        try Gate.computeMetrics(
            evidence,
            trustedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    private func makeEvidence() -> Qwen38MTPPerformanceScorecardEvidence {
        var evidence = Qwen38MTPPerformanceScorecardEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Gate.requiredArtifact,
            candidate: trustedEngineIdentities.candidate,
            reference: trustedEngineIdentities.reference,
            liveExactnessProof: trustedProof,
            measurementClass: Gate.measurementClass,
            hardware: .init(
                className: Gate.measurementClass,
                chip: trustedRunIdentity.hardwareChip,
                ramBytes: Gate.requiredRAMBytes,
                osBuild: trustedRunIdentity.hardwareOSBuild,
                hostIdentityDigest: trustedRunIdentity.hostIdentityDigest),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            workload: Gate.requiredWorkload,
            settings: Gate.requiredSettings,
            runPlan: Gate.runPlan,
            pairs: makePairs(),
            metrics: .empty,
            verdict: .unqualified)
        evidence.metrics = try! computeTrustedMetrics(evidence)
        evidence.verdict = try! Gate.evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        return evidence
    }

    private func recomputed(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) -> Qwen38MTPPerformanceScorecardEvidence {
        var updated = evidence
        updated.metrics = try! computeTrustedMetrics(updated)
        updated.verdict = try! Gate.evaluateCandidate(
            updated,
            trustedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        return updated
    }

    private func makePairs() -> [Qwen38MTPPerformanceScorecardPair] {
        Gate.runPlan.schedules.enumerated().map { offset, schedule in
            Qwen38MTPPerformanceScorecardPair(
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                warmup: schedule.pairIndex < Gate.runPlan.droppedWarmupPairs,
                order: schedule.order,
                scheduledCaseIDs: schedule.caseIDs,
                candidate: makeEngine(
                    identity: trustedEngineIdentities.candidate,
                    schedule: schedule,
                    offset: offset,
                    candidate: true),
                reference: makeEngine(
                    identity: trustedEngineIdentities.reference,
                    schedule: schedule,
                    offset: offset,
                    candidate: false))
        }
    }

    private func makeEngine(
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        offset: Int,
        candidate: Bool
    ) -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let e2e = candidate ? 9.0 : 10.0
        let decodeSeconds = candidate ? 0.90 : 1.0
        let requests = schedule.caseIDs.enumerated().map { requestIndex, caseID in
            makeRequest(
                caseID: caseID,
                requestIndex: requestIndex,
                seed: offset + requestIndex,
                e2eSeconds: e2e,
                decodeSeconds: decodeSeconds)
        }
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: identity,
            requests: requests,
            wallSeconds: e2e,
            peakRSSBytes: 210_000_000_000,
            peakMetalBytes: 190_000_000_000,
            thermalBefore: "nominal",
            thermalAfter: "fair",
            proposalCount: candidate ? 12 * schedule.concurrency : 0,
            acceptedCount: candidate ? 8 * schedule.concurrency : 0,
            fallbackUsed: false,
            passthroughUsed: false)
    }

    private func makeRequest(
        caseID: String,
        requestIndex: Int,
        seed: Int,
        e2eSeconds: Double,
        decodeSeconds: Double
    ) -> Qwen38MTPPerformanceScorecardRequestMeasurement {
        Qwen38MTPPerformanceScorecardRequestMeasurement(
            caseID: caseID,
            requestIndex: requestIndex,
            promptSeconds: 0.25,
            prefillSeconds: 0.80,
            ttftSeconds: 1.00,
            decodeTokenCount: 100,
            decodeSeconds: decodeSeconds,
            e2eSeconds: e2eSeconds,
            outputDigest: digest(seed, "output"),
            cacheDigest: digest(seed, "cache"),
            outputProvenanceID: digest(seed, "output-provenance"),
            cacheProvenanceID: digest(seed, "cache-provenance"))
    }

    private func firstMeasuredPairIndex(concurrency: Int) -> Int {
        measuredPairIndices(concurrency: concurrency)[0]
    }

    private func measuredPairIndices(concurrency: Int) -> [Int] {
        Gate.runPlan.schedules.enumerated().compactMap { index, schedule in
            schedule.concurrency == concurrency
                && schedule.pairIndex >= Gate.runPlan.droppedWarmupPairs
                ? index
                : nil
        }
    }

    private func assertMeasuredHalvesAreBalanced(
        _ runPlan: Qwen38MTPPerformanceScorecardRunPlan,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for concurrency in runPlan.concurrencies {
            let schedules = runPlan.schedules.filter { $0.concurrency == concurrency }
            XCTAssertEqual(schedules.count, 42, file: file, line: line)
            XCTAssertEqual(
                schedules.filter { $0.pairIndex < runPlan.droppedWarmupPairs }.count,
                2,
                file: file,
                line: line)
            let measured = schedules.dropFirst(runPlan.droppedWarmupPairs)
            XCTAssertEqual(measured.count, 40, file: file, line: line)
            for half in [Array(measured.prefix(20)), Array(measured.suffix(20))] {
                XCTAssertEqual(half.count, 20, file: file, line: line)
                XCTAssertEqual(
                    half.filter { $0.order == .candidateThenReference }.count,
                    10,
                    file: file,
                    line: line)
                XCTAssertEqual(
                    half.filter { $0.order == .referenceThenCandidate }.count,
                    10,
                    file: file,
                    line: line)
            }
        }
    }

    private func makeProvenance(
        modelPath: String? = nil,
        harnessGitSHA: String? = nil
    ) -> Provenance {
        Provenance(
            date: "2026-08-24T00:00:00Z",
            hardwareChip: trustedRunIdentity.hardwareChip,
            hardwareRAMBytes: trustedRunIdentity.hardwareRAMBytes,
            hardwareOS: trustedRunIdentity.hardwareOSBuild,
            harnessGitSHA: harnessGitSHA ?? trustedRunIdentity.harnessGitSHA,
            mlxSwiftVersion: trustedRunIdentity.candidateMLXSwiftVersion,
            referenceMLXVersion: trustedRunIdentity.referenceMLXVersion,
            referenceMLXLMVersion: trustedRunIdentity.referenceMLXLMVersion,
            modelPath: modelPath ?? trustedRunIdentity.modelLabel,
            modelConfigHash: trustedRunIdentity.modelConfigHash,
            modelCheckpointManifestHash: trustedRunIdentity.modelCheckpointManifestHash,
            modelQuant: trustedRunIdentity.modelQuant,
            corpusId: trustedRunIdentity.corpusID,
            corpusContentHash: trustedRunIdentity.corpusContentHash,
            nonce: "scorecard-fixture")
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    private func digest(_ seed: Int, _ salt: String) -> String {
        let bytes = Array("\(salt)-\(seed)".utf8)
        let alphabet = Array("0123456789abcdef")
        return String((0..<64).map { alphabet[Int(bytes[$0 % bytes.count]) % alphabet.count] })
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}
