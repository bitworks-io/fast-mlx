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
        XCTAssertEqual(evidence.comparisonAxis, .executionMode)
        XCTAssertEqual(evidence.candidate.executionMode, .exactMTP)
        XCTAssertEqual(evidence.reference.executionMode, .scalar)
        XCTAssertNotEqual(evidence.candidate.executionDigest, evidence.reference.executionDigest)
        XCTAssertEqual(evidence.candidate.sourceDigest, evidence.reference.sourceDigest)
        XCTAssertEqual(evidence.candidate.gdnMode, .gdnOn)
        XCTAssertEqual(evidence.reference.gdnMode, .gdnOn)
        XCTAssertEqual(evidence.candidate.launchBinding?.mode, .gdnOn)
        XCTAssertEqual(evidence.reference.launchBinding?.mode, .gdnOn)
        XCTAssertEqual(evidence.candidate.launchBinding?.observedEnv, .enabled)
        XCTAssertEqual(evidence.reference.launchBinding?.observedEnv, .enabled)
        XCTAssertEqual(evidence.liveExactnessProof, trustedProof)
        XCTAssertEqual(evidence.liveExactnessProof?.gdnMode, .gdnOn)
        XCTAssertEqual(evidence.liveExactnessProof?.launchBinding, evidence.candidate.launchBinding)
        XCTAssertEqual(evidence.measurementClass, "dedicated-heavy-256gib")
        XCTAssertEqual(evidence.hardware.chip, trustedRunIdentity.hardwareChip)
        XCTAssertEqual(evidence.hardware.ramBytes, 274_877_906_944)
        XCTAssertEqual(evidence.hardware.osBuild, trustedRunIdentity.hardwareOSBuild)
        XCTAssertEqual(evidence.hardware.hostIdentityDigest, trustedRunIdentity.hostIdentityDigest)
        XCTAssertEqual(evidence.workload, Gate.requiredWorkload)
        XCTAssertEqual(
            evidence.workload.chatTemplateSHA256,
            "b426d0bb02412efa9e44777312cc7df1bf95ea332dc0d2e46376c801f273599d")
        XCTAssertEqual(evidence.workload.contextTokenLimit, 32_768)
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
        XCTAssertEqual(
            Set(evidence.runPlan.schedules.flatMap(\.benchmarkCells).map(\.contextTokens)),
            [.tokens4096, .tokens16384, .tokens32768])
        XCTAssertEqual(
            Set(evidence.runPlan.schedules.flatMap(\.benchmarkCells).map(\.prefixKind)),
            [.cold, .exactWarmPrefix])
        XCTAssertTrue(evidence.runPlan.schedules.allSatisfy {
            $0.lane.kind == .syntheticInProcess && $0.lane.width == $0.concurrency
        })
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

    func testBenchmarkIntegritySchemaFailsClosedAgainstLegacyUntypedEvidence() throws {
        XCTAssertEqual(Gate.schemaVersion, 3)

        var legacy = makeEvidence()
        legacy = recomputed(legacy)
        let legacyJSON = try JSONEncoder().encode(legacy)
        var legacyObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: legacyJSON) as? [String: Any])
        legacyObject["schemaVersion"] = 2
        legacyObject.removeValue(forKey: "comparisonAxis")
        let data = try JSONSerialization.data(withJSONObject: legacyObject)

        XCTAssertThrowsError(
            try JSONDecoder().decode(Qwen38MTPPerformanceScorecardEvidence.self, from: data)
        ) { error in
            XCTAssertTrue(error is DecodingError)
        }

        let record = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: legacy)
        let legacyLine = try record.jsonLine().replacingOccurrences(
            of: "\"schemaVersion\":3",
            with: "\"schemaVersion\":2")
        XCTAssertThrowsError(
            try Gate.validateJSONL(
                Data((legacyLine + "\n").utf8),
                authority: trustedAuthority)) { error in
            XCTAssertEqual(error as? GateError, .schemaVersionMismatch(2))
        }
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

    func testComparisonIdentityVariesOnlyExecutionModeAndDoesNotConflateGDNWithMTP() {
        XCTAssertNoThrow(try validateTrusted(makeEvidence()))

        let labelOnlyAuthority = authority(
            candidate: model(
                executionMode: .exactMTP,
                gdnMode: nil,
                binding: nil,
                label: "candidate-gdn-on"),
            reference: model(
                executionMode: .scalar,
                gdnMode: nil,
                binding: nil,
                label: "reference-gdn-off"))
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: labelOnlyAuthority), authority: labelOnlyAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var sameExecutionModeAuthority = trustedAuthority
        sameExecutionModeAuthority.trustedEngineIdentities.reference.executionMode = .exactMTP
        XCTAssertThrowsError(
            try Gate.validate(
                makeEvidence(authority: sameExecutionModeAuthority),
                authority: sameExecutionModeAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var gdnAxisDriftAuthority = trustedAuthority
        gdnAxisDriftAuthority.trustedEngineIdentities.reference.gdnMode = .gdnOff
        gdnAxisDriftAuthority.trustedEngineIdentities.reference.launchBinding =
            launchBinding(mode: .gdnOff, processIsolationEvidenceID: hex("8"))
        XCTAssertThrowsError(
            try Gate.validate(
                makeEvidence(authority: gdnAxisDriftAuthority),
                authority: gdnAxisDriftAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var unboundAuthority = trustedAuthority
        unboundAuthority.trustedEngineIdentities.candidate.launchBinding = nil
        unboundAuthority.trustedEngineIdentities.reference.launchBinding = nil
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: unboundAuthority), authority: unboundAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var sameProcessAuthority = trustedAuthority
        sameProcessAuthority.trustedEngineIdentities.reference.launchBinding =
            launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4"))
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: sameProcessAuthority), authority: sameProcessAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var envMismatchAuthority = trustedAuthority
        envMismatchAuthority.trustedEngineIdentities.reference.launchBinding =
            launchBinding(
                mode: .gdnOn,
                observedEnv: .disabled,
                processIsolationEvidenceID: hex("5"))
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: envMismatchAuthority), authority: envMismatchAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var forgedDigestAuthority = trustedAuthority
        forgedDigestAuthority.trustedEngineIdentities.candidate.launchBinding =
            launchBinding(
                mode: .gdnOn,
                processIsolationEvidenceID: hex("4"),
                launchDigest: hex("9"))
        forgedDigestAuthority.acceptedLiveExactnessProof.launchBinding =
            forgedDigestAuthority.trustedEngineIdentities.candidate.launchBinding
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: forgedDigestAuthority), authority: forgedDigestAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        var fakeSourceAuthority = trustedAuthority
        fakeSourceAuthority.trustedEngineIdentities.reference.sourceDigest =
            Gate.promptSHA256("different unauthenticated source")
        fakeSourceAuthority.trustedEngineIdentities.reference.launchBinding =
            launchBinding(
                mode: .gdnOn,
                sourceDigest: fakeSourceAuthority.trustedEngineIdentities.reference.sourceDigest,
                processIsolationEvidenceID: hex("9"))
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: fakeSourceAuthority), authority: fakeSourceAuthority)) { error in
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }
    }

    func testFusionOnPerformanceCannotConsumeUnboundOrOffExactnessProof() {
        var unbound = trustedAuthority
        unbound.acceptedLiveExactnessProof.gdnMode = nil
        unbound.acceptedLiveExactnessProof.launchBinding = nil
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: unbound), authority: unbound)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var off = trustedAuthority
        off.acceptedLiveExactnessProof.gdnMode = .gdnOff
        off.acceptedLiveExactnessProof.launchBinding =
            launchBinding(mode: .gdnOff, processIsolationEvidenceID: hex("a"))
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: off), authority: off)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var arbitraryArtifact = trustedAuthority
        arbitraryArtifact.acceptedLiveExactnessProof.artifactID = hex("b")
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: arbitraryArtifact), authority: arbitraryArtifact)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        var arbitrarySource = trustedAuthority
        arbitrarySource.acceptedLiveExactnessProof.sourceID = hex("d")
        XCTAssertThrowsError(try Gate.validate(makeEvidence(authority: arbitrarySource), authority: arbitrarySource)) { error in
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
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

    func testBenchmarkCellsAreCompactTypedAndMustMatchScheduledFixture() {
        let soloPair = firstMeasuredPairIndex(concurrency: 1)
        let c2Pair = firstMeasuredPairIndex(concurrency: 2)

        var candidateCellDrift = makeEvidence()
        candidateCellDrift.pairs[soloPair].candidate.requests[0].benchmarkCell =
            Qwen38MTPPerformanceScorecardBenchmarkCellIdentity(
                contextTokens: .tokens16384,
                prefixKind: .cold,
                fixtureRef: candidateCellDrift.pairs[soloPair].candidate.requests[0].caseID)
        XCTAssertThrowsError(try validateTrusted(candidateCellDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "benchmark cell"))
        }

        var fixtureDrift = makeEvidence()
        fixtureDrift.pairs[soloPair].candidate.requests[0].benchmarkCell =
            Qwen38MTPPerformanceScorecardBenchmarkCellIdentity(
                contextTokens: fixtureDrift.pairs[soloPair].scheduledBenchmarkCells[0].contextTokens,
                prefixKind: fixtureDrift.pairs[soloPair].scheduledBenchmarkCells[0].prefixKind,
                fixtureRef: "case-legacy")
        XCTAssertThrowsError(try validateTrusted(fixtureDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "benchmark cell"))
        }

        var candidateReferenceMismatch = makeEvidence()
        candidateReferenceMismatch.pairs[c2Pair].reference.requests[1].benchmarkCell =
            Qwen38MTPPerformanceScorecardBenchmarkCellIdentity(
                contextTokens: .tokens32768,
                prefixKind: .exactWarmPrefix,
                fixtureRef: candidateReferenceMismatch.pairs[c2Pair].reference.requests[1].caseID)
        XCTAssertThrowsError(try validateTrusted(candidateReferenceMismatch)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: c2Pair, reason: "benchmark cell"))
        }
    }

    func testLaneIdentityDistinguishesSyntheticConcurrencyFromProductionSchedulerLanes() {
        XCTAssertTrue(Gate.runPlan.schedules.allSatisfy {
            $0.lane.kind == .syntheticInProcess
        })

        let soloPair = firstMeasuredPairIndex(concurrency: 1)
        var laneDrift = makeEvidence()
        laneDrift.pairs[soloPair].lane = Qwen38MTPPerformanceScorecardLaneIdentity(
            kind: .productionScheduler,
            width: laneDrift.pairs[soloPair].concurrency,
            laneRef: "prod-lane-a")

        XCTAssertThrowsError(try validateTrusted(laneDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidPair(index: soloPair, reason: "lane identity"))
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
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            evidenceID: hex("c"),
            accepted: true,
            gdnMode: .gdnOn,
            launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4")))
    }

    private var trustedAuthority: Qwen38MTPPerformanceScorecardAuthorityBundle {
        authority(
            candidate: model(executionMode: .exactMTP, gdnMode: .gdnOn),
            reference: model(executionMode: .scalar, gdnMode: .gdnOn))
    }

    private var trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities {
        trustedAuthority.trustedEngineIdentities
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

    private func makeEvidence(
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle? = nil
    ) -> Qwen38MTPPerformanceScorecardEvidence {
        let suppliedAuthority = authority
        let authority = authority ?? trustedAuthority
        var evidence = Qwen38MTPPerformanceScorecardEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Gate.requiredArtifact,
            candidate: authority.trustedEngineIdentities.candidate,
            reference: authority.trustedEngineIdentities.reference,
            liveExactnessProof: authority.acceptedLiveExactnessProof,
            comparisonAxis: .executionMode,
            measurementClass: Gate.measurementClass,
            hardware: .init(
                className: Gate.measurementClass,
                chip: authority.trustedRunIdentity.hardwareChip,
                ramBytes: Gate.requiredRAMBytes,
                osBuild: authority.trustedRunIdentity.hardwareOSBuild,
                hostIdentityDigest: authority.trustedRunIdentity.hostIdentityDigest),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            workload: Gate.requiredWorkload,
            settings: Gate.requiredSettings,
            runPlan: Gate.runPlan,
            pairs: makePairs(authority: authority),
            metrics: .empty,
            verdict: .unqualified)
        if suppliedAuthority == nil {
            evidence.metrics = try! Gate.computeMetrics(evidence, authority: authority)
            evidence.verdict = try! Gate.evaluateCandidate(evidence, authority: authority)
        }
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

    private func makePairs(
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) -> [Qwen38MTPPerformanceScorecardPair] {
        Gate.runPlan.schedules.enumerated().map { offset, schedule in
            Qwen38MTPPerformanceScorecardPair(
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                warmup: schedule.pairIndex < Gate.runPlan.droppedWarmupPairs,
                order: schedule.order,
                scheduledCaseIDs: schedule.caseIDs,
                scheduledBenchmarkCells: schedule.benchmarkCells,
                lane: schedule.lane,
                candidate: makeEngine(
                    identity: authority.trustedEngineIdentities.candidate,
                    schedule: schedule,
                    offset: offset,
                    candidate: true),
                reference: makeEngine(
                    identity: authority.trustedEngineIdentities.reference,
                    schedule: schedule,
                    offset: offset,
                    candidate: false))
        }
    }

    private func authority(
        candidate: Qwen38MTPPerformanceScorecardModel,
        reference: Qwen38MTPPerformanceScorecardModel
    ) -> Qwen38MTPPerformanceScorecardAuthorityBundle {
        Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities(
                candidate: candidate,
                reference: reference),
            trustedRunIdentity: trustedRunIdentity)
    }

    private func model(
        executionMode: Qwen38MTPPerformanceScorecardExecutionMode,
        gdnMode: Qwen38MTPPerformanceScorecardGDNMode?,
        binding: Qwen38MTPPerformanceScorecardLaunchBinding? = nil,
        label: String = "engine"
    ) -> Qwen38MTPPerformanceScorecardModel {
        let isExactMTP = executionMode == .exactMTP
        return Qwen38MTPPerformanceScorecardModel(
            label: label,
            executionMode: executionMode,
            artifact: Gate.requiredArtifact,
            executionDigest: Gate.promptSHA256(
                isExactMTP
                    ? "generic exact mtp execution identity"
                    : "generic scalar execution identity"),
            sourceDigest: sharedSourceDigest,
            gdnMode: gdnMode,
            launchBinding: binding ?? gdnMode.map {
                launchBinding(
                    mode: $0,
                    processIsolationEvidenceID: isExactMTP ? hex("4") : hex("5"))
            })
    }

    private func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        sourceDigest: String? = nil,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv? = nil,
        processIsolationEvidenceID: String,
        launchDigest: String? = nil
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let sourceDigest = sourceDigest ?? sharedSourceDigest
        let observedEnv = observedEnv ?? (mode == .gdnOn ? .enabled : .disabled)
        let processIsolationEvidenceID = processIsolationEvidenceID
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: sourceDigest,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: launchDigest ?? Gate.launchDigest(
                mode: mode,
                sourceDigest: sourceDigest,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }

    private var sharedSourceDigest: String {
        Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
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
                benchmarkCell: schedule.benchmarkCells[requestIndex],
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
        benchmarkCell: Qwen38MTPPerformanceScorecardBenchmarkCellIdentity,
        requestIndex: Int,
        seed: Int,
        e2eSeconds: Double,
        decodeSeconds: Double
    ) -> Qwen38MTPPerformanceScorecardRequestMeasurement {
        Qwen38MTPPerformanceScorecardRequestMeasurement(
            caseID: caseID,
            benchmarkCell: benchmarkCell,
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

    // MARK: - Runner launch equality (chain Slice 4a)

    // The canonical evidence fixture's worker-side launch bindings carry
    // processIsolationEvidenceID hex("4") (candidate) / hex("5")
    // (reference); the runner observations below play the independently
    // minted runner-side IDs.
    private func runnerObservations(
        candidate: String,
        reference: String
    ) -> Qwen38MTPPerformanceScorecardRunnerLaunchObservations {
        Qwen38MTPPerformanceScorecardRunnerLaunchObservations(
            candidateProcessIsolationEvidenceID: candidate,
            referenceProcessIsolationEvidenceID: reference)
    }

    func testRunnerLaunchEqualityAcceptsMatchingObservations() throws {
        let evidence = makeEvidence()
        XCTAssertNoThrow(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("4"),
                    reference: hex("5"))))
    }

    func testRunnerLaunchEqualityRejectsCandidateMismatch() {
        let evidence = makeEvidence()
        XCTAssertThrowsError(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("6"),
                    reference: hex("5")))
        ) { error in
            XCTAssertEqual(
                error as? GateError,
                .runnerLaunchEqualityRejected("candidate"))
        }
    }

    func testRunnerLaunchEqualityRejectsReferenceMismatch() {
        let evidence = makeEvidence()
        XCTAssertThrowsError(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("4"),
                    reference: hex("6")))
        ) { error in
            XCTAssertEqual(
                error as? GateError,
                .runnerLaunchEqualityRejected("reference"))
        }
    }

    func testRunnerLaunchEqualityRejectsSwappedObservations() {
        let evidence = makeEvidence()
        XCTAssertThrowsError(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("5"),
                    reference: hex("4")))
        ) { error in
            XCTAssertEqual(
                error as? GateError,
                .runnerLaunchEqualityRejected("candidate"))
        }
    }

    func testRunnerLaunchEqualityRejectsNonCanonicalOrEqualObservationIDs() {
        let evidence = makeEvidence()
        let rejected: [(String, String, String)] = [
            ("uppercase", String(repeating: "A", count: 64), hex("5")),
            ("short", String(repeating: "4", count: 63), hex("5")),
            ("long", String(repeating: "4", count: 65), hex("5")),
            ("empty", "", hex("5")),
            ("equal IDs", hex("4"), hex("4")),
        ]
        for (label, candidate, reference) in rejected {
            XCTAssertThrowsError(
                try Gate.validateRunnerLaunchEquality(
                    evidence,
                    observations: runnerObservations(
                        candidate: candidate,
                        reference: reference)),
                label
            ) { error in
                XCTAssertEqual(
                    error as? GateError,
                    .invalidRunnerLaunchObservation,
                    label)
            }
        }
    }

    func testRunnerLaunchEqualityRejectsMissingLaunchBindings() {
        let noBindingAuthority = authority(
            candidate: model(executionMode: .exactMTP, gdnMode: nil),
            reference: model(executionMode: .scalar, gdnMode: nil))
        let evidence = makeEvidence(authority: noBindingAuthority)
        XCTAssertThrowsError(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("4"),
                    reference: hex("5")))
        ) { error in
            XCTAssertEqual(
                error as? GateError,
                .runnerLaunchEqualityRejected("candidate launch binding missing"))
        }
    }

    /// Candidate binding present (and matching) so the check reaches the
    /// reference guard — covers the reference-missing fail-closed branch
    /// on its own.
    func testRunnerLaunchEqualityRejectsReferenceLaunchBindingMissingAlone() {
        let mixedAuthority = authority(
            candidate: model(executionMode: .exactMTP, gdnMode: .gdnOn),
            reference: model(executionMode: .scalar, gdnMode: nil))
        let evidence = makeEvidence(authority: mixedAuthority)
        XCTAssertThrowsError(
            try Gate.validateRunnerLaunchEquality(
                evidence,
                observations: runnerObservations(
                    candidate: hex("4"),
                    reference: hex("5")))
        ) { error in
            XCTAssertEqual(
                error as? GateError,
                .runnerLaunchEqualityRejected("reference launch binding missing"))
        }
    }
}
