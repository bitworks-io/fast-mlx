import Foundation
import XCTest
@testable import HarnessCore

final class QwenMTPCorpusGateTests: XCTestCase {
    func testFrozenCorpusIdentityAndCaseOrder() throws {
        XCTAssertEqual(QwenMTPCorpusGate.schemaVersion, 4)
        XCTAssertEqual(QwenMTPCorpusGate.corpusID, "qwen3.5-9b-mtp-consumer-corpus-v1")
        XCTAssertEqual(QwenMTPCorpusGate.corpusContentHash, "5e3bc2fbb016d5e0")

        let cases = QwenMTPCorpusGate.cases
        XCTAssertEqual(cases.map(\.id), [
            "prime-sequence",
            "swift-code",
            "strict-json-list",
            "math-reasoning",
            "long-retrieval",
            "dialogue-tool-like",
            "length-boundary-1",
            "length-boundary-2",
            "cancel-retained-1",
            "cancel-after-accepted-draft",
            "forced-fallback-seeded",
        ])
        XCTAssertEqual(cases.filter { $0.kind == .fullGreedy }.count, 6)
        XCTAssertEqual(cases.filter { $0.maxTokens == 128 }.count, 9)
        XCTAssertEqual(cases.first { $0.id == "length-boundary-1" }?.maxTokens, 1)
        XCTAssertEqual(cases.first { $0.id == "length-boundary-2" }?.maxTokens, 2)
        XCTAssertGreaterThanOrEqual(cases.first { $0.id == "long-retrieval" }?.prompt.utf8.count ?? 0, 6_000)
        XCTAssertEqual(Set(cases.map(\.id)).count, cases.count)
    }

    func testProfilePlanFreezesWarmupAndMeasuredOrder() {
        let plan = QwenMTPCorpusGate.profilePlan
        XCTAssertEqual(plan.droppedWarmupPairs, 2)
        XCTAssertEqual(plan.measuredPairs, 5)
        XCTAssertEqual(plan.totalPairsPerCase, 7)
        XCTAssertEqual(plan.caseIDs, [
            "prime-sequence",
            "swift-code",
            "strict-json-list",
            "math-reasoning",
            "long-retrieval",
            "dialogue-tool-like",
        ])
        XCTAssertEqual(plan.orders, [
            .scalarThenMTP,
            .mtpThenScalar,
            .scalarThenMTP,
            .mtpThenScalar,
            .scalarThenMTP,
            .mtpThenScalar,
            .scalarThenMTP,
        ])
    }

    func testScalarRetainedLimitPolicyForCancellationCases() {
        XCTAssertEqual(
            QwenMTPCorpusGate.scalarRetainedTokenLimit(forCaseID: "cancel-retained-1"),
            1)
        XCTAssertNil(QwenMTPCorpusGate.scalarRetainedTokenLimit(forCaseID: "prime-sequence"))
        XCTAssertNil(QwenMTPCorpusGate.scalarRetainedTokenLimit(forCaseID: "length-boundary-1"))
    }

    func testCompleteCorrectnessEvidencePassesClosedGate() throws {
        let payload = makePayload()

        let verdict = try QwenMTPCorpusGate.validate(payload)

        XCTAssertEqual(verdict.correctness, .pass)
        XCTAssertNil(verdict.profile)
    }

    func testEvidenceValidationFailsClosedOnMissingReorderedOrMismatchedCases() {
        var missing = makePayload()
        missing.caseResults.removeLast()
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(missing))

        var reordered = makePayload()
        reordered.caseResults.swapAt(0, 1)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(reordered))

        var mismatch = makePayload()
        mismatch.caseResults[0] = mismatch.caseResults[0].withTokenExactness(
            .init(exact: false, comparedTokens: 3, firstDivergenceIndex: 2, lengthMatched: true))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(mismatch))
    }

    func testBindingMustMatchExactReviewedQwenLock() {
        var wrongBlock = makePayload()
        wrongBlock.binding = .init(
            targetModelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            drafterModelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            targetRevision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33",
            sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
            blockSize: 2,
            maxAcceptedDrafts: 1)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(wrongBlock))

        var wrongRevision = makePayload()
        wrongRevision.binding = .init(
            targetModelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            drafterModelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            targetRevision: "drift",
            drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33",
            sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
            blockSize: 3,
            maxAcceptedDrafts: 2)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(wrongRevision))
    }

    func testTokenDecodedAndCacheDigestsMustBeWellFormedAndEqual() {
        var malformedTokenDigest = makePayload()
        malformedTokenDigest.caseResults[0] = malformedTokenDigest.caseResults[0].withTokenIDDigests(
            scalar: "not-hex",
            mtp: hex("token-prime-sequence"))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(malformedTokenDigest))

        var mismatchedTokenDigest = makePayload()
        mismatchedTokenDigest.caseResults[0] = mismatchedTokenDigest.caseResults[0].withTokenIDDigests(
            scalar: hex("token-a"),
            mtp: hex("token-b"))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(mismatchedTokenDigest))

        var malformedDecodedDigest = makePayload()
        malformedDecodedDigest.caseResults[1] = malformedDecodedDigest.caseResults[1].withDecodedDigests(
            scalar: "sha-swift-code",
            mtp: "sha-swift-code")
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(malformedDecodedDigest))

        var malformedCacheDigest = makePayload()
        malformedCacheDigest.caseResults[2] = malformedCacheDigest.caseResults[2].withCacheFingerprints(
            scalar: .init(digest: "cache", entries: []),
            mtp: .init(digest: "cache", entries: []),
            firstMismatch: nil)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(malformedCacheDigest))
    }

    func testStopOutcomesAndLengthBoundarySemanticsAreValidated() {
        XCTAssertNoThrow(try QwenMTPCorpusGate.validate(makePayload()))

        var cancelWrong = makePayload()
        cancelWrong.caseResults[8] = cancelWrong.caseResults[8].withStopOutcomes(scalar: .length, mtp: .cancelled)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(cancelWrong))

        var lengthWrong = makePayload()
        lengthWrong.caseResults[6] = lengthWrong.caseResults[6].withStopOutcomes(scalar: .cancelled, mtp: .cancelled)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(lengthWrong))

        var fullWrong = makePayload()
        fullWrong.caseResults[0] = fullWrong.caseResults[0].withStopOutcomes(scalar: .cancelled, mtp: .cancelled)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fullWrong))
    }

    func testTelemetryMustBeCompleteAndInternallyConsistent() {
        var inconsistent = makePayload()
        inconsistent.caseResults[0] = inconsistent.caseResults[0].withMTPTelemetry(.init(
            proposedDraftTokens: 4,
            acceptedDraftTokens: 3,
            rejectedDraftTokens: 99,
            roundCount: 2,
            targetModelCallCount: 4,
            draftModelCallCount: 2,
            targetVerifiedTokenCount: 6,
            emittedTokenCount: 8))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(inconsistent))

        var fallbackWithRounds = makePayload()
        fallbackWithRounds.caseResults[10] = fallbackWithRounds.caseResults[10].withMTPTelemetry(.init(
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            rejectedDraftTokens: 0,
            roundCount: 1,
            targetModelCallCount: 4,
            draftModelCallCount: 0,
            targetVerifiedTokenCount: 0,
            emittedTokenCount: 8))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fallbackWithRounds))

        var fallbackWithFabricatedTargetCalls = makePayload()
        fallbackWithFabricatedTargetCalls.caseResults[10] =
            fallbackWithFabricatedTargetCalls.caseResults[10].withMTPTelemetry(.init(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                rejectedDraftTokens: 0,
                roundCount: 0,
                targetModelCallCount: 8,
                draftModelCallCount: 0,
                targetVerifiedTokenCount: 0,
                emittedTokenCount: 8))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fallbackWithFabricatedTargetCalls))

        var forgedCalls = makePayload()
        forgedCalls.caseResults[0] = forgedCalls.caseResults[0]
            .withMTPTelemetry(.init(
                proposedDraftTokens: 4,
                acceptedDraftTokens: 3,
                rejectedDraftTokens: 1,
                roundCount: 2,
                targetModelCallCount: 4,
                draftModelCallCount: 2,
                targetVerifiedTokenCount: 6,
                emittedTokenCount: 8))
            .withMTPPhaseAttribution(phaseAttribution(targetVerificationCount: 4))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(forgedCalls))
    }

    func testPhaseAttributionMustBeFiniteNonNegativeAndCountCoherent() {
        var negative = makePayload()
        negative.caseResults[0] = negative.caseResults[0].withMTPPhaseAttribution(
            phaseAttribution(draftBlockSeconds: -0.001))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(negative))

        var nonFinite = makePayload()
        nonFinite.caseResults[0] = nonFinite.caseResults[0].withMTPPhaseAttribution(
            phaseAttribution(targetVerificationSeconds: .nan))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(nonFinite))

        var wrongDraftCount = makePayload()
        wrongDraftCount.caseResults[0] = wrongDraftCount.caseResults[0].withMTPPhaseAttribution(
            phaseAttribution(draftBlockCount: 1))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(wrongDraftCount))

        var missingFingerprint = makePayload()
        missingFingerprint.caseResults[0] = missingFingerprint.caseResults[0].withMTPPhaseAttribution(
            phaseAttribution(cacheFingerprintSeconds: 0, cacheFingerprintCount: 0))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(missingFingerprint))
    }

    func testPromptPreparationAttributionMustBePresentAndInternallyCoherent() {
        var missing = makePayload()
        missing.caseResults[0] = missing.caseResults[0].withMTPPhaseAttribution(
            phaseAttribution(includeTargetPromptPreparation: false))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(missing))

        var wrongPromptLength = makePayload()
        wrongPromptLength.caseResults[0] = wrongPromptLength.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    promptTokenCount: 15)))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(wrongPromptLength))

        var outOfOrderChunks = makePayload()
        outOfOrderChunks.caseResults[0] = outOfOrderChunks.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(chunks: [
                    .init(
                        tokenOffset: 0, tokenCount: 8,
                        targetForwardSchedulingSeconds: 0.0002),
                    .init(
                        tokenOffset: 9, tokenCount: 7,
                        targetForwardSchedulingSeconds: 0.0002),
                ])))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(outOfOrderChunks))

        var wrongHiddenShape = makePayload()
        wrongHiddenShape.caseResults[0] = wrongHiddenShape.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    hiddenShape: [1, 15, 4])))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(wrongHiddenShape))

        var impossibleHiddenBytes = makePayload()
        impossibleHiddenBytes.caseResults[0] = impossibleHiddenBytes.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    hiddenByteCount: 255)))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(impossibleHiddenBytes))

        var forgedEnvelope = makePayload()
        forgedEnvelope.caseResults[0] = forgedEnvelope.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    concatenatedHiddenEvaluationSeconds: 0.002,
                    targetPrefillResidualSeconds: 0.001)))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(forgedEnvelope))

        var negativeSubphase = makePayload()
        negativeSubphase.caseResults[0] = negativeSubphase.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    cacheEvaluationSeconds: -0.0001)))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(negativeSubphase))

        var fabricatedSync = makePayload()
        fabricatedSync.caseResults[0] = fabricatedSync.caseResults[0]
            .withMTPPhaseAttribution(phaseAttribution(
                targetPromptPreparation: promptPreparationAttribution(
                    phaseBoundarySynchronizationSeconds: 0.002,
                    targetPrefillResidualSeconds: 0)))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fabricatedSync))

        var fallbackFabrication = makePayload()
        fallbackFabrication.caseResults[10] = fallbackFabrication.caseResults[10]
            .withMTPPhaseAttribution(fallbackPhaseAttribution(
                targetPromptPreparation: promptPreparationAttribution()))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fallbackFabrication))
    }

    func testNormalFullCasesRequireDraftActivityAndNoPassthrough() {
        var noDrafts = makePayload()
        noDrafts.caseResults[0] = noDrafts.caseResults[0].withDrafts(proposed: 0, accepted: 0)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(noDrafts))

        var passthrough = makePayload()
        passthrough.caseResults[0] = passthrough.caseResults[0].withPassthroughReason("unexpected")
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(passthrough))
    }

    func testForcedFallbackRequiresStickyPassthroughAndZeroDraftActivity() throws {
        let valid = makePayload()
        XCTAssertNoThrow(try QwenMTPCorpusGate.validate(valid))

        var noReason = makePayload()
        noReason.caseResults[10] = noReason.caseResults[10].withPassthroughReason(nil)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(noReason))

        var drafted = makePayload()
        drafted.caseResults[10] = drafted.caseResults[10].withDrafts(proposed: 1, accepted: 0)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(drafted))

        var fabricatedSpeculationCost = makePayload()
        fabricatedSpeculationCost.caseResults[10] = fabricatedSpeculationCost.caseResults[10]
            .withMTPPhaseAttribution(fallbackPhaseAttribution(draftBlockSeconds: 0.01))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(fabricatedSpeculationCost))

        var underreportedTailCalls = makePayload()
        underreportedTailCalls.caseResults[10] = underreportedTailCalls.caseResults[10]
            .withMTPPhaseAttribution(fallbackPhaseAttribution(targetTailCount: 1))
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(underreportedTailCalls))
    }

    func testProfileEvidenceQualifiesOnlyAfterCorrectnessPasses() throws {
        var payload = makePayload()
        payload.profile = makeProfileEvidence(ratioByCase: [:])

        let verdict = try QwenMTPCorpusGate.validate(payload)

        XCTAssertEqual(verdict.correctness, .pass)
        XCTAssertEqual(verdict.profile?.qualified, true)
        XCTAssertGreaterThanOrEqual(verdict.profile?.aggregatePairedMedian ?? 0, 1.08)
        XCTAssertEqual(
            verdict.profile?.hiddenMaterializationSecondsTotal ?? 0,
            Double(QwenMTPCorpusGate.profilePlan.caseIDs.count
                * QwenMTPCorpusGate.profilePlan.measuredPairs) * 0.0004,
            accuracy: 1e-12)
        XCTAssertEqual(verdict.profile?.promptOverheadSecondsTotal ?? -1, 0, accuracy: 1e-12)
        XCTAssertEqual(
            verdict.profile?.hiddenMaterializationShareOfPromptOverhead ?? -1,
            0,
            accuracy: 1e-12)
        XCTAssertEqual(verdict.profile?.hiddenMaterializationCandidateQualified, false)

        var incorrect = payload
        incorrect.caseResults[1] = incorrect.caseResults[1].withDecodedDigests(
            scalar: "a", mtp: "b")
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(incorrect))
    }

    func testPromptHiddenReuseGateAggregatesMeasuredPrimingOnly() throws {
        var payload = makePayload()
        payload.profile = makeProfileEvidence(ratioByCase: [:])

        let verdict = try QwenMTPCorpusGate.validatePromptHiddenReuse(payload)

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(
            verdict.measuredDrafterPromptPrimingSeconds,
            Double(QwenMTPCorpusGate.profilePlan.caseIDs.count
                * QwenMTPCorpusGate.profilePlan.measuredPairs) * 0.002,
            accuracy: 1e-12)
        XCTAssertEqual(
            verdict.baselineDrafterPromptPrimingSeconds,
            QwenMTPCorpusGate.promptHiddenReusePrimingBaselineSeconds)
        XCTAssertGreaterThanOrEqual(
            verdict.reductionSeconds,
            QwenMTPCorpusGate.promptHiddenReuseRequiredReductionSeconds)
    }

    func testPromptHiddenReuseThresholdIsFailClosedAtEightySecondReduction() {
        let maximumMeasured =
            QwenMTPCorpusGate.promptHiddenReusePrimingBaselineSeconds
            - QwenMTPCorpusGate.promptHiddenReuseRequiredReductionSeconds

        XCTAssertTrue(
            QwenMTPCorpusGate.promptHiddenReuseVerdict(
                measuredPrimingSeconds: maximumMeasured).qualified)
        XCTAssertFalse(
            QwenMTPCorpusGate.promptHiddenReuseVerdict(
                measuredPrimingSeconds: maximumMeasured.nextUp).qualified)
        XCTAssertFalse(
            QwenMTPCorpusGate.promptHiddenReuseVerdict(
                measuredPrimingSeconds: .nan).qualified)
    }

    func testPromptHiddenReuseRequiresAQualifiedProfile() {
        XCTAssertThrowsError(
            try QwenMTPCorpusGate.validatePromptHiddenReuse(makePayload())) { error in
                XCTAssertEqual(
                    error as? QwenMTPCorpusGateError,
                    .promptHiddenReuseProfileRequired)
            }

        var weak = makePayload()
        weak.profile = makeProfileEvidence(ratioByCase: [:], defaultRatio: 1.07)
        XCTAssertThrowsError(
            try QwenMTPCorpusGate.validatePromptHiddenReuse(weak)) { error in
                guard case QwenMTPCorpusGateError.unqualifiedPerformance = error else {
                    XCTFail("expected the frozen profile gate to fail first, got \(error)")
                    return
                }
            }
    }

    func testProfileEvidenceRequiresReleaseOnlyContract() {
        let validProfile = makeProfileEvidence(ratioByCase: [:])

        for (required, observed) in [(false, false), (false, true), (true, false)] {
            var payload = makePayload()
            payload.profile = .init(
                releaseBuildRequired: required,
                releaseBuildObserved: observed,
                samples: validProfile.samples)

            XCTAssertThrowsError(try QwenMTPCorpusGate.validate(payload))
        }
    }

    func testGreedyBatchedVerificationGateAggregatesMeasuredVerificationOnly() throws {
        var payload = makePayload()
        payload.profile = makeProfileEvidence(ratioByCase: [:])

        let verdict = try QwenMTPCorpusGate.validateGreedyBatchedVerification(payload)

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(
            verdict.measuredTargetVerificationSeconds,
            Double(QwenMTPCorpusGate.profilePlan.caseIDs.count
                * QwenMTPCorpusGate.profilePlan.measuredPairs) * 0.006,
            accuracy: 1e-12)
        XCTAssertEqual(
            verdict.baselineTargetVerificationSeconds,
            QwenMTPCorpusGate.greedyBatchedVerificationBaselineSeconds)
        XCTAssertGreaterThanOrEqual(
            verdict.reductionSeconds,
            QwenMTPCorpusGate.greedyBatchedVerificationRequiredReductionSeconds)
    }

    func testGreedyBatchedVerificationThresholdIsFailClosedAtFiveSecondReduction() {
        let maximumMeasured =
            QwenMTPCorpusGate.greedyBatchedVerificationBaselineSeconds
            - QwenMTPCorpusGate.greedyBatchedVerificationRequiredReductionSeconds

        XCTAssertTrue(
            QwenMTPCorpusGate.greedyBatchedVerificationVerdict(
                measuredVerificationSeconds: maximumMeasured).qualified)
        XCTAssertFalse(
            QwenMTPCorpusGate.greedyBatchedVerificationVerdict(
                measuredVerificationSeconds: maximumMeasured.nextUp).qualified)
        XCTAssertFalse(
            QwenMTPCorpusGate.greedyBatchedVerificationVerdict(
                measuredVerificationSeconds: .nan).qualified)
    }

    func testGreedyBatchedVerificationRequiresAQualifiedProfile() {
        XCTAssertThrowsError(
            try QwenMTPCorpusGate.validateGreedyBatchedVerification(makePayload())) { error in
                XCTAssertEqual(
                    error as? QwenMTPCorpusGateError,
                    .greedyBatchedVerificationProfileRequired)
            }

        var weak = makePayload()
        weak.profile = makeProfileEvidence(ratioByCase: [:], defaultRatio: 1.07)
        XCTAssertThrowsError(
            try QwenMTPCorpusGate.validateGreedyBatchedVerification(weak)) { error in
                guard case QwenMTPCorpusGateError.unqualifiedPerformance = error else {
                    XCTFail("expected the frozen profile gate to fail first, got \(error)")
                    return
                }
            }

        let maximumMeasured =
            QwenMTPCorpusGate.greedyBatchedVerificationBaselineSeconds
            - QwenMTPCorpusGate.greedyBatchedVerificationRequiredReductionSeconds
        var insufficient = makePayload()
        insufficient.profile = makeProfileEvidence(
            ratioByCase: [:],
            targetVerificationSeconds: maximumMeasured
                / Double(QwenMTPCorpusGate.profilePlan.caseIDs.count
                    * QwenMTPCorpusGate.profilePlan.measuredPairs)
                + 0.001)
        XCTAssertThrowsError(
            try QwenMTPCorpusGate.validateGreedyBatchedVerification(insufficient)) { error in
                guard case QwenMTPCorpusGateError
                    .insufficientGreedyBatchedVerificationReduction = error
                else {
                    XCTFail("expected the frozen verifier reduction gate, got \(error)")
                    return
                }
            }
    }

    func testCanonicalCorrectnessFailurePayloadDropsProfileBeforeProfiling() {
        var payload = makePayload()
        payload.profile = makeProfileEvidence(ratioByCase: [:])

        let failed = QwenMTPCorpusGate.canonicalCorrectnessFailurePayload(from: payload)

        XCTAssertEqual(failed.correctness, .fail)
        XCTAssertNil(failed.profile)
        XCTAssertEqual(failed.caseResults, payload.caseResults)
        XCTAssertEqual(failed.binding, payload.binding)
    }

    func testProfileSamplesMustBeExactBeforePerformanceQualification() {
        var payload = makePayload()
        var profile = makeProfileEvidence(ratioByCase: [:])
        profile.samples[0] = profile.samples[0].withExactness(.init(
            scalarTokenCount: 128,
            mtpTokenCount: 127,
            scalarTokenIDsSHA256: hex("profile-token"),
            mtpTokenIDsSHA256: hex("profile-token"),
            scalarDecodedBytesSHA256: hex("profile-decoded"),
            mtpDecodedBytesSHA256: hex("profile-decoded"),
            scalarStopOutcome: .length,
            mtpStopOutcome: .length,
            scalarCacheFingerprint: cacheFingerprint("profile-cache"),
            mtpCacheFingerprint: cacheFingerprint("profile-cache"),
            firstCacheMismatch: nil))
        payload.profile = profile

        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(payload))
    }

    func testProfileReportedPerformanceMustMatchExactCountsAndTimings() {
        var payload = makePayload()
        var profile = makeProfileEvidence(ratioByCase: [:])
        profile.samples[0] = profile.samples[0].withReportedPerformance(
            scalarTokensPerSecond: 128,
            mtpTokensPerSecond: 256,
            decodeOnlyRatio: 2,
            e2eRatio: 2)
        payload.profile = profile

        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(payload))
    }

    func testProfileEvidenceRejectsWeakAggregateChronologicalAndPerPromptRatios() {
        var weakAggregate = makePayload()
        weakAggregate.profile = makeProfileEvidence(ratioByCase: [:], defaultRatio: 1.07)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(weakAggregate))

        var weakChronologicalHalf = makePayload()
        weakChronologicalHalf.profile = makeProfileEvidence(
            ratioByCase: [:],
            measuredRatioByPair: [0: 1.09, 1: 1.09, 2: 1.09, 3: 1.04, 4: 1.04])
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(weakChronologicalHalf))

        var tooManyWeakPrompts = makePayload()
        tooManyWeakPrompts.profile = makeProfileEvidence(
            ratioByCase: ["prime-sequence": 0.96, "swift-code": 0.96])
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(tooManyWeakPrompts))
    }

    func testProfileEvidenceValidatesExactSampleCardinalityOrderAndTimings() {
        var missingSample = makePayload()
        var profile = makeProfileEvidence(ratioByCase: [:])
        profile.samples.removeLast()
        missingSample.profile = profile
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(missingSample))

        var badOrder = makePayload()
        profile = makeProfileEvidence(ratioByCase: [:])
        profile.samples[3] = profile.samples[3].withOrder(.scalarThenMTP)
        badOrder.profile = profile
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(badOrder))

        var badTiming = makePayload()
        profile = makeProfileEvidence(ratioByCase: [:])
        profile.samples[0] = profile.samples[0].withScalarTiming(.init(
            promptSeconds: 0.01,
            generationSeconds: .infinity,
            wallSeconds: 0.03,
            e2eSeconds: 0.04))
        badTiming.profile = profile
        XCTAssertThrowsError(try QwenMTPCorpusGate.validate(badTiming))
    }

    func testJSONLEvidenceDecoderFailsClosedOnMalformedOrIncompleteLines() throws {
        let record = ResultRecord(
            subcommand: "qwen-mtp-corpus",
            provenance: makeProvenance(),
            payload: makePayload())
        let valid = try (record.jsonLine() + "\n").data(using: .utf8).unwrap()
        XCTAssertNoThrow(try QwenMTPCorpusGate.validateJSONL(valid))

        let rejectedRecord = ResultRecord(
            subcommand: "qwen-mtp-corpus-rejected",
            provenance: makeProvenance(),
            payload: makePayload())
        let rejected = try (rejectedRecord.jsonLine() + "\n").data(using: .utf8).unwrap()
        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(rejected)) { error in
            XCTAssertEqual(
                error as? QwenMTPCorpusGateError,
                .wrongSubcommand("qwen-mtp-corpus-rejected"))
        }

        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(Data("{bad json}\n".utf8)))

        let incomplete = try record.jsonLine().data(using: .utf8).unwrap()
        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(incomplete))

        let localPathRecord = ResultRecord(
            subcommand: "qwen-mtp-corpus",
            provenance: makeProvenance(modelPath: "/models/qwen"),
            payload: makePayload())
        let localPath = try (localPathRecord.jsonLine() + "\n").data(using: .utf8).unwrap()
        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(localPath)) { error in
            XCTAssertFalse(String(describing: error).contains("/models/qwen"))
        }

        var oldKeysRoot = try XCTUnwrap(
            JSONSerialization.jsonObject(with: valid) as? [String: Any])
        var oldKeysPayload = try XCTUnwrap(oldKeysRoot["payload"] as? [String: Any])
        var oldKeysCases = try XCTUnwrap(oldKeysPayload["caseResults"] as? [[String: Any]])
        var oldKeysPhase = try XCTUnwrap(
            oldKeysCases[0]["mtpPhaseAttribution"] as? [String: Any])
        var oldKeysPreparation = try XCTUnwrap(
            oldKeysPhase["targetPromptPreparation"] as? [String: Any])
        oldKeysPreparation["cacheHiddenEvaluationSeconds"] =
            oldKeysPreparation.removeValue(forKey: "cacheEvaluationSeconds")
        oldKeysPreparation["hiddenConcatenationSeconds"] =
            oldKeysPreparation.removeValue(forKey: "concatenatedHiddenEvaluationSeconds")
        oldKeysPhase["targetPromptPreparation"] = oldKeysPreparation
        oldKeysCases[0]["mtpPhaseAttribution"] = oldKeysPhase
        oldKeysPayload["caseResults"] = oldKeysCases
        oldKeysRoot["payload"] = oldKeysPayload
        var oldKeys = try JSONSerialization.data(withJSONObject: oldKeysRoot)
        oldKeys.append(0x0a)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(oldKeys))

        var legacyRoot = try XCTUnwrap(
            JSONSerialization.jsonObject(with: valid) as? [String: Any])
        var legacyPayload = try XCTUnwrap(legacyRoot["payload"] as? [String: Any])
        legacyPayload["schemaVersion"] = 3
        var legacyCases = try XCTUnwrap(legacyPayload["caseResults"] as? [[String: Any]])
        for index in legacyCases.indices {
            legacyCases[index].removeValue(forKey: "mtpPhaseAttribution")
        }
        legacyPayload["caseResults"] = legacyCases
        legacyRoot["payload"] = legacyPayload
        var legacy = try JSONSerialization.data(withJSONObject: legacyRoot)
        legacy.append(0x0a)
        XCTAssertThrowsError(try QwenMTPCorpusGate.validateJSONL(legacy)) { error in
            XCTAssertEqual(error as? QwenMTPCorpusGateError, .schemaVersionMismatch(3))
        }
    }

    private func makePayload() -> QwenMTPCorpusEvidencePayload {
        QwenMTPCorpusEvidencePayload(
            schemaVersion: QwenMTPCorpusGate.schemaVersion,
            corpusID: QwenMTPCorpusGate.corpusID,
            corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
            binding: .init(
                targetModelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
                drafterModelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
                targetRevision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
                drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33",
                sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
                blockSize: 3,
                maxAcceptedDrafts: 2),
            host: .init(chip: "Apple M5 Max", ramBytes: 128 * 1024 * 1024 * 1024, os: "macOS 16.0.0"),
            caseResults: QwenMTPCorpusGate.cases.map(makePassingResult),
            correctness: .pass,
            profile: nil)
    }

    private func makePassingResult(_ spec: QwenMTPCorpusCaseSpec) -> QwenMTPCorpusCaseResult {
        let tokenDigest = hex("token-\(spec.id)")
        let decodedDigest = hex("decoded-\(spec.id)")
        let isFallback = spec.kind == .forcedFallback
        let tokenCount = spec.kind == .cancellationRetainedToken
            ? 1
            : min(spec.maxTokens, 8)
        let stopOutcome: QwenMTPCorpusStopOutcome = {
            switch spec.kind {
            case .lengthBoundary:
                return .length
            case .cancellationRetainedToken, .cancellationAcceptedDraft:
                return .cancelled
            default:
                return .length
            }
        }()
        let telemetry = isFallback
            ? QwenMTPCorpusMTPTelemetry(
                proposedDraftTokens: 0,
                acceptedDraftTokens: 0,
                rejectedDraftTokens: 0,
                roundCount: 0,
                targetModelCallCount: 0,
                draftModelCallCount: 0,
                targetVerifiedTokenCount: 0,
                emittedTokenCount: tokenCount)
            : QwenMTPCorpusMTPTelemetry(
                proposedDraftTokens: 4,
                acceptedDraftTokens: 3,
                rejectedDraftTokens: 1,
                roundCount: 2,
                targetModelCallCount: 2,
                draftModelCallCount: 2,
                targetVerifiedTokenCount: 6,
                emittedTokenCount: tokenCount)
        return QwenMTPCorpusCaseResult(
            caseID: spec.id,
            kind: spec.kind,
            maxTokens: spec.maxTokens,
            promptTokenCount: 16,
            scalarTokenCount: tokenCount,
            mtpTokenCount: tokenCount,
            scalarTokenIDsSHA256: tokenDigest,
            mtpTokenIDsSHA256: tokenDigest,
            tokenExactness: .init(
                exact: true,
                comparedTokens: tokenCount,
                firstDivergenceIndex: nil,
                lengthMatched: true),
            scalarDecodedBytesSHA256: decodedDigest,
            mtpDecodedBytesSHA256: decodedDigest,
            scalarStopOutcome: stopOutcome,
            mtpStopOutcome: stopOutcome,
            scalarCacheFingerprint: cacheFingerprint("cache-\(spec.id)"),
            mtpCacheFingerprint: cacheFingerprint("cache-\(spec.id)"),
            firstCacheMismatch: nil,
            scalarTiming: .init(promptSeconds: 0.01, generationSeconds: 0.02, wallSeconds: 0.03, e2eSeconds: 0.04),
            mtpTiming: .init(promptSeconds: 0.01, generationSeconds: 0.02, wallSeconds: 0.03, e2eSeconds: 0.04),
            mtpTelemetry: telemetry,
            mtpPhaseAttribution: isFallback
                ? fallbackPhaseAttribution()
                : phaseAttribution(),
            passthroughReason: isFallback ? "Qwen MTP currently requires temperature == 0; generating without speculation" : nil)
    }

    private func makeProfileEvidence(
        ratioByCase: [String: Double],
        defaultRatio: Double = 1.12,
        measuredRatioByPair: [Int: Double] = [:],
        targetVerificationSeconds: Double = 0.006
    ) -> QwenMTPCorpusProfileEvidence {
        var samples: [QwenMTPCorpusProfileSample] = []
        for caseID in QwenMTPCorpusGate.profilePlan.caseIDs {
            for pairIndex in 0..<QwenMTPCorpusGate.profilePlan.totalPairsPerCase {
                let measuredPairIndex = pairIndex - QwenMTPCorpusGate.profilePlan.droppedWarmupPairs
                let isWarmup = pairIndex < QwenMTPCorpusGate.profilePlan.droppedWarmupPairs
                let ratio = isWarmup
                    ? defaultRatio
                    : measuredRatioByPair[measuredPairIndex, default: ratioByCase[caseID, default: defaultRatio]]
                let mtpGenerationSeconds = max(
                    1.0 / ratio,
                    targetVerificationSeconds + 0.007)
                let scalarGenerationSeconds = mtpGenerationSeconds * ratio
                samples.append(.init(
                    caseID: caseID,
                    pairIndex: pairIndex,
                    warmup: isWarmup,
                    order: QwenMTPCorpusGate.profilePlan.orders[pairIndex],
                    exactness: .init(
                        scalarTokenCount: 128,
                        mtpTokenCount: 128,
                        scalarTokenIDsSHA256: hex("profile-token-\(caseID)-\(pairIndex)"),
                        mtpTokenIDsSHA256: hex("profile-token-\(caseID)-\(pairIndex)"),
                        scalarDecodedBytesSHA256: hex("profile-decoded-\(caseID)-\(pairIndex)"),
                        mtpDecodedBytesSHA256: hex("profile-decoded-\(caseID)-\(pairIndex)"),
                        scalarStopOutcome: .length,
                        mtpStopOutcome: .length,
                        scalarCacheFingerprint: cacheFingerprint("profile-cache-\(caseID)-\(pairIndex)"),
                        mtpCacheFingerprint: cacheFingerprint("profile-cache-\(caseID)-\(pairIndex)"),
                        firstCacheMismatch: nil),
                    scalarTiming: .init(
                        promptSeconds: 0.01,
                        generationSeconds: scalarGenerationSeconds,
                        wallSeconds: scalarGenerationSeconds,
                        e2eSeconds: scalarGenerationSeconds),
                    mtpTiming: .init(
                        promptSeconds: 0.01,
                        generationSeconds: mtpGenerationSeconds,
                        wallSeconds: mtpGenerationSeconds,
                        e2eSeconds: mtpGenerationSeconds),
                    scalarTokensPerSecond: 128 / scalarGenerationSeconds,
                    mtpTokensPerSecond: 128 / mtpGenerationSeconds,
                    decodeOnlyRatio: ratio,
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
                    mtpPhaseAttribution: phaseAttribution(
                        targetVerificationSeconds: targetVerificationSeconds,
                        draftBlockCount: 5,
                        targetVerificationCount: 5),
                    passthroughReason: nil))
            }
        }
        return QwenMTPCorpusProfileEvidence(
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            samples: samples)
    }

    private func makeProvenance(
        modelPath: String = QwenMTPCorpusGate.requiredBinding.targetModelID
    ) -> Provenance {
        Provenance(
            date: "2026-08-23T00:00:00Z",
            hardwareChip: "Apple M5 Max",
            hardwareRAMBytes: 128 * 1024 * 1024 * 1024,
            hardwareOS: "macOS 16.0.0",
            harnessGitSHA: "abc",
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: modelPath,
            modelConfigHash: "hash",
            modelQuant: .init(bits: 4, groupSize: 64),
            corpusId: QwenMTPCorpusGate.corpusID,
            corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
            nonce: "nonce")
    }

    private func phaseAttribution(
        targetPrefillSeconds: Double = 0.002,
        drafterPromptPrimingSeconds: Double = 0.002,
        draftBlockSeconds: Double = 0.004,
        targetVerificationSeconds: Double = 0.006,
        targetTailSeconds: Double = 0,
        hybridRewindReplaySeconds: Double = 0.001,
        finalizationSeconds: Double = 0.001,
        cacheFingerprintSeconds: Double = 0.001,
        draftBlockCount: Int = 2,
        targetVerificationCount: Int = 2,
        targetTailCount: Int = 0,
        cacheFingerprintCount: Int = 1,
        includeTargetPromptPreparation: Bool = true,
        targetPromptPreparation: QwenMTPPromptPreparationAttribution? = nil
    ) -> QwenMTPCorpusMTPPhaseAttribution {
        QwenMTPCorpusMTPPhaseAttribution(
            targetPrefillSeconds: targetPrefillSeconds,
            drafterPromptPrimingSeconds: drafterPromptPrimingSeconds,
            draftBlockSeconds: draftBlockSeconds,
            targetVerificationSeconds: targetVerificationSeconds,
            targetTailSeconds: targetTailSeconds,
            hybridRewindReplaySeconds: hybridRewindReplaySeconds,
            finalizationSeconds: finalizationSeconds,
            cacheFingerprintSeconds: cacheFingerprintSeconds,
            targetPrefillCount: 1,
            drafterPromptPrimingCount: 1,
            draftBlockCount: draftBlockCount,
            targetVerificationCount: targetVerificationCount,
            targetTailCount: targetTailCount,
            hybridRewindReplayCount: 1,
            finalizationCount: 1,
            cacheFingerprintCount: cacheFingerprintCount,
            targetPromptPreparation: includeTargetPromptPreparation
                ? (targetPromptPreparation ?? promptPreparationAttribution())
                : nil)
    }

    private func fallbackPhaseAttribution(
        draftBlockSeconds: Double = 0,
        targetTailCount: Int = 7,
        targetPromptPreparation: QwenMTPPromptPreparationAttribution? = nil
    ) -> QwenMTPCorpusMTPPhaseAttribution {
        QwenMTPCorpusMTPPhaseAttribution(
            targetPrefillSeconds: 0.002,
            drafterPromptPrimingSeconds: 0,
            draftBlockSeconds: draftBlockSeconds,
            targetVerificationSeconds: 0,
            targetTailSeconds: 0.01,
            hybridRewindReplaySeconds: 0,
            finalizationSeconds: 0.001,
            cacheFingerprintSeconds: 0.001,
            targetPrefillCount: 1,
            drafterPromptPrimingCount: 0,
            draftBlockCount: 0,
            targetVerificationCount: 0,
            targetTailCount: targetTailCount,
            hybridRewindReplayCount: 0,
            finalizationCount: 1,
            cacheFingerprintCount: 1,
            targetPromptPreparation: targetPromptPreparation)
    }

    private func promptPreparationAttribution(
        promptTokenCount: Int = 16,
        hiddenShape: [Int] = [1, 16, 4],
        hiddenByteCount: Int = 256,
        chunks: [QwenMTPPromptPreparationChunkAttribution] = [
            .init(
                tokenOffset: 0, tokenCount: 8,
                targetForwardSchedulingSeconds: 0.0002),
            .init(
                tokenOffset: 8, tokenCount: 8,
                targetForwardSchedulingSeconds: 0.0002),
        ],
        cacheEvaluationSeconds: Double = 0.0003,
        hiddenEvaluationSeconds: Double = 0.0002,
        concatenatedHiddenEvaluationSeconds: Double = 0.0002,
        preparedCacheHandoffSeconds: Double = 0.0004,
        phaseBoundarySynchronizationSeconds: Double = 0.0001,
        targetPrefillResidualSeconds: Double = 0.0004
    ) -> QwenMTPPromptPreparationAttribution {
        QwenMTPPromptPreparationAttribution(
            promptTokenCount: promptTokenCount,
            hiddenShape: hiddenShape,
            hiddenByteCount: hiddenByteCount,
            chunks: chunks,
            cacheEvaluationSeconds: cacheEvaluationSeconds,
            hiddenEvaluationSeconds: hiddenEvaluationSeconds,
            concatenatedHiddenEvaluationSeconds: concatenatedHiddenEvaluationSeconds,
            preparedCacheHandoffSeconds: preparedCacheHandoffSeconds,
            phaseBoundarySynchronizationSeconds: phaseBoundarySynchronizationSeconds,
            targetPrefillResidualSeconds: targetPrefillResidualSeconds)
    }
}

private extension Optional {
    func unwrap(file: StaticString = #filePath, line: UInt = #line) throws -> Wrapped {
        guard let value = self else {
            XCTFail("unexpected nil", file: file, line: line)
            throw UnwrapFailure()
        }
        return value
    }
}

private struct UnwrapFailure: Error {}

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
