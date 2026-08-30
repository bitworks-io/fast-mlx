import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38PerformanceAttributionScorecardGateTests: XCTestCase {
    private typealias Gate = Qwen38PerformanceAttributionScorecardGate
    private typealias Error = Qwen38PerformanceAttributionScorecardGateError

    func testValidAttributionScorecardQualifiesRequiredClaimsAndIgnoresExploratoryBestStack() throws {
        let scorecard = try makeScorecard(includeExploratory: true)

        let verdict = try validate(scorecard)

        XCTAssertTrue(verdict.qualified)
        XCTAssertEqual(Set(scorecard.claims.map(\.kind)), Set(Gate.requiredClaimKinds))
        XCTAssertEqual(scorecard.exploratoryBestStack?.kind, .bestStackExploratory)
        XCTAssertEqual(scorecard.exploratoryBestStack?.verdict.qualified, false)
        XCTAssertEqual(scorecard.exploratoryBestStack?.verdict.exploratory, true)
        XCTAssertEqual(
            scorecard.envelopeDigest,
            Gate.envelopeDigest(
                artifact: artifact,
                runIdentity: runIdentity,
                promotionPolicyDigest: scorecard.promotionPolicyDigest,
                productionRouteReceiptDigest: scorecard.productionRouteReceiptDigest,
                claims: scorecard.claims,
                exploratoryBestStack: scorecard.exploratoryBestStack))
    }

    func testRejectsMissingAndDuplicateRequiredClaim() throws {
        var missing = try makeScorecard()
        missing.claims.removeAll { $0.kind == .prefixMatrix }
        missing.envelopeDigest = Gate.envelopeDigest(
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: missing.promotionPolicyDigest,
            productionRouteReceiptDigest: missing.productionRouteReceiptDigest,
            claims: missing.claims,
            exploratoryBestStack: missing.exploratoryBestStack)
        missing.verdict = .unqualified
        XCTAssertThrowsError(try validate(missing)) { error in
            XCTAssertEqual(error as? Error, .missingClaim(.prefixMatrix))
        }

        var duplicate = try makeScorecard()
        duplicate.claims.append(duplicate.claims[0])
        duplicate.envelopeDigest = Gate.envelopeDigest(
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: duplicate.promotionPolicyDigest,
            productionRouteReceiptDigest: duplicate.productionRouteReceiptDigest,
            claims: duplicate.claims,
            exploratoryBestStack: duplicate.exploratoryBestStack)
        XCTAssertThrowsError(try validate(duplicate)) { error in
            XCTAssertEqual(error as? Error, .duplicateClaim(.scalarGDN))
        }
    }

    func testEnvelopeBindsClaimBodiesAndTopLevelArtifact() throws {
        var tamperedClaim = try makeScorecard()
        tamperedClaim.claims[0].measurements[0].candidate.requests[0].prefillSeconds = 0.4
        tamperedClaim.claims[0] = try recompute(tamperedClaim.claims[0])
        XCTAssertThrowsError(try validate(tamperedClaim)) { error in
            XCTAssertEqual(error as? Error, .invalidEnvelope)
        }

        var mismatchedArtifact = try makeScorecard()
        mismatchedArtifact.artifact.targetRevision = String(repeating: "9", count: 40)
        mismatchedArtifact.envelopeDigest = Gate.envelopeDigest(
            artifact: mismatchedArtifact.artifact,
            runIdentity: mismatchedArtifact.runIdentity,
            promotionPolicyDigest: mismatchedArtifact.promotionPolicyDigest,
            productionRouteReceiptDigest: mismatchedArtifact.productionRouteReceiptDigest,
            claims: mismatchedArtifact.claims,
            exploratoryBestStack: mismatchedArtifact.exploratoryBestStack)
        XCTAssertThrowsError(try validate(mismatchedArtifact)) { error in
            XCTAssertEqual(error as? Error, .invalidPromotionPolicy)
        }
    }

    func testRejectsMislabeledBlendedExactMTPAndWrongScalarGDNSingleAxisIdentity() throws {
        var blended = try makeClaim(.exactMTP)
        blended.reference.gdnMode = .gdnOff
        blended.reference.launchBinding = launchBinding(mode: .gdnOff, isolation: hex("9"))
        XCTAssertThrowsError(try Gate.evaluateClaim(blended)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.exactMTP))
        }

        var wrongScalarAxis = try makeClaim(.scalarGDN)
        wrongScalarAxis.candidate.executionMode = .exactMTP
        wrongScalarAxis.candidate.executionDigest = digest("exact")
        XCTAssertThrowsError(try Gate.evaluateClaim(wrongScalarAxis)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.scalarGDN))
        }
    }

    func testScalarGDNAxisRequiresOrderedModesAndValidIsolatedLaunchBindings() throws {
        var missingMode = try makeClaim(.scalarGDN)
        var missingModeCandidate = missingMode.candidate
        missingModeCandidate.gdnMode = nil
        missingModeCandidate.launchBinding = nil
        rebindCandidate(&missingMode, to: missingModeCandidate)
        XCTAssertThrowsError(try Gate.evaluateClaim(missingMode)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.scalarGDN))
        }

        var inverted = try makeClaim(.scalarGDN)
        let invertedCandidate = model(.scalar, .gdnOff, "scalar-inverted-candidate")
        var invertedReference = model(.scalar, .gdnOn, "scalar-inverted-reference")
        invertedReference.executionDigest = invertedCandidate.executionDigest
        rebindCandidate(&inverted, to: invertedCandidate)
        rebindReference(&inverted, to: invertedReference)
        XCTAssertThrowsError(try Gate.evaluateClaim(inverted)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.scalarGDN))
        }

        var invalidLaunch = try makeClaim(.scalarGDN)
        var invalidLaunchCandidate = invalidLaunch.candidate
        invalidLaunchCandidate.launchBinding?.observedEnv = .disabled
        rebindCandidate(&invalidLaunch, to: invalidLaunchCandidate)
        XCTAssertThrowsError(try Gate.evaluateClaim(invalidLaunch)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.scalarGDN))
        }
    }

    func testRejectsSyntheticOrUnboundContinuousBatchProof() throws {
        var unbound = try makeClaim(.continuousBatchNoSpec)
        unbound.candidateRoute.backendObservationDigest = nil
        XCTAssertThrowsError(try Gate.evaluateClaim(unbound)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.continuousBatchNoSpec))
        }

        var syntheticEvidence = try makeClaim(.continuousBatchNoSpec)
        let syntheticRoute = Gate.routeIdentity(
            kind: .continuousBatchNoSpec,
            backendEvidenceKind: .syntheticPathProof,
            backendObservationDigest: syntheticEvidence.candidateRoute.backendObservationDigest)
        syntheticEvidence.candidateRoute = syntheticRoute
        for pairIndex in syntheticEvidence.measurements.indices {
            syntheticEvidence.measurements[pairIndex].candidate.route = syntheticRoute
        }
        XCTAssertThrowsError(try Gate.evaluateClaim(syntheticEvidence)) { error in
            XCTAssertEqual(error as? Error, .invalidClaimIdentity(.continuousBatchNoSpec))
        }

        var syntheticFanout = try makeClaim(.continuousBatchNoSpec)
        syntheticFanout.measurements[0].candidate.requests[0].routeObservation.overlapObserved = false
        bindContinuousRouteEvidence(&syntheticFanout)
        XCTAssertThrowsError(try Gate.evaluateClaim(syntheticFanout)) { error in
            XCTAssertEqual(error as? Error, .invalidRouteEvidence(.continuousBatchNoSpec))
        }

        var speculative = try makeClaim(.continuousBatchNoSpec)
        speculative.measurements[1].candidate.requests[0].routeObservation.speculationUsed = true
        bindContinuousRouteEvidence(&speculative)
        XCTAssertThrowsError(try Gate.evaluateClaim(speculative)) { error in
            XCTAssertEqual(error as? Error, .invalidRouteEvidence(.continuousBatchNoSpec))
        }


        var forgedDigest = try makeClaim(.continuousBatchNoSpec)
        let forged = hex("f")
        forgedDigest.candidateRoute.backendObservationDigest = forged
        for pairIndex in forgedDigest.measurements.indices {
            forgedDigest.measurements[pairIndex].candidate.route = forgedDigest.candidateRoute
            for requestIndex in forgedDigest.measurements[pairIndex].candidate.requests.indices {
                forgedDigest.measurements[pairIndex].candidate.requests[requestIndex]
                    .routeObservation.backendObservationDigest = forged
            }
        }
        XCTAssertThrowsError(try Gate.evaluateClaim(forgedDigest))
    }

    func testRejectsFabricatedSchedulerEvidenceOnNonContinuousRoutes() throws {
        var scalar = try makeClaim(.scalarGDN)
        scalar.measurements[0].candidate.requests[0].routeObservation.planRevisionAfter = 1
        scalar.measurements[0].candidate.requests[0].routeObservation.stateRevisionAfter = 1
        scalar.measurements[0].candidate.requests[0].routeObservation.sharedOccupancy = 1
        XCTAssertThrowsError(try Gate.evaluateClaim(scalar)) { error in
            XCTAssertEqual(error as? Error, .invalidRouteEvidence(.scalarGDN))
        }

        var continuous = try makeClaim(.continuousBatchNoSpec)
        continuous.measurements[0].reference.requests[0].routeObservation.planRevisionAfter = 1
        continuous.measurements[0].reference.requests[0].routeObservation.stateRevisionAfter = 1
        XCTAssertThrowsError(try Gate.evaluateClaim(continuous)) { error in
            XCTAssertEqual(error as? Error, .invalidRouteEvidence(.continuousBatchNoSpec))
        }
    }

    func testRejectsIncompleteRepeatedSampleCoverage() throws {
        var scalar = try makeClaim(.scalarGDN)
        scalar.measurements.removeLast()
        scalar.absoluteAuthority = absoluteAuthority(for: scalar)
        XCTAssertThrowsError(try Gate.evaluateClaim(scalar)) { error in
            XCTAssertEqual(error as? Error, .invalidCell(.scalarGDN, "4096-cold"))
        }
    }

    func testRejectsMissingPrefixMatrixCellAndFailingPerCellBand() throws {
        var missingCell = try makeClaim(.prefixMatrix)
        let removedCellID = missingCell.scheduledCells.removeLast().id
        missingCell.measurements.removeAll { $0.cellID == removedCellID }
        missingCell.absoluteAuthority = absoluteAuthority(for: missingCell)
        XCTAssertThrowsError(try Gate.evaluateClaim(missingCell)) { error in
            XCTAssertEqual(error as? Error, .invalidCell(.prefixMatrix, "prefix-matrix"))
        }

        var slowCell = try makeClaim(.prefixMatrix)
        slowCell.measurements[0].candidate.requests[0].prefillSeconds = 9.0
        XCTAssertThrowsError(try Gate.evaluateClaim(slowCell)) { error in
            XCTAssertEqual(error as? Error, .invalidAbsoluteAuthority(.prefixMatrix))
        }
    }

    func testRejectsForgedWarmSnapshotDigestAndColdWarmProof() throws {
        var forgedWarm = try makeClaim(.prefixMatrix)
        let warmIndex = try XCTUnwrap(forgedWarm.measurements.firstIndex {
            $0.cellID.contains("warm")
        })
        forgedWarm.measurements[warmIndex].candidate.requests[0].warmPrefixEvidence?
            .rebuildCanonicalDigest = hex("8")
        forgedWarm.measurements[warmIndex].reference.requests[0].warmPrefixEvidence?
            .rebuildCanonicalDigest = hex("8")
        XCTAssertThrowsError(try Gate.evaluateClaim(forgedWarm)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidWarmPrefixEvidence(.prefixMatrix, forgedWarm.measurements[warmIndex].cellID))
        }

        var coldWithWarm = try makeClaim(.prefixMatrix)
        let coldIndex = try XCTUnwrap(coldWithWarm.measurements.firstIndex {
            $0.cellID.contains("cold")
        })
        let warm = warmEvidence(tokenCount: coldWithWarm.scheduledCells[0].renderedPromptTokenCount)
        coldWithWarm.measurements[coldIndex].candidate.requests[0].warmPrefixEvidence = warm
        coldWithWarm.measurements[coldIndex].reference.requests[0].warmPrefixEvidence = warm
        XCTAssertThrowsError(try Gate.evaluateClaim(coldWithWarm)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidWarmPrefixEvidence(.prefixMatrix, coldWithWarm.measurements[coldIndex].cellID))
        }
    }

    func testRejectsPromptDigestMismatchAndWarmTokensThatDoNotBindToPromptDigest() throws {
        var promptMismatch = try makeClaim(.prefixMatrix)
        promptMismatch.measurements[0].candidate.requests[0].promptTokenDigest = hex("7")
        XCTAssertThrowsError(try Gate.evaluateClaim(promptMismatch)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidCell(.prefixMatrix, promptMismatch.measurements[0].cellID))
        }

        var forgedWarmTokens = try makeClaim(.prefixMatrix)
        let warmIndex = try XCTUnwrap(forgedWarmTokens.measurements.firstIndex {
            $0.cellID.contains("warm")
        })
        forgedWarmTokens.measurements[warmIndex].candidate.requests[0].warmPrefixEvidence?
            .tokenIDs[0] = -1
        forgedWarmTokens.measurements[warmIndex].reference.requests[0].warmPrefixEvidence?
            .tokenIDs[0] = -1
        XCTAssertThrowsError(try Gate.evaluateClaim(forgedWarmTokens)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidWarmPrefixEvidence(.prefixMatrix, forgedWarmTokens.measurements[warmIndex].cellID))
        }


        var forgedColdDigest = try makeClaim(.scalarGDN)
        forgedColdDigest.scheduledCells[0].promptTokenDigest = hex("6")
        for pairIndex in forgedColdDigest.measurements.indices {
            forgedColdDigest.measurements[pairIndex].candidate.requests[0].promptTokenDigest = hex("6")
            forgedColdDigest.measurements[pairIndex].reference.requests[0].promptTokenDigest = hex("6")
        }
        XCTAssertThrowsError(try Gate.evaluateClaim(forgedColdDigest)) { error in
            XCTAssertEqual(error as? Error, .invalidCell(.scalarGDN, "4096-cold"))
        }
    }

    func testRejectsUnbackedDecodeCountsAndNonPositiveTimings() throws {
        var unbackedCount = try makeClaim(.scalarGDN)
        unbackedCount.measurements[0].candidate.requests[0].decodeTokenCount += 1
        unbackedCount.measurements[0].reference.requests[0].decodeTokenCount += 1
        XCTAssertThrowsError(try Gate.evaluateClaim(unbackedCount)) { error in
            XCTAssertEqual(error as? Error, .invalidCell(.scalarGDN, "4096-cold"))
        }

        let mutations: [(String, (inout Qwen38PerformanceAttributionRequestMeasurement) -> Void)] = [
            ("negative-prefill", { $0.prefillSeconds = -1 }),
            ("zero-ttft", { $0.ttftSeconds = 0 }),
            ("zero-decode", { $0.decodeSeconds = 0 }),
            ("zero-e2e", { $0.e2eSeconds = 0 }),
        ]
        for (label, mutate) in mutations {
            var claim = try makeClaim(.scalarGDN)
            mutate(&claim.measurements[0].candidate.requests[0])
            XCTAssertThrowsError(try Gate.evaluateClaim(claim), label) { error in
                XCTAssertEqual(error as? Error, .invalidCell(.scalarGDN, "4096-cold"), label)
            }
        }
    }

    func testRejectsFiniteInputsThatOverflowDerivedMetrics() throws {
        var decodeOverflow = try makeClaim(.scalarGDN)
        decodeOverflow.measurements[0].candidate.requests[0].decodeSeconds =
            Double.leastNonzeroMagnitude
        XCTAssertThrowsError(try Gate.evaluateClaim(decodeOverflow)) { error in
            XCTAssertEqual(error as? Error, .nonFiniteMetrics(.scalarGDN))
        }

        var aggregateOverflow = try makeClaim(.continuousBatchNoSpec)
        for index in aggregateOverflow.measurements.indices
        where !aggregateOverflow.measurements[index].warmup {
            aggregateOverflow.measurements[index].candidate.wallSeconds =
                Double.leastNonzeroMagnitude
        }
        XCTAssertThrowsError(try Gate.evaluateClaim(aggregateOverflow)) { error in
            XCTAssertEqual(error as? Error, .nonFiniteMetrics(.continuousBatchNoSpec))
        }
    }

    func testContinuousBatchUsesPerRequestP95LatencyGate() throws {
        var claim = try makeClaim(.continuousBatchNoSpec)
        let measuredIndex = Gate.droppedWarmupSamplesPerCell
        claim.measurements[measuredIndex].candidate.requests[0].e2eSeconds = 100

        let verdict = try Gate.evaluateClaim(claim)

        XCTAssertFalse(verdict.qualified)
    }

    func testRejectsPerEngineCleanupFailureAndMonotonicCounterRegression() throws {
        var rssLeak = try makeClaim(.scalarGDN)
        rssLeak.measurements[0].candidate.cleanup.finalRSSBytes =
            rssLeak.measurements[0].candidate.cleanup.baselineRSSBytes
                + rssLeak.cleanupAuthority.maxRSSDeltaBytes
                + 1
        XCTAssertThrowsError(try Gate.evaluateClaim(rssLeak)) { error in
            XCTAssertEqual(error as? Error, .invalidCleanup(.scalarGDN))
        }

        var counterRegression = try makeClaim(.scalarGDN)
        counterRegression.measurements[0].reference.cleanup.finalPageouts =
            counterRegression.measurements[0].reference.cleanup.baselinePageouts - 1
        XCTAssertThrowsError(try Gate.evaluateClaim(counterRegression)) { error in
            XCTAssertEqual(error as? Error, .invalidCleanup(.scalarGDN))
        }

        var swapDecrease = try makeClaim(.scalarGDN)
        swapDecrease.measurements[0].candidate.cleanup.baselineSwapBytes = 10
        swapDecrease.measurements[0].candidate.cleanup.finalSwapBytes = 0
        swapDecrease.measurements[0].candidate.cleanup.evidenceDigest =
            Gate.cleanupEvidenceDigest(
                swapDecrease.measurements[0].candidate.cleanup)
        XCTAssertNoThrow(try Gate.evaluateClaim(swapDecrease))

        var swapGrowth = try makeClaim(.scalarGDN)
        swapGrowth.measurements[0].candidate.cleanup.finalSwapBytes = 1
        swapGrowth.measurements[0].candidate.cleanup.evidenceDigest =
            Gate.cleanupEvidenceDigest(
                swapGrowth.measurements[0].candidate.cleanup)
        XCTAssertThrowsError(try Gate.evaluateClaim(swapGrowth)) { error in
            XCTAssertEqual(error as? Error, .invalidCleanup(.scalarGDN))
        }

        var unsealedCleanupValues = try makeClaim(.scalarGDN)
        unsealedCleanupValues.measurements[0].candidate.cleanup.finalRSSBytes += 1
        XCTAssertThrowsError(try Gate.evaluateClaim(unsealedCleanupValues)) { error in
            XCTAssertEqual(error as? Error, .invalidCleanup(.scalarGDN))
        }
    }

    func testRejectsDuplicateAbsoluteBandKeyAndUnsealedAuthority() throws {
        var duplicateBand = try makeClaim(.exactMTP)
        duplicateBand.absoluteAuthority.bands.append(duplicateBand.absoluteAuthority.bands[0])
        duplicateBand.absoluteAuthority = Gate.absoluteAuthority(
            evidenceID: duplicateBand.absoluteAuthority.evidenceID,
            bands: duplicateBand.absoluteAuthority.bands)
        XCTAssertThrowsError(try Gate.evaluateClaim(duplicateBand)) { error in
            XCTAssertEqual(error as? Error, .invalidAbsoluteAuthority(.exactMTP))
        }

        var unsealed = try makeClaim(.exactMTP)
        unsealed.absoluteAuthority = Gate.absoluteAuthority(
            evidenceID: unsealed.absoluteAuthority.evidenceID,
            sealedBeforeMeasurements: false,
            bands: unsealed.absoluteAuthority.bands)
        XCTAssertThrowsError(try Gate.evaluateClaim(unsealed)) { error in
            XCTAssertEqual(error as? Error, .invalidAbsoluteAuthority(.exactMTP))
        }

        var nonFinite = try makeClaim(.exactMTP)
        nonFinite.absoluteAuthority.bands[0].maxPrefillSeconds = .nan
        nonFinite.absoluteAuthority = Gate.absoluteAuthority(
            evidenceID: nonFinite.absoluteAuthority.evidenceID,
            bands: nonFinite.absoluteAuthority.bands)
        XCTAssertThrowsError(try Gate.evaluateClaim(nonFinite)) { error in
            XCTAssertEqual(error as? Error, .invalidAbsoluteAuthority(.exactMTP))
        }
    }

    func testExploratoryBestStackCanBePresentButNeverPromotes() throws {
        let exploratory = try makeClaim(.bestStackExploratory)

        XCTAssertEqual(exploratory.verdict, .init(qualified: false, exploratory: true))

        var promoted = exploratory
        promoted.verdict = .init(qualified: true, exploratory: true)
        var scorecard = try makeScorecard(includeExploratory: true)
        scorecard.exploratoryBestStack = promoted
        scorecard.envelopeDigest = Gate.envelopeDigest(
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: scorecard.promotionPolicyDigest,
            productionRouteReceiptDigest: scorecard.productionRouteReceiptDigest,
            claims: scorecard.claims,
            exploratoryBestStack: promoted)
        XCTAssertThrowsError(try validate(scorecard)) { error in
            XCTAssertEqual(error as? Error, .verdictMismatch(.bestStackExploratory))
        }
    }

    func testFlagshipPromotionRejectsSelfIssuedPolicyRouteReceiptAndFixtureRun() throws {
        let original = try makeScorecard()
        let pinnedContext = try promotionContext(for: original)

        var widened = original
        let scalarIndex = try XCTUnwrap(widened.claims.firstIndex {
            $0.kind == .scalarGDN
        })
        widened.claims[scalarIndex].absoluteAuthority.bands[0].maxPrefillSeconds = 10_000
        widened.claims[scalarIndex].absoluteAuthority = Gate.absoluteAuthority(
            evidenceID: widened.claims[scalarIndex].absoluteAuthority.evidenceID,
            bands: widened.claims[scalarIndex].absoluteAuthority.bands)
        widened.claims[scalarIndex] = try recompute(widened.claims[scalarIndex])
        let widenedPolicy = frozenPolicy(
            claims: widened.claims,
            runIdentity: widened.runIdentity,
            artifact: widened.artifact)
        widened.promotionPolicyDigest = widenedPolicy.digest
        reEnvelope(&widened)
        let selfIssuedPolicyContext = Qwen38FlagshipPromotionContext(
            frozenPolicy: widenedPolicy,
            expectedPolicyDigest: pinnedContext.expectedPolicyDigest,
            productionRouteReceipt: pinnedContext.productionRouteReceipt,
            expectedReceiptDigest: pinnedContext.expectedReceiptDigest)
        XCTAssertThrowsError(try Gate.validateForFlagshipPromotion(
            widened,
            context: selfIssuedPolicyContext)
        ) { error in
            XCTAssertEqual(error as? Error, .invalidPromotionPolicy)
        }

        var relabeledRoute = original
        let continuousIndex = try XCTUnwrap(relabeledRoute.claims.firstIndex {
            $0.kind == .continuousBatchNoSpec
        })
        let originalRoute = relabeledRoute.claims[continuousIndex].candidateRoute
        let forgedReceiptDigest = hex("f")
        let forgedRoute = Gate.routeIdentity(
            kind: .continuousBatchNoSpec,
            backendEvidenceKind: .liveProductionRoute,
            backendEvidenceID: originalRoute.backendEvidenceID,
            backendObservationDigest: originalRoute.backendObservationDigest,
            backendReceiptDigest: forgedReceiptDigest)
        relabeledRoute.claims[continuousIndex].candidateRoute = forgedRoute
        for pairIndex in relabeledRoute.claims[continuousIndex].measurements.indices {
            relabeledRoute.claims[continuousIndex].measurements[pairIndex].candidate.route =
                forgedRoute
        }
        relabeledRoute.productionRouteReceiptDigest = forgedReceiptDigest
        reEnvelope(&relabeledRoute)
        XCTAssertThrowsError(try Gate.validateForFlagshipPromotion(
            relabeledRoute,
            context: pinnedContext)
        ) { error in
            XCTAssertEqual(error as? Error, .invalidProductionRouteReceipt)
        }

        var fixtureRun = original
        let liveRun = fixtureRun.runIdentity
        fixtureRun.runIdentity = Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: "synthetic-fixture",
            hardwareChip: liveRun.hardwareChip,
            hardwareRAMBytes: 1,
            hardwareOSBuild: liveRun.hardwareOSBuild,
            hostIdentityDigest: liveRun.hostIdentityDigest,
            harnessGitSHA: liveRun.harnessGitSHA,
            candidateMLXSwiftVersion: liveRun.candidateMLXSwiftVersion,
            referenceMLXVersion: liveRun.referenceMLXVersion,
            referenceMLXLMVersion: liveRun.referenceMLXLMVersion,
            modelLabel: liveRun.modelLabel,
            modelConfigHash: liveRun.modelConfigHash,
            modelCheckpointManifestHash: liveRun.modelCheckpointManifestHash,
            modelQuant: liveRun.modelQuant,
            corpusID: liveRun.corpusID,
            corpusContentHash: liveRun.corpusContentHash)
        let fixturePolicy = frozenPolicy(
            claims: fixtureRun.claims,
            runIdentity: fixtureRun.runIdentity,
            artifact: fixtureRun.artifact)
        let continuous = try XCTUnwrap(fixtureRun.claims.first {
            $0.kind == .continuousBatchNoSpec
        })
        let fixtureReceipt = routeReceipt(
            observationDigest: try XCTUnwrap(
                continuous.candidateRoute.backendObservationDigest),
            runIdentity: fixtureRun.runIdentity,
            artifact: fixtureRun.artifact)
        fixtureRun.promotionPolicyDigest = fixturePolicy.digest
        fixtureRun.productionRouteReceiptDigest = fixtureReceipt.digest
        reEnvelope(&fixtureRun)
        let fixtureContext = Qwen38FlagshipPromotionContext(
            frozenPolicy: fixturePolicy,
            expectedPolicyDigest: fixturePolicy.digest,
            productionRouteReceipt: fixtureReceipt,
            expectedReceiptDigest: fixtureReceipt.digest)
        XCTAssertThrowsError(try Gate.validateForFlagshipPromotion(
            fixtureRun,
            context: fixtureContext)
        ) { error in
            XCTAssertEqual(error as? Error, .invalidPromotionPolicy)
        }
    }

    private func makeScorecard(
        includeExploratory: Bool = false
    ) throws -> Qwen38PerformanceAttributionScorecard {
        let claims = try Gate.requiredClaimKinds.map { try makeClaim($0) }
        let exploratory = includeExploratory ? try makeClaim(.bestStackExploratory) : nil
        let policy = frozenPolicy(claims: claims)
        let continuous = try XCTUnwrap(claims.first {
            $0.kind == .continuousBatchNoSpec
        })
        let receipt = routeReceipt(
            observationDigest: try XCTUnwrap(
                continuous.candidateRoute.backendObservationDigest))
        let envelopeDigest = Gate.envelopeDigest(
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: policy.digest,
            productionRouteReceiptDigest: receipt.digest,
            claims: claims,
            exploratoryBestStack: exploratory)
        return Qwen38PerformanceAttributionScorecard(
            schemaVersion: Gate.schemaVersion,
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: policy.digest,
            productionRouteReceiptDigest: receipt.digest,
            envelopeDigest: envelopeDigest,
            claims: claims,
            exploratoryBestStack: exploratory,
            verdict: .init(qualified: true))
    }

    private func validate(
        _ scorecard: Qwen38PerformanceAttributionScorecard
    ) throws -> Qwen38PerformanceAttributionScorecardVerdict {
        try Gate.validateForFlagshipPromotion(
            scorecard,
            context: promotionContext(for: scorecard))
    }

    private func promotionContext(
        for scorecard: Qwen38PerformanceAttributionScorecard
    ) throws -> Qwen38FlagshipPromotionContext {
        let policy = frozenPolicy(
            claims: scorecard.claims,
            runIdentity: scorecard.runIdentity,
            artifact: scorecard.artifact)
        let continuous = try XCTUnwrap(scorecard.claims.first {
            $0.kind == .continuousBatchNoSpec
        })
        let receipt = routeReceipt(
            observationDigest: try XCTUnwrap(
                continuous.candidateRoute.backendObservationDigest),
            runIdentity: scorecard.runIdentity,
            artifact: scorecard.artifact)
        return Qwen38FlagshipPromotionContext(
            frozenPolicy: policy,
            expectedPolicyDigest: scorecard.promotionPolicyDigest,
            productionRouteReceipt: receipt,
            expectedReceiptDigest: scorecard.productionRouteReceiptDigest)
    }

    private func frozenPolicy(
        claims: [Qwen38PerformanceAttributionClaim],
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity? = nil,
        artifact: Qwen38MTPPerformanceScorecardArtifact? = nil
    ) -> Qwen38PerformanceAttributionFrozenPromotionPolicy {
        Gate.frozenPromotionPolicy(
            evidenceID: promotionPolicyEvidenceID,
            artifact: artifact ?? self.artifact,
            runIdentity: runIdentity ?? self.runIdentity,
            claimAuthorities: Gate.requiredClaimKinds.compactMap { kind in
                claims.first(where: { $0.kind == kind }).map {
                    Qwen38PerformanceAttributionFrozenClaimAuthority(
                        claimKind: kind,
                        absoluteAuthority: $0.absoluteAuthority,
                        cleanupAuthority: $0.cleanupAuthority)
                }
            })
    }

    private func routeReceipt(
        observationDigest: String,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity? = nil,
        artifact: Qwen38MTPPerformanceScorecardArtifact? = nil
    ) -> Qwen38PerformanceAttributionProductionRouteReceipt {
        Gate.productionRouteReceipt(
            evidenceID: productionRouteEvidenceID,
            artifact: artifact ?? self.artifact,
            runIdentity: runIdentity ?? self.runIdentity,
            backendBuildIdentityDigest: backendBuildIdentityDigest,
            observationDigest: observationDigest)
    }

    private func reEnvelope(
        _ scorecard: inout Qwen38PerformanceAttributionScorecard
    ) {
        scorecard.envelopeDigest = Gate.envelopeDigest(
            artifact: scorecard.artifact,
            runIdentity: scorecard.runIdentity,
            promotionPolicyDigest: scorecard.promotionPolicyDigest,
            productionRouteReceiptDigest: scorecard.productionRouteReceiptDigest,
            claims: scorecard.claims,
            exploratoryBestStack: scorecard.exploratoryBestStack)
    }

    private func makeClaim(
        _ kind: Qwen38PerformanceAttributionClaimKind
    ) throws -> Qwen38PerformanceAttributionClaim {
        let identities = claimIdentities(kind)
        let routes = routeIdentities(kind)
        let cells = cells(for: kind)
        let measurements = cells.flatMap { cell -> [Qwen38PerformanceAttributionPairMeasurement] in
            concurrencies(for: kind).flatMap { concurrency in
                (0..<sampleCount).map { sampleIndex in
                    pair(
                        kind: kind,
                        cell: cell,
                        concurrency: concurrency,
                        sampleIndex: sampleIndex,
                        identities: identities,
                        routes: routes)
                }
            }
        }
        var claim = Qwen38PerformanceAttributionClaim(
            kind: kind,
            candidate: identities.candidate,
            reference: identities.reference,
            candidateRoute: routes.candidate,
            referenceRoute: routes.reference,
            scheduledCells: cells,
            measurements: measurements,
            absoluteAuthority: Gate.absoluteAuthority(evidenceID: hex("1"), bands: []),
            cleanupAuthority: cleanupAuthority,
            metrics: .empty,
            verdict: .unqualified)
        if kind == .continuousBatchNoSpec {
            bindContinuousRouteEvidence(&claim)
        }
        claim.absoluteAuthority = absoluteAuthority(for: claim)
        return try recompute(claim)
    }

    private func recompute(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Qwen38PerformanceAttributionClaim {
        var updated = claim
        updated.metrics = try Gate.computeMetrics(updated)
        updated.verdict = try Gate.evaluateClaim(updated)
        return updated
    }

    private func rebindCandidate(
        _ claim: inout Qwen38PerformanceAttributionClaim,
        to identity: Qwen38MTPPerformanceScorecardModel
    ) {
        claim.candidate = identity
        for index in claim.measurements.indices {
            claim.measurements[index].candidate.identity = identity
        }
    }

    private func rebindReference(
        _ claim: inout Qwen38PerformanceAttributionClaim,
        to identity: Qwen38MTPPerformanceScorecardModel
    ) {
        claim.reference = identity
        for index in claim.measurements.indices {
            claim.measurements[index].reference.identity = identity
        }
    }

    private func bindContinuousRouteEvidence(
        _ claim: inout Qwen38PerformanceAttributionClaim
    ) {
        let observationDigest = Gate.continuousBackendObservationDigest(claim.measurements)
        let receipt = routeReceipt(observationDigest: observationDigest)
        let route = Gate.routeIdentity(
            kind: .continuousBatchNoSpec,
            backendEvidenceKind: .liveProductionRoute,
            backendEvidenceID: receipt.evidenceID,
            backendObservationDigest: observationDigest,
            backendReceiptDigest: receipt.digest)
        claim.candidateRoute = route
        for pairIndex in claim.measurements.indices {
            claim.measurements[pairIndex].candidate.route = route
            for requestIndex in claim.measurements[pairIndex].candidate.requests.indices {
                claim.measurements[pairIndex].candidate.requests[requestIndex]
                    .routeObservation.backendObservationDigest = observationDigest
            }
        }
    }

    private func claimIdentities(
        _ kind: Qwen38PerformanceAttributionClaimKind
    ) -> Qwen38MTPPerformanceScorecardTrustedEngineIdentities {
        switch kind {
        case .scalarGDN:
            return .init(
                candidate: model(.scalar, .gdnOn, "scalar"),
                reference: model(.scalar, .gdnOff, "scalar"))
        case .exactMTP:
            return .init(
                candidate: model(.exactMTP, .gdnOn, "exact-mtp"),
                reference: model(.scalar, .gdnOn, "scalar"))
        case .continuousBatchNoSpec:
            return .init(
                candidate: model(.scalar, .gdnOn, "continuous-batch"),
                reference: model(.scalar, .gdnOn, "scalar"))
        case .prefixMatrix:
            let candidate = model(.scalar, .gdnOn, "prefix")
            var reference = model(.scalar, .gdnOn, "prefix-reference")
            reference.executionDigest = candidate.executionDigest
            return .init(
                candidate: candidate,
                reference: reference)
        case .bestStackExploratory:
            return .init(
                candidate: model(.exactMTP, .gdnOn, "exploratory"),
                reference: model(.scalar, .gdnOff, "scalar"))
        }
    }

    private func routeIdentities(
        _ kind: Qwen38PerformanceAttributionClaimKind
    ) -> (candidate: Qwen38PerformanceAttributionRouteIdentity, reference: Qwen38PerformanceAttributionRouteIdentity) {
        switch kind {
        case .scalarGDN:
            let route = Gate.routeIdentity(kind: .scalar)
            return (
                route,
                route
            )
        case .exactMTP:
            return (
                Gate.routeIdentity(kind: .exactMTP),
                Gate.routeIdentity(kind: .scalar)
            )
        case .continuousBatchNoSpec:
            return (
                Gate.routeIdentity(
                    kind: .continuousBatchNoSpec,
                    backendEvidenceKind: .liveProductionRoute,
                    backendEvidenceID: productionRouteEvidenceID,
                    backendObservationDigest: hex("0"),
                    backendReceiptDigest: hex("0")),
                Gate.routeIdentity(kind: .scalar)
            )
        case .prefixMatrix:
            let route = Gate.routeIdentity(kind: .prefixMatrix)
            return (
                route,
                route
            )
        case .bestStackExploratory:
            return (
                Gate.routeIdentity(kind: .exploratoryBestStack),
                Gate.routeIdentity(kind: .scalar)
            )
        }
    }

    private func cells(
        for kind: Qwen38PerformanceAttributionClaimKind
    ) -> [Qwen38PerformanceAttributionCellIdentity] {
        if kind == .prefixMatrix {
            return contexts.flatMap { context in
                prefixes.map { prefix in
                    cell(context: context, prefix: prefix)
                }
            }
        }
        return [cell(context: .tokens4096, prefix: kind == .exactMTP ? .cold : .cold)]
    }

    private func cell(
        context: Qwen38MTPPerformanceScorecardBenchmarkContextTokens,
        prefix: Qwen38MTPPerformanceScorecardPrefixKind
    ) -> Qwen38PerformanceAttributionCellIdentity {
        let tokenIDs = promptTokenIDs(count: context.rawValue)
        return Qwen38PerformanceAttributionCellIdentity(
            id: "\(context.rawValue)-\(prefix == .cold ? "cold" : "warm")",
            contextTokens: context,
            prefixKind: prefix,
            renderedPromptTokenCount: context.rawValue,
            promptTokenIDs: tokenIDs,
            promptTokenDigest: Gate.canonicalDigest(tokenIDs))
    }

    private func concurrencies(for kind: Qwen38PerformanceAttributionClaimKind) -> [Int] {
        kind == .continuousBatchNoSpec ? [2, 4] : [1]
    }

    private func pair(
        kind: Qwen38PerformanceAttributionClaimKind,
        cell: Qwen38PerformanceAttributionCellIdentity,
        concurrency: Int,
        sampleIndex: Int,
        identities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        routes: (candidate: Qwen38PerformanceAttributionRouteIdentity, reference: Qwen38PerformanceAttributionRouteIdentity)
    ) -> Qwen38PerformanceAttributionPairMeasurement {
        let order: Qwen38MTPPerformanceScorecardRunOrder =
            sampleIndex.isMultiple(of: 2) ? .candidateThenReference : .referenceThenCandidate
        return Qwen38PerformanceAttributionPairMeasurement(
            concurrency: concurrency,
            cellID: cell.id,
            sampleIndex: sampleIndex,
            warmup: sampleIndex < Gate.droppedWarmupSamplesPerCell,
            order: order,
            candidate: engine(
                identity: identities.candidate,
                route: routes.candidate,
                kind: kind,
                cell: cell,
                concurrency: concurrency,
                sampleIndex: sampleIndex,
                candidate: true),
            reference: engine(
                identity: identities.reference,
                route: routes.reference,
                kind: kind,
                cell: cell,
                concurrency: concurrency,
                sampleIndex: sampleIndex,
                candidate: false))
    }

    private func engine(
        identity: Qwen38MTPPerformanceScorecardModel,
        route: Qwen38PerformanceAttributionRouteIdentity,
        kind: Qwen38PerformanceAttributionClaimKind,
        cell: Qwen38PerformanceAttributionCellIdentity,
        concurrency: Int,
        sampleIndex: Int,
        candidate: Bool
    ) -> Qwen38PerformanceAttributionEngineMeasurement {
        let decodeSeconds = candidate ? 0.8 : 1.0
        let e2eSeconds = candidate ? 8.0 : 10.0
        let requests = (0..<concurrency).map { requestIndex in
            request(
                kind: kind,
                route: route,
                cell: cell,
                concurrency: concurrency,
                sampleIndex: sampleIndex,
                requestIndex: requestIndex,
                decodeSeconds: decodeSeconds,
                e2eSeconds: e2eSeconds)
        }
        return Qwen38PerformanceAttributionEngineMeasurement(
            identity: identity,
            route: route,
            requests: requests,
            wallSeconds: e2eSeconds,
            proposalCount: kind == .exactMTP && candidate ? 16 : 0,
            acceptedCount: kind == .exactMTP && candidate ? 12 : 0,
            cleanup: cleanup(seed: "\(kind.rawValue)-\(candidate)-\(cell.id)-\(concurrency)-\(sampleIndex)"))
    }

    private func request(
        kind: Qwen38PerformanceAttributionClaimKind,
        route: Qwen38PerformanceAttributionRouteIdentity,
        cell: Qwen38PerformanceAttributionCellIdentity,
        concurrency: Int,
        sampleIndex: Int,
        requestIndex: Int,
        decodeSeconds: Double,
        e2eSeconds: Double
    ) -> Qwen38PerformanceAttributionRequestMeasurement {
        Qwen38PerformanceAttributionRequestMeasurement(
            cellID: cell.id,
            promptTokenCount: cell.renderedPromptTokenCount,
            promptTokenDigest: cell.promptTokenDigest,
            routeObservation: routeObservation(
                kind: kind,
                route: route,
                concurrency: concurrency,
                sampleIndex: sampleIndex,
                requestIndex: requestIndex),
            warmPrefixEvidence: cell.prefixKind == .exactWarmPrefix
                ? warmEvidence(tokenCount: cell.renderedPromptTokenCount)
                : nil,
            prefillSeconds: 0.5,
            ttftSeconds: 0.7,
            decodeTokenCount: 100,
            decodeSeconds: decodeSeconds,
            e2eSeconds: e2eSeconds,
            outputTokenIDs: Array(0..<100),
            outputBytesDigest: digest("output-\(cell.id)-\(requestIndex)"),
            cacheDigest: digest("cache-\(cell.id)-\(requestIndex)"))
    }

    private func routeObservation(
        kind: Qwen38PerformanceAttributionClaimKind,
        route: Qwen38PerformanceAttributionRouteIdentity,
        concurrency: Int,
        sampleIndex: Int,
        requestIndex: Int
    ) -> Qwen38PerformanceAttributionRequestRouteObservation {
        let hasSchedulerEvidence = kind == .continuousBatchNoSpec
            && route.kind == .continuousBatchNoSpec
        let planBase = sampleIndex * 100
        return Qwen38PerformanceAttributionRequestRouteObservation(
            routeKind: route.kind,
            requestID: "request-\(kind.rawValue)-\(concurrency)-\(sampleIndex)-\(requestIndex)",
            planRevisionBefore: hasSchedulerEvidence ? planBase : 0,
            planRevisionAfter: hasSchedulerEvidence ? planBase + 2 : 0,
            stateRevisionBefore: hasSchedulerEvidence ? planBase + requestIndex : 0,
            stateRevisionAfter: hasSchedulerEvidence ? planBase + requestIndex + 1 : 0,
            sharedBatchPlanSequence: hasSchedulerEvidence ? planBase + 1 : 0,
            sharedOccupancy: hasSchedulerEvidence ? concurrency : 0,
            overlapObserved: hasSchedulerEvidence,
            speculationUsed: false,
            backendObservationDigest: hasSchedulerEvidence
                ? digest("placeholder-backend-observation")
                : nil)
    }

    private func absoluteAuthority(
        for claim: Qwen38PerformanceAttributionClaim
    ) -> Qwen38PerformanceAttributionAbsoluteAuthority {
        let bands = claim.scheduledCells.flatMap { cell in
            concurrencies(for: claim.kind).map { concurrency in
                Qwen38PerformanceAttributionAbsoluteBand(
                    claimKind: claim.kind,
                    concurrency: concurrency,
                    contextTokens: cell.contextTokens,
                    prefixKind: cell.prefixKind,
                    maxPrefillSeconds: 1.0,
                    maxTTFTSeconds: 1.0,
                    minDecodeTokensPerSecond: 50.0,
                    minAggregateThroughputTokensPerSecond: 10.0)
            }
        }
        return Gate.absoluteAuthority(evidenceID: hex("1"), bands: bands)
    }

    private var cleanupAuthority: Qwen38PerformanceAttributionCleanupAuthority {
        Gate.cleanupAuthority(
            evidenceID: hex("2"),
            minIdleSamples: 3,
            cooldownSeconds: 5.0,
            maxRSSDeltaBytes: 1_000,
            maxActiveMetalDeltaBytes: 1_000,
            maxCachedMetalDeltaBytes: 1_000,
            maxSwapDeltaBytes: 0,
            maxPageoutDelta: 0,
            allowedPressureStates: ["normal"],
            allowedThermalStates: ["nominal", "fair"])
    }

    private func cleanup(seed: String) -> Qwen38PerformanceAttributionCleanupEvidence {
        var evidence = Qwen38PerformanceAttributionCleanupEvidence(
            evidenceDigest: "",
            baselineRSSBytes: 1_000,
            finalRSSBytes: 1_000,
            baselineActiveMetalBytes: 2_000,
            finalActiveMetalBytes: 2_000,
            baselineCachedMetalBytes: 500,
            finalCachedMetalBytes: 500,
            baselineSwapBytes: 0,
            finalSwapBytes: 0,
            baselinePageouts: 10,
            finalPageouts: 10,
            pressureBefore: "normal",
            pressureAfter: "normal",
            thermalBefore: "nominal",
            thermalAfter: "fair",
            cooldownSeconds: 5.0,
            idleSampleCount: 3,
            boundedCooldownObserved: true)
        evidence.evidenceDigest = Gate.cleanupEvidenceDigest(evidence)
        return evidence
    }

    private func warmEvidence(tokenCount: Int) -> Qwen38PerformanceAttributionWarmPrefixEvidence {
        let ids = promptTokenIDs(count: tokenCount)
        let snapshotDigest = Gate.canonicalDigest(ids)
        return Qwen38PerformanceAttributionWarmPrefixEvidence(
            snapshotCanonicalDigest: snapshotDigest,
            tokenIDs: ids,
            tokenCount: tokenCount,
            rebuildTokenIDs: ids,
            rebuildTokenCount: tokenCount,
            restoredCanonicalDigest: snapshotDigest,
            rebuildCanonicalDigest: snapshotDigest)
    }

    private var sampleCount: Int {
        Gate.droppedWarmupSamplesPerCell + Gate.measuredSamplesPerCell
    }

    private func promptTokenIDs(count: Int) -> [Int] {
        Array(0..<count)
    }

    private func model(
        _ executionMode: Qwen38MTPPerformanceScorecardExecutionMode,
        _ gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
        _ executionSalt: String
    ) -> Qwen38MTPPerformanceScorecardModel {
        Qwen38MTPPerformanceScorecardModel(
            label: "qwen38-attribution-fixture",
            executionMode: executionMode,
            artifact: artifact,
            executionDigest: digest("execution-\(executionSalt)"),
            sourceDigest: sourceDigest,
            gdnMode: gdnMode,
            launchBinding: launchBinding(
                mode: gdnMode,
                isolation: digest("isolation-\(executionSalt)-\(gdnMode.rawValue)")))
    }

    private func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        isolation: String
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let observed: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            mode == .gdnOn ? .enabled : .disabled
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: sourceDigest,
            observedEnv: observed,
            processIsolationEvidenceID: isolation,
            launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: mode,
                sourceDigest: sourceDigest,
                observedEnv: observed,
                processIsolationEvidenceID: isolation))
    }

    private var artifact: Qwen38MTPPerformanceScorecardArtifact {
        Qwen38MTPPerformanceScorecardGate.requiredArtifact
    }

    private var sourceDigest: String {
        Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
    }

    private var promotionPolicyEvidenceID: String {
        digest("externally-pinned-promotion-policy")
    }

    private var productionRouteEvidenceID: String {
        digest("production-route-receipt")
    }

    private var backendBuildIdentityDigest: String {
        digest("production-continuous-backend-build")
    }

    private var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: Gate.flagshipMeasurementClass,
            hardwareChip: "Apple M3 Ultra fixture",
            hardwareRAMBytes: Gate.flagshipMinimumRAMBytes,
            hardwareOSBuild: "fixture-os",
            hostIdentityDigest: digest("host"),
            harnessGitSHA: String(repeating: "1", count: 40),
            candidateMLXSwiftVersion: "fixture-mlx-swift",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: "qwen38-attribution-fixture",
            modelConfigHash: artifact.targetConfigSHA256,
            modelCheckpointManifestHash: artifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: "fixture-corpus",
            corpusContentHash: digest("corpus"))
    }

    private var contexts: [Qwen38MTPPerformanceScorecardBenchmarkContextTokens] {
        [.tokens4096, .tokens16384, .tokens32768]
    }

    private var prefixes: [Qwen38MTPPerformanceScorecardPrefixKind] {
        [.cold, .exactWarmPrefix]
    }

    private func digest(_ value: String) -> String {
        Qwen38PerformanceAttributionScorecardGate.canonicalDigest(["value": value])
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}
