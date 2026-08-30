import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38PerformanceAttributionScorecardProducerTests: XCTestCase {
    private typealias Producer = Qwen38PerformanceAttributionScorecardProducer
    private typealias Gate = Qwen38PerformanceAttributionScorecardGate
    private typealias Error = Qwen38PerformanceAttributionScorecardGateError

    func testProducerFailsClosedWhenRequiredClaimsAreAbsent() {
        XCTAssertThrowsError(try Producer.sealEvaluatedForFlagshipPromotion(
            artifact: artifact,
            runIdentity: runIdentity,
            claims: [],
            context: invalidContext)) { error in
            XCTAssertEqual(error as? Error, .missingClaim(.scalarGDN))
        }
    }

    func testProducerDoesNotAcceptCallerSuppliedQualifiedVerdictForIncompleteClaims() {
        XCTAssertThrowsError(try Producer.sealForFlagshipPromotion(
            artifact: artifact,
            runIdentity: runIdentity,
            claims: [],
            verdict: Qwen38PerformanceAttributionScorecardVerdict(qualified: true),
            context: invalidContext)) { error in
            XCTAssertEqual(error as? Error, .missingClaim(.scalarGDN))
        }
    }

    func testAuthorityDigestsAreCanonicalAndDeterministic() {
        let band = Qwen38PerformanceAttributionAbsoluteBand(
            claimKind: .scalarGDN,
            concurrency: 1,
            contextTokens: .tokens4096,
            prefixKind: .cold,
            maxPrefillSeconds: 1.0,
            maxTTFTSeconds: 1.0,
            minDecodeTokensPerSecond: 1.0,
            minAggregateThroughputTokensPerSecond: 1.0)

        let first = Gate.absoluteAuthority(evidenceID: hex("1"), bands: [band])
        let second = Gate.absoluteAuthority(evidenceID: hex("1"), bands: [band])

        XCTAssertEqual(first.digest, second.digest)
        XCTAssertEqual(
            Gate.cleanupAuthority(
                evidenceID: hex("2"),
                minIdleSamples: 3,
                cooldownSeconds: 5.0,
                maxRSSDeltaBytes: 1,
                maxActiveMetalDeltaBytes: 1,
                maxCachedMetalDeltaBytes: 1,
                maxSwapDeltaBytes: 0,
                maxPageoutDelta: 0,
                allowedPressureStates: ["normal"],
                allowedThermalStates: ["nominal"]).digest,
            Gate.cleanupAuthority(
                evidenceID: hex("2"),
                minIdleSamples: 3,
                cooldownSeconds: 5.0,
                maxRSSDeltaBytes: 1,
                maxActiveMetalDeltaBytes: 1,
                maxCachedMetalDeltaBytes: 1,
                maxSwapDeltaBytes: 0,
                maxPageoutDelta: 0,
                allowedPressureStates: ["normal"],
                allowedThermalStates: ["nominal"]).digest)

        let policyFirst = Gate.frozenPromotionPolicy(
            evidenceID: hex("3"),
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: [])
        let policySecond = Gate.frozenPromotionPolicy(
            evidenceID: hex("3"),
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: [])
        XCTAssertEqual(policyFirst.digest, policySecond.digest)

        let receiptFirst = Gate.productionRouteReceipt(
            evidenceID: hex("4"),
            artifact: artifact,
            runIdentity: runIdentity,
            backendBuildIdentityDigest: hex("5"),
            observationDigest: hex("6"))
        let receiptSecond = Gate.productionRouteReceipt(
            evidenceID: hex("4"),
            artifact: artifact,
            runIdentity: runIdentity,
            backendBuildIdentityDigest: hex("5"),
            observationDigest: hex("6"))
        XCTAssertEqual(receiptFirst.digest, receiptSecond.digest)
    }

    private var invalidContext: Qwen38FlagshipPromotionContext {
        let policy = Gate.frozenPromotionPolicy(
            evidenceID: hex("3"),
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: [])
        let receipt = Gate.productionRouteReceipt(
            evidenceID: hex("4"),
            artifact: artifact,
            runIdentity: runIdentity,
            backendBuildIdentityDigest: hex("5"),
            observationDigest: hex("6"))
        return Qwen38FlagshipPromotionContext(
            frozenPolicy: policy,
            expectedPolicyDigest: policy.digest,
            productionRouteReceipt: receipt,
            expectedReceiptDigest: receipt.digest)
    }

    private var artifact: Qwen38MTPPerformanceScorecardArtifact {
        Qwen38MTPPerformanceScorecardGate.requiredArtifact
    }

    private var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: "producer-fixture",
            hardwareChip: "fixture-chip",
            hardwareRAMBytes: 1,
            hardwareOSBuild: "fixture-os",
            hostIdentityDigest: digest("host"),
            harnessGitSHA: String(repeating: "1", count: 40),
            candidateMLXSwiftVersion: "fixture-mlx-swift",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: "fixture-model",
            modelConfigHash: artifact.targetConfigSHA256,
            modelCheckpointManifestHash: artifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: "fixture-corpus",
            corpusContentHash: digest("corpus"))
    }

    private func digest(_ value: String) -> String {
        Gate.canonicalDigest(["value": value])
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}
