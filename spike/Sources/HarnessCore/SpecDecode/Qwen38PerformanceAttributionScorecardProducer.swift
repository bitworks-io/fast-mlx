import Foundation

public enum Qwen38PerformanceAttributionScorecardProducer {
    public static func sealForFlagshipPromotion(
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        claims: [Qwen38PerformanceAttributionClaim],
        exploratoryBestStack: Qwen38PerformanceAttributionClaim? = nil,
        verdict: Qwen38PerformanceAttributionScorecardVerdict,
        context: Qwen38FlagshipPromotionContext
    ) throws -> Qwen38PerformanceAttributionScorecard {
        let envelopeDigest = Qwen38PerformanceAttributionScorecardGate.envelopeDigest(
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: context.expectedPolicyDigest,
            productionRouteReceiptDigest: context.expectedReceiptDigest,
            claims: claims,
            exploratoryBestStack: exploratoryBestStack)
        let scorecard = Qwen38PerformanceAttributionScorecard(
            schemaVersion: Qwen38PerformanceAttributionScorecardGate.schemaVersion,
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: context.expectedPolicyDigest,
            productionRouteReceiptDigest: context.expectedReceiptDigest,
            envelopeDigest: envelopeDigest,
            claims: claims,
            exploratoryBestStack: exploratoryBestStack,
            verdict: verdict)
        _ = try Qwen38PerformanceAttributionScorecardGate.validateForFlagshipPromotion(
            scorecard,
            context: context)
        return scorecard
    }

    public static func sealEvaluatedForFlagshipPromotion(
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        claims: [Qwen38PerformanceAttributionClaim],
        exploratoryBestStack: Qwen38PerformanceAttributionClaim? = nil,
        context: Qwen38FlagshipPromotionContext
    ) throws -> Qwen38PerformanceAttributionScorecard {
        let evaluatedClaims = try claims.map(evaluatedClaim)
        let evaluatedExploratory = try exploratoryBestStack.map(evaluatedClaim)
        let verdict = Qwen38PerformanceAttributionScorecardVerdict(
            qualified: evaluatedClaims.allSatisfy(\.verdict.qualified))
        return try sealForFlagshipPromotion(
            artifact: artifact,
            runIdentity: runIdentity,
            claims: evaluatedClaims,
            exploratoryBestStack: evaluatedExploratory,
            verdict: verdict,
            context: context)
    }

    static func evaluatedClaim(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Qwen38PerformanceAttributionClaim {
        var updated = claim
        updated.metrics = try Qwen38PerformanceAttributionScorecardGate.computeMetrics(updated)
        updated.verdict = try Qwen38PerformanceAttributionScorecardGate.evaluateClaim(updated)
        return updated
    }
}
