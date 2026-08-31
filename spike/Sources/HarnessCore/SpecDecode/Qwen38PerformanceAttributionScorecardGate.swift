import CryptoKit
import Foundation

public enum Qwen38PerformanceAttributionClaimKind: String, Codable, Equatable, Hashable, Sendable {
    case scalarGDN
    case exactMTP
    case continuousBatchNoSpec
    case prefixMatrix
    case bestStackExploratory
}

public enum Qwen38PerformanceAttributionRouteKind: String, Codable, Equatable, Sendable {
    case scalar
    case exactMTP
    case continuousBatchNoSpec
    case prefixMatrix
    case exploratoryBestStack
}

public enum Qwen38PerformanceAttributionBackendEvidenceKind: String, Codable, Equatable, Sendable {
    case liveProductionRoute
    case syntheticPathProof
}

public struct Qwen38PerformanceAttributionRouteIdentity: Codable, Equatable, Sendable {
    public var kind: Qwen38PerformanceAttributionRouteKind
    public var routeDigest: String
    public var backendEvidenceKind: Qwen38PerformanceAttributionBackendEvidenceKind?
    public var backendEvidenceID: String?
    public var backendObservationDigest: String?
    public var backendReceiptDigest: String?

    public init(
        kind: Qwen38PerformanceAttributionRouteKind,
        routeDigest: String,
        backendEvidenceKind: Qwen38PerformanceAttributionBackendEvidenceKind? = nil,
        backendEvidenceID: String? = nil,
        backendObservationDigest: String? = nil,
        backendReceiptDigest: String? = nil
    ) {
        self.kind = kind
        self.routeDigest = routeDigest
        self.backendEvidenceKind = backendEvidenceKind
        self.backendEvidenceID = backendEvidenceID
        self.backendObservationDigest = backendObservationDigest
        self.backendReceiptDigest = backendReceiptDigest
    }
}

public struct Qwen38PerformanceAttributionCellIdentity: Codable, Equatable, Sendable {
    public var id: String
    public var contextTokens: Qwen38MTPPerformanceScorecardBenchmarkContextTokens
    public var prefixKind: Qwen38MTPPerformanceScorecardPrefixKind
    public var renderedPromptTokenCount: Int
    public var promptTokenIDs: [Int]
    public var promptTokenDigest: String

    public init(
        id: String,
        contextTokens: Qwen38MTPPerformanceScorecardBenchmarkContextTokens,
        prefixKind: Qwen38MTPPerformanceScorecardPrefixKind,
        renderedPromptTokenCount: Int,
        promptTokenIDs: [Int],
        promptTokenDigest: String
    ) {
        self.id = id
        self.contextTokens = contextTokens
        self.prefixKind = prefixKind
        self.renderedPromptTokenCount = renderedPromptTokenCount
        self.promptTokenIDs = promptTokenIDs
        self.promptTokenDigest = promptTokenDigest
    }
}

public struct Qwen38PerformanceAttributionWarmPrefixEvidence: Codable, Equatable, Sendable {
    public var snapshotCanonicalDigest: String
    public var tokenIDs: [Int]
    public var tokenCount: Int
    public var rebuildTokenIDs: [Int]
    public var rebuildTokenCount: Int
    public var restoredCanonicalDigest: String
    public var rebuildCanonicalDigest: String

    public init(
        snapshotCanonicalDigest: String,
        tokenIDs: [Int],
        tokenCount: Int,
        rebuildTokenIDs: [Int],
        rebuildTokenCount: Int,
        restoredCanonicalDigest: String,
        rebuildCanonicalDigest: String
    ) {
        self.snapshotCanonicalDigest = snapshotCanonicalDigest
        self.tokenIDs = tokenIDs
        self.tokenCount = tokenCount
        self.rebuildTokenIDs = rebuildTokenIDs
        self.rebuildTokenCount = rebuildTokenCount
        self.restoredCanonicalDigest = restoredCanonicalDigest
        self.rebuildCanonicalDigest = rebuildCanonicalDigest
    }
}

public struct Qwen38PerformanceAttributionRequestRouteObservation: Codable, Equatable, Sendable {
    public var routeKind: Qwen38PerformanceAttributionRouteKind
    public var requestID: String
    public var planRevisionBefore: Int
    public var planRevisionAfter: Int
    public var stateRevisionBefore: Int
    public var stateRevisionAfter: Int
    public var sharedBatchPlanSequence: Int
    public var sharedOccupancy: Int
    public var overlapObserved: Bool
    public var speculationUsed: Bool
    public var backendObservationDigest: String?

    public init(
        routeKind: Qwen38PerformanceAttributionRouteKind,
        requestID: String,
        planRevisionBefore: Int,
        planRevisionAfter: Int,
        stateRevisionBefore: Int,
        stateRevisionAfter: Int,
        sharedBatchPlanSequence: Int = 0,
        sharedOccupancy: Int,
        overlapObserved: Bool,
        speculationUsed: Bool,
        backendObservationDigest: String? = nil
    ) {
        self.routeKind = routeKind
        self.requestID = requestID
        self.planRevisionBefore = planRevisionBefore
        self.planRevisionAfter = planRevisionAfter
        self.stateRevisionBefore = stateRevisionBefore
        self.stateRevisionAfter = stateRevisionAfter
        self.sharedBatchPlanSequence = sharedBatchPlanSequence
        self.sharedOccupancy = sharedOccupancy
        self.overlapObserved = overlapObserved
        self.speculationUsed = speculationUsed
        self.backendObservationDigest = backendObservationDigest
    }
}

public struct Qwen38PerformanceAttributionRequestMeasurement: Codable, Equatable, Sendable {
    public var cellID: String
    public var promptTokenCount: Int
    public var promptTokenDigest: String
    public var routeObservation: Qwen38PerformanceAttributionRequestRouteObservation
    public var warmPrefixEvidence: Qwen38PerformanceAttributionWarmPrefixEvidence?
    public var prefillSeconds: Double
    public var ttftSeconds: Double
    public var decodeTokenCount: Int
    public var decodeSeconds: Double
    public var e2eSeconds: Double
    public var outputTokenIDs: [Int]
    public var outputBytesDigest: String
    public var cacheDigest: String

    public init(
        cellID: String,
        promptTokenCount: Int,
        promptTokenDigest: String,
        routeObservation: Qwen38PerformanceAttributionRequestRouteObservation,
        warmPrefixEvidence: Qwen38PerformanceAttributionWarmPrefixEvidence? = nil,
        prefillSeconds: Double,
        ttftSeconds: Double,
        decodeTokenCount: Int,
        decodeSeconds: Double,
        e2eSeconds: Double,
        outputTokenIDs: [Int],
        outputBytesDigest: String,
        cacheDigest: String
    ) {
        self.cellID = cellID
        self.promptTokenCount = promptTokenCount
        self.promptTokenDigest = promptTokenDigest
        self.routeObservation = routeObservation
        self.warmPrefixEvidence = warmPrefixEvidence
        self.prefillSeconds = prefillSeconds
        self.ttftSeconds = ttftSeconds
        self.decodeTokenCount = decodeTokenCount
        self.decodeSeconds = decodeSeconds
        self.e2eSeconds = e2eSeconds
        self.outputTokenIDs = outputTokenIDs
        self.outputBytesDigest = outputBytesDigest
        self.cacheDigest = cacheDigest
    }

    public var decodeTokensPerSecond: Double {
        Double(decodeTokenCount) / decodeSeconds
    }
}

public struct Qwen38PerformanceAttributionCleanupEvidence: Codable, Equatable, Sendable {
    public var evidenceDigest: String
    public var baselineRSSBytes: UInt64
    public var finalRSSBytes: UInt64
    public var baselineActiveMetalBytes: UInt64
    public var finalActiveMetalBytes: UInt64
    public var baselineCachedMetalBytes: UInt64
    public var finalCachedMetalBytes: UInt64
    public var baselineSwapBytes: UInt64
    public var finalSwapBytes: UInt64
    public var baselinePageouts: UInt64
    public var finalPageouts: UInt64
    public var pressureBefore: String
    public var pressureAfter: String
    public var thermalBefore: String
    public var thermalAfter: String
    public var cooldownSeconds: Double
    public var idleSampleCount: Int
    public var boundedCooldownObserved: Bool

    public init(
        evidenceDigest: String,
        baselineRSSBytes: UInt64,
        finalRSSBytes: UInt64,
        baselineActiveMetalBytes: UInt64,
        finalActiveMetalBytes: UInt64,
        baselineCachedMetalBytes: UInt64,
        finalCachedMetalBytes: UInt64,
        baselineSwapBytes: UInt64,
        finalSwapBytes: UInt64,
        baselinePageouts: UInt64,
        finalPageouts: UInt64,
        pressureBefore: String,
        pressureAfter: String,
        thermalBefore: String,
        thermalAfter: String,
        cooldownSeconds: Double,
        idleSampleCount: Int,
        boundedCooldownObserved: Bool
    ) {
        self.evidenceDigest = evidenceDigest
        self.baselineRSSBytes = baselineRSSBytes
        self.finalRSSBytes = finalRSSBytes
        self.baselineActiveMetalBytes = baselineActiveMetalBytes
        self.finalActiveMetalBytes = finalActiveMetalBytes
        self.baselineCachedMetalBytes = baselineCachedMetalBytes
        self.finalCachedMetalBytes = finalCachedMetalBytes
        self.baselineSwapBytes = baselineSwapBytes
        self.finalSwapBytes = finalSwapBytes
        self.baselinePageouts = baselinePageouts
        self.finalPageouts = finalPageouts
        self.pressureBefore = pressureBefore
        self.pressureAfter = pressureAfter
        self.thermalBefore = thermalBefore
        self.thermalAfter = thermalAfter
        self.cooldownSeconds = cooldownSeconds
        self.idleSampleCount = idleSampleCount
        self.boundedCooldownObserved = boundedCooldownObserved
    }
}

public struct Qwen38PerformanceAttributionEngineMeasurement: Codable, Equatable, Sendable {
    public var identity: Qwen38MTPPerformanceScorecardModel
    public var route: Qwen38PerformanceAttributionRouteIdentity
    public var requests: [Qwen38PerformanceAttributionRequestMeasurement]
    public var wallSeconds: Double
    public var proposalCount: Int
    public var acceptedCount: Int
    public var cleanup: Qwen38PerformanceAttributionCleanupEvidence

    public init(
        identity: Qwen38MTPPerformanceScorecardModel,
        route: Qwen38PerformanceAttributionRouteIdentity,
        requests: [Qwen38PerformanceAttributionRequestMeasurement],
        wallSeconds: Double,
        proposalCount: Int,
        acceptedCount: Int,
        cleanup: Qwen38PerformanceAttributionCleanupEvidence
    ) {
        self.identity = identity
        self.route = route
        self.requests = requests
        self.wallSeconds = wallSeconds
        self.proposalCount = proposalCount
        self.acceptedCount = acceptedCount
        self.cleanup = cleanup
    }
}

public struct Qwen38PerformanceAttributionPairMeasurement: Codable, Equatable, Sendable {
    public var concurrency: Int
    public var cellID: String
    public var sampleIndex: Int
    public var warmup: Bool
    public var order: Qwen38MTPPerformanceScorecardRunOrder
    public var candidate: Qwen38PerformanceAttributionEngineMeasurement
    public var reference: Qwen38PerformanceAttributionEngineMeasurement

    public init(
        concurrency: Int,
        cellID: String,
        sampleIndex: Int,
        warmup: Bool,
        order: Qwen38MTPPerformanceScorecardRunOrder,
        candidate: Qwen38PerformanceAttributionEngineMeasurement,
        reference: Qwen38PerformanceAttributionEngineMeasurement
    ) {
        self.concurrency = concurrency
        self.cellID = cellID
        self.sampleIndex = sampleIndex
        self.warmup = warmup
        self.order = order
        self.candidate = candidate
        self.reference = reference
    }
}

public struct Qwen38PerformanceAttributionAbsoluteBand: Codable, Equatable, Sendable {
    public var claimKind: Qwen38PerformanceAttributionClaimKind
    public var concurrency: Int
    public var contextTokens: Qwen38MTPPerformanceScorecardBenchmarkContextTokens
    public var prefixKind: Qwen38MTPPerformanceScorecardPrefixKind
    public var maxPrefillSeconds: Double
    public var maxTTFTSeconds: Double
    public var minDecodeTokensPerSecond: Double
    public var minAggregateThroughputTokensPerSecond: Double

    public init(
        claimKind: Qwen38PerformanceAttributionClaimKind,
        concurrency: Int,
        contextTokens: Qwen38MTPPerformanceScorecardBenchmarkContextTokens,
        prefixKind: Qwen38MTPPerformanceScorecardPrefixKind,
        maxPrefillSeconds: Double,
        maxTTFTSeconds: Double,
        minDecodeTokensPerSecond: Double,
        minAggregateThroughputTokensPerSecond: Double
    ) {
        self.claimKind = claimKind
        self.concurrency = concurrency
        self.contextTokens = contextTokens
        self.prefixKind = prefixKind
        self.maxPrefillSeconds = maxPrefillSeconds
        self.maxTTFTSeconds = maxTTFTSeconds
        self.minDecodeTokensPerSecond = minDecodeTokensPerSecond
        self.minAggregateThroughputTokensPerSecond = minAggregateThroughputTokensPerSecond
    }
}

public struct Qwen38PerformanceAttributionAbsoluteAuthority: Codable, Equatable, Sendable {
    public var evidenceID: String
    public var sealedBeforeMeasurements: Bool
    public var bands: [Qwen38PerformanceAttributionAbsoluteBand]
    public var digest: String

    public init(
        evidenceID: String,
        sealedBeforeMeasurements: Bool,
        bands: [Qwen38PerformanceAttributionAbsoluteBand],
        digest: String
    ) {
        self.evidenceID = evidenceID
        self.sealedBeforeMeasurements = sealedBeforeMeasurements
        self.bands = bands
        self.digest = digest
    }
}

public struct Qwen38PerformanceAttributionCleanupAuthority: Codable, Equatable, Sendable {
    public var evidenceID: String
    public var digest: String
    public var minIdleSamples: Int
    public var cooldownSeconds: Double
    public var maxRSSDeltaBytes: UInt64
    public var maxActiveMetalDeltaBytes: UInt64
    public var maxCachedMetalDeltaBytes: UInt64
    public var maxSwapDeltaBytes: UInt64
    public var maxPageoutDelta: UInt64
    public var allowedPressureStates: [String]
    public var allowedThermalStates: [String]

    public init(
        evidenceID: String,
        digest: String,
        minIdleSamples: Int,
        cooldownSeconds: Double,
        maxRSSDeltaBytes: UInt64,
        maxActiveMetalDeltaBytes: UInt64,
        maxCachedMetalDeltaBytes: UInt64,
        maxSwapDeltaBytes: UInt64,
        maxPageoutDelta: UInt64,
        allowedPressureStates: [String],
        allowedThermalStates: [String]
    ) {
        self.evidenceID = evidenceID
        self.digest = digest
        self.minIdleSamples = minIdleSamples
        self.cooldownSeconds = cooldownSeconds
        self.maxRSSDeltaBytes = maxRSSDeltaBytes
        self.maxActiveMetalDeltaBytes = maxActiveMetalDeltaBytes
        self.maxCachedMetalDeltaBytes = maxCachedMetalDeltaBytes
        self.maxSwapDeltaBytes = maxSwapDeltaBytes
        self.maxPageoutDelta = maxPageoutDelta
        self.allowedPressureStates = allowedPressureStates
        self.allowedThermalStates = allowedThermalStates
    }
}

/// A pre-measurement policy whose expected digest must be pinned outside the scorecard producer.
/// The policy duplicates the exact artifact, run identity, and per-claim authorities so a result
/// cannot widen its own target or cleanup bands and then self-seal.
public struct Qwen38PerformanceAttributionFrozenPromotionPolicy:
    Codable, Equatable, Sendable
{
    public var schemaVersion: Int
    public var evidenceID: String
    public var sealedBeforeMeasurements: Bool
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    public var claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]
    public var digest: String

    public init(
        schemaVersion: Int,
        evidenceID: String,
        sealedBeforeMeasurements: Bool,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority],
        digest: String
    ) {
        self.schemaVersion = schemaVersion
        self.evidenceID = evidenceID
        self.sealedBeforeMeasurements = sealedBeforeMeasurements
        self.artifact = artifact
        self.runIdentity = runIdentity
        self.claimAuthorities = claimAuthorities
        self.digest = digest
    }
}

public struct Qwen38PerformanceAttributionFrozenClaimAuthority:
    Codable, Equatable, Sendable
{
    public var claimKind: Qwen38PerformanceAttributionClaimKind
    public var absoluteAuthority: Qwen38PerformanceAttributionAbsoluteAuthority
    public var cleanupAuthority: Qwen38PerformanceAttributionCleanupAuthority

    public init(
        claimKind: Qwen38PerformanceAttributionClaimKind,
        absoluteAuthority: Qwen38PerformanceAttributionAbsoluteAuthority,
        cleanupAuthority: Qwen38PerformanceAttributionCleanupAuthority
    ) {
        self.claimKind = claimKind
        self.absoluteAuthority = absoluteAuthority
        self.cleanupAuthority = cleanupAuthority
    }
}

/// A post-run receipt emitted at the production continuous-backend boundary. Its expected digest
/// is independently pinned by the evidence controller; a canonical digest is not a signature.
public struct Qwen38PerformanceAttributionProductionRouteReceipt:
    Codable, Equatable, Sendable
{
    public var schemaVersion: Int
    public var evidenceID: String
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var runIdentityDigest: String
    public var backendBuildIdentityDigest: String
    public var observationDigest: String
    public var digest: String

    public init(
        schemaVersion: Int,
        evidenceID: String,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentityDigest: String,
        backendBuildIdentityDigest: String,
        observationDigest: String,
        digest: String
    ) {
        self.schemaVersion = schemaVersion
        self.evidenceID = evidenceID
        self.artifact = artifact
        self.runIdentityDigest = runIdentityDigest
        self.backendBuildIdentityDigest = backendBuildIdentityDigest
        self.observationDigest = observationDigest
        self.digest = digest
    }
}

/// Trust material supplied from outside result collection. The expected digests must come from a
/// compiled registry or separately pinned operator input, never from the scorecard being checked.
public struct Qwen38FlagshipPromotionContext: Equatable, Sendable {
    public let frozenPolicy: Qwen38PerformanceAttributionFrozenPromotionPolicy
    public let expectedPolicyDigest: String
    public let productionRouteReceipt: Qwen38PerformanceAttributionProductionRouteReceipt
    public let expectedReceiptDigest: String

    public init(
        frozenPolicy: Qwen38PerformanceAttributionFrozenPromotionPolicy,
        expectedPolicyDigest: String,
        productionRouteReceipt: Qwen38PerformanceAttributionProductionRouteReceipt,
        expectedReceiptDigest: String
    ) {
        self.frozenPolicy = frozenPolicy
        self.expectedPolicyDigest = expectedPolicyDigest
        self.productionRouteReceipt = productionRouteReceipt
        self.expectedReceiptDigest = expectedReceiptDigest
    }
}

public struct Qwen38PerformanceAttributionCellMetrics: Codable, Equatable, Sendable {
    public var cellID: String
    public var concurrency: Int
    public var aggregateThroughputRatio: Double
    public var e2ELatencyRatio: Double
    public var e2EP95LatencyRatio: Double
    public var ttftLatencyRatio: Double
    public var decodeTokensPerSecondRatio: Double
    public var candidateAggregateThroughput: Double
    public var referenceAggregateThroughput: Double

    public init(
        cellID: String,
        concurrency: Int,
        aggregateThroughputRatio: Double,
        e2ELatencyRatio: Double,
        e2EP95LatencyRatio: Double,
        ttftLatencyRatio: Double,
        decodeTokensPerSecondRatio: Double,
        candidateAggregateThroughput: Double,
        referenceAggregateThroughput: Double
    ) {
        self.cellID = cellID
        self.concurrency = concurrency
        self.aggregateThroughputRatio = aggregateThroughputRatio
        self.e2ELatencyRatio = e2ELatencyRatio
        self.e2EP95LatencyRatio = e2EP95LatencyRatio
        self.ttftLatencyRatio = ttftLatencyRatio
        self.decodeTokensPerSecondRatio = decodeTokensPerSecondRatio
        self.candidateAggregateThroughput = candidateAggregateThroughput
        self.referenceAggregateThroughput = referenceAggregateThroughput
    }
}

public struct Qwen38PerformanceAttributionClaimMetrics: Codable, Equatable, Sendable {
    public var cells: [Qwen38PerformanceAttributionCellMetrics]

    public init(cells: [Qwen38PerformanceAttributionCellMetrics]) {
        self.cells = cells
    }

    public static let empty = Qwen38PerformanceAttributionClaimMetrics(cells: [])
}

public struct Qwen38PerformanceAttributionClaimVerdict: Codable, Equatable, Sendable {
    public var qualified: Bool
    public var exploratory: Bool

    public init(qualified: Bool, exploratory: Bool) {
        self.qualified = qualified
        self.exploratory = exploratory
    }

    public static let unqualified = Qwen38PerformanceAttributionClaimVerdict(
        qualified: false,
        exploratory: false)
}

public struct Qwen38PerformanceAttributionScorecardVerdict: Codable, Equatable, Sendable {
    public var qualified: Bool

    public init(qualified: Bool) {
        self.qualified = qualified
    }

    public static let unqualified = Qwen38PerformanceAttributionScorecardVerdict(qualified: false)
}

public struct Qwen38PerformanceAttributionClaim: Codable, Equatable, Sendable {
    public var kind: Qwen38PerformanceAttributionClaimKind
    public var candidate: Qwen38MTPPerformanceScorecardModel
    public var reference: Qwen38MTPPerformanceScorecardModel
    public var candidateRoute: Qwen38PerformanceAttributionRouteIdentity
    public var referenceRoute: Qwen38PerformanceAttributionRouteIdentity
    public var scheduledCells: [Qwen38PerformanceAttributionCellIdentity]
    public var measurements: [Qwen38PerformanceAttributionPairMeasurement]
    public var absoluteAuthority: Qwen38PerformanceAttributionAbsoluteAuthority
    public var cleanupAuthority: Qwen38PerformanceAttributionCleanupAuthority
    public var metrics: Qwen38PerformanceAttributionClaimMetrics
    public var verdict: Qwen38PerformanceAttributionClaimVerdict

    public init(
        kind: Qwen38PerformanceAttributionClaimKind,
        candidate: Qwen38MTPPerformanceScorecardModel,
        reference: Qwen38MTPPerformanceScorecardModel,
        candidateRoute: Qwen38PerformanceAttributionRouteIdentity,
        referenceRoute: Qwen38PerformanceAttributionRouteIdentity,
        scheduledCells: [Qwen38PerformanceAttributionCellIdentity],
        measurements: [Qwen38PerformanceAttributionPairMeasurement],
        absoluteAuthority: Qwen38PerformanceAttributionAbsoluteAuthority,
        cleanupAuthority: Qwen38PerformanceAttributionCleanupAuthority,
        metrics: Qwen38PerformanceAttributionClaimMetrics,
        verdict: Qwen38PerformanceAttributionClaimVerdict
    ) {
        self.kind = kind
        self.candidate = candidate
        self.reference = reference
        self.candidateRoute = candidateRoute
        self.referenceRoute = referenceRoute
        self.scheduledCells = scheduledCells
        self.measurements = measurements
        self.absoluteAuthority = absoluteAuthority
        self.cleanupAuthority = cleanupAuthority
        self.metrics = metrics
        self.verdict = verdict
    }
}

public struct Qwen38PerformanceAttributionScorecard: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    public var promotionPolicyDigest: String
    public var productionRouteReceiptDigest: String
    public var envelopeDigest: String
    public var claims: [Qwen38PerformanceAttributionClaim]
    public var exploratoryBestStack: Qwen38PerformanceAttributionClaim?
    public var verdict: Qwen38PerformanceAttributionScorecardVerdict

    public init(
        schemaVersion: Int,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        promotionPolicyDigest: String,
        productionRouteReceiptDigest: String,
        envelopeDigest: String,
        claims: [Qwen38PerformanceAttributionClaim],
        exploratoryBestStack: Qwen38PerformanceAttributionClaim? = nil,
        verdict: Qwen38PerformanceAttributionScorecardVerdict
    ) {
        self.schemaVersion = schemaVersion
        self.artifact = artifact
        self.runIdentity = runIdentity
        self.promotionPolicyDigest = promotionPolicyDigest
        self.productionRouteReceiptDigest = productionRouteReceiptDigest
        self.envelopeDigest = envelopeDigest
        self.claims = claims
        self.exploratoryBestStack = exploratoryBestStack
        self.verdict = verdict
    }
}

public enum Qwen38PerformanceAttributionScorecardGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case schemaVersionMismatch(Int)
    case invalidEnvelope
    case invalidPromotionPolicy
    case invalidProductionRouteReceipt
    case missingClaim(Qwen38PerformanceAttributionClaimKind)
    case duplicateClaim(Qwen38PerformanceAttributionClaimKind)
    case invalidExploratoryClaim
    case invalidClaimIdentity(Qwen38PerformanceAttributionClaimKind)
    case invalidRouteEvidence(Qwen38PerformanceAttributionClaimKind)
    case invalidCell(Qwen38PerformanceAttributionClaimKind, String)
    case invalidWarmPrefixEvidence(Qwen38PerformanceAttributionClaimKind, String)
    case invalidAbsoluteAuthority(Qwen38PerformanceAttributionClaimKind)
    case invalidCleanup(Qwen38PerformanceAttributionClaimKind)
    case outputParityMismatch(Qwen38PerformanceAttributionClaimKind)
    case nonFiniteMetrics(Qwen38PerformanceAttributionClaimKind)
    case metricsMismatch(Qwen38PerformanceAttributionClaimKind)
    case verdictMismatch(Qwen38PerformanceAttributionClaimKind)
    case scorecardVerdictMismatch
    case unqualifiedScorecard

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let version): return "schemaVersion mismatch: \(version)"
        case .invalidEnvelope: return "invalid attribution scorecard envelope"
        case .invalidPromotionPolicy: return "invalid frozen flagship promotion policy"
        case .invalidProductionRouteReceipt: return "invalid production route receipt"
        case .missingClaim(let kind): return "missing required claim: \(kind.rawValue)"
        case .duplicateClaim(let kind): return "duplicate claim: \(kind.rawValue)"
        case .invalidExploratoryClaim: return "invalid exploratory best-stack claim"
        case .invalidClaimIdentity(let kind): return "invalid claim identity: \(kind.rawValue)"
        case .invalidRouteEvidence(let kind): return "invalid route evidence: \(kind.rawValue)"
        case .invalidCell(let kind, let id): return "invalid cell \(id): \(kind.rawValue)"
        case .invalidWarmPrefixEvidence(let kind, let id):
            return "invalid warm prefix evidence \(id): \(kind.rawValue)"
        case .invalidAbsoluteAuthority(let kind): return "invalid absolute authority: \(kind.rawValue)"
        case .invalidCleanup(let kind): return "invalid cleanup: \(kind.rawValue)"
        case .outputParityMismatch(let kind): return "output parity mismatch: \(kind.rawValue)"
        case .nonFiniteMetrics(let kind): return "non-finite metrics: \(kind.rawValue)"
        case .metricsMismatch(let kind): return "metrics mismatch: \(kind.rawValue)"
        case .verdictMismatch(let kind): return "verdict mismatch: \(kind.rawValue)"
        case .scorecardVerdictMismatch: return "scorecard verdict mismatch"
        case .unqualifiedScorecard: return "unqualified scorecard"
        }
    }
}

private struct Qwen38PerformanceAttributionAbsoluteDigestBasis: Codable, Equatable, Sendable {
    var evidenceID: String
    var sealedBeforeMeasurements: Bool
    var bands: [Qwen38PerformanceAttributionAbsoluteBand]
}

private struct Qwen38PerformanceAttributionCleanupDigestBasis: Codable, Equatable, Sendable {
    var evidenceID: String
    var minIdleSamples: Int
    var cooldownSeconds: Double
    var maxRSSDeltaBytes: UInt64
    var maxActiveMetalDeltaBytes: UInt64
    var maxCachedMetalDeltaBytes: UInt64
    var maxSwapDeltaBytes: UInt64
    var maxPageoutDelta: UInt64
    var allowedPressureStates: [String]
    var allowedThermalStates: [String]
}

private struct Qwen38PerformanceAttributionCleanupEvidenceDigestBasis: Codable, Equatable, Sendable {
    var baselineRSSBytes: UInt64
    var finalRSSBytes: UInt64
    var baselineActiveMetalBytes: UInt64
    var finalActiveMetalBytes: UInt64
    var baselineCachedMetalBytes: UInt64
    var finalCachedMetalBytes: UInt64
    var baselineSwapBytes: UInt64
    var finalSwapBytes: UInt64
    var baselinePageouts: UInt64
    var finalPageouts: UInt64
    var pressureBefore: String
    var pressureAfter: String
    var thermalBefore: String
    var thermalAfter: String
    var cooldownSeconds: Double
    var idleSampleCount: Int
    var boundedCooldownObserved: Bool
}

private struct Qwen38PerformanceAttributionRouteDigestBasis: Codable, Equatable, Sendable {
    var kind: Qwen38PerformanceAttributionRouteKind
    var backendEvidenceKind: Qwen38PerformanceAttributionBackendEvidenceKind?
    var backendEvidenceID: String?
    var backendObservationDigest: String?
    var backendReceiptDigest: String?
}

private struct Qwen38PerformanceAttributionFrozenPromotionPolicyDigestBasis:
    Codable, Equatable, Sendable
{
    var schemaVersion: Int
    var evidenceID: String
    var sealedBeforeMeasurements: Bool
    var artifact: Qwen38MTPPerformanceScorecardArtifact
    var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    var claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]
}

private struct Qwen38PerformanceAttributionProductionRouteReceiptDigestBasis:
    Codable, Equatable, Sendable
{
    var schemaVersion: Int
    var evidenceID: String
    var artifact: Qwen38MTPPerformanceScorecardArtifact
    var runIdentityDigest: String
    var backendBuildIdentityDigest: String
    var observationDigest: String
}

private struct Qwen38PerformanceAttributionBackendRequestDigestBasis: Codable, Equatable, Sendable {
    var cellID: String
    var concurrency: Int
    var sampleIndex: Int
    var warmup: Bool
    var requestID: String
    var planRevisionBefore: Int
    var planRevisionAfter: Int
    var stateRevisionBefore: Int
    var stateRevisionAfter: Int
    var sharedBatchPlanSequence: Int
    var sharedOccupancy: Int
    var overlapObserved: Bool
    var speculationUsed: Bool
}

private struct Qwen38PerformanceAttributionEnvelopeDigestBasis: Codable, Equatable, Sendable {
    var schemaVersion: Int
    var artifact: Qwen38MTPPerformanceScorecardArtifact
    var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    var promotionPolicyDigest: String
    var productionRouteReceiptDigest: String
    var claims: [Qwen38PerformanceAttributionClaim]
    var exploratoryBestStack: Qwen38PerformanceAttributionClaim?
}

public enum Qwen38PerformanceAttributionScorecardGate {
    public static let schemaVersion = 2
    public static let flagshipMeasurementClass = "qwen38-27b-flagship-live-v1"
    public static let flagshipMinimumRAMBytes: UInt64 = 256 * 1_024 * 1_024 * 1_024
    public static let droppedWarmupSamplesPerCell = 2
    public static let measuredSamplesPerCell = 4
    public static let requiredClaimKinds: [Qwen38PerformanceAttributionClaimKind] = [
        .scalarGDN,
        .exactMTP,
        .continuousBatchNoSpec,
        .prefixMatrix,
    ]

    public static func routeIdentity(
        kind: Qwen38PerformanceAttributionRouteKind,
        backendEvidenceKind: Qwen38PerformanceAttributionBackendEvidenceKind? = nil,
        backendEvidenceID: String? = nil,
        backendObservationDigest: String? = nil,
        backendReceiptDigest: String? = nil
    ) -> Qwen38PerformanceAttributionRouteIdentity {
        Qwen38PerformanceAttributionRouteIdentity(
            kind: kind,
            routeDigest: canonicalDigest(Qwen38PerformanceAttributionRouteDigestBasis(
                kind: kind,
                backendEvidenceKind: backendEvidenceKind,
                backendEvidenceID: backendEvidenceID,
                backendObservationDigest: backendObservationDigest,
                backendReceiptDigest: backendReceiptDigest)),
            backendEvidenceKind: backendEvidenceKind,
            backendEvidenceID: backendEvidenceID,
            backendObservationDigest: backendObservationDigest,
            backendReceiptDigest: backendReceiptDigest)
    }

    public static func frozenPromotionPolicy(
        evidenceID: String,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]
    ) -> Qwen38PerformanceAttributionFrozenPromotionPolicy {
        let ordered = claimAuthorities.sorted {
            claimOrder($0.claimKind) < claimOrder($1.claimKind)
        }
        return Qwen38PerformanceAttributionFrozenPromotionPolicy(
            schemaVersion: schemaVersion,
            evidenceID: evidenceID,
            sealedBeforeMeasurements: true,
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: ordered,
            digest: frozenPromotionPolicyDigest(
                evidenceID: evidenceID,
                sealedBeforeMeasurements: true,
                artifact: artifact,
                runIdentity: runIdentity,
                claimAuthorities: ordered))
    }

    public static func productionRouteReceipt(
        evidenceID: String,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        backendBuildIdentityDigest: String,
        observationDigest: String
    ) -> Qwen38PerformanceAttributionProductionRouteReceipt {
        let runDigest = canonicalDigest(runIdentity)
        return Qwen38PerformanceAttributionProductionRouteReceipt(
            schemaVersion: schemaVersion,
            evidenceID: evidenceID,
            artifact: artifact,
            runIdentityDigest: runDigest,
            backendBuildIdentityDigest: backendBuildIdentityDigest,
            observationDigest: observationDigest,
            digest: productionRouteReceiptDigest(
                evidenceID: evidenceID,
                artifact: artifact,
                runIdentityDigest: runDigest,
                backendBuildIdentityDigest: backendBuildIdentityDigest,
                observationDigest: observationDigest))
    }

    public static func cleanupEvidenceDigest(
        _ evidence: Qwen38PerformanceAttributionCleanupEvidence
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionCleanupEvidenceDigestBasis(
            baselineRSSBytes: evidence.baselineRSSBytes,
            finalRSSBytes: evidence.finalRSSBytes,
            baselineActiveMetalBytes: evidence.baselineActiveMetalBytes,
            finalActiveMetalBytes: evidence.finalActiveMetalBytes,
            baselineCachedMetalBytes: evidence.baselineCachedMetalBytes,
            finalCachedMetalBytes: evidence.finalCachedMetalBytes,
            baselineSwapBytes: evidence.baselineSwapBytes,
            finalSwapBytes: evidence.finalSwapBytes,
            baselinePageouts: evidence.baselinePageouts,
            finalPageouts: evidence.finalPageouts,
            pressureBefore: evidence.pressureBefore,
            pressureAfter: evidence.pressureAfter,
            thermalBefore: evidence.thermalBefore,
            thermalAfter: evidence.thermalAfter,
            cooldownSeconds: evidence.cooldownSeconds,
            idleSampleCount: evidence.idleSampleCount,
            boundedCooldownObserved: evidence.boundedCooldownObserved))
    }

    public static func continuousBackendObservationDigest(
        _ measurements: [Qwen38PerformanceAttributionPairMeasurement]
    ) -> String {
        let basis = measurements.flatMap { pair in
            pair.candidate.requests.map { request in
                let route = request.routeObservation
                return Qwen38PerformanceAttributionBackendRequestDigestBasis(
                    cellID: pair.cellID,
                    concurrency: pair.concurrency,
                    sampleIndex: pair.sampleIndex,
                    warmup: pair.warmup,
                    requestID: route.requestID,
                    planRevisionBefore: route.planRevisionBefore,
                    planRevisionAfter: route.planRevisionAfter,
                    stateRevisionBefore: route.stateRevisionBefore,
                    stateRevisionAfter: route.stateRevisionAfter,
                    sharedBatchPlanSequence: route.sharedBatchPlanSequence,
                    sharedOccupancy: route.sharedOccupancy,
                    overlapObserved: route.overlapObserved,
                    speculationUsed: route.speculationUsed)
            }
        }.sorted {
            ($0.cellID, $0.concurrency, $0.sampleIndex, $0.requestID)
                < ($1.cellID, $1.concurrency, $1.sampleIndex, $1.requestID)
        }
        return canonicalDigest(basis)
    }

    public static func absoluteAuthority(
        evidenceID: String,
        sealedBeforeMeasurements: Bool = true,
        bands: [Qwen38PerformanceAttributionAbsoluteBand]
    ) -> Qwen38PerformanceAttributionAbsoluteAuthority {
        Qwen38PerformanceAttributionAbsoluteAuthority(
            evidenceID: evidenceID,
            sealedBeforeMeasurements: sealedBeforeMeasurements,
            bands: bands,
            digest: absoluteAuthorityDigest(
                evidenceID: evidenceID,
                sealedBeforeMeasurements: sealedBeforeMeasurements,
                bands: bands))
    }

    public static func cleanupAuthority(
        evidenceID: String,
        minIdleSamples: Int,
        cooldownSeconds: Double,
        maxRSSDeltaBytes: UInt64,
        maxActiveMetalDeltaBytes: UInt64,
        maxCachedMetalDeltaBytes: UInt64,
        maxSwapDeltaBytes: UInt64,
        maxPageoutDelta: UInt64,
        allowedPressureStates: [String],
        allowedThermalStates: [String]
    ) -> Qwen38PerformanceAttributionCleanupAuthority {
        Qwen38PerformanceAttributionCleanupAuthority(
            evidenceID: evidenceID,
            digest: cleanupAuthorityDigest(
                evidenceID: evidenceID,
                minIdleSamples: minIdleSamples,
                cooldownSeconds: cooldownSeconds,
                maxRSSDeltaBytes: maxRSSDeltaBytes,
                maxActiveMetalDeltaBytes: maxActiveMetalDeltaBytes,
                maxCachedMetalDeltaBytes: maxCachedMetalDeltaBytes,
                maxSwapDeltaBytes: maxSwapDeltaBytes,
                maxPageoutDelta: maxPageoutDelta,
                allowedPressureStates: allowedPressureStates,
                allowedThermalStates: allowedThermalStates),
            minIdleSamples: minIdleSamples,
            cooldownSeconds: cooldownSeconds,
            maxRSSDeltaBytes: maxRSSDeltaBytes,
            maxActiveMetalDeltaBytes: maxActiveMetalDeltaBytes,
            maxCachedMetalDeltaBytes: maxCachedMetalDeltaBytes,
            maxSwapDeltaBytes: maxSwapDeltaBytes,
            maxPageoutDelta: maxPageoutDelta,
            allowedPressureStates: allowedPressureStates,
            allowedThermalStates: allowedThermalStates)
    }

    public static func envelopeDigest(
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        promotionPolicyDigest: String,
        productionRouteReceiptDigest: String,
        claims: [Qwen38PerformanceAttributionClaim],
        exploratoryBestStack: Qwen38PerformanceAttributionClaim?
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionEnvelopeDigestBasis(
            schemaVersion: schemaVersion,
            artifact: artifact,
            runIdentity: runIdentity,
            promotionPolicyDigest: promotionPolicyDigest,
            productionRouteReceiptDigest: productionRouteReceiptDigest,
            claims: claims,
            exploratoryBestStack: exploratoryBestStack))
    }

    /// The only public API that can return a promotion-qualified verdict. `context` must be
    /// assembled from trust inputs outside result collection; never derive either expected digest
    /// from the policy, receipt, or scorecard passed to this call.
    public static func validateForFlagshipPromotion(
        _ scorecard: Qwen38PerformanceAttributionScorecard,
        context: Qwen38FlagshipPromotionContext
    ) throws -> Qwen38PerformanceAttributionScorecardVerdict {
        guard scorecard.schemaVersion == schemaVersion else {
            throw Qwen38PerformanceAttributionScorecardGateError.schemaVersionMismatch(
                scorecard.schemaVersion)
        }
        guard scorecard.envelopeDigest == envelopeDigest(
            artifact: scorecard.artifact,
            runIdentity: scorecard.runIdentity,
            promotionPolicyDigest: scorecard.promotionPolicyDigest,
            productionRouteReceiptDigest: scorecard.productionRouteReceiptDigest,
            claims: scorecard.claims,
            exploratoryBestStack: scorecard.exploratoryBestStack)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidEnvelope
        }
        try validateClaimCardinality(scorecard.claims)
        try validatePromotionContext(context, scorecard: scorecard)
        try validateScorecardAuthority(scorecard)
        var requiredVerdicts: [Qwen38PerformanceAttributionClaimVerdict] = []
        for claim in scorecard.claims {
            let verdict = try validateClaim(claim)
            requiredVerdicts.append(verdict)
        }
        if let exploratory = scorecard.exploratoryBestStack {
            guard exploratory.kind == .bestStackExploratory else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidExploratoryClaim
            }
            let verdict = try validateClaim(exploratory)
            guard verdict.exploratory, !verdict.qualified else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidExploratoryClaim
            }
        }
        let expected = Qwen38PerformanceAttributionScorecardVerdict(
            qualified: requiredVerdicts.allSatisfy(\.qualified))
        guard scorecard.verdict == expected else {
            throw Qwen38PerformanceAttributionScorecardGateError.scorecardVerdictMismatch
        }
        guard expected.qualified else {
            throw Qwen38PerformanceAttributionScorecardGateError.unqualifiedScorecard
        }
        return expected
    }

    public static func computeMetrics(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Qwen38PerformanceAttributionClaimMetrics {
        try validateClaimEnvelope(claim)
        try validateMeasurementInputs(claim)
        let cells = try metricGroups(for: claim).map { group in
            let pairs = group.pairs.filter { !$0.warmup }
            let candidateRequests = pairs.flatMap(\.candidate.requests)
            let referenceRequests = pairs.flatMap(\.reference.requests)
            guard !pairs.isEmpty else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    claim.kind,
                    group.cellID)
            }
            let candidateTPS = Double(try tokenSum(candidateRequests))
                / pairs.reduce(0) { $0 + $1.candidate.wallSeconds }
            let referenceTPS = Double(try tokenSum(referenceRequests))
                / pairs.reduce(0) { $0 + $1.reference.wallSeconds }
            let cell = Qwen38PerformanceAttributionCellMetrics(
                cellID: group.cellID,
                concurrency: group.concurrency,
                aggregateThroughputRatio: candidateTPS / referenceTPS,
                e2ELatencyRatio: median(matchedRequestRatios(pairs) {
                    $0.reference.e2eSeconds / $0.candidate.e2eSeconds
                }),
                e2EP95LatencyRatio:
                    nearestRank(referenceRequests.map(\.e2eSeconds), percentile: 0.95)
                    / nearestRank(candidateRequests.map(\.e2eSeconds), percentile: 0.95),
                ttftLatencyRatio:
                    nearestRank(referenceRequests.map(\.ttftSeconds), percentile: 0.95)
                    / nearestRank(candidateRequests.map(\.ttftSeconds), percentile: 0.95),
                decodeTokensPerSecondRatio:
                    median(matchedRequestRatios(pairs) {
                        $0.candidate.decodeTokensPerSecond / $0.reference.decodeTokensPerSecond
                    }),
                candidateAggregateThroughput: candidateTPS,
                referenceAggregateThroughput: referenceTPS)
            guard [
                cell.aggregateThroughputRatio,
                cell.e2ELatencyRatio,
                cell.e2EP95LatencyRatio,
                cell.ttftLatencyRatio,
                cell.decodeTokensPerSecondRatio,
                cell.candidateAggregateThroughput,
                cell.referenceAggregateThroughput,
            ].allSatisfy({ $0.isFinite && $0 > 0 }) else {
                throw Qwen38PerformanceAttributionScorecardGateError.nonFiniteMetrics(
                    claim.kind)
            }
            return cell
        }
        return Qwen38PerformanceAttributionClaimMetrics(cells: cells)
    }

    /// Structural evaluation for assembly and tests. This is intentionally not public because it
    /// cannot authorize a flagship claim without the external promotion context.
    static func evaluateClaim(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Qwen38PerformanceAttributionClaimVerdict {
        let metrics = try computeMetrics(claim)
        try validateMeasurements(claim, metrics: metrics)
        let qualified: Bool
        switch claim.kind {
        case .scalarGDN:
            let halvesQualified = try balancedHalvesQualify(claim)
            qualified = metrics.cells.allSatisfy {
                $0.concurrency == 1
                    && $0.e2ELatencyRatio >= 1
                    && $0.decodeTokensPerSecondRatio >= 1
                    && $0.ttftLatencyRatio >= 1 / 1.05
            } && halvesQualified
        case .prefixMatrix:
            qualified = true
        case .exactMTP:
            let halvesQualified = try balancedHalvesQualify(claim)
            qualified = metrics.cells.allSatisfy {
                $0.concurrency == 1
                    && $0.e2ELatencyRatio >= 1
                    && $0.decodeTokensPerSecondRatio >= 1
                    && $0.ttftLatencyRatio >= 1 / 1.05
            } && halvesQualified
        case .continuousBatchNoSpec:
            let halvesQualified = try balancedHalvesQualify(claim)
            qualified = metrics.cells.allSatisfy {
                ($0.concurrency == 2 || $0.concurrency == 4)
                    && $0.aggregateThroughputRatio >= 1
                    && $0.e2EP95LatencyRatio >= 1 / 1.05
            } && halvesQualified
        case .bestStackExploratory:
            qualified = false
        }
        return Qwen38PerformanceAttributionClaimVerdict(
            qualified: qualified,
            exploratory: claim.kind == .bestStackExploratory)
    }

    public static func canonicalDigest<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "+inf",
            negativeInfinity: "-inf",
            nan: "nan")
        guard let data = try? encoder.encode(value) else { return "" }
        return sha256Hex(data)
    }

    private static func validateClaim(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Qwen38PerformanceAttributionClaimVerdict {
        let metrics = try computeMetrics(claim)
        guard claim.metrics == metrics else {
            throw Qwen38PerformanceAttributionScorecardGateError.metricsMismatch(claim.kind)
        }
        let verdict = try evaluateClaim(claim)
        guard claim.verdict == verdict else {
            throw Qwen38PerformanceAttributionScorecardGateError.verdictMismatch(claim.kind)
        }
        return verdict
    }

    private static func validateClaimCardinality(
        _ claims: [Qwen38PerformanceAttributionClaim]
    ) throws {
        var counts: [Qwen38PerformanceAttributionClaimKind: Int] = [:]
        for claim in claims {
            guard requiredClaimKinds.contains(claim.kind) else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidExploratoryClaim
            }
            counts[claim.kind, default: 0] += 1
        }
        for kind in requiredClaimKinds {
            switch counts[kind, default: 0] {
            case 0:
                throw Qwen38PerformanceAttributionScorecardGateError.missingClaim(kind)
            case 1:
                continue
            default:
                throw Qwen38PerformanceAttributionScorecardGateError.duplicateClaim(kind)
            }
        }
    }

    private static func validatePromotionContext(
        _ context: Qwen38FlagshipPromotionContext,
        scorecard: Qwen38PerformanceAttributionScorecard
    ) throws {
        let policy = context.frozenPolicy
        guard isLowerHex(context.expectedPolicyDigest, count: 64),
            policy.schemaVersion == schemaVersion,
            isLowerHex(policy.evidenceID, count: 64),
            policy.sealedBeforeMeasurements,
            policy.artifact == Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            policy.artifact == scorecard.artifact,
            policy.runIdentity == scorecard.runIdentity,
            policy.runIdentity.measurementClass == flagshipMeasurementClass,
            policy.runIdentity.hardwareRAMBytes >= flagshipMinimumRAMBytes,
            policy.claimAuthorities.map(\.claimKind) == requiredClaimKinds,
            policy.digest == context.expectedPolicyDigest,
            scorecard.promotionPolicyDigest == context.expectedPolicyDigest,
            policy.digest == frozenPromotionPolicyDigest(
                evidenceID: policy.evidenceID,
                sealedBeforeMeasurements: policy.sealedBeforeMeasurements,
                artifact: policy.artifact,
                runIdentity: policy.runIdentity,
                claimAuthorities: policy.claimAuthorities)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidPromotionPolicy
        }
        for binding in policy.claimAuthorities {
            guard let claim = scorecard.claims.first(where: { $0.kind == binding.claimKind }),
                claim.absoluteAuthority == binding.absoluteAuthority,
                claim.cleanupAuthority == binding.cleanupAuthority
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidPromotionPolicy
            }
        }

        let receipt = context.productionRouteReceipt
        guard isLowerHex(context.expectedReceiptDigest, count: 64),
            receipt.schemaVersion == schemaVersion,
            isLowerHex(receipt.evidenceID, count: 64),
            receipt.evidenceID != policy.evidenceID,
            receipt.artifact == scorecard.artifact,
            receipt.runIdentityDigest == canonicalDigest(scorecard.runIdentity),
            isLowerHex(receipt.backendBuildIdentityDigest, count: 64),
            isLowerHex(receipt.observationDigest, count: 64),
            receipt.digest == context.expectedReceiptDigest,
            scorecard.productionRouteReceiptDigest == context.expectedReceiptDigest,
            receipt.digest == productionRouteReceiptDigest(
                evidenceID: receipt.evidenceID,
                artifact: receipt.artifact,
                runIdentityDigest: receipt.runIdentityDigest,
                backendBuildIdentityDigest: receipt.backendBuildIdentityDigest,
                observationDigest: receipt.observationDigest),
            let continuous = scorecard.claims.first(where: {
                $0.kind == .continuousBatchNoSpec
            }),
            continuous.candidateRoute.backendEvidenceKind == .liveProductionRoute,
            continuous.candidateRoute.backendEvidenceID == receipt.evidenceID,
            continuous.candidateRoute.backendObservationDigest == receipt.observationDigest,
            continuous.candidateRoute.backendReceiptDigest == receipt.digest,
            receipt.observationDigest == continuousBackendObservationDigest(
                continuous.measurements)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidProductionRouteReceipt
        }
    }

    private static func validateScorecardAuthority(
        _ scorecard: Qwen38PerformanceAttributionScorecard
    ) throws {
        let run = scorecard.runIdentity
        guard scorecard.artifact == Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            !run.measurementClass.isEmpty,
            !run.hardwareChip.isEmpty,
            run.hardwareRAMBytes > 0,
            !run.hardwareOSBuild.isEmpty,
            isLowerHex(run.hostIdentityDigest, count: 64),
            isLowerHex(run.harnessGitSHA, count: 40),
            !run.candidateMLXSwiftVersion.isEmpty,
            !run.modelLabel.isEmpty,
            run.modelConfigHash == scorecard.artifact.targetConfigSHA256,
            run.modelCheckpointManifestHash == scorecard.artifact.targetTensorManifestSHA256,
            run.modelQuant == ModelQuantInfo(
                bits: scorecard.artifact.targetQuantizationBits,
                groupSize: scorecard.artifact.targetQuantizationGroupSize),
            !run.corpusID.isEmpty,
            isLowerHex(run.corpusContentHash, count: 64)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidEnvelope
        }
        let allClaims = scorecard.claims
            + (scorecard.exploratoryBestStack.map { [$0] } ?? [])
        guard allClaims.allSatisfy({ claim in
            claim.candidate.artifact == scorecard.artifact
                && claim.reference.artifact == scorecard.artifact
                && claim.candidate.label == run.modelLabel
                && claim.reference.label == run.modelLabel
                && claim.candidate.sourceDigest == Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
                && claim.reference.sourceDigest == Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
        }) else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidEnvelope
        }
    }

    private static func validateClaimEnvelope(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        guard !claim.scheduledCells.isEmpty,
            !claim.measurements.isEmpty,
            claim.scheduledCells.map(\.id).count == Set(claim.scheduledCells.map(\.id)).count,
            claim.measurements.allSatisfy({ measurement in
                claim.scheduledCells.contains { $0.id == measurement.cellID }
            })
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(claim.kind, "schedule")
        }
        try validateIdentity(claim)
        try validateAbsoluteAuthority(claim)
        try validateCleanupAuthority(claim.cleanupAuthority, kind: claim.kind)
        if claim.kind == .prefixMatrix {
            try validatePrefixMatrixSchedule(claim)
        }
    }

    private static func validateIdentity(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        guard claim.candidate.label == claim.reference.label,
            !claim.candidate.label.isEmpty,
            claim.candidate.artifact == claim.reference.artifact,
            isLowerHex(claim.candidate.executionDigest, count: 64),
            isLowerHex(claim.reference.executionDigest, count: 64),
            isLowerHex(claim.candidate.sourceDigest, count: 64),
            claim.candidate.sourceDigest == claim.reference.sourceDigest,
            let candidateMode = claim.candidate.gdnMode,
            let referenceMode = claim.reference.gdnMode,
            let candidateLaunch = claim.candidate.launchBinding,
            let referenceLaunch = claim.reference.launchBinding,
            isValidLaunchBinding(
                candidateLaunch,
                expectedMode: candidateMode,
                sourceDigest: claim.candidate.sourceDigest),
            isValidLaunchBinding(
                referenceLaunch,
                expectedMode: referenceMode,
                sourceDigest: claim.reference.sourceDigest),
            candidateLaunch.processIsolationEvidenceID
                != referenceLaunch.processIsolationEvidenceID,
            isValidRouteIdentity(claim.candidateRoute),
            isValidRouteIdentity(claim.referenceRoute)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidClaimIdentity(claim.kind)
        }
        switch claim.kind {
        case .scalarGDN:
            guard claim.candidate.executionMode == .scalar,
                claim.reference.executionMode == .scalar,
                claim.candidate.executionDigest == claim.reference.executionDigest,
                candidateMode == .gdnOn,
                referenceMode == .gdnOff,
                claim.candidateRoute.kind == .scalar,
                claim.referenceRoute.kind == .scalar,
                claim.candidateRoute == claim.referenceRoute,
                hasNoBackendEvidence(claim.candidateRoute),
                hasNoBackendEvidence(claim.referenceRoute)
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidClaimIdentity(claim.kind)
            }
        case .exactMTP:
            guard claim.candidate.executionMode == .exactMTP,
                claim.reference.executionMode == .scalar,
                claim.candidate.executionDigest != claim.reference.executionDigest,
                candidateMode == .gdnOn,
                referenceMode == .gdnOn,
                claim.candidateRoute.kind == .exactMTP,
                claim.referenceRoute.kind == .scalar,
                hasNoBackendEvidence(claim.candidateRoute),
                hasNoBackendEvidence(claim.referenceRoute)
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidClaimIdentity(claim.kind)
            }
        case .continuousBatchNoSpec:
            guard claim.candidate.executionMode == .scalar,
                claim.reference.executionMode == .scalar,
                claim.candidate.executionDigest != claim.reference.executionDigest,
                candidateMode == .gdnOn,
                referenceMode == .gdnOn,
                claim.candidateRoute.kind == .continuousBatchNoSpec,
                claim.referenceRoute.kind == .scalar,
                claim.candidateRoute.backendEvidenceKind == .liveProductionRoute,
                isLowerHex(claim.candidateRoute.backendEvidenceID ?? "", count: 64),
                isLowerHex(claim.candidateRoute.backendObservationDigest ?? "", count: 64),
                isLowerHex(claim.candidateRoute.backendReceiptDigest ?? "", count: 64),
                claim.candidateRoute.backendObservationDigest
                    == continuousBackendObservationDigest(claim.measurements),
                hasNoBackendEvidence(claim.referenceRoute)
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidClaimIdentity(claim.kind)
            }
        case .prefixMatrix:
            guard claim.candidate.executionMode == claim.reference.executionMode,
                claim.candidate.executionDigest == claim.reference.executionDigest,
                candidateMode == .gdnOn,
                referenceMode == .gdnOn,
                claim.candidateRoute.kind == .prefixMatrix,
                claim.referenceRoute.kind == .prefixMatrix,
                claim.candidateRoute == claim.referenceRoute,
                hasNoBackendEvidence(claim.candidateRoute),
                hasNoBackendEvidence(claim.referenceRoute)
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidClaimIdentity(claim.kind)
            }
        case .bestStackExploratory:
            guard claim.candidateRoute.kind == .exploratoryBestStack,
                hasNoBackendEvidence(claim.candidateRoute),
                hasNoBackendEvidence(claim.referenceRoute)
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidExploratoryClaim
            }
        }
    }

    private static func isValidLaunchBinding(
        _ binding: Qwen38MTPPerformanceScorecardLaunchBinding,
        expectedMode: Qwen38MTPPerformanceScorecardGDNMode,
        sourceDigest: String
    ) -> Bool {
        let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            expectedMode == .gdnOn ? .enabled : .disabled
        return binding.mode == expectedMode
            && binding.sourceDigest == sourceDigest
            && binding.observedEnv == observedEnv
            && isLowerHex(binding.processIsolationEvidenceID, count: 64)
            && isLowerHex(binding.launchDigest, count: 64)
            && binding.launchDigest == Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: binding.mode,
                sourceDigest: binding.sourceDigest,
                observedEnv: binding.observedEnv,
                processIsolationEvidenceID: binding.processIsolationEvidenceID)
    }

    private static func isValidRouteIdentity(
        _ route: Qwen38PerformanceAttributionRouteIdentity
    ) -> Bool {
        isLowerHex(route.routeDigest, count: 64)
            && route.routeDigest == routeIdentity(
                kind: route.kind,
                backendEvidenceKind: route.backendEvidenceKind,
                backendEvidenceID: route.backendEvidenceID,
                backendObservationDigest: route.backendObservationDigest,
                backendReceiptDigest: route.backendReceiptDigest).routeDigest
    }

    private static func hasNoBackendEvidence(
        _ route: Qwen38PerformanceAttributionRouteIdentity
    ) -> Bool {
        route.backendEvidenceKind == nil
            && route.backendEvidenceID == nil
            && route.backendObservationDigest == nil
            && route.backendReceiptDigest == nil
    }

    private static func validateMeasurementInputs(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        for pair in claim.measurements {
            guard pair.concurrency > 0,
                pair.sampleIndex >= 0,
                pair.candidate.wallSeconds.isFinite,
                pair.reference.wallSeconds.isFinite,
                pair.candidate.wallSeconds > 0,
                pair.reference.wallSeconds > 0,
                pair.candidate.proposalCount >= 0,
                pair.candidate.acceptedCount >= 0,
                pair.reference.proposalCount >= 0,
                pair.reference.acceptedCount >= 0
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    claim.kind,
                    pair.cellID)
            }
            for request in pair.candidate.requests + pair.reference.requests {
                guard request.promptTokenCount > 0,
                    request.prefillSeconds.isFinite,
                    request.prefillSeconds > 0,
                    request.ttftSeconds.isFinite,
                    request.ttftSeconds > 0,
                    request.decodeSeconds.isFinite,
                    request.decodeSeconds > 0,
                    request.e2eSeconds.isFinite,
                    request.e2eSeconds > 0,
                    request.decodeTokenCount > 0,
                    request.decodeTokenCount == request.outputTokenIDs.count,
                    request.outputTokenIDs.allSatisfy({
                        $0 >= 0 && Int32(exactly: $0) != nil
                    }),
                    isLowerHex(request.outputBytesDigest, count: 64),
                    isLowerHex(request.cacheDigest, count: 64)
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                        claim.kind,
                        pair.cellID)
                }
                guard request.decodeTokensPerSecond.isFinite,
                    request.decodeTokensPerSecond > 0
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.nonFiniteMetrics(
                        claim.kind)
                }
            }
        }
    }

    private static func validateMeasurements(
        _ claim: Qwen38PerformanceAttributionClaim,
        metrics: Qwen38PerformanceAttributionClaimMetrics
    ) throws {
        try validateSampleCoverage(claim)
        for pair in claim.measurements {
            guard pair.candidate.identity == claim.candidate,
                pair.reference.identity == claim.reference,
                pair.candidate.route == claim.candidateRoute,
                pair.reference.route == claim.referenceRoute,
                pair.candidate.wallSeconds.isFinite,
                pair.reference.wallSeconds.isFinite,
                pair.candidate.wallSeconds > 0,
                pair.reference.wallSeconds > 0,
                pair.candidate.requests.count == pair.concurrency,
                pair.reference.requests.count == pair.concurrency
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    claim.kind,
                    pair.cellID)
            }
            try validateRouteEvidence(claim, pair: pair)
            try validateCleanup(pair.candidate.cleanup, authority: claim.cleanupAuthority, kind: claim.kind)
            try validateCleanup(pair.reference.cleanup, authority: claim.cleanupAuthority, kind: claim.kind)
            try validatePairOutputParity(claim, pair: pair)
            try validatePairCellsAndWarmEvidence(claim, pair: pair)
            try validateAbsoluteBands(claim, pair: pair, metrics: metrics)
            try validateDraftActivity(claim, pair: pair)
        }
    }

    private struct MetricGroup {
        var cellID: String
        var concurrency: Int
        var pairs: [Qwen38PerformanceAttributionPairMeasurement]
    }

    private static func metricGroups(
        for claim: Qwen38PerformanceAttributionClaim
    ) throws -> [MetricGroup] {
        try validateSampleCoverage(claim)
        return requiredCellConcurrencyKeys(for: claim).map { key in
            MetricGroup(
                cellID: key.cellID,
                concurrency: key.concurrency,
                pairs: claim.measurements.filter {
                    $0.cellID == key.cellID && $0.concurrency == key.concurrency
                })
        }
    }

    private static func validateSampleCoverage(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        for key in requiredCellConcurrencyKeys(for: claim) {
            let pairs = claim.measurements
                .filter { $0.cellID == key.cellID && $0.concurrency == key.concurrency }
                .sorted { $0.sampleIndex < $1.sampleIndex }
            let expectedTotal = droppedWarmupSamplesPerCell + measuredSamplesPerCell
            guard pairs.count == expectedTotal,
                pairs.map(\.sampleIndex) == Array(0..<expectedTotal),
                pairs.filter(\.warmup).count == droppedWarmupSamplesPerCell,
                pairs.filter({ !$0.warmup }).count == measuredSamplesPerCell,
                pairs.allSatisfy({
                    $0.warmup == ($0.sampleIndex < droppedWarmupSamplesPerCell)
                })
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    claim.kind,
                    key.cellID)
            }
            let measured = pairs.filter { !$0.warmup }
            let firstHalf = measured.prefix(measured.count / 2)
            let secondHalf = measured.suffix(measured.count / 2)
            guard firstHalf.filter({ $0.order == .candidateThenReference }).count
                    == firstHalf.count / 2,
                firstHalf.filter({ $0.order == .referenceThenCandidate }).count
                    == firstHalf.count / 2,
                secondHalf.filter({ $0.order == .candidateThenReference }).count
                    == secondHalf.count / 2,
                secondHalf.filter({ $0.order == .referenceThenCandidate }).count
                    == secondHalf.count / 2
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    claim.kind,
                    key.cellID)
            }
        }
        let required = Set(requiredCellConcurrencyKeys(for: claim).map { "\($0.cellID):\($0.concurrency)" })
        let observed = Set(claim.measurements.map { "\($0.cellID):\($0.concurrency)" })
        guard observed == required else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                claim.kind,
                "coverage")
        }
    }

    private static func requiredCellConcurrencyKeys(
        for claim: Qwen38PerformanceAttributionClaim
    ) -> [(cellID: String, concurrency: Int)] {
        claim.scheduledCells.flatMap { cell in
            requiredConcurrencies(for: claim.kind).map { (cell.id, $0) }
        }
    }

    private static func requiredConcurrencies(
        for kind: Qwen38PerformanceAttributionClaimKind
    ) -> [Int] {
        kind == .continuousBatchNoSpec ? [2, 4] : [1]
    }

    private struct MatchedRequestSample {
        var candidate: Qwen38PerformanceAttributionRequestMeasurement
        var reference: Qwen38PerformanceAttributionRequestMeasurement
    }

    private static func matchedRequestRatios(
        _ pairs: [Qwen38PerformanceAttributionPairMeasurement],
        _ ratio: (MatchedRequestSample) -> Double
    ) -> [Double] {
        pairs.flatMap { pair in
            zip(pair.candidate.requests, pair.reference.requests).map { candidate, reference in
                ratio(MatchedRequestSample(candidate: candidate, reference: reference))
            }
        }
    }

    private static func balancedHalvesQualify(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws -> Bool {
        for group in try metricGroups(for: claim) {
            let measured = group.pairs
                .filter { !$0.warmup }
                .sorted { $0.sampleIndex < $1.sampleIndex }
            let midpoint = measured.count / 2
            for half in [Array(measured.prefix(midpoint)), Array(measured.suffix(midpoint))] {
                guard !half.isEmpty else {
                    return false
                }
                switch claim.kind {
                case .scalarGDN, .exactMTP:
                    let e2e = median(matchedRequestRatios(half) {
                        $0.reference.e2eSeconds / $0.candidate.e2eSeconds
                    })
                    let decode = median(matchedRequestRatios(half) {
                        $0.candidate.decodeTokensPerSecond / $0.reference.decodeTokensPerSecond
                    })
                    guard e2e >= 0.98, decode >= 0.98 else {
                        return false
                    }
                case .continuousBatchNoSpec:
                    let candidateTPS = Double(try tokenSum(half.flatMap(\.candidate.requests)))
                        / half.reduce(0) { $0 + $1.candidate.wallSeconds }
                    let referenceTPS = Double(try tokenSum(half.flatMap(\.reference.requests)))
                        / half.reduce(0) { $0 + $1.reference.wallSeconds }
                    let candidateRequests = half.flatMap(\.candidate.requests)
                    let referenceRequests = half.flatMap(\.reference.requests)
                    let e2eP95 = nearestRank(
                        referenceRequests.map(\.e2eSeconds),
                        percentile: 0.95)
                        / nearestRank(
                            candidateRequests.map(\.e2eSeconds),
                            percentile: 0.95)
                    guard candidateTPS / referenceTPS >= 0.98, e2eP95 >= 0.98 else {
                        return false
                    }
                case .prefixMatrix, .bestStackExploratory:
                    continue
                }
            }
        }
        return true
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else {
            return .nan
        }
        let sorted = values.sorted()
        let midpoint = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[midpoint - 1] + sorted[midpoint]) / 2
        }
        return sorted[midpoint]
    }

    private static func nearestRank(_ values: [Double], percentile: Double) -> Double {
        guard !values.isEmpty else {
            return .nan
        }
        let sorted = values.sorted()
        let rank = max(1, Int(ceil(percentile * Double(sorted.count))))
        return sorted[min(rank - 1, sorted.count - 1)]
    }

    private static func validateRouteEvidence(
        _ claim: Qwen38PerformanceAttributionClaim,
        pair: Qwen38PerformanceAttributionPairMeasurement
    ) throws {
        switch claim.kind {
        case .exactMTP:
            guard pair.concurrency == 1 else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(claim.kind)
            }
        case .continuousBatchNoSpec:
            guard pair.concurrency == 2 || pair.concurrency == 4 else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(claim.kind)
            }
        default:
            break
        }

        try validateRequestRoutes(
            pair.candidate.requests,
            engineRoute: pair.candidate.route,
            concurrency: pair.concurrency,
            kind: claim.kind,
            candidate: true)
        try validateRequestRoutes(
            pair.reference.requests,
            engineRoute: pair.reference.route,
            concurrency: pair.concurrency,
            kind: claim.kind,
            candidate: false)
    }

    private static func validateRequestRoutes(
        _ requests: [Qwen38PerformanceAttributionRequestMeasurement],
        engineRoute: Qwen38PerformanceAttributionRouteIdentity,
        concurrency: Int,
        kind: Qwen38PerformanceAttributionClaimKind,
        candidate: Bool
    ) throws {
        var requestIDs = Set<String>()
        if kind == .continuousBatchNoSpec && candidate {
            let sharedPlans = Set(requests.map(\.routeObservation.sharedBatchPlanSequence))
            guard sharedPlans.count == 1,
                let sharedPlan = sharedPlans.first,
                sharedPlan > 0
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(kind)
            }
        }
        for request in requests {
            let route = request.routeObservation
            guard route.routeKind == engineRoute.kind,
                !route.requestID.isEmpty,
                requestIDs.insert(route.requestID).inserted
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(kind)
            }
            if kind == .continuousBatchNoSpec && candidate {
                guard route.backendObservationDigest == engineRoute.backendObservationDigest,
                    isLowerHex(route.backendObservationDigest ?? "", count: 64),
                    route.planRevisionBefore >= 0,
                    route.planRevisionAfter > route.planRevisionBefore,
                    route.sharedBatchPlanSequence > route.planRevisionBefore,
                    route.sharedBatchPlanSequence <= route.planRevisionAfter,
                    route.stateRevisionBefore >= 0,
                    route.stateRevisionAfter > route.stateRevisionBefore,
                    route.sharedOccupancy == concurrency,
                    route.overlapObserved,
                    !route.speculationUsed
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(kind)
                }
            } else {
                guard route.backendObservationDigest == nil,
                    route.planRevisionBefore == 0,
                    route.planRevisionAfter == 0,
                    route.stateRevisionBefore == 0,
                    route.stateRevisionAfter == 0,
                    route.sharedBatchPlanSequence == 0,
                    route.sharedOccupancy == 0,
                    !route.overlapObserved,
                    !route.speculationUsed
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(kind)
                }
            }
        }
    }

    private static func validatePairOutputParity(
        _ claim: Qwen38PerformanceAttributionClaim,
        pair: Qwen38PerformanceAttributionPairMeasurement
    ) throws {
        for (candidate, reference) in zip(pair.candidate.requests, pair.reference.requests) {
            guard candidate.outputTokenIDs == reference.outputTokenIDs,
                candidate.decodeTokenCount == reference.decodeTokenCount,
                candidate.outputBytesDigest == reference.outputBytesDigest,
                candidate.cacheDigest == reference.cacheDigest
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.outputParityMismatch(claim.kind)
            }
        }
    }

    private static func validatePairCellsAndWarmEvidence(
        _ claim: Qwen38PerformanceAttributionClaim,
        pair: Qwen38PerformanceAttributionPairMeasurement
    ) throws {
        guard let cell = claim.scheduledCells.first(where: { $0.id == pair.cellID }),
            cell.renderedPromptTokenCount == cell.contextTokens.rawValue,
            cell.promptTokenIDs.count == cell.renderedPromptTokenCount,
            cell.promptTokenIDs.allSatisfy({ $0 >= 0 && Int32(exactly: $0) != nil }),
            canonicalDigest(cell.promptTokenIDs) == cell.promptTokenDigest,
            isLowerHex(cell.promptTokenDigest, count: 64)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(claim.kind, pair.cellID)
        }
        for (candidate, reference) in zip(pair.candidate.requests, pair.reference.requests) {
            guard candidate.cellID == pair.cellID,
                reference.cellID == pair.cellID,
                candidate.promptTokenCount == cell.renderedPromptTokenCount,
                reference.promptTokenCount == cell.renderedPromptTokenCount,
                candidate.promptTokenDigest == cell.promptTokenDigest,
                reference.promptTokenDigest == cell.promptTokenDigest
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(claim.kind, pair.cellID)
            }
            switch cell.prefixKind {
            case .cold:
                guard candidate.warmPrefixEvidence == nil,
                    reference.warmPrefixEvidence == nil
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.invalidWarmPrefixEvidence(
                        claim.kind,
                        pair.cellID)
                }
            case .exactWarmPrefix:
                guard let candidateWarm = candidate.warmPrefixEvidence,
                    candidateWarm == reference.warmPrefixEvidence,
                    candidateWarm.tokenCount == cell.renderedPromptTokenCount,
                    candidateWarm.tokenIDs.count == candidateWarm.tokenCount,
                    candidateWarm.tokenIDs == cell.promptTokenIDs,
                    canonicalDigest(candidateWarm.tokenIDs) == cell.promptTokenDigest,
                    candidateWarm.rebuildTokenCount == candidateWarm.tokenCount,
                    candidateWarm.rebuildTokenIDs == candidateWarm.tokenIDs,
                    candidateWarm.restoredCanonicalDigest
                        == candidateWarm.snapshotCanonicalDigest,
                    candidateWarm.rebuildCanonicalDigest == candidateWarm.snapshotCanonicalDigest,
                    isLowerHex(candidateWarm.snapshotCanonicalDigest, count: 64),
                    isLowerHex(candidateWarm.restoredCanonicalDigest, count: 64),
                    isLowerHex(candidateWarm.rebuildCanonicalDigest, count: 64)
                else {
                    throw Qwen38PerformanceAttributionScorecardGateError.invalidWarmPrefixEvidence(
                        claim.kind,
                        pair.cellID)
                }
            }
        }
    }

    private static func validateAbsoluteBands(
        _ claim: Qwen38PerformanceAttributionClaim,
        pair: Qwen38PerformanceAttributionPairMeasurement,
        metrics: Qwen38PerformanceAttributionClaimMetrics
    ) throws {
        guard let cell = claim.scheduledCells.first(where: { $0.id == pair.cellID }),
            let band = claim.absoluteAuthority.bands.first(where: {
                bandKey($0) == bandKey(
                    kind: claim.kind,
                    concurrency: pair.concurrency,
                    contextTokens: cell.contextTokens,
                    prefixKind: cell.prefixKind)
            }),
            let cellMetrics = metrics.cells.first(where: {
                $0.cellID == pair.cellID && $0.concurrency == pair.concurrency
            })
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(claim.kind)
        }
        for request in pair.candidate.requests {
            guard request.prefillSeconds <= band.maxPrefillSeconds,
                request.ttftSeconds <= band.maxTTFTSeconds,
                request.decodeTokensPerSecond >= band.minDecodeTokensPerSecond
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(
                    claim.kind)
            }
        }
        guard cellMetrics.candidateAggregateThroughput
            >= band.minAggregateThroughputTokensPerSecond
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(claim.kind)
        }
    }

    private static func validateDraftActivity(
        _ claim: Qwen38PerformanceAttributionClaim,
        pair: Qwen38PerformanceAttributionPairMeasurement
    ) throws {
        if claim.kind == .exactMTP {
            guard pair.candidate.proposalCount > 0,
                pair.candidate.acceptedCount > 0,
                pair.candidate.acceptedCount <= pair.candidate.proposalCount,
                pair.reference.proposalCount == 0,
                pair.reference.acceptedCount == 0
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(claim.kind)
            }
        } else {
            guard pair.candidate.proposalCount == 0,
                pair.candidate.acceptedCount == 0,
                pair.reference.proposalCount == 0,
                pair.reference.acceptedCount == 0
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidRouteEvidence(claim.kind)
            }
        }
    }

    private static func validatePrefixMatrixSchedule(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        let expected = Set(
            Qwen38MTPPerformanceScorecardBenchmarkContextTokens.allMatrixContexts.flatMap { context in
                Qwen38MTPPerformanceScorecardPrefixKind.allMatrixPrefixes.map {
                    "\(context.rawValue):\($0.rawValue)"
                }
            })
        let observed = Set(claim.scheduledCells.map {
            "\($0.contextTokens.rawValue):\($0.prefixKind.rawValue)"
        })
        guard observed == expected,
            claim.scheduledCells.count == expected.count
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                claim.kind,
                "prefix-matrix")
        }
    }

    private static func validateAbsoluteAuthority(
        _ claim: Qwen38PerformanceAttributionClaim
    ) throws {
        let authority = claim.absoluteAuthority
        guard isLowerHex(authority.evidenceID, count: 64),
            isLowerHex(authority.digest, count: 64),
            authority.sealedBeforeMeasurements,
            authority.digest == absoluteAuthorityDigest(
                evidenceID: authority.evidenceID,
                sealedBeforeMeasurements: authority.sealedBeforeMeasurements,
                bands: authority.bands)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(claim.kind)
        }
        var keys = Set<String>()
        for band in authority.bands {
            guard band.claimKind == claim.kind,
                band.concurrency > 0,
                band.maxPrefillSeconds.isFinite,
                band.maxPrefillSeconds > 0,
                band.maxTTFTSeconds.isFinite,
                band.maxTTFTSeconds > 0,
                band.minDecodeTokensPerSecond.isFinite,
                band.minDecodeTokensPerSecond > 0,
                band.minAggregateThroughputTokensPerSecond.isFinite,
                band.minAggregateThroughputTokensPerSecond > 0,
                keys.insert(bandKey(band)).inserted
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(
                    claim.kind)
            }
        }
        let expectedKeys = Set(claim.scheduledCells.flatMap { cell in
            requiredConcurrencies(for: claim.kind).map {
                bandKey(
                    kind: claim.kind,
                    concurrency: $0,
                    contextTokens: cell.contextTokens,
                    prefixKind: cell.prefixKind)
            }
        })
        guard keys == expectedKeys else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(
                claim.kind)
        }
        for pair in claim.measurements {
            guard let cell = claim.scheduledCells.first(where: { $0.id == pair.cellID }),
                keys.contains(bandKey(
                    kind: claim.kind,
                    concurrency: pair.concurrency,
                    contextTokens: cell.contextTokens,
                    prefixKind: cell.prefixKind))
            else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidAbsoluteAuthority(
                    claim.kind)
            }
        }
    }

    private static func validateCleanupAuthority(
        _ authority: Qwen38PerformanceAttributionCleanupAuthority,
        kind: Qwen38PerformanceAttributionClaimKind
    ) throws {
        guard isLowerHex(authority.evidenceID, count: 64),
            isLowerHex(authority.digest, count: 64),
            authority.minIdleSamples > 0,
            authority.cooldownSeconds.isFinite,
            authority.cooldownSeconds > 0,
            !authority.allowedPressureStates.isEmpty,
            !authority.allowedThermalStates.isEmpty,
            authority.digest == cleanupAuthorityDigest(
                evidenceID: authority.evidenceID,
                minIdleSamples: authority.minIdleSamples,
                cooldownSeconds: authority.cooldownSeconds,
                maxRSSDeltaBytes: authority.maxRSSDeltaBytes,
                maxActiveMetalDeltaBytes: authority.maxActiveMetalDeltaBytes,
                maxCachedMetalDeltaBytes: authority.maxCachedMetalDeltaBytes,
                maxSwapDeltaBytes: authority.maxSwapDeltaBytes,
                maxPageoutDelta: authority.maxPageoutDelta,
                allowedPressureStates: authority.allowedPressureStates,
                allowedThermalStates: authority.allowedThermalStates)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCleanup(kind)
        }
    }

    private static func validateCleanup(
        _ evidence: Qwen38PerformanceAttributionCleanupEvidence,
        authority: Qwen38PerformanceAttributionCleanupAuthority,
        kind: Qwen38PerformanceAttributionClaimKind
    ) throws {
        guard isLowerHex(evidence.evidenceDigest, count: 64),
            evidence.evidenceDigest == cleanupEvidenceDigest(evidence),
            evidence.boundedCooldownObserved,
            evidence.cooldownSeconds.isFinite,
            evidence.cooldownSeconds >= authority.cooldownSeconds,
            evidence.idleSampleCount >= authority.minIdleSamples,
            evidence.finalPageouts >= evidence.baselinePageouts,
            gaugeDelta(evidence.finalRSSBytes, evidence.baselineRSSBytes) <= authority.maxRSSDeltaBytes,
            gaugeDelta(evidence.finalActiveMetalBytes, evidence.baselineActiveMetalBytes)
                <= authority.maxActiveMetalDeltaBytes,
            gaugeDelta(evidence.finalCachedMetalBytes, evidence.baselineCachedMetalBytes)
                <= authority.maxCachedMetalDeltaBytes,
            gaugeDelta(evidence.finalSwapBytes, evidence.baselineSwapBytes)
                <= authority.maxSwapDeltaBytes,
            evidence.finalPageouts - evidence.baselinePageouts <= authority.maxPageoutDelta,
            authority.allowedPressureStates.contains(evidence.pressureBefore),
            authority.allowedPressureStates.contains(evidence.pressureAfter),
            authority.allowedThermalStates.contains(evidence.thermalBefore),
            authority.allowedThermalStates.contains(evidence.thermalAfter)
        else {
            throw Qwen38PerformanceAttributionScorecardGateError.invalidCleanup(kind)
        }
    }

    private static func tokenSum(
        _ requests: [Qwen38PerformanceAttributionRequestMeasurement]
    ) throws -> Int {
        var total = 0
        for request in requests {
            let result = total.addingReportingOverflow(request.decodeTokenCount)
            guard !result.overflow else {
                throw Qwen38PerformanceAttributionScorecardGateError.invalidCell(
                    .prefixMatrix,
                    request.cellID)
            }
            total = result.partialValue
        }
        return total
    }

    private static func bandKey(_ band: Qwen38PerformanceAttributionAbsoluteBand) -> String {
        bandKey(
            kind: band.claimKind,
            concurrency: band.concurrency,
            contextTokens: band.contextTokens,
            prefixKind: band.prefixKind)
    }

    private static func bandKey(
        kind: Qwen38PerformanceAttributionClaimKind,
        concurrency: Int,
        contextTokens: Qwen38MTPPerformanceScorecardBenchmarkContextTokens,
        prefixKind: Qwen38MTPPerformanceScorecardPrefixKind
    ) -> String {
        "\(kind.rawValue):\(concurrency):\(contextTokens.rawValue):\(prefixKind.rawValue)"
    }

    private static func absoluteAuthorityDigest(
        evidenceID: String,
        sealedBeforeMeasurements: Bool,
        bands: [Qwen38PerformanceAttributionAbsoluteBand]
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionAbsoluteDigestBasis(
            evidenceID: evidenceID,
            sealedBeforeMeasurements: sealedBeforeMeasurements,
            bands: bands))
    }

    private static func cleanupAuthorityDigest(
        evidenceID: String,
        minIdleSamples: Int,
        cooldownSeconds: Double,
        maxRSSDeltaBytes: UInt64,
        maxActiveMetalDeltaBytes: UInt64,
        maxCachedMetalDeltaBytes: UInt64,
        maxSwapDeltaBytes: UInt64,
        maxPageoutDelta: UInt64,
        allowedPressureStates: [String],
        allowedThermalStates: [String]
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionCleanupDigestBasis(
            evidenceID: evidenceID,
            minIdleSamples: minIdleSamples,
            cooldownSeconds: cooldownSeconds,
            maxRSSDeltaBytes: maxRSSDeltaBytes,
            maxActiveMetalDeltaBytes: maxActiveMetalDeltaBytes,
            maxCachedMetalDeltaBytes: maxCachedMetalDeltaBytes,
            maxSwapDeltaBytes: maxSwapDeltaBytes,
            maxPageoutDelta: maxPageoutDelta,
            allowedPressureStates: allowedPressureStates,
            allowedThermalStates: allowedThermalStates))
    }

    private static func frozenPromotionPolicyDigest(
        evidenceID: String,
        sealedBeforeMeasurements: Bool,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionFrozenPromotionPolicyDigestBasis(
            schemaVersion: schemaVersion,
            evidenceID: evidenceID,
            sealedBeforeMeasurements: sealedBeforeMeasurements,
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: claimAuthorities))
    }

    private static func productionRouteReceiptDigest(
        evidenceID: String,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        runIdentityDigest: String,
        backendBuildIdentityDigest: String,
        observationDigest: String
    ) -> String {
        canonicalDigest(Qwen38PerformanceAttributionProductionRouteReceiptDigestBasis(
            schemaVersion: schemaVersion,
            evidenceID: evidenceID,
            artifact: artifact,
            runIdentityDigest: runIdentityDigest,
            backendBuildIdentityDigest: backendBuildIdentityDigest,
            observationDigest: observationDigest))
    }

    private static func claimOrder(
        _ kind: Qwen38PerformanceAttributionClaimKind
    ) -> Int {
        requiredClaimKinds.firstIndex(of: kind) ?? Int.max
    }

    private static func gaugeDelta(_ final: UInt64, _ baseline: UInt64) -> UInt64 {
        final > baseline ? final - baseline : 0
    }

    private static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private extension Qwen38MTPPerformanceScorecardBenchmarkContextTokens {
    static let allMatrixContexts: [Qwen38MTPPerformanceScorecardBenchmarkContextTokens] = [
        .tokens4096,
        .tokens16384,
        .tokens32768,
    ]
}

private extension Qwen38MTPPerformanceScorecardPrefixKind {
    static let allMatrixPrefixes: [Qwen38MTPPerformanceScorecardPrefixKind] = [
        .cold,
        .exactWarmPrefix,
    ]
}
