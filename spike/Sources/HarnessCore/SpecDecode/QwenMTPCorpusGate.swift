import Foundation

extension MTPStreamExactness.Result: Codable {
    private enum CodingKeys: String, CodingKey {
        case exact
        case comparedTokens
        case firstDivergenceIndex
        case lengthMatched
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            exact: try container.decode(Bool.self, forKey: .exact),
            comparedTokens: try container.decode(Int.self, forKey: .comparedTokens),
            firstDivergenceIndex: try container.decodeIfPresent(Int.self, forKey: .firstDivergenceIndex),
            lengthMatched: try container.decode(Bool.self, forKey: .lengthMatched))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(exact, forKey: .exact)
        try container.encode(comparedTokens, forKey: .comparedTokens)
        try container.encodeIfPresent(firstDivergenceIndex, forKey: .firstDivergenceIndex)
        try container.encode(lengthMatched, forKey: .lengthMatched)
    }
}

public enum QwenMTPCorpusCaseKind: String, Codable, Equatable, Sendable {
    case fullGreedy
    case lengthBoundary
    case cancellationRetainedToken
    case cancellationAcceptedDraft
    case forcedFallback
}

public enum QwenMTPCorpusStopOutcome: String, Codable, Equatable, Sendable {
    case stop
    case length
    case cancelled
}

public struct QwenMTPCorpusCaseSpec: Codable, Equatable, Sendable {
    public let id: String
    public let kind: QwenMTPCorpusCaseKind
    public let prompt: String
    public let maxTokens: Int

    public init(id: String, kind: QwenMTPCorpusCaseKind, prompt: String, maxTokens: Int) {
        self.id = id
        self.kind = kind
        self.prompt = prompt
        self.maxTokens = maxTokens
    }
}

public enum QwenMTPCorpusRunOrder: String, Codable, Equatable, Sendable {
    case scalarThenMTP
    case mtpThenScalar
}

public struct QwenMTPCorpusProfilePlan: Codable, Equatable, Sendable {
    public let caseIDs: [String]
    public let droppedWarmupPairs: Int
    public let measuredPairs: Int
    public let orders: [QwenMTPCorpusRunOrder]

    public init(
        caseIDs: [String],
        droppedWarmupPairs: Int,
        measuredPairs: Int,
        orders: [QwenMTPCorpusRunOrder]
    ) {
        self.caseIDs = caseIDs
        self.droppedWarmupPairs = droppedWarmupPairs
        self.measuredPairs = measuredPairs
        self.orders = orders
    }

    public var totalPairsPerCase: Int { droppedWarmupPairs + measuredPairs }
}

public struct QwenMTPCorpusRuntimeBinding: Codable, Equatable, Sendable {
    public let targetModelID: String
    public let drafterModelID: String
    public let targetRevision: String
    public let drafterRevision: String
    public let sourceRevision: String
    public let blockSize: Int
    public let maxAcceptedDrafts: Int

    public init(
        targetModelID: String,
        drafterModelID: String,
        targetRevision: String,
        drafterRevision: String,
        sourceRevision: String,
        blockSize: Int,
        maxAcceptedDrafts: Int
    ) {
        self.targetModelID = targetModelID
        self.drafterModelID = drafterModelID
        self.targetRevision = targetRevision
        self.drafterRevision = drafterRevision
        self.sourceRevision = sourceRevision
        self.blockSize = blockSize
        self.maxAcceptedDrafts = maxAcceptedDrafts
    }
}

public struct QwenMTPCorpusHostEvidence: Codable, Equatable, Sendable {
    public let chip: String
    public let ramBytes: UInt64
    public let os: String

    public init(chip: String, ramBytes: UInt64, os: String) {
        self.chip = chip
        self.ramBytes = ramBytes
        self.os = os
    }
}

public struct QwenMTPCorpusCacheStateFingerprint: Codable, Equatable, Sendable {
    public let stateIndex: Int
    public let shape: [Int]
    public let dtype: String
    public let byteCount: Int
    public let sha256: String

    public init(stateIndex: Int, shape: [Int], dtype: String, byteCount: Int, sha256: String) {
        self.stateIndex = stateIndex
        self.shape = shape
        self.dtype = dtype
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct QwenMTPCorpusCacheLayerFingerprint: Codable, Equatable, Sendable {
    public let layerIndex: Int
    public let cacheType: String
    public let offset: Int
    public let metaStateSHA256: String
    public let stateCount: Int
    public let states: [QwenMTPCorpusCacheStateFingerprint]

    public init(
        layerIndex: Int,
        cacheType: String,
        offset: Int,
        metaStateSHA256: String,
        stateCount: Int,
        states: [QwenMTPCorpusCacheStateFingerprint]
    ) {
        self.layerIndex = layerIndex
        self.cacheType = cacheType
        self.offset = offset
        self.metaStateSHA256 = metaStateSHA256
        self.stateCount = stateCount
        self.states = states
    }
}

public struct QwenMTPCorpusCacheFingerprint: Codable, Equatable, Sendable {
    public let digest: String
    public let entries: [QwenMTPCorpusCacheLayerFingerprint]

    public init(digest: String, entries: [QwenMTPCorpusCacheLayerFingerprint]) {
        self.digest = digest
        self.entries = entries
    }

    public init(digest: String, entryCount: Int) {
        let entries = (0..<entryCount).map { index in
            QwenMTPCorpusCacheLayerFingerprint(
                layerIndex: index,
                cacheType: "legacy",
                offset: 0,
                metaStateSHA256: digest,
                stateCount: 1,
                states: [
                    .init(
                        stateIndex: 0,
                        shape: [1],
                        dtype: "legacy",
                        byteCount: 0,
                        sha256: digest),
                ])
        }
        self.init(digest: digest, entries: entries)
    }

    public var entryCount: Int { entries.reduce(0) { $0 + $1.states.count } }
}

public struct QwenMTPCorpusTiming: Codable, Equatable, Sendable {
    public let promptSeconds: Double
    public let generationSeconds: Double
    public let wallSeconds: Double
    public let e2eSeconds: Double

    public init(
        promptSeconds: Double,
        generationSeconds: Double,
        wallSeconds: Double,
        e2eSeconds: Double
    ) {
        self.promptSeconds = promptSeconds
        self.generationSeconds = generationSeconds
        self.wallSeconds = wallSeconds
        self.e2eSeconds = e2eSeconds
    }

    public var allFinitePositive: Bool {
        [promptSeconds, generationSeconds, wallSeconds, e2eSeconds].allSatisfy {
            $0.isFinite && $0 > 0
        }
    }
}

/// Measurement-only attribution for the MTP path. Cache fingerprinting is
/// intentionally separated from the decode envelope because it is proof
/// overhead, not user-visible generation latency.
public struct QwenMTPPromptPreparationChunkAttribution: Codable, Equatable, Sendable {
    public let tokenOffset: Int
    public let tokenCount: Int
    public let targetForwardSchedulingSeconds: Double

    public init(
        tokenOffset: Int,
        tokenCount: Int,
        targetForwardSchedulingSeconds: Double
    ) {
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
        self.targetForwardSchedulingSeconds = targetForwardSchedulingSeconds
    }
}

public struct QwenMTPPromptPreparationAttribution: Codable, Equatable, Sendable {
    public let promptTokenCount: Int
    public let hiddenShape: [Int]
    public let hiddenByteCount: Int
    public let chunks: [QwenMTPPromptPreparationChunkAttribution]
    public let cacheEvaluationSeconds: Double
    public let hiddenEvaluationSeconds: Double
    public let concatenatedHiddenEvaluationSeconds: Double
    public let preparedCacheHandoffSeconds: Double
    public let phaseBoundarySynchronizationSeconds: Double
    public let targetPrefillResidualSeconds: Double

    public init(
        promptTokenCount: Int,
        hiddenShape: [Int],
        hiddenByteCount: Int,
        chunks: [QwenMTPPromptPreparationChunkAttribution],
        cacheEvaluationSeconds: Double,
        hiddenEvaluationSeconds: Double,
        concatenatedHiddenEvaluationSeconds: Double,
        preparedCacheHandoffSeconds: Double,
        phaseBoundarySynchronizationSeconds: Double,
        targetPrefillResidualSeconds: Double
    ) {
        self.promptTokenCount = promptTokenCount
        self.hiddenShape = hiddenShape
        self.hiddenByteCount = hiddenByteCount
        self.chunks = chunks
        self.cacheEvaluationSeconds = cacheEvaluationSeconds
        self.hiddenEvaluationSeconds = hiddenEvaluationSeconds
        self.concatenatedHiddenEvaluationSeconds = concatenatedHiddenEvaluationSeconds
        self.preparedCacheHandoffSeconds = preparedCacheHandoffSeconds
        self.phaseBoundarySynchronizationSeconds = phaseBoundarySynchronizationSeconds
        self.targetPrefillResidualSeconds = targetPrefillResidualSeconds
    }

    public init(
        promptTokenCount: Int,
        hiddenShape: [Int],
        hiddenByteCount: Int,
        chunks: [QwenMTPPromptPreparationChunkAttribution],
        cacheHiddenEvaluationSeconds: Double,
        hiddenConcatenationSeconds: Double,
        targetPrefillResidualSeconds: Double
    ) {
        self.init(
            promptTokenCount: promptTokenCount,
            hiddenShape: hiddenShape,
            hiddenByteCount: hiddenByteCount,
            chunks: chunks,
            cacheEvaluationSeconds: cacheHiddenEvaluationSeconds,
            hiddenEvaluationSeconds: 0,
            concatenatedHiddenEvaluationSeconds: hiddenConcatenationSeconds,
            preparedCacheHandoffSeconds: 0,
            phaseBoundarySynchronizationSeconds: 0,
            targetPrefillResidualSeconds: targetPrefillResidualSeconds)
    }

    public var targetForwardSchedulingSeconds: Double {
        chunks.reduce(0) { $0 + $1.targetForwardSchedulingSeconds }
    }

    public var cacheHiddenEvaluationSeconds: Double {
        cacheEvaluationSeconds + hiddenEvaluationSeconds
    }

    public var hiddenConcatenationSeconds: Double {
        concatenatedHiddenEvaluationSeconds
    }

    public var attributedSeconds: Double {
        targetForwardSchedulingSeconds
            + cacheEvaluationSeconds
            + hiddenEvaluationSeconds
            + concatenatedHiddenEvaluationSeconds
            + preparedCacheHandoffSeconds
            + phaseBoundarySynchronizationSeconds
    }
}

public struct QwenMTPCorpusMTPPhaseAttribution: Codable, Equatable, Sendable {
    public let targetPrefillSeconds: Double
    public let drafterPromptPrimingSeconds: Double
    public let draftBlockSeconds: Double
    public let targetVerificationSeconds: Double
    public let targetTailSeconds: Double
    public let hybridRewindReplaySeconds: Double
    public let finalizationSeconds: Double
    public let cacheFingerprintSeconds: Double
    public let targetPrefillCount: Int
    public let drafterPromptPrimingCount: Int
    public let draftBlockCount: Int
    public let targetVerificationCount: Int
    public let targetTailCount: Int
    public let hybridRewindReplayCount: Int
    public let finalizationCount: Int
    public let cacheFingerprintCount: Int
    public let targetPromptPreparation: QwenMTPPromptPreparationAttribution?

    public init(
        targetPrefillSeconds: Double,
        drafterPromptPrimingSeconds: Double,
        draftBlockSeconds: Double,
        targetVerificationSeconds: Double,
        targetTailSeconds: Double,
        hybridRewindReplaySeconds: Double,
        finalizationSeconds: Double,
        cacheFingerprintSeconds: Double,
        targetPrefillCount: Int,
        drafterPromptPrimingCount: Int,
        draftBlockCount: Int,
        targetVerificationCount: Int,
        targetTailCount: Int,
        hybridRewindReplayCount: Int,
        finalizationCount: Int,
        cacheFingerprintCount: Int,
        targetPromptPreparation: QwenMTPPromptPreparationAttribution?
    ) {
        self.targetPrefillSeconds = targetPrefillSeconds
        self.drafterPromptPrimingSeconds = drafterPromptPrimingSeconds
        self.draftBlockSeconds = draftBlockSeconds
        self.targetVerificationSeconds = targetVerificationSeconds
        self.targetTailSeconds = targetTailSeconds
        self.hybridRewindReplaySeconds = hybridRewindReplaySeconds
        self.finalizationSeconds = finalizationSeconds
        self.cacheFingerprintSeconds = cacheFingerprintSeconds
        self.targetPrefillCount = targetPrefillCount
        self.drafterPromptPrimingCount = drafterPromptPrimingCount
        self.draftBlockCount = draftBlockCount
        self.targetVerificationCount = targetVerificationCount
        self.targetTailCount = targetTailCount
        self.hybridRewindReplayCount = hybridRewindReplayCount
        self.finalizationCount = finalizationCount
        self.cacheFingerprintCount = cacheFingerprintCount
        self.targetPromptPreparation = targetPromptPreparation
    }

    public var promptSeconds: Double {
        targetPrefillSeconds + drafterPromptPrimingSeconds
    }

    public var generationSeconds: Double {
        draftBlockSeconds + targetVerificationSeconds + targetTailSeconds
            + hybridRewindReplaySeconds + finalizationSeconds
    }
}

public struct QwenMTPCorpusMTPTelemetry: Codable, Equatable, Sendable {
    public let proposedDraftTokens: Int
    public let acceptedDraftTokens: Int
    public let rejectedDraftTokens: Int
    public let roundCount: Int
    public let targetModelCallCount: Int
    public let draftModelCallCount: Int
    public let targetVerifiedTokenCount: Int
    public let emittedTokenCount: Int

    public init(
        proposedDraftTokens: Int,
        acceptedDraftTokens: Int,
        rejectedDraftTokens: Int,
        roundCount: Int,
        targetModelCallCount: Int,
        draftModelCallCount: Int,
        targetVerifiedTokenCount: Int,
        emittedTokenCount: Int
    ) {
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.rejectedDraftTokens = rejectedDraftTokens
        self.roundCount = roundCount
        self.targetModelCallCount = targetModelCallCount
        self.draftModelCallCount = draftModelCallCount
        self.targetVerifiedTokenCount = targetVerifiedTokenCount
        self.emittedTokenCount = emittedTokenCount
    }
}

public struct QwenMTPCorpusExactnessEvidence: Codable, Equatable, Sendable {
    public let scalarTokenCount: Int
    public let mtpTokenCount: Int
    public let scalarTokenIDsSHA256: String
    public let mtpTokenIDsSHA256: String
    public let scalarDecodedBytesSHA256: String
    public let mtpDecodedBytesSHA256: String
    public let scalarStopOutcome: QwenMTPCorpusStopOutcome
    public let mtpStopOutcome: QwenMTPCorpusStopOutcome
    public let scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let firstCacheMismatch: String?

    public init(
        scalarTokenCount: Int,
        mtpTokenCount: Int,
        scalarTokenIDsSHA256: String,
        mtpTokenIDsSHA256: String,
        scalarDecodedBytesSHA256: String,
        mtpDecodedBytesSHA256: String,
        scalarStopOutcome: QwenMTPCorpusStopOutcome,
        mtpStopOutcome: QwenMTPCorpusStopOutcome,
        scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        firstCacheMismatch: String?
    ) {
        self.scalarTokenCount = scalarTokenCount
        self.mtpTokenCount = mtpTokenCount
        self.scalarTokenIDsSHA256 = scalarTokenIDsSHA256
        self.mtpTokenIDsSHA256 = mtpTokenIDsSHA256
        self.scalarDecodedBytesSHA256 = scalarDecodedBytesSHA256
        self.mtpDecodedBytesSHA256 = mtpDecodedBytesSHA256
        self.scalarStopOutcome = scalarStopOutcome
        self.mtpStopOutcome = mtpStopOutcome
        self.scalarCacheFingerprint = scalarCacheFingerprint
        self.mtpCacheFingerprint = mtpCacheFingerprint
        self.firstCacheMismatch = firstCacheMismatch
    }
}

public struct QwenMTPCorpusCaseResult: Codable, Equatable, Sendable {
    public let caseID: String
    public let kind: QwenMTPCorpusCaseKind
    public let maxTokens: Int
    public let promptTokenCount: Int
    public let scalarTokenCount: Int
    public let mtpTokenCount: Int
    public let scalarTokenIDsSHA256: String
    public let mtpTokenIDsSHA256: String
    public let tokenExactness: MTPStreamExactness.Result
    public let scalarDecodedBytesSHA256: String
    public let mtpDecodedBytesSHA256: String
    public let scalarStopOutcome: QwenMTPCorpusStopOutcome
    public let mtpStopOutcome: QwenMTPCorpusStopOutcome
    public let scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let firstCacheMismatch: String?
    public let scalarTiming: QwenMTPCorpusTiming
    public let mtpTiming: QwenMTPCorpusTiming
    public let mtpTelemetry: QwenMTPCorpusMTPTelemetry
    public let mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution
    public let passthroughReason: String?

    public var draftsProposed: Int { mtpTelemetry.proposedDraftTokens }
    public var draftsAccepted: Int { mtpTelemetry.acceptedDraftTokens }

    public init(
        caseID: String,
        kind: QwenMTPCorpusCaseKind,
        maxTokens: Int,
        promptTokenCount: Int,
        scalarTokenCount: Int,
        mtpTokenCount: Int,
        scalarTokenIDsSHA256: String,
        mtpTokenIDsSHA256: String,
        tokenExactness: MTPStreamExactness.Result,
        scalarDecodedBytesSHA256: String,
        mtpDecodedBytesSHA256: String,
        scalarStopOutcome: QwenMTPCorpusStopOutcome,
        mtpStopOutcome: QwenMTPCorpusStopOutcome,
        scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        firstCacheMismatch: String?,
        scalarTiming: QwenMTPCorpusTiming,
        mtpTiming: QwenMTPCorpusTiming,
        mtpTelemetry: QwenMTPCorpusMTPTelemetry,
        mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution,
        passthroughReason: String?
    ) {
        self.caseID = caseID
        self.kind = kind
        self.maxTokens = maxTokens
        self.promptTokenCount = promptTokenCount
        self.scalarTokenCount = scalarTokenCount
        self.mtpTokenCount = mtpTokenCount
        self.scalarTokenIDsSHA256 = scalarTokenIDsSHA256
        self.mtpTokenIDsSHA256 = mtpTokenIDsSHA256
        self.tokenExactness = tokenExactness
        self.scalarDecodedBytesSHA256 = scalarDecodedBytesSHA256
        self.mtpDecodedBytesSHA256 = mtpDecodedBytesSHA256
        self.scalarStopOutcome = scalarStopOutcome
        self.mtpStopOutcome = mtpStopOutcome
        self.scalarCacheFingerprint = scalarCacheFingerprint
        self.mtpCacheFingerprint = mtpCacheFingerprint
        self.firstCacheMismatch = firstCacheMismatch
        self.scalarTiming = scalarTiming
        self.mtpTiming = mtpTiming
        self.mtpTelemetry = mtpTelemetry
        self.mtpPhaseAttribution = mtpPhaseAttribution
        self.passthroughReason = passthroughReason
    }

    private func copy(
        scalarTokenIDsSHA256: String? = nil,
        mtpTokenIDsSHA256: String? = nil,
        tokenExactness: MTPStreamExactness.Result? = nil,
        scalarDecodedBytesSHA256: String? = nil,
        mtpDecodedBytesSHA256: String? = nil,
        scalarStopOutcome: QwenMTPCorpusStopOutcome? = nil,
        mtpStopOutcome: QwenMTPCorpusStopOutcome? = nil,
        scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint? = nil,
        mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint? = nil,
        firstCacheMismatch: String?? = nil,
        mtpTelemetry: QwenMTPCorpusMTPTelemetry? = nil,
        mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution? = nil,
        passthroughReason: String?? = nil
    ) -> QwenMTPCorpusCaseResult {
        QwenMTPCorpusCaseResult(
            caseID: caseID,
            kind: kind,
            maxTokens: maxTokens,
            promptTokenCount: promptTokenCount,
            scalarTokenCount: scalarTokenCount,
            mtpTokenCount: mtpTokenCount,
            scalarTokenIDsSHA256: scalarTokenIDsSHA256 ?? self.scalarTokenIDsSHA256,
            mtpTokenIDsSHA256: mtpTokenIDsSHA256 ?? self.mtpTokenIDsSHA256,
            tokenExactness: tokenExactness ?? self.tokenExactness,
            scalarDecodedBytesSHA256: scalarDecodedBytesSHA256 ?? self.scalarDecodedBytesSHA256,
            mtpDecodedBytesSHA256: mtpDecodedBytesSHA256 ?? self.mtpDecodedBytesSHA256,
            scalarStopOutcome: scalarStopOutcome ?? self.scalarStopOutcome,
            mtpStopOutcome: mtpStopOutcome ?? self.mtpStopOutcome,
            scalarCacheFingerprint: scalarCacheFingerprint ?? self.scalarCacheFingerprint,
            mtpCacheFingerprint: mtpCacheFingerprint ?? self.mtpCacheFingerprint,
            firstCacheMismatch: firstCacheMismatch ?? self.firstCacheMismatch,
            scalarTiming: scalarTiming,
            mtpTiming: mtpTiming,
            mtpTelemetry: mtpTelemetry ?? self.mtpTelemetry,
            mtpPhaseAttribution: mtpPhaseAttribution ?? self.mtpPhaseAttribution,
            passthroughReason: passthroughReason ?? self.passthroughReason)
    }

    public func withTokenExactness(_ tokenExactness: MTPStreamExactness.Result) -> QwenMTPCorpusCaseResult {
        copy(tokenExactness: tokenExactness)
    }

    public func withTokenIDDigests(scalar: String, mtp: String) -> QwenMTPCorpusCaseResult {
        copy(scalarTokenIDsSHA256: scalar, mtpTokenIDsSHA256: mtp)
    }

    public func withDrafts(proposed: Int, accepted: Int) -> QwenMTPCorpusCaseResult {
        copy(mtpTelemetry: .init(
            proposedDraftTokens: proposed,
            acceptedDraftTokens: accepted,
            rejectedDraftTokens: max(0, proposed - accepted),
            roundCount: proposed > 0 ? max(1, mtpTelemetry.roundCount) : 0,
            targetModelCallCount: proposed > 0 ? max(1, mtpTelemetry.targetModelCallCount) : mtpTelemetry.targetModelCallCount,
            draftModelCallCount: proposed > 0 ? max(1, mtpTelemetry.draftModelCallCount) : 0,
            targetVerifiedTokenCount: proposed > 0 ? max(1, mtpTelemetry.targetVerifiedTokenCount) : 0,
            emittedTokenCount: mtpTelemetry.emittedTokenCount))
    }

    public func withPassthroughReason(_ passthroughReason: String?) -> QwenMTPCorpusCaseResult {
        copy(passthroughReason: .some(passthroughReason))
    }

    public func withDecodedDigests(scalar: String, mtp: String) -> QwenMTPCorpusCaseResult {
        copy(scalarDecodedBytesSHA256: scalar, mtpDecodedBytesSHA256: mtp)
    }

    public func withCacheFingerprints(
        scalar: QwenMTPCorpusCacheFingerprint,
        mtp: QwenMTPCorpusCacheFingerprint,
        firstMismatch: String?
    ) -> QwenMTPCorpusCaseResult {
        copy(
            scalarCacheFingerprint: scalar,
            mtpCacheFingerprint: mtp,
            firstCacheMismatch: .some(firstMismatch))
    }

    public func withStopOutcomes(
        scalar: QwenMTPCorpusStopOutcome,
        mtp: QwenMTPCorpusStopOutcome
    ) -> QwenMTPCorpusCaseResult {
        copy(scalarStopOutcome: scalar, mtpStopOutcome: mtp)
    }

    public func withMTPTelemetry(_ telemetry: QwenMTPCorpusMTPTelemetry) -> QwenMTPCorpusCaseResult {
        copy(mtpTelemetry: telemetry)
    }

    public func withMTPPhaseAttribution(
        _ attribution: QwenMTPCorpusMTPPhaseAttribution
    ) -> QwenMTPCorpusCaseResult {
        copy(mtpPhaseAttribution: attribution)
    }
}

public enum QwenMTPCorpusCorrectnessVerdict: String, Codable, Equatable, Sendable {
    case pass
    case fail
}

public struct QwenMTPCorpusProfileSample: Codable, Equatable, Sendable {
    public let caseID: String
    public let pairIndex: Int
    public let warmup: Bool
    public let order: QwenMTPCorpusRunOrder
    public let exactness: QwenMTPCorpusExactnessEvidence
    public let scalarTiming: QwenMTPCorpusTiming
    public let mtpTiming: QwenMTPCorpusTiming
    public let scalarTokensPerSecond: Double
    public let mtpTokensPerSecond: Double
    public let decodeOnlyRatio: Double
    public let e2eRatio: Double
    public let mtpTelemetry: QwenMTPCorpusMTPTelemetry
    public let mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution
    public let passthroughReason: String?

    public var draftsProposed: Int { mtpTelemetry.proposedDraftTokens }
    public var draftsAccepted: Int { mtpTelemetry.acceptedDraftTokens }

    public init(
        caseID: String,
        pairIndex: Int,
        warmup: Bool,
        order: QwenMTPCorpusRunOrder,
        exactness: QwenMTPCorpusExactnessEvidence,
        scalarTiming: QwenMTPCorpusTiming,
        mtpTiming: QwenMTPCorpusTiming,
        scalarTokensPerSecond: Double,
        mtpTokensPerSecond: Double,
        decodeOnlyRatio: Double,
        e2eRatio: Double,
        mtpTelemetry: QwenMTPCorpusMTPTelemetry,
        mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution,
        passthroughReason: String?
    ) {
        self.caseID = caseID
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.order = order
        self.exactness = exactness
        self.scalarTiming = scalarTiming
        self.mtpTiming = mtpTiming
        self.scalarTokensPerSecond = scalarTokensPerSecond
        self.mtpTokensPerSecond = mtpTokensPerSecond
        self.decodeOnlyRatio = decodeOnlyRatio
        self.e2eRatio = e2eRatio
        self.mtpTelemetry = mtpTelemetry
        self.mtpPhaseAttribution = mtpPhaseAttribution
        self.passthroughReason = passthroughReason
    }

    private func copy(
        order: QwenMTPCorpusRunOrder? = nil,
        exactness: QwenMTPCorpusExactnessEvidence? = nil,
        scalarTiming: QwenMTPCorpusTiming? = nil,
        scalarTokensPerSecond: Double? = nil,
        mtpTokensPerSecond: Double? = nil,
        decodeOnlyRatio: Double? = nil,
        e2eRatio: Double? = nil
    ) -> QwenMTPCorpusProfileSample {
        QwenMTPCorpusProfileSample(
            caseID: caseID,
            pairIndex: pairIndex,
            warmup: warmup,
            order: order ?? self.order,
            exactness: exactness ?? self.exactness,
            scalarTiming: scalarTiming ?? self.scalarTiming,
            mtpTiming: mtpTiming,
            scalarTokensPerSecond: scalarTokensPerSecond ?? self.scalarTokensPerSecond,
            mtpTokensPerSecond: mtpTokensPerSecond ?? self.mtpTokensPerSecond,
            decodeOnlyRatio: decodeOnlyRatio ?? self.decodeOnlyRatio,
            e2eRatio: e2eRatio ?? self.e2eRatio,
            mtpTelemetry: mtpTelemetry,
            mtpPhaseAttribution: mtpPhaseAttribution,
            passthroughReason: passthroughReason)
    }

    public func withOrder(_ order: QwenMTPCorpusRunOrder) -> QwenMTPCorpusProfileSample {
        copy(order: order)
    }

    public func withScalarTiming(_ scalarTiming: QwenMTPCorpusTiming) -> QwenMTPCorpusProfileSample {
        copy(scalarTiming: scalarTiming)
    }

    public func withExactness(_ exactness: QwenMTPCorpusExactnessEvidence) -> QwenMTPCorpusProfileSample {
        copy(exactness: exactness)
    }

    public func withReportedPerformance(
        scalarTokensPerSecond: Double,
        mtpTokensPerSecond: Double,
        decodeOnlyRatio: Double,
        e2eRatio: Double
    ) -> QwenMTPCorpusProfileSample {
        copy(
            scalarTokensPerSecond: scalarTokensPerSecond,
            mtpTokensPerSecond: mtpTokensPerSecond,
            decodeOnlyRatio: decodeOnlyRatio,
            e2eRatio: e2eRatio)
    }
}

public struct QwenMTPCorpusProfileEvidence: Codable, Equatable, Sendable {
    public let releaseBuildRequired: Bool
    public let releaseBuildObserved: Bool
    public var samples: [QwenMTPCorpusProfileSample]

    public init(
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        samples: [QwenMTPCorpusProfileSample]
    ) {
        self.releaseBuildRequired = releaseBuildRequired
        self.releaseBuildObserved = releaseBuildObserved
        self.samples = samples
    }
}

public struct QwenMTPCorpusEvidencePayload: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let corpusID: String
    public let corpusContentHash: String
    public var binding: QwenMTPCorpusRuntimeBinding
    public let host: QwenMTPCorpusHostEvidence
    public var caseResults: [QwenMTPCorpusCaseResult]
    public var correctness: QwenMTPCorpusCorrectnessVerdict
    public var profile: QwenMTPCorpusProfileEvidence?

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        binding: QwenMTPCorpusRuntimeBinding,
        host: QwenMTPCorpusHostEvidence,
        caseResults: [QwenMTPCorpusCaseResult],
        correctness: QwenMTPCorpusCorrectnessVerdict,
        profile: QwenMTPCorpusProfileEvidence?
    ) {
        self.schemaVersion = schemaVersion
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
        self.binding = binding
        self.host = host
        self.caseResults = caseResults
        self.correctness = correctness
        self.profile = profile
    }
}

public struct QwenMTPCorpusProfileVerdict: Codable, Equatable, Sendable {
    public let qualified: Bool
    public let aggregatePairedMedian: Double
    public let chronologicalFirstHalfMedian: Double
    public let chronologicalSecondHalfMedian: Double
    public let perPromptMedians: [String: Double]
    public let perPromptMedianBelowFloorCount: Int
    public let aggregateThreshold: Double
    public let chronologicalHalfThreshold: Double
    public let perPromptFloor: Double
    public let hiddenMaterializationSecondsTotal: Double
    public let promptOverheadSecondsTotal: Double
    public let hiddenMaterializationShareOfPromptOverhead: Double
    public let hiddenMaterializationCandidateQualified: Bool
    public let hiddenMaterializationCandidateThresholdSeconds: Double

    public init(
        qualified: Bool,
        aggregatePairedMedian: Double,
        chronologicalFirstHalfMedian: Double,
        chronologicalSecondHalfMedian: Double,
        perPromptMedians: [String: Double],
        perPromptMedianBelowFloorCount: Int,
        aggregateThreshold: Double,
        chronologicalHalfThreshold: Double,
        perPromptFloor: Double,
        hiddenMaterializationSecondsTotal: Double,
        promptOverheadSecondsTotal: Double,
        hiddenMaterializationShareOfPromptOverhead: Double,
        hiddenMaterializationCandidateQualified: Bool,
        hiddenMaterializationCandidateThresholdSeconds: Double
    ) {
        self.qualified = qualified
        self.aggregatePairedMedian = aggregatePairedMedian
        self.chronologicalFirstHalfMedian = chronologicalFirstHalfMedian
        self.chronologicalSecondHalfMedian = chronologicalSecondHalfMedian
        self.perPromptMedians = perPromptMedians
        self.perPromptMedianBelowFloorCount = perPromptMedianBelowFloorCount
        self.aggregateThreshold = aggregateThreshold
        self.chronologicalHalfThreshold = chronologicalHalfThreshold
        self.perPromptFloor = perPromptFloor
        self.hiddenMaterializationSecondsTotal = hiddenMaterializationSecondsTotal
        self.promptOverheadSecondsTotal = promptOverheadSecondsTotal
        self.hiddenMaterializationShareOfPromptOverhead =
            hiddenMaterializationShareOfPromptOverhead
        self.hiddenMaterializationCandidateQualified =
            hiddenMaterializationCandidateQualified
        self.hiddenMaterializationCandidateThresholdSeconds =
            hiddenMaterializationCandidateThresholdSeconds
    }
}

public struct QwenMTPPromptHiddenReuseVerdict: Codable, Equatable, Sendable {
    public let qualified: Bool
    public let measuredDrafterPromptPrimingSeconds: Double
    public let baselineDrafterPromptPrimingSeconds: Double
    public let reductionSeconds: Double
    public let requiredReductionSeconds: Double

    public init(
        qualified: Bool,
        measuredDrafterPromptPrimingSeconds: Double,
        baselineDrafterPromptPrimingSeconds: Double,
        reductionSeconds: Double,
        requiredReductionSeconds: Double
    ) {
        self.qualified = qualified
        self.measuredDrafterPromptPrimingSeconds = measuredDrafterPromptPrimingSeconds
        self.baselineDrafterPromptPrimingSeconds = baselineDrafterPromptPrimingSeconds
        self.reductionSeconds = reductionSeconds
        self.requiredReductionSeconds = requiredReductionSeconds
    }
}

public struct QwenMTPGreedyBatchedVerificationVerdict: Codable, Equatable, Sendable {
    public let qualified: Bool
    public let measuredTargetVerificationSeconds: Double
    public let baselineTargetVerificationSeconds: Double
    public let reductionSeconds: Double
    public let requiredReductionSeconds: Double

    public init(
        qualified: Bool,
        measuredTargetVerificationSeconds: Double,
        baselineTargetVerificationSeconds: Double,
        reductionSeconds: Double,
        requiredReductionSeconds: Double
    ) {
        self.qualified = qualified
        self.measuredTargetVerificationSeconds = measuredTargetVerificationSeconds
        self.baselineTargetVerificationSeconds = baselineTargetVerificationSeconds
        self.reductionSeconds = reductionSeconds
        self.requiredReductionSeconds = requiredReductionSeconds
    }
}

public struct QwenMTPCorpusGateDecision: Codable, Equatable, Sendable {
    public let correctness: QwenMTPCorpusCorrectnessVerdict
    public let profile: QwenMTPCorpusProfileVerdict?

    public init(correctness: QwenMTPCorpusCorrectnessVerdict, profile: QwenMTPCorpusProfileVerdict?) {
        self.correctness = correctness
        self.profile = profile
    }
}

public enum QwenMTPCorpusGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case schemaVersionMismatch(Int)
    case corpusIdentityMismatch
    case invalidBinding(String)
    case invalidHost
    case invalidCaseCardinality(expected: Int, actual: Int)
    case invalidCaseOrder(index: Int, expected: String, actual: String)
    case invalidCaseMetadata(String)
    case correctnessMismatch(String)
    case profilePresentAfterCorrectnessFailure
    case releaseBuildRequired
    case invalidProfileCardinality(expected: Int, actual: Int)
    case invalidProfileSample(index: Int, reason: String)
    case invalidProvenanceModelIdentity
    case unqualifiedPerformance(QwenMTPCorpusProfileVerdict)
    case promptHiddenReuseProfileRequired
    case insufficientPromptHiddenReuseReduction(QwenMTPPromptHiddenReuseVerdict)
    case greedyBatchedVerificationProfileRequired
    case insufficientGreedyBatchedVerificationReduction(
        QwenMTPGreedyBatchedVerificationVerdict)
    case malformedJSONL(line: Int)
    case unterminatedJSONL
    case wrongSubcommand(String)

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let value):
            return "schemaVersion mismatch: \(value)"
        case .corpusIdentityMismatch:
            return "corpus identity mismatch"
        case .invalidBinding(let field):
            return "invalid runtime binding: \(field)"
        case .invalidHost:
            return "invalid host evidence"
        case .invalidCaseCardinality(let expected, let actual):
            return "invalid case cardinality: expected \(expected), got \(actual)"
        case .invalidCaseOrder(let index, let expected, let actual):
            return "invalid case order at \(index): expected \(expected), got \(actual)"
        case .invalidCaseMetadata(let reason):
            return "invalid case metadata: \(reason)"
        case .correctnessMismatch(let reason):
            return "correctness mismatch: \(reason)"
        case .profilePresentAfterCorrectnessFailure:
            return "profile evidence cannot be accepted after correctness failure"
        case .releaseBuildRequired:
            return "profile evidence requires a Release build"
        case .invalidProfileCardinality(let expected, let actual):
            return "invalid profile sample cardinality: expected \(expected), got \(actual)"
        case .invalidProfileSample(let index, let reason):
            return "invalid profile sample \(index): \(reason)"
        case .invalidProvenanceModelIdentity:
            return "invalid provenance model identity"
        case .unqualifiedPerformance(let verdict):
            let promptSummary = verdict.perPromptMedians
                .sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: ",")
            return "unqualified performance: aggregate=\(verdict.aggregatePairedMedian) "
                + "firstHalf=\(verdict.chronologicalFirstHalfMedian) "
                + "secondHalf=\(verdict.chronologicalSecondHalfMedian) "
                + "belowFloor=\(verdict.perPromptMedianBelowFloorCount) "
                + "perPrompt=[\(promptSummary)] "
                + "hiddenMaterialization=\(verdict.hiddenMaterializationSecondsTotal) "
                + "promptOverhead=\(verdict.promptOverheadSecondsTotal) "
                + "hiddenShare=\(verdict.hiddenMaterializationShareOfPromptOverhead) "
                + "thresholds=\(verdict.aggregateThreshold)/"
                + "\(verdict.chronologicalHalfThreshold)/"
                + "\(verdict.perPromptFloor)"
        case .promptHiddenReuseProfileRequired:
            return "prompt-hidden reuse requires complete profile evidence"
        case .insufficientPromptHiddenReuseReduction(let verdict):
            return "insufficient prompt-hidden reuse reduction: \(verdict.reductionSeconds) seconds"
        case .greedyBatchedVerificationProfileRequired:
            return "greedy batched verification requires complete profile evidence"
        case .insufficientGreedyBatchedVerificationReduction(let verdict):
            return "insufficient greedy batched verification reduction: "
                + "\(verdict.reductionSeconds) seconds"
        case .malformedJSONL(let line):
            return "malformed JSONL at line \(line)"
        case .unterminatedJSONL:
            return "unterminated JSONL"
        case .wrongSubcommand(let subcommand):
            return "wrong subcommand: \(subcommand)"
        }
    }
}

private struct QwenMTPCorpusSchemaProbe: Codable, Sendable {
    let schemaVersion: Int
}

public enum QwenMTPCorpusGate {
    public static let schemaVersion = 4
    public static let corpusID = "qwen3.5-9b-mtp-consumer-corpus-v1"
    public static let corpusContentHash = "5e3bc2fbb016d5e0"
    public static let promptHiddenReusePrimingBaselineSeconds = 88.6121
    public static let promptHiddenReuseRequiredReductionSeconds = 80.0
    public static let greedyBatchedVerificationBaselineSeconds = 204.43435170891462
    public static let greedyBatchedVerificationRequiredReductionSeconds = 5.0
    public static let hiddenMaterializationCandidateThresholdSeconds = 5.0

    public static let requiredBinding = QwenMTPCorpusRuntimeBinding(
        targetModelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
        drafterModelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
        targetRevision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
        drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33",
        sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
        blockSize: 3,
        maxAcceptedDrafts: 2)

    public static let cases: [QwenMTPCorpusCaseSpec] = [
        .init(id: "prime-sequence", kind: .fullGreedy, prompt: "Continue the sequence with one integer per line, preserving the pattern and no prose:\n2\n3\n5\n7\n11\n13\n17\n19\n23\n29\n", maxTokens: 128),
        .init(id: "swift-code", kind: .fullGreedy, prompt: "Write a compact Swift function that validates a JSON Lines evidence file. Requirements: reject embedded newlines, reject unterminated final lines, decode each row into a typed envelope, and return the count of accepted rows. Include a tiny example call.", maxTokens: 128),
        .init(id: "strict-json-list", kind: .fullGreedy, prompt: "Return strict JSON only. Produce an array of six objects. Each object must have keys \"id\", \"risk\", \"mitigation\", and \"owner\". The topic is a runtime gate for speculative decoding evidence.", maxTokens: 128),
        .init(id: "math-reasoning", kind: .fullGreedy, prompt: "Solve step by step. A benchmark runs 2 warmup pairs and 5 measured pairs for each of 6 prompts. Each pair runs scalar once and MTP once. How many scalar runs and MTP runs are measured, excluding warmups? Then give the total including warmups.", maxTokens: 128),
        .init(id: "long-retrieval", kind: .fullGreedy, prompt: longRetrievalPrompt, maxTokens: 128),
        .init(id: "dialogue-tool-like", kind: .fullGreedy, prompt: "User: I need a deployment gate that refuses to publish speedups until correctness evidence is complete.\nAssistant: I can help. Available tool schema: {\"name\":\"record_evidence\",\"arguments\":{\"case_id\":\"string\",\"passed\":\"boolean\",\"digest\":\"string\"}}\nUser: Draft the next assistant message, including exactly one tool-like JSON call and then a short plain-language summary.", maxTokens: 128),
        .init(id: "length-boundary-1", kind: .lengthBoundary, prompt: "Return exactly one concise completion token after this colon:", maxTokens: 1),
        .init(id: "length-boundary-2", kind: .lengthBoundary, prompt: "Return exactly two concise completion tokens after this colon:", maxTokens: 2),
        .init(id: "cancel-retained-1", kind: .cancellationRetainedToken, prompt: "Generate a numbered checklist for cache finalization evidence. The consumer will retain one generated token and then cancel.", maxTokens: 128),
        .init(id: "cancel-after-accepted-draft", kind: .cancellationAcceptedDraft, prompt: "Repeat this patterned phrase to encourage accepted speculative drafts: alpha beta gamma alpha beta gamma alpha beta", maxTokens: 128),
        .init(id: "forced-fallback-seeded", kind: .forcedFallback, prompt: "Sample a brief non-greedy continuation for a speculative decoder fallback check with a fixed seed.", maxTokens: 128),
    ]

    public static let profilePlan = QwenMTPCorpusProfilePlan(
        caseIDs: ["prime-sequence", "swift-code", "strict-json-list", "math-reasoning", "long-retrieval", "dialogue-tool-like"],
        droppedWarmupPairs: 2,
        measuredPairs: 5,
        orders: [.scalarThenMTP, .mtpThenScalar, .scalarThenMTP, .mtpThenScalar, .scalarThenMTP, .mtpThenScalar, .scalarThenMTP])

    public static func scalarRetainedTokenLimit(forCaseID caseID: String) -> Int? {
        caseID == "cancel-retained-1" ? 1 : nil
    }

    public static func canonicalCorrectnessFailurePayload(
        from payload: QwenMTPCorpusEvidencePayload
    ) -> QwenMTPCorpusEvidencePayload {
        var failedPayload = payload
        failedPayload.correctness = .fail
        failedPayload.profile = nil
        return failedPayload
    }

    public static func validate(_ payload: QwenMTPCorpusEvidencePayload) throws -> QwenMTPCorpusGateDecision {
        try validateCorrectness(payload)
        if let profile = payload.profile {
            let verdict = try validateProfile(
                profile,
                correctness: payload.correctness,
                promptTokenCounts: Dictionary(
                    uniqueKeysWithValues: payload.caseResults.map {
                        ($0.caseID, $0.promptTokenCount)
                    }))
            guard verdict.qualified else {
                throw QwenMTPCorpusGateError.unqualifiedPerformance(verdict)
            }
            return QwenMTPCorpusGateDecision(correctness: .pass, profile: verdict)
        }
        return QwenMTPCorpusGateDecision(correctness: .pass, profile: nil)
    }

    /// Apply the optimization-specific promotion gate without changing the
    /// frozen corpus identity or its general MTP qualification thresholds.
    public static func evaluatePromptHiddenReuse(
        _ payload: QwenMTPCorpusEvidencePayload
    ) throws -> QwenMTPPromptHiddenReuseVerdict {
        let decision = try validate(payload)
        guard decision.profile != nil, let profile = payload.profile else {
            throw QwenMTPCorpusGateError.promptHiddenReuseProfileRequired
        }
        let measuredSeconds = profile.samples.lazy
            .filter { !$0.warmup }
            .reduce(0) { partial, sample in
                partial + sample.mtpPhaseAttribution.drafterPromptPrimingSeconds
            }
        return promptHiddenReuseVerdict(measuredPrimingSeconds: measuredSeconds)
    }

    public static func validatePromptHiddenReuse(
        _ payload: QwenMTPCorpusEvidencePayload
    ) throws -> QwenMTPPromptHiddenReuseVerdict {
        let verdict = try evaluatePromptHiddenReuse(payload)
        guard verdict.qualified else {
            throw QwenMTPCorpusGateError.insufficientPromptHiddenReuseReduction(verdict)
        }
        return verdict
    }

    static func promptHiddenReuseVerdict(
        measuredPrimingSeconds: Double
    ) -> QwenMTPPromptHiddenReuseVerdict {
        let reductionSeconds =
            promptHiddenReusePrimingBaselineSeconds - measuredPrimingSeconds
        let maximumMeasuredSeconds =
            promptHiddenReusePrimingBaselineSeconds
            - promptHiddenReuseRequiredReductionSeconds
        return QwenMTPPromptHiddenReuseVerdict(
            qualified: measuredPrimingSeconds.isFinite
                && measuredPrimingSeconds >= 0
                && measuredPrimingSeconds <= maximumMeasuredSeconds,
            measuredDrafterPromptPrimingSeconds: measuredPrimingSeconds,
            baselineDrafterPromptPrimingSeconds: promptHiddenReusePrimingBaselineSeconds,
            reductionSeconds: reductionSeconds,
            requiredReductionSeconds: promptHiddenReuseRequiredReductionSeconds)
    }

    public static func evaluateGreedyBatchedVerification(
        _ payload: QwenMTPCorpusEvidencePayload
    ) throws -> QwenMTPGreedyBatchedVerificationVerdict {
        let decision = try validate(payload)
        guard decision.profile != nil, let profile = payload.profile else {
            throw QwenMTPCorpusGateError.greedyBatchedVerificationProfileRequired
        }
        let measuredSeconds = profile.samples.lazy
            .filter { !$0.warmup }
            .reduce(0) { partial, sample in
                partial + sample.mtpPhaseAttribution.targetVerificationSeconds
            }
        return greedyBatchedVerificationVerdict(
            measuredVerificationSeconds: measuredSeconds)
    }

    public static func validateGreedyBatchedVerification(
        _ payload: QwenMTPCorpusEvidencePayload
    ) throws -> QwenMTPGreedyBatchedVerificationVerdict {
        let verdict = try evaluateGreedyBatchedVerification(payload)
        guard verdict.qualified else {
            throw QwenMTPCorpusGateError.insufficientGreedyBatchedVerificationReduction(
                verdict)
        }
        return verdict
    }

    static func greedyBatchedVerificationVerdict(
        measuredVerificationSeconds: Double
    ) -> QwenMTPGreedyBatchedVerificationVerdict {
        let reductionSeconds =
            greedyBatchedVerificationBaselineSeconds - measuredVerificationSeconds
        let maximumMeasuredSeconds =
            greedyBatchedVerificationBaselineSeconds
            - greedyBatchedVerificationRequiredReductionSeconds
        return QwenMTPGreedyBatchedVerificationVerdict(
            qualified: measuredVerificationSeconds.isFinite
                && measuredVerificationSeconds >= 0
                && measuredVerificationSeconds <= maximumMeasuredSeconds,
            measuredTargetVerificationSeconds: measuredVerificationSeconds,
            baselineTargetVerificationSeconds: greedyBatchedVerificationBaselineSeconds,
            reductionSeconds: reductionSeconds,
            requiredReductionSeconds: greedyBatchedVerificationRequiredReductionSeconds)
    }

    public static func validateJSONL(_ data: Data) throws -> [QwenMTPCorpusGateDecision] {
        guard data.last == 0x0a else { throw QwenMTPCorpusGateError.unterminatedJSONL }
        let rows = data.split(separator: 0x0a, omittingEmptySubsequences: false).dropLast()
        var decisions: [QwenMTPCorpusGateDecision] = []
        decisions.reserveCapacity(rows.count)
        let decoder = JSONDecoder()
        for (index, row) in rows.enumerated() {
            guard !row.isEmpty else { throw QwenMTPCorpusGateError.malformedJSONL(line: index + 1) }
            let rowData = Data(row)
            let probe: ResultRecord<QwenMTPCorpusSchemaProbe>
            do {
                probe = try decoder.decode(
                    ResultRecord<QwenMTPCorpusSchemaProbe>.self, from: rowData)
            } catch {
                throw QwenMTPCorpusGateError.malformedJSONL(line: index + 1)
            }
            guard probe.payload.schemaVersion == schemaVersion else {
                throw QwenMTPCorpusGateError.schemaVersionMismatch(
                    probe.payload.schemaVersion)
            }
            let record: ResultRecord<QwenMTPCorpusEvidencePayload>
            do {
                record = try decoder.decode(
                    ResultRecord<QwenMTPCorpusEvidencePayload>.self, from: rowData)
            } catch {
                throw QwenMTPCorpusGateError.malformedJSONL(line: index + 1)
            }
            guard record.subcommand == "qwen-mtp-corpus" else {
                throw QwenMTPCorpusGateError.wrongSubcommand(record.subcommand)
            }
            guard record.provenance.modelPath == record.payload.binding.targetModelID else {
                throw QwenMTPCorpusGateError.invalidProvenanceModelIdentity
            }
            decisions.append(try validate(record.payload))
        }
        return decisions
    }

    private static func validateCorrectness(_ payload: QwenMTPCorpusEvidencePayload) throws {
        guard payload.schemaVersion == schemaVersion else {
            throw QwenMTPCorpusGateError.schemaVersionMismatch(payload.schemaVersion)
        }
        guard payload.corpusID == corpusID, payload.corpusContentHash == corpusContentHash else {
            throw QwenMTPCorpusGateError.corpusIdentityMismatch
        }
        try validateBinding(payload.binding)
        guard !payload.host.chip.isEmpty, payload.host.ramBytes > 0, !payload.host.os.isEmpty else {
            throw QwenMTPCorpusGateError.invalidHost
        }
        guard payload.caseResults.count == cases.count else {
            throw QwenMTPCorpusGateError.invalidCaseCardinality(expected: cases.count, actual: payload.caseResults.count)
        }
        for (index, expected) in cases.enumerated() {
            let actual = payload.caseResults[index]
            guard actual.caseID == expected.id else {
                throw QwenMTPCorpusGateError.invalidCaseOrder(index: index, expected: expected.id, actual: actual.caseID)
            }
            guard actual.kind == expected.kind, actual.maxTokens == expected.maxTokens else {
                throw QwenMTPCorpusGateError.invalidCaseMetadata(actual.caseID)
            }
            try validateCaseResult(actual)
        }
        guard payload.correctness == .pass else {
            throw QwenMTPCorpusGateError.correctnessMismatch("payload correctness is \(payload.correctness.rawValue)")
        }
    }

    private static func validateBinding(_ binding: QwenMTPCorpusRuntimeBinding) throws {
        guard binding == requiredBinding else {
            throw QwenMTPCorpusGateError.invalidBinding("exact qwen35_9BDepth1 binding")
        }
    }

    private static func validateCaseResult(_ result: QwenMTPCorpusCaseResult) throws {
        guard result.promptTokenCount > 0 else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(result.caseID) promptTokenCount")
        }
        let exactness = QwenMTPCorpusExactnessEvidence(
            scalarTokenCount: result.scalarTokenCount,
            mtpTokenCount: result.mtpTokenCount,
            scalarTokenIDsSHA256: result.scalarTokenIDsSHA256,
            mtpTokenIDsSHA256: result.mtpTokenIDsSHA256,
            scalarDecodedBytesSHA256: result.scalarDecodedBytesSHA256,
            mtpDecodedBytesSHA256: result.mtpDecodedBytesSHA256,
            scalarStopOutcome: result.scalarStopOutcome,
            mtpStopOutcome: result.mtpStopOutcome,
            scalarCacheFingerprint: result.scalarCacheFingerprint,
            mtpCacheFingerprint: result.mtpCacheFingerprint,
            firstCacheMismatch: result.firstCacheMismatch)
        try validateExactness(exactness, context: result.caseID)
        guard result.scalarTokenCount <= result.maxTokens else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) token count exceeds maxTokens")
        }
        guard result.tokenExactness.exact,
            result.tokenExactness.lengthMatched,
            result.tokenExactness.comparedTokens == result.scalarTokenCount,
            result.tokenExactness.firstDivergenceIndex == nil
        else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) token exactness")
        }
        guard result.scalarTiming.allFinitePositive, result.mtpTiming.allFinitePositive else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(result.caseID) timing")
        }
        try validateTelemetry(result.mtpTelemetry, emittedTokenCount: result.mtpTokenCount, context: result.caseID)
        try validatePhaseAttribution(
            result.mtpPhaseAttribution,
            telemetry: result.mtpTelemetry,
            timing: result.mtpTiming,
            expectsDrafterPriming: result.kind != .forcedFallback,
            expectedPromptTokenCount: result.promptTokenCount,
            context: result.caseID)

        switch result.kind {
        case .lengthBoundary:
            guard result.scalarStopOutcome == .length, result.mtpStopOutcome == .length, result.scalarTokenCount == result.maxTokens else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) length outcome")
            }
            guard result.passthroughReason == nil else { throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) passthrough") }
        case .cancellationRetainedToken:
            guard result.scalarStopOutcome == .cancelled,
                result.mtpStopOutcome == .cancelled,
                result.scalarTokenCount == scalarRetainedTokenLimit(forCaseID: result.caseID)
            else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) cancellation outcome")
            }
            guard result.passthroughReason == nil else { throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) passthrough") }
        case .cancellationAcceptedDraft:
            guard result.scalarStopOutcome == .cancelled, result.mtpStopOutcome == .cancelled else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) cancellation outcome")
            }
            guard result.mtpTelemetry.proposedDraftTokens > 0, result.mtpTelemetry.acceptedDraftTokens > 0 else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) no accepted draft")
            }
            guard result.passthroughReason == nil else { throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) passthrough") }
        case .fullGreedy:
            guard result.scalarStopOutcome != .cancelled, result.mtpStopOutcome != .cancelled else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) cancelled full case")
            }
            guard result.mtpTelemetry.proposedDraftTokens > 0, result.mtpTelemetry.roundCount > 0 else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) no draft activity")
            }
            guard result.passthroughReason == nil else { throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) passthrough") }
        case .forcedFallback:
            guard result.scalarStopOutcome != .cancelled, result.mtpStopOutcome != .cancelled else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) cancelled fallback")
            }
            guard result.mtpTelemetry.proposedDraftTokens == 0,
                result.mtpTelemetry.acceptedDraftTokens == 0,
                result.mtpTelemetry.rejectedDraftTokens == 0,
                result.mtpTelemetry.roundCount == 0,
                result.mtpTelemetry.targetModelCallCount == 0,
                result.mtpTelemetry.draftModelCallCount == 0,
                result.mtpTelemetry.targetVerifiedTokenCount == 0
            else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) fallback drafted")
            }
            guard let reason = result.passthroughReason, !reason.isEmpty else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(result.caseID) fallback reason")
            }
            guard result.mtpPhaseAttribution.targetTailCount
                == max(0, result.mtpTokenCount - 1)
            else {
                throw QwenMTPCorpusGateError.correctnessMismatch(
                    "\(result.caseID) fallback target-tail attribution")
            }
        }
    }

    private static func validateExactness(_ exactness: QwenMTPCorpusExactnessEvidence, context: String) throws {
        guard exactness.scalarTokenCount == exactness.mtpTokenCount, exactness.scalarTokenCount > 0 else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) token counts")
        }
        guard isLowerHexSHA256(exactness.scalarTokenIDsSHA256),
            isLowerHexSHA256(exactness.mtpTokenIDsSHA256),
            exactness.scalarTokenIDsSHA256 == exactness.mtpTokenIDsSHA256
        else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) token ID digest")
        }
        guard isLowerHexSHA256(exactness.scalarDecodedBytesSHA256),
            isLowerHexSHA256(exactness.mtpDecodedBytesSHA256),
            exactness.scalarDecodedBytesSHA256 == exactness.mtpDecodedBytesSHA256
        else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) decoded bytes")
        }
        guard exactness.scalarStopOutcome == exactness.mtpStopOutcome else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) stop outcome")
        }
        try validateCacheFingerprint(exactness.scalarCacheFingerprint, context: "\(context) scalar cache")
        try validateCacheFingerprint(exactness.mtpCacheFingerprint, context: "\(context) mtp cache")
        guard exactness.scalarCacheFingerprint == exactness.mtpCacheFingerprint, exactness.firstCacheMismatch == nil else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) cache fingerprint")
        }
    }

    private static func validateCacheFingerprint(_ fingerprint: QwenMTPCorpusCacheFingerprint, context: String) throws {
        guard isLowerHexSHA256(fingerprint.digest), !fingerprint.entries.isEmpty else {
            throw QwenMTPCorpusGateError.correctnessMismatch("\(context) aggregate digest")
        }
        for (layerOrdinal, layer) in fingerprint.entries.enumerated() {
            guard layer.layerIndex == layerOrdinal else { throw QwenMTPCorpusGateError.correctnessMismatch("\(context) layer index") }
            guard !layer.cacheType.isEmpty, layer.offset >= 0, isLowerHexSHA256(layer.metaStateSHA256) else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(context) layer metadata")
            }
            guard layer.stateCount == layer.states.count, !layer.states.isEmpty else {
                throw QwenMTPCorpusGateError.correctnessMismatch("\(context) state count")
            }
            for (stateOrdinal, state) in layer.states.enumerated() {
                guard state.stateIndex == stateOrdinal else { throw QwenMTPCorpusGateError.correctnessMismatch("\(context) state index") }
                guard !state.shape.isEmpty,
                    state.shape.allSatisfy({ $0 >= 0 }),
                    !state.dtype.isEmpty,
                    state.byteCount >= 0,
                    isLowerHexSHA256(state.sha256)
                else {
                    throw QwenMTPCorpusGateError.correctnessMismatch("\(context) state metadata")
                }
            }
        }
    }

    private static func validateTelemetry(_ telemetry: QwenMTPCorpusMTPTelemetry, emittedTokenCount: Int, context: String) throws {
        let counts = [
            telemetry.proposedDraftTokens,
            telemetry.acceptedDraftTokens,
            telemetry.rejectedDraftTokens,
            telemetry.roundCount,
            telemetry.targetModelCallCount,
            telemetry.draftModelCallCount,
            telemetry.targetVerifiedTokenCount,
            telemetry.emittedTokenCount,
        ]
        guard counts.allSatisfy({ $0 >= 0 }) else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) telemetry negative")
        }
        guard telemetry.emittedTokenCount == emittedTokenCount else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) telemetry emitted")
        }
        guard telemetry.acceptedDraftTokens <= telemetry.proposedDraftTokens,
            telemetry.rejectedDraftTokens == telemetry.proposedDraftTokens - telemetry.acceptedDraftTokens
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) telemetry draft totals")
        }
        guard telemetry.targetModelCallCount == telemetry.roundCount,
            telemetry.draftModelCallCount == telemetry.roundCount,
            telemetry.targetVerifiedTokenCount
                == telemetry.proposedDraftTokens + telemetry.roundCount
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) telemetry call coherence")
        }
    }

    private static func validatePhaseAttribution(
        _ phase: QwenMTPCorpusMTPPhaseAttribution,
        telemetry: QwenMTPCorpusMTPTelemetry,
        timing: QwenMTPCorpusTiming,
        expectsDrafterPriming: Bool,
        expectedPromptTokenCount: Int,
        context: String
    ) throws {
        let seconds = [
            phase.targetPrefillSeconds,
            phase.drafterPromptPrimingSeconds,
            phase.draftBlockSeconds,
            phase.targetVerificationSeconds,
            phase.targetTailSeconds,
            phase.hybridRewindReplaySeconds,
            phase.finalizationSeconds,
            phase.cacheFingerprintSeconds,
        ]
        guard seconds.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) phase timing")
        }

        let counts = [
            phase.targetPrefillCount,
            phase.drafterPromptPrimingCount,
            phase.draftBlockCount,
            phase.targetVerificationCount,
            phase.targetTailCount,
            phase.hybridRewindReplayCount,
            phase.finalizationCount,
            phase.cacheFingerprintCount,
        ]
        guard counts.allSatisfy({ $0 >= 0 }) else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) phase count")
        }
        guard phase.targetPrefillCount == 1,
            phase.drafterPromptPrimingCount == (expectsDrafterPriming ? 1 : 0),
            phase.draftBlockCount == telemetry.draftModelCallCount,
            phase.targetVerificationCount == telemetry.targetModelCallCount,
            phase.targetTailCount <= telemetry.emittedTokenCount,
            phase.hybridRewindReplayCount <= telemetry.targetModelCallCount + 1,
            phase.finalizationCount == 1,
            phase.cacheFingerprintCount == 1
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) phase count coherence")
        }

        let zeroCountPhases = [
            (phase.drafterPromptPrimingCount, phase.drafterPromptPrimingSeconds),
            (phase.draftBlockCount, phase.draftBlockSeconds),
            (phase.targetVerificationCount, phase.targetVerificationSeconds),
            (phase.targetTailCount, phase.targetTailSeconds),
            (phase.hybridRewindReplayCount, phase.hybridRewindReplaySeconds),
        ]
        guard zeroCountPhases.allSatisfy({ count, elapsed in count > 0 || elapsed == 0 }) else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata("\(context) phase time without work")
        }

        let tolerance = 0.005
        guard phase.promptSeconds <= timing.promptSeconds + tolerance,
            phase.generationSeconds <= timing.generationSeconds + tolerance,
            phase.promptSeconds + phase.generationSeconds <= timing.wallSeconds + tolerance
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) phase envelope "
                    + "prompt=\(phase.promptSeconds)/\(timing.promptSeconds) "
                    + "generation=\(phase.generationSeconds)/\(timing.generationSeconds) "
                    + "total=\(phase.promptSeconds + phase.generationSeconds)/\(timing.wallSeconds)")
        }

        if expectsDrafterPriming {
            guard let preparation = phase.targetPromptPreparation else {
                throw QwenMTPCorpusGateError.invalidCaseMetadata(
                    "\(context) target prompt preparation missing")
            }
            try validatePromptPreparation(
                preparation,
                targetPrefillSeconds: phase.targetPrefillSeconds,
                expectedPromptTokenCount: expectedPromptTokenCount,
                context: context)
        } else if phase.targetPromptPreparation != nil {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) target prompt preparation without MTP")
        }
    }

    private static func validatePromptPreparation(
        _ preparation: QwenMTPPromptPreparationAttribution,
        targetPrefillSeconds: Double,
        expectedPromptTokenCount: Int,
        context: String
    ) throws {
        guard preparation.promptTokenCount == expectedPromptTokenCount,
            preparation.promptTokenCount > 0,
            preparation.hiddenShape.count == 3,
            preparation.hiddenShape[0] == 1,
            preparation.hiddenShape[1] == preparation.promptTokenCount,
            preparation.hiddenShape[2] > 0,
            preparation.hiddenByteCount > 0,
            !preparation.chunks.isEmpty,
            preparation.cacheEvaluationSeconds.isFinite,
            preparation.cacheEvaluationSeconds >= 0,
            preparation.hiddenEvaluationSeconds.isFinite,
            preparation.hiddenEvaluationSeconds >= 0,
            preparation.concatenatedHiddenEvaluationSeconds.isFinite,
            preparation.concatenatedHiddenEvaluationSeconds >= 0,
            preparation.preparedCacheHandoffSeconds.isFinite,
            preparation.preparedCacheHandoffSeconds >= 0,
            preparation.phaseBoundarySynchronizationSeconds.isFinite,
            preparation.phaseBoundarySynchronizationSeconds >= 0,
            preparation.targetPrefillResidualSeconds.isFinite,
            preparation.targetPrefillResidualSeconds >= 0
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) target prompt preparation geometry")
        }

        var elementCount = 1
        for dimension in preparation.hiddenShape {
            let (next, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
            guard !overflow, next > 0 else {
                throw QwenMTPCorpusGateError.invalidCaseMetadata(
                    "\(context) target prompt hidden size")
            }
            elementCount = next
        }
        guard preparation.hiddenByteCount.isMultiple(of: elementCount),
            [1, 2, 4, 8].contains(preparation.hiddenByteCount / elementCount)
        else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) target prompt hidden bytes")
        }

        var expectedOffset = 0
        for chunk in preparation.chunks {
            guard chunk.tokenOffset == expectedOffset,
                chunk.tokenCount > 0,
                chunk.targetForwardSchedulingSeconds.isFinite,
                chunk.targetForwardSchedulingSeconds >= 0
            else {
                throw QwenMTPCorpusGateError.invalidCaseMetadata(
                    "\(context) target prompt chunk")
            }
            let (nextOffset, overflow) = expectedOffset.addingReportingOverflow(
                chunk.tokenCount)
            guard !overflow else {
                throw QwenMTPCorpusGateError.invalidCaseMetadata(
                    "\(context) target prompt chunk overflow")
            }
            expectedOffset = nextOffset
        }
        guard expectedOffset == preparation.promptTokenCount else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) target prompt chunk coverage")
        }

        let reconstructed = preparation.attributedSeconds
            + preparation.targetPrefillResidualSeconds
        guard approximatelyEqual(reconstructed, targetPrefillSeconds) else {
            throw QwenMTPCorpusGateError.invalidCaseMetadata(
                "\(context) target prompt phase envelope")
        }
    }

    private static func validateProfile(
        _ profile: QwenMTPCorpusProfileEvidence,
        correctness: QwenMTPCorpusCorrectnessVerdict,
        promptTokenCounts: [String: Int]
    ) throws -> QwenMTPCorpusProfileVerdict {
        guard correctness == .pass else { throw QwenMTPCorpusGateError.profilePresentAfterCorrectnessFailure }
        guard profile.releaseBuildRequired, profile.releaseBuildObserved else {
            throw QwenMTPCorpusGateError.releaseBuildRequired
        }
        let expectedCount = profilePlan.caseIDs.count * profilePlan.totalPairsPerCase
        guard profile.samples.count == expectedCount else {
            throw QwenMTPCorpusGateError.invalidProfileCardinality(expected: expectedCount, actual: profile.samples.count)
        }

        var sampleIndex = 0
        var measuredByCase: [String: [Double]] = [:]
        var measuredByPairIndex: [Int: [Double]] = [:]
        var measuredRatios: [Double] = []
        var hiddenMaterializationSecondsTotal = 0.0
        var promptOverheadSecondsTotal = 0.0

        for caseID in profilePlan.caseIDs {
            for pairIndex in 0..<profilePlan.totalPairsPerCase {
                let sample = profile.samples[sampleIndex]
                guard sample.caseID == caseID else { throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "caseID") }
                guard sample.pairIndex == pairIndex else { throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "pairIndex") }
                guard sample.order == profilePlan.orders[pairIndex] else { throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "order") }
                guard sample.warmup == (pairIndex < profilePlan.droppedWarmupPairs) else { throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "warmup") }
                do {
                    try validateExactness(sample.exactness, context: "\(sample.caseID) profile[\(sample.pairIndex)]")
                    try validateTelemetry(sample.mtpTelemetry, emittedTokenCount: sample.exactness.mtpTokenCount, context: "\(sample.caseID) profile[\(sample.pairIndex)]")
                    try validatePhaseAttribution(
                        sample.mtpPhaseAttribution,
                        telemetry: sample.mtpTelemetry,
                        timing: sample.mtpTiming,
                        expectsDrafterPriming: true,
                        expectedPromptTokenCount: promptTokenCounts[caseID] ?? 0,
                        context: "\(sample.caseID) profile[\(sample.pairIndex)]")
                } catch {
                    throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: String(describing: error))
                }
                guard sample.scalarTiming.allFinitePositive,
                    sample.mtpTiming.allFinitePositive,
                    sample.scalarTokensPerSecond.isFinite,
                    sample.scalarTokensPerSecond > 0,
                    sample.mtpTokensPerSecond.isFinite,
                    sample.mtpTokensPerSecond > 0,
                    sample.decodeOnlyRatio.isFinite,
                    sample.decodeOnlyRatio > 0,
                    sample.e2eRatio.isFinite,
                    sample.e2eRatio > 0
                else {
                    throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "timing")
                }
                let recomputedScalarTPS =
                    Double(sample.exactness.scalarTokenCount) / sample.scalarTiming.e2eSeconds
                let recomputedMTPTPS =
                    Double(sample.exactness.mtpTokenCount) / sample.mtpTiming.e2eSeconds
                let recomputedDecodeRatio =
                    sample.scalarTiming.generationSeconds / sample.mtpTiming.generationSeconds
                let recomputedE2ERatio = recomputedMTPTPS / recomputedScalarTPS
                guard approximatelyEqual(sample.scalarTokensPerSecond, recomputedScalarTPS),
                    approximatelyEqual(sample.mtpTokensPerSecond, recomputedMTPTPS),
                    approximatelyEqual(sample.decodeOnlyRatio, recomputedDecodeRatio),
                    approximatelyEqual(sample.e2eRatio, recomputedE2ERatio)
                else {
                    throw QwenMTPCorpusGateError.invalidProfileSample(
                        index: sampleIndex,
                        reason: "reported performance does not match counts and timings")
                }
                guard sample.mtpTelemetry.proposedDraftTokens > 0,
                    sample.mtpTelemetry.acceptedDraftTokens >= 0,
                    sample.mtpTelemetry.acceptedDraftTokens <= sample.mtpTelemetry.proposedDraftTokens,
                    sample.passthroughReason == nil
                else {
                    throw QwenMTPCorpusGateError.invalidProfileSample(index: sampleIndex, reason: "draft telemetry")
                }

                if !sample.warmup {
                    measuredRatios.append(recomputedE2ERatio)
                    measuredByCase[caseID, default: []].append(recomputedE2ERatio)
                    measuredByPairIndex[pairIndex - profilePlan.droppedWarmupPairs, default: []].append(recomputedE2ERatio)
                    if let preparation = sample.mtpPhaseAttribution.targetPromptPreparation {
                        hiddenMaterializationSecondsTotal +=
                            preparation.hiddenEvaluationSeconds
                            + preparation.concatenatedHiddenEvaluationSeconds
                    }
                    promptOverheadSecondsTotal += max(
                        0,
                        sample.mtpTiming.promptSeconds
                            - sample.scalarTiming.promptSeconds)
                }
                sampleIndex += 1
            }
        }

        let aggregateMedian = median(measuredRatios)
        let firstHalfMedian = median([0, 1].flatMap { measuredByPairIndex[$0] ?? [] })
        let secondHalfMedian = median([2, 3, 4].flatMap { measuredByPairIndex[$0] ?? [] })
        let perPromptMedians = profilePlan.caseIDs.reduce(into: [String: Double]()) { out, caseID in
            out[caseID] = median(measuredByCase[caseID] ?? [])
        }
        let weakPromptCount = perPromptMedians.values.filter { $0 < 0.97 }.count
        let qualified = aggregateMedian >= 1.08
            && firstHalfMedian >= 1.05
            && secondHalfMedian >= 1.05
            && weakPromptCount <= 1
        let hiddenMaterializationShare = promptOverheadSecondsTotal > 0
            ? hiddenMaterializationSecondsTotal / promptOverheadSecondsTotal
            : 0

        return QwenMTPCorpusProfileVerdict(
            qualified: qualified,
            aggregatePairedMedian: aggregateMedian,
            chronologicalFirstHalfMedian: firstHalfMedian,
            chronologicalSecondHalfMedian: secondHalfMedian,
            perPromptMedians: perPromptMedians,
            perPromptMedianBelowFloorCount: weakPromptCount,
            aggregateThreshold: 1.08,
            chronologicalHalfThreshold: 1.05,
            perPromptFloor: 0.97,
            hiddenMaterializationSecondsTotal: hiddenMaterializationSecondsTotal,
            promptOverheadSecondsTotal: promptOverheadSecondsTotal,
            hiddenMaterializationShareOfPromptOverhead:
                hiddenMaterializationShare,
            hiddenMaterializationCandidateQualified:
                hiddenMaterializationSecondsTotal
                    >= hiddenMaterializationCandidateThresholdSeconds,
            hiddenMaterializationCandidateThresholdSeconds:
                hiddenMaterializationCandidateThresholdSeconds)
    }

    private static func isLowerHexSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func approximatelyEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        let scale = max(1, abs(lhs), abs(rhs))
        return abs(lhs - rhs) <= scale * 1e-9
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return .nan }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static let longRetrievalPrompt: String = {
        let paragraph = """
        Evidence packet section: the runtime integrator must compare scalar decoding with native MTP decoding, retain only case identifiers and digests, reject raw prompt persistence, fingerprint every cache layer, and refuse performance claims until correctness passes. The record includes token counts, decoded byte digests, cache state digests, draft telemetry, passthrough reason, and timing fields.
        """
        return (1...18).map { index in
            "\(index). \(paragraph)"
        }.joined(separator: "\n")
            + "\nQuestion: Which fields prove equivalence without storing raw generated output?"
    }()
}
