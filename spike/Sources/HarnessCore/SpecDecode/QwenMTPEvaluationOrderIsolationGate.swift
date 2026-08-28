import Foundation

public enum QwenMTPPromptEvaluationOrderEvidence: String, Codable, Equatable, Sendable {
    case cacheFirst = "cache-first"
    case hiddenFirst = "hidden-first"
    case combined = "combined"
}

public enum QwenMTPEvaluationOrderPairRunOrder: String, Codable, Equatable, Sendable {
    case cacheFirstThenHiddenFirst = "cache-first-then-hidden-first"
    case hiddenFirstThenCacheFirst = "hidden-first-then-cache-first"
}

public struct QwenMTPEvaluationOrderRunEvidence: Codable, Equatable, Sendable {
    public let evaluationOrder: QwenMTPPromptEvaluationOrderEvidence
    public let timing: QwenMTPCorpusTiming
    public let exactness: QwenMTPCorpusExactnessEvidence
    public let telemetry: QwenMTPCorpusMTPTelemetry
    public let phaseAttribution: QwenMTPCorpusMTPPhaseAttribution
    public let passthroughReason: String?

    public init(
        evaluationOrder: QwenMTPPromptEvaluationOrderEvidence,
        timing: QwenMTPCorpusTiming,
        exactness: QwenMTPCorpusExactnessEvidence,
        telemetry: QwenMTPCorpusMTPTelemetry,
        phaseAttribution: QwenMTPCorpusMTPPhaseAttribution,
        passthroughReason: String?
    ) {
        self.evaluationOrder = evaluationOrder
        self.timing = timing
        self.exactness = exactness
        self.telemetry = telemetry
        self.phaseAttribution = phaseAttribution
        self.passthroughReason = passthroughReason
    }

    public func copy(
        evaluationOrder: QwenMTPPromptEvaluationOrderEvidence? = nil,
        timing: QwenMTPCorpusTiming? = nil,
        exactness: QwenMTPCorpusExactnessEvidence? = nil,
        telemetry: QwenMTPCorpusMTPTelemetry? = nil,
        phaseAttribution: QwenMTPCorpusMTPPhaseAttribution? = nil,
        passthroughReason: String?? = nil
    ) -> Self {
        .init(
            evaluationOrder: evaluationOrder ?? self.evaluationOrder,
            timing: timing ?? self.timing,
            exactness: exactness ?? self.exactness,
            telemetry: telemetry ?? self.telemetry,
            phaseAttribution: phaseAttribution ?? self.phaseAttribution,
            passthroughReason: passthroughReason ?? self.passthroughReason)
    }
}

public struct QwenMTPEvaluationOrderPairEvidence: Codable, Equatable, Sendable {
    public let pairIndex: Int
    public let warmup: Bool
    public let runOrder: QwenMTPEvaluationOrderPairRunOrder
    public let cacheFirst: QwenMTPEvaluationOrderRunEvidence
    public let hiddenFirst: QwenMTPEvaluationOrderRunEvidence

    public init(
        pairIndex: Int,
        warmup: Bool,
        runOrder: QwenMTPEvaluationOrderPairRunOrder,
        cacheFirst: QwenMTPEvaluationOrderRunEvidence,
        hiddenFirst: QwenMTPEvaluationOrderRunEvidence
    ) {
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.runOrder = runOrder
        self.cacheFirst = cacheFirst
        self.hiddenFirst = hiddenFirst
    }

    public func copy(
        runOrder: QwenMTPEvaluationOrderPairRunOrder? = nil,
        cacheFirst: QwenMTPEvaluationOrderRunEvidence? = nil,
        hiddenFirst: QwenMTPEvaluationOrderRunEvidence? = nil
    ) -> Self {
        .init(
            pairIndex: pairIndex,
            warmup: warmup,
            runOrder: runOrder ?? self.runOrder,
            cacheFirst: cacheFirst ?? self.cacheFirst,
            hiddenFirst: hiddenFirst ?? self.hiddenFirst)
    }
}

public struct QwenMTPEvaluationOrderIsolationPayload: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var corpusID: String
    public var corpusContentHash: String
    public var binding: QwenMTPCorpusRuntimeBinding
    public var host: QwenMTPCorpusHostEvidence
    public var releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var pairs: [QwenMTPEvaluationOrderPairEvidence]

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        binding: QwenMTPCorpusRuntimeBinding,
        host: QwenMTPCorpusHostEvidence,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        pairs: [QwenMTPEvaluationOrderPairEvidence]
    ) {
        self.schemaVersion = schemaVersion
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
        self.binding = binding
        self.host = host
        self.releaseBuildRequired = releaseBuildRequired
        self.releaseBuildObserved = releaseBuildObserved
        self.pairs = pairs
    }
}

public struct QwenMTPEvaluationOrderIsolationVerdict: Codable, Equatable, Sendable {
    public let qualified: Bool
    public let aggregatePromptImprovementSeconds: Double
    public let medianPromptImprovementSeconds: Double
    public let requiredAggregatePromptImprovementSeconds: Double
    public let requiredMedianPromptImprovementSeconds: Double

    public init(
        qualified: Bool,
        aggregatePromptImprovementSeconds: Double,
        medianPromptImprovementSeconds: Double,
        requiredAggregatePromptImprovementSeconds: Double,
        requiredMedianPromptImprovementSeconds: Double
    ) {
        self.qualified = qualified
        self.aggregatePromptImprovementSeconds = aggregatePromptImprovementSeconds
        self.medianPromptImprovementSeconds = medianPromptImprovementSeconds
        self.requiredAggregatePromptImprovementSeconds =
            requiredAggregatePromptImprovementSeconds
        self.requiredMedianPromptImprovementSeconds =
            requiredMedianPromptImprovementSeconds
    }
}

public enum QwenMTPEvaluationOrderIsolationGateError: Error, Equatable, Sendable {
    case schemaIdentity
    case binding
    case host
    case releaseBuild
    case pairCardinality(expected: Int, actual: Int)
    case invalidPair(index: Int, reason: String)
}

public enum QwenMTPEvaluationOrderIsolationGate {
    public static let schemaVersion = 1
    public static let corpusID = "qwen3.5-9b-mtp-eval-order-isolation-v1"
    public static let corpusContentHash = QwenMTPCorpusGate.corpusContentHash
    public static let droppedWarmupPairs = 2
    public static let measuredPairs = 5
    public static let requiredPromptTokenCount = 1_353
    public static let requiredAggregatePromptImprovementSeconds = 2.5
    public static let requiredMedianPromptImprovementSeconds = 0.40
    public static let pairOrders: [QwenMTPEvaluationOrderPairRunOrder] = [
        .cacheFirstThenHiddenFirst,
        .hiddenFirstThenCacheFirst,
        .cacheFirstThenHiddenFirst,
        .hiddenFirstThenCacheFirst,
        .cacheFirstThenHiddenFirst,
        .hiddenFirstThenCacheFirst,
        .cacheFirstThenHiddenFirst,
    ]

    public static func validate(
        _ payload: QwenMTPEvaluationOrderIsolationPayload
    ) throws -> QwenMTPEvaluationOrderIsolationVerdict {
        guard payload.schemaVersion == schemaVersion,
            payload.corpusID == corpusID,
            payload.corpusContentHash == corpusContentHash
        else {
            throw QwenMTPEvaluationOrderIsolationGateError.schemaIdentity
        }
        guard payload.binding == QwenMTPCorpusGate.requiredBinding else {
            throw QwenMTPEvaluationOrderIsolationGateError.binding
        }
        guard !payload.host.chip.isEmpty, payload.host.ramBytes > 0,
            !payload.host.os.isEmpty
        else {
            throw QwenMTPEvaluationOrderIsolationGateError.host
        }
        guard payload.releaseBuildRequired, payload.releaseBuildObserved else {
            throw QwenMTPEvaluationOrderIsolationGateError.releaseBuild
        }
        guard payload.pairs.count == pairOrders.count else {
            throw QwenMTPEvaluationOrderIsolationGateError.pairCardinality(
                expected: pairOrders.count,
                actual: payload.pairs.count)
        }

        var measuredImprovements = [Double]()
        measuredImprovements.reserveCapacity(measuredPairs)
        for (index, pair) in payload.pairs.enumerated() {
            try validatePair(pair, at: index)
            if !pair.warmup {
                let improvement = pair.cacheFirst.timing.promptSeconds
                    - pair.hiddenFirst.timing.promptSeconds
                guard improvement.isFinite else {
                    throw invalidPair(index, "paired prompt improvement")
                }
                measuredImprovements.append(improvement)
            }
        }

        let aggregate = measuredImprovements.reduce(0, +)
        let median = median(measuredImprovements)
        return QwenMTPEvaluationOrderIsolationVerdict(
            qualified: aggregate >= requiredAggregatePromptImprovementSeconds
                && median >= requiredMedianPromptImprovementSeconds,
            aggregatePromptImprovementSeconds: aggregate,
            medianPromptImprovementSeconds: median,
            requiredAggregatePromptImprovementSeconds:
                requiredAggregatePromptImprovementSeconds,
            requiredMedianPromptImprovementSeconds:
                requiredMedianPromptImprovementSeconds)
    }

    /// Validate one completed scalar/cache-first/hidden-first pair so a live
    /// producer can stop at the first correctness or evidence failure.
    public static func validatePair(
        _ pair: QwenMTPEvaluationOrderPairEvidence,
        at index: Int
    ) throws {
        guard pairOrders.indices.contains(index) else {
            throw invalidPair(index, "pair index out of range")
        }
        guard pair.pairIndex == index else {
            throw invalidPair(index, "pair index")
        }
        guard pair.warmup == (index < droppedWarmupPairs) else {
            throw invalidPair(index, "warmup")
        }
        guard pair.runOrder == pairOrders[index] else {
            throw invalidPair(index, "run order")
        }
        try validateDiagnosticRun(
            pair.cacheFirst,
            expectedOrder: .cacheFirst,
            pairIndex: index)
        try validateDiagnosticRun(
            pair.hiddenFirst,
            expectedOrder: .hiddenFirst,
            pairIndex: index)
        guard pair.cacheFirst.exactness.scalarTokenIDsSHA256
            == pair.hiddenFirst.exactness.scalarTokenIDsSHA256,
            pair.cacheFirst.exactness.scalarDecodedBytesSHA256
                == pair.hiddenFirst.exactness.scalarDecodedBytesSHA256,
            pair.cacheFirst.exactness.scalarStopOutcome
                == pair.hiddenFirst.exactness.scalarStopOutcome,
            pair.cacheFirst.exactness.scalarCacheFingerprint
                == pair.hiddenFirst.exactness.scalarCacheFingerprint
        else {
            throw invalidPair(index, "scalar control identity")
        }
    }

    static func validateDiagnosticRun(
        _ run: QwenMTPEvaluationOrderRunEvidence,
        expectedOrder: QwenMTPPromptEvaluationOrderEvidence,
        pairIndex: Int
    ) throws {
        guard run.evaluationOrder == expectedOrder else {
            throw invalidPair(pairIndex, "evaluation order")
        }
        guard run.passthroughReason == nil else {
            throw invalidPair(pairIndex, "passthrough")
        }
        guard run.timing.allFinitePositive,
            run.timing.promptSeconds + run.timing.generationSeconds
                <= run.timing.wallSeconds + 0.005,
            run.timing.wallSeconds <= run.timing.e2eSeconds + 0.005
        else {
            throw invalidPair(pairIndex, "timing")
        }
        try validateExactness(run.exactness, pairIndex: pairIndex)
        try validateTelemetry(run.telemetry, pairIndex: pairIndex)
        try validatePhase(
            run.phaseAttribution,
            telemetry: run.telemetry,
            timing: run.timing,
            pairIndex: pairIndex)
    }

    private static func validateExactness(
        _ exactness: QwenMTPCorpusExactnessEvidence,
        pairIndex: Int
    ) throws {
        guard exactness.scalarTokenCount == 128,
            exactness.mtpTokenCount == 128,
            exactness.scalarTokenIDsSHA256 == exactness.mtpTokenIDsSHA256,
            exactness.scalarDecodedBytesSHA256 == exactness.mtpDecodedBytesSHA256,
            exactness.scalarStopOutcome == .length,
            exactness.mtpStopOutcome == .length,
            exactness.scalarCacheFingerprint == exactness.mtpCacheFingerprint,
            exactness.scalarCacheFingerprint.entries.count == 32,
            exactness.firstCacheMismatch == nil
        else {
            throw invalidPair(pairIndex, "scalar exactness")
        }
    }

    private static func validateTelemetry(
        _ telemetry: QwenMTPCorpusMTPTelemetry,
        pairIndex: Int
    ) throws {
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
        guard counts.allSatisfy({ $0 >= 0 }),
            telemetry.proposedDraftTokens > 0,
            telemetry.acceptedDraftTokens > 0,
            telemetry.acceptedDraftTokens <= telemetry.proposedDraftTokens,
            telemetry.rejectedDraftTokens
                == telemetry.proposedDraftTokens - telemetry.acceptedDraftTokens,
            telemetry.roundCount > 0,
            telemetry.targetModelCallCount == telemetry.roundCount,
            telemetry.draftModelCallCount == telemetry.roundCount,
            telemetry.targetVerifiedTokenCount
                == telemetry.proposedDraftTokens + telemetry.roundCount,
            telemetry.emittedTokenCount == 128
        else {
            throw invalidPair(pairIndex, "draft telemetry")
        }
    }

    private static func validatePhase(
        _ phase: QwenMTPCorpusMTPPhaseAttribution,
        telemetry: QwenMTPCorpusMTPTelemetry,
        timing: QwenMTPCorpusTiming,
        pairIndex: Int
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
        guard seconds.allSatisfy({ $0.isFinite && $0 >= 0 }),
            phase.targetPrefillCount == 1,
            phase.drafterPromptPrimingCount == 1,
            phase.draftBlockCount == telemetry.draftModelCallCount,
            phase.targetVerificationCount == telemetry.targetModelCallCount,
            phase.targetTailCount <= telemetry.emittedTokenCount,
            phase.hybridRewindReplayCount <= telemetry.targetModelCallCount + 1,
            phase.finalizationCount == 1,
            phase.cacheFingerprintCount == 1,
            phase.promptSeconds <= timing.promptSeconds + 0.005,
            phase.generationSeconds <= timing.generationSeconds + 0.005,
            phase.promptSeconds + phase.generationSeconds <= timing.wallSeconds + 0.005,
            let preparation = phase.targetPromptPreparation
        else {
            throw invalidPair(pairIndex, "phase envelope")
        }
        try validatePreparation(
            preparation,
            targetPrefillSeconds: phase.targetPrefillSeconds,
            pairIndex: pairIndex)
    }

    private static func validatePreparation(
        _ preparation: QwenMTPPromptPreparationAttribution,
        targetPrefillSeconds: Double,
        pairIndex: Int
    ) throws {
        guard preparation.promptTokenCount == requiredPromptTokenCount,
            preparation.hiddenShape.count == 3,
            preparation.hiddenShape[0] == 1,
            preparation.hiddenShape[1] == preparation.promptTokenCount,
            preparation.hiddenShape[2] > 0,
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
            throw invalidPair(pairIndex, "prompt preparation")
        }

        var hiddenElementCount = 1
        for dimension in preparation.hiddenShape {
            let (nextCount, overflow) = hiddenElementCount.multipliedReportingOverflow(
                by: dimension)
            guard !overflow, nextCount > 0 else {
                throw invalidPair(pairIndex, "hidden shape overflow")
            }
            hiddenElementCount = nextCount
        }
        guard preparation.hiddenByteCount > 0,
            preparation.hiddenByteCount.isMultiple(of: hiddenElementCount),
            [1, 2, 4, 8].contains(preparation.hiddenByteCount / hiddenElementCount)
        else {
            throw invalidPair(pairIndex, "hidden byte geometry")
        }

        var expectedOffset = 0
        for chunk in preparation.chunks {
            guard chunk.tokenOffset == expectedOffset,
                chunk.tokenCount > 0,
                chunk.targetForwardSchedulingSeconds.isFinite,
                chunk.targetForwardSchedulingSeconds >= 0
            else {
                throw invalidPair(pairIndex, "prompt chunk")
            }
            let (nextOffset, overflow) = expectedOffset.addingReportingOverflow(
                chunk.tokenCount)
            guard !overflow else {
                throw invalidPair(pairIndex, "prompt chunk overflow")
            }
            expectedOffset = nextOffset
        }
        guard expectedOffset == preparation.promptTokenCount,
            approximatelyEqual(
                preparation.attributedSeconds
                    + preparation.targetPrefillResidualSeconds,
                targetPrefillSeconds)
        else {
            throw invalidPair(pairIndex, "prompt phase envelope")
        }
    }

    private static func invalidPair(
        _ index: Int,
        _ reason: String
    ) -> QwenMTPEvaluationOrderIsolationGateError {
        .invalidPair(index: index, reason: reason)
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
}
