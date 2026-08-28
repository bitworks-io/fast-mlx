import Foundation

public enum QwenMTPCombinedEvaluationPairRunOrder: String, Codable, Equatable, Sendable {
    case cacheFirstThenCombined = "cache-first-then-combined"
    case combinedThenCacheFirst = "combined-then-cache-first"
}

public struct QwenMTPCombinedEvaluationPairEvidence: Codable, Equatable, Sendable {
    public let pairIndex: Int
    public let warmup: Bool
    public let runOrder: QwenMTPCombinedEvaluationPairRunOrder
    public let cacheFirst: QwenMTPEvaluationOrderRunEvidence
    public let combined: QwenMTPEvaluationOrderRunEvidence

    public init(
        pairIndex: Int,
        warmup: Bool,
        runOrder: QwenMTPCombinedEvaluationPairRunOrder,
        cacheFirst: QwenMTPEvaluationOrderRunEvidence,
        combined: QwenMTPEvaluationOrderRunEvidence
    ) {
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.runOrder = runOrder
        self.cacheFirst = cacheFirst
        self.combined = combined
    }

    public func copy(
        runOrder: QwenMTPCombinedEvaluationPairRunOrder? = nil,
        cacheFirst: QwenMTPEvaluationOrderRunEvidence? = nil,
        combined: QwenMTPEvaluationOrderRunEvidence? = nil
    ) -> Self {
        .init(
            pairIndex: pairIndex,
            warmup: warmup,
            runOrder: runOrder ?? self.runOrder,
            cacheFirst: cacheFirst ?? self.cacheFirst,
            combined: combined ?? self.combined)
    }
}

public struct QwenMTPCombinedEvaluationIsolationPayload: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var corpusID: String
    public var corpusContentHash: String
    public var binding: QwenMTPCorpusRuntimeBinding
    public var host: QwenMTPCorpusHostEvidence
    public var releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var pairs: [QwenMTPCombinedEvaluationPairEvidence]

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        binding: QwenMTPCorpusRuntimeBinding,
        host: QwenMTPCorpusHostEvidence,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        pairs: [QwenMTPCombinedEvaluationPairEvidence]
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

public enum QwenMTPCombinedEvaluationIsolationGateError: Error, Equatable, Sendable {
    case schemaIdentity
    case binding
    case host
    case releaseBuild
    case pairCardinality(expected: Int, actual: Int)
    case invalidPair(index: Int, reason: String)
}

/// Fail-closed diagnostic gate for one joint `eval(cache, hidden)` wait.
/// This is deliberately separate from the accepted cache/hidden-order schema.
public enum QwenMTPCombinedEvaluationIsolationGate {
    public static let schemaVersion = 1
    public static let corpusID = "qwen3.5-9b-mtp-combined-eval-isolation-v1"
    public static let corpusContentHash = QwenMTPCorpusGate.corpusContentHash
    public static let droppedWarmupPairs = 2
    public static let measuredPairs = 5
    public static let requiredPromptTokenCount = 1_353
    public static let requiredAggregatePromptImprovementSeconds = 2.5
    public static let requiredMedianPromptImprovementSeconds = 0.40
    public static let pairOrders: [QwenMTPCombinedEvaluationPairRunOrder] = [
        .cacheFirstThenCombined,
        .combinedThenCacheFirst,
        .cacheFirstThenCombined,
        .combinedThenCacheFirst,
        .cacheFirstThenCombined,
        .combinedThenCacheFirst,
        .cacheFirstThenCombined,
    ]

    public static func validate(
        _ payload: QwenMTPCombinedEvaluationIsolationPayload
    ) throws -> QwenMTPEvaluationOrderIsolationVerdict {
        guard payload.schemaVersion == schemaVersion,
            payload.corpusID == corpusID,
            payload.corpusContentHash == corpusContentHash
        else {
            throw QwenMTPCombinedEvaluationIsolationGateError.schemaIdentity
        }
        guard payload.binding == QwenMTPCorpusGate.requiredBinding else {
            throw QwenMTPCombinedEvaluationIsolationGateError.binding
        }
        guard !payload.host.chip.isEmpty, payload.host.ramBytes > 0,
            !payload.host.os.isEmpty
        else {
            throw QwenMTPCombinedEvaluationIsolationGateError.host
        }
        guard payload.releaseBuildRequired, payload.releaseBuildObserved else {
            throw QwenMTPCombinedEvaluationIsolationGateError.releaseBuild
        }
        guard payload.pairs.count == pairOrders.count else {
            throw QwenMTPCombinedEvaluationIsolationGateError.pairCardinality(
                expected: pairOrders.count,
                actual: payload.pairs.count)
        }

        var measuredImprovements = [Double]()
        measuredImprovements.reserveCapacity(measuredPairs)
        for (index, pair) in payload.pairs.enumerated() {
            try validatePair(pair, at: index)
            if !pair.warmup {
                let improvement = pair.cacheFirst.timing.promptSeconds
                    - pair.combined.timing.promptSeconds
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

    public static func validatePair(
        _ pair: QwenMTPCombinedEvaluationPairEvidence,
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
        do {
            try QwenMTPEvaluationOrderIsolationGate.validateDiagnosticRun(
                pair.cacheFirst,
                expectedOrder: .cacheFirst,
                pairIndex: index)
            try QwenMTPEvaluationOrderIsolationGate.validateDiagnosticRun(
                pair.combined,
                expectedOrder: .combined,
                pairIndex: index)
        } catch {
            throw invalidPair(index, "diagnostic run: \(error)")
        }
        guard let combinedPreparation = pair.combined.phaseAttribution
            .targetPromptPreparation,
            combinedPreparation.hiddenEvaluationSeconds == 0
        else {
            throw invalidPair(index, "combined evaluation telemetry")
        }
        guard pair.cacheFirst.exactness.scalarTokenIDsSHA256
            == pair.combined.exactness.scalarTokenIDsSHA256,
            pair.cacheFirst.exactness.scalarDecodedBytesSHA256
                == pair.combined.exactness.scalarDecodedBytesSHA256,
            pair.cacheFirst.exactness.scalarStopOutcome
                == pair.combined.exactness.scalarStopOutcome,
            pair.cacheFirst.exactness.scalarCacheFingerprint
                == pair.combined.exactness.scalarCacheFingerprint
        else {
            throw invalidPair(index, "scalar control identity")
        }
    }

    private static func invalidPair(
        _ index: Int,
        _ reason: String
    ) -> QwenMTPCombinedEvaluationIsolationGateError {
        .invalidPair(index: index, reason: reason)
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
