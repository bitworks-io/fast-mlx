import Foundation

public enum QwenMTPHiddenFirstRuntimePairRunOrder: String, Codable, Equatable, Sendable {
    case defaultThenHiddenFirst = "default-then-hidden-first"
    case hiddenFirstThenDefault = "hidden-first-then-default"
}

public struct QwenMTPHiddenFirstRuntimePairEvidence: Codable, Equatable, Sendable {
    public let pairIndex: Int
    public let warmup: Bool
    public let runOrder: QwenMTPHiddenFirstRuntimePairRunOrder
    /// The ordinary iterator construction path with no evaluation-order override.
    public let defaultRuntime: QwenMTPEvaluationOrderRunEvidence
    /// The same runtime with the explicit telemetry-gated hidden-first override.
    public let hiddenFirstRuntime: QwenMTPEvaluationOrderRunEvidence

    public init(
        pairIndex: Int,
        warmup: Bool,
        runOrder: QwenMTPHiddenFirstRuntimePairRunOrder,
        defaultRuntime: QwenMTPEvaluationOrderRunEvidence,
        hiddenFirstRuntime: QwenMTPEvaluationOrderRunEvidence
    ) {
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.runOrder = runOrder
        self.defaultRuntime = defaultRuntime
        self.hiddenFirstRuntime = hiddenFirstRuntime
    }

    public func copy(
        runOrder: QwenMTPHiddenFirstRuntimePairRunOrder? = nil,
        defaultRuntime: QwenMTPEvaluationOrderRunEvidence? = nil,
        hiddenFirstRuntime: QwenMTPEvaluationOrderRunEvidence? = nil
    ) -> Self {
        .init(
            pairIndex: pairIndex,
            warmup: warmup,
            runOrder: runOrder ?? self.runOrder,
            defaultRuntime: defaultRuntime ?? self.defaultRuntime,
            hiddenFirstRuntime: hiddenFirstRuntime ?? self.hiddenFirstRuntime)
    }
}

public struct QwenMTPHiddenFirstRuntimeEquivalencePayload: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var corpusID: String
    public var corpusContentHash: String
    public var binding: QwenMTPCorpusRuntimeBinding
    public var host: QwenMTPCorpusHostEvidence
    public var releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var pairs: [QwenMTPHiddenFirstRuntimePairEvidence]

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        binding: QwenMTPCorpusRuntimeBinding,
        host: QwenMTPCorpusHostEvidence,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        pairs: [QwenMTPHiddenFirstRuntimePairEvidence]
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

public enum QwenMTPHiddenFirstRuntimeEquivalenceGateError:
    Error, Equatable, Sendable
{
    case schemaIdentity
    case binding
    case host
    case releaseBuild
    case pairCardinality(expected: Int, actual: Int)
    case invalidPair(index: Int, reason: String)
    case unterminatedJSONL
    case invalidRecordCardinality(Int)
    case malformedJSONL
    case wrongSubcommand(String)
    case invalidProvenance(String)
    case qualificationMismatch(expected: Bool, actual: Bool)
}

/// Fail-closed promotion gate for the opt-in hidden-first runtime candidate.
///
/// This is deliberately distinct from evaluation-order diagnosis: the control
/// is produced through the iterator's ordinary default construction path,
/// while the candidate must be explicitly hidden-first and telemetry-gated.
/// Passing this gate authorizes neither serving exposure nor a default change.
public enum QwenMTPHiddenFirstRuntimeEquivalenceGate {
    public static let schemaVersion = 1
    public static let corpusID = "qwen3.5-9b-mtp-hidden-first-runtime-equivalence-v1"
    public static let corpusContentHash = QwenMTPCorpusGate.corpusContentHash
    public static let subcommand = "qwen-mtp-hidden-first-runtime"
    public static let rejectedSubcommand = "qwen-mtp-hidden-first-runtime-rejected"
    public static let droppedWarmupPairs = 2
    public static let measuredPairs = 5
    public static let requiredPromptTokenCount = 1_353
    public static let requiredAggregatePromptImprovementSeconds = 2.5
    public static let requiredMedianPromptImprovementSeconds = 0.40
    public static let pairOrders: [QwenMTPHiddenFirstRuntimePairRunOrder] = [
        .defaultThenHiddenFirst,
        .hiddenFirstThenDefault,
        .defaultThenHiddenFirst,
        .hiddenFirstThenDefault,
        .defaultThenHiddenFirst,
        .hiddenFirstThenDefault,
        .defaultThenHiddenFirst,
    ]

    public static func validate(
        _ payload: QwenMTPHiddenFirstRuntimeEquivalencePayload
    ) throws -> QwenMTPEvaluationOrderIsolationVerdict {
        guard payload.schemaVersion == schemaVersion,
            payload.corpusID == corpusID,
            payload.corpusContentHash == corpusContentHash
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.schemaIdentity
        }
        guard payload.binding == QwenMTPCorpusGate.requiredBinding else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.binding
        }
        guard !payload.host.chip.isEmpty, payload.host.ramBytes > 0,
            !payload.host.os.isEmpty
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.host
        }
        guard payload.releaseBuildRequired, payload.releaseBuildObserved else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.releaseBuild
        }
        guard payload.pairs.count == pairOrders.count else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.pairCardinality(
                expected: pairOrders.count,
                actual: payload.pairs.count)
        }

        var measuredImprovements = [Double]()
        measuredImprovements.reserveCapacity(measuredPairs)
        for (index, pair) in payload.pairs.enumerated() {
            try validatePair(pair, at: index)
            if !pair.warmup {
                let improvement = pair.defaultRuntime.timing.promptSeconds
                    - pair.hiddenFirstRuntime.timing.promptSeconds
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

    /// Validate each completed pair before a producer advances to the next run.
    public static func validatePair(
        _ pair: QwenMTPHiddenFirstRuntimePairEvidence,
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
                pair.defaultRuntime,
                expectedOrder: .cacheFirst,
                pairIndex: index)
            try QwenMTPEvaluationOrderIsolationGate.validateDiagnosticRun(
                pair.hiddenFirstRuntime,
                expectedOrder: .hiddenFirst,
                pairIndex: index)
        } catch {
            throw invalidPair(index, "runtime run: \(error)")
        }
        guard pair.defaultRuntime.exactness.scalarTokenIDsSHA256
            == pair.hiddenFirstRuntime.exactness.scalarTokenIDsSHA256,
            pair.defaultRuntime.exactness.scalarDecodedBytesSHA256
                == pair.hiddenFirstRuntime.exactness.scalarDecodedBytesSHA256,
            pair.defaultRuntime.exactness.scalarStopOutcome
                == pair.hiddenFirstRuntime.exactness.scalarStopOutcome,
            pair.defaultRuntime.exactness.scalarCacheFingerprint
                == pair.hiddenFirstRuntime.exactness.scalarCacheFingerprint
        else {
            throw invalidPair(index, "scalar control identity")
        }
    }

    public static func validateJSONL(
        _ data: Data
    ) throws -> [QwenMTPEvaluationOrderIsolationVerdict] {
        try validateJSONL(
            data,
            expectedSubcommand: subcommand,
            expectedQualified: true)
    }

    public static func validateRejectedJSONL(
        _ data: Data
    ) throws -> [QwenMTPEvaluationOrderIsolationVerdict] {
        try validateJSONL(
            data,
            expectedSubcommand: rejectedSubcommand,
            expectedQualified: false)
    }

    private static func validateJSONL(
        _ data: Data,
        expectedSubcommand: String,
        expectedQualified: Bool
    ) throws -> [QwenMTPEvaluationOrderIsolationVerdict] {
        guard data.last == 0x0a else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.unterminatedJSONL
        }
        let rows = data.split(
            separator: 0x0a,
            omittingEmptySubsequences: false).dropLast()
        guard rows.count == 1 else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidRecordCardinality(rows.count)
        }
        guard let row = rows.first, !row.isEmpty else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.malformedJSONL
        }
        let record: ResultRecord<QwenMTPHiddenFirstRuntimeEquivalencePayload>
        do {
            record = try JSONDecoder().decode(
                ResultRecord<QwenMTPHiddenFirstRuntimeEquivalencePayload>.self,
                from: Data(row))
        } catch {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.malformedJSONL
        }
        guard record.subcommand == expectedSubcommand else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .wrongSubcommand(record.subcommand)
        }
        try validateProvenance(record.provenance, payload: record.payload)
        let verdict = try validate(record.payload)
        guard verdict.qualified == expectedQualified else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError.qualificationMismatch(
                expected: expectedQualified,
                actual: verdict.qualified)
        }
        return [verdict]
    }

    private static func validateProvenance(
        _ provenance: Provenance,
        payload: QwenMTPHiddenFirstRuntimeEquivalencePayload
    ) throws {
        guard provenance.hardwareChip == payload.host.chip,
            provenance.hardwareRAMBytes == payload.host.ramBytes,
            provenance.hardwareOS == payload.host.os
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("host")
        }
        guard provenance.harnessGitSHA.range(
            of: "^[0-9a-f]{40}$",
            options: .regularExpression) != nil
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("harnessGitSHA")
        }
        guard provenance.mlxSwiftVersion == "0.31.6",
            provenance.referenceMLXVersion == nil,
            provenance.referenceMLXLMVersion
                == "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("source versions")
        }
        guard provenance.modelPath == payload.binding.targetModelID else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("model path")
        }
        guard provenance.modelConfigHash == "5a99be4477ebdac8" else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("model config hash")
        }
        guard provenance.modelCheckpointManifestHash == "db2b2480a8525194" else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("model checkpoint manifest hash")
        }
        guard provenance.modelQuant == ModelQuantInfo(bits: 4, groupSize: 64) else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("model quantization")
        }
        guard provenance.corpusId == payload.corpusID,
            provenance.corpusContentHash == payload.corpusContentHash,
            !provenance.date.isEmpty,
            !provenance.nonce.isEmpty
        else {
            throw QwenMTPHiddenFirstRuntimeEquivalenceGateError
                .invalidProvenance("record")
        }
    }

    private static func invalidPair(
        _ index: Int,
        _ reason: String
    ) -> QwenMTPHiddenFirstRuntimeEquivalenceGateError {
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
