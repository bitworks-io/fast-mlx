import Foundation

public enum QwenMTPServingLatencyRouteKind: String, Codable, Equatable, Sendable {
    case exactQwen35MTP
    case scalarFallback
}

public struct QwenMTPServingLatencyRouteEvidence: Codable, Equatable, Sendable {
    public let kind: QwenMTPServingLatencyRouteKind

    public init(kind: QwenMTPServingLatencyRouteKind) {
        self.kind = kind
    }
}

public struct QwenMTPServingLatencyFallbackEvidence: Codable, Equatable, Sendable {
    public let scalarFallbackStartCount: Int

    public init(scalarFallbackStartCount: Int) {
        self.scalarFallbackStartCount = scalarFallbackStartCount
    }
}

public struct QwenMTPServingLatencyRequestEvidence: Codable, Equatable, Sendable {
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let seed: Int64?
    public let toolsEmpty: Bool
    public let penaltiesDisabled: Bool

    public init(
        temperature: Double?,
        topP: Double?,
        topK: Int?,
        minP: Double?,
        seed: Int64?,
        toolsEmpty: Bool,
        penaltiesDisabled: Bool
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.toolsEmpty = toolsEmpty
        self.penaltiesDisabled = penaltiesDisabled
    }

    public var greedy: Bool {
        guard let temperature else { return true }
        return temperature == 0
    }

    public var samplingKnobsUnset: Bool {
        topP == nil && topK == nil && minP == nil && seed == nil
    }
}

public struct QwenMTPServingLatencyUsageEvidence: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    public init(promptTokens: Int, completionTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }
}

public enum QwenMTPServingLatencyTokenObservationMode: String, Codable, Equatable, Sendable {
    /// The serving contract exposes decoded chunks, not raw token IDs. The recorded
    /// MTP token digest is therefore a tokenizer round-trip of the decoded bytes and
    /// is admitted only when it matches the directly observed scalar token IDs.
    case decodedRoundTrip
}

public struct QwenMTPServingLatencyLowerLevelProof: Codable, Equatable, Sendable {
    public let acceptedCorpusSubcommand: String
    public let acceptedCorpusSchemaVersion: Int
    public let acceptedCorpusID: String
    public let acceptedCorpusContentHash: String
    public let acceptedCorpusBinding: QwenMTPCorpusRuntimeBinding
    public let acceptedCorpusJSONLSHA256: String
    public let acceptedCorpusHarnessGitSHA: String

    public init(
        acceptedCorpusSubcommand: String,
        acceptedCorpusSchemaVersion: Int,
        acceptedCorpusID: String,
        acceptedCorpusContentHash: String,
        acceptedCorpusBinding: QwenMTPCorpusRuntimeBinding,
        acceptedCorpusJSONLSHA256: String,
        acceptedCorpusHarnessGitSHA: String
    ) {
        self.acceptedCorpusSubcommand = acceptedCorpusSubcommand
        self.acceptedCorpusSchemaVersion = acceptedCorpusSchemaVersion
        self.acceptedCorpusID = acceptedCorpusID
        self.acceptedCorpusContentHash = acceptedCorpusContentHash
        self.acceptedCorpusBinding = acceptedCorpusBinding
        self.acceptedCorpusJSONLSHA256 = acceptedCorpusJSONLSHA256
        self.acceptedCorpusHarnessGitSHA = acceptedCorpusHarnessGitSHA
    }
}

public struct QwenMTPServingLatencyExactnessEvidence: Codable, Equatable, Sendable {
    public let tokenObservationMode: QwenMTPServingLatencyTokenObservationMode
    public let scalarDirectTokenCount: Int
    public let mtpUsageCompletionTokenCount: Int
    public let scalarDirectTokenIDsSHA256: String
    public let mtpDecodedRoundTripTokenIDsSHA256: String
    public let scalarDecodedBytesSHA256: String
    public let mtpDecodedBytesSHA256: String
    public let scalarStopOutcome: QwenMTPCorpusStopOutcome
    public let mtpStopOutcome: QwenMTPCorpusStopOutcome
    public let scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint
    public let firstCacheMismatch: String?

    public init(
        tokenObservationMode: QwenMTPServingLatencyTokenObservationMode,
        scalarDirectTokenCount: Int,
        mtpUsageCompletionTokenCount: Int,
        scalarDirectTokenIDsSHA256: String,
        mtpDecodedRoundTripTokenIDsSHA256: String,
        scalarDecodedBytesSHA256: String,
        mtpDecodedBytesSHA256: String,
        scalarStopOutcome: QwenMTPCorpusStopOutcome,
        mtpStopOutcome: QwenMTPCorpusStopOutcome,
        scalarCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        mtpCacheFingerprint: QwenMTPCorpusCacheFingerprint,
        firstCacheMismatch: String?
    ) {
        self.tokenObservationMode = tokenObservationMode
        self.scalarDirectTokenCount = scalarDirectTokenCount
        self.mtpUsageCompletionTokenCount = mtpUsageCompletionTokenCount
        self.scalarDirectTokenIDsSHA256 = scalarDirectTokenIDsSHA256
        self.mtpDecodedRoundTripTokenIDsSHA256 = mtpDecodedRoundTripTokenIDsSHA256
        self.scalarDecodedBytesSHA256 = scalarDecodedBytesSHA256
        self.mtpDecodedBytesSHA256 = mtpDecodedBytesSHA256
        self.scalarStopOutcome = scalarStopOutcome
        self.mtpStopOutcome = mtpStopOutcome
        self.scalarCacheFingerprint = scalarCacheFingerprint
        self.mtpCacheFingerprint = mtpCacheFingerprint
        self.firstCacheMismatch = firstCacheMismatch
    }
}

public struct QwenMTPServingLatencySample: Codable, Equatable, Sendable {
    public let caseID: String
    public let pairIndex: Int
    public let warmup: Bool
    public let order: QwenMTPCorpusRunOrder
    public var route: QwenMTPServingLatencyRouteEvidence
    public var fallback: QwenMTPServingLatencyFallbackEvidence
    public var exactness: QwenMTPServingLatencyExactnessEvidence
    public let scalarUsage: QwenMTPServingLatencyUsageEvidence
    public let mtpUsage: QwenMTPServingLatencyUsageEvidence
    public let scalarE2ESeconds: Double
    public let mtpE2ESeconds: Double
    public let scalarTokensPerSecond: Double
    public let mtpTokensPerSecond: Double
    public let e2eRatio: Double
    public var mtpTelemetry: QwenMTPCorpusMTPTelemetry
    public var passthroughReason: String?

    public init(
        caseID: String,
        pairIndex: Int,
        warmup: Bool,
        order: QwenMTPCorpusRunOrder,
        route: QwenMTPServingLatencyRouteEvidence,
        fallback: QwenMTPServingLatencyFallbackEvidence,
        exactness: QwenMTPServingLatencyExactnessEvidence,
        scalarUsage: QwenMTPServingLatencyUsageEvidence,
        mtpUsage: QwenMTPServingLatencyUsageEvidence,
        scalarE2ESeconds: Double,
        mtpE2ESeconds: Double,
        scalarTokensPerSecond: Double,
        mtpTokensPerSecond: Double,
        e2eRatio: Double,
        mtpTelemetry: QwenMTPCorpusMTPTelemetry,
        passthroughReason: String?
    ) {
        self.caseID = caseID
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.order = order
        self.route = route
        self.fallback = fallback
        self.exactness = exactness
        self.scalarUsage = scalarUsage
        self.mtpUsage = mtpUsage
        self.scalarE2ESeconds = scalarE2ESeconds
        self.mtpE2ESeconds = mtpE2ESeconds
        self.scalarTokensPerSecond = scalarTokensPerSecond
        self.mtpTokensPerSecond = mtpTokensPerSecond
        self.e2eRatio = e2eRatio
        self.mtpTelemetry = mtpTelemetry
        self.passthroughReason = passthroughReason
    }
}

public struct QwenMTPServingLatencyVerdict: Codable, Equatable, Sendable {
    public let qualified: Bool
    public let aggregatePairedMedian: Double
    public let chronologicalFirstHalfMedian: Double
    public let chronologicalSecondHalfMedian: Double
    public let perPromptMedians: [String: Double]
    public let perPromptMedianBelowFloorCount: Int
    public let aggregateThreshold: Double
    public let chronologicalHalfThreshold: Double
    public let perPromptFloor: Double

    public init(
        qualified: Bool,
        aggregatePairedMedian: Double,
        chronologicalFirstHalfMedian: Double,
        chronologicalSecondHalfMedian: Double,
        perPromptMedians: [String: Double],
        perPromptMedianBelowFloorCount: Int,
        aggregateThreshold: Double,
        chronologicalHalfThreshold: Double,
        perPromptFloor: Double
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
    }
}

public struct QwenMTPServingLatencyCheckpointIdentity: Codable, Equatable, Sendable {
    public let targetCheckpointContentSHA256: String
    public let drafterCheckpointContentSHA256: String

    public init(
        targetCheckpointContentSHA256: String,
        drafterCheckpointContentSHA256: String
    ) {
        self.targetCheckpointContentSHA256 = targetCheckpointContentSHA256
        self.drafterCheckpointContentSHA256 = drafterCheckpointContentSHA256
    }
}

public struct QwenMTPServingLatencyEvidence: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let corpusID: String
    public let corpusContentHash: String
    public let binding: QwenMTPCorpusRuntimeBinding
    public let lowerLevelProof: QwenMTPServingLatencyLowerLevelProof
    public let checkpointIdentity: QwenMTPServingLatencyCheckpointIdentity
    public let measurementClass: String
    public let host: QwenMTPCorpusHostEvidence
    public let profilePlan: QwenMTPCorpusProfilePlan
    public let releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var request: QwenMTPServingLatencyRequestEvidence
    public var samples: [QwenMTPServingLatencySample]
    public let verdict: QwenMTPServingLatencyVerdict?

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        binding: QwenMTPCorpusRuntimeBinding,
        lowerLevelProof: QwenMTPServingLatencyLowerLevelProof,
        checkpointIdentity: QwenMTPServingLatencyCheckpointIdentity,
        measurementClass: String,
        host: QwenMTPCorpusHostEvidence,
        profilePlan: QwenMTPCorpusProfilePlan,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        request: QwenMTPServingLatencyRequestEvidence,
        samples: [QwenMTPServingLatencySample],
        verdict: QwenMTPServingLatencyVerdict?
    ) {
        self.schemaVersion = schemaVersion
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
        self.binding = binding
        self.lowerLevelProof = lowerLevelProof
        self.checkpointIdentity = checkpointIdentity
        self.measurementClass = measurementClass
        self.host = host
        self.profilePlan = profilePlan
        self.releaseBuildRequired = releaseBuildRequired
        self.releaseBuildObserved = releaseBuildObserved
        self.request = request
        self.samples = samples
        self.verdict = verdict
    }
}

public enum QwenMTPServingLatencyGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case schemaVersionMismatch(Int)
    case corpusIdentityMismatch
    case invalidBinding
    case invalidLowerLevelProof
    case invalidCheckpointIdentity
    case invalidMeasurementClass(String)
    case invalidHost
    case invalidProfilePlan
    case releaseBuildRequired
    case invalidRequest(String)
    case invalidSampleCardinality(expected: Int, actual: Int)
    case invalidSample(index: Int, reason: String)
    case invalidRoute(index: Int, kind: String)
    case scalarFallbackUsed(index: Int, startCount: Int)
    case unqualifiedPerformance
    case verdictMismatch
    case malformedJSONL(line: Int)
    case unterminatedJSONL
    case invalidRecordCardinality(Int)
    case wrongSubcommand(String)
    case invalidProvenance(String)

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let value):
            return "schemaVersion mismatch: \(value)"
        case .corpusIdentityMismatch:
            return "corpus identity mismatch"
        case .invalidBinding:
            return "invalid runtime binding"
        case .invalidLowerLevelProof:
            return "invalid accepted lower-level corpus proof"
        case .invalidCheckpointIdentity:
            return "invalid target/drafter checkpoint identity"
        case .invalidMeasurementClass(let value):
            return "invalid measurement class: \(value)"
        case .invalidHost:
            return "invalid host evidence"
        case .invalidProfilePlan:
            return "invalid profile plan"
        case .releaseBuildRequired:
            return "serving latency evidence requires a Release build"
        case .invalidRequest(let field):
            return "invalid serving request evidence: \(field)"
        case .invalidSampleCardinality(let expected, let actual):
            return "invalid sample cardinality: expected \(expected), got \(actual)"
        case .invalidSample(let index, let reason):
            return "invalid sample \(index): \(reason)"
        case .invalidRoute(let index, let kind):
            return "invalid route at sample \(index): \(kind)"
        case .scalarFallbackUsed(let index, let startCount):
            return "scalar fallback used at sample \(index): \(startCount)"
        case .unqualifiedPerformance:
            return "unqualified serving latency performance"
        case .verdictMismatch:
            return "stored verdict does not match recomputed verdict"
        case .malformedJSONL(let line):
            return "malformed JSONL at line \(line)"
        case .unterminatedJSONL:
            return "unterminated JSONL"
        case .invalidRecordCardinality(let count):
            return "invalid JSONL record cardinality: \(count)"
        case .wrongSubcommand(let subcommand):
            return "wrong subcommand: \(subcommand)"
        case .invalidProvenance(let field):
            return "invalid provenance: \(field)"
        }
    }
}

private struct QwenMTPServingLatencySchemaProbe: Codable, Sendable {
    let schemaVersion: Int
}

public enum QwenMTPServingLatencyGate {
    public static let schemaVersion = 1
    public static let subcommand = "qwen-mtp-serving-latency"
    public static let rejectedSubcommand = "qwen-mtp-serving-latency-rejected"
    public static let measurementClass = "consumer-24gib"
    public static let requiredRAMBytes: UInt64 = 25_769_803_776
    public static let aggregateThreshold = 1.08
    public static let chronologicalHalfThreshold = 1.05
    public static let perPromptFloor = 0.97
    public static let requiredLowerLevelProof = QwenMTPServingLatencyLowerLevelProof(
        acceptedCorpusSubcommand: "qwen-mtp-corpus",
        acceptedCorpusSchemaVersion: QwenMTPCorpusGate.schemaVersion,
        acceptedCorpusID: QwenMTPCorpusGate.corpusID,
        acceptedCorpusContentHash: QwenMTPCorpusGate.corpusContentHash,
        acceptedCorpusBinding: QwenMTPCorpusGate.requiredBinding,
        acceptedCorpusJSONLSHA256:
            "19308a47b23c61fab593775cd790ebf8c8a3fa18309c26b78a4d62ab469704e3",
        acceptedCorpusHarnessGitSHA:
            "fb1a66a9713081494927e5e7d7f819ca062b5e97")
    public static let requiredCheckpointIdentity = QwenMTPServingLatencyCheckpointIdentity(
        targetCheckpointContentSHA256:
            "fdb9fe71c724f81ea0945804e781a7eb9db17c0a2564c9e828e6f2f7e347d834",
        drafterCheckpointContentSHA256:
            "0fe02a87bd145239bb54c64267674ac176a79329ec003df21c9704326649326a")

    public static func validate(
        _ evidence: QwenMTPServingLatencyEvidence
    ) throws -> QwenMTPServingLatencyVerdict {
        let verdict = try evaluateCandidate(evidence)
        guard let stored = evidence.verdict else {
            throw QwenMTPServingLatencyGateError.verdictMismatch
        }
        guard stored == verdict else {
            throw QwenMTPServingLatencyGateError.verdictMismatch
        }
        return verdict
    }

    public static func evaluateCandidate(
        _ evidence: QwenMTPServingLatencyEvidence
    ) throws -> QwenMTPServingLatencyVerdict {
        guard evidence.schemaVersion == schemaVersion else {
            throw QwenMTPServingLatencyGateError.schemaVersionMismatch(
                evidence.schemaVersion)
        }
        guard evidence.corpusID == QwenMTPCorpusGate.corpusID,
            evidence.corpusContentHash == QwenMTPCorpusGate.corpusContentHash
        else {
            throw QwenMTPServingLatencyGateError.corpusIdentityMismatch
        }
        guard evidence.binding == QwenMTPCorpusGate.requiredBinding else {
            throw QwenMTPServingLatencyGateError.invalidBinding
        }
        guard evidence.lowerLevelProof == requiredLowerLevelProof else {
            throw QwenMTPServingLatencyGateError.invalidLowerLevelProof
        }
        guard evidence.checkpointIdentity == requiredCheckpointIdentity else {
            throw QwenMTPServingLatencyGateError.invalidCheckpointIdentity
        }
        guard evidence.measurementClass == measurementClass else {
            throw QwenMTPServingLatencyGateError.invalidMeasurementClass(
                evidence.measurementClass)
        }
        guard evidence.host.ramBytes == requiredRAMBytes,
            evidence.host.chip.hasPrefix("Apple"),
            !evidence.host.os.isEmpty
        else {
            throw QwenMTPServingLatencyGateError.invalidHost
        }
        guard evidence.profilePlan == QwenMTPCorpusGate.profilePlan else {
            throw QwenMTPServingLatencyGateError.invalidProfilePlan
        }
        guard evidence.releaseBuildRequired, evidence.releaseBuildObserved else {
            throw QwenMTPServingLatencyGateError.releaseBuildRequired
        }
        try validateRequest(evidence.request)

        let verdict = try validateSamples(evidence.samples)
        guard verdict.qualified else {
            throw QwenMTPServingLatencyGateError.unqualifiedPerformance
        }
        return verdict
    }

    public static func validateJSONL(_ data: Data) throws -> [QwenMTPServingLatencyVerdict] {
        guard data.last == 0x0a else {
            throw QwenMTPServingLatencyGateError.unterminatedJSONL
        }
        let rows = data.split(separator: 0x0a, omittingEmptySubsequences: false).dropLast()
        guard rows.count == 1 else {
            throw QwenMTPServingLatencyGateError.invalidRecordCardinality(rows.count)
        }
        var verdicts: [QwenMTPServingLatencyVerdict] = []
        verdicts.reserveCapacity(rows.count)
        let decoder = JSONDecoder()
        for (index, row) in rows.enumerated() {
            guard !row.isEmpty else {
                throw QwenMTPServingLatencyGateError.malformedJSONL(line: index + 1)
            }
            let rowData = Data(row)
            let probe: ResultRecord<QwenMTPServingLatencySchemaProbe>
            do {
                probe = try decoder.decode(
                    ResultRecord<QwenMTPServingLatencySchemaProbe>.self,
                    from: rowData)
            } catch {
                throw QwenMTPServingLatencyGateError.malformedJSONL(line: index + 1)
            }
            guard probe.payload.schemaVersion == schemaVersion else {
                throw QwenMTPServingLatencyGateError.schemaVersionMismatch(
                    probe.payload.schemaVersion)
            }
            let record: ResultRecord<QwenMTPServingLatencyEvidence>
            do {
                record = try decoder.decode(
                    ResultRecord<QwenMTPServingLatencyEvidence>.self,
                    from: rowData)
            } catch {
                throw QwenMTPServingLatencyGateError.malformedJSONL(line: index + 1)
            }
            guard record.subcommand == subcommand else {
                throw QwenMTPServingLatencyGateError.wrongSubcommand(
                    record.subcommand)
            }
            try validateProvenance(record.provenance, evidence: record.payload)
            verdicts.append(try validate(record.payload))
        }
        return verdicts
    }

    private static func validateRequest(
        _ request: QwenMTPServingLatencyRequestEvidence
    ) throws {
        guard request.greedy else {
            throw QwenMTPServingLatencyGateError.invalidRequest("greedy")
        }
        guard request.samplingKnobsUnset else {
            throw QwenMTPServingLatencyGateError.invalidRequest("sampling")
        }
        guard request.toolsEmpty else {
            throw QwenMTPServingLatencyGateError.invalidRequest("tools")
        }
        guard request.penaltiesDisabled else {
            throw QwenMTPServingLatencyGateError.invalidRequest("penalties")
        }
    }

    private static func validateProvenance(
        _ provenance: Provenance,
        evidence: QwenMTPServingLatencyEvidence
    ) throws {
        guard provenance.hardwareChip == evidence.host.chip,
            provenance.hardwareRAMBytes == evidence.host.ramBytes,
            provenance.hardwareOS == evidence.host.os
        else {
            throw QwenMTPServingLatencyGateError.invalidProvenance("host")
        }
        guard isLowerHex(provenance.harnessGitSHA, count: 40) else {
            throw QwenMTPServingLatencyGateError.invalidProvenance("harnessGitSHA")
        }
        guard provenance.mlxSwiftVersion == "0.31.6",
            provenance.referenceMLXVersion == nil,
            provenance.referenceMLXLMVersion
                == "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"
        else {
            throw QwenMTPServingLatencyGateError.invalidProvenance("source versions")
        }
        guard provenance.modelPath == evidence.binding.targetModelID,
            provenance.modelConfigHash == "5a99be4477ebdac8",
            provenance.modelCheckpointManifestHash == "db2b2480a8525194",
            provenance.modelQuant == ModelQuantInfo(bits: 4, groupSize: 64)
        else {
            throw QwenMTPServingLatencyGateError.invalidProvenance("model")
        }
        guard provenance.corpusId == evidence.corpusID,
            provenance.corpusContentHash == evidence.corpusContentHash,
            !provenance.date.isEmpty,
            !provenance.nonce.isEmpty
        else {
            throw QwenMTPServingLatencyGateError.invalidProvenance("record")
        }
    }

    private static func validateSamples(
        _ samples: [QwenMTPServingLatencySample]
    ) throws -> QwenMTPServingLatencyVerdict {
        let plan = QwenMTPCorpusGate.profilePlan
        let expectedCount = plan.caseIDs.count * plan.totalPairsPerCase
        guard samples.count == expectedCount else {
            throw QwenMTPServingLatencyGateError.invalidSampleCardinality(
                expected: expectedCount,
                actual: samples.count)
        }

        var sampleIndex = 0
        var measuredRatios: [Double] = []
        var measuredByCase: [String: [Double]] = [:]
        var measuredByPairIndex: [Int: [Double]] = [:]

        for caseID in plan.caseIDs {
            for pairIndex in 0..<plan.totalPairsPerCase {
                let sample = samples[sampleIndex]
                try validateSample(
                    sample,
                    sampleIndex: sampleIndex,
                    expectedCaseID: caseID,
                    expectedPairIndex: pairIndex)
                if !sample.warmup {
                    let ratio = sample.scalarE2ESeconds / sample.mtpE2ESeconds
                    measuredRatios.append(ratio)
                    measuredByCase[caseID, default: []].append(ratio)
                    measuredByPairIndex[pairIndex - plan.droppedWarmupPairs, default: []]
                        .append(ratio)
                }
                sampleIndex += 1
            }
        }

        let aggregateMedian = median(measuredRatios)
        let firstHalfMedian = median([0, 1].flatMap { measuredByPairIndex[$0] ?? [] })
        let secondHalfMedian = median([2, 3, 4].flatMap { measuredByPairIndex[$0] ?? [] })
        let perPromptMedians = plan.caseIDs.reduce(into: [String: Double]()) {
            output,
            caseID in
            output[caseID] = median(measuredByCase[caseID] ?? [])
        }
        let weakPromptCount = perPromptMedians.values.filter {
            $0 < perPromptFloor
        }.count
        let qualified = aggregateMedian >= aggregateThreshold
            && firstHalfMedian >= chronologicalHalfThreshold
            && secondHalfMedian >= chronologicalHalfThreshold
            && weakPromptCount <= 1

        return QwenMTPServingLatencyVerdict(
            qualified: qualified,
            aggregatePairedMedian: aggregateMedian,
            chronologicalFirstHalfMedian: firstHalfMedian,
            chronologicalSecondHalfMedian: secondHalfMedian,
            perPromptMedians: perPromptMedians,
            perPromptMedianBelowFloorCount: weakPromptCount,
            aggregateThreshold: aggregateThreshold,
            chronologicalHalfThreshold: chronologicalHalfThreshold,
            perPromptFloor: perPromptFloor)
    }

    private static func validateSample(
        _ sample: QwenMTPServingLatencySample,
        sampleIndex: Int,
        expectedCaseID: String,
        expectedPairIndex: Int
    ) throws {
        let plan = QwenMTPCorpusGate.profilePlan
        guard sample.caseID == expectedCaseID else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "caseID")
        }
        guard sample.pairIndex == expectedPairIndex else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "pairIndex")
        }
        guard sample.warmup == (expectedPairIndex < plan.droppedWarmupPairs) else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "warmup")
        }
        guard sample.order == plan.orders[expectedPairIndex] else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "order")
        }
        guard sample.route.kind == .exactQwen35MTP else {
            throw QwenMTPServingLatencyGateError.invalidRoute(
                index: sampleIndex,
                kind: sample.route.kind.rawValue)
        }
        guard sample.fallback.scalarFallbackStartCount == 0 else {
            throw QwenMTPServingLatencyGateError.scalarFallbackUsed(
                index: sampleIndex,
                startCount: sample.fallback.scalarFallbackStartCount)
        }
        do {
            try validateExactness(sample.exactness, context: sample.caseID)
            try validateUsage(sample)
            try validateTelemetry(
                sample.mtpTelemetry,
                emittedTokenCount: sample.exactness.mtpUsageCompletionTokenCount,
                context: sample.caseID)
        } catch {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: String(describing: error))
        }
        guard sample.mtpTelemetry.proposedDraftTokens > 0,
            sample.mtpTelemetry.acceptedDraftTokens > 0,
            sample.passthroughReason == nil
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "draft activity")
        }
        guard sample.scalarE2ESeconds.isFinite,
            sample.scalarE2ESeconds > 0,
            sample.mtpE2ESeconds.isFinite,
            sample.mtpE2ESeconds > 0,
            sample.scalarTokensPerSecond.isFinite,
            sample.scalarTokensPerSecond > 0,
            sample.mtpTokensPerSecond.isFinite,
            sample.mtpTokensPerSecond > 0,
            sample.e2eRatio.isFinite,
            sample.e2eRatio > 0
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "timing")
        }

        let scalarTPS =
            Double(sample.exactness.scalarDirectTokenCount) / sample.scalarE2ESeconds
        let mtpTPS =
            Double(sample.exactness.mtpUsageCompletionTokenCount) / sample.mtpE2ESeconds
        let ratio = sample.scalarE2ESeconds / sample.mtpE2ESeconds
        guard approximatelyEqual(sample.scalarTokensPerSecond, scalarTPS),
            approximatelyEqual(sample.mtpTokensPerSecond, mtpTPS),
            approximatelyEqual(sample.e2eRatio, ratio)
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: sampleIndex,
                reason: "reported performance")
        }
    }

    private static func validateExactness(
        _ exactness: QwenMTPServingLatencyExactnessEvidence,
        context: String
    ) throws {
        guard exactness.tokenObservationMode == .decodedRoundTrip,
            exactness.scalarDirectTokenCount == exactness.mtpUsageCompletionTokenCount,
            exactness.scalarDirectTokenCount > 0
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) token counts")
        }
        guard isLowerHexSHA256(exactness.scalarDirectTokenIDsSHA256),
            isLowerHexSHA256(exactness.mtpDecodedRoundTripTokenIDsSHA256),
            exactness.scalarDirectTokenIDsSHA256
                == exactness.mtpDecodedRoundTripTokenIDsSHA256
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) token ID digest")
        }
        guard isLowerHexSHA256(exactness.scalarDecodedBytesSHA256),
            isLowerHexSHA256(exactness.mtpDecodedBytesSHA256),
            exactness.scalarDecodedBytesSHA256 == exactness.mtpDecodedBytesSHA256
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) decoded bytes")
        }
        guard exactness.scalarStopOutcome == .length,
            exactness.mtpStopOutcome == .length
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) incomplete stop outcome")
        }
        try validateCacheFingerprint(
            exactness.scalarCacheFingerprint,
            context: "\(context) scalar cache")
        try validateCacheFingerprint(
            exactness.mtpCacheFingerprint,
            context: "\(context) mtp cache")
        guard exactness.scalarCacheFingerprint == exactness.mtpCacheFingerprint,
            exactness.firstCacheMismatch == nil
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) cache fingerprint")
        }
    }

    private static func validateUsage(
        _ sample: QwenMTPServingLatencySample
    ) throws {
        guard sample.scalarUsage == sample.mtpUsage,
            sample.scalarUsage.promptTokens > 0,
            sample.scalarUsage.completionTokens
                == sample.exactness.scalarDirectTokenCount,
            sample.scalarUsage.totalTokens
                == sample.scalarUsage.promptTokens + sample.scalarUsage.completionTokens
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(sample.caseID) usage")
        }
    }

    private static func validateCacheFingerprint(
        _ fingerprint: QwenMTPCorpusCacheFingerprint,
        context: String
    ) throws {
        guard isLowerHexSHA256(fingerprint.digest), !fingerprint.entries.isEmpty else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) aggregate digest")
        }
        for (layerOrdinal, layer) in fingerprint.entries.enumerated() {
            guard layer.layerIndex == layerOrdinal,
                !layer.cacheType.isEmpty,
                layer.offset >= 0,
                isLowerHexSHA256(layer.metaStateSHA256),
                layer.stateCount == layer.states.count,
                !layer.states.isEmpty
            else {
                throw QwenMTPServingLatencyGateError.invalidSample(
                    index: -1,
                    reason: "\(context) layer metadata")
            }
            for (stateOrdinal, state) in layer.states.enumerated() {
                guard state.stateIndex == stateOrdinal,
                    !state.shape.isEmpty,
                    state.shape.allSatisfy({ $0 >= 0 }),
                    !state.dtype.isEmpty,
                    state.byteCount >= 0,
                    isLowerHexSHA256(state.sha256)
                else {
                    throw QwenMTPServingLatencyGateError.invalidSample(
                        index: -1,
                        reason: "\(context) state metadata")
                }
            }
        }
    }

    private static func validateTelemetry(
        _ telemetry: QwenMTPCorpusMTPTelemetry,
        emittedTokenCount: Int,
        context: String
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
        guard counts.allSatisfy({ $0 >= 0 }) else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) telemetry negative")
        }
        guard telemetry.emittedTokenCount == emittedTokenCount,
            telemetry.acceptedDraftTokens <= telemetry.proposedDraftTokens,
            telemetry.rejectedDraftTokens
                == telemetry.proposedDraftTokens - telemetry.acceptedDraftTokens,
            telemetry.targetModelCallCount == telemetry.roundCount,
            telemetry.draftModelCallCount == telemetry.roundCount,
            telemetry.targetVerifiedTokenCount
                == telemetry.proposedDraftTokens + telemetry.roundCount
        else {
            throw QwenMTPServingLatencyGateError.invalidSample(
                index: -1,
                reason: "\(context) telemetry coherence")
        }
    }

    private static func isLowerHexSHA256(_ value: String) -> Bool {
        isLowerHex(value, count: 64)
    }

    private static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.utf8.allSatisfy {
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
}
