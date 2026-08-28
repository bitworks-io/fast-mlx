import CryptoKit
import Foundation

public struct Qwen38MTPPerformanceScorecardArtifact: Codable, Equatable, Sendable {
    public var lockSourceRevision: String
    public var targetRevision: String
    public var drafterRevision: String
    public var targetConfigSHA256: String
    public var drafterConfigSHA256: String
    public var tokenizerSHA256: String
    public var targetTensorManifestSHA256: String
    public var drafterTensorManifestSHA256: String
    public var targetQuantizationBits: Int
    public var targetQuantizationGroupSize: Int
    public var targetQuantizationMode: String
    public var drafterQuantizationBits: Int
    public var drafterQuantizationGroupSize: Int
    public var drafterQuantizationMode: String
    public var depth: Int
    public var blockSize: Int
    public var maxAcceptedDrafts: Int

    public init(
        lockSourceRevision: String,
        targetRevision: String,
        drafterRevision: String,
        targetConfigSHA256: String,
        drafterConfigSHA256: String,
        tokenizerSHA256: String,
        targetTensorManifestSHA256: String,
        drafterTensorManifestSHA256: String,
        targetQuantizationBits: Int,
        targetQuantizationGroupSize: Int,
        targetQuantizationMode: String,
        drafterQuantizationBits: Int,
        drafterQuantizationGroupSize: Int,
        drafterQuantizationMode: String,
        depth: Int,
        blockSize: Int,
        maxAcceptedDrafts: Int
    ) {
        self.lockSourceRevision = lockSourceRevision
        self.targetRevision = targetRevision
        self.drafterRevision = drafterRevision
        self.targetConfigSHA256 = targetConfigSHA256
        self.drafterConfigSHA256 = drafterConfigSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.targetTensorManifestSHA256 = targetTensorManifestSHA256
        self.drafterTensorManifestSHA256 = drafterTensorManifestSHA256
        self.targetQuantizationBits = targetQuantizationBits
        self.targetQuantizationGroupSize = targetQuantizationGroupSize
        self.targetQuantizationMode = targetQuantizationMode
        self.drafterQuantizationBits = drafterQuantizationBits
        self.drafterQuantizationGroupSize = drafterQuantizationGroupSize
        self.drafterQuantizationMode = drafterQuantizationMode
        self.depth = depth
        self.blockSize = blockSize
        self.maxAcceptedDrafts = maxAcceptedDrafts
    }
}

public struct Qwen38MTPPerformanceScorecardModel: Codable, Equatable, Sendable {
    public let label: String
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var executionDigest: String
    public var sourceDigest: String

    public init(
        label: String,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        executionDigest: String,
        sourceDigest: String
    ) {
        self.label = label
        self.artifact = artifact
        self.executionDigest = executionDigest
        self.sourceDigest = sourceDigest
    }
}

public struct Qwen38MTPPerformanceScorecardTrustedEngineIdentities: Codable, Equatable, Sendable {
    public var candidate: Qwen38MTPPerformanceScorecardModel
    public var reference: Qwen38MTPPerformanceScorecardModel

    public init(
        candidate: Qwen38MTPPerformanceScorecardModel,
        reference: Qwen38MTPPerformanceScorecardModel
    ) {
        self.candidate = candidate
        self.reference = reference
    }
}

public struct Qwen38MTPPerformanceScorecardTrustedRunIdentity: Codable, Equatable, Sendable {
    public let measurementClass: String
    public let hardwareChip: String
    public let hardwareRAMBytes: UInt64
    public let hardwareOSBuild: String
    public var hostIdentityDigest: String
    public let harnessGitSHA: String
    public let candidateMLXSwiftVersion: String
    public let referenceMLXVersion: String?
    public let referenceMLXLMVersion: String?
    public let modelLabel: String
    public let modelConfigHash: String
    public let modelCheckpointManifestHash: String
    public let modelQuant: ModelQuantInfo
    public let corpusID: String
    public let corpusContentHash: String

    public init(
        measurementClass: String,
        hardwareChip: String,
        hardwareRAMBytes: UInt64,
        hardwareOSBuild: String,
        hostIdentityDigest: String,
        harnessGitSHA: String,
        candidateMLXSwiftVersion: String,
        referenceMLXVersion: String?,
        referenceMLXLMVersion: String?,
        modelLabel: String,
        modelConfigHash: String,
        modelCheckpointManifestHash: String,
        modelQuant: ModelQuantInfo,
        corpusID: String,
        corpusContentHash: String
    ) {
        self.measurementClass = measurementClass
        self.hardwareChip = hardwareChip
        self.hardwareRAMBytes = hardwareRAMBytes
        self.hardwareOSBuild = hardwareOSBuild
        self.hostIdentityDigest = hostIdentityDigest
        self.harnessGitSHA = harnessGitSHA
        self.candidateMLXSwiftVersion = candidateMLXSwiftVersion
        self.referenceMLXVersion = referenceMLXVersion
        self.referenceMLXLMVersion = referenceMLXLMVersion
        self.modelLabel = modelLabel
        self.modelConfigHash = modelConfigHash
        self.modelCheckpointManifestHash = modelCheckpointManifestHash
        self.modelQuant = modelQuant
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
    }
}

public struct Qwen38MTPPerformanceScorecardLiveExactnessProof: Codable, Equatable, Sendable {
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var artifactID: String
    public var sourceID: String
    public var evidenceID: String
    public var accepted: Bool

    public init(
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        artifactID: String,
        sourceID: String,
        evidenceID: String,
        accepted: Bool
    ) {
        self.artifact = artifact
        self.artifactID = artifactID
        self.sourceID = sourceID
        self.evidenceID = evidenceID
        self.accepted = accepted
    }
}

public struct Qwen38MTPPerformanceScorecardAuthorityBundle: Codable, Equatable, Sendable {
    public var acceptedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    public var trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities
    public var trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity

    public init(
        acceptedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) {
        self.acceptedLiveExactnessProof = acceptedLiveExactnessProof
        self.trustedEngineIdentities = trustedEngineIdentities
        self.trustedRunIdentity = trustedRunIdentity
    }
}

public struct Qwen38MTPPerformanceScorecardHardware: Codable, Equatable, Sendable {
    public let className: String
    public let chip: String
    public let ramBytes: UInt64
    public let osBuild: String
    public let hostIdentityDigest: String

    public init(
        className: String,
        chip: String = "",
        ramBytes: UInt64,
        osBuild: String = "",
        hostIdentityDigest: String = ""
    ) {
        self.className = className
        self.chip = chip
        self.ramBytes = ramBytes
        self.osBuild = osBuild
        self.hostIdentityDigest = hostIdentityDigest
    }
}

public struct Qwen38MTPPerformanceScorecardWorkloadCase: Codable, Equatable, Sendable {
    public let id: String
    public var prompt: String
    public var promptSHA256: String
    public let maxCompletionTokens: Int

    public init(
        id: String,
        prompt: String,
        promptSHA256: String,
        maxCompletionTokens: Int
    ) {
        self.id = id
        self.prompt = prompt
        self.promptSHA256 = promptSHA256
        self.maxCompletionTokens = maxCompletionTokens
    }
}

public struct Qwen38MTPPerformanceScorecardWorkload: Codable, Equatable, Sendable {
    public let id: String
    public var contentSHA256: String
    public let tokenizerSHA256: String
    public var chatTemplateSHA256: String
    public var contextTokenLimit: Int
    public let thinkingEnabled: Bool
    public var cases: [Qwen38MTPPerformanceScorecardWorkloadCase]

    public init(
        id: String,
        contentSHA256: String,
        tokenizerSHA256: String,
        chatTemplateSHA256: String,
        contextTokenLimit: Int,
        thinkingEnabled: Bool,
        cases: [Qwen38MTPPerformanceScorecardWorkloadCase]
    ) {
        self.id = id
        self.contentSHA256 = contentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.chatTemplateSHA256 = chatTemplateSHA256
        self.contextTokenLimit = contextTokenLimit
        self.thinkingEnabled = thinkingEnabled
        self.cases = cases
    }
}

public struct Qwen38MTPPerformanceScorecardSettings: Codable, Equatable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var seed: Int64?
    public var toolsEmpty: Bool
    public var penaltiesDisabled: Bool
    public var streaming: Bool

    public init(
        temperature: Double?,
        topP: Double?,
        topK: Int?,
        minP: Double?,
        seed: Int64?,
        toolsEmpty: Bool,
        penaltiesDisabled: Bool,
        streaming: Bool
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.toolsEmpty = toolsEmpty
        self.penaltiesDisabled = penaltiesDisabled
        self.streaming = streaming
    }
}

public enum Qwen38MTPPerformanceScorecardRunOrder: String, Codable, Equatable, Sendable {
    case candidateThenReference
    case referenceThenCandidate
}

public struct Qwen38MTPPerformanceScorecardPairSchedule: Codable, Equatable, Sendable {
    public let concurrency: Int
    public let pairIndex: Int
    public let order: Qwen38MTPPerformanceScorecardRunOrder
    public let caseIDs: [String]

    public init(
        concurrency: Int,
        pairIndex: Int,
        order: Qwen38MTPPerformanceScorecardRunOrder,
        caseIDs: [String]
    ) {
        self.concurrency = concurrency
        self.pairIndex = pairIndex
        self.order = order
        self.caseIDs = caseIDs
    }
}

public struct Qwen38MTPPerformanceScorecardRunPlan: Codable, Equatable, Sendable {
    public let concurrencies: [Int]
    public let droppedWarmupPairs: Int
    public let measuredPairs: Int
    public let orders: [Qwen38MTPPerformanceScorecardRunOrder]
    public let schedules: [Qwen38MTPPerformanceScorecardPairSchedule]

    public init(
        concurrencies: [Int],
        droppedWarmupPairs: Int,
        measuredPairs: Int,
        orders: [Qwen38MTPPerformanceScorecardRunOrder],
        schedules: [Qwen38MTPPerformanceScorecardPairSchedule]
    ) {
        self.concurrencies = concurrencies
        self.droppedWarmupPairs = droppedWarmupPairs
        self.measuredPairs = measuredPairs
        self.orders = orders
        self.schedules = schedules
    }

    public var totalPairsPerConcurrency: Int {
        droppedWarmupPairs + measuredPairs
    }

    /// The complete source-derived cost of executing this plan. A pair owns
    /// one candidate and one reference engine measurement; each engine
    /// measurement owns one request measurement per scheduled case ID.
    public var budget: Qwen38MTPPerformanceScorecardRunBudget {
        let perConcurrency = concurrencies.map { concurrency in
            let rows = schedules.filter { $0.concurrency == concurrency }
            let warmups = rows.filter { $0.pairIndex < droppedWarmupPairs }
            let measured = rows.filter { $0.pairIndex >= droppedWarmupPairs }
            let requestCount: ([Qwen38MTPPerformanceScorecardPairSchedule]) -> Int = { schedules in
                schedules.reduce(0) { $0 + (2 * $1.caseIDs.count) }
            }
            return Qwen38MTPPerformanceScorecardConcurrencyBudget(
                concurrency: concurrency,
                pairRecords: rows.count,
                warmupPairs: warmups.count,
                measuredPairs: measured.count,
                engineMeasurements: 2 * rows.count,
                requestMeasurementsIncludingWarmups: requestCount(rows),
                warmupRequestMeasurements: requestCount(warmups),
                measuredRequestMeasurements: requestCount(measured))
        }
        return Qwen38MTPPerformanceScorecardRunBudget(
            pairRecords: perConcurrency.reduce(0) { $0 + $1.pairRecords },
            engineMeasurements: perConcurrency.reduce(0) { $0 + $1.engineMeasurements },
            requestMeasurementsIncludingWarmups: perConcurrency.reduce(0) {
                $0 + $1.requestMeasurementsIncludingWarmups
            },
            warmupRequestMeasurements: perConcurrency.reduce(0) {
                $0 + $1.warmupRequestMeasurements
            },
            measuredRequestMeasurements: perConcurrency.reduce(0) {
                $0 + $1.measuredRequestMeasurements
            },
            perConcurrency: perConcurrency)
    }
}

public struct Qwen38MTPPerformanceScorecardConcurrencyBudget: Codable, Equatable, Sendable {
    public let concurrency: Int
    public let pairRecords: Int
    public let warmupPairs: Int
    public let measuredPairs: Int
    public let engineMeasurements: Int
    public let requestMeasurementsIncludingWarmups: Int
    public let warmupRequestMeasurements: Int
    public let measuredRequestMeasurements: Int

    public init(
        concurrency: Int,
        pairRecords: Int,
        warmupPairs: Int,
        measuredPairs: Int,
        engineMeasurements: Int,
        requestMeasurementsIncludingWarmups: Int,
        warmupRequestMeasurements: Int,
        measuredRequestMeasurements: Int
    ) {
        self.concurrency = concurrency
        self.pairRecords = pairRecords
        self.warmupPairs = warmupPairs
        self.measuredPairs = measuredPairs
        self.engineMeasurements = engineMeasurements
        self.requestMeasurementsIncludingWarmups = requestMeasurementsIncludingWarmups
        self.warmupRequestMeasurements = warmupRequestMeasurements
        self.measuredRequestMeasurements = measuredRequestMeasurements
    }
}

public struct Qwen38MTPPerformanceScorecardRunBudget: Codable, Equatable, Sendable {
    public let pairRecords: Int
    public let engineMeasurements: Int
    public let requestMeasurementsIncludingWarmups: Int
    public let warmupRequestMeasurements: Int
    public let measuredRequestMeasurements: Int
    public let perConcurrency: [Qwen38MTPPerformanceScorecardConcurrencyBudget]

    public init(
        pairRecords: Int,
        engineMeasurements: Int,
        requestMeasurementsIncludingWarmups: Int,
        warmupRequestMeasurements: Int,
        measuredRequestMeasurements: Int,
        perConcurrency: [Qwen38MTPPerformanceScorecardConcurrencyBudget]
    ) {
        self.pairRecords = pairRecords
        self.engineMeasurements = engineMeasurements
        self.requestMeasurementsIncludingWarmups = requestMeasurementsIncludingWarmups
        self.warmupRequestMeasurements = warmupRequestMeasurements
        self.measuredRequestMeasurements = measuredRequestMeasurements
        self.perConcurrency = perConcurrency
    }
}

public struct Qwen38MTPPerformanceScorecardRequestMeasurement: Codable, Equatable, Sendable {
    public var caseID: String
    public let requestIndex: Int
    public let promptSeconds: Double
    public let prefillSeconds: Double
    public var ttftSeconds: Double
    public var decodeTokenCount: Int
    public let decodeSeconds: Double
    public var e2eSeconds: Double
    public var outputDigest: String
    public var cacheDigest: String
    public let outputProvenanceID: String
    public let cacheProvenanceID: String

    public init(
        caseID: String,
        requestIndex: Int,
        promptSeconds: Double,
        prefillSeconds: Double,
        ttftSeconds: Double,
        decodeTokenCount: Int,
        decodeSeconds: Double,
        e2eSeconds: Double,
        outputDigest: String,
        cacheDigest: String,
        outputProvenanceID: String,
        cacheProvenanceID: String
    ) {
        self.caseID = caseID
        self.requestIndex = requestIndex
        self.promptSeconds = promptSeconds
        self.prefillSeconds = prefillSeconds
        self.ttftSeconds = ttftSeconds
        self.decodeTokenCount = decodeTokenCount
        self.decodeSeconds = decodeSeconds
        self.e2eSeconds = e2eSeconds
        self.outputDigest = outputDigest
        self.cacheDigest = cacheDigest
        self.outputProvenanceID = outputProvenanceID
        self.cacheProvenanceID = cacheProvenanceID
    }

    public var decodeTokensPerSecond: Double {
        Double(decodeTokenCount) / decodeSeconds
    }
}

public struct Qwen38MTPPerformanceScorecardEngineMeasurement: Codable, Equatable, Sendable {
    public var identity: Qwen38MTPPerformanceScorecardModel
    public var requests: [Qwen38MTPPerformanceScorecardRequestMeasurement]
    public var wallSeconds: Double
    public let peakRSSBytes: UInt64
    public let peakMetalBytes: UInt64
    public let thermalBefore: String
    public var thermalAfter: String
    public var proposalCount: Int
    public var acceptedCount: Int
    public var fallbackUsed: Bool
    public var passthroughUsed: Bool

    public init(
        identity: Qwen38MTPPerformanceScorecardModel,
        requests: [Qwen38MTPPerformanceScorecardRequestMeasurement],
        wallSeconds: Double,
        peakRSSBytes: UInt64,
        peakMetalBytes: UInt64,
        thermalBefore: String,
        thermalAfter: String,
        proposalCount: Int,
        acceptedCount: Int,
        fallbackUsed: Bool,
        passthroughUsed: Bool
    ) {
        self.identity = identity
        self.requests = requests
        self.wallSeconds = wallSeconds
        self.peakRSSBytes = peakRSSBytes
        self.peakMetalBytes = peakMetalBytes
        self.thermalBefore = thermalBefore
        self.thermalAfter = thermalAfter
        self.proposalCount = proposalCount
        self.acceptedCount = acceptedCount
        self.fallbackUsed = fallbackUsed
        self.passthroughUsed = passthroughUsed
    }
}

public struct Qwen38MTPPerformanceScorecardPair: Codable, Equatable, Sendable {
    public let concurrency: Int
    public let pairIndex: Int
    public let warmup: Bool
    public let order: Qwen38MTPPerformanceScorecardRunOrder
    public let scheduledCaseIDs: [String]
    public var candidate: Qwen38MTPPerformanceScorecardEngineMeasurement
    public var reference: Qwen38MTPPerformanceScorecardEngineMeasurement

    public init(
        concurrency: Int,
        pairIndex: Int,
        warmup: Bool,
        order: Qwen38MTPPerformanceScorecardRunOrder,
        scheduledCaseIDs: [String],
        candidate: Qwen38MTPPerformanceScorecardEngineMeasurement,
        reference: Qwen38MTPPerformanceScorecardEngineMeasurement
    ) {
        self.concurrency = concurrency
        self.pairIndex = pairIndex
        self.warmup = warmup
        self.order = order
        self.scheduledCaseIDs = scheduledCaseIDs
        self.candidate = candidate
        self.reference = reference
    }
}

public struct Qwen38MTPPerformanceScorecardPercentiles: Codable, Equatable, Sendable {
    public var p50: Double
    public var p95: Double

    public init(p50: Double, p95: Double) {
        self.p50 = p50
        self.p95 = p95
    }
}

public struct Qwen38MTPPerformanceScorecardEngineMetrics: Codable, Equatable, Sendable {
    public var prompt: Qwen38MTPPerformanceScorecardPercentiles
    public var prefill: Qwen38MTPPerformanceScorecardPercentiles
    public var ttft: Qwen38MTPPerformanceScorecardPercentiles
    public var decodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles
    public var e2e: Qwen38MTPPerformanceScorecardPercentiles
    public var aggregateThroughputTokensPerSecond: Double
    public var peakRSSBytes: UInt64
    public var peakMetalBytes: UInt64

    public init(
        prompt: Qwen38MTPPerformanceScorecardPercentiles,
        prefill: Qwen38MTPPerformanceScorecardPercentiles,
        ttft: Qwen38MTPPerformanceScorecardPercentiles,
        decodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles,
        e2e: Qwen38MTPPerformanceScorecardPercentiles,
        aggregateThroughputTokensPerSecond: Double,
        peakRSSBytes: UInt64,
        peakMetalBytes: UInt64
    ) {
        self.prompt = prompt
        self.prefill = prefill
        self.ttft = ttft
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.e2e = e2e
        self.aggregateThroughputTokensPerSecond = aggregateThroughputTokensPerSecond
        self.peakRSSBytes = peakRSSBytes
        self.peakMetalBytes = peakMetalBytes
    }
}

public struct Qwen38MTPPerformanceScorecardConcurrencyMetrics: Codable, Equatable, Sendable {
    public let concurrency: Int
    public var candidate: Qwen38MTPPerformanceScorecardEngineMetrics
    public var reference: Qwen38MTPPerformanceScorecardEngineMetrics
    public let candidateTTFT: Qwen38MTPPerformanceScorecardPercentiles
    public let referenceTTFT: Qwen38MTPPerformanceScorecardPercentiles
    public let candidatePrefill: Qwen38MTPPerformanceScorecardPercentiles
    public let referencePrefill: Qwen38MTPPerformanceScorecardPercentiles
    public let candidateDecodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles
    public let referenceDecodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles
    public let candidateE2E: Qwen38MTPPerformanceScorecardPercentiles
    public let referenceE2E: Qwen38MTPPerformanceScorecardPercentiles
    public let aggregateThroughputRatio: Double
    public let requestE2EP95LatencyRatio: Double
    public let e2EPairedMedianLatencyRatio: Double
    public let decodeTokensPerSecondRatio: Double
    public let peakRSSBytes: UInt64
    public let peakMetalBytes: UInt64
    public let thermalStates: [String]

    public init(
        concurrency: Int,
        candidate: Qwen38MTPPerformanceScorecardEngineMetrics,
        reference: Qwen38MTPPerformanceScorecardEngineMetrics,
        candidateTTFT: Qwen38MTPPerformanceScorecardPercentiles,
        referenceTTFT: Qwen38MTPPerformanceScorecardPercentiles,
        candidatePrefill: Qwen38MTPPerformanceScorecardPercentiles,
        referencePrefill: Qwen38MTPPerformanceScorecardPercentiles,
        candidateDecodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles,
        referenceDecodeTokensPerSecond: Qwen38MTPPerformanceScorecardPercentiles,
        candidateE2E: Qwen38MTPPerformanceScorecardPercentiles,
        referenceE2E: Qwen38MTPPerformanceScorecardPercentiles,
        aggregateThroughputRatio: Double,
        requestE2EP95LatencyRatio: Double,
        e2EPairedMedianLatencyRatio: Double,
        decodeTokensPerSecondRatio: Double,
        peakRSSBytes: UInt64,
        peakMetalBytes: UInt64,
        thermalStates: [String]
    ) {
        self.concurrency = concurrency
        self.candidate = candidate
        self.reference = reference
        self.candidateTTFT = candidateTTFT
        self.referenceTTFT = referenceTTFT
        self.candidatePrefill = candidatePrefill
        self.referencePrefill = referencePrefill
        self.candidateDecodeTokensPerSecond = candidateDecodeTokensPerSecond
        self.referenceDecodeTokensPerSecond = referenceDecodeTokensPerSecond
        self.candidateE2E = candidateE2E
        self.referenceE2E = referenceE2E
        self.aggregateThroughputRatio = aggregateThroughputRatio
        self.requestE2EP95LatencyRatio = requestE2EP95LatencyRatio
        self.e2EPairedMedianLatencyRatio = e2EPairedMedianLatencyRatio
        self.decodeTokensPerSecondRatio = decodeTokensPerSecondRatio
        self.peakRSSBytes = peakRSSBytes
        self.peakMetalBytes = peakMetalBytes
        self.thermalStates = thermalStates
    }
}

public struct Qwen38MTPPerformanceScorecardMetrics: Codable, Equatable, Sendable {
    public var perConcurrency: [Qwen38MTPPerformanceScorecardConcurrencyMetrics]

    public init(perConcurrency: [Qwen38MTPPerformanceScorecardConcurrencyMetrics]) {
        self.perConcurrency = perConcurrency
    }

    public static let empty = Qwen38MTPPerformanceScorecardMetrics(perConcurrency: [])
}

public struct Qwen38MTPPerformanceScorecardHalfRatios: Codable, Equatable, Sendable {
    public let soloE2ELatency: Double
    public let soloDecodeTokensPerSecond: Double
    public let soloTTFTLatency: Double
    public let c2AggregateThroughput: Double
    public let c2RequestE2EP95Latency: Double
    public let c4AggregateThroughput: Double
    public let c4RequestE2EP95Latency: Double

    public init(
        soloE2ELatency: Double,
        soloDecodeTokensPerSecond: Double,
        soloTTFTLatency: Double,
        c2AggregateThroughput: Double,
        c2RequestE2EP95Latency: Double,
        c4AggregateThroughput: Double,
        c4RequestE2EP95Latency: Double
    ) {
        self.soloE2ELatency = soloE2ELatency
        self.soloDecodeTokensPerSecond = soloDecodeTokensPerSecond
        self.soloTTFTLatency = soloTTFTLatency
        self.c2AggregateThroughput = c2AggregateThroughput
        self.c2RequestE2EP95Latency = c2RequestE2EP95Latency
        self.c4AggregateThroughput = c4AggregateThroughput
        self.c4RequestE2EP95Latency = c4RequestE2EP95Latency
    }

    public var minimum: Double {
        [
            soloE2ELatency,
            soloDecodeTokensPerSecond,
            soloTTFTLatency,
            c2AggregateThroughput,
            c2RequestE2EP95Latency,
            c4AggregateThroughput,
            c4RequestE2EP95Latency,
        ].min() ?? 0
    }
}

public struct Qwen38MTPPerformanceScorecardVerdict: Codable, Equatable, Sendable {
    public var qualified: Bool
    public let soloE2EPairedMedianLatencyRatio: Double
    public let soloDecodeTokensPerSecondRatio: Double
    public let soloTTFTP95LatencyRatio: Double
    public let c2AggregateThroughputRatio: Double
    public let c2RequestE2EP95LatencyRatio: Double
    public let c4AggregateThroughputRatio: Double
    public let c4RequestE2EP95LatencyRatio: Double
    public let chronologicalFirstHalf: Qwen38MTPPerformanceScorecardHalfRatios
    public let chronologicalSecondHalf: Qwen38MTPPerformanceScorecardHalfRatios

    public init(
        qualified: Bool,
        soloE2EPairedMedianLatencyRatio: Double,
        soloDecodeTokensPerSecondRatio: Double,
        soloTTFTP95LatencyRatio: Double,
        c2AggregateThroughputRatio: Double,
        c2RequestE2EP95LatencyRatio: Double,
        c4AggregateThroughputRatio: Double,
        c4RequestE2EP95LatencyRatio: Double,
        chronologicalFirstHalf: Qwen38MTPPerformanceScorecardHalfRatios,
        chronologicalSecondHalf: Qwen38MTPPerformanceScorecardHalfRatios
    ) {
        self.qualified = qualified
        self.soloE2EPairedMedianLatencyRatio = soloE2EPairedMedianLatencyRatio
        self.soloDecodeTokensPerSecondRatio = soloDecodeTokensPerSecondRatio
        self.soloTTFTP95LatencyRatio = soloTTFTP95LatencyRatio
        self.c2AggregateThroughputRatio = c2AggregateThroughputRatio
        self.c2RequestE2EP95LatencyRatio = c2RequestE2EP95LatencyRatio
        self.c4AggregateThroughputRatio = c4AggregateThroughputRatio
        self.c4RequestE2EP95LatencyRatio = c4RequestE2EP95LatencyRatio
        self.chronologicalFirstHalf = chronologicalFirstHalf
        self.chronologicalSecondHalf = chronologicalSecondHalf
    }

    public static let unqualified = Qwen38MTPPerformanceScorecardVerdict(
        qualified: false,
        soloE2EPairedMedianLatencyRatio: 0,
        soloDecodeTokensPerSecondRatio: 0,
        soloTTFTP95LatencyRatio: 0,
        c2AggregateThroughputRatio: 0,
        c2RequestE2EP95LatencyRatio: 0,
        c4AggregateThroughputRatio: 0,
        c4RequestE2EP95LatencyRatio: 0,
        chronologicalFirstHalf: .zero,
        chronologicalSecondHalf: .zero)
}

public extension Qwen38MTPPerformanceScorecardHalfRatios {
    static let zero = Qwen38MTPPerformanceScorecardHalfRatios(
        soloE2ELatency: 0,
        soloDecodeTokensPerSecond: 0,
        soloTTFTLatency: 0,
        c2AggregateThroughput: 0,
        c2RequestE2EP95Latency: 0,
        c4AggregateThroughput: 0,
        c4RequestE2EP95Latency: 0)
}

public struct Qwen38MTPPerformanceScorecardEvidence: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var candidate: Qwen38MTPPerformanceScorecardModel
    public var reference: Qwen38MTPPerformanceScorecardModel
    public var liveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof?
    public let measurementClass: String
    public let hardware: Qwen38MTPPerformanceScorecardHardware
    public let releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var workload: Qwen38MTPPerformanceScorecardWorkload
    public var settings: Qwen38MTPPerformanceScorecardSettings
    public let runPlan: Qwen38MTPPerformanceScorecardRunPlan
    public var pairs: [Qwen38MTPPerformanceScorecardPair]
    public var metrics: Qwen38MTPPerformanceScorecardMetrics
    public var verdict: Qwen38MTPPerformanceScorecardVerdict

    public init(
        schemaVersion: Int,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        candidate: Qwen38MTPPerformanceScorecardModel,
        reference: Qwen38MTPPerformanceScorecardModel,
        liveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof?,
        measurementClass: String,
        hardware: Qwen38MTPPerformanceScorecardHardware,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        workload: Qwen38MTPPerformanceScorecardWorkload,
        settings: Qwen38MTPPerformanceScorecardSettings,
        runPlan: Qwen38MTPPerformanceScorecardRunPlan,
        pairs: [Qwen38MTPPerformanceScorecardPair],
        metrics: Qwen38MTPPerformanceScorecardMetrics,
        verdict: Qwen38MTPPerformanceScorecardVerdict
    ) {
        self.schemaVersion = schemaVersion
        self.artifact = artifact
        self.candidate = candidate
        self.reference = reference
        self.liveExactnessProof = liveExactnessProof
        self.measurementClass = measurementClass
        self.hardware = hardware
        self.releaseBuildRequired = releaseBuildRequired
        self.releaseBuildObserved = releaseBuildObserved
        self.workload = workload
        self.settings = settings
        self.runPlan = runPlan
        self.pairs = pairs
        self.metrics = metrics
        self.verdict = verdict
    }
}

public enum Qwen38MTPPerformanceScorecardGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case schemaVersionMismatch(Int)
    case invalidArtifactBinding
    case invalidModelIdentity
    case liveExactnessNotPromoted
    case performanceIdentityNotPromoted
    case runIdentityNotPromoted
    case invalidLiveExactnessProof
    case invalidRunIdentity
    case invalidMeasurementClass(String)
    case invalidHardware
    case releaseBuildRequired
    case invalidWorkload
    case invalidGenerationSettings(String)
    case invalidRunPlan
    case invalidPairCardinality(expected: Int, actual: Int)
    case invalidPair(index: Int, reason: String)
    case metricsMismatch
    case verdictMismatch
    case unqualifiedPerformance
    case malformedJSONL(line: Int)
    case unterminatedJSONL
    case invalidRecordCardinality(Int)
    case wrongSubcommand(String)
    case invalidProvenance(String)

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let value): return "schemaVersion mismatch: \(value)"
        case .invalidArtifactBinding: return "invalid artifact binding"
        case .invalidModelIdentity: return "invalid model identity"
        case .liveExactnessNotPromoted: return "accepted live exactness proof is not promoted"
        case .performanceIdentityNotPromoted: return "performance engine identities are not promoted"
        case .runIdentityNotPromoted: return "performance run identity is not promoted"
        case .invalidLiveExactnessProof: return "invalid accepted live exactness proof"
        case .invalidRunIdentity: return "invalid performance run identity"
        case .invalidMeasurementClass(let value): return "invalid measurement class: \(value)"
        case .invalidHardware: return "invalid hardware"
        case .releaseBuildRequired: return "Release build required"
        case .invalidWorkload: return "invalid workload"
        case .invalidGenerationSettings(let field): return "invalid generation settings: \(field)"
        case .invalidRunPlan: return "invalid run plan"
        case .invalidPairCardinality(let expected, let actual):
            return "invalid pair cardinality: expected \(expected), got \(actual)"
        case .invalidPair(let index, let reason): return "invalid pair \(index): \(reason)"
        case .metricsMismatch: return "stored metrics do not match recomputed metrics"
        case .verdictMismatch: return "stored verdict does not match recomputed verdict"
        case .unqualifiedPerformance: return "unqualified performance"
        case .malformedJSONL(let line): return "malformed JSONL at line \(line)"
        case .unterminatedJSONL: return "unterminated JSONL"
        case .invalidRecordCardinality(let count): return "invalid JSONL record cardinality: \(count)"
        case .wrongSubcommand(let subcommand): return "wrong subcommand: \(subcommand)"
        case .invalidProvenance(let field): return "invalid provenance: \(field)"
        }
    }
}

private struct Qwen38MTPPerformanceScorecardSchemaProbe: Codable, Sendable {
    let schemaVersion: Int
}

public enum Qwen38MTPPerformanceScorecardGate {
    public static let schemaVersion = 2
    public static let subcommand = "qwen38-mtp-performance-scorecard"
    public static let rejectedSubcommand = "qwen38-mtp-performance-scorecard-rejected"
    public static let measurementClass = "dedicated-heavy-256gib"
    public static let requiredRAMBytes: UInt64 = 256 * 1024 * 1024 * 1024
    public static let settingsID = "greedy-no-tools-no-penalties-no-streaming-v1"
    public static let requiredAcceptedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof? = nil
    public static let requiredTrustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities? = nil
    public static let requiredTrustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity? = nil
    public static let modelArtifactLabel = "qwen38-27b-scorecard-artifact"
    public static let requiredArtifactLock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
    public static let requiredArtifact = artifact(from: requiredArtifactLock)
    public static let requiredSettings = Qwen38MTPPerformanceScorecardSettings(
        temperature: 0,
        topP: nil,
        topK: nil,
        minP: nil,
        seed: nil,
        toolsEmpty: true,
        penaltiesDisabled: true,
        streaming: false)
    public static let requiredWorkloadCases: [Qwen38MTPPerformanceScorecardWorkloadCase] = [
        .init(
            id: "case-01",
            prompt: "Summarize a generic maintenance checklist in three concise bullets.",
            promptSHA256: promptSHA256(
                "Summarize a generic maintenance checklist in three concise bullets."),
            maxCompletionTokens: 128),
        .init(
            id: "case-02",
            prompt: "Explain how a small team should triage a delayed batch job.",
            promptSHA256: promptSHA256(
                "Explain how a small team should triage a delayed batch job."),
            maxCompletionTokens: 160),
        .init(
            id: "case-03",
            prompt: "Draft a neutral incident update with impact, status, and next steps.",
            promptSHA256: promptSHA256(
                "Draft a neutral incident update with impact, status, and next steps."),
            maxCompletionTokens: 192),
        .init(
            id: "case-04",
            prompt: "Compare two generic caching strategies for a read-heavy service.",
            promptSHA256: promptSHA256(
                "Compare two generic caching strategies for a read-heavy service."),
            maxCompletionTokens: 224),
        .init(
            id: "case-05",
            prompt: "Write a short runbook for validating a deterministic command-line report.",
            promptSHA256: promptSHA256(
                "Write a short runbook for validating a deterministic command-line report."),
            maxCompletionTokens: 256),
        .init(
            id: "case-06",
            prompt: "Describe a safe rollback plan for a generic local performance experiment.",
            promptSHA256: promptSHA256(
                "Describe a safe rollback plan for a generic local performance experiment."),
            maxCompletionTokens: 288),
    ]
    public static let requiredWorkload = Qwen38MTPPerformanceScorecardWorkload(
        id: "qwen38-27b-frozen-scorecard-workload-v2",
        contentSHA256: canonicalWorkloadContentSHA256(requiredWorkloadCases),
        tokenizerSHA256: requiredArtifact.tokenizerSHA256,
        chatTemplateSHA256: "b426d0bb02412efa9e44777312cc7df1bf95ea332dc0d2e46376c801f273599d",
        contextTokenLimit: 40_960,
        thinkingEnabled: false,
        cases: requiredWorkloadCases)
    public static let runPlan = Qwen38MTPPerformanceScorecardRunPlan(
        concurrencies: [1, 2, 4],
        droppedWarmupPairs: 2,
        measuredPairs: 40,
        orders: makeOrders(),
        schedules: makeSchedules())

    private static let requiredWorkloadCaseByID = Dictionary(
        uniqueKeysWithValues: requiredWorkloadCases.map { ($0.id, $0) })

    public static func promptSHA256(_ prompt: String) -> String {
        sha256Hex(Data(prompt.utf8))
    }

    public static func canonicalWorkloadContentSHA256(
        _ cases: [Qwen38MTPPerformanceScorecardWorkloadCase]
    ) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try! encoder.encode(cases)
        return sha256Hex(data)
    }

    public static func validate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedLiveExactnessProof = requiredAcceptedLiveExactnessProof else {
            throw Qwen38MTPPerformanceScorecardGateError.liveExactnessNotPromoted
        }
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try validate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    public static func validate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        try validate(
            evidence,
            trustedLiveExactnessProof: authority.acceptedLiveExactnessProof,
            trustedEngineIdentities: authority.trustedEngineIdentities,
            trustedRunIdentity: authority.trustedRunIdentity)
    }

    public static func validateAuthority(
        _ authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) throws {
        try validateAuthority(
            trustedLiveExactnessProof: authority.acceptedLiveExactnessProof,
            trustedEngineIdentities: authority.trustedEngineIdentities,
            trustedRunIdentity: authority.trustedRunIdentity)
    }

    public static func validatePreflight(
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle,
        provenance: Provenance,
        releaseBuildObserved: Bool
    ) throws {
        try validateAuthority(authority)
        try validateProvenance(provenance, trustedRunIdentity: authority.trustedRunIdentity)
        guard releaseBuildObserved else {
            throw Qwen38MTPPerformanceScorecardGateError.releaseBuildRequired
        }
    }

    static func validate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        return try validate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities)
    }

    static func validate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try validate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    static func validate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        let metrics = try computeMetrics(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        guard evidence.metrics == metrics else {
            throw Qwen38MTPPerformanceScorecardGateError.metricsMismatch
        }
        let verdict = try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity,
            metrics: metrics)
        guard evidence.verdict == verdict else {
            throw Qwen38MTPPerformanceScorecardGateError.verdictMismatch
        }
        guard verdict.qualified else {
            throw Qwen38MTPPerformanceScorecardGateError.unqualifiedPerformance
        }
        return verdict
    }

    public static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedLiveExactnessProof = requiredAcceptedLiveExactnessProof else {
            throw Qwen38MTPPerformanceScorecardGateError.liveExactnessNotPromoted
        }
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    public static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: authority.acceptedLiveExactnessProof,
            trustedEngineIdentities: authority.trustedEngineIdentities,
            trustedRunIdentity: authority.trustedRunIdentity)
    }

    static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        return try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities)
    }

    static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        try evaluateCandidate(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity,
            metrics: computeMetrics(
                evidence,
                trustedLiveExactnessProof: trustedLiveExactnessProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity))
    }

    public static func computeMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        guard let trustedLiveExactnessProof = requiredAcceptedLiveExactnessProof else {
            throw Qwen38MTPPerformanceScorecardGateError.liveExactnessNotPromoted
        }
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try computeMetrics(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    public static func computeMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        try computeMetrics(
            evidence,
            trustedLiveExactnessProof: authority.acceptedLiveExactnessProof,
            trustedEngineIdentities: authority.trustedEngineIdentities,
            trustedRunIdentity: authority.trustedRunIdentity)
    }

    static func computeMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        guard let trustedEngineIdentities = requiredTrustedEngineIdentities else {
            throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
        }
        return try computeMetrics(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities)
    }

    static func computeMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        guard let trustedRunIdentity = requiredTrustedRunIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
        }
        return try computeMetrics(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    static func computeMetrics(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws -> Qwen38MTPPerformanceScorecardMetrics {
        try validateEnvelopeBeforeTimings(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        try validatePairs(evidence, trustedEngineIdentities: trustedEngineIdentities)

        let perConcurrency = try runPlan.concurrencies.map { concurrency in
            let measured = evidence.pairs.filter { $0.concurrency == concurrency && !$0.warmup }
            return try metrics(for: concurrency, measured: measured)
        }
        return Qwen38MTPPerformanceScorecardMetrics(perConcurrency: perConcurrency)
    }

    public static func validateJSONL(_ data: Data) throws -> [Qwen38MTPPerformanceScorecardVerdict] {
        try validateJSONL(
            data,
            trustedLiveExactnessProof: requiredAcceptedLiveExactnessProof,
            trustedEngineIdentities: requiredTrustedEngineIdentities,
            trustedRunIdentity: requiredTrustedRunIdentity)
    }

    public static func validateJSONL(
        _ data: Data,
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    ) throws -> [Qwen38MTPPerformanceScorecardVerdict] {
        try validateJSONL(
            data,
            trustedLiveExactnessProof: authority.acceptedLiveExactnessProof,
            trustedEngineIdentities: authority.trustedEngineIdentities,
            trustedRunIdentity: authority.trustedRunIdentity)
    }

    static func validateJSONL(
        _ data: Data,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof?
    ) throws -> [Qwen38MTPPerformanceScorecardVerdict] {
        try validateJSONL(
            data,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: requiredTrustedEngineIdentities,
            trustedRunIdentity: requiredTrustedRunIdentity)
    }

    static func validateJSONL(
        _ data: Data,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof?,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities?
    ) throws -> [Qwen38MTPPerformanceScorecardVerdict] {
        try validateJSONL(
            data,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: requiredTrustedRunIdentity)
    }

    static func validateJSONL(
        _ data: Data,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof?,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities?,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity?
    ) throws -> [Qwen38MTPPerformanceScorecardVerdict] {
        guard data.last == 0x0a else {
            throw Qwen38MTPPerformanceScorecardGateError.unterminatedJSONL
        }
        let rows = data.split(separator: 0x0a, omittingEmptySubsequences: false).dropLast()
        guard rows.count == 1 else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRecordCardinality(rows.count)
        }

        let decoder = JSONDecoder()
        var verdicts: [Qwen38MTPPerformanceScorecardVerdict] = []
        for (index, row) in rows.enumerated() {
            guard !row.isEmpty else {
                throw Qwen38MTPPerformanceScorecardGateError.malformedJSONL(line: index + 1)
            }
            let rowData = Data(row)
            let probe: ResultRecord<Qwen38MTPPerformanceScorecardSchemaProbe>
            do {
                probe = try decoder.decode(
                    ResultRecord<Qwen38MTPPerformanceScorecardSchemaProbe>.self,
                    from: rowData)
            } catch {
                throw Qwen38MTPPerformanceScorecardGateError.malformedJSONL(line: index + 1)
            }
            guard probe.payload.schemaVersion == schemaVersion else {
                throw Qwen38MTPPerformanceScorecardGateError.schemaVersionMismatch(
                    probe.payload.schemaVersion)
            }
            let record: ResultRecord<Qwen38MTPPerformanceScorecardEvidence>
            do {
                record = try decoder.decode(
                    ResultRecord<Qwen38MTPPerformanceScorecardEvidence>.self,
                    from: rowData)
            } catch {
                throw Qwen38MTPPerformanceScorecardGateError.malformedJSONL(line: index + 1)
            }
            guard record.subcommand == subcommand else {
                throw Qwen38MTPPerformanceScorecardGateError.wrongSubcommand(
                    record.subcommand)
            }
            guard let trustedLiveExactnessProof else {
                throw Qwen38MTPPerformanceScorecardGateError.liveExactnessNotPromoted
            }
            guard let trustedEngineIdentities else {
                throw Qwen38MTPPerformanceScorecardGateError.performanceIdentityNotPromoted
            }
            guard let trustedRunIdentity else {
                throw Qwen38MTPPerformanceScorecardGateError.runIdentityNotPromoted
            }
            try validateProvenance(
                record.provenance,
                evidence: record.payload,
                trustedRunIdentity: trustedRunIdentity)
            verdicts.append(try validate(
                record.payload,
                trustedLiveExactnessProof: trustedLiveExactnessProof,
                trustedEngineIdentities: trustedEngineIdentities,
                trustedRunIdentity: trustedRunIdentity))
        }
        return verdicts
    }

    private static func evaluateCandidate(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        metrics: Qwen38MTPPerformanceScorecardMetrics
    ) throws -> Qwen38MTPPerformanceScorecardVerdict {
        try validateEnvelopeBeforeTimings(
            evidence,
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        let byConcurrency = Dictionary(uniqueKeysWithValues: metrics.perConcurrency.map {
            ($0.concurrency, $0)
        })
        guard let solo = byConcurrency[1],
            let c2 = byConcurrency[2],
            let c4 = byConcurrency[4]
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunPlan
        }
        let halves = try chronologicalHalfRatios(evidence.pairs)
        let ttftFloor = 1 / 1.05
        let qualified = solo.e2EPairedMedianLatencyRatio >= 1
            && solo.decodeTokensPerSecondRatio >= 1
            && solo.referenceTTFT.p95 / solo.candidateTTFT.p95 >= ttftFloor
            && c2.aggregateThroughputRatio >= 1
            && c2.requestE2EP95LatencyRatio >= ttftFloor
            && c4.aggregateThroughputRatio >= 1
            && c4.requestE2EP95LatencyRatio >= ttftFloor
            && halves.first.minimum >= 0.98
            && halves.second.minimum >= 0.98

        return Qwen38MTPPerformanceScorecardVerdict(
            qualified: qualified,
            soloE2EPairedMedianLatencyRatio: solo.e2EPairedMedianLatencyRatio,
            soloDecodeTokensPerSecondRatio: solo.decodeTokensPerSecondRatio,
            soloTTFTP95LatencyRatio: solo.referenceTTFT.p95 / solo.candidateTTFT.p95,
            c2AggregateThroughputRatio: c2.aggregateThroughputRatio,
            c2RequestE2EP95LatencyRatio: c2.requestE2EP95LatencyRatio,
            c4AggregateThroughputRatio: c4.aggregateThroughputRatio,
            c4RequestE2EP95LatencyRatio: c4.requestE2EP95LatencyRatio,
            chronologicalFirstHalf: halves.first,
            chronologicalSecondHalf: halves.second)
    }

    private static func validateEnvelopeBeforeTimings(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws {
        guard evidence.schemaVersion == schemaVersion else {
            throw Qwen38MTPPerformanceScorecardGateError.schemaVersionMismatch(
                evidence.schemaVersion)
        }
        guard evidence.artifact == requiredArtifact else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidArtifactBinding
        }
        try validateAuthority(
            trustedLiveExactnessProof: trustedLiveExactnessProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
        guard evidence.candidate == trustedEngineIdentities.candidate,
            evidence.reference == trustedEngineIdentities.reference,
            evidence.candidate.artifact == evidence.reference.artifact,
            evidence.candidate.executionDigest != evidence.reference.executionDigest,
            evidence.candidate.sourceDigest != evidence.reference.sourceDigest
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidModelIdentity
        }
        guard evidence.liveExactnessProof == trustedLiveExactnessProof else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidLiveExactnessProof
        }
        guard evidence.measurementClass == measurementClass else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidMeasurementClass(
                evidence.measurementClass)
        }
        guard evidence.hardware.className == measurementClass,
            evidence.hardware.ramBytes == requiredRAMBytes
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidHardware
        }
        guard evidence.hardware.chip == trustedRunIdentity.hardwareChip,
            evidence.hardware.osBuild == trustedRunIdentity.hardwareOSBuild,
            evidence.hardware.hostIdentityDigest == trustedRunIdentity.hostIdentityDigest
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunIdentity
        }
        guard evidence.releaseBuildRequired, evidence.releaseBuildObserved else {
            throw Qwen38MTPPerformanceScorecardGateError.releaseBuildRequired
        }
        guard evidence.workload == requiredWorkload,
            evidence.workload.contentSHA256
                == canonicalWorkloadContentSHA256(evidence.workload.cases),
            evidence.workload.cases.allSatisfy({ $0.promptSHA256 == promptSHA256($0.prompt) })
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidWorkload
        }
        try validateSettings(evidence.settings)
        guard evidence.runPlan == runPlan else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunPlan
        }
    }

    private static func validateAuthority(
        trustedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws {
        guard trustedEngineIdentities.candidate.artifact == requiredArtifact,
            trustedEngineIdentities.reference.artifact == requiredArtifact,
            isLowerHex(trustedEngineIdentities.candidate.executionDigest, count: 64),
            isLowerHex(trustedEngineIdentities.candidate.sourceDigest, count: 64),
            isLowerHex(trustedEngineIdentities.reference.executionDigest, count: 64),
            isLowerHex(trustedEngineIdentities.reference.sourceDigest, count: 64),
            trustedEngineIdentities.candidate.executionDigest
                != trustedEngineIdentities.reference.executionDigest,
            trustedEngineIdentities.candidate.sourceDigest
                != trustedEngineIdentities.reference.sourceDigest
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidModelIdentity
        }
        guard trustedLiveExactnessProof.artifact == requiredArtifact,
            trustedLiveExactnessProof.accepted,
            isLowerHex(trustedLiveExactnessProof.artifactID, count: 64),
            isLowerHex(trustedLiveExactnessProof.sourceID, count: 64),
            isLowerHex(trustedLiveExactnessProof.evidenceID, count: 64)
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidLiveExactnessProof
        }
        guard trustedRunIdentity.measurementClass == measurementClass,
            trustedRunIdentity.hardwareRAMBytes == requiredRAMBytes,
            !trustedRunIdentity.hardwareChip.isEmpty,
            !trustedRunIdentity.hardwareOSBuild.isEmpty,
            isLowerHex(trustedRunIdentity.hostIdentityDigest, count: 64),
            isLowerHex(trustedRunIdentity.harnessGitSHA, count: 40),
            !trustedRunIdentity.candidateMLXSwiftVersion.isEmpty,
            trustedRunIdentity.modelLabel == modelArtifactLabel,
            trustedRunIdentity.modelConfigHash == requiredArtifact.targetConfigSHA256,
            trustedRunIdentity.modelCheckpointManifestHash
                == requiredArtifact.targetTensorManifestSHA256,
            trustedRunIdentity.modelQuant == ModelQuantInfo(bits: 8, groupSize: 32),
            trustedRunIdentity.corpusID == requiredWorkload.id,
            trustedRunIdentity.corpusContentHash == requiredWorkload.contentSHA256
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunIdentity
        }
    }

    private static func validateSettings(
        _ settings: Qwen38MTPPerformanceScorecardSettings
    ) throws {
        guard settings == requiredSettings else {
            if settings.temperature != nil && settings.temperature != 0 {
                throw Qwen38MTPPerformanceScorecardGateError.invalidGenerationSettings("greedy")
            }
            if !settings.toolsEmpty {
                throw Qwen38MTPPerformanceScorecardGateError.invalidGenerationSettings("tools")
            }
            if !settings.penaltiesDisabled {
                throw Qwen38MTPPerformanceScorecardGateError.invalidGenerationSettings("penalties")
            }
            if settings.streaming {
                throw Qwen38MTPPerformanceScorecardGateError.invalidGenerationSettings("streaming")
            }
            throw Qwen38MTPPerformanceScorecardGateError.invalidGenerationSettings("sampling")
        }
    }

    private static func validatePairs(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities
    ) throws {
        let expectedCount = runPlan.schedules.count
        guard evidence.pairs.count == expectedCount else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPairCardinality(
                expected: expectedCount,
                actual: evidence.pairs.count)
        }
        for (index, schedule) in runPlan.schedules.enumerated() {
            let pair = evidence.pairs[index]
            guard pair.concurrency == schedule.concurrency,
                pair.pairIndex == schedule.pairIndex,
                pair.warmup == (schedule.pairIndex < runPlan.droppedWarmupPairs),
                pair.order == schedule.order,
                pair.scheduledCaseIDs == schedule.caseIDs
            else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "order/cardinality")
            }
            try validateEngine(
                pair.candidate,
                expectedIdentity: trustedEngineIdentities.candidate,
                schedule: schedule,
                index: index,
                candidate: true)
            try validateEngine(
                pair.reference,
                expectedIdentity: trustedEngineIdentities.reference,
                schedule: schedule,
                index: index,
                candidate: false)
            for requestIndex in 0..<schedule.concurrency {
                let candidate = pair.candidate.requests[requestIndex]
                let reference = pair.reference.requests[requestIndex]
                guard candidate.caseID == reference.caseID else {
                    throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                        index: index,
                        reason: "request schedule")
                }
                guard candidate.outputDigest == reference.outputDigest,
                    candidate.cacheDigest == reference.cacheDigest,
                    candidate.outputProvenanceID == reference.outputProvenanceID,
                    candidate.cacheProvenanceID == reference.cacheProvenanceID
                else {
                    throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                        index: index,
                        reason: "output/cache provenance")
                }
                guard candidate.decodeTokenCount == reference.decodeTokenCount else {
                    throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                        index: index,
                        reason: "decode token parity")
                }
            }
        }
    }

    private static func validateEngine(
        _ engine: Qwen38MTPPerformanceScorecardEngineMeasurement,
        expectedIdentity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        index: Int,
        candidate: Bool
    ) throws {
        guard engine.identity == expectedIdentity else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "engine identity")
        }
        guard engine.requests.count == schedule.concurrency else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "request count")
        }
        guard engine.wallSeconds.isFinite, engine.wallSeconds > 0,
            engine.peakRSSBytes > 0,
            engine.peakMetalBytes > 0
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(index: index, reason: "timing")
        }
        guard isAllowedThermal(engine.thermalBefore),
            isAllowedThermal(engine.thermalAfter)
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(index: index, reason: "thermal")
        }
        guard !engine.fallbackUsed, !engine.passthroughUsed else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "fallback/passthrough")
        }
        if candidate {
            guard engine.proposalCount > 0,
                engine.acceptedCount > 0,
                engine.acceptedCount <= engine.proposalCount
            else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "draft activity")
            }
        } else {
            guard engine.proposalCount == 0, engine.acceptedCount == 0 else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "draft activity")
            }
        }
        _ = try checkedTokenSum(engine.requests, index: index)
        var maxE2E = 0.0
        for (requestIndex, request) in engine.requests.enumerated() {
            guard request.requestIndex == requestIndex,
                request.caseID == schedule.caseIDs[requestIndex]
            else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "request schedule")
            }
            guard let workloadCase = requiredWorkloadCaseByID[request.caseID] else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "request schedule")
            }
            try validateRequest(
                request,
                index: index,
                maxCompletionTokens: workloadCase.maxCompletionTokens)
            maxE2E = max(maxE2E, request.e2eSeconds)
        }
        guard engine.wallSeconds >= maxE2E else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "wall/request timing")
        }
    }

    private static func validateRequest(
        _ request: Qwen38MTPPerformanceScorecardRequestMeasurement,
        index: Int,
        maxCompletionTokens: Int
    ) throws {
        guard [request.promptSeconds, request.prefillSeconds, request.ttftSeconds,
               request.decodeSeconds, request.e2eSeconds].allSatisfy({ $0.isFinite && $0 > 0 }),
            request.decodeTokenCount > 0
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(index: index, reason: "timing")
        }
        guard request.promptSeconds <= request.prefillSeconds,
            request.prefillSeconds <= request.ttftSeconds,
            request.ttftSeconds <= request.e2eSeconds,
            request.ttftSeconds + request.decodeSeconds <= request.e2eSeconds
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "request timing")
        }
        guard request.decodeTokenCount <= maxCompletionTokens else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "max completion tokens")
        }
        guard request.decodeTokensPerSecond.isFinite, request.decodeTokensPerSecond > 0 else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(index: index, reason: "timing")
        }
        guard isLowerHex(request.outputDigest, count: 64),
            isLowerHex(request.cacheDigest, count: 64),
            isLowerHex(request.outputProvenanceID, count: 64),
            isLowerHex(request.cacheProvenanceID, count: 64)
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: index,
                reason: "output/cache provenance")
        }
    }

    private static func metrics(
        for concurrency: Int,
        measured: [Qwen38MTPPerformanceScorecardPair]
    ) throws -> Qwen38MTPPerformanceScorecardConcurrencyMetrics {
        guard measured.count == runPlan.measuredPairs else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunPlan
        }
        let candidateRequests = measured.flatMap(\.candidate.requests)
        let referenceRequests = measured.flatMap(\.reference.requests)
        let candidateTokens = try checkedTokenSum(candidateRequests, index: -1)
        let referenceTokens = try checkedTokenSum(referenceRequests, index: -1)
        let candidateWall = measured.reduce(0) { $0 + $1.candidate.wallSeconds }
        let referenceWall = measured.reduce(0) { $0 + $1.reference.wallSeconds }
        let aggregateCandidateThroughput = Double(candidateTokens) / candidateWall
        let aggregateReferenceThroughput = Double(referenceTokens) / referenceWall
        let candidate = Qwen38MTPPerformanceScorecardEngineMetrics(
            prompt: percentiles(candidateRequests.map(\.promptSeconds)),
            prefill: percentiles(candidateRequests.map(\.prefillSeconds)),
            ttft: percentiles(candidateRequests.map(\.ttftSeconds)),
            decodeTokensPerSecond: percentiles(candidateRequests.map(\.decodeTokensPerSecond)),
            e2e: percentiles(candidateRequests.map(\.e2eSeconds)),
            aggregateThroughputTokensPerSecond: aggregateCandidateThroughput,
            peakRSSBytes: measured.map(\.candidate.peakRSSBytes).max() ?? 0,
            peakMetalBytes: measured.map(\.candidate.peakMetalBytes).max() ?? 0)
        let reference = Qwen38MTPPerformanceScorecardEngineMetrics(
            prompt: percentiles(referenceRequests.map(\.promptSeconds)),
            prefill: percentiles(referenceRequests.map(\.prefillSeconds)),
            ttft: percentiles(referenceRequests.map(\.ttftSeconds)),
            decodeTokensPerSecond: percentiles(referenceRequests.map(\.decodeTokensPerSecond)),
            e2e: percentiles(referenceRequests.map(\.e2eSeconds)),
            aggregateThroughputTokensPerSecond: aggregateReferenceThroughput,
            peakRSSBytes: measured.map(\.reference.peakRSSBytes).max() ?? 0,
            peakMetalBytes: measured.map(\.reference.peakMetalBytes).max() ?? 0)

        return Qwen38MTPPerformanceScorecardConcurrencyMetrics(
            concurrency: concurrency,
            candidate: candidate,
            reference: reference,
            candidateTTFT: candidate.ttft,
            referenceTTFT: reference.ttft,
            candidatePrefill: candidate.prefill,
            referencePrefill: reference.prefill,
            candidateDecodeTokensPerSecond: candidate.decodeTokensPerSecond,
            referenceDecodeTokensPerSecond: reference.decodeTokensPerSecond,
            candidateE2E: candidate.e2e,
            referenceE2E: reference.e2e,
            aggregateThroughputRatio: aggregateCandidateThroughput / aggregateReferenceThroughput,
            requestE2EP95LatencyRatio:
                nearestRank(referenceRequests.map(\.e2eSeconds), percentile: 0.95)
                / nearestRank(candidateRequests.map(\.e2eSeconds), percentile: 0.95),
            e2EPairedMedianLatencyRatio: median(matchedRequestRatios(measured) {
                $0.reference.e2eSeconds / $0.candidate.e2eSeconds
            }),
            decodeTokensPerSecondRatio: median(matchedRequestRatios(measured) {
                $0.candidate.decodeTokensPerSecond / $0.reference.decodeTokensPerSecond
            }),
            peakRSSBytes: max(candidate.peakRSSBytes, reference.peakRSSBytes),
            peakMetalBytes: max(candidate.peakMetalBytes, reference.peakMetalBytes),
            thermalStates: Array(Set(measured.flatMap {
                [
                    $0.candidate.thermalBefore,
                    $0.candidate.thermalAfter,
                    $0.reference.thermalBefore,
                    $0.reference.thermalAfter,
                ]
            })).sorted())
    }

    private struct MatchedRequest {
        let candidate: Qwen38MTPPerformanceScorecardRequestMeasurement
        let reference: Qwen38MTPPerformanceScorecardRequestMeasurement
    }

    private static func matchedRequestRatios(
        _ pairs: [Qwen38MTPPerformanceScorecardPair],
        _ ratio: (MatchedRequest) -> Double
    ) -> [Double] {
        pairs.flatMap { pair in
            zip(pair.candidate.requests, pair.reference.requests).map {
                ratio(MatchedRequest(candidate: $0.0, reference: $0.1))
            }
        }
    }

    private static func chronologicalHalfRatios(
        _ pairs: [Qwen38MTPPerformanceScorecardPair]
    ) throws -> (first: Qwen38MTPPerformanceScorecardHalfRatios, second: Qwen38MTPPerformanceScorecardHalfRatios) {
        let solo = pairs.filter { $0.concurrency == 1 && !$0.warmup }
        let c2 = pairs.filter { $0.concurrency == 2 && !$0.warmup }
        let c4 = pairs.filter { $0.concurrency == 4 && !$0.warmup }
        let first = try halfRatios(solo: firstHalf(solo), c2: firstHalf(c2), c4: firstHalf(c4))
        let second = try halfRatios(solo: secondHalf(solo), c2: secondHalf(c2), c4: secondHalf(c4))
        return (first, second)
    }

    private static func halfRatios(
        solo: [Qwen38MTPPerformanceScorecardPair],
        c2: [Qwen38MTPPerformanceScorecardPair],
        c4: [Qwen38MTPPerformanceScorecardPair]
    ) throws -> Qwen38MTPPerformanceScorecardHalfRatios {
        Qwen38MTPPerformanceScorecardHalfRatios(
            soloE2ELatency: median(matchedRequestRatios(solo) {
                $0.reference.e2eSeconds / $0.candidate.e2eSeconds
            }),
            soloDecodeTokensPerSecond: median(matchedRequestRatios(solo) {
                $0.candidate.decodeTokensPerSecond / $0.reference.decodeTokensPerSecond
            }),
            soloTTFTLatency: requestP95LatencyRatio(solo, \.ttftSeconds),
            c2AggregateThroughput: try aggregateThroughputRatio(c2),
            c2RequestE2EP95Latency: requestP95LatencyRatio(c2, \.e2eSeconds),
            c4AggregateThroughput: try aggregateThroughputRatio(c4),
            c4RequestE2EP95Latency: requestP95LatencyRatio(c4, \.e2eSeconds))
    }

    private static func aggregateThroughputRatio(
        _ pairs: [Qwen38MTPPerformanceScorecardPair]
    ) throws -> Double {
        guard !pairs.isEmpty else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidRunPlan
        }
        let candidateTokens = try checkedTokenSum(pairs.flatMap(\.candidate.requests), index: -1)
        let referenceTokens = try checkedTokenSum(pairs.flatMap(\.reference.requests), index: -1)
        let candidateWall = pairs.reduce(0) { $0 + $1.candidate.wallSeconds }
        let referenceWall = pairs.reduce(0) { $0 + $1.reference.wallSeconds }
        return (Double(candidateTokens) / candidateWall)
            / (Double(referenceTokens) / referenceWall)
    }

    private static func requestP95LatencyRatio(
        _ pairs: [Qwen38MTPPerformanceScorecardPair],
        _ keyPath: KeyPath<Qwen38MTPPerformanceScorecardRequestMeasurement, Double>
    ) -> Double {
        let reference = pairs.flatMap(\.reference.requests).map { $0[keyPath: keyPath] }
        let candidate = pairs.flatMap(\.candidate.requests).map { $0[keyPath: keyPath] }
        return nearestRank(reference, percentile: 0.95) / nearestRank(candidate, percentile: 0.95)
    }

    private static func firstHalf<T>(_ values: [T]) -> [T] {
        Array(values.prefix(values.count / 2))
    }

    private static func secondHalf<T>(_ values: [T]) -> [T] {
        Array(values.dropFirst(values.count / 2))
    }

    private static func percentiles(
        _ values: [Double]
    ) -> Qwen38MTPPerformanceScorecardPercentiles {
        Qwen38MTPPerformanceScorecardPercentiles(
            p50: median(values),
            p95: nearestRank(values, percentile: 0.95))
    }

    private static func artifact(
        from lock: QwenMTPArtifactLock
    ) -> Qwen38MTPPerformanceScorecardArtifact {
        Qwen38MTPPerformanceScorecardArtifact(
            lockSourceRevision: lock.sourceRevision,
            targetRevision: lock.targetIdentity.revision,
            drafterRevision: lock.drafterIdentity.revision,
            targetConfigSHA256: lock.targetIdentity.configSHA256,
            drafterConfigSHA256: lock.drafterIdentity.configSHA256,
            tokenizerSHA256: lock.targetIdentity.tokenizerSHA256,
            targetTensorManifestSHA256: lock.targetIdentity.tensorManifestSHA256,
            drafterTensorManifestSHA256: lock.drafterIdentity.tensorManifestSHA256,
            targetQuantizationBits: lock.targetQuantization.bits,
            targetQuantizationGroupSize: lock.targetQuantization.groupSize,
            targetQuantizationMode: lock.targetQuantization.mode,
            drafterQuantizationBits: lock.drafterQuantization.bits,
            drafterQuantizationGroupSize: lock.drafterQuantization.groupSize,
            drafterQuantizationMode: lock.drafterQuantization.mode,
            depth: 1,
            blockSize: 3,
            maxAcceptedDrafts: 2)
    }

    private static func makeSchedules() -> [Qwen38MTPPerformanceScorecardPairSchedule] {
        let ids = requiredWorkloadCases.map(\.id)
        let orders = makeOrders()
        return [1, 2, 4].flatMap { concurrency in
            (0..<orders.count).map { pairIndex in
                Qwen38MTPPerformanceScorecardPairSchedule(
                    concurrency: concurrency,
                    pairIndex: pairIndex,
                    order: orders[pairIndex],
                    caseIDs: (0..<concurrency).map {
                        ids[(pairIndex + $0 + concurrency) % ids.count]
                    })
            }
        }
    }

    private static func makeOrders() -> [Qwen38MTPPerformanceScorecardRunOrder] {
        (0..<42).map {
            $0.isMultiple(of: 2) ? .candidateThenReference : .referenceThenCandidate
        }
    }

    private static func validateProvenance(
        _ provenance: Provenance,
        evidence: Qwen38MTPPerformanceScorecardEvidence,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws {
        try validateProvenance(provenance, trustedRunIdentity: trustedRunIdentity)
        guard evidence.hardware.chip == trustedRunIdentity.hardwareChip,
            evidence.hardware.osBuild == trustedRunIdentity.hardwareOSBuild,
            evidence.hardware.hostIdentityDigest == trustedRunIdentity.hostIdentityDigest
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("hardware")
        }
        guard provenance.corpusContentHash == evidence.workload.contentSHA256 else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("workload")
        }
    }

    private static func validateProvenance(
        _ provenance: Provenance,
        trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    ) throws {
        guard provenance.hardwareChip == trustedRunIdentity.hardwareChip,
            provenance.hardwareRAMBytes == trustedRunIdentity.hardwareRAMBytes,
            provenance.hardwareOS == trustedRunIdentity.hardwareOSBuild
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("hardware")
        }
        guard provenance.harnessGitSHA == trustedRunIdentity.harnessGitSHA,
            isLowerHex(provenance.harnessGitSHA, count: 40),
            !provenance.date.isEmpty,
            !provenance.nonce.isEmpty
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("record")
        }
        guard provenance.modelPath == trustedRunIdentity.modelLabel,
            provenance.modelConfigHash == trustedRunIdentity.modelConfigHash,
            provenance.modelCheckpointManifestHash == trustedRunIdentity.modelCheckpointManifestHash,
            provenance.modelQuant == trustedRunIdentity.modelQuant,
            provenance.mlxSwiftVersion == trustedRunIdentity.candidateMLXSwiftVersion,
            provenance.referenceMLXVersion == trustedRunIdentity.referenceMLXVersion,
            provenance.referenceMLXLMVersion == trustedRunIdentity.referenceMLXLMVersion
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("model")
        }
        guard provenance.corpusId == trustedRunIdentity.corpusID,
            provenance.corpusContentHash == trustedRunIdentity.corpusContentHash
        else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidProvenance("workload")
        }
    }

    private static func checkedTokenSum(
        _ requests: [Qwen38MTPPerformanceScorecardRequestMeasurement],
        index: Int
    ) throws -> Int {
        var total = 0
        for request in requests {
            let result = total.addingReportingOverflow(request.decodeTokenCount)
            guard !result.overflow else {
                throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                    index: index,
                    reason: "token overflow")
            }
            total = result.partialValue
        }
        return total
    }

    private static func nearestRank(_ values: [Double], percentile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let raw = Int(ceil(percentile * Double(sorted.count)))
        let index = min(max(raw - 1, 0), sorted.count - 1)
        return sorted[index]
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func isAllowedThermal(_ value: String) -> Bool {
        value == "nominal" || value == "fair"
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
