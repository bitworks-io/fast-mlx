import CryptoKit
import Foundation

public struct Qwen38MTPLiveExactnessArrayFingerprint: Codable, Equatable, Sendable {
    public var stateIndex: Int
    public var shape: [Int]
    public var dtype: String
    public var byteCount: Int
    public var sha256: String

    public init(
        stateIndex: Int,
        shape: [Int],
        dtype: String,
        byteCount: Int,
        sha256: String
    ) {
        self.stateIndex = stateIndex
        self.shape = shape
        self.dtype = dtype
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct Qwen38MTPLiveExactnessCacheFingerprint: Codable, Equatable, Sendable {
    public var layerIndex: Int
    public var cacheType: String
    public var offset: Int
    public var metaStateSHA256: String
    public var stateFingerprints: [Qwen38MTPLiveExactnessArrayFingerprint]

    public init(
        layerIndex: Int,
        cacheType: String,
        offset: Int,
        metaStateSHA256: String,
        stateFingerprints: [Qwen38MTPLiveExactnessArrayFingerprint]
    ) {
        self.layerIndex = layerIndex
        self.cacheType = cacheType
        self.offset = offset
        self.metaStateSHA256 = metaStateSHA256
        self.stateFingerprints = stateFingerprints
    }
}

public struct Qwen38MTPLiveExactnessCaseSpec: Codable, Equatable, Sendable {
    public let id: String
    public let promptSHA256: String
    public let maxTokens: Int

    public init(id: String, promptSHA256: String, maxTokens: Int) {
        self.id = id
        self.promptSHA256 = promptSHA256
        self.maxTokens = maxTokens
    }
}

public struct Qwen38MTPLiveExactnessCaseEvidence: Codable, Equatable, Sendable {
    public var id: String
    public var promptSHA256: String
    public var maxTokens: Int
    public var scalarTokenIDs: [Int]
    public var mtpTokenIDs: [Int]
    public var scalarDecodedUTF8Base64: String
    public var mtpDecodedUTF8Base64: String
    public var decodedUTF8SHA256: String
    public var proposedDraftTokens: Int
    public var acceptedDraftTokens: Int
    public var passthroughReason: String?
    public var scalarCacheFingerprints: [Qwen38MTPLiveExactnessCacheFingerprint]
    public var mtpCacheFingerprints: [Qwen38MTPLiveExactnessCacheFingerprint]

    public init(
        id: String,
        promptSHA256: String,
        maxTokens: Int,
        scalarTokenIDs: [Int],
        mtpTokenIDs: [Int],
        scalarDecodedUTF8Base64: String,
        mtpDecodedUTF8Base64: String,
        decodedUTF8SHA256: String,
        proposedDraftTokens: Int,
        acceptedDraftTokens: Int,
        passthroughReason: String?,
        scalarCacheFingerprints: [Qwen38MTPLiveExactnessCacheFingerprint],
        mtpCacheFingerprints: [Qwen38MTPLiveExactnessCacheFingerprint]
    ) {
        self.id = id
        self.promptSHA256 = promptSHA256
        self.maxTokens = maxTokens
        self.scalarTokenIDs = scalarTokenIDs
        self.mtpTokenIDs = mtpTokenIDs
        self.scalarDecodedUTF8Base64 = scalarDecodedUTF8Base64
        self.mtpDecodedUTF8Base64 = mtpDecodedUTF8Base64
        self.decodedUTF8SHA256 = decodedUTF8SHA256
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.passthroughReason = passthroughReason
        self.scalarCacheFingerprints = scalarCacheFingerprints
        self.mtpCacheFingerprints = mtpCacheFingerprints
    }
}

public struct Qwen38MTPLiveExactnessSourceIdentity: Codable, Equatable, Sendable {
    public var selection: String
    public var targetRepositoryID: String
    public var targetRevision: String
    public var drafterRepositoryID: String
    public var drafterRevision: String
    public var lockSourceRevision: String
    public var artifactID: String
    public var sourceID: String

    public init(
        selection: String,
        targetRepositoryID: String,
        targetRevision: String,
        drafterRepositoryID: String,
        drafterRevision: String,
        lockSourceRevision: String,
        artifactID: String,
        sourceID: String
    ) {
        self.selection = selection
        self.targetRepositoryID = targetRepositoryID
        self.targetRevision = targetRevision
        self.drafterRepositoryID = drafterRepositoryID
        self.drafterRevision = drafterRevision
        self.lockSourceRevision = lockSourceRevision
        self.artifactID = artifactID
        self.sourceID = sourceID
    }
}

public struct Qwen38MTPLiveExactnessProcessIsolationEvidence: Codable, Equatable, Sendable {
    public var processID: Int
    public var parentProcessID: Int
    public var processStartUptimeNanoseconds: UInt64
    public var bootTimeUnixSeconds: Int64
    public var executableIdentitySource: Qwen38MTPLiveExactnessExecutableIdentitySource
    public var executableSHA256: String
    public var harnessGitSHA: String
    public var sourceID: String
    public var gdnMode: Qwen38MTPPerformanceScorecardGDNMode
    public var observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv

    public init(
        processID: Int,
        parentProcessID: Int,
        processStartUptimeNanoseconds: UInt64,
        bootTimeUnixSeconds: Int64,
        executableIdentitySource: Qwen38MTPLiveExactnessExecutableIdentitySource,
        executableSHA256: String,
        harnessGitSHA: String,
        sourceID: String,
        gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv
    ) {
        self.processID = processID
        self.parentProcessID = parentProcessID
        self.processStartUptimeNanoseconds = processStartUptimeNanoseconds
        self.bootTimeUnixSeconds = bootTimeUnixSeconds
        self.executableIdentitySource = executableIdentitySource
        self.executableSHA256 = executableSHA256
        self.harnessGitSHA = harnessGitSHA
        self.sourceID = sourceID
        self.gdnMode = gdnMode
        self.observedEnv = observedEnv
    }
}

public enum Qwen38MTPLiveExactnessExecutableIdentitySource: String, Codable, Equatable, Sendable {
    case procPIDPath = "proc_pidpath"
    case argumentVector = "argument-vector"
}

public struct Qwen38MTPLiveExactnessMLXMemoryBudget: Codable, Equatable, Sendable {
    public var memoryLimitBytes: Int
    public var cacheLimitBytes: Int

    public init(memoryLimitBytes: Int, cacheLimitBytes: Int) {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
    }
}

public enum Qwen38MTPLiveExactnessObservationProvenance: String, Codable, Equatable, Sendable {
    case measured
    case synthesized
}

public struct Qwen38MTPLiveExactnessHostMemoryObservation: Codable, Equatable, Sendable {
    public var hostUse: String
    public var hostUseSource: String
    public var hostUsePolicyVersion: String
    public var physicalRAMBytes: UInt64
    public var wiredLimitMB: Int
    public var wiredLimitProvenance: Qwen38MTPLiveExactnessObservationProvenance
    public var metalRecommendedMaxWorkingSetSizeBytes: UInt64
    public var metalCurrentAllocatedSizeBytes: UInt64
    public var memoryLimitBytes: UInt64
    public var cacheLimitBytes: UInt64
    public var reservedKVBytes: UInt64
    public var reservedIOBytes: UInt64
    public var reservedPrefetchBytes: UInt64
    public var osServiceReserveBytes: UInt64

    public init(
        hostUse: String,
        hostUseSource: String,
        hostUsePolicyVersion: String,
        physicalRAMBytes: UInt64,
        wiredLimitMB: Int,
        wiredLimitProvenance: Qwen38MTPLiveExactnessObservationProvenance,
        metalRecommendedMaxWorkingSetSizeBytes: UInt64,
        metalCurrentAllocatedSizeBytes: UInt64,
        memoryLimitBytes: UInt64,
        cacheLimitBytes: UInt64,
        reservedKVBytes: UInt64,
        reservedIOBytes: UInt64,
        reservedPrefetchBytes: UInt64,
        osServiceReserveBytes: UInt64
    ) {
        self.hostUse = hostUse
        self.hostUseSource = hostUseSource
        self.hostUsePolicyVersion = hostUsePolicyVersion
        self.physicalRAMBytes = physicalRAMBytes
        self.wiredLimitMB = wiredLimitMB
        self.wiredLimitProvenance = wiredLimitProvenance
        self.metalRecommendedMaxWorkingSetSizeBytes = metalRecommendedMaxWorkingSetSizeBytes
        self.metalCurrentAllocatedSizeBytes = metalCurrentAllocatedSizeBytes
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.reservedKVBytes = reservedKVBytes
        self.reservedIOBytes = reservedIOBytes
        self.reservedPrefetchBytes = reservedPrefetchBytes
        self.osServiceReserveBytes = osServiceReserveBytes
    }

    public var mlxMemoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget {
        Qwen38MTPLiveExactnessMLXMemoryBudget(
            memoryLimitBytes: Int(memoryLimitBytes),
            cacheLimitBytes: Int(cacheLimitBytes))
    }
}

public struct Qwen38MTPLiveExactnessEvidence: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var artifact: Qwen38MTPPerformanceScorecardArtifact
    public var artifactID: String
    public var source: Qwen38MTPLiveExactnessSourceIdentity
    public var gdnMode: Qwen38MTPPerformanceScorecardGDNMode
    public var launchBinding: Qwen38MTPPerformanceScorecardLaunchBinding
    public var processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    public var mlxMemoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget
    public var hostMemoryObservation: Qwen38MTPLiveExactnessHostMemoryObservation
    public var cases: [Qwen38MTPLiveExactnessCaseEvidence]

    public init(
        schemaVersion: Int,
        artifact: Qwen38MTPPerformanceScorecardArtifact,
        artifactID: String,
        source: Qwen38MTPLiveExactnessSourceIdentity,
        gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
        launchBinding: Qwen38MTPPerformanceScorecardLaunchBinding,
        processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence,
        mlxMemoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget,
        hostMemoryObservation: Qwen38MTPLiveExactnessHostMemoryObservation,
        cases: [Qwen38MTPLiveExactnessCaseEvidence]
    ) {
        self.schemaVersion = schemaVersion
        self.artifact = artifact
        self.artifactID = artifactID
        self.source = source
        self.gdnMode = gdnMode
        self.launchBinding = launchBinding
        self.processIsolation = processIsolation
        self.mlxMemoryBudget = mlxMemoryBudget
        self.hostMemoryObservation = hostMemoryObservation
        self.cases = cases
    }

}

public enum Qwen38MTPLiveExactnessGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case schemaVersionMismatch(Int)
    case invalidArtifactIdentity
    case invalidSourceIdentity
    case invalidCaseOrder
    case invalidCase(String, String)
    case malformedJSONL(line: Int)
    case unterminatedJSONL
    case invalidRecordCardinality(Int)
    case wrongSubcommand(String)
    case invalidProvenance(String)
    case nonCanonicalJSONL
    case invalidLaunchBinding(String)
    case invalidMemoryBudget(String)
    case invalidHostMemoryObservation(String)

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let value): return "schemaVersion mismatch: \(value)"
        case .invalidArtifactIdentity: return "invalid artifact identity"
        case .invalidSourceIdentity: return "invalid source identity"
        case .invalidCaseOrder: return "invalid live exactness case order"
        case .invalidCase(let id, let field): return "invalid live exactness case \(id): \(field)"
        case .malformedJSONL(let line): return "malformed JSONL at line \(line)"
        case .unterminatedJSONL: return "unterminated JSONL"
        case .invalidRecordCardinality(let count): return "invalid JSONL record cardinality: \(count)"
        case .wrongSubcommand(let subcommand): return "wrong subcommand: \(subcommand)"
        case .invalidProvenance(let field): return "invalid provenance: \(field)"
        case .nonCanonicalJSONL: return "non-canonical JSONL"
        case .invalidLaunchBinding(let field): return "invalid launch binding: \(field)"
        case .invalidMemoryBudget(let field): return "invalid memory budget: \(field)"
        case .invalidHostMemoryObservation(let field):
            return "invalid host memory observation: \(field)"
        }
    }
}

private struct Qwen38MTPLiveExactnessSchemaProbe: Codable, Sendable {
    let schemaVersion: Int
}

private struct Qwen38MTPLiveExactnessSourceBasis: Codable, Equatable, Sendable {
    let selection: String
    let targetRepositoryID: String
    let targetRevision: String
    let drafterRepositoryID: String
    let drafterRevision: String
    let lockSourceRevision: String
    let artifactID: String
    let cases: [Qwen38MTPLiveExactnessCaseSpec]
}

public enum Qwen38MTPLiveExactnessGate {
    public static let schemaVersion = 2
    public static let subcommand = "qwen38-mtp-live-exactness"
    public static let requiredHostUsePolicyVersion = "host-use/v1"
    public static let modelPathSentinel = "qwen38-27b-mxfp8-depth1-live-exactness"
    public static let requiredCases: [Qwen38MTPLiveExactnessCaseSpec] = [
        makeCaseSpec(
            id: "numbers",
            prompt: "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,",
            maxTokens: 12),
        makeCaseSpec(
            id: "sentence",
            prompt: "Complete this sentence in a few words: The fastest reliable test is",
            maxTokens: 12),
    ]
    public static let requiredCasesByID = Dictionary(
        uniqueKeysWithValues: requiredCases.map { ($0.id, $0) })
    public static let requiredCaseIDs = requiredCases.map(\.id)
    public static let requiredCacheLayerCount = 64
    public static let requiredArtifact = Qwen38MTPPerformanceScorecardGate.requiredArtifact
    public static let requiredArtifactID = sha256Hex(canonicalData(requiredArtifact))
    public static let requiredSourceIdentity = makeRequiredSourceIdentity()

    public static func validateJSONL(
        _ data: Data
    ) throws -> Qwen38MTPPerformanceScorecardLiveExactnessProof {
        guard data.last == 0x0a else {
            throw Qwen38MTPLiveExactnessGateError.unterminatedJSONL
        }
        let rows = data.split(separator: 0x0a, omittingEmptySubsequences: false).dropLast()
        guard rows.count == 1 else {
            throw Qwen38MTPLiveExactnessGateError.invalidRecordCardinality(rows.count)
        }
        guard let row = rows.first, !row.isEmpty else {
            throw Qwen38MTPLiveExactnessGateError.malformedJSONL(line: 1)
        }

        let rowData = Data(row)
        let decoder = JSONDecoder()
        let probe: ResultRecord<Qwen38MTPLiveExactnessSchemaProbe>
        do {
            probe = try decoder.decode(
                ResultRecord<Qwen38MTPLiveExactnessSchemaProbe>.self,
                from: rowData)
        } catch {
            throw Qwen38MTPLiveExactnessGateError.malformedJSONL(line: 1)
        }
        guard probe.subcommand == subcommand else {
            throw Qwen38MTPLiveExactnessGateError.wrongSubcommand(probe.subcommand)
        }
        guard probe.payload.schemaVersion == schemaVersion else {
            throw Qwen38MTPLiveExactnessGateError.schemaVersionMismatch(
                probe.payload.schemaVersion)
        }

        let record: ResultRecord<Qwen38MTPLiveExactnessEvidence>
        do {
            record = try decoder.decode(
                ResultRecord<Qwen38MTPLiveExactnessEvidence>.self,
                from: rowData)
        } catch {
            throw Qwen38MTPLiveExactnessGateError.malformedJSONL(line: 1)
        }
        guard try record.jsonLine() == String(decoding: rowData, as: UTF8.self) else {
            throw Qwen38MTPLiveExactnessGateError.nonCanonicalJSONL
        }
        try validateProvenance(record.provenance)
        try validateEvidence(record.payload, provenance: record.provenance)
        return Qwen38MTPPerformanceScorecardLiveExactnessProof(
            artifact: requiredArtifact,
            artifactID: requiredArtifactID,
            sourceID: requiredSourceIdentity.sourceID,
            evidenceID: sha256Hex(rowData),
            accepted: true,
            gdnMode: .gdnOn,
            launchBinding: record.payload.launchBinding)
    }

    private static func validateEvidence(
        _ evidence: Qwen38MTPLiveExactnessEvidence,
        provenance: Provenance
    ) throws {
        guard evidence.schemaVersion == schemaVersion else {
            throw Qwen38MTPLiveExactnessGateError.schemaVersionMismatch(evidence.schemaVersion)
        }
        guard evidence.artifact == requiredArtifact,
            evidence.artifactID == requiredArtifactID,
            isLowerHex(evidence.artifactID, count: 64)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidArtifactIdentity
        }
        guard evidence.source == requiredSourceIdentity,
            evidence.source.artifactID == requiredArtifactID,
            evidence.source.sourceID == sourceID(for: requiredSourceBasis),
            isLowerHex(evidence.source.sourceID, count: 64)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidSourceIdentity
        }
        guard evidence.processIsolation.harnessGitSHA == provenance.harnessGitSHA else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance(
                "processIsolation.harnessGitSHA")
        }
        try validateLaunchBinding(evidence)
        try validateMemoryBudget(evidence.mlxMemoryBudget)
        try validateHostMemoryObservation(
            evidence.hostMemoryObservation,
            memoryBudget: evidence.mlxMemoryBudget)
        guard evidence.cases.map(\.id) == requiredCaseIDs else {
            throw Qwen38MTPLiveExactnessGateError.invalidCaseOrder
        }
        for liveCase in evidence.cases {
            try validateCase(liveCase)
        }
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func processIsolationEvidenceID(
        for evidence: Qwen38MTPLiveExactnessProcessIsolationEvidence
    ) -> String {
        sha256Hex(canonicalData(evidence))
    }

    private static let requiredSourceBasis = Qwen38MTPLiveExactnessSourceBasis(
        selection: "qwen38_27BMXFP8Depth1",
        targetRepositoryID: "mlx-community/Qwen3.8-27B-mxfp8",
        targetRevision: requiredArtifact.targetRevision,
        drafterRepositoryID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
        drafterRevision: requiredArtifact.drafterRevision,
        lockSourceRevision: requiredArtifact.lockSourceRevision,
        artifactID: requiredArtifactID,
        cases: requiredCases)

    private static func makeRequiredSourceIdentity() -> Qwen38MTPLiveExactnessSourceIdentity {
        Qwen38MTPLiveExactnessSourceIdentity(
            selection: requiredSourceBasis.selection,
            targetRepositoryID: requiredSourceBasis.targetRepositoryID,
            targetRevision: requiredSourceBasis.targetRevision,
            drafterRepositoryID: requiredSourceBasis.drafterRepositoryID,
            drafterRevision: requiredSourceBasis.drafterRevision,
            lockSourceRevision: requiredSourceBasis.lockSourceRevision,
            artifactID: requiredSourceBasis.artifactID,
            sourceID: sourceID(for: requiredSourceBasis))
    }

    private static func sourceID(for basis: Qwen38MTPLiveExactnessSourceBasis) -> String {
        sha256Hex(canonicalData(basis))
    }

    private static func makeCaseSpec(
        id: String,
        prompt: String,
        maxTokens: Int
    ) -> Qwen38MTPLiveExactnessCaseSpec {
        Qwen38MTPLiveExactnessCaseSpec(
            id: id,
            promptSHA256: sha256Hex(Data(prompt.utf8)),
            maxTokens: maxTokens)
    }

    private static func validateCase(
        _ liveCase: Qwen38MTPLiveExactnessCaseEvidence
    ) throws {
        guard let requiredCase = requiredCasesByID[liveCase.id],
            liveCase.promptSHA256 == requiredCase.promptSHA256,
            liveCase.maxTokens == requiredCase.maxTokens
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "workload")
        }
        guard !liveCase.scalarTokenIDs.isEmpty,
            !liveCase.mtpTokenIDs.isEmpty,
            liveCase.scalarTokenIDs == liveCase.mtpTokenIDs,
            liveCase.scalarTokenIDs.allSatisfy({ $0 >= 0 })
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "tokens")
        }

        guard let scalarBytes = Data(base64Encoded: liveCase.scalarDecodedUTF8Base64),
            let mtpBytes = Data(base64Encoded: liveCase.mtpDecodedUTF8Base64),
            !scalarBytes.isEmpty,
            scalarBytes == mtpBytes,
            sha256Hex(scalarBytes) == liveCase.decodedUTF8SHA256,
            isLowerHex(liveCase.decodedUTF8SHA256, count: 64)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "decoded")
        }

        guard liveCase.proposedDraftTokens > 0,
            liveCase.acceptedDraftTokens > 0,
            liveCase.acceptedDraftTokens <= liveCase.proposedDraftTokens
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "drafts")
        }
        guard liveCase.passthroughReason == nil else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "passthrough")
        }
        guard validCacheFingerprints(liveCase.scalarCacheFingerprints),
            liveCase.scalarCacheFingerprints == liveCase.mtpCacheFingerprints
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidCase(liveCase.id, "cache")
        }
    }

    private static func validateLaunchBinding(
        _ evidence: Qwen38MTPLiveExactnessEvidence
    ) throws {
        guard evidence.gdnMode == .gdnOn else {
            throw Qwen38MTPLiveExactnessGateError.invalidLaunchBinding("gdnMode")
        }
        guard validProcessIsolation(evidence.processIsolation) else {
            throw Qwen38MTPLiveExactnessGateError.invalidLaunchBinding("processIsolation")
        }
        let processIsolationEvidenceID = processIsolationEvidenceID(
            for: evidence.processIsolation)
        guard evidence.launchBinding.mode == .gdnOn,
            evidence.launchBinding.sourceDigest == requiredSourceIdentity.sourceID,
            evidence.launchBinding.observedEnv == .enabled,
            evidence.launchBinding.processIsolationEvidenceID == processIsolationEvidenceID,
            isLowerHex(evidence.launchBinding.processIsolationEvidenceID, count: 64)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidLaunchBinding(
                "processIsolationEvidenceID")
        }
        let expectedLaunchDigest = Qwen38MTPPerformanceScorecardGate.launchDigest(
            mode: .gdnOn,
            sourceDigest: requiredSourceIdentity.sourceID,
            observedEnv: .enabled,
            processIsolationEvidenceID: processIsolationEvidenceID)
        guard evidence.launchBinding.launchDigest == expectedLaunchDigest,
            isLowerHex(evidence.launchBinding.launchDigest, count: 64)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidLaunchBinding("launchDigest")
        }
    }

    private static func validateMemoryBudget(
        _ budget: Qwen38MTPLiveExactnessMLXMemoryBudget
    ) throws {
        guard budget.memoryLimitBytes > 0 else {
            throw Qwen38MTPLiveExactnessGateError.invalidMemoryBudget("memoryLimitBytes")
        }
        guard budget.cacheLimitBytes > 0,
            budget.cacheLimitBytes <= budget.memoryLimitBytes
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidMemoryBudget("cacheLimitBytes")
        }
    }

    private static func validateHostMemoryObservation(
        _ observation: Qwen38MTPLiveExactnessHostMemoryObservation,
        memoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget
    ) throws {
        guard observation.hostUse == "dedicated-serving",
            observation.hostUseSource == "operator-assertion",
            observation.hostUsePolicyVersion == requiredHostUsePolicyVersion
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("hostUse")
        }
        guard observation.physicalRAMBytes == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("physicalRAMBytes")
        }
        guard observation.wiredLimitMB >= 0,
            observation.wiredLimitProvenance == .measured
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("wiredLimit")
        }
        let wiredLimitBytes = observation.wiredLimitMB == 0
            ? Optional(UInt64.max)
            : mebibytesToBytes(observation.wiredLimitMB)
        guard let wiredLimitBytes else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("wiredLimit")
        }
        guard observation.metalRecommendedMaxWorkingSetSizeBytes > 0,
            observation.metalCurrentAllocatedSizeBytes
                < observation.metalRecommendedMaxWorkingSetSizeBytes
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("metal")
        }
        guard UInt64(memoryBudget.memoryLimitBytes) == observation.memoryLimitBytes,
            UInt64(memoryBudget.cacheLimitBytes) == observation.cacheLimitBytes
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("memoryBudget")
        }
        guard observation.memoryLimitBytes > 0,
            observation.cacheLimitBytes > 0,
            observation.cacheLimitBytes <= observation.memoryLimitBytes,
            observation.reservedKVBytes > 0,
            observation.reservedIOBytes > 0,
            observation.reservedPrefetchBytes > 0,
            observation.osServiceReserveBytes > 0
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("budget")
        }
        guard let memoryResident = checkedSum([
            observation.cacheLimitBytes,
            observation.reservedKVBytes,
        ]),
            memoryResident <= observation.memoryLimitBytes
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("budget")
        }
        let effectiveCeiling = min(
            observation.physicalRAMBytes,
            min(wiredLimitBytes, observation.metalRecommendedMaxWorkingSetSizeBytes))
        guard let totalPlanned = checkedSum([
            observation.metalCurrentAllocatedSizeBytes,
            observation.memoryLimitBytes,
            observation.reservedIOBytes,
            observation.reservedPrefetchBytes,
            observation.osServiceReserveBytes,
        ]),
            totalPlanned <= effectiveCeiling
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidHostMemoryObservation("budget")
        }
    }

    private static func validProcessIsolation(
        _ evidence: Qwen38MTPLiveExactnessProcessIsolationEvidence
    ) -> Bool {
        evidence.processID > 0
            && evidence.parentProcessID > 0
            && evidence.processStartUptimeNanoseconds > 0
            && evidence.bootTimeUnixSeconds > 0
            && evidence.executableIdentitySource == .procPIDPath
            && isLowerHex(evidence.executableSHA256, count: 64)
            && evidence.executableSHA256 != String(repeating: "0", count: 64)
            && isLowerHex(evidence.harnessGitSHA, count: 40)
            && evidence.harnessGitSHA != String(repeating: "0", count: 40)
            && evidence.sourceID == requiredSourceIdentity.sourceID
            && evidence.gdnMode == .gdnOn
            && evidence.observedEnv == .enabled
    }

    private static func validCacheFingerprints(
        _ fingerprints: [Qwen38MTPLiveExactnessCacheFingerprint]
    ) -> Bool {
        guard fingerprints.count == requiredCacheLayerCount else { return false }
        for (index, fingerprint) in fingerprints.enumerated() {
            let isDenseAttention = (index + 1).isMultiple(of: 4)
            let requiredCacheType = isDenseAttention ? "dense-attention" : "recurrent-mamba"
            guard fingerprint.layerIndex == index,
                fingerprint.cacheType == requiredCacheType,
                (isDenseAttention ? fingerprint.offset > 0 : fingerprint.offset == 0),
                isLowerHex(fingerprint.metaStateSHA256, count: 64),
                fingerprint.stateFingerprints.count == 2
            else {
                return false
            }
            for (stateIndex, state) in fingerprint.stateFingerprints.enumerated() {
                guard state.stateIndex == stateIndex,
                    validCacheStateGeometry(
                        state,
                        cacheType: requiredCacheType,
                        offset: fingerprint.offset),
                    isLowerHex(state.sha256, count: 64)
                else {
                    return false
                }
            }
        }
        return true
    }

    private static func validCacheStateGeometry(
        _ state: Qwen38MTPLiveExactnessArrayFingerprint,
        cacheType: String,
        offset: Int
    ) -> Bool {
        let expectedShape: [Int]
        let expectedDType: String
        switch (cacheType, state.stateIndex) {
        case ("recurrent-mamba", 0):
            expectedShape = [1, 3, 10_240]
            expectedDType = "bfloat16"
        case ("recurrent-mamba", 1):
            expectedShape = [1, 48, 128, 128]
            expectedDType = "float32"
        case ("dense-attention", 0), ("dense-attention", 1):
            expectedShape = [1, 4, offset, 256]
            expectedDType = "bfloat16"
        default:
            return false
        }
        let bytesPerElement = expectedDType == "float32" ? 4 : 2
        guard let expectedByteCount = exactElementCount(expectedShape).flatMap({ count in
            let (bytes, overflow) = count.multipliedReportingOverflow(by: bytesPerElement)
            return overflow ? nil : bytes
        }) else { return false }
        return state.shape == expectedShape
            && state.dtype == expectedDType
            && state.byteCount == expectedByteCount
    }

    private static func exactElementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape where dimension > 0 {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            if overflow { return nil }
            count = next
        }
        return shape.allSatisfy({ $0 > 0 }) ? count : nil
    }

    private static func checkedSum(_ values: [UInt64]) -> UInt64? {
        var total: UInt64 = 0
        for value in values {
            let result = total.addingReportingOverflow(value)
            if result.overflow { return nil }
            total = result.partialValue
        }
        return total
    }

    private static func mebibytesToBytes(_ value: Int) -> UInt64? {
        guard value > 0 else { return nil }
        return UInt64(value).multipliedReportingOverflow(by: 1024 * 1024).overflow
            ? nil
            : UInt64(value) * 1024 * 1024
    }

    private static func validateProvenance(_ provenance: Provenance) throws {
        guard ISO8601DateFormatter().date(from: provenance.date) != nil else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("date")
        }
        guard provenance.hardwareChip.contains("M3 Ultra"),
            provenance.hardwareRAMBytes == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
            !provenance.hardwareOS.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("hardware")
        }
        guard isLowerHex(provenance.harnessGitSHA, count: 40),
            provenance.harnessGitSHA != String(repeating: "0", count: 40)
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("harnessGitSHA")
        }
        guard provenance.mlxSwiftVersion == "0.31.6" else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("mlxSwiftVersion")
        }
        guard provenance.modelPath == modelPathSentinel else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("modelPath")
        }
        guard provenance.modelConfigHash == requiredArtifact.targetConfigSHA256 else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("modelConfigHash")
        }
        guard provenance.modelCheckpointManifestHash
            == requiredArtifact.targetTensorManifestSHA256
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance(
                "modelCheckpointManifestHash")
        }
        guard provenance.modelQuant == ModelQuantInfo(bits: 8, groupSize: 32) else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("modelQuant")
        }
        guard provenance.referenceMLXVersion == nil,
            provenance.referenceMLXLMVersion == nil,
            provenance.corpusId == nil,
            provenance.corpusContentHash == nil,
            provenance.nonce == requiredSourceIdentity.sourceID
        else {
            throw Qwen38MTPLiveExactnessGateError.invalidProvenance("identity")
        }
    }

    private static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func canonicalData<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try! encoder.encode(value)
    }
}
