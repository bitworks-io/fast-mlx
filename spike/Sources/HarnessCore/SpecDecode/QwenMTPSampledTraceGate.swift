import CryptoKit
import Foundation

public enum QwenMTPSampledTraceBranch: String, Codable, Sendable, Equatable {
    case accepted
    case rejected
}

public struct QwenMTPSampledTraceSource: Codable, Sendable, Equatable {
    public var targetModelID: String
    public var targetRevision: String
    public var targetConfigSHA256: String
    public var targetTensorManifestSHA256: String
    public var drafterModelID: String
    public var drafterRevision: String
    public var drafterConfigSHA256: String
    public var drafterTensorManifestSHA256: String
    public var sourceRevision: String
    public var runtimeBlockSize: Int
    public var maximumAcceptedDraftTokens: Int

    public init(
        targetModelID: String,
        targetRevision: String,
        targetConfigSHA256: String,
        targetTensorManifestSHA256: String,
        drafterModelID: String,
        drafterRevision: String,
        drafterConfigSHA256: String,
        drafterTensorManifestSHA256: String,
        sourceRevision: String,
        runtimeBlockSize: Int,
        maximumAcceptedDraftTokens: Int
    ) {
        self.targetModelID = targetModelID
        self.targetRevision = targetRevision
        self.targetConfigSHA256 = targetConfigSHA256
        self.targetTensorManifestSHA256 = targetTensorManifestSHA256
        self.drafterModelID = drafterModelID
        self.drafterRevision = drafterRevision
        self.drafterConfigSHA256 = drafterConfigSHA256
        self.drafterTensorManifestSHA256 = drafterTensorManifestSHA256
        self.sourceRevision = sourceRevision
        self.runtimeBlockSize = runtimeBlockSize
        self.maximumAcceptedDraftTokens = maximumAcceptedDraftTokens
    }
}

public struct QwenMTPSampledTraceSampling: Codable, Sendable, Equatable {
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var minP: Double
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var probabilityDType: String
    public var probabilityNormalization: String
    public var rejectionProposalSelection: String
    public var proposalContext: String
    public var drawOrder: [String]

    public init(
        temperature: Double,
        topP: Double,
        topK: Int,
        minP: Double,
        repetitionPenalty: Double?,
        presencePenalty: Double?,
        frequencyPenalty: Double?,
        probabilityDType: String,
        probabilityNormalization: String,
        rejectionProposalSelection: String,
        proposalContext: String,
        drawOrder: [String]
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.probabilityDType = probabilityDType
        self.probabilityNormalization = probabilityNormalization
        self.rejectionProposalSelection = rejectionProposalSelection
        self.proposalContext = proposalContext
        self.drawOrder = drawOrder
    }
}

public struct QwenMTPSampledTraceArrayFingerprint: Codable, Sendable, Equatable {
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

public struct QwenMTPSampledTraceCacheFingerprint: Codable, Sendable, Equatable {
    public var layerIndex: Int
    public var cacheType: String
    public var offset: Int
    public var metaStateSHA256: String
    public var states: [QwenMTPSampledTraceArrayFingerprint]

    public init(
        layerIndex: Int,
        cacheType: String,
        offset: Int,
        metaStateSHA256: String,
        states: [QwenMTPSampledTraceArrayFingerprint]
    ) {
        self.layerIndex = layerIndex
        self.cacheType = cacheType
        self.offset = offset
        self.metaStateSHA256 = metaStateSHA256
        self.states = states
    }
}

public struct QwenMTPSampledTraceCaseEvidence: Codable, Sendable, Equatable {
    public var branch: QwenMTPSampledTraceBranch
    public var promptSHA256: String
    public var promptTokenCount: Int
    public var bonusToken: Int
    public var proposedToken: Int
    public var emittedToken: Int
    public var proposalUniform: Double
    public var acceptanceUniform: Double
    public var residualUniform: Double?
    public var targetProbabilities: [Double]
    public var draftProbabilities: [Double]
    public var candidateCache: [QwenMTPSampledTraceCacheFingerprint]
    public var scalarCache: [QwenMTPSampledTraceCacheFingerprint]

    public init(
        branch: QwenMTPSampledTraceBranch,
        promptSHA256: String,
        promptTokenCount: Int,
        bonusToken: Int,
        proposedToken: Int,
        emittedToken: Int,
        proposalUniform: Double,
        acceptanceUniform: Double,
        residualUniform: Double?,
        targetProbabilities: [Double],
        draftProbabilities: [Double],
        candidateCache: [QwenMTPSampledTraceCacheFingerprint],
        scalarCache: [QwenMTPSampledTraceCacheFingerprint]
    ) {
        self.branch = branch
        self.promptSHA256 = promptSHA256
        self.promptTokenCount = promptTokenCount
        self.bonusToken = bonusToken
        self.proposedToken = proposedToken
        self.emittedToken = emittedToken
        self.proposalUniform = proposalUniform
        self.acceptanceUniform = acceptanceUniform
        self.residualUniform = residualUniform
        self.targetProbabilities = targetProbabilities
        self.draftProbabilities = draftProbabilities
        self.candidateCache = candidateCache
        self.scalarCache = scalarCache
    }
}

public struct QwenMTPSampledTraceEvidence: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var source: QwenMTPSampledTraceSource
    public var sampling: QwenMTPSampledTraceSampling
    public var cases: [QwenMTPSampledTraceCaseEvidence]

    public init(
        schemaVersion: Int,
        source: QwenMTPSampledTraceSource,
        sampling: QwenMTPSampledTraceSampling,
        cases: [QwenMTPSampledTraceCaseEvidence]
    ) {
        self.schemaVersion = schemaVersion
        self.source = source
        self.sampling = sampling
        self.cases = cases
    }
}

public enum QwenMTPSampledTraceGateError: Error, Sendable, Equatable {
    case schemaMismatch
    case sourceMismatch
    case samplingMismatch
    case branchSetMismatch
    case caseContextMismatch
    case promptMismatch(branch: QwenMTPSampledTraceBranch)
    case invalidPromptTokenCount(branch: QwenMTPSampledTraceBranch)
    case vocabularyMismatch(branch: QwenMTPSampledTraceBranch)
    case proposalDrawMismatch(branch: QwenMTPSampledTraceBranch)
    case drawPlanMismatch(branch: QwenMTPSampledTraceBranch)
    case decisionMismatch(branch: QwenMTPSampledTraceBranch)
    case distributionLawMismatch(index: Int)
    case cacheLayerCountMismatch(branch: QwenMTPSampledTraceBranch)
    case malformedCache(branch: QwenMTPSampledTraceBranch, layer: Int)
    case cacheMismatch(branch: QwenMTPSampledTraceBranch, layer: Int)
    case malformedJSONL
    case wrongSubcommand
    case invalidProvenance
    case nonCanonicalJSONL
}

public enum QwenMTPSampledTraceGate {
    public static let schemaVersion = 2
    public static let subcommand = "qwen-mtp-sampled-trace"
    public static let corpusID = "qwen35-sampled-mtp-depth1-v1"
    public static let requiredVocabularyCount = 248_320
    public static let requiredCacheLayerCount = 32
    public static let requiredPrompt =
        "Complete this sentence in a few words: The fastest reliable test is"
    public static let requiredPromptSHA256 = sha256Hex(Data(requiredPrompt.utf8))
    public static let requiredModelConfigHash = "5a99be4477ebdac8"
    public static let requiredCheckpointManifestHash = "db2b2480a8525194"
    public static let requiredMLXSwiftVersion = "0.31.6"
    public static let requiredMLXSwiftLMRevision =
        "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"

    public static let requiredSource = QwenMTPSampledTraceSource(
        targetModelID: QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.modelID,
        targetRevision: QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.revision,
        targetConfigSHA256:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.configSHA256,
        targetTensorManifestSHA256:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.tensorManifestSHA256,
        drafterModelID: QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.modelID,
        drafterRevision: QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.revision,
        drafterConfigSHA256:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.configSHA256,
        drafterTensorManifestSHA256:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.tensorManifestSHA256,
        sourceRevision: QwenMTPKnownArtifactLocks.qwen35_9BDepth1.sourceRevision,
        runtimeBlockSize: 3,
        maximumAcceptedDraftTokens: 2)

    public static let requiredSampling = QwenMTPSampledTraceSampling(
        temperature: 1,
        topP: 1,
        topK: 0,
        minP: 0,
        repetitionPenalty: nil,
        presencePenalty: nil,
        frequencyPenalty: nil,
        probabilityDType: "float32",
        probabilityNormalization: "softmax",
        rejectionProposalSelection: "max-positive-q-minus-p-cdf-midpoint",
        proposalContext: "after-committed-bonus",
        drawOrder: ["proposal", "acceptance", "residual-on-rejection"])

    @discardableResult
    public static func validateJSONL(
        _ data: Data
    ) throws -> QwenMTPSampledTraceEvidence {
        guard data.last == 0x0a,
            data.split(separator: 0x0a, omittingEmptySubsequences: false).count == 2,
            let record = try? JSONDecoder().decode(
                ResultRecord<QwenMTPSampledTraceEvidence>.self,
                from: Data(data.dropLast()))
        else {
            throw QwenMTPSampledTraceGateError.malformedJSONL
        }
        guard record.subcommand == subcommand else {
            throw QwenMTPSampledTraceGateError.wrongSubcommand
        }
        guard isValidProvenance(record.provenance) else {
            throw QwenMTPSampledTraceGateError.invalidProvenance
        }
        guard record.provenance.modelPath == requiredSource.targetModelID,
            record.provenance.corpusId == corpusID,
            record.provenance.corpusContentHash == requiredPromptSHA256,
            record.provenance.modelConfigHash == requiredModelConfigHash,
            record.provenance.modelCheckpointManifestHash
                == requiredCheckpointManifestHash,
            record.provenance.modelQuant == ModelQuantInfo(bits: 4, groupSize: 64)
        else {
            throw QwenMTPSampledTraceGateError.invalidProvenance
        }
        _ = try validate(record.payload)
        guard Data((try record.jsonLine() + "\n").utf8) == data else {
            throw QwenMTPSampledTraceGateError.nonCanonicalJSONL
        }
        return record.payload
    }

    @discardableResult
    public static func validate(
        _ evidence: QwenMTPSampledTraceEvidence
    ) throws -> QwenMTPSampledTraceEvidence {
        guard evidence.schemaVersion == schemaVersion else { throw QwenMTPSampledTraceGateError.schemaMismatch }
        guard evidence.source == requiredSource else { throw QwenMTPSampledTraceGateError.sourceMismatch }
        guard evidence.sampling == requiredSampling else { throw QwenMTPSampledTraceGateError.samplingMismatch }
        guard evidence.cases.map(\.branch) == [.accepted, .rejected] else {
            throw QwenMTPSampledTraceGateError.branchSetMismatch
        }
        for traceCase in evidence.cases {
            try validate(traceCase)
        }
        guard evidence.cases[0].promptTokenCount == evidence.cases[1].promptTokenCount,
            evidence.cases[0].bonusToken == evidence.cases[1].bonusToken,
            evidence.cases[0].targetProbabilities == evidence.cases[1].targetProbabilities,
            evidence.cases[0].draftProbabilities == evidence.cases[1].draftProbabilities
        else {
            throw QwenMTPSampledTraceGateError.caseContextMismatch
        }
        try validateDistributionLaw(evidence.cases[0])
        return evidence
    }

    private static func validate(
        _ traceCase: QwenMTPSampledTraceCaseEvidence
    ) throws {
        let branch = traceCase.branch
        guard traceCase.promptSHA256 == requiredPromptSHA256 else {
            throw QwenMTPSampledTraceGateError.promptMismatch(branch: branch)
        }
        guard traceCase.promptTokenCount > 0 else {
            throw QwenMTPSampledTraceGateError.invalidPromptTokenCount(branch: branch)
        }
        guard traceCase.targetProbabilities.count == requiredVocabularyCount,
            traceCase.draftProbabilities.count == requiredVocabularyCount,
            traceCase.targetProbabilities.allSatisfy({ $0.isFinite && $0 >= 0 }),
            traceCase.draftProbabilities.allSatisfy({ $0.isFinite && $0 >= 0 })
        else {
            throw QwenMTPSampledTraceGateError.vocabularyMismatch(branch: branch)
        }
        guard traceCase.proposalUniform.isFinite,
            (0 ..< 1).contains(traceCase.proposalUniform),
            traceCase.acceptanceUniform.isFinite,
            (0 ... 1).contains(traceCase.acceptanceUniform),
            traceCase.residualUniform.map({ $0.isFinite && (0 ..< 1).contains($0) }) ?? true
        else {
            throw QwenMTPSampledTraceGateError.drawPlanMismatch(branch: branch)
        }
        try validateDrawPlan(traceCase)
        let proposal = categoricalSample(
            traceCase.draftProbabilities,
            uniform: traceCase.proposalUniform)
        guard proposal == traceCase.proposedToken else {
            throw QwenMTPSampledTraceGateError.proposalDrawMismatch(branch: branch)
        }

        let decision = try SampledMTPResidualCorrection.decide(
            target: traceCase.targetProbabilities,
            draft: traceCase.draftProbabilities,
            proposedToken: traceCase.proposedToken,
            acceptanceUniform: traceCase.acceptanceUniform,
            residualUniform: traceCase.residualUniform)
        switch (branch, decision.decision) {
        case (.accepted, .accepted(let token)) where token == traceCase.emittedToken:
            guard traceCase.residualUniform == nil else {
                throw QwenMTPSampledTraceGateError.decisionMismatch(branch: branch)
            }
        case (.rejected, .rejected(let token)) where token == traceCase.emittedToken:
            guard traceCase.residualUniform != nil else {
                throw QwenMTPSampledTraceGateError.decisionMismatch(branch: branch)
            }
        default:
            throw QwenMTPSampledTraceGateError.decisionMismatch(branch: branch)
        }

        guard traceCase.candidateCache.count == requiredCacheLayerCount,
            traceCase.scalarCache.count == requiredCacheLayerCount
        else {
            throw QwenMTPSampledTraceGateError.cacheLayerCountMismatch(branch: branch)
        }
        for layer in 0 ..< requiredCacheLayerCount {
            let candidate = traceCase.candidateCache[layer]
            let scalar = traceCase.scalarCache[layer]
            try validateCacheFingerprint(
                candidate, branch: branch, layer: layer,
                promptTokenCount: traceCase.promptTokenCount)
            try validateCacheFingerprint(
                scalar, branch: branch, layer: layer,
                promptTokenCount: traceCase.promptTokenCount)
            guard candidate == scalar else {
                throw QwenMTPSampledTraceGateError.cacheMismatch(branch: branch, layer: layer)
            }
        }
    }

    private static func validateDrawPlan(
        _ traceCase: QwenMTPSampledTraceCaseEvidence
    ) throws {
        let branch = traceCase.branch
        switch branch {
        case .accepted:
            guard traceCase.proposalUniform == 0.25,
                traceCase.acceptanceUniform == 0,
                traceCase.residualUniform == nil
            else {
                throw QwenMTPSampledTraceGateError.drawPlanMismatch(branch: branch)
            }
        case .rejected:
            var expectedToken: Int?
            var greatestPositiveDifference = 0.0
            for index in traceCase.draftProbabilities.indices {
                let difference = traceCase.draftProbabilities[index]
                    - traceCase.targetProbabilities[index]
                if difference > greatestPositiveDifference {
                    greatestPositiveDifference = difference
                    expectedToken = index
                }
            }
            guard let expectedToken,
                traceCase.proposedToken == expectedToken
            else {
                throw QwenMTPSampledTraceGateError.drawPlanMismatch(branch: branch)
            }
            let lowerCDF = traceCase.draftProbabilities[..<expectedToken].reduce(0, +)
            let expectedUniform = lowerCDF
                + traceCase.draftProbabilities[expectedToken] / 2
            guard traceCase.proposalUniform == expectedUniform,
                traceCase.acceptanceUniform == Double(1).nextDown,
                traceCase.residualUniform == 0.75
            else {
                throw QwenMTPSampledTraceGateError.drawPlanMismatch(branch: branch)
            }
        }
    }

    private static func validateCacheFingerprint(
        _ fingerprint: QwenMTPSampledTraceCacheFingerprint,
        branch: QwenMTPSampledTraceBranch,
        layer: Int,
        promptTokenCount: Int
    ) throws {
        let finalTokenCount = promptTokenCount + 2
        let expectedStates: [(shape: [Int], dtype: String, byteCount: Int)]
        let expectedType: String
        let expectedOffset: Int
        if (layer + 1).isMultiple(of: 4) {
            expectedType = "KVCacheSimple"
            expectedOffset = finalTokenCount
            let shape = [1, 4, finalTokenCount, 256]
            expectedStates = [
                (shape, "bfloat16", 1 * 4 * finalTokenCount * 256 * 2),
                (shape, "bfloat16", 1 * 4 * finalTokenCount * 256 * 2),
            ]
        } else {
            expectedType = "MambaCache"
            expectedOffset = 0
            expectedStates = [
                ([1, 3, 8192], "bfloat16", 49_152),
                ([1, 32, 128, 128], "float32", 2_097_152),
            ]
        }
        guard fingerprint.layerIndex == layer,
            fingerprint.cacheType == expectedType,
            fingerprint.offset == expectedOffset,
            isLowercaseSHA256(fingerprint.metaStateSHA256),
            fingerprint.states.count == expectedStates.count
        else {
            throw QwenMTPSampledTraceGateError.malformedCache(branch: branch, layer: layer)
        }
        for (index, state) in fingerprint.states.enumerated() {
            let expected = expectedStates[index]
            guard state.stateIndex == index,
                state.shape == expected.shape,
                state.dtype == expected.dtype,
                state.byteCount == expected.byteCount,
                isLowercaseSHA256(state.sha256)
            else {
                throw QwenMTPSampledTraceGateError.malformedCache(branch: branch, layer: layer)
            }
        }
    }

    private static func isValidProvenance(_ provenance: Provenance) -> Bool {
        isLowercaseHex(provenance.harnessGitSHA, count: 40)
            && provenance.mlxSwiftVersion == requiredMLXSwiftVersion
            && provenance.referenceMLXVersion == nil
            && provenance.referenceMLXLMVersion == requiredMLXSwiftLMRevision
            && !provenance.hardwareChip.isEmpty
            && provenance.hardwareChip != "unknown"
            && provenance.hardwareRAMBytes > 0
            && !provenance.hardwareOS.isEmpty
            && ISO8601DateFormatter().date(from: provenance.date) != nil
            && !provenance.nonce.isEmpty
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        isLowercaseHex(value, count: 64)
    }

    private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }

    private static func validateDistributionLaw(
        _ traceCase: QwenMTPSampledTraceCaseEvidence
    ) throws {
        var acceptance = Array(repeating: 0.0, count: requiredVocabularyCount)
        for index in traceCase.draftProbabilities.indices
        where traceCase.draftProbabilities[index] > 0 {
            acceptance[index] = min(
                1,
                traceCase.targetProbabilities[index] / traceCase.draftProbabilities[index])
        }
        let rejectedMass = zip(traceCase.draftProbabilities, acceptance).reduce(0.0) {
            $0 + $1.0 * (1 - $1.1)
        }
        let residual: [Double]
        if rejectedMass > 0 {
            residual = try SampledMTPResidualCorrection.residualDistribution(
                target: traceCase.targetProbabilities,
                draft: traceCase.draftProbabilities)
        } else {
            residual = Array(repeating: 0, count: requiredVocabularyCount)
        }
        for index in 0 ..< requiredVocabularyCount {
            let reconstructed = traceCase.draftProbabilities[index] * acceptance[index]
                + rejectedMass * residual[index]
            guard abs(reconstructed - traceCase.targetProbabilities[index]) <= 1e-9 else {
                throw QwenMTPSampledTraceGateError.distributionLawMismatch(index: index)
            }
        }
    }

    private static func categoricalSample(
        _ distribution: [Double],
        uniform: Double
    ) -> Int {
        var cumulative = 0.0
        for (index, probability) in distribution.enumerated() {
            cumulative += probability
            if uniform < cumulative { return index }
        }
        return distribution.index(before: distribution.endIndex)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
