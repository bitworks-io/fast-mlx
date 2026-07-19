import CryptoKit
import Foundation

/// Stable lowercase SHA-256 used for task artifacts and preserved scored outputs.
public func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// Stable, architecture-independent digest of the exact tokenizer IDs fed to the model. Fixed
/// eight-byte big-endian lanes preserve ordering and boundaries without relying on JSON number
/// formatting, making fp16/candidate prompt equality independently auditable.
public func taskTokenIDsSHA256(_ tokenIDs: [Int]) -> String {
    var data = Data()
    data.reserveCapacity(tokenIDs.count * MemoryLayout<Int64>.size)
    for tokenID in tokenIDs {
        var encoded = Int64(tokenID).bigEndian
        withUnsafeBytes(of: &encoded) { data.append(contentsOf: $0) }
    }
    return sha256Hex(data)
}

public enum TaskCoherenceEvidenceError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidIdentifier(String)
    case invalidDigest(String)
    case invalidPromptLayout
    case invalidRuntimeEvidence(String)
    case invalidArtifact
    case incompleteArtifact
    case duplicateItemID(String)
    case mismatchedArtifact(String)
    case missingReferenceArtifact
    case referenceArtifactMismatch
    case hardFloorFailed
    case invalidPromotionProvenance(String)
    case klEvidenceMismatch(String)
}

/// Token-space proof that the task-bearing material was not left in KVarN's fp16 sink or
/// incomplete tail. The current correctness path has a fixed 128-token sink and tile width.
public struct TaskCoherencePromptLayoutEvidence:
    Codable, Equatable, Sendable
{
    public static let sinkTokens = 128
    public static let tileTokens = 128

    public let promptTokens: Int
    public let materialStartToken: Int
    public let materialEndToken: Int
    public let compressedRegionEndToken: Int
    public let minimumCompletedTileCount: Int

    public init(
        promptTokens: Int,
        materialStartToken: Int,
        materialEndToken: Int,
        compressedRegionEndToken: Int,
        minimumCompletedTileCount: Int
    ) {
        self.promptTokens = promptTokens
        self.materialStartToken = materialStartToken
        self.materialEndToken = materialEndToken
        self.compressedRegionEndToken = compressedRegionEndToken
        self.minimumCompletedTileCount = minimumCompletedTileCount
    }

    /// Derives a conservative material span from four tokenizer passes. Tokenizers may resegment
    /// several tokens at either concatenation boundary, so the material ends immediately before
    /// the suffix/query tokens that remain an exact common suffix of the complete prompt. The
    /// prefix+material common prefix is retained as an independent lower bound, preventing a
    /// repeated suffix token sequence from understating the span.
    public static func derive(
        prefixTokenIDs: [Int],
        prefixAndMaterialTokenIDs: [Int],
        suffixAndQueryTokenIDs: [Int],
        promptTokenIDs: [Int]
    ) throws -> TaskCoherencePromptLayoutEvidence {
        func commonPrefix(_ lhs: [Int], _ rhs: [Int]) -> Int {
            var count = 0
            while count < lhs.count, count < rhs.count,
                lhs[count] == rhs[count]
            {
                count += 1
            }
            return count
        }

        func commonSuffix(_ lhs: [Int], _ rhs: [Int]) -> Int {
            var count = 0
            while count < lhs.count, count < rhs.count,
                lhs[lhs.count - count - 1] == rhs[rhs.count - count - 1]
            {
                count += 1
            }
            return count
        }

        let materialStart = commonPrefix(prefixTokenIDs, promptTokenIDs)
        let prefixMaterialCommon = commonPrefix(
            prefixAndMaterialTokenIDs, promptTokenIDs)
        let stableSuffixTokens = commonSuffix(
            suffixAndQueryTokenIDs, promptTokenIDs)
        let suffixBoundary = promptTokenIDs.count - stableSuffixTokens
        let materialEnd = max(prefixMaterialCommon, suffixBoundary)
        let sink = Self.sinkTokens
        let tile = Self.tileTokens
        guard promptTokenIDs.count > sink, materialEnd >= sink else {
            throw TaskCoherenceEvidenceError.invalidPromptLayout
        }
        let compressedEnd = sink
            + ((promptTokenIDs.count - sink) / tile) * tile
        let minimumTiles = (materialEnd - sink + tile - 1) / tile
        return try TaskCoherencePromptLayoutEvidence(
            promptTokens: promptTokenIDs.count,
            materialStartToken: materialStart,
            materialEndToken: materialEnd,
            compressedRegionEndToken: compressedEnd,
            minimumCompletedTileCount: minimumTiles).validated()
    }

    @discardableResult
    public func validated() throws -> Self {
        let sink = Self.sinkTokens
        let tile = Self.tileTokens
        guard promptTokens > sink,
            materialStartToken >= sink,
            materialEndToken > materialStartToken,
            materialEndToken <= promptTokens
        else { throw TaskCoherenceEvidenceError.invalidPromptLayout }

        let expectedCompressedEnd = sink + ((promptTokens - sink) / tile) * tile
        let materialSpanFromSink = materialEndToken - sink
        let expectedMinimumTiles = (materialSpanFromSink + tile - 1) / tile
        guard compressedRegionEndToken == expectedCompressedEnd,
            materialEndToken <= expectedCompressedEnd,
            expectedMinimumTiles >= 1,
            minimumCompletedTileCount == expectedMinimumTiles
        else { throw TaskCoherenceEvidenceError.invalidPromptLayout }
        return self
    }
}

/// Scalar cache facts captured after each generated task case. These values prove the requested
/// tier performed real cache work; a nominal tier label is never sufficient promotion evidence.
public struct TaskCoherenceCacheEngagementEvidence:
    Codable, Equatable, Sendable
{
    /// Native quantized-cache token counter. It is nil for fp16 because that path does not expose
    /// a quantized engagement marker and must not manufacture one from prompt length.
    public let cachedTokens: Int?
    public let affineTokens: Int?
    public let kvtunerTokens: Int?
    public let kvtunerLayerCount: Int?
    public let kvarnCompletedTileCount: Int?
    public let kvarnCompressedTokens: Int?
    public let kvarnCodecIterations: Int?
    public let kvarnExecutionMode: String?
    /// Native engagement captured from the fresh, full-prompt cache used to score a restricted
    /// choice. Structured-tool rows do not run this second scoring pass and leave these nil.
    public let scoringCachedTokens: Int?
    public let scoringKVTunerLayerCount: Int?
    public let scoringKVarNCompletedTileCount: Int?
    public let scoringKVarNCompressedTokens: Int?

    public init(
        cachedTokens: Int?,
        affineTokens: Int?,
        kvtunerTokens: Int? = nil,
        kvtunerLayerCount: Int? = nil,
        kvarnCompletedTileCount: Int?,
        kvarnCompressedTokens: Int?,
        kvarnCodecIterations: Int?,
        kvarnExecutionMode: String?,
        scoringCachedTokens: Int? = nil,
        scoringKVTunerLayerCount: Int? = nil,
        scoringKVarNCompletedTileCount: Int? = nil,
        scoringKVarNCompressedTokens: Int? = nil
    ) {
        self.cachedTokens = cachedTokens
        self.affineTokens = affineTokens
        self.kvtunerTokens = kvtunerTokens
        self.kvtunerLayerCount = kvtunerLayerCount
        self.kvarnCompletedTileCount = kvarnCompletedTileCount
        self.kvarnCompressedTokens = kvarnCompressedTokens
        self.kvarnCodecIterations = kvarnCodecIterations
        self.kvarnExecutionMode = kvarnExecutionMode
        self.scoringCachedTokens = scoringCachedTokens
        self.scoringKVTunerLayerCount = scoringKVTunerLayerCount
        self.scoringKVarNCompletedTileCount =
            scoringKVarNCompletedTileCount
        self.scoringKVarNCompressedTokens = scoringKVarNCompressedTokens
    }
}

/// Exact tokenizer boundary for one case. Model/config/checkpoint hashes do not cover mutable
/// tokenizer files, so task evidence also binds the actual prompt IDs and the four globally pinned
/// restricted-choice token IDs. Candidate/reference pairing requires this value byte-for-byte.
public struct TaskCoherenceTokenizationEvidence:
    Codable, Equatable, Sendable
{
    public let tokenizerManifestSHA256: String
    public let promptTokenIDsSHA256: String
    public let restrictedChoiceLabelTokenIDs: [String: Int]?

    public init(
        tokenizerManifestSHA256: String,
        promptTokenIDsSHA256: String,
        restrictedChoiceLabelTokenIDs: [String: Int]?
    ) {
        self.tokenizerManifestSHA256 = tokenizerManifestSHA256
        self.promptTokenIDsSHA256 = promptTokenIDsSHA256
        self.restrictedChoiceLabelTokenIDs = restrictedChoiceLabelTokenIDs
    }
}

/// Frozen decoding/scoring controls shared by every case in a qualification run. These fields are
/// evidence, not CLI defaults: candidate/reference comparison is invalid when truncation, sampling,
/// label spelling, or tokenizer special-token handling differs.
public enum TaskCoherencePromptFormat: String, Codable, Sendable {
    /// Historical task qualification behavior: tokenize the frozen prompt text directly.
    case rawV1 = "raw-v1"

    /// Render the frozen prompt as one user message through the checkpoint chat template. The
    /// MLXLM tokenizer bridge adds the assistant generation prompt, while the caller supplies
    /// `enable_thinking=false` so structured output starts at the answer rather than a reasoning
    /// preamble. The long name is intentional evidence: it freezes all three controls.
    case checkpointChatTemplateGenerationPromptThinkingDisabledV1 =
        "checkpoint-chat-template-generation-prompt-thinking-disabled-v1"
}

public struct TaskCoherenceRunConfiguration:
    Codable, Equatable, Sendable
{
    public let temperature: Double
    public let restrictedChoiceMaxTokens: Int
    public let structuredToolMaxTokens: Int
    public let restrictedChoiceLabelTokenSpellings: [String: String]
    public let restrictedChoiceAddsSpecialTokens: Bool
    public let structuredToolSkipsSpecialTokens: Bool
    public let restrictedChoicePromptFormat: TaskCoherencePromptFormat
    public let structuredToolPromptFormat: TaskCoherencePromptFormat

    public init(
        temperature: Double,
        restrictedChoiceMaxTokens: Int,
        structuredToolMaxTokens: Int,
        restrictedChoiceLabelTokenSpellings: [String: String],
        restrictedChoiceAddsSpecialTokens: Bool,
        structuredToolSkipsSpecialTokens: Bool,
        restrictedChoicePromptFormat: TaskCoherencePromptFormat = .rawV1,
        structuredToolPromptFormat: TaskCoherencePromptFormat = .rawV1
    ) {
        self.temperature = temperature
        self.restrictedChoiceMaxTokens = restrictedChoiceMaxTokens
        self.structuredToolMaxTokens = structuredToolMaxTokens
        self.restrictedChoiceLabelTokenSpellings =
            restrictedChoiceLabelTokenSpellings
        self.restrictedChoiceAddsSpecialTokens =
            restrictedChoiceAddsSpecialTokens
        self.structuredToolSkipsSpecialTokens = structuredToolSkipsSpecialTokens
        self.restrictedChoicePromptFormat = restrictedChoicePromptFormat
        self.structuredToolPromptFormat = structuredToolPromptFormat
    }

    public static func qualificationV2(
        structuredToolMaxTokens: Int
    ) -> TaskCoherenceRunConfiguration {
        TaskCoherenceRunConfiguration(
            temperature: 0,
            restrictedChoiceMaxTokens: 1,
            structuredToolMaxTokens: structuredToolMaxTokens,
            restrictedChoiceLabelTokenSpellings:
                TaskRestrictedChoiceScorer.labelTokenSpellings,
            restrictedChoiceAddsSpecialTokens: false,
            structuredToolSkipsSpecialTokens: true,
            restrictedChoicePromptFormat: .rawV1,
            structuredToolPromptFormat: .rawV1)
    }

    public static func qualificationV3(
        structuredToolMaxTokens: Int
    ) -> TaskCoherenceRunConfiguration {
        TaskCoherenceRunConfiguration(
            temperature: 0,
            restrictedChoiceMaxTokens: 1,
            structuredToolMaxTokens: structuredToolMaxTokens,
            restrictedChoiceLabelTokenSpellings:
                TaskRestrictedChoiceScorer.labelTokenSpellings,
            restrictedChoiceAddsSpecialTokens: false,
            structuredToolSkipsSpecialTokens: true,
            restrictedChoicePromptFormat: .rawV1,
            structuredToolPromptFormat:
                .checkpointChatTemplateGenerationPromptThinkingDisabledV1)
    }

    @discardableResult
    public func validated() throws -> Self {
        guard temperature.isFinite, temperature == 0,
            restrictedChoiceMaxTokens == 1,
            (1 ... 512).contains(structuredToolMaxTokens),
            restrictedChoiceLabelTokenSpellings
                == TaskRestrictedChoiceScorer.labelTokenSpellings,
            restrictedChoiceAddsSpecialTokens == false,
            structuredToolSkipsSpecialTokens,
            restrictedChoicePromptFormat == .rawV1,
            structuredToolPromptFormat == .rawV1
                || structuredToolPromptFormat
                    == .checkpointChatTemplateGenerationPromptThinkingDisabledV1
        else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact(
                "run-configuration")
        }
        return self
    }

    private enum CodingKeys: String, CodingKey {
        case temperature
        case restrictedChoiceMaxTokens
        case structuredToolMaxTokens
        case restrictedChoiceLabelTokenSpellings
        case restrictedChoiceAddsSpecialTokens
        case structuredToolSkipsSpecialTokens
        case restrictedChoicePromptFormat
        case structuredToolPromptFormat
    }

    /// Schema-2 V1/V2 evidence predates explicit prompt-format fields. Missing fields decode to
    /// the historical raw path, preserving durable artifact readability without allowing those
    /// rows to compare equal to the new chat-templated qualification configuration.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            temperature: try container.decode(
                Double.self, forKey: .temperature),
            restrictedChoiceMaxTokens: try container.decode(
                Int.self, forKey: .restrictedChoiceMaxTokens),
            structuredToolMaxTokens: try container.decode(
                Int.self, forKey: .structuredToolMaxTokens),
            restrictedChoiceLabelTokenSpellings: try container.decode(
                [String: String].self,
                forKey: .restrictedChoiceLabelTokenSpellings),
            restrictedChoiceAddsSpecialTokens: try container.decode(
                Bool.self, forKey: .restrictedChoiceAddsSpecialTokens),
            structuredToolSkipsSpecialTokens: try container.decode(
                Bool.self, forKey: .structuredToolSkipsSpecialTokens),
            restrictedChoicePromptFormat: try container.decodeIfPresent(
                TaskCoherencePromptFormat.self,
                forKey: .restrictedChoicePromptFormat) ?? .rawV1,
            structuredToolPromptFormat: try container.decodeIfPresent(
                TaskCoherencePromptFormat.self,
                forKey: .structuredToolPromptFormat) ?? .rawV1)
    }
}

/// One append-only task row. A run writes exactly one row per frozen case, allowing interrupted
/// runs to be inspected without treating a partial artifact as a complete assessment.
public struct TaskCoherenceCasePayload: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let matrixID: String
    public let cellID: String
    public let identity: TaskCoherenceRunIdentity
    /// Nil only for the fp16 task baseline. Every lossy row names the exact raw fp16 JSONL digest.
    public let referenceArtifactSHA256: String?
    public let promptContentHash: String
    public let runConfiguration: TaskCoherenceRunConfiguration
    public let tokenization: TaskCoherenceTokenizationEvidence
    public let layout: TaskCoherencePromptLayoutEvidence
    public let generatedTokenCount: Int
    /// The exact scorer input: one of A/B/C/D or the decoded structured-tool generation.
    public let scoredOutput: String
    public let outputSHA256: String
    public let score: TaskItemScore
    public let engagement: TaskCoherenceCacheEngagementEvidence
    /// Explicit attention request plus the operation observed after this exact case. Optional
    /// keeps historical task artifacts readable; Phase-2 qualification requires it separately.
    public let compressedKVAttention:
        CompressedKVAttentionRuntimeBinding?

    public init(
        schemaVersion: Int,
        matrixID: String,
        cellID: String,
        identity: TaskCoherenceRunIdentity,
        referenceArtifactSHA256: String?,
        promptContentHash: String,
        runConfiguration: TaskCoherenceRunConfiguration,
        tokenization: TaskCoherenceTokenizationEvidence,
        layout: TaskCoherencePromptLayoutEvidence,
        generatedTokenCount: Int,
        scoredOutput: String,
        outputSHA256: String,
        score: TaskItemScore,
        engagement: TaskCoherenceCacheEngagementEvidence,
        compressedKVAttention:
            CompressedKVAttentionRuntimeBinding? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.cellID = cellID
        self.identity = identity
        self.referenceArtifactSHA256 = referenceArtifactSHA256
        self.promptContentHash = promptContentHash
        self.runConfiguration = runConfiguration
        self.tokenization = tokenization
        self.layout = layout
        self.generatedTokenCount = generatedTokenCount
        self.scoredOutput = scoredOutput
        self.outputSHA256 = outputSHA256
        self.score = score
        self.engagement = engagement
        self.compressedKVAttention = compressedKVAttention
    }
}

/// Canonical reduction of a raw 80-row task artifact. `scores` are ordered exactly like the
/// frozen corpus even if the append-only input arrived in a different order.
public struct TaskCoherenceArtifactSummary: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let matrixID: String
    public let cellID: String
    public let artifactSHA256: String
    public let referenceArtifactSHA256: String?
    public let identity: TaskCoherenceRunIdentity
    public let runConfiguration: TaskCoherenceRunConfiguration
    public let provenance: Provenance
    public let caseCount: Int
    /// Canonical corpus-order cases keep layout, output, and engagement inside the validation
    /// boundary after Codable round trips; a detached all-pass score table cannot promote.
    public let cases: [TaskCoherenceCasePayload]
    public let scores: [TaskItemScore]

    init(
        schemaVersion: Int,
        matrixID: String,
        cellID: String,
        artifactSHA256: String,
        referenceArtifactSHA256: String?,
        identity: TaskCoherenceRunIdentity,
        runConfiguration: TaskCoherenceRunConfiguration,
        provenance: Provenance,
        caseCount: Int,
        cases: [TaskCoherenceCasePayload],
        scores: [TaskItemScore]
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.cellID = cellID
        self.artifactSHA256 = artifactSHA256
        self.referenceArtifactSHA256 = referenceArtifactSHA256
        self.identity = identity
        self.runConfiguration = runConfiguration
        self.provenance = provenance
        self.caseCount = caseCount
        self.cases = cases
        self.scores = scores
    }
}

public enum TaskCoherenceArtifact {
    /// Schema 1 predated authenticated tokenizer/layout/run controls and never shipped from a
    /// runnable task CLI. Schema 2 is the first qualification-capable artifact and rejects any
    /// partial legacy rows rather than guessing missing evidence.
    public static let schemaVersion = 2
    /// Schema 3 carries exact checkpoint-content identity. Affine and KVTuner rows additionally
    /// require an authenticated request/observed-operation binding; an fp16 reference carries
    /// the exact content identity without claiming a compressed-attention operation. Schema 2
    /// remains readable for the frozen pre-compressed-attention corpus.
    public static let compressedAttentionSchemaVersion = 3

    private static let affineTiers: Set<String> = [
        "affine-k4v2-g64",
        "affine-k4v2-g128",
        "affine-k8v2-g64",
        "affine-k8v2-g128",
        "affine-k4v4-g128",
    ]
    private static let kvarnIterations: [String: Int] = [
        "kvarn-k4v2-g128": 8,
        "kvarn-k4v2-g128-i16": 16,
    ]
    private static let tierCellIDs: [String: String] = [
        "fp16": "fp16",
        "affine-k4v2-g64": "affine-k4v2-g64",
        "affine-k4v2-g128": "affine-k4v2-g128",
        "affine-k8v2-g64": "affine-k8v2-g64",
        "affine-k8v2-g128": "affine-k8v2-g128",
        "affine-k4v4-g128": "affine-k4v4-g128",
        "kvarn-k4v2-g128": "kvarn-k4v2-g128-i8",
        "kvarn-k4v2-g128-i16": "kvarn-k4v2-g128-i16",
    ]

    /// Closed runnable mapping shared by preflight CLI validation and artifact adjudication.
    /// Uniform tiers use the fixed table; canonical KVTuner cells use their authenticated
    /// schedule-bound cell spelling as both tier and cell identity.
    public static func expectedCellID(forTier tier: String) -> String? {
        if let cellID = tierCellIDs[tier] { return cellID }
        return isCanonicalKVTunerCellID(tier) ? tier : nil
    }

    /// Historical candidates without an exact checkpoint identity may still consume historical
    /// references. Once a candidate declares exact checkpoint content, however, a contentless or
    /// differently sourced fp16 reference must fail closed even when its cheap manifest matches.
    public static func referenceModelIdentityMatches(
        reference: TaskCoherenceRunIdentity,
        candidate: KVModelEvidenceIdentity
    ) -> Bool {
        guard reference.modelConfigHash == candidate.configHash,
            reference.modelCheckpointManifestHash
                == candidate.checkpointManifestHash
        else { return false }
        guard let checkpointContentSHA256 =
            candidate.checkpointContentSHA256
        else { return true }
        return reference.modelCheckpointContentSHA256
            == checkpointContentSHA256
    }

    /// Authenticate a decoded task receipt against the exact run identity and tokenizer. The
    /// caller derives the observed operation from post-forward counters; this validator prevents
    /// a forged Codable payload from turning the request into its own proof.
    static func validateCompressedKVAttention(
        _ binding: CompressedKVAttentionRuntimeBinding?,
        identity: TaskCoherenceRunIdentity,
        tokenization: TaskCoherenceTokenizationEvidence
    ) throws {
        guard let binding else { return }
        do {
            try binding.validated()
            guard binding.admission.modelConfigHash
                    == identity.modelConfigHash,
                binding.admission.checkpointManifestHash
                    == identity.modelCheckpointManifestHash,
                binding.admission.checkpointContentSHA256
                    == identity.modelCheckpointContentSHA256,
                binding.admission.tokenizerSHA256
                    == tokenization.tokenizerManifestSHA256
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "compressed-attention")
            }
            if affineTiers.contains(identity.kvQuantTier) {
                let groupSize: Int
                if identity.kvQuantTier.hasSuffix("-g64") {
                    groupSize = 64
                } else if identity.kvQuantTier.hasSuffix("-g128") {
                    groupSize = 128
                } else {
                    throw TaskCoherenceEvidenceError
                        .invalidRuntimeEvidence("compressed-attention")
                }
                try binding.admission.validateAffineGeometry(
                    keyGroupSize: groupSize,
                    valueGroupSize: groupSize)
            } else if isCanonicalKVTunerCellID(
                identity.kvQuantTier)
            {
                guard let schedule = identity.kvtunerSchedule else {
                    throw TaskCoherenceEvidenceError
                        .invalidRuntimeEvidence("compressed-attention")
                }
                try binding.admission.validateScheduleIdentity(
                    modelConfigHash: schedule.modelConfigHash,
                    modelConfigSHA256: schedule.modelConfigSHA256,
                    checkpointManifestHash:
                        schedule.checkpointManifestHash,
                    checkpointContentSHA256:
                        schedule.checkpointContentSHA256,
                    tokenizerSHA256: schedule.tokenizerSHA256,
                    layerCount: schedule.layers.count,
                    groupSize: schedule.groupSize)
            } else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "compressed-attention")
            }
        } catch let error as TaskCoherenceEvidenceError {
            throw error
        } catch {
            throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                "compressed-attention")
        }
    }

    /// Strict JSON Lines decoding: exactly one optional trailing newline is accepted. Blank rows,
    /// partial writes, and malformed envelopes fail the whole artifact.
    public static func decodeJSONL(
        _ data: Data
    ) throws -> [ResultRecord<TaskCoherenceCasePayload>] {
        guard let contents = String(data: data, encoding: .utf8),
            !contents.isEmpty
        else { throw TaskCoherenceEvidenceError.invalidArtifact }
        var lines = contents.split(
            separator: "\n", omittingEmptySubsequences: false)
        if lines.last?.isEmpty == true {
            lines.removeLast()
        }
        guard !lines.isEmpty,
            lines.allSatisfy({
                !String($0).trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty
            })
        else { throw TaskCoherenceEvidenceError.invalidArtifact }

        do {
            let decoder = JSONDecoder()
            return try lines.map {
                try decoder.decode(
                    ResultRecord<TaskCoherenceCasePayload>.self,
                    from: Data($0.utf8))
            }
        } catch {
            throw TaskCoherenceEvidenceError.invalidArtifact
        }
    }

    /// Authenticates the exact raw JSONL bytes. Required writers already emit this canonical
    /// sorted-key form; accepting alternate whitespace/order would make the stored digest
    /// impossible to reproduce from a decoded summary.
    public static func summarize(
        _ data: Data,
        corpus: TaskCoherenceCorpus
    ) throws -> TaskCoherenceArtifactSummary {
        let rows = try decodeJSONL(data)
        let canonical = try canonicalData(rows)
        guard data == canonical,
            rows.map({ $0.payload.score.itemID }) == corpus.items.map(\.id)
        else { throw TaskCoherenceEvidenceError.invalidArtifact }
        return try summarizeRows(
            rows, corpus: corpus, artifactSHA256: sha256Hex(data))
    }

    private static func summarizeRows(
        _ rows: [ResultRecord<TaskCoherenceCasePayload>],
        corpus: TaskCoherenceCorpus,
        artifactSHA256: String
    ) throws -> TaskCoherenceArtifactSummary {
        guard rows.count == corpus.items.count, let first = rows.first else {
            throw TaskCoherenceEvidenceError.incompleteArtifact
        }

        try validateProvenance(first.provenance, corpus: corpus)
        try validateCase(first, corpus: corpus)
        var payloadByID: [String: TaskCoherenceCasePayload] = [:]
        for row in rows {
            try validateCase(row, corpus: corpus)
            guard row.subcommand == first.subcommand,
                row.provenance == first.provenance,
                row.payload.schemaVersion == first.payload.schemaVersion,
                row.payload.matrixID == first.payload.matrixID,
                row.payload.cellID == first.payload.cellID,
                row.payload.identity == first.payload.identity,
                row.payload.runConfiguration
                    == first.payload.runConfiguration,
                row.payload.referenceArtifactSHA256
                    == first.payload.referenceArtifactSHA256,
                row.payload.compressedKVAttention
                    == first.payload.compressedKVAttention
            else {
                throw TaskCoherenceEvidenceError.mismatchedArtifact("run")
            }
            guard payloadByID[row.payload.score.itemID] == nil else {
                throw TaskCoherenceEvidenceError.duplicateItemID(
                    row.payload.score.itemID)
            }
            payloadByID[row.payload.score.itemID] = row.payload
        }

        guard Set(payloadByID.keys) == Set(corpus.items.map(\.id)) else {
            throw TaskCoherenceEvidenceError.incompleteArtifact
        }
        let cases = try corpus.items.map { item -> TaskCoherenceCasePayload in
            guard let payload = payloadByID[item.id] else {
                throw TaskCoherenceEvidenceError.incompleteArtifact
            }
            return payload
        }
        let restrictedTokenizations = zip(corpus.items, cases).compactMap {
            item, payload in
            item.scoringMode == .restrictedChoice
                ? payload.tokenization.restrictedChoiceLabelTokenIDs : nil
        }
        guard Set(cases.map {
            $0.tokenization.tokenizerManifestSHA256
        }).count == 1,
            let firstTokenization = restrictedTokenizations.first,
            restrictedTokenizations.allSatisfy({ $0 == firstTokenization })
        else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact(
                "label-tokenization")
        }
        let scores = cases.map(\.score)
        return TaskCoherenceArtifactSummary(
            schemaVersion: first.payload.schemaVersion,
            matrixID: first.payload.matrixID,
            cellID: first.payload.cellID,
            artifactSHA256: artifactSHA256,
            referenceArtifactSHA256:
                first.payload.referenceArtifactSHA256,
            identity: first.payload.identity,
            runConfiguration: first.payload.runConfiguration,
            provenance: first.provenance,
            caseCount: rows.count,
            cases: cases,
            scores: scores)
    }

    private static func canonicalData(
        _ rows: [ResultRecord<TaskCoherenceCasePayload>]
    ) throws -> Data {
        Data((try rows.map { try $0.jsonLine() }
            .joined(separator: "\n") + "\n").utf8)
    }

    private static func validateCase(
        _ record: ResultRecord<TaskCoherenceCasePayload>,
        corpus: TaskCoherenceCorpus
    ) throws {
        let payload = record.payload
        guard record.subcommand == "task-coherence" else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact("subcommand")
        }
        guard [schemaVersion, compressedAttentionSchemaVersion].contains(
            payload.schemaVersion)
        else {
            throw TaskCoherenceEvidenceError.unsupportedSchema(
                payload.schemaVersion)
        }
        guard isIdentifier(payload.matrixID), isIdentifier(payload.cellID),
            isIdentifier(payload.identity.corpusID),
            isIdentifier(payload.identity.corpusContentHash),
            isIdentifier(payload.identity.modelConfigHash),
            isIdentifier(payload.identity.modelCheckpointManifestHash),
            isIdentifier(payload.identity.kvQuantTier),
            expectedCellID(forTier: payload.identity.kvQuantTier)
                == payload.cellID,
            payload.generatedTokenCount > 0,
            isHex(payload.outputSHA256, length: 64),
            payload.outputSHA256
                == sha256Hex(Data(payload.scoredOutput.utf8))
        else {
            throw TaskCoherenceEvidenceError.invalidIdentifier(
                payload.score.itemID)
        }
        guard payload.identity.corpusID == corpus.id,
            payload.identity.corpusContentHash == corpus.contentHash,
            payload.identity.modelConfigHash
                == record.provenance.modelConfigHash,
            payload.identity.modelCheckpointManifestHash
                == record.provenance.modelCheckpointManifestHash,
            record.provenance.corpusId == corpus.id,
            record.provenance.corpusContentHash == corpus.contentHash
        else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact("identity")
        }
        guard let item = corpus.items.first(where: {
            $0.id == payload.score.itemID
        }), item.domain == payload.score.domain,
            payload.promptContentHash == fnv1a64(item.prompt.utf8)
        else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact("prompt")
        }
        try payload.runConfiguration.validated()
        let isKVTuner = isCanonicalKVTunerCellID(
            payload.identity.kvQuantTier)
        if isKVTuner {
            guard let schedule = payload.identity.kvtunerSchedule,
                let layerCount = payload.engagement.kvtunerLayerCount,
                payload.schemaVersion == compressedAttentionSchemaVersion,
                let compressedBinding = payload.compressedKVAttention,
                let checkpointContentSHA256 =
                    payload.identity.modelCheckpointContentSHA256,
                schedule.tokenizerSHA256
                    == payload.tokenization.tokenizerManifestSHA256
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "kvtuner-schedule")
            }
            do {
                try schedule.validated(
                    expectedMatrixID: payload.matrixID,
                    expectedCellID: payload.cellID,
                    expectedModelConfigHash:
                        payload.identity.modelConfigHash,
                    expectedModelConfigSHA256:
                        compressedBinding.admission.modelConfigSHA256,
                    expectedCheckpointManifestHash:
                        payload.identity.modelCheckpointManifestHash,
                    expectedCheckpointContentSHA256:
                        checkpointContentSHA256,
                    expectedLayerCount: layerCount,
                    requiredEvaluationCorpus:
                        try corpus.kvtunerEvaluationCorpusIdentity)
            } catch {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "kvtuner-schedule")
            }
        } else if payload.identity.kvtunerSchedule != nil {
            throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                "kvtuner-schedule")
        }
        guard isHex(
            payload.tokenization.tokenizerManifestSHA256, length: 64),
            isHex(payload.tokenization.promptTokenIDsSHA256, length: 64)
        else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact(
                "prompt-tokenization")
        }
        let requiresCompressedBinding = affineTiers.contains(
            payload.identity.kvQuantTier)
            || isCanonicalKVTunerCellID(payload.identity.kvQuantTier)
        switch payload.schemaVersion {
        case schemaVersion:
            guard payload.compressedKVAttention == nil,
                payload.identity.modelCheckpointContentSHA256 == nil
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "compressed-attention-schema")
            }
        case compressedAttentionSchemaVersion:
            let requiresCheckpointContent = requiresCompressedBinding
                || payload.identity.kvQuantTier == "fp16"
            guard requiresCompressedBinding
                == (payload.compressedKVAttention != nil),
                requiresCheckpointContent
                    == (payload.identity.modelCheckpointContentSHA256 != nil),
                (payload.identity.modelCheckpointContentSHA256.map {
                    isHex($0, length: 64)
                } ?? true)
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "compressed-attention-schema")
            }
        default:
            throw TaskCoherenceEvidenceError.unsupportedSchema(
                payload.schemaVersion)
        }
        try validateCompressedKVAttention(
            payload.compressedKVAttention,
            identity: payload.identity,
            tokenization: payload.tokenization)
        let derivedScore: TaskItemScore
        switch item.scoringMode {
        case .restrictedChoice:
            guard let labelTokenIDs =
                payload.tokenization.restrictedChoiceLabelTokenIDs,
                Set(labelTokenIDs.keys) == Set(["A", "B", "C", "D"]),
                Set(labelTokenIDs.values).count == 4,
                labelTokenIDs.values.allSatisfy({ $0 >= 0 }),
                payload.generatedTokenCount
                    == payload.runConfiguration.restrictedChoiceMaxTokens
            else {
                throw TaskCoherenceEvidenceError.mismatchedArtifact(
                    "label-tokenization")
            }
            guard ["A", "B", "C", "D"].contains(payload.scoredOutput),
                let expected = item.expectedChoice
            else {
                throw TaskCoherenceEvidenceError.mismatchedArtifact("score")
            }
            derivedScore = TaskItemScore(
                itemID: item.id, domain: item.domain,
                correct: payload.scoredOutput == expected,
                syntacticallyValid: nil)
        case .structuredTool:
            guard payload.tokenization.restrictedChoiceLabelTokenIDs == nil,
                payload.generatedTokenCount
                    <= payload.runConfiguration.structuredToolMaxTokens
            else {
                throw TaskCoherenceEvidenceError.mismatchedArtifact(
                    "label-tokenization")
            }
            guard let expected = item.expectedTool else {
                throw TaskCoherenceEvidenceError.mismatchedArtifact("score")
            }
            let structured = TaskStructuredToolScorer.score(
                payload.scoredOutput, expected: expected)
            derivedScore = TaskItemScore(
                itemID: item.id, domain: item.domain,
                correct: structured.correct,
                syntacticallyValid: structured.syntacticallyValid)
        }
        guard payload.score == derivedScore else {
            throw TaskCoherenceEvidenceError.mismatchedArtifact("score")
        }
        try payload.layout.validated()
        try validateEngagement(
            payload.engagement, tier: payload.identity.kvQuantTier,
            kvtunerSchedule: payload.identity.kvtunerSchedule,
            layout: payload.layout,
            generatedTokenCount: payload.generatedTokenCount,
            scoringMode: item.scoringMode)

        if payload.identity.kvQuantTier == "fp16" {
            guard payload.referenceArtifactSHA256 == nil else {
                throw TaskCoherenceEvidenceError.referenceArtifactMismatch
            }
        } else {
            guard let digest = payload.referenceArtifactSHA256 else {
                throw TaskCoherenceEvidenceError.missingReferenceArtifact
            }
            guard isHex(digest, length: 64) else {
                throw TaskCoherenceEvidenceError.invalidDigest(digest)
            }
        }
    }

    private static func validateEngagement(
        _ engagement: TaskCoherenceCacheEngagementEvidence,
        tier: String,
        kvtunerSchedule: KVTunerScheduleBinding?,
        layout: TaskCoherencePromptLayoutEvidence,
        generatedTokenCount: Int,
        scoringMode: TaskCoherenceScoringMode
    ) throws {
        // The submit-first production decoder consumes every emitted token into KV before making
        // it visible to the caller. Its exact post-run cache offset is therefore prompt + N.
        let (expectedCachedTokens, cachedOverflow) = layout.promptTokens
            .addingReportingOverflow(generatedTokenCount)
        guard generatedTokenCount > 0, !cachedOverflow
        else {
            throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                "cachedTokens")
        }
        let kvarnValues = [
            engagement.kvarnCompletedTileCount,
            engagement.kvarnCompressedTokens,
            engagement.kvarnCodecIterations,
        ]
        let scoringKVarNValues = [
            engagement.scoringKVarNCompletedTileCount,
            engagement.scoringKVarNCompressedTokens,
        ]
        let kvtunerValues = [
            engagement.kvtunerTokens,
            engagement.kvtunerLayerCount,
        ]
        if tier == "fp16" {
            guard engagement.cachedTokens == nil,
                engagement.affineTokens == nil,
                kvtunerValues.allSatisfy({ $0 == nil }),
                kvarnValues.allSatisfy({ $0 == nil }),
                engagement.kvarnExecutionMode == nil,
                engagement.scoringCachedTokens == nil,
                engagement.scoringKVTunerLayerCount == nil,
                scoringKVarNValues.allSatisfy({ $0 == nil })
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence("fp16")
            }
            return
        }
        if affineTiers.contains(tier) {
            let validScoringEngagement: Bool
            switch scoringMode {
            case .restrictedChoice:
                validScoringEngagement =
                    engagement.scoringCachedTokens == layout.promptTokens
                    && engagement.scoringKVTunerLayerCount == nil
                    && scoringKVarNValues.allSatisfy({ $0 == nil })
            case .structuredTool:
                validScoringEngagement =
                    engagement.scoringCachedTokens == nil
                    && engagement.scoringKVTunerLayerCount == nil
                    && scoringKVarNValues.allSatisfy({ $0 == nil })
            }
            guard let cachedTokens = engagement.cachedTokens,
                let affineTokens = engagement.affineTokens,
                cachedTokens == expectedCachedTokens,
                cachedTokens == affineTokens,
                kvtunerValues.allSatisfy({ $0 == nil }),
                kvarnValues.allSatisfy({ $0 == nil }),
                engagement.kvarnExecutionMode == nil,
                validScoringEngagement
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "affine")
            }
            return
        }
        if isCanonicalKVTunerCellID(tier) {
            guard let schedule = kvtunerSchedule else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "kvtuner")
            }
            let validScoringEngagement: Bool
            switch scoringMode {
            case .restrictedChoice:
                validScoringEngagement =
                    engagement.scoringCachedTokens == layout.promptTokens
                    && engagement.scoringKVTunerLayerCount
                        == schedule.layers.count
                    && scoringKVarNValues.allSatisfy({ $0 == nil })
            case .structuredTool:
                validScoringEngagement =
                    engagement.scoringCachedTokens == nil
                    && engagement.scoringKVTunerLayerCount == nil
                    && scoringKVarNValues.allSatisfy({ $0 == nil })
            }
            guard let cachedTokens = engagement.cachedTokens,
                let kvtunerTokens = engagement.kvtunerTokens,
                let layerCount = engagement.kvtunerLayerCount,
                cachedTokens == expectedCachedTokens,
                kvtunerTokens == cachedTokens,
                layerCount == schedule.layers.count,
                layerCount > 0,
                engagement.affineTokens == nil,
                kvarnValues.allSatisfy({ $0 == nil }),
                engagement.kvarnExecutionMode == nil,
                validScoringEngagement
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "kvtuner")
            }
            return
        }
        if let expectedIterations = kvarnIterations[tier] {
            let validScoringEngagement: Bool
            switch scoringMode {
            case .restrictedChoice:
                if let scoringCachedTokens = engagement.scoringCachedTokens,
                    let scoringCompleted =
                        engagement.scoringKVarNCompletedTileCount,
                    let scoringCompressed =
                        engagement.scoringKVarNCompressedTokens
                {
                    validScoringEngagement =
                        scoringCachedTokens == layout.promptTokens
                        && engagement.scoringKVTunerLayerCount == nil
                        && checkedKVarNGeometry(
                            cachedTokens: scoringCachedTokens,
                            completedTiles: scoringCompleted,
                            compressedTokens: scoringCompressed)
                        && scoringCompleted >= layout.minimumCompletedTileCount
                } else {
                    validScoringEngagement = false
                }
            case .structuredTool:
                validScoringEngagement =
                    engagement.scoringCachedTokens == nil
                    && engagement.scoringKVTunerLayerCount == nil
                    && scoringKVarNValues.allSatisfy({ $0 == nil })
            }
            guard let cachedTokens = engagement.cachedTokens,
                engagement.affineTokens == nil,
                kvtunerValues.allSatisfy({ $0 == nil }),
                let completed = engagement.kvarnCompletedTileCount,
                let compressed = engagement.kvarnCompressedTokens,
                let iterations = engagement.kvarnCodecIterations,
                cachedTokens == expectedCachedTokens,
                cachedTokens >= TaskCoherencePromptLayoutEvidence.sinkTokens,
                completed >= 0,
                compressed >= 0,
                checkedKVarNGeometry(
                    cachedTokens: cachedTokens,
                    completedTiles: completed,
                    compressedTokens: compressed),
                completed >= layout.minimumCompletedTileCount,
                iterations == expectedIterations,
                engagement.kvarnExecutionMode == "uncompiled-correctness",
                validScoringEngagement
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "kvarn")
            }
            return
        }
        throw TaskCoherenceEvidenceError.invalidRuntimeEvidence("tier")
    }

    private static func checkedKVarNGeometry(
        cachedTokens: Int,
        completedTiles: Int,
        compressedTokens: Int
    ) -> Bool {
        let sink = TaskCoherencePromptLayoutEvidence.sinkTokens
        let tile = TaskCoherencePromptLayoutEvidence.tileTokens
        guard cachedTokens >= sink, completedTiles >= 0,
            compressedTokens >= 0,
            completedTiles == (cachedTokens - sink) / tile
        else { return false }
        let (expectedCompressed, multiplyOverflow) = completedTiles
            .multipliedReportingOverflow(by: tile)
        let (minimumCached, addOverflow) = sink
            .addingReportingOverflow(expectedCompressed)
        return !multiplyOverflow && !addOverflow
            && compressedTokens == expectedCompressed
            && cachedTokens >= minimumCached
    }

    fileprivate static func validateSummary(
        _ summary: TaskCoherenceArtifactSummary,
        corpus: TaskCoherenceCorpus
    ) throws {
        guard [schemaVersion, compressedAttentionSchemaVersion].contains(
            summary.schemaVersion)
        else {
            throw TaskCoherenceEvidenceError.unsupportedSchema(
                summary.schemaVersion)
        }
        guard summary.caseCount == corpus.items.count,
            summary.cases.count == summary.caseCount,
            summary.scores == summary.cases.map(\.score)
        else { throw TaskCoherenceEvidenceError.invalidArtifact }
        let rows = summary.cases.map {
            ResultRecord(
                subcommand: "task-coherence",
                provenance: summary.provenance,
                payload: $0)
        }
        let canonical = try canonicalData(rows)
        guard sha256Hex(canonical) == summary.artifactSHA256 else {
            throw TaskCoherenceEvidenceError.invalidDigest(
                summary.artifactSHA256)
        }
        let rebuilt = try summarize(canonical, corpus: corpus)
        guard rebuilt == summary else {
            throw TaskCoherenceEvidenceError.invalidArtifact
        }
    }

    fileprivate static func matchingRuntime(
        _ lhs: Provenance, _ rhs: Provenance
    ) -> Bool {
        lhs.hardwareChip == rhs.hardwareChip
            && lhs.hardwareRAMBytes == rhs.hardwareRAMBytes
            && lhs.hardwareOS == rhs.hardwareOS
            && lhs.harnessGitSHA == rhs.harnessGitSHA
            && lhs.mlxSwiftVersion == rhs.mlxSwiftVersion
            && lhs.modelPath == rhs.modelPath
            && lhs.modelConfigHash == rhs.modelConfigHash
            && lhs.modelCheckpointManifestHash
                == rhs.modelCheckpointManifestHash
            && lhs.modelQuant == rhs.modelQuant
    }

    fileprivate static func expectedFormatKind(
        for tier: String
    ) -> KVStorageFormatKind? {
        if affineTiers.contains(tier) { return .affine }
        if isCanonicalKVTunerCellID(tier) { return .kvtuner }
        if kvarnIterations[tier] != nil { return .kvarn }
        return tier == "fp16" ? .fp16 : nil
    }

    private static func isCanonicalKVTunerCellID(_ value: String) -> Bool {
        let prefix = "kvtuner-g"
        guard value.hasPrefix(prefix) else { return false }
        let fields = value.dropFirst(prefix.count).split(
            separator: "-", omittingEmptySubsequences: false)
        guard fields.count == 2, fields[1].first == "b" else {
            return false
        }
        let groupText = String(fields[0])
        let bitsText = String(fields[1].dropFirst())
        guard let groupSize = Int(groupText),
            [64, 128].contains(groupSize),
            String(groupSize) == groupText,
            let nominalAverageBits = Double(bitsText),
            nominalAverageBits.isFinite,
            nominalAverageBits > 0,
            String(nominalAverageBits) == bitsText
        else { return false }
        return true
    }

    fileprivate static func isCleanGitSHA(_ value: String) -> Bool {
        ([40, 64].contains(value.count))
            && value.unicodeScalars.allSatisfy {
                CharacterSet(charactersIn: "0123456789abcdefABCDEF")
                    .contains($0)
            }
    }

    fileprivate static func isIdentifier(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty && trimmed == value && value != "unknown"
            && !value.contains("\n") && !value.contains("\r")
    }

    fileprivate static func isHex(_ value: String, length: Int) -> Bool {
        value.count == length && value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdefABCDEF")
                .contains($0)
        }
    }

    private static func validateProvenance(
        _ provenance: Provenance,
        corpus: TaskCoherenceCorpus
    ) throws {
        guard isIdentifier(provenance.date),
            isIdentifier(provenance.hardwareChip),
            provenance.hardwareRAMBytes > 0,
            isIdentifier(provenance.hardwareOS),
            isIdentifier(provenance.harnessGitSHA),
            isIdentifier(provenance.mlxSwiftVersion),
            isIdentifier(provenance.modelPath),
            isIdentifier(provenance.modelConfigHash),
            provenance.modelCheckpointManifestHash.map(isIdentifier) == true,
            provenance.corpusId == corpus.id,
            provenance.corpusContentHash == corpus.contentHash,
            isIdentifier(provenance.nonce)
        else {
            throw TaskCoherenceEvidenceError.invalidPromotionProvenance(
                "runtime")
        }
    }
}

/// Paired fp16/candidate task result. The hard-floor result is preserved even when false so a
/// rejected cell can be documented, but `validated(with:corpus:)` refuses promotion in that state.
public struct TaskCoherencePromotionEvidence:
    Codable, Equatable, Sendable
{
    public let schemaVersion: Int
    public let candidate: TaskCoherenceArtifactSummary
    public let reference: TaskCoherenceArtifactSummary
    public let assessment: TaskCoherenceAssessment

    private init(
        schemaVersion: Int,
        candidate: TaskCoherenceArtifactSummary,
        reference: TaskCoherenceArtifactSummary,
        assessment: TaskCoherenceAssessment
    ) {
        self.schemaVersion = schemaVersion
        self.candidate = candidate
        self.reference = reference
        self.assessment = assessment
    }

    public static func derive(
        candidate: TaskCoherenceArtifactSummary,
        reference: TaskCoherenceArtifactSummary,
        corpus: TaskCoherenceCorpus
    ) throws -> TaskCoherencePromotionEvidence {
        try TaskCoherenceArtifact.validateSummary(candidate, corpus: corpus)
        try TaskCoherenceArtifact.validateSummary(reference, corpus: corpus)
        guard candidate.identity.kvQuantTier != "fp16",
            reference.identity.kvQuantTier == "fp16",
            reference.cellID == "fp16",
            candidate.matrixID == reference.matrixID,
            candidate.runConfiguration == reference.runConfiguration,
            candidate.referenceArtifactSHA256 == reference.artifactSHA256,
            TaskCoherenceArtifact.referenceModelIdentityMatches(
                reference: reference.identity,
                candidate: KVModelEvidenceIdentity(
                    configHash: candidate.identity.modelConfigHash,
                    checkpointManifestHash:
                        candidate.identity.modelCheckpointManifestHash,
                    checkpointContentSHA256:
                        candidate.identity
                            .modelCheckpointContentSHA256)),
            TaskCoherenceArtifact.matchingRuntime(
                candidate.provenance, reference.provenance),
            zip(candidate.cases, reference.cases).allSatisfy({
                $0.tokenization == $1.tokenization
                    && $0.layout == $1.layout
            })
        else {
            throw TaskCoherenceEvidenceError.referenceArtifactMismatch
        }
        let assessment = try TaskCoherenceAssessment.derive(
            candidate: TaskCoherenceScoredRun(
                identity: candidate.identity, scores: candidate.scores),
            reference: TaskCoherenceScoredRun(
                identity: reference.identity, scores: reference.scores),
            corpus: corpus)
        return TaskCoherencePromotionEvidence(
            schemaVersion: 1, candidate: candidate,
            reference: reference, assessment: assessment)
    }

    @discardableResult
    public func validated(
        with klRecord: ResultRecord<KLPayload>,
        corpus: TaskCoherenceCorpus
    ) throws -> Self {
        guard schemaVersion == 1 else {
            throw TaskCoherenceEvidenceError.unsupportedSchema(schemaVersion)
        }
        let derived = try Self.derive(
            candidate: candidate, reference: reference, corpus: corpus)
        guard derived == self else {
            throw TaskCoherenceEvidenceError.invalidArtifact
        }
        guard assessment.hardFloorPassed else {
            throw TaskCoherenceEvidenceError.hardFloorFailed
        }
        try klRecord.validatedForPromotionEvidence()
        guard TaskCoherenceArtifact.isCleanGitSHA(
            candidate.provenance.harnessGitSHA),
            TaskCoherenceArtifact.isCleanGitSHA(
                reference.provenance.harnessGitSHA),
            TaskCoherenceArtifact.matchingRuntime(
                candidate.provenance, klRecord.provenance),
            let frontier = klRecord.payload.frontier
        else {
            throw TaskCoherenceEvidenceError.invalidPromotionProvenance(
                "runtime")
        }
        let schedulePairMatches: Bool
        switch (
            candidate.identity.kvtunerSchedule,
            frontier.candidateKVTunerSchedule
        ) {
        case (.none, .none):
            schedulePairMatches = true
        case (.some(let taskSchedule), .some(let klSchedule)):
            schedulePairMatches = taskSchedule.sameSchedule(as: klSchedule)
        default:
            schedulePairMatches = false
        }
        guard schedulePairMatches else {
            throw TaskCoherenceEvidenceError.klEvidenceMismatch(
                "kvtuner-schedule")
        }
        guard candidate.matrixID == frontier.matrixID,
            candidate.cellID == frontier.cellID,
            candidate.identity.kvQuantTier == klRecord.payload.kvQuantTier,
            frontier.candidateFormat?.kind
                == TaskCoherenceArtifact.expectedFormatKind(
                    for: candidate.identity.kvQuantTier),
            reference.identity.kvQuantTier
                == frontier.referenceKVQuantTier,
            candidate.identity.modelConfigHash
                == frontier.candidateModel.configHash,
            candidate.identity.modelCheckpointManifestHash
                == frontier.candidateModel.checkpointManifestHash,
            candidate.identity.modelCheckpointContentSHA256
                == frontier.candidateModel.checkpointContentSHA256,
            reference.identity.modelConfigHash
                == frontier.referenceModel.configHash,
            reference.identity.modelCheckpointManifestHash
                == frontier.referenceModel.checkpointManifestHash,
            reference.identity.modelCheckpointContentSHA256
                == frontier.referenceModel.checkpointContentSHA256
        else {
            throw TaskCoherenceEvidenceError.klEvidenceMismatch("matrix")
        }
        return self
    }
}
