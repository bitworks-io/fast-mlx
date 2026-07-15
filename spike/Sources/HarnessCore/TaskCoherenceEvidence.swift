import CryptoKit
import Foundation

/// Stable lowercase SHA-256 used for task artifacts and preserved scored outputs.
public func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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
    public let kvarnCompletedTileCount: Int?
    public let kvarnCompressedTokens: Int?
    public let kvarnCodecIterations: Int?
    public let kvarnExecutionMode: String?

    public init(
        cachedTokens: Int?,
        affineTokens: Int?,
        kvarnCompletedTileCount: Int?,
        kvarnCompressedTokens: Int?,
        kvarnCodecIterations: Int?,
        kvarnExecutionMode: String?
    ) {
        self.cachedTokens = cachedTokens
        self.affineTokens = affineTokens
        self.kvarnCompletedTileCount = kvarnCompletedTileCount
        self.kvarnCompressedTokens = kvarnCompressedTokens
        self.kvarnCodecIterations = kvarnCodecIterations
        self.kvarnExecutionMode = kvarnExecutionMode
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
    public let layout: TaskCoherencePromptLayoutEvidence
    public let generatedTokenCount: Int
    /// The exact scorer input: one of A/B/C/D or the decoded structured-tool generation.
    public let scoredOutput: String
    public let outputSHA256: String
    public let score: TaskItemScore
    public let engagement: TaskCoherenceCacheEngagementEvidence

    public init(
        schemaVersion: Int,
        matrixID: String,
        cellID: String,
        identity: TaskCoherenceRunIdentity,
        referenceArtifactSHA256: String?,
        promptContentHash: String,
        layout: TaskCoherencePromptLayoutEvidence,
        generatedTokenCount: Int,
        scoredOutput: String,
        outputSHA256: String,
        score: TaskItemScore,
        engagement: TaskCoherenceCacheEngagementEvidence
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.cellID = cellID
        self.identity = identity
        self.referenceArtifactSHA256 = referenceArtifactSHA256
        self.promptContentHash = promptContentHash
        self.layout = layout
        self.generatedTokenCount = generatedTokenCount
        self.scoredOutput = scoredOutput
        self.outputSHA256 = outputSHA256
        self.score = score
        self.engagement = engagement
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
        self.provenance = provenance
        self.caseCount = caseCount
        self.cases = cases
        self.scores = scores
    }
}

public enum TaskCoherenceArtifact {
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
                row.payload.referenceArtifactSHA256
                    == first.payload.referenceArtifactSHA256
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
        let scores = cases.map(\.score)
        return TaskCoherenceArtifactSummary(
            schemaVersion: first.payload.schemaVersion,
            matrixID: first.payload.matrixID,
            cellID: first.payload.cellID,
            artifactSHA256: artifactSHA256,
            referenceArtifactSHA256:
                first.payload.referenceArtifactSHA256,
            identity: first.payload.identity,
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
        guard payload.schemaVersion == 1 else {
            throw TaskCoherenceEvidenceError.unsupportedSchema(
                payload.schemaVersion)
        }
        guard isIdentifier(payload.matrixID), isIdentifier(payload.cellID),
            isIdentifier(payload.identity.corpusID),
            isIdentifier(payload.identity.corpusContentHash),
            isIdentifier(payload.identity.modelConfigHash),
            isIdentifier(payload.identity.modelCheckpointManifestHash),
            isIdentifier(payload.identity.kvQuantTier),
            tierCellIDs[payload.identity.kvQuantTier] == payload.cellID,
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
        let derivedScore: TaskItemScore
        switch item.scoringMode {
        case .restrictedChoice:
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
            layout: payload.layout,
            generatedTokenCount: payload.generatedTokenCount)

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
        layout: TaskCoherencePromptLayoutEvidence,
        generatedTokenCount: Int
    ) throws {
        let (generatedInputs, generatedUnderflow) = generatedTokenCount
            .subtractingReportingOverflow(1)
        let (expectedCachedTokens, cachedOverflow) = layout.promptTokens
            .addingReportingOverflow(generatedInputs)
        guard !generatedUnderflow, !cachedOverflow,
            generatedInputs >= 0
        else {
            throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                "cachedTokens")
        }
        let kvarnValues = [
            engagement.kvarnCompletedTileCount,
            engagement.kvarnCompressedTokens,
            engagement.kvarnCodecIterations,
        ]
        if tier == "fp16" {
            guard engagement.cachedTokens == nil,
                engagement.affineTokens == nil,
                kvarnValues.allSatisfy({ $0 == nil }),
                engagement.kvarnExecutionMode == nil
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence("fp16")
            }
            return
        }
        if affineTiers.contains(tier) {
            guard let cachedTokens = engagement.cachedTokens,
                let affineTokens = engagement.affineTokens,
                cachedTokens == expectedCachedTokens,
                cachedTokens == affineTokens,
                kvarnValues.allSatisfy({ $0 == nil }),
                engagement.kvarnExecutionMode == nil
            else {
                throw TaskCoherenceEvidenceError.invalidRuntimeEvidence(
                    "affine")
            }
            return
        }
        if let expectedIterations = kvarnIterations[tier] {
            guard let cachedTokens = engagement.cachedTokens,
                engagement.affineTokens == nil,
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
                engagement.kvarnExecutionMode == "uncompiled-correctness"
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
        guard summary.schemaVersion == 1 else {
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
        if kvarnIterations[tier] != nil { return .kvarn }
        return tier == "fp16" ? .fp16 : nil
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
            candidate.referenceArtifactSHA256 == reference.artifactSHA256,
            TaskCoherenceArtifact.matchingRuntime(
                candidate.provenance, reference.provenance)
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
            reference.identity.modelConfigHash
                == frontier.referenceModel.configHash,
            reference.identity.modelCheckpointManifestHash
                == frontier.referenceModel.checkpointManifestHash
        else {
            throw TaskCoherenceEvidenceError.klEvidenceMismatch("matrix")
        }
        return self
    }
}
