import Foundation

public enum KVTunerRuntimeSelectionError: Error, Equatable, Sendable {
    case malformedArtifact
    case malformedQualificationBundle
    case unsupportedQualificationBundleSchema(Int)
    case qualificationArtifactMismatch(String)
    case tokenizerIdentityMismatch
    case promptTokenizationMismatch(phase: String, position: Int)
    case missingEvaluationCorpus
    case invalidSchedule(KVTunerScheduleError)
    case invalidEvaluationCorpus(index: Int, reason: KVTunerScheduleError)
    case unexpectedValidationFailure
}

private struct KVTunerQualificationBundleDiscriminator: Decodable {
    let schemaVersion: Int
}

/// Schema-3 KVTuner schedules pin prompt fingerprints to FNV-1a-64 over the exact UTF-8 text.
/// Accepting an unlabelled second digest width would let identical calibration/evaluation text
/// appear disjoint merely because one side used SHA-256 and the other used FNV.
public enum KVTunerPromptDigest: Sendable {
    public static let algorithm = "fnv1a64-utf8-v1"

    public static func exactText(_ value: String) -> String {
        fnv1a64(value.utf8)
    }

    public static func isCanonical(_ value: String) -> Bool {
        guard value.count == 16 else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}

/// Stable evaluation provenance supplied at the same trust boundary as the schedule artifact.
/// There is deliberately no public initializer: custom callers can select canonical source rows,
/// while the audited value is created only after an exact checked-in corpus passes its SHA pin.
public struct KVTunerEvaluationSourceProvenance: Hashable, Sendable {
    fileprivate enum Storage: String, Sendable {
        case firstPartyAuditedNoGSM8K =
            "first-party-audited-no-gsm8k-v1"
        case canonicalSourceItems = "canonical-source-item-digests-v1"
    }

    fileprivate let storage: Storage

    /// The caller supplies canonical upstream source-row digests (for example GSM8K indices).
    public static let canonicalSourceItems = Self(
        storage: .canonicalSourceItems)

    /// The checked-in fast-mlx corpus is audited as first-party material with no GSM8K rows.
    /// Internal so public/custom callers cannot assert the privilege directly.
    static let firstPartyAuditedNoGSM8K = Self(
        storage: .firstPartyAuditedNoGSM8K)

    static func auditedSourceItemDigests(
        entryDigests: [String]
    ) -> [String] {
        entryDigests.map { entryDigest in
            var data = Data(
                "fast-mlx.kvtuner-evaluation-source.first-party.v1\0".utf8)
            data.append(contentsOf: entryDigest.utf8)
            return sha256Hex(data)
        }.sorted()
    }
}

public enum KVTunerEvaluationCorpusIdentityError: Error, Equatable, Sendable {
    case canonicalSourceItemsRequired
    case invalidCanonicalSourceItemDigest(String)
    case duplicateCanonicalSourceItemDigest
}

private enum KVTunerEvaluationCorpusAudit {
    // These pins are generated from the full, domain-separated canonical transcripts below.
    // They are not the legacy FNV aggregate identifiers used in result presentation.
    static let measurementTranscriptSHA256 =
        "f862ae55be658c3c352beb881423cb3077d57cef893d605525cc4963bfcdfd90"
    static let taskCoherenceV1TranscriptSHA256 =
        "e13644bccc0865c77935982dd5d717a7b8350fb8c6a204f6628dafcdf33fd809"
    static let taskCoherenceV2TranscriptSHA256 =
        "e3207cfa924523c461b100a48afde8cb21d8f056baf7870ad3edd47f7992a431"
    static let measurementIdentitySHA256 =
        "a6621294b3c3ab887cd6ae95906c5b15328fbd0eab6d4a7aeed24fab59bfafa3"
    static let taskCoherenceV1IdentitySHA256 =
        "d6af1e7afad23011869b2744dd55b96107740ecfa073c3a4e6e9581af3f6d681"
    static let taskCoherenceV2IdentitySHA256 =
        "a146bc7605ac9962ad66aa832ab77e0b17d69f9a11a39dc98706ed772c5174c9"

    static func expectedTaskCoherenceTranscriptSHA256(
        id: String,
        aggregateDigest: String
    ) -> String? {
        switch (id, aggregateDigest) {
        case ("kvarn-task-coherence-v1", "0f16e3abc00ec7c8"):
            taskCoherenceV1TranscriptSHA256
        case ("kvarn-task-coherence-v2", "1740d0d07f586def"):
            taskCoherenceV2TranscriptSHA256
        default:
            nil
        }
    }

    static func isRecognizedAuditedIdentity(
        transcriptSHA256: String,
        identitySHA256: String
    ) -> Bool {
        (transcriptSHA256 == measurementTranscriptSHA256
            && identitySHA256 == measurementIdentitySHA256)
            || (transcriptSHA256 == taskCoherenceV1TranscriptSHA256
                && identitySHA256 == taskCoherenceV1IdentitySHA256)
            || (transcriptSHA256 == taskCoherenceV2TranscriptSHA256
                && identitySHA256 == taskCoherenceV2IdentitySHA256)
    }

    static func measurementTranscriptSHA256(
        _ corpus: MeasurementCorpus
    ) -> String {
        var transcript = Data(
            "fast-mlx.kvtuner.audited-measurement-corpus.v1\0".utf8)
        appendField(corpus.corpusId, to: &transcript)
        let entries = corpus.entries.sorted {
            ($0.id, $0.tag.rawValue, $0.text)
                < ($1.id, $1.tag.rawValue, $1.text)
        }
        appendField(String(entries.count), to: &transcript)
        for entry in entries {
            appendField(entry.id, to: &transcript)
            appendField(entry.tag.rawValue, to: &transcript)
            appendField(entry.text, to: &transcript)
        }
        return sha256Hex(transcript)
    }

    static func taskCoherenceTranscriptSHA256(
        _ corpus: TaskCoherenceCorpus
    ) -> String {
        var transcript = Data(
            "fast-mlx.kvtuner.audited-task-coherence-corpus.v1\0".utf8)
        appendField(String(corpus.schemaVersion), to: &transcript)
        appendField(corpus.id, to: &transcript)
        appendField(String(corpus.items.count), to: &transcript)
        for item in corpus.items {
            appendField(item.id, to: &transcript)
            appendField(item.domain.rawValue, to: &transcript)
            appendField(item.scoringMode.rawValue, to: &transcript)
            appendField(item.prefix, to: &transcript)
            appendField(item.material, to: &transcript)
            appendField(item.suffix, to: &transcript)
            appendField(item.query, to: &transcript)
            if let expectedChoice = item.expectedChoice {
                appendField("choice", to: &transcript)
                appendField(expectedChoice, to: &transcript)
            } else {
                appendField("no-choice", to: &transcript)
            }
            if let expectedTool = item.expectedTool {
                appendField("tool", to: &transcript)
                appendField(expectedTool.name, to: &transcript)
                appendField(
                    String(expectedTool.arguments.count),
                    to: &transcript)
                for key in expectedTool.arguments.keys.sorted() {
                    appendField(key, to: &transcript)
                    appendField(
                        expectedTool.arguments[key] ?? "",
                        to: &transcript)
                }
            } else {
                appendField("no-tool", to: &transcript)
            }
        }
        return sha256Hex(transcript)
    }

    static func identitySHA256(
        id: String,
        aggregateDigest: String,
        canonicalEntryDigests: [String],
        canonicalSourceItemDigests: [String],
        auditedContentTranscriptSHA256: String
    ) -> String {
        var transcript = Data(
            "fast-mlx.kvtuner.audited-evaluation-identity.v1\0".utf8)
        appendField(id, to: &transcript)
        appendField(aggregateDigest, to: &transcript)
        appendField(
            String(canonicalEntryDigests.count), to: &transcript)
        for digest in canonicalEntryDigests {
            appendField(digest, to: &transcript)
        }
        appendField(
            String(canonicalSourceItemDigests.count), to: &transcript)
        for digest in canonicalSourceItemDigests {
            appendField(digest, to: &transcript)
        }
        appendField(auditedContentTranscriptSHA256, to: &transcript)
        return sha256Hex(transcript)
    }

    private static func appendField(_ value: String, to data: inout Data) {
        let bytes = Data(value.utf8)
        var length = UInt64(bytes.count).bigEndian
        withUnsafeBytes(of: &length) {
            data.append(contentsOf: $0)
        }
        data.append(bytes)
    }
}

public struct KVTunerEvaluationCorpusIdentity: Codable, Hashable, Sendable {
    public let id: String
    public let aggregateDigest: String
    public let canonicalEntryDigests: [String]
    let sourceProvenance: KVTunerEvaluationSourceProvenance
    public let canonicalSourceItemDigests: [String]
    private let auditedContentTranscriptSHA256: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case aggregateDigest
        case canonicalEntryDigests
        case sourceProvenance
        case canonicalSourceItemDigests
        case auditedContentTranscriptSHA256
    }

    /// Constructs a custom identity only when canonical upstream source rows are present.
    public init(
        id: String,
        aggregateDigest: String,
        canonicalEntryDigests: [String],
        canonicalSourceItemDigests: [String]
    ) throws {
        guard !canonicalSourceItemDigests.isEmpty else {
            throw KVTunerEvaluationCorpusIdentityError
                .canonicalSourceItemsRequired
        }
        guard Set(canonicalSourceItemDigests).count
            == canonicalSourceItemDigests.count
        else {
            throw KVTunerEvaluationCorpusIdentityError
                .duplicateCanonicalSourceItemDigest
        }
        for digest in canonicalSourceItemDigests {
            guard Self.isLowercaseHex(digest, length: 64) else {
                throw KVTunerEvaluationCorpusIdentityError
                    .invalidCanonicalSourceItemDigest(digest)
            }
        }
        self.id = id
        self.aggregateDigest = aggregateDigest
        self.canonicalEntryDigests = canonicalEntryDigests
        sourceProvenance = .canonicalSourceItems
        self.canonicalSourceItemDigests = canonicalSourceItemDigests.sorted()
        auditedContentTranscriptSHA256 = nil
    }

    private init(
        id: String,
        aggregateDigest: String,
        canonicalEntryDigests: [String],
        auditedContentTranscriptSHA256: String
    ) {
        self.id = id
        self.aggregateDigest = aggregateDigest
        self.canonicalEntryDigests = canonicalEntryDigests
        sourceProvenance = .firstPartyAuditedNoGSM8K
        canonicalSourceItemDigests = KVTunerEvaluationSourceProvenance
            .auditedSourceItemDigests(entryDigests: canonicalEntryDigests)
        self.auditedContentTranscriptSHA256 =
            auditedContentTranscriptSHA256
    }

    /// Stable leakage identity for the exact strings supplied to measurement tokenization.
    /// Entry IDs are deliberately excluded: renaming a reused prompt cannot hide overlap with
    /// the schedule's calibration corpus.
    public static func measurementCorpus(
        _ corpus: MeasurementCorpus
    ) throws -> KVTunerEvaluationCorpusIdentity {
        let transcriptSHA256 = KVTunerEvaluationCorpusAudit
            .measurementTranscriptSHA256(corpus)
        let isAuditedBuiltIn = corpus.corpusId == "measurement-corpus-v2"
            && corpus.contentHash == "8dd73ade100742f2"
            && corpus.entries.count == 5
            && transcriptSHA256 == KVTunerEvaluationCorpusAudit
                .measurementTranscriptSHA256
        guard isAuditedBuiltIn else {
            throw KVTunerEvaluationCorpusIdentityError
                .canonicalSourceItemsRequired
        }
        let identity = KVTunerEvaluationCorpusIdentity(
            id: corpus.corpusId,
            aggregateDigest: corpus.contentHash,
            canonicalEntryDigests: corpus.entries
                .map { KVTunerPromptDigest.exactText($0.text) }
                .sorted(),
            auditedContentTranscriptSHA256: transcriptSHA256)
        return identity
    }

    /// Leakage identity for the exact salted prompts used by the batch-1 runtime frontier.
    /// Only the pinned first-party default source is admitted; custom prompts must supply real
    /// canonical upstream source rows through the public initializer instead of borrowing this
    /// audited identity.
    public static func benchWorkload(
        _ workload: BenchWorkloadIdentity
    ) throws -> KVTunerEvaluationCorpusIdentity {
        let expectedPromptSHA256 =
            "aaa70310381eb25ba917e680397c141e494ae62e174dc42de3e5f0b2a4a261a4"
        guard workload.basePrompt == defaultBenchPrompt,
            sha256Hex(Data(workload.basePrompt.utf8)) == expectedPromptSHA256
        else {
            throw KVTunerEvaluationCorpusIdentityError
                .canonicalSourceItemsRequired
        }
        let id = "fastmlx-bench-decode-v2"
        let prompts = workload.prompts
        let entryDigests = prompts
            .map(KVTunerPromptDigest.exactText)
            .sorted()
        let aggregateDigest = fnv1a64(
            ([id] + prompts).joined(separator: "\0").utf8)
        var sourceTranscript = Data(
            "fast-mlx.kvtuner-bench-source.v2\0".utf8)
        sourceTranscript.append(contentsOf: workload.basePrompt.utf8)
        return try KVTunerEvaluationCorpusIdentity(
            id: id,
            aggregateDigest: aggregateDigest,
            canonicalEntryDigests: entryDigests,
            canonicalSourceItemDigests: [sha256Hex(sourceTranscript)])
    }

    public static func measurementCorpus(
        _ corpus: MeasurementCorpus,
        canonicalSourceItemDigests: [String]
    ) throws -> KVTunerEvaluationCorpusIdentity {
        try KVTunerEvaluationCorpusIdentity(
            id: corpus.corpusId,
            aggregateDigest: corpus.contentHash,
            canonicalEntryDigests: corpus.entries
                .map { KVTunerPromptDigest.exactText($0.text) }
                .sorted(),
            canonicalSourceItemDigests: canonicalSourceItemDigests)
    }

    /// Stable leakage identity for the fully expanded task prompts actually passed to the model.
    /// Item and corpus renames may change the aggregate identity, but cannot change these exact
    /// prompt fingerprints.
    public static func taskCoherenceCorpus(
        _ corpus: TaskCoherenceCorpus
    ) throws -> KVTunerEvaluationCorpusIdentity {
        let transcriptSHA256 = KVTunerEvaluationCorpusAudit
            .taskCoherenceTranscriptSHA256(corpus)
        let expectedTranscriptSHA256 = KVTunerEvaluationCorpusAudit
            .expectedTaskCoherenceTranscriptSHA256(
                id: corpus.id,
                aggregateDigest: corpus.contentHash)
        let isAuditedBuiltIn = corpus.schemaVersion == 1
            && expectedTranscriptSHA256 != nil
            && corpus.items.count
                == TaskCoherenceCorpus.requiredItemsPerDomain
                    * TaskCoherenceDomain.allCases.count
            && transcriptSHA256 == expectedTranscriptSHA256
        guard isAuditedBuiltIn else {
            throw KVTunerEvaluationCorpusIdentityError
                .canonicalSourceItemsRequired
        }
        return KVTunerEvaluationCorpusIdentity(
            id: corpus.id,
            aggregateDigest: corpus.contentHash,
            canonicalEntryDigests: corpus.items
                .map { KVTunerPromptDigest.exactText($0.prompt) }
                .sorted(),
            auditedContentTranscriptSHA256: transcriptSHA256)
    }

    public static func taskCoherenceCorpus(
        _ corpus: TaskCoherenceCorpus,
        canonicalSourceItemDigests: [String]
    ) throws -> KVTunerEvaluationCorpusIdentity {
        try KVTunerEvaluationCorpusIdentity(
            id: corpus.id,
            aggregateDigest: corpus.contentHash,
            canonicalEntryDigests: corpus.items
                .map { KVTunerPromptDigest.exactText($0.prompt) }
                .sorted(),
            canonicalSourceItemDigests: canonicalSourceItemDigests)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let id = try container.decode(String.self, forKey: .id)
        let aggregateDigest = try container.decode(
            String.self, forKey: .aggregateDigest)
        let entryDigests = try container.decode(
            [String].self, forKey: .canonicalEntryDigests)
        let provenance = try container.decode(
            String.self, forKey: .sourceProvenance)
        let sourceDigests = try container.decode(
            [String].self, forKey: .canonicalSourceItemDigests)

        if provenance
            == KVTunerEvaluationSourceProvenance.Storage
                .canonicalSourceItems.rawValue
        {
            guard !container.contains(.auditedContentTranscriptSHA256) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .sourceProvenance,
                    in: container,
                    debugDescription: "canonical identity carries audited content")
            }
            do {
                self = try KVTunerEvaluationCorpusIdentity(
                    id: id,
                    aggregateDigest: aggregateDigest,
                    canonicalEntryDigests: entryDigests,
                    canonicalSourceItemDigests: sourceDigests)
            } catch {
                throw DecodingError.dataCorruptedError(
                    forKey: .canonicalSourceItemDigests,
                    in: container,
                    debugDescription: "invalid canonical source provenance")
            }
            return
        }

        guard provenance
            == KVTunerEvaluationSourceProvenance.Storage
                .firstPartyAuditedNoGSM8K.rawValue,
            let transcriptSHA256 = try container.decodeIfPresent(
                String.self, forKey: .auditedContentTranscriptSHA256)
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .sourceProvenance,
                in: container,
                debugDescription: "unknown evaluation provenance")
        }
        let decoded = KVTunerEvaluationCorpusIdentity(
            id: id,
            aggregateDigest: aggregateDigest,
            canonicalEntryDigests: entryDigests,
            auditedContentTranscriptSHA256: transcriptSHA256)
        let identitySHA256 = KVTunerEvaluationCorpusAudit.identitySHA256(
            id: decoded.id,
            aggregateDigest: decoded.aggregateDigest,
            canonicalEntryDigests: decoded.canonicalEntryDigests,
            canonicalSourceItemDigests:
                decoded.canonicalSourceItemDigests,
            auditedContentTranscriptSHA256: transcriptSHA256)
        guard sourceDigests == decoded.canonicalSourceItemDigests,
            KVTunerEvaluationCorpusAudit.isRecognizedAuditedIdentity(
                transcriptSHA256: transcriptSHA256,
                identitySHA256: identitySHA256)
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .sourceProvenance,
                in: container,
                debugDescription: "unrecognized audited corpus identity")
        }
        self = decoded
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(aggregateDigest, forKey: .aggregateDigest)
        try container.encode(
            canonicalEntryDigests, forKey: .canonicalEntryDigests)
        try container.encode(
            sourceProvenance.storage.rawValue,
            forKey: .sourceProvenance)
        try container.encode(
            canonicalSourceItemDigests,
            forKey: .canonicalSourceItemDigests)
        try container.encodeIfPresent(
            auditedContentTranscriptSHA256,
            forKey: .auditedContentTranscriptSHA256)
    }

    private static func isLowercaseHex(
        _ value: String, length: Int
    ) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef")
                .contains($0)
        }
    }
}

/// Runtime-only copy of one validated layer decision. Unlike `KVLayerPrecision`, these fields
/// cannot be changed after the authenticated selection is constructed.
public struct KVTunerRuntimeLayerPolicy: Codable, Hashable, Sendable {
    public let layer: Int
    public let keyBits: Int
    public let valueBits: Int

    public init(layer: Int, keyBits: Int, valueBits: Int) {
        self.layer = layer
        self.keyBits = keyBits
        self.valueBits = valueBits
    }
}

/// Immutable result of authenticating exact KVTuner JSON bytes for one runtime invocation.
/// There is intentionally no public memberwise initializer: callers can obtain a selection only
/// by decoding the bytes, matching every requested identity, and passing the calibration-leakage
/// gate for every declared evaluation corpus.
public struct KVTunerRuntimeSelection: Hashable, Sendable {
    /// SHA-256 of the exact schedule bytes that define runtime cache behavior.
    public let artifactSHA256: String
    /// SHA-256 of the complete qualification bundle used to authenticate this selection.
    /// This is provenance, not the runtime schedule identity: reformatting the outer JSON must
    /// not make an otherwise identical policy look different to task/KL schedule pairing.
    public let qualificationBundleSHA256: String
    public let schemaVersion: Int
    public let matrixID: String
    public let cellID: String
    public let modelConfigHash: String
    public let checkpointManifestHash: String
    public let tokenizerSHA256: String
    public let groupSize: Int
    public let promptDigestAlgorithm: String
    public let calibrationCorpusID: String
    public let calibrationCorpusHash: String
    public let calibrationEntryDigests: [String]
    public let calibrationSourceItemDigests: [String]
    public let seed: UInt64
    public let objective: String
    public let nominalAverageBits: Double
    public let sourceSensitivityArtifactSHA256: String
    public let sourceSearchArtifactSHA256: String
    public let evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    public let layers: [KVTunerRuntimeLayerPolicy]

    private init(
        artifactSHA256: String,
        qualificationBundleSHA256: String,
        schedule: KVTunerSchedule,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    ) {
        self.artifactSHA256 = artifactSHA256
        self.qualificationBundleSHA256 = qualificationBundleSHA256
        schemaVersion = schedule.schemaVersion
        matrixID = schedule.matrixID
        cellID = schedule.cellID
        modelConfigHash = schedule.modelConfigHash
        checkpointManifestHash = schedule.checkpointManifestHash
        tokenizerSHA256 = schedule.tokenizerSHA256
        groupSize = schedule.groupSize
        promptDigestAlgorithm = KVTunerPromptDigest.algorithm
        calibrationCorpusID = schedule.calibrationCorpusID
        calibrationCorpusHash = schedule.calibrationCorpusHash
        calibrationEntryDigests = schedule.calibrationEntryHashes
        calibrationSourceItemDigests =
            schedule.calibrationSourceItemDigests
        seed = schedule.seed
        objective = schedule.objective
        nominalAverageBits = schedule.nominalAverageBits
        sourceSensitivityArtifactSHA256 =
            schedule.sourceSensitivityArtifactSHA256
        sourceSearchArtifactSHA256 = schedule.sourceSearchArtifactSHA256
        self.evaluationCorpora = evaluationCorpora
        layers = schedule.layers.map {
            KVTunerRuntimeLayerPolicy(
                layer: $0.layer,
                keyBits: $0.keyBits,
                valueBits: $0.valueBits)
        }
    }

    /// Internal structural loader used only by unit tests that exercise downstream binding/cache
    /// behavior in isolation. Production callers cannot bypass `loadQualified`.
    static func loadForTesting(
        artifactData: Data,
        expectedLayerCount: Int,
        expectedMatrixID: String,
        expectedCellID: String,
        expectedModelConfigHash: String,
        expectedCheckpointManifestHash: String,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    ) throws -> KVTunerRuntimeSelection {
        let schedule: KVTunerSchedule
        do {
            schedule = try JSONDecoder().decode(
                KVTunerSchedule.self,
                from: artifactData)
        } catch let reason as KVTunerScheduleError {
            throw KVTunerRuntimeSelectionError.invalidSchedule(reason)
        } catch {
            throw KVTunerRuntimeSelectionError.malformedArtifact
        }

        let validatedSchedule: KVTunerSchedule
        do {
            validatedSchedule = try schedule.validated(
                expectedLayerCount: expectedLayerCount,
                expectedMatrixID: expectedMatrixID,
                expectedCellID: expectedCellID,
                expectedModelConfigHash: expectedModelConfigHash,
                expectedCheckpointManifestHash:
                    expectedCheckpointManifestHash)
        } catch let reason as KVTunerScheduleError {
            throw KVTunerRuntimeSelectionError.invalidSchedule(reason)
        } catch {
            throw KVTunerRuntimeSelectionError.unexpectedValidationFailure
        }

        guard !evaluationCorpora.isEmpty else {
            throw KVTunerRuntimeSelectionError.missingEvaluationCorpus
        }
        for (index, corpus) in evaluationCorpora.enumerated() {
            do {
                try validatedSchedule.validateEvaluationCorpus(
                    id: corpus.id,
                    hash: corpus.aggregateDigest,
                    entryHashes: corpus.canonicalEntryDigests,
                    sourceProvenance: corpus.sourceProvenance,
                    sourceItemHashes:
                        corpus.canonicalSourceItemDigests)
            } catch let reason as KVTunerScheduleError {
                throw KVTunerRuntimeSelectionError.invalidEvaluationCorpus(
                    index: index,
                    reason: reason)
            } catch {
                throw KVTunerRuntimeSelectionError.unexpectedValidationFailure
            }
        }

        let artifactSHA256 = sha256Hex(artifactData)
        return KVTunerRuntimeSelection(
            artifactSHA256: artifactSHA256,
            qualificationBundleSHA256: artifactSHA256,
            schedule: validatedSchedule,
            evaluationCorpora: evaluationCorpora)
    }

    /// Re-derives the runtime schedule from every exact qualification artifact before any policy
    /// can reach MLX. The schedule inside the bundle is evidence output, never its own trust root.
    public static func loadQualified(
        artifactData: Data,
        exactModelConfigData: Data,
        expectedMatrixID: String,
        expectedCellID: String,
        expectedCheckpointManifestHash: String,
        expectedTokenizerSHA256: String,
        expectedEOSTokenID: Int,
        tokenizePrompt: (String) throws -> [Int],
        decodeTokenIDs: ([Int]) throws -> String,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    ) throws -> KVTunerRuntimeSelection {
        let discriminator: KVTunerQualificationBundleDiscriminator
        do {
            discriminator = try JSONDecoder().decode(
                KVTunerQualificationBundleDiscriminator.self,
                from: artifactData)
        } catch {
            throw KVTunerRuntimeSelectionError
                .malformedQualificationBundle
        }
        guard discriminator.schemaVersion == 1 else {
            throw KVTunerRuntimeSelectionError
                .unsupportedQualificationBundleSchema(
                    discriminator.schemaVersion)
        }
        let bundle: KVTunerQualificationBundle
        do {
            bundle = try JSONDecoder().decode(
                KVTunerQualificationBundle.self, from: artifactData)
        } catch {
            throw KVTunerRuntimeSelectionError
                .malformedQualificationBundle
        }
        // The discriminator is checked before current-schema fields are decoded so a genuine
        // older/newer bundle is never mislabeled as malformed merely because its shape differs.

        let manifest: KVTunerCalibrationManifest
        let sensitivity: KVTunerSensitivityArtifact
        let search: KVTunerSearchArtifact
        let suppliedSchedule: KVTunerSchedule
        do {
            manifest = try JSONDecoder().decode(
                KVTunerCalibrationManifest.self,
                from: bundle.calibrationManifestData)
            sensitivity = try JSONDecoder().decode(
                KVTunerSensitivityArtifact.self,
                from: bundle.sensitivityArtifactData)
            search = try JSONDecoder().decode(
                KVTunerSearchArtifact.self,
                from: bundle.searchArtifactData)
            suppliedSchedule = try JSONDecoder().decode(
                KVTunerSchedule.self, from: bundle.scheduleData)
        } catch {
            throw KVTunerRuntimeSelectionError
                .qualificationArtifactMismatch("decode")
        }

        let candidateRuntimePolicies: [KVTunerCandidateRuntimePolicy]
        do {
            candidateRuntimePolicies = try search.candidates.indices.map {
                candidateOrdinal in
                try KVTunerCandidateRuntimePolicy.load(
                    exactCalibrationManifestData:
                        bundle.calibrationManifestData,
                    exactSensitivityArtifactData:
                        bundle.sensitivityArtifactData,
                    exactModelConfigData: exactModelConfigData,
                    expectedCheckpointManifestHash:
                        expectedCheckpointManifestHash,
                    expectedTokenizerSHA256:
                        expectedTokenizerSHA256,
                    targetPairBitTotal: search.targetPairBitTotal,
                    maxCandidates: search.candidates.count,
                    candidateOrdinal: candidateOrdinal,
                    tokenizePrompt: tokenizePrompt)
            }
        } catch let error as KVTunerCandidateRuntimePolicyError {
            switch error {
            case .promptTokenizationMismatch(let phase, let position):
                throw KVTunerRuntimeSelectionError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            case .tokenizerIdentityMismatch:
                throw KVTunerRuntimeSelectionError
                    .tokenizerIdentityMismatch
            default:
                throw KVTunerRuntimeSelectionError
                    .qualificationArtifactMismatch("candidate-policies")
            }
        } catch {
            throw KVTunerRuntimeSelectionError
                .qualificationArtifactMismatch("candidate-policies")
        }

        let derivedSchedule: KVTunerSchedule
        do {
            derivedSchedule = try KVTunerScheduleSearch.makeSchedule(
                searchArtifact: search,
                exactSearchArtifactData: bundle.searchArtifactData,
                sensitivityArtifact: sensitivity,
                exactSensitivityArtifactData:
                    bundle.sensitivityArtifactData,
                calibrationManifest: manifest,
                exactCalibrationManifestData:
                    bundle.calibrationManifestData,
                exactCandidateEvaluationArtifactData:
                    bundle.candidateEvaluationArtifactData,
                candidateRuntimePolicies: candidateRuntimePolicies,
                eosTokenID: expectedEOSTokenID,
                decodeTokenIDs: decodeTokenIDs,
                exactModelConfigData: exactModelConfigData,
                expectedCheckpointManifestHash:
                    expectedCheckpointManifestHash)
        } catch {
            throw KVTunerRuntimeSelectionError
                .qualificationArtifactMismatch("derivation")
        }
        guard suppliedSchedule == derivedSchedule else {
            throw KVTunerRuntimeSelectionError
                .qualificationArtifactMismatch("schedule")
        }
        guard expectedTokenizerSHA256.count == 64,
            expectedTokenizerSHA256.unicodeScalars.allSatisfy({
                CharacterSet(charactersIn: "0123456789abcdef")
                    .contains($0)
            }),
            suppliedSchedule.tokenizerSHA256 == expectedTokenizerSHA256
        else {
            throw KVTunerRuntimeSelectionError.tokenizerIdentityMismatch
        }
        try validateLiveTokenization(
            manifest.sensitivityPrompts,
            phase: "sensitivity",
            tokenizePrompt: tokenizePrompt)
        try validateLiveTokenization(
            manifest.searchPrompts,
            phase: "search",
            tokenizePrompt: tokenizePrompt)

        let expectedLayerCount: Int
        do {
            expectedLayerCount = try KVTunerModelConfigPreflight.load(
                from: exactModelConfigData)
        } catch {
            throw KVTunerRuntimeSelectionError
                .qualificationArtifactMismatch("model-config")
        }
        let validatedSchedule: KVTunerSchedule
        do {
            validatedSchedule = try suppliedSchedule.validated(
                expectedLayerCount: expectedLayerCount,
                expectedMatrixID: expectedMatrixID,
                expectedCellID: expectedCellID,
                expectedModelConfigHash: fnv1a64(exactModelConfigData),
                expectedCheckpointManifestHash:
                    expectedCheckpointManifestHash)
        } catch let reason as KVTunerScheduleError {
            throw KVTunerRuntimeSelectionError.invalidSchedule(reason)
        } catch {
            throw KVTunerRuntimeSelectionError.unexpectedValidationFailure
        }

        guard !evaluationCorpora.isEmpty else {
            throw KVTunerRuntimeSelectionError.missingEvaluationCorpus
        }
        for (index, corpus) in evaluationCorpora.enumerated() {
            do {
                try validatedSchedule.validateEvaluationCorpus(
                    id: corpus.id,
                    hash: corpus.aggregateDigest,
                    entryHashes: corpus.canonicalEntryDigests,
                    sourceProvenance: corpus.sourceProvenance,
                    sourceItemHashes:
                        corpus.canonicalSourceItemDigests)
            } catch let reason as KVTunerScheduleError {
                throw KVTunerRuntimeSelectionError.invalidEvaluationCorpus(
                    index: index, reason: reason)
            } catch {
                throw KVTunerRuntimeSelectionError
                    .unexpectedValidationFailure
            }
        }
        return KVTunerRuntimeSelection(
            artifactSHA256: sha256Hex(bundle.scheduleData),
            qualificationBundleSHA256: sha256Hex(artifactData),
            schedule: validatedSchedule,
            evaluationCorpora: evaluationCorpora)
    }

    private static func validateLiveTokenization(
        _ prompts: [KVTunerCalibrationPromptIdentity],
        phase: String,
        tokenizePrompt: (String) throws -> [Int]
    ) throws {
        for (position, prompt) in prompts.enumerated() {
            guard let text = String(
                data: prompt.promptUTF8, encoding: .utf8)
            else {
                throw KVTunerRuntimeSelectionError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
            let liveTokenIDs: [Int]
            do {
                liveTokenIDs = try tokenizePrompt(text)
            } catch {
                throw KVTunerRuntimeSelectionError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
            guard liveTokenIDs == prompt.tokenIDs else {
                throw KVTunerRuntimeSelectionError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
        }
    }
}
