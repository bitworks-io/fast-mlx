import Foundation

public enum KVTunerScheduleBindingError: Error, Equatable, Sendable {
    case malformedBinding
    case unsupportedBindingSchema(Int)
    case invalidArtifactSHA256
    case invalidSchedule(KVTunerScheduleError)
    case invalidEvaluationCorpus(index: Int, reason: KVTunerScheduleError)
    case requiredEvaluationCorpusMismatch
}

/// Portable evidence identity for one already-authenticated KVTuner runtime selection.
///
/// This is not a second path for constructing a runtime policy. Live inference still obtains its
/// policy exclusively through `KVTunerRuntimeSelection.loadQualified`. The binding copies that immutable
/// result into task/KL evidence, then revalidates its own decoded structure and all externally
/// expected identities before the evidence is accepted.
public struct KVTunerScheduleBinding: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 4

    public let schemaVersion: Int
    public let scheduleSchemaVersion: Int
    /// Exact schedule-byte identity used for runtime policy pairing.
    public let artifactSHA256: String
    /// Exact outer qualification-bundle identity retained for provenance and auditing.
    public let qualificationBundleSHA256: String
    public let matrixID: String
    public let cellID: String
    public let modelConfigHash: String
    public let modelConfigSHA256: String
    public let checkpointManifestHash: String
    public let checkpointContentSHA256: String
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

    public init(selection: KVTunerRuntimeSelection) {
        schemaVersion = Self.currentSchemaVersion
        scheduleSchemaVersion = selection.schemaVersion
        artifactSHA256 = selection.artifactSHA256
        qualificationBundleSHA256 = selection.qualificationBundleSHA256
        matrixID = selection.matrixID
        cellID = selection.cellID
        modelConfigHash = selection.modelConfigHash
        modelConfigSHA256 = selection.modelConfigSHA256
        checkpointManifestHash = selection.checkpointManifestHash
        checkpointContentSHA256 = selection.checkpointContentSHA256
        tokenizerSHA256 = selection.tokenizerSHA256
        groupSize = selection.groupSize
        promptDigestAlgorithm = selection.promptDigestAlgorithm
        calibrationCorpusID = selection.calibrationCorpusID
        calibrationCorpusHash = selection.calibrationCorpusHash
        calibrationEntryDigests = selection.calibrationEntryDigests
        calibrationSourceItemDigests =
            selection.calibrationSourceItemDigests
        seed = selection.seed
        objective = selection.objective
        nominalAverageBits = selection.nominalAverageBits
        sourceSensitivityArtifactSHA256 =
            selection.sourceSensitivityArtifactSHA256
        sourceSearchArtifactSHA256 = selection.sourceSearchArtifactSHA256
        evaluationCorpora = selection.evaluationCorpora
        layers = selection.layers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(
            Int.self, forKey: .schemaVersion)
        guard schemaVersion == Self.currentSchemaVersion else {
            throw KVTunerScheduleBindingError.unsupportedBindingSchema(
                schemaVersion)
        }
        scheduleSchemaVersion = try container.decode(
            Int.self, forKey: .scheduleSchemaVersion)
        guard scheduleSchemaVersion == 4 else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .unsupportedSchema(scheduleSchemaVersion))
        }
        artifactSHA256 = try container.decode(
            String.self, forKey: .artifactSHA256)
        qualificationBundleSHA256 = try container.decode(
            String.self, forKey: .qualificationBundleSHA256)
        matrixID = try container.decode(String.self, forKey: .matrixID)
        cellID = try container.decode(String.self, forKey: .cellID)
        modelConfigHash = try container.decode(
            String.self, forKey: .modelConfigHash)
        modelConfigSHA256 = try container.decode(
            String.self, forKey: .modelConfigSHA256)
        checkpointManifestHash = try container.decode(
            String.self, forKey: .checkpointManifestHash)
        checkpointContentSHA256 = try container.decode(
            String.self, forKey: .checkpointContentSHA256)
        tokenizerSHA256 = try container.decode(
            String.self, forKey: .tokenizerSHA256)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        promptDigestAlgorithm = try container.decode(
            String.self, forKey: .promptDigestAlgorithm)
        calibrationCorpusID = try container.decode(
            String.self, forKey: .calibrationCorpusID)
        calibrationCorpusHash = try container.decode(
            String.self, forKey: .calibrationCorpusHash)
        calibrationEntryDigests = try container.decode(
            [String].self, forKey: .calibrationEntryDigests)
        calibrationSourceItemDigests = try container.decode(
            [String].self, forKey: .calibrationSourceItemDigests)
        seed = try container.decode(UInt64.self, forKey: .seed)
        objective = try container.decode(String.self, forKey: .objective)
        nominalAverageBits = try container.decode(
            Double.self, forKey: .nominalAverageBits)
        sourceSensitivityArtifactSHA256 = try container.decode(
            String.self, forKey: .sourceSensitivityArtifactSHA256)
        sourceSearchArtifactSHA256 = try container.decode(
            String.self, forKey: .sourceSearchArtifactSHA256)
        evaluationCorpora = try container.decode(
            [KVTunerEvaluationCorpusIdentity].self,
            forKey: .evaluationCorpora)
        layers = try container.decode(
            [KVTunerRuntimeLayerPolicy].self, forKey: .layers)
        try validateStructure()
    }

    /// Rebinds decoded evidence to both portable and exact model identities. Runtime/task/KL
    /// consumers still validate the same values against an independently captured loaded-model
    /// admission before touching MLX cache state.
    @discardableResult
    public func validated(
        expectedMatrixID: String,
        expectedCellID: String,
        expectedModelConfigHash: String,
        expectedModelConfigSHA256: String,
        expectedCheckpointManifestHash: String,
        expectedCheckpointContentSHA256: String,
        expectedLayerCount: Int,
        requiredEvaluationCorpus: KVTunerEvaluationCorpusIdentity
    ) throws -> KVTunerScheduleBinding {
        try validateStructure()

        guard Self.isIdentifier(expectedMatrixID) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidIdentifier(expectedMatrixID))
        }
        guard Self.isIdentifier(expectedCellID) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidIdentifier(expectedCellID))
        }
        for digest in [
            expectedModelConfigHash,
            expectedCheckpointManifestHash,
        ] {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidDigest(digest))
            }
        }
        guard expectedLayerCount > 0 else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidLayerCount)
        }
        guard matrixID == expectedMatrixID else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .matrixIDMismatch)
        }
        guard cellID == expectedCellID else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .cellIDMismatch)
        }
        guard modelConfigHash == expectedModelConfigHash else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .modelConfigHashMismatch)
        }
        guard Self.isLowercaseHex(expectedModelConfigSHA256, length: 64),
            modelConfigSHA256 == expectedModelConfigSHA256
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .modelConfigSHA256Mismatch)
        }
        guard checkpointManifestHash == expectedCheckpointManifestHash else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .checkpointManifestHashMismatch)
        }
        guard Self.isLowercaseHex(
                expectedCheckpointContentSHA256, length: 64),
            checkpointContentSHA256 == expectedCheckpointContentSHA256
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .checkpointContentSHA256Mismatch)
        }
        guard layers.count == expectedLayerCount else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidLayerCount)
        }
        guard evaluationCorpora.contains(requiredEvaluationCorpus) else {
            throw KVTunerScheduleBindingError
                .requiredEvaluationCorpusMismatch
        }
        return self
    }

    public static func load(
        from data: Data,
        expectedMatrixID: String,
        expectedCellID: String,
        expectedModelConfigHash: String,
        expectedModelConfigSHA256: String,
        expectedCheckpointManifestHash: String,
        expectedCheckpointContentSHA256: String,
        expectedLayerCount: Int,
        requiredEvaluationCorpus: KVTunerEvaluationCorpusIdentity
    ) throws -> KVTunerScheduleBinding {
        let binding: KVTunerScheduleBinding
        do {
            binding = try JSONDecoder().decode(
                KVTunerScheduleBinding.self, from: data)
        } catch let error as KVTunerScheduleBindingError {
            throw error
        } catch {
            throw KVTunerScheduleBindingError.malformedBinding
        }
        return try binding.validated(
            expectedMatrixID: expectedMatrixID,
            expectedCellID: expectedCellID,
            expectedModelConfigHash: expectedModelConfigHash,
            expectedModelConfigSHA256: expectedModelConfigSHA256,
            expectedCheckpointManifestHash:
                expectedCheckpointManifestHash,
            expectedCheckpointContentSHA256:
                expectedCheckpointContentSHA256,
            expectedLayerCount: expectedLayerCount,
            requiredEvaluationCorpus: requiredEvaluationCorpus)
    }

    /// Schedule pairing deliberately ignores command-specific evaluation corpus lists. Every
    /// field that can change runtime cache behavior or select a different artifact remains bound.
    public func sameSchedule(as other: KVTunerScheduleBinding) -> Bool {
        schemaVersion == other.schemaVersion
            && scheduleSchemaVersion == other.scheduleSchemaVersion
            && artifactSHA256 == other.artifactSHA256
            && matrixID == other.matrixID
            && cellID == other.cellID
            && modelConfigHash == other.modelConfigHash
            && modelConfigSHA256 == other.modelConfigSHA256
            && checkpointManifestHash == other.checkpointManifestHash
            && checkpointContentSHA256
                == other.checkpointContentSHA256
            && tokenizerSHA256 == other.tokenizerSHA256
            && groupSize == other.groupSize
            && promptDigestAlgorithm == other.promptDigestAlgorithm
            && calibrationCorpusID == other.calibrationCorpusID
            && calibrationCorpusHash == other.calibrationCorpusHash
            && calibrationEntryDigests == other.calibrationEntryDigests
            && calibrationSourceItemDigests
                == other.calibrationSourceItemDigests
            && seed == other.seed
            && objective == other.objective
            && nominalAverageBits == other.nominalAverageBits
            && sourceSensitivityArtifactSHA256
                == other.sourceSensitivityArtifactSHA256
            && sourceSearchArtifactSHA256
                == other.sourceSearchArtifactSHA256
            && layers == other.layers
    }

    private func validateStructure() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw KVTunerScheduleBindingError.unsupportedBindingSchema(
                schemaVersion)
        }
        guard scheduleSchemaVersion == 4 else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .unsupportedSchema(scheduleSchemaVersion))
        }
        guard Self.isLowercaseHex(artifactSHA256, length: 64) else {
            throw KVTunerScheduleBindingError.invalidArtifactSHA256
        }
        guard Self.isLowercaseHex(
            qualificationBundleSHA256, length: 64)
        else {
            throw KVTunerScheduleBindingError.invalidArtifactSHA256
        }
        for identifier in [matrixID, cellID, objective] {
            guard Self.isIdentifier(identifier) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidIdentifier(identifier))
            }
        }
        for digest in [modelConfigHash, checkpointManifestHash] {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidDigest(digest))
            }
        }
        for digest in [
            modelConfigSHA256,
            checkpointContentSHA256,
            sourceSensitivityArtifactSHA256,
            sourceSearchArtifactSHA256,
            tokenizerSHA256,
        ] {
            guard Self.isLowercaseHex(digest, length: 64) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidDigest(digest))
            }
        }
        guard [64, 128].contains(groupSize) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .unsupportedGroupSize(groupSize))
        }
        guard let descriptor = Self.parseCellDescriptor(cellID) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidCellDescriptor(cellID))
        }
        guard descriptor.groupSize == groupSize else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .cellGroupSizeMismatch(
                    cell: descriptor.groupSize, schedule: groupSize))
        }

        guard promptDigestAlgorithm == KVTunerPromptDigest.algorithm else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidProvenance)
        }
        guard Self.isIdentifier(calibrationCorpusID) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidIdentifier(calibrationCorpusID))
        }
        guard Self.isIdentityDigest(calibrationCorpusHash) else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidDigest(calibrationCorpusHash))
        }
        guard !calibrationEntryDigests.isEmpty,
            Set(calibrationEntryDigests).count
                == calibrationEntryDigests.count,
            calibrationEntryDigests == calibrationEntryDigests.sorted()
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidProvenance)
        }
        for digest in calibrationEntryDigests {
            guard KVTunerPromptDigest.isCanonical(digest) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidDigest(digest))
            }
        }
        guard calibrationSourceItemDigests.count == 200,
            Set(calibrationSourceItemDigests).count == 200,
            calibrationSourceItemDigests
                == calibrationSourceItemDigests.sorted()
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidProvenance)
        }
        for digest in calibrationSourceItemDigests {
            guard Self.isLowercaseHex(digest, length: 64) else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .invalidDigest(digest))
            }
        }

        guard !layers.isEmpty else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidLayerCount)
        }
        var totalBits = 0.0
        for (position, policy) in layers.enumerated() {
            guard policy.layer == position else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .nonCanonicalLayerOrder(
                        position: position,
                        declaredLayer: policy.layer))
            }
            guard Self.supportedPrecisionPairs.contains(
                PrecisionPair(
                    keyBits: policy.keyBits,
                    valueBits: policy.valueBits))
            else {
                throw KVTunerScheduleBindingError.invalidSchedule(
                    .unsupportedPrecision(
                        layer: policy.layer,
                        keyBits: policy.keyBits,
                        valueBits: policy.valueBits))
            }
            totalBits += Double(policy.keyBits) / 2
                + Double(policy.valueBits) / 2
        }
        let computedNominalAverageBits = totalBits / Double(layers.count)
        guard nominalAverageBits.isFinite, nominalAverageBits > 0,
            nominalAverageBits == computedNominalAverageBits,
            nominalAverageBits == descriptor.nominalAverageBits
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .cellNominalAverageBitsMismatch(
                    cell: descriptor.nominalAverageBits,
                    schedule: nominalAverageBits))
        }
        guard seed == KVTunerScheduleSearch.requiredFewShotSeed else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidSearchProtocol("seed"))
        }
        guard objective
            == "maximize-gsm8k-accuracy-at-b\(nominalAverageBits)"
        else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .invalidSearchProtocol("objective"))
        }

        guard !evaluationCorpora.isEmpty else {
            throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                index: 0, reason: .invalidProvenance)
        }
        var seenCorpora = Set<KVTunerEvaluationCorpusIdentity>()
        for (index, corpus) in evaluationCorpora.enumerated() {
            guard seenCorpora.insert(corpus).inserted else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index, reason: .invalidProvenance)
            }
            guard Self.isIdentifier(corpus.id) else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index,
                    reason: .invalidIdentifier(corpus.id))
            }
            guard Self.isIdentityDigest(corpus.aggregateDigest) else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index,
                    reason: .invalidDigest(corpus.aggregateDigest))
            }
            guard !corpus.canonicalEntryDigests.isEmpty,
                Set(corpus.canonicalEntryDigests).count
                    == corpus.canonicalEntryDigests.count,
                corpus.canonicalEntryDigests
                    == corpus.canonicalEntryDigests.sorted()
            else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index, reason: .invalidProvenance)
            }
            for digest in corpus.canonicalEntryDigests {
                guard KVTunerPromptDigest.isCanonical(digest) else {
                    throw KVTunerScheduleBindingError
                        .invalidEvaluationCorpus(
                            index: index,
                            reason: .invalidDigest(digest))
                }
            }
            let expectedSourceItems: [String]
            if corpus.sourceProvenance == .firstPartyAuditedNoGSM8K {
                expectedSourceItems = KVTunerEvaluationSourceProvenance
                    .auditedSourceItemDigests(
                        entryDigests: corpus.canonicalEntryDigests)
                guard corpus.canonicalSourceItemDigests
                    == expectedSourceItems
                else {
                    throw KVTunerScheduleBindingError
                        .invalidEvaluationCorpus(
                            index: index, reason: .invalidProvenance)
                }
            } else {
                expectedSourceItems = corpus.canonicalSourceItemDigests
            }
            guard !expectedSourceItems.isEmpty,
                Set(expectedSourceItems).count
                    == corpus.canonicalSourceItemDigests.count,
                corpus.canonicalSourceItemDigests
                    == corpus.canonicalSourceItemDigests.sorted()
            else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index, reason: .invalidProvenance)
            }
            for digest in corpus.canonicalSourceItemDigests {
                guard Self.isLowercaseHex(digest, length: 64) else {
                    throw KVTunerScheduleBindingError
                        .invalidEvaluationCorpus(
                            index: index,
                            reason: .invalidDigest(digest))
                }
            }
            guard corpus.id != calibrationCorpusID,
                corpus.aggregateDigest != calibrationCorpusHash,
                Set(calibrationEntryDigests).isDisjoint(
                    with: corpus.canonicalEntryDigests),
                Set(calibrationSourceItemDigests).isDisjoint(
                    with: corpus.canonicalSourceItemDigests)
            else {
                throw KVTunerScheduleBindingError.invalidEvaluationCorpus(
                    index: index,
                    reason: .evaluationCorpusLeaksCalibration)
            }
        }
    }

    private struct PrecisionPair: Hashable {
        let keyBits: Int
        let valueBits: Int
    }

    private struct CellDescriptor {
        let groupSize: Int
        let nominalAverageBits: Double
    }

    private static let supportedPrecisionPairs: Set<PrecisionPair> = [
        PrecisionPair(keyBits: 8, valueBits: 4),
        PrecisionPair(keyBits: 8, valueBits: 2),
        PrecisionPair(keyBits: 4, valueBits: 2),
    ]

    private static func parseCellDescriptor(
        _ value: String
    ) -> CellDescriptor? {
        let prefix = "kvtuner-g"
        guard value.hasPrefix(prefix) else { return nil }
        let fields = value.dropFirst(prefix.count).split(
            separator: "-", omittingEmptySubsequences: false)
        guard fields.count == 2, fields[1].first == "b" else {
            return nil
        }

        let groupText = String(fields[0])
        let bitsText = String(fields[1].dropFirst())
        guard let groupSize = Int(groupText), groupSize > 0,
            String(groupSize) == groupText,
            let nominalAverageBits = Double(bitsText),
            nominalAverageBits.isFinite, nominalAverageBits > 0,
            String(nominalAverageBits) == bitsText
        else { return nil }
        return CellDescriptor(
            groupSize: groupSize,
            nominalAverageBits: nominalAverageBits)
    }

    private static func isIdentifier(_ value: String) -> Bool {
        guard !value.isEmpty, value != "unknown",
            value == value.trimmingCharacters(
                in: .whitespacesAndNewlines),
            !value.contains("\n"), !value.contains("\r")
        else { return false }
        let punctuation = CharacterSet(charactersIn: "-._:/@+")
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0)
                || punctuation.contains($0)
        }
    }

    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(
        _ value: String,
        length: Int
    ) -> Bool {
        guard value.count == length else { return false }
        let lowercaseHex = CharacterSet(charactersIn: "0123456789abcdef")
        return value.unicodeScalars.allSatisfy(lowercaseHex.contains)
    }
}
