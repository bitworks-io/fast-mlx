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
/// policy exclusively through `KVTunerRuntimeSelection.load`. The binding copies that immutable
/// result into task/KL evidence, then revalidates its own decoded structure and all externally
/// expected identities before the evidence is accepted.
public struct KVTunerScheduleBinding: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 2

    public let schemaVersion: Int
    public let scheduleSchemaVersion: Int
    public let artifactSHA256: String
    public let matrixID: String
    public let cellID: String
    public let modelConfigHash: String
    public let checkpointManifestHash: String
    public let groupSize: Int
    public let promptDigestAlgorithm: String
    public let calibrationCorpusID: String
    public let calibrationCorpusHash: String
    public let calibrationEntryDigests: [String]
    public let evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    public let layers: [KVTunerRuntimeLayerPolicy]

    public init(selection: KVTunerRuntimeSelection) {
        schemaVersion = Self.currentSchemaVersion
        scheduleSchemaVersion = selection.schemaVersion
        artifactSHA256 = selection.artifactSHA256
        matrixID = selection.matrixID
        cellID = selection.cellID
        modelConfigHash = selection.modelConfigHash
        checkpointManifestHash = selection.checkpointManifestHash
        groupSize = selection.groupSize
        promptDigestAlgorithm = selection.promptDigestAlgorithm
        calibrationCorpusID = selection.calibrationCorpusID
        calibrationCorpusHash = selection.calibrationCorpusHash
        calibrationEntryDigests = selection.calibrationEntryDigests
        evaluationCorpora = selection.evaluationCorpora
        layers = selection.layers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(
            Int.self, forKey: .schemaVersion)
        scheduleSchemaVersion = try container.decode(
            Int.self, forKey: .scheduleSchemaVersion)
        artifactSHA256 = try container.decode(
            String.self, forKey: .artifactSHA256)
        matrixID = try container.decode(String.self, forKey: .matrixID)
        cellID = try container.decode(String.self, forKey: .cellID)
        modelConfigHash = try container.decode(
            String.self, forKey: .modelConfigHash)
        checkpointManifestHash = try container.decode(
            String.self, forKey: .checkpointManifestHash)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        promptDigestAlgorithm = try container.decode(
            String.self, forKey: .promptDigestAlgorithm)
        calibrationCorpusID = try container.decode(
            String.self, forKey: .calibrationCorpusID)
        calibrationCorpusHash = try container.decode(
            String.self, forKey: .calibrationCorpusHash)
        calibrationEntryDigests = try container.decode(
            [String].self, forKey: .calibrationEntryDigests)
        evaluationCorpora = try container.decode(
            [KVTunerEvaluationCorpusIdentity].self,
            forKey: .evaluationCorpora)
        layers = try container.decode(
            [KVTunerRuntimeLayerPolicy].self, forKey: .layers)
        try validateStructure()
    }

    /// Rebinds decoded evidence to the exact runtime/corpus identities expected by its consumer.
    /// Matching only a tier label is insufficient: every model and schedule field must match, and
    /// the exact aggregate plus per-prompt corpus identity must be present.
    @discardableResult
    public func validated(
        expectedMatrixID: String,
        expectedCellID: String,
        expectedModelConfigHash: String,
        expectedCheckpointManifestHash: String,
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
        guard checkpointManifestHash == expectedCheckpointManifestHash else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .checkpointManifestHashMismatch)
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
        expectedCheckpointManifestHash: String,
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
            expectedCheckpointManifestHash:
                expectedCheckpointManifestHash,
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
            && checkpointManifestHash == other.checkpointManifestHash
            && groupSize == other.groupSize
            && promptDigestAlgorithm == other.promptDigestAlgorithm
            && calibrationCorpusID == other.calibrationCorpusID
            && calibrationCorpusHash == other.calibrationCorpusHash
            && calibrationEntryDigests == other.calibrationEntryDigests
            && layers == other.layers
    }

    private func validateStructure() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw KVTunerScheduleBindingError.unsupportedBindingSchema(
                schemaVersion)
        }
        guard scheduleSchemaVersion == 2 else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .unsupportedSchema(scheduleSchemaVersion))
        }
        guard Self.isLowercaseHex(artifactSHA256, length: 64) else {
            throw KVTunerScheduleBindingError.invalidArtifactSHA256
        }
        for identifier in [matrixID, cellID] {
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
        let nominalAverageBits = totalBits / Double(layers.count)
        guard nominalAverageBits == descriptor.nominalAverageBits else {
            throw KVTunerScheduleBindingError.invalidSchedule(
                .cellNominalAverageBitsMismatch(
                    cell: descriptor.nominalAverageBits,
                    schedule: nominalAverageBits))
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
            guard corpus.id != calibrationCorpusID,
                corpus.aggregateDigest != calibrationCorpusHash,
                Set(calibrationEntryDigests).isDisjoint(
                    with: corpus.canonicalEntryDigests)
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
