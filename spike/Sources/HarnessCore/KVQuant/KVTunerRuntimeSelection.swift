import Foundation

public enum KVTunerRuntimeSelectionError: Error, Equatable, Sendable {
    case malformedArtifact
    case missingEvaluationCorpus
    case invalidSchedule(KVTunerScheduleError)
    case invalidEvaluationCorpus(index: Int, reason: KVTunerScheduleError)
    case unexpectedValidationFailure
}

/// Stable evaluation identity supplied at the same trust boundary as the schedule artifact.
/// Validation remains centralized in `KVTunerSchedule.validateEvaluationCorpus`.
public struct KVTunerEvaluationCorpusIdentity: Hashable, Sendable {
    public let id: String
    public let aggregateDigest: String
    public let canonicalEntryDigests: [String]

    public init(
        id: String,
        aggregateDigest: String,
        canonicalEntryDigests: [String]
    ) {
        self.id = id
        self.aggregateDigest = aggregateDigest
        self.canonicalEntryDigests = canonicalEntryDigests
    }
}

/// Runtime-only copy of one validated layer decision. Unlike `KVLayerPrecision`, these fields
/// cannot be changed after the authenticated selection is constructed.
public struct KVTunerRuntimeLayerPolicy: Hashable, Sendable {
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
    public let artifactSHA256: String
    public let schemaVersion: Int
    public let matrixID: String
    public let cellID: String
    public let modelConfigHash: String
    public let checkpointManifestHash: String
    public let groupSize: Int
    public let evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    public let layers: [KVTunerRuntimeLayerPolicy]

    private init(
        artifactSHA256: String,
        schedule: KVTunerSchedule,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]
    ) {
        self.artifactSHA256 = artifactSHA256
        schemaVersion = schedule.schemaVersion
        matrixID = schedule.matrixID
        cellID = schedule.cellID
        modelConfigHash = schedule.modelConfigHash
        checkpointManifestHash = schedule.checkpointManifestHash
        groupSize = schedule.groupSize
        self.evaluationCorpora = evaluationCorpora
        layers = schedule.layers.map {
            KVTunerRuntimeLayerPolicy(
                layer: $0.layer,
                keyBits: $0.keyBits,
                valueBits: $0.valueBits)
        }
    }

    public static func load(
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
                    entryHashes: corpus.canonicalEntryDigests)
            } catch let reason as KVTunerScheduleError {
                throw KVTunerRuntimeSelectionError.invalidEvaluationCorpus(
                    index: index,
                    reason: reason)
            } catch {
                throw KVTunerRuntimeSelectionError.unexpectedValidationFailure
            }
        }

        return KVTunerRuntimeSelection(
            artifactSHA256: sha256Hex(artifactData),
            schedule: validatedSchedule,
            evaluationCorpora: evaluationCorpora)
    }
}
