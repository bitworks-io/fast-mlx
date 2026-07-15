import Foundation

public enum KVTunerScheduleError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidIdentifier(String)
    case invalidDigest(String)
    case invalidProvenance
    case matrixIDMismatch
    case cellIDMismatch
    case invalidCellDescriptor(String)
    case cellGroupSizeMismatch(cell: Int, schedule: Int)
    case cellNominalAverageBitsMismatch(cell: Double, schedule: Double)
    case modelConfigHashMismatch
    case checkpointManifestHashMismatch
    case unsupportedGroupSize(Int)
    case invalidLayerCount
    case nonCanonicalLayerOrder(position: Int, declaredLayer: Int)
    case unsupportedPrecision(layer: Int, keyBits: Int, valueBits: Int)
    case nominalAverageMismatch(declared: Double, computed: Double)
    case evaluationCorpusLeaksCalibration
}

public struct KVLayerPrecision: Codable, Equatable, Sendable {
    public var layer: Int
    public var keyBits: Int
    public var valueBits: Int

    public init(layer: Int, keyBits: Int, valueBits: Int) {
        self.layer = layer
        self.keyBits = keyBits
        self.valueBits = valueBits
    }
}

/// Frozen, portable output of the KVTuner control plane. Search-database state is not a runtime
/// contract; this versioned JSON artifact carries one complete per-layer policy plus enough
/// identity to reject a schedule selected for another model, matrix cell, or calibration set.
public struct KVTunerSchedule: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var matrixID: String
    public var cellID: String
    public var modelConfigHash: String
    public var checkpointManifestHash: String
    public var groupSize: Int
    public var calibrationCorpusID: String
    public var calibrationCorpusHash: String
    public var calibrationEntryHashes: [String]
    public var seed: UInt64
    public var objective: String
    public var nominalAverageBits: Double
    public var sourceSensitivityArtifactSHA256: String
    public var layers: [KVLayerPrecision]

    public init(
        schemaVersion: Int,
        matrixID: String,
        cellID: String,
        modelConfigHash: String,
        checkpointManifestHash: String,
        groupSize: Int,
        calibrationCorpusID: String,
        calibrationCorpusHash: String,
        calibrationEntryHashes: [String],
        seed: UInt64,
        objective: String,
        nominalAverageBits: Double,
        sourceSensitivityArtifactSHA256: String,
        layers: [KVLayerPrecision]
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.cellID = cellID
        self.modelConfigHash = modelConfigHash
        self.checkpointManifestHash = checkpointManifestHash
        self.groupSize = groupSize
        self.calibrationCorpusID = calibrationCorpusID
        self.calibrationCorpusHash = calibrationCorpusHash
        self.calibrationEntryHashes = calibrationEntryHashes
        self.seed = seed
        self.objective = objective
        self.nominalAverageBits = nominalAverageBits
        self.sourceSensitivityArtifactSHA256 = sourceSensitivityArtifactSHA256
        self.layers = layers
    }

    /// KVTuner's paper/repository budget convention: the nominal mean of K and V bit widths
    /// across all layers. It intentionally excludes runtime scale/bias/alignment bytes; the
    /// format accountant supplies those actual bytes separately.
    ///
    /// The calculation divides before accumulating, avoiding an integer `K + V` overflow on a
    /// decoded but not-yet-validated artifact. Validation admits only the closed declared pairs;
    /// runtime consumption of the schedule is a separate integration gate.
    public var computedNominalAverageBits: Double {
        guard !layers.isEmpty else { return .nan }
        var total = 0.0
        for layer in layers {
            let layerAverage = Double(layer.keyBits) / 2
                + Double(layer.valueBits) / 2
            total += layerAverage
            guard total.isFinite else { return .nan }
        }
        let layerCount = Double(layers.count)
        guard layerCount.isFinite, layerCount > 0 else { return .nan }
        return total / layerCount
    }

    @discardableResult
    public func validated(
        expectedLayerCount: Int,
        expectedMatrixID: String,
        expectedCellID: String,
        expectedModelConfigHash: String,
        expectedCheckpointManifestHash: String
    ) throws -> KVTunerSchedule {
        guard schemaVersion == 2 else {
            throw KVTunerScheduleError.unsupportedSchema(schemaVersion)
        }
        guard expectedLayerCount > 0 else {
            throw KVTunerScheduleError.invalidLayerCount
        }

        for identifier in [
            matrixID,
            cellID,
            expectedMatrixID,
            expectedCellID,
            objective,
        ] {
            guard Self.isIdentifier(identifier) else {
                throw KVTunerScheduleError.invalidIdentifier(identifier)
            }
        }
        for digest in [
            modelConfigHash,
            checkpointManifestHash,
            expectedModelConfigHash,
            expectedCheckpointManifestHash,
        ] {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerScheduleError.invalidDigest(digest)
            }
        }
        guard Self.isLowercaseHex(
            sourceSensitivityArtifactSHA256, length: 64)
        else {
            throw KVTunerScheduleError.invalidDigest(
                sourceSensitivityArtifactSHA256)
        }
        guard matrixID == expectedMatrixID else {
            throw KVTunerScheduleError.matrixIDMismatch
        }
        guard cellID == expectedCellID else {
            throw KVTunerScheduleError.cellIDMismatch
        }
        guard modelConfigHash == expectedModelConfigHash else {
            throw KVTunerScheduleError.modelConfigHashMismatch
        }
        guard checkpointManifestHash == expectedCheckpointManifestHash else {
            throw KVTunerScheduleError.checkpointManifestHashMismatch
        }
        guard let cellDescriptor = Self.parseCellDescriptor(cellID) else {
            throw KVTunerScheduleError.invalidCellDescriptor(cellID)
        }
        guard Self.supportedGroupSizes.contains(groupSize) else {
            throw KVTunerScheduleError.unsupportedGroupSize(groupSize)
        }
        guard cellDescriptor.groupSize == groupSize else {
            throw KVTunerScheduleError.cellGroupSizeMismatch(
                cell: cellDescriptor.groupSize, schedule: groupSize)
        }
        try validateCalibrationIdentity()

        guard layers.count == expectedLayerCount else {
            throw KVTunerScheduleError.invalidLayerCount
        }
        for (position, precision) in layers.enumerated() {
            guard precision.layer == position else {
                throw KVTunerScheduleError.nonCanonicalLayerOrder(
                    position: position, declaredLayer: precision.layer)
            }
            guard Self.supportedPrecisionPairs.contains(
                PrecisionPair(
                    keyBits: precision.keyBits,
                    valueBits: precision.valueBits))
            else {
                throw KVTunerScheduleError.unsupportedPrecision(
                    layer: precision.layer,
                    keyBits: precision.keyBits,
                    valueBits: precision.valueBits)
            }
        }

        let computed = computedNominalAverageBits
        guard nominalAverageBits.isFinite, nominalAverageBits > 0,
            computed.isFinite, computed > 0,
            computed == nominalAverageBits
        else {
            throw KVTunerScheduleError.nominalAverageMismatch(
                declared: nominalAverageBits, computed: computed)
        }
        guard cellDescriptor.nominalAverageBits == nominalAverageBits else {
            throw KVTunerScheduleError.cellNominalAverageBitsMismatch(
                cell: cellDescriptor.nominalAverageBits,
                schedule: nominalAverageBits)
        }
        return self
    }

    /// Evaluation must be disjoint from calibration by stable corpus identity, aggregate content,
    /// and every individual entry fingerprint. Renaming a corpus or moving one calibration prompt
    /// into another corpus therefore cannot evade the leakage gate.
    public func validateEvaluationCorpus(
        id: String,
        hash: String,
        entryHashes: [String]
    ) throws {
        try validateCalibrationIdentity()
        guard Self.isIdentifier(id) else {
            throw KVTunerScheduleError.invalidIdentifier(id)
        }
        guard Self.isIdentityDigest(hash) else {
            throw KVTunerScheduleError.invalidDigest(hash)
        }
        guard !entryHashes.isEmpty,
            Set(entryHashes).count == entryHashes.count
        else {
            throw KVTunerScheduleError.invalidProvenance
        }
        for digest in entryHashes {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerScheduleError.invalidDigest(digest)
            }
        }

        let calibrationHashes = Set(calibrationEntryHashes)
        guard id != calibrationCorpusID,
            hash != calibrationCorpusHash,
            calibrationHashes.isDisjoint(with: entryHashes)
        else {
            throw KVTunerScheduleError.evaluationCorpusLeaksCalibration
        }
    }

    private func validateCalibrationIdentity() throws {
        guard Self.isIdentifier(calibrationCorpusID) else {
            throw KVTunerScheduleError.invalidIdentifier(
                calibrationCorpusID)
        }
        guard Self.isIdentityDigest(calibrationCorpusHash) else {
            throw KVTunerScheduleError.invalidDigest(calibrationCorpusHash)
        }
        guard !calibrationEntryHashes.isEmpty,
            Set(calibrationEntryHashes).count == calibrationEntryHashes.count,
            calibrationEntryHashes == calibrationEntryHashes.sorted()
        else {
            throw KVTunerScheduleError.invalidProvenance
        }
        for digest in calibrationEntryHashes {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerScheduleError.invalidDigest(digest)
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
    private static let supportedGroupSizes: Set<Int> = [64, 128]

    private static func parseCellDescriptor(
        _ value: String
    ) -> CellDescriptor? {
        let prefix = "kvtuner-g"
        guard value.hasPrefix(prefix) else { return nil }

        let fields = value.dropFirst(prefix.count).split(
            separator: "-", omittingEmptySubsequences: false)
        guard fields.count == 2, fields[1].first == "b" else { return nil }

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
            value == value.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.contains("\n"), !value.contains("\r")
        else { return false }
        let punctuation = CharacterSet(charactersIn: "-._:/@+")
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0)
                || punctuation.contains($0)
        }
    }

    /// Current harness model/corpus identities are FNV-1a-64 (16 lowercase hex characters).
    /// Accepting 64 lowercase hex characters permits a later SHA-256 identity migration without
    /// weakening the artifact boundary or accepting descriptive placeholders.
    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(_ value: String, length: Int) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}
