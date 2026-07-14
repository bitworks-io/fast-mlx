import Foundation

public enum KVTunerScheduleError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidProvenance
    case modelConfigHashMismatch
    case invalidLayerCount
    case invalidLayerIndex(Int)
    case duplicateLayer(Int)
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
/// contract; this versioned JSON artifact carries the selected per-layer policy plus the inputs
/// needed to detect model mismatch and calibration/evaluation leakage.
public struct KVTunerSchedule: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var modelConfigHash: String
    public var calibrationCorpusID: String
    public var calibrationCorpusHash: String
    public var seed: UInt64
    public var objective: String
    public var nominalAverageBits: Double
    public var layers: [KVLayerPrecision]

    public init(
        schemaVersion: Int, modelConfigHash: String,
        calibrationCorpusID: String, calibrationCorpusHash: String,
        seed: UInt64, objective: String, nominalAverageBits: Double,
        layers: [KVLayerPrecision]
    ) {
        self.schemaVersion = schemaVersion
        self.modelConfigHash = modelConfigHash
        self.calibrationCorpusID = calibrationCorpusID
        self.calibrationCorpusHash = calibrationCorpusHash
        self.seed = seed
        self.objective = objective
        self.nominalAverageBits = nominalAverageBits
        self.layers = layers
    }

    /// KVTuner's paper/repository budget convention: the nominal mean of K and V bit widths
    /// across all layers. It intentionally excludes runtime scale/bias/alignment bytes; the
    /// format accountant supplies those actual bytes separately.
    public var computedNominalAverageBits: Double {
        guard !layers.isEmpty else { return .nan }
        let total = layers.reduce(0.0) { partial, layer in
            partial + Double(layer.keyBits) + Double(layer.valueBits)
        }
        return total / (2 * Double(layers.count))
    }

    @discardableResult
    public func validated(
        expectedLayerCount: Int, expectedModelConfigHash: String
    ) throws -> KVTunerSchedule {
        guard schemaVersion == 1 else {
            throw KVTunerScheduleError.unsupportedSchema(schemaVersion)
        }
        guard expectedLayerCount > 0, !modelConfigHash.isEmpty,
            !calibrationCorpusID.isEmpty, !calibrationCorpusHash.isEmpty,
            !objective.isEmpty, nominalAverageBits.isFinite
        else { throw KVTunerScheduleError.invalidProvenance }
        guard modelConfigHash == expectedModelConfigHash else {
            throw KVTunerScheduleError.modelConfigHashMismatch
        }
        guard layers.count == expectedLayerCount else {
            throw KVTunerScheduleError.invalidLayerCount
        }

        var seen = Set<Int>()
        for precision in layers {
            guard precision.layer >= 0, precision.layer < expectedLayerCount else {
                throw KVTunerScheduleError.invalidLayerIndex(precision.layer)
            }
            guard seen.insert(precision.layer).inserted else {
                throw KVTunerScheduleError.duplicateLayer(precision.layer)
            }
            guard Self.supportedBits.contains(precision.keyBits),
                Self.supportedBits.contains(precision.valueBits)
            else {
                throw KVTunerScheduleError.unsupportedPrecision(
                    layer: precision.layer,
                    keyBits: precision.keyBits,
                    valueBits: precision.valueBits)
            }
        }
        let computed = computedNominalAverageBits
        guard abs(computed - nominalAverageBits) <= 1e-12 else {
            throw KVTunerScheduleError.nominalAverageMismatch(
                declared: nominalAverageBits, computed: computed)
        }
        return self
    }

    /// Evaluation must be disjoint from calibration by both stable identity and content hash.
    /// Renaming the same bytes or changing bytes under the same ID remains a leakage failure.
    public func validateEvaluationCorpus(id: String, hash: String) throws {
        guard !id.isEmpty, !hash.isEmpty else {
            throw KVTunerScheduleError.invalidProvenance
        }
        guard id != calibrationCorpusID, hash != calibrationCorpusHash else {
            throw KVTunerScheduleError.evaluationCorpusLeaksCalibration
        }
    }

    private static let supportedBits: Set<Int> = [2, 4, 8, 16]
}
