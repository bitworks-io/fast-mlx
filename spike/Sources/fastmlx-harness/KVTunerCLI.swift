import Foundation
import HarnessCore

struct PreparedKVTunerRun {
    let selection: KVTunerRuntimeSelection
    let binding: KVTunerScheduleBinding
}

enum KVTunerCLIError: Error, CustomStringConvertible {
    case missingSchedule
    case unexpectedSchedule(tier: String)
    case tierCellMismatch(tier: String, cell: String)
    case unreadableModelConfig(String)
    case modelConfigIdentityMismatch
    case outputPathCollision(String)

    var description: String {
        switch self {
        case .missingSchedule:
            return "KVTuner requires --kvtuner-schedule <JSON>"
        case .unexpectedSchedule(let tier):
            return "--kvtuner-schedule is valid only for a kvtuner-* tier, not \(tier)"
        case .tierCellMismatch(let tier, let cell):
            return "KVTuner requires --kv-quant and --cell-id to match exactly; got \(tier) and \(cell)"
        case .unreadableModelConfig(let path):
            return "unable to read the exact model config at \(path)/config.json"
        case .modelConfigIdentityMismatch:
            return "model config bytes do not match the preflight model identity"
        case .outputPathCollision(let path):
            return "KVTuner schedule input must not alias an evidence output: \(path)"
        }
    }
}

func isKVTunerTier(_ tier: String) -> Bool {
    tier.hasPrefix("kvtuner-")
}

func validateKVTunerScheduleFlag(
    tier: String,
    cellID: String,
    schedulePath: String
) throws {
    if isKVTunerTier(tier) {
        guard !schedulePath.isEmpty else {
            throw KVTunerCLIError.missingSchedule
        }
        guard tier == cellID else {
            throw KVTunerCLIError.tierCellMismatch(tier: tier, cell: cellID)
        }
    } else if !schedulePath.isEmpty {
        throw KVTunerCLIError.unexpectedSchedule(tier: tier)
    }
}

/// Authenticates one exact schedule against model and evaluation identities before MLX model
/// loading. The expected layer count comes from the model's own config bytes, never the schedule.
func prepareKVTunerRun(
    schedulePath: String,
    modelPath: String,
    matrixID: String,
    cellID: String,
    modelIdentity: KVModelEvidenceIdentity,
    evaluationCorpus: KVTunerEvaluationCorpusIdentity
) throws -> PreparedKVTunerRun {
    let configURL = URL(fileURLWithPath: modelPath)
        .appendingPathComponent("config.json")
    guard let configData = try? Data(contentsOf: configURL) else {
        throw KVTunerCLIError.unreadableModelConfig(modelPath)
    }
    guard fnv1a64(configData) == modelIdentity.configHash else {
        throw KVTunerCLIError.modelConfigIdentityMismatch
    }
    let layerCount = try KVTunerModelConfigPreflight.load(from: configData)
    let scheduleData = try Data(
        contentsOf: URL(fileURLWithPath: schedulePath))
    let selection = try KVTunerRuntimeSelection.load(
        artifactData: scheduleData,
        expectedLayerCount: layerCount,
        expectedMatrixID: matrixID,
        expectedCellID: cellID,
        expectedModelConfigHash: modelIdentity.configHash,
        expectedCheckpointManifestHash:
            modelIdentity.checkpointManifestHash,
        evaluationCorpora: [evaluationCorpus])
    let binding = try KVTunerScheduleBinding(selection: selection).validated(
        expectedMatrixID: matrixID,
        expectedCellID: cellID,
        expectedModelConfigHash: modelIdentity.configHash,
        expectedCheckpointManifestHash:
            modelIdentity.checkpointManifestHash,
        expectedLayerCount: layerCount,
        requiredEvaluationCorpus: evaluationCorpus)
    return PreparedKVTunerRun(selection: selection, binding: binding)
}
