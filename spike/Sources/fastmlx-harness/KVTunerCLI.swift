import Foundation
import HarnessCore
import Tokenizers

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
    case missingEOSToken
    case outputPathCollision(String)

    var description: String {
        switch self {
        case .missingSchedule:
            return "KVTuner requires --kvtuner-schedule <QUALIFICATION-BUNDLE.json>"
        case .unexpectedSchedule(let tier):
            return "--kvtuner-schedule is valid only for a kvtuner-* tier, not \(tier)"
        case .tierCellMismatch(let tier, let cell):
            return "KVTuner requires --kv-quant and --cell-id to match exactly; got \(tier) and \(cell)"
        case .unreadableModelConfig(let path):
            return "unable to read the exact model config at \(path)/config.json"
        case .modelConfigIdentityMismatch:
            return "model config bytes do not match the preflight model identity"
        case .missingEOSToken:
            return "KVTuner qualification requires a tokenizer EOS token"
        case .outputPathCollision(let path):
            return "KVTuner qualification bundle must not alias an evidence output: \(path)"
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

/// Re-derives one exact schedule from its qualification bundle before MLX model loading. Model
/// geometry and tokenizer identity come from the runtime model files, never the bundled schedule.
func prepareKVTunerRun(
    schedulePath: String,
    modelPath: String,
    matrixID: String,
    cellID: String,
    modelIdentity: KVModelEvidenceIdentity,
    evaluationCorpus: KVTunerEvaluationCorpusIdentity
) async throws -> PreparedKVTunerRun {
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
    let tokenizerSHA256 = try ProvenanceCLI.tokenizerManifestSHA256(
        at: modelPath)
    // Load only the CPU tokenizer before the heavyweight MLX model. This lets qualification
    // replay every authenticated prompt against the live tokenizer while retaining the
    // fail-before-model-load behavior for a bad schedule.
    let tokenizer = try await AutoTokenizer.from(
        modelFolder: URL(fileURLWithPath: modelPath))
    guard let eosTokenID = tokenizer.eosToken.flatMap({
        tokenizer.convertTokenToId($0)
    }) else {
        throw KVTunerCLIError.missingEOSToken
    }
    let selection = try KVTunerRuntimeSelection.loadQualified(
        artifactData: scheduleData,
        exactModelConfigData: configData,
        expectedMatrixID: matrixID,
        expectedCellID: cellID,
        expectedCheckpointManifestHash:
            modelIdentity.checkpointManifestHash,
        expectedTokenizerSHA256: tokenizerSHA256,
        expectedEOSTokenID: eosTokenID,
        tokenizePrompt: {
            tokenizer.encode(text: $0, addSpecialTokens: true)
        },
        decodeTokenIDs: {
            tokenizer.decode(tokens: $0)
        },
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
