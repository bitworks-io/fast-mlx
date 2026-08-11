import Foundation

public enum KVTunerModelConfigPreflightError: Error, Equatable, Sendable {
    case malformedConfig
    case missingLayerCount
    case invalidLayerCount(Int)
}

/// Pure pre-model-load validation for the schedule's required per-layer policy count.
/// Filesystem ownership stays with the CLI so this parser is deterministic and off-box testable.
public enum KVTunerModelConfigPreflight {
    private struct RawConfig: Decodable {
        let numHiddenLayers: Int?

        private enum CodingKeys: String, CodingKey {
            case numHiddenLayers = "num_hidden_layers"
        }
    }

    public static func load(from data: Data) throws -> Int {
        let config: RawConfig
        do {
            config = try JSONDecoder().decode(RawConfig.self, from: data)
        } catch {
            throw KVTunerModelConfigPreflightError.malformedConfig
        }
        guard let layerCount = config.numHiddenLayers else {
            throw KVTunerModelConfigPreflightError.missingLayerCount
        }
        guard layerCount > 0 else {
            throw KVTunerModelConfigPreflightError.invalidLayerCount(
                layerCount)
        }
        return layerCount
    }
}
