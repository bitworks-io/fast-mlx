// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The safetensors files that constitute the model's weights.
///
/// When `model.safetensors.index.json` exists, exactly its mapped shard files are loaded
/// (matching Python mlx-lm). Repositories can ship additional index-external safetensors —
/// separate drafter/vision sidecars, calibration extras — whose foreign keys must never
/// reach the model: a recursive sweep let one factory route fail on them and the silent
/// fallback route then misread the model format (double-shifted norm weights), producing a
/// model that loads and decodes but is quality-destroyed. Without an index, fall back to
/// enumerating `*.safetensors` as before (single-file checkpoints).
func modelWeightFileURLs(modelDirectory: URL) throws -> [URL] {
    struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if let indexData = try? Data(contentsOf: indexURL) {
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: indexData)
        let shards = Set(index.weightMap.values).sorted()
        let urls = shards.map { modelDirectory.appendingPathComponent($0) }
        for url in urls {
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "model.safetensors.index.json maps missing shard \(url.lastPathComponent)")
            }
        }
        return urls
    }

    var urls = [URL]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            urls.append(url)
        }
    }
    return urls
}

public func resolvedWeightQuantizationPath(
    sanitizedPath: String,
    sourcePath: String?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) -> String {
    if perLayerQuantization?
        .perLayerQuantization[sanitizedPath] != nil
    {
        return sanitizedPath
    }
    return sourcePath ?? sanitizedPath
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFilter: (String) -> Bool = { _ in true }
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for url in try modelWeightFileURLs(modelDirectory: modelDirectory) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w where weightFilter(key) {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)
    if let validator = model as? SanitizedWeightQuantizationValidator {
        try validator.validateSanitizedWeightQuantization(
            weights: weights,
            quantization: quantization,
            perLayerQuantization: perLayerQuantization)
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    let sourcePath =
                        (model as? WeightQuantizationPathResolver)?
                        .sourceWeightQuantizationPath(for: path)
                    let quantizationPath = resolvedWeightQuantizationPath(
                        sanitizedPath: path,
                        sourcePath: sourcePath,
                        perLayerQuantization: perLayerQuantization)
                    return perLayerQuantization
                        .quantization(layer: quantizationPath)?
                        .asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    if let languageModel = model as? LanguageModel {
        try languageModel.prepare()
    }

    eval(model)
}
