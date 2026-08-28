// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w where weightFilter(key) {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
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

    eval(model)
}
