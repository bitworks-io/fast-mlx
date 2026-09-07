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
///
/// The index decoding itself lives in ``SafetensorsWeightIndex`` so callers outside this
/// module can obtain the artifact's declared key set and key->shard mapping without reading
/// a tensor.
func modelWeightFileURLs(modelDirectory: URL) throws -> [URL] {
    if let index = try SafetensorsWeightIndex.load(modelDirectory: modelDirectory) {
        return try index.shardURLs(in: modelDirectory)
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
    try applySanitizedWeights(
        model: model,
        sanitizedWeights: weights,
        quantization: quantization,
        perLayerQuantization: perLayerQuantization)
}

/// Validates, quantizes, and binds already-sanitized weights onto `model`, then prepares and
/// evaluates it.
///
/// This is everything ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFilter:)``
/// does from quantization validation onward, extracted so there is exactly ONE implementation
/// of the validate/quantize/bind sequence.
///
/// A model family whose drafter weights live inside its TARGET checkpoint needs this same
/// sequence but cannot route through `loadWeights` itself, for two reasons. It must narrow to
/// the specific shards holding those weights rather than reading the artifact's whole shard
/// set, and it must sanitize through a THROWING family sanitizer rather than the non-throwing
/// ``BaseLanguageModel/sanitize(weights:metadata:)`` entry point `loadWeights` calls above --
/// a drafter's implementation of that protocol method may deliberately swallow sanitize
/// failures and return `[:]`, which would turn a precise diagnostic into a generic "unset
/// parameters" failure here. Such a loader sanitizes on its own terms and then calls this
/// function; without the extraction it would have to duplicate the quantize/bind logic below,
/// and the two copies could drift apart.
public func applySanitizedWeights(
    model: BaseLanguageModel,
    sanitizedWeights: [String: MLXArray],
    quantization: BaseConfiguration.Quantization?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) throws {
    if let validator = model as? SanitizedWeightQuantizationValidator {
        try validator.validateSanitizedWeightQuantization(
            weights: sanitizedWeights,
            quantization: quantization,
            perLayerQuantization: perLayerQuantization)
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if sanitizedWeights["\(path).scales"] != nil {
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
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.update(parameters: parameters, verify: [.all])

    if let languageModel = model as? LanguageModel {
        try languageModel.prepare()
    }

    eval(model)
}
