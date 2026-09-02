import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

// Load-path parity for affine per-layer (mixed-bit) Qwen3.5 VLM checkpoints.
//
// Reproduction target (2026-09-02): the mlx-community calibrated 4-bit
// Qwen3.8-27B checkpoints (affine, group size 64, nested per-tensor overrides,
// quantized embed_tokens + lm_head with quantization biases) load and decode
// fluently through the Swift engine but produce logits unrelated to the same
// artifact under Python mlx-lm (teacher-forced top-1 1/328), while the
// uniform-mxfp8 target loads exactly. These tests round-trip a tiny model in
// the candidates' exact config idiom through the REAL factory load seam
// (BaseConfiguration -> loadWeights(modelDirectory:model:perLayerQuantization:))
// and require the loaded model's logits to equal the in-memory source model's
// logits bit for bit. The uniform-4-bit control brackets the failing factor.

private func qwen35AffineParityConfigJSON(quantization: String) -> String {
    // Every quantized Linear/Embedding input dimension is a multiple of 64 so
    // the affine group size of the real candidates applies unchanged.
    // linear_key_head_dim >= 32 keeps the GDN Metal kernel off the unsupported
    // zero-work specialization (see Qwen35MTPTests fixture note).
    """
    {
      "model_type": "qwen3_5",
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
        "attention_bias": false,
        "full_attention_interval": 2,
        "mtp_num_hidden_layers": 0,
        "mtp_use_dedicated_embeddings": false,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "rope_parameters": {
          "type": "default",
          "rope_theta": 100000.0,
          "partial_rotary_factor": 0.25
        }
      },
      "vision_config": {
        "model_type": "qwen3_5_vit",
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "out_hidden_size": 16,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 16
      },
      "quantization": \(quantization)
    }
    """
}

private let qwen35AffineUniformQuantizationJSON = """
    {"group_size": 64, "bits": 4, "mode": "affine"}
    """

// Mirrors the OptiQ idiom: top-level affine 4/64 default, nested per-tensor
// 8-bit overrides including the quantized embedding and untied lm_head, and a
// mixed (4,4,8,8) qkv/z/b/a pattern on a GDN layer.
private let qwen35AffineMixedQuantizationJSON = """
    {
      "group_size": 64, "bits": 4, "mode": "affine",
      "language_model.model.embed_tokens": {"group_size": 64, "bits": 8},
      "language_model.lm_head": {"group_size": 64, "bits": 8},
      "language_model.model.layers.0.linear_attn.in_proj_b": {"group_size": 64, "bits": 8},
      "language_model.model.layers.0.linear_attn.in_proj_a": {"group_size": 64, "bits": 8},
      "language_model.model.layers.1.self_attn.q_proj": {"group_size": 64, "bits": 8},
      "language_model.model.layers.1.mlp.down_proj": {"group_size": 64, "bits": 8}
    }
    """

private func qwen35AffineParityOverrides(from quantizationJSON: String) throws -> [String: (
    groupSize: Int, bits: Int
)] {
    let raw =
        try JSONSerialization.jsonObject(with: Data(quantizationJSON.utf8)) as? [String: Any] ?? [:]
    var overrides = [String: (groupSize: Int, bits: Int)]()
    for (key, value) in raw {
        guard let nested = value as? [String: Any],
            let groupSize = nested["group_size"] as? Int,
            let bits = nested["bits"] as? Int
        else { continue }
        overrides[key] = (groupSize: groupSize, bits: bits)
    }
    return overrides
}

private struct Qwen35AffineParityFixture {
    let directory: URL
    let configData: Data
    let source: MLXVLM.Qwen35

    /// Builds the source model, quantizes it per the config idiom, saves the
    /// checkpoint exactly as the artifacts ship (format=mlx metadata), and
    /// returns everything needed to reload through the real seam.
    init(quantizationJSON: String) throws {
        let configJSON = qwen35AffineParityConfigJSON(quantization: quantizationJSON)
        configData = Data(configJSON.utf8)
        let cfg = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: configData)
        source = MLXVLM.Qwen35(cfg)

        let overrides = try qwen35AffineParityOverrides(from: quantizationJSON)
        quantize(model: source) { path, module in
            guard path.hasPrefix("language_model.") else { return nil }
            guard module is Linear || module is Embedding else { return nil }
            if let override = overrides[path] {
                return (groupSize: override.groupSize, bits: override.bits, mode: .affine)
            }
            return (groupSize: 64, bits: 4, mode: .affine)
        }
        try (source as LanguageModel).prepare()

        directory = FileManager.default.temporaryDirectory.appending(
            path: "qwen35-affine-parity-\(UUID().uuidString)",
            directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(configJSON.utf8).write(to: directory.appending(path: "config.json"))

        let arrays = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        try save(
            arrays: arrays,
            metadata: ["format": "mlx"],
            url: directory.appending(path: "model.safetensors"))
    }

    func tearDown() {
        try? FileManager.default.removeItem(at: directory)
    }

    func loadThroughRealSeam() throws -> MLXVLM.Qwen35 {
        let cfg = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: configData)
        let loaded = MLXVLM.Qwen35(cfg)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        try loadWeights(
            modelDirectory: directory, model: loaded,
            perLayerQuantization: baseConfig.perLayerQuantization)
        return loaded
    }
}

private func qwen35AffineParityLogits(_ model: MLXVLM.Qwen35, tokens: MLXArray) -> MLXArray {
    let output = model.languageModel(tokens, cache: nil, state: nil, positionIds: nil)
    eval(output.logits)
    return output.logits
}

/// Per-module structural + numeric diff between source and loaded models,
/// so a parity failure names the corrupted modules directly.
private func qwen35AffineParityModuleDiff(
    source: MLXVLM.Qwen35, loaded: MLXVLM.Qwen35
) -> [String] {
    let sourceModules = Dictionary(uniqueKeysWithValues: source.leafModules().flattened())
    let loadedModules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
    var report: [String] = []
    for (path, sourceModule) in sourceModules.sorted(by: { $0.key < $1.key }) {
        guard let loadedModule = loadedModules[path] else {
            report.append("\(path): missing from loaded model")
            continue
        }
        if type(of: sourceModule) != type(of: loadedModule) {
            report.append(
                "\(path): class \(type(of: sourceModule)) -> \(type(of: loadedModule))")
            continue
        }
        if let sourceQuantized = sourceModule as? Quantized,
            let loadedQuantized = loadedModule as? Quantized
        {
            if sourceQuantized.groupSize != loadedQuantized.groupSize
                || sourceQuantized.bits != loadedQuantized.bits
                || sourceQuantized.mode != loadedQuantized.mode
            {
                report.append(
                    "\(path): quant (gs \(sourceQuantized.groupSize), bits "
                        + "\(sourceQuantized.bits), \(sourceQuantized.mode)) -> (gs "
                        + "\(loadedQuantized.groupSize), bits \(loadedQuantized.bits), "
                        + "\(loadedQuantized.mode))")
                continue
            }
        }
        let sourceParams = Dictionary(
            uniqueKeysWithValues: sourceModule.parameters().flattened())
        let loadedParams = Dictionary(
            uniqueKeysWithValues: loadedModule.parameters().flattened())
        for (name, sourceValue) in sourceParams.sorted(by: { $0.key < $1.key }) {
            guard let loadedValue = loadedParams[name] else {
                report.append("\(path).\(name): parameter missing after load")
                continue
            }
            if sourceValue.shape != loadedValue.shape {
                report.append(
                    "\(path).\(name): shape \(sourceValue.shape) -> \(loadedValue.shape)")
            } else if !allClose(
                sourceValue.asType(.float32), loadedValue.asType(.float32),
                rtol: 0, atol: 0
            ).item(Bool.self) {
                report.append("\(path).\(name): values differ")
            }
        }
    }
    return report
}

private func expectQwen35AffineLoadParity(quantizationJSON: String) throws {
    let fixture = try Qwen35AffineParityFixture(quantizationJSON: quantizationJSON)
    defer { fixture.tearDown() }

    let tokens = MLXArray([Int32(1), 2, 3, 5, 8, 13, 21]).reshaped([1, 7])
    let sourceLogits = qwen35AffineParityLogits(fixture.source, tokens: tokens)
    let loaded = try fixture.loadThroughRealSeam()
    let loadedLogits = qwen35AffineParityLogits(loaded, tokens: tokens)

    let exact = allClose(sourceLogits, loadedLogits, rtol: 0, atol: 0).item(Bool.self)
    if !exact {
        let diff = qwen35AffineParityModuleDiff(source: fixture.source, loaded: loaded)
        let report = diff.joined(separator: "\n")
        Issue.record(
            Comment(
                rawValue:
                    "loaded logits diverge from source; module diff (\(diff.count) findings):\n\(report)"
            ))
    }
    #expect(exact)
}

@Test
func testQwen35UniformAffine4BitLoadReproducesSourceLogitsExactly() throws {
    try expectQwen35AffineLoadParity(quantizationJSON: qwen35AffineUniformQuantizationJSON)
}

@Test
func testQwen35PerLayerMixedAffineLoadReproducesSourceLogitsExactly() throws {
    try expectQwen35AffineLoadParity(quantizationJSON: qwen35AffineMixedQuantizationJSON)
}

// Root-cause regression (2026-09-02): mlx-community calibrated Qwen3.8-27B repos ship
// index-external sidecar safetensors (e.g. `optiq/mtp.safetensors`, the separate drafter's
// weights). Python mlx-lm loads strictly via model.safetensors.index.json and never reads
// them; `loadWeights`' recursive *.safetensors sweep pulled them in, the VLM route then
// failed on the unused `mtp.*` keys, factory dispatch silently fell through to the LLM
// route, and its sanitize misread the stray `mtp.` keys as an unconverted checkpoint —
// applying a second +1 shift to already-shifted norm weights. Result: a model that loads
// and decodes but is quality-destroyed. When an index is present, loadWeights must load
// exactly the index-mapped shard files.

@Test
func testIndexMappedLoadIgnoresSidecarSafetensorsOutsideTheIndex() throws {
    let fixture = try Qwen35AffineParityFixture(
        quantizationJSON: qwen35AffineMixedQuantizationJSON)
    defer { fixture.tearDown() }

    // Split the checkpoint into two indexed shards to mirror the real repos' shape.
    let arrays = try loadArrays(url: fixture.directory.appending(path: "model.safetensors"))
    let keys = arrays.keys.sorted()
    let half = keys.count / 2
    let shardA = Dictionary(uniqueKeysWithValues: keys[..<half].map { ($0, arrays[$0]!) })
    let shardB = Dictionary(uniqueKeysWithValues: keys[half...].map { ($0, arrays[$0]!) })
    try FileManager.default.removeItem(
        at: fixture.directory.appending(path: "model.safetensors"))
    try save(
        arrays: shardA, metadata: ["format": "mlx"],
        url: fixture.directory.appending(path: "model-00001-of-00002.safetensors"))
    try save(
        arrays: shardB, metadata: ["format": "mlx"],
        url: fixture.directory.appending(path: "model-00002-of-00002.safetensors"))
    var weightMap = [String: String]()
    for key in keys[..<half] { weightMap[key] = "model-00001-of-00002.safetensors" }
    for key in keys[half...] { weightMap[key] = "model-00002-of-00002.safetensors" }
    let index = try JSONSerialization.data(
        withJSONObject: ["metadata": ["total_size": 0], "weight_map": weightMap])
    try index.write(to: fixture.directory.appending(path: "model.safetensors.index.json"))

    // Index-external sidecar in a subdirectory, exactly like the real repos: foreign
    // `mtp.*` keys (the route-flip/norm-shift trigger) plus a decoy tensor whose key
    // collides with a real weight but carries garbage values.
    let sidecarDirectory = fixture.directory.appending(path: "optiq")
    try FileManager.default.createDirectory(
        at: sidecarDirectory, withIntermediateDirectories: true)
    let collidingKey = "language_model.model.norm.weight"
    let realNorm = arrays[collidingKey]!
    try save(
        arrays: [
            "mtp.fc.weight": MLXArray.zeros([8, 16]),
            "mtp.layers.0.input_layernorm.weight": MLXArray.ones([8]),
            collidingKey: MLXArray.full(realNorm.shape, values: MLXArray(Float(42))),
        ],
        url: sidecarDirectory.appending(path: "mtp.safetensors"))

    let tokens = MLXArray([Int32(1), 2, 3, 5, 8, 13, 21]).reshaped([1, 7])
    let sourceLogits = qwen35AffineParityLogits(fixture.source, tokens: tokens)
    let loaded = try fixture.loadThroughRealSeam()
    let loadedLogits = qwen35AffineParityLogits(loaded, tokens: tokens)

    let exact = allClose(sourceLogits, loadedLogits, rtol: 0, atol: 0).item(Bool.self)
    if !exact {
        let diff = qwen35AffineParityModuleDiff(source: fixture.source, loaded: loaded)
        let report = diff.joined(separator: "\n")
        Issue.record(
            Comment(
                rawValue:
                    "sidecar safetensors leaked into an index-mapped load (\(diff.count) findings):\n\(report)"
            ))
    }
    #expect(exact)
}

@Test
func testIndexlessSingleFileLoadStillWorks() throws {
    // The enumeration fallback must keep loading single-file checkpoints without an index.
    try expectQwen35AffineLoadParity(quantizationJSON: qwen35AffineUniformQuantizationJSON)
}
