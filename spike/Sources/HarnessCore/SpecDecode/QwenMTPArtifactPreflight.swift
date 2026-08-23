import Foundation

public enum QwenMTPArtifactRole: String, Equatable, Sendable {
    case target
    case drafter
}

public struct QwenMTPArtifactIdentity: Equatable, Sendable {
    public let modelID: String
    public let revision: String
    public let configSHA256: String
    public let tokenizerSHA256: String
    public let tensorManifestSHA256: String

    public init(
        modelID: String,
        revision: String,
        configSHA256: String,
        tokenizerSHA256: String,
        tensorManifestSHA256: String
    ) {
        self.modelID = modelID
        self.revision = revision
        self.configSHA256 = configSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.tensorManifestSHA256 = tensorManifestSHA256
    }
}

public struct QwenMTPTensorDescriptor: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: String

    public init(name: String, shape: [Int], dtype: String) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
    }
}

public struct QwenMTPQuantization: Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    public init(bits: Int, groupSize: Int, mode: String) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }
}

public struct QwenMTPArchitecture: Equatable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let targetLayerCount: Int
    public let fullAttentionInterval: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let headDimension: Int
    public let usesDedicatedMTPEmbeddings: Bool

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        vocabularySize: Int,
        targetLayerCount: Int,
        fullAttentionInterval: Int,
        attentionHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        usesDedicatedMTPEmbeddings: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.targetLayerCount = targetLayerCount
        self.fullAttentionInterval = fullAttentionInterval
        self.attentionHeadCount = attentionHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        self.usesDedicatedMTPEmbeddings = usesDedicatedMTPEmbeddings
    }
}

public struct QwenMTPArtifactCandidate: Equatable, Sendable {
    public let identity: QwenMTPArtifactIdentity
    public let configJSON: Data
    public let tensors: [QwenMTPTensorDescriptor]

    public init(
        identity: QwenMTPArtifactIdentity,
        configJSON: Data,
        tensors: [QwenMTPTensorDescriptor]
    ) {
        self.identity = identity
        self.configJSON = configJSON
        self.tensors = tensors
    }
}

/// One exact target/drafter compatibility row. A lock is evidence, not model discovery: callers
/// must populate it from reviewed immutable source identities before attempting model construction.
public struct QwenMTPArtifactLock: Equatable, Sendable {
    public let sourceRevision: String
    public let targetIdentity: QwenMTPArtifactIdentity
    public let drafterIdentity: QwenMTPArtifactIdentity
    public let architecture: QwenMTPArchitecture
    public let targetQuantization: QwenMTPQuantization
    public let drafterQuantization: QwenMTPQuantization
    public let drafterTensors: [QwenMTPTensorDescriptor]

    public init(
        sourceRevision: String,
        targetIdentity: QwenMTPArtifactIdentity,
        drafterIdentity: QwenMTPArtifactIdentity,
        architecture: QwenMTPArchitecture,
        targetQuantization: QwenMTPQuantization,
        drafterQuantization: QwenMTPQuantization,
        drafterTensors: [QwenMTPTensorDescriptor]
    ) {
        self.sourceRevision = sourceRevision
        self.targetIdentity = targetIdentity
        self.drafterIdentity = drafterIdentity
        self.architecture = architecture
        self.targetQuantization = targetQuantization
        self.drafterQuantization = drafterQuantization
        self.drafterTensors = drafterTensors
    }
}

/// Reviewed compatibility rows. Each row is intentionally artifact-specific; adding another target
/// or quantization requires a separate source/provenance review rather than shape-based admission.
public enum QwenMTPKnownArtifactLocks {
    // gitleaks:allow -- canonical public token-to-id vocabulary SHA-256, not a credential.
    private static let qwen35TokenizerVocabularySHA256 =
        "38eaf282b2679a1ede9fb3ee9418fc72f656f3fabfb9c33e16d34239ca88ddc1"

    /// Qwen3.5 9B affine-4bit target plus its standalone affine-5bit native MTP artifact.
    ///
    /// `tokenizerSHA256` is the canonical sorted token→id vocabulary digest shared by both artifacts,
    /// not either repository's byte-level `tokenizer.json` hash. The MTP factory borrows the target
    /// tokenizer; binding token IDs is the relevant compatibility invariant.
    public static let qwen35_9BDepth1 = QwenMTPArtifactLock(
        sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
        targetIdentity: .init(
            modelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            configSHA256: "a96942cb6a8a1d3f1d17514d81a1925d04362a6a3233b389d13012211baaa9f8",
            tokenizerSHA256: qwen35TokenizerVocabularySHA256,
            tensorManifestSHA256: "74435498cbb5085d6bc9b9cd175c80ddf010296471f5ce18d6109b2d93fb3a51"),
        drafterIdentity: .init(
            modelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            revision: "994730d199bff7799aa3ddef33a96723967a3e33",
            configSHA256: "8c6ea7ff5e52111c5286d5f3b3035c0a4e1c7744f207f035b8313cd2583374ea",
            tokenizerSHA256: qwen35TokenizerVocabularySHA256,
            tensorManifestSHA256: "11a1fdf9bb3569781d6046b49ce8934ae0be47c7964d129c1aad34f999897c1d"),
        architecture: .init(
            hiddenSize: 4096,
            intermediateSize: 12288,
            vocabularySize: 248_320,
            targetLayerCount: 32,
            fullAttentionInterval: 4,
            attentionHeadCount: 16,
            keyValueHeadCount: 4,
            headDimension: 256),
        targetQuantization: .init(bits: 4, groupSize: 64, mode: "affine"),
        drafterQuantization: .init(bits: 5, groupSize: 64, mode: "affine"),
        drafterTensors: qwen35_9BMTP5BitTensors)

    private static let qwen35_9BMTP5BitTensors: [QwenMTPTensorDescriptor] = [
        .init(name: "fc.biases", shape: [4096, 128], dtype: "BF16"),
        .init(name: "fc.scales", shape: [4096, 128], dtype: "BF16"),
        .init(name: "fc.weight", shape: [4096, 1280], dtype: "U32"),
        .init(name: "layers.0.input_layernorm.weight", shape: [4096], dtype: "BF16"),
        .init(name: "layers.0.mlp.down_proj.biases", shape: [4096, 192], dtype: "BF16"),
        .init(name: "layers.0.mlp.down_proj.scales", shape: [4096, 192], dtype: "BF16"),
        .init(name: "layers.0.mlp.down_proj.weight", shape: [4096, 1920], dtype: "U32"),
        .init(name: "layers.0.mlp.gate_proj.biases", shape: [12288, 64], dtype: "BF16"),
        .init(name: "layers.0.mlp.gate_proj.scales", shape: [12288, 64], dtype: "BF16"),
        .init(name: "layers.0.mlp.gate_proj.weight", shape: [12288, 640], dtype: "U32"),
        .init(name: "layers.0.mlp.up_proj.biases", shape: [12288, 64], dtype: "BF16"),
        .init(name: "layers.0.mlp.up_proj.scales", shape: [12288, 64], dtype: "BF16"),
        .init(name: "layers.0.mlp.up_proj.weight", shape: [12288, 640], dtype: "U32"),
        .init(name: "layers.0.post_attention_layernorm.weight", shape: [4096], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_norm.weight", shape: [256], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_proj.biases", shape: [1024, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_proj.scales", shape: [1024, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_proj.weight", shape: [1024, 640], dtype: "U32"),
        .init(name: "layers.0.self_attn.o_proj.biases", shape: [4096, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.o_proj.scales", shape: [4096, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.o_proj.weight", shape: [4096, 640], dtype: "U32"),
        .init(name: "layers.0.self_attn.q_norm.weight", shape: [256], dtype: "BF16"),
        .init(name: "layers.0.self_attn.q_proj.biases", shape: [8192, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.q_proj.scales", shape: [8192, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.q_proj.weight", shape: [8192, 640], dtype: "U32"),
        .init(name: "layers.0.self_attn.v_proj.biases", shape: [1024, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.v_proj.scales", shape: [1024, 64], dtype: "BF16"),
        .init(name: "layers.0.self_attn.v_proj.weight", shape: [1024, 640], dtype: "U32"),
        .init(name: "norm.weight", shape: [4096], dtype: "BF16"),
        .init(name: "pre_fc_norm_embedding.weight", shape: [4096], dtype: "BF16"),
        .init(name: "pre_fc_norm_hidden.weight", shape: [4096], dtype: "BF16"),
    ]
}

public struct QwenMTPArtifactBinding: Equatable, Sendable {
    public let targetModelID: String
    public let drafterModelID: String
    public let targetRevision: String
    public let drafterRevision: String
    public let sourceRevision: String
    public let architecture: QwenMTPArchitecture
    /// One committed input plus one drafted token. Depth-k remains closed.
    public let runtimeBlockSize: Int
    public let maximumAcceptedDraftTokens: Int
}

public enum QwenMTPArtifactPreflightError: Error, Equatable, Sendable {
    case invalidSourceRevision
    case invalidIdentity(role: QwenMTPArtifactRole, field: String)
    case identityMismatch(role: QwenMTPArtifactRole, field: String)
    case configDigestMismatch(role: QwenMTPArtifactRole)
    case malformedConfig(role: QwenMTPArtifactRole)
    case unsupportedModelType(role: QwenMTPArtifactRole, actual: String)
    case unsupportedTextModelType(role: QwenMTPArtifactRole, actual: String)
    case unsupportedMTPDepth(Int)
    case unsupportedBlockSize(Int)
    case architectureMismatch(field: String)
    case quantizationMismatch(role: QwenMTPArtifactRole)
    case tokenizerMismatch
    case duplicateTensor(String)
    case missingTensor(String)
    case unexpectedTensor(String)
    case tensorDescriptorMismatch(String)
    case tensorManifestDigestMismatch(role: QwenMTPArtifactRole)
}

public enum QwenMTPArtifactPreflight {
    /// Validate one exact Qwen3.5 target plus standalone native MTP artifact before any model use.
    ///
    /// This gate deliberately admits depth one only. It authenticates the source/config/tokenizer/
    /// tensor boundary and returns immutable binding data; it does not load weights or expose MTP to
    /// serving. Both artifacts' complete observed tensor descriptors must reproduce the reviewed
    /// checkpoint-manifest digest; the small standalone drafter additionally has every descriptor
    /// compared with the embedded allow-list.
    public static func validate(
        lock: QwenMTPArtifactLock,
        target: QwenMTPArtifactCandidate,
        drafter: QwenMTPArtifactCandidate
    ) throws -> QwenMTPArtifactBinding {
        guard isLowercaseHex(lock.sourceRevision, count: 40) else {
            throw QwenMTPArtifactPreflightError.invalidSourceRevision
        }

        try validateIdentity(lock.targetIdentity, role: .target)
        try validateIdentity(lock.drafterIdentity, role: .drafter)
        try validateIdentity(target.identity, role: .target)
        try validateIdentity(drafter.identity, role: .drafter)
        try requireIdentity(
            expected: lock.targetIdentity, actual: target.identity, role: .target,
            deferTensorManifest: false)
        try requireIdentity(
            expected: lock.drafterIdentity, actual: drafter.identity, role: .drafter,
            deferTensorManifest: true)

        guard sha256Hex(target.configJSON) == target.identity.configSHA256 else {
            throw QwenMTPArtifactPreflightError.configDigestMismatch(role: .target)
        }
        guard sha256Hex(drafter.configJSON) == drafter.identity.configSHA256 else {
            throw QwenMTPArtifactPreflightError.configDigestMismatch(role: .drafter)
        }

        let targetConfig = try decodeConfig(target.configJSON, role: .target)
        let drafterConfig = try decodeConfig(drafter.configJSON, role: .drafter)
        guard targetConfig.modelType == "qwen3_5" else {
            throw QwenMTPArtifactPreflightError.unsupportedModelType(
                role: .target, actual: targetConfig.modelType)
        }
        guard drafterConfig.modelType == "qwen3_5_mtp" else {
            throw QwenMTPArtifactPreflightError.unsupportedModelType(
                role: .drafter, actual: drafterConfig.modelType)
        }
        try validateTextModelType(targetConfig, role: .target)
        try validateTextModelType(drafterConfig, role: .drafter)

        let targetDepth = targetConfig.textConfig.mtpNumHiddenLayers
        let drafterDepth = drafterConfig.textConfig.mtpNumHiddenLayers
        guard targetDepth == 1 else {
            throw QwenMTPArtifactPreflightError.unsupportedMTPDepth(targetDepth)
        }
        guard drafterDepth == 1 else {
            throw QwenMTPArtifactPreflightError.unsupportedMTPDepth(drafterDepth)
        }
        guard let declaredBlockSize = drafterConfig.blockSize, declaredBlockSize >= 2 else {
            throw QwenMTPArtifactPreflightError.unsupportedBlockSize(drafterConfig.blockSize ?? 0)
        }

        try requireArchitecture(
            expected: lock.architecture,
            actual: targetConfig.textConfig.architecture,
            role: .target)
        try requireArchitecture(
            expected: lock.architecture,
            actual: drafterConfig.textConfig.architecture,
            role: .drafter)
        guard targetConfig.quantization == lock.targetQuantization else {
            throw QwenMTPArtifactPreflightError.quantizationMismatch(role: .target)
        }
        guard drafterConfig.quantization == lock.drafterQuantization else {
            throw QwenMTPArtifactPreflightError.quantizationMismatch(role: .drafter)
        }
        guard target.identity.tokenizerSHA256 == drafter.identity.tokenizerSHA256 else {
            throw QwenMTPArtifactPreflightError.tokenizerMismatch
        }

        _ = try tensorMap(target.tensors)
        let targetManifest = tensorManifestSHA256(target.tensors)
        guard targetManifest == target.identity.tensorManifestSHA256,
            targetManifest == lock.targetIdentity.tensorManifestSHA256
        else {
            throw QwenMTPArtifactPreflightError.tensorManifestDigestMismatch(role: .target)
        }

        let expectedByName = try tensorMap(lock.drafterTensors)
        let actualByName = try tensorMap(drafter.tensors)
        for name in expectedByName.keys.sorted() where actualByName[name] == nil {
            throw QwenMTPArtifactPreflightError.missingTensor(name)
        }
        for name in actualByName.keys.sorted() where expectedByName[name] == nil {
            throw QwenMTPArtifactPreflightError.unexpectedTensor(name)
        }
        for name in expectedByName.keys.sorted()
        where expectedByName[name] != actualByName[name] {
            throw QwenMTPArtifactPreflightError.tensorDescriptorMismatch(name)
        }

        let expectedManifest = tensorManifestSHA256(lock.drafterTensors)
        guard expectedManifest == lock.drafterIdentity.tensorManifestSHA256 else {
            throw QwenMTPArtifactPreflightError.tensorManifestDigestMismatch(role: .drafter)
        }
        let actualManifest = tensorManifestSHA256(drafter.tensors)
        guard actualManifest == drafter.identity.tensorManifestSHA256,
            actualManifest == lock.drafterIdentity.tensorManifestSHA256
        else {
            throw QwenMTPArtifactPreflightError.tensorManifestDigestMismatch(role: .drafter)
        }

        return QwenMTPArtifactBinding(
            targetModelID: target.identity.modelID,
            drafterModelID: drafter.identity.modelID,
            targetRevision: target.identity.revision,
            drafterRevision: drafter.identity.revision,
            sourceRevision: lock.sourceRevision,
            architecture: lock.architecture,
            runtimeBlockSize: 2,
            maximumAcceptedDraftTokens: 1)
    }

    /// Architecture-independent digest of the ordered tensor name/dtype/shape contract.
    public static func tensorManifestSHA256(_ tensors: [QwenMTPTensorDescriptor]) -> String {
        var data = Data()
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            data.append(Data(tensor.name.utf8))
            data.append(0)
            data.append(Data(tensor.dtype.utf8))
            data.append(0)
            data.append(Data(tensor.shape.map(String.init).joined(separator: ",").utf8))
            data.append(10)
        }
        return sha256Hex(data)
    }

    private static func validateIdentity(
        _ identity: QwenMTPArtifactIdentity,
        role: QwenMTPArtifactRole
    ) throws {
        guard !identity.modelID.isEmpty,
            identity.modelID.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value != 0x7f })
        else {
            throw QwenMTPArtifactPreflightError.invalidIdentity(role: role, field: "modelID")
        }
        guard isLowercaseHex(identity.revision, count: 40) else {
            throw QwenMTPArtifactPreflightError.invalidIdentity(role: role, field: "revision")
        }
        for (field, value) in [
            ("configSHA256", identity.configSHA256),
            ("tokenizerSHA256", identity.tokenizerSHA256),
            ("tensorManifestSHA256", identity.tensorManifestSHA256),
        ] where !isLowercaseHex(value, count: 64) {
            throw QwenMTPArtifactPreflightError.invalidIdentity(role: role, field: field)
        }
    }

    private static func requireIdentity(
        expected: QwenMTPArtifactIdentity,
        actual: QwenMTPArtifactIdentity,
        role: QwenMTPArtifactRole,
        deferTensorManifest: Bool
    ) throws {
        for (field, matches) in [
            ("modelID", expected.modelID == actual.modelID),
            ("revision", expected.revision == actual.revision),
            ("configSHA256", expected.configSHA256 == actual.configSHA256),
            ("tokenizerSHA256", expected.tokenizerSHA256 == actual.tokenizerSHA256),
            ("tensorManifestSHA256",
                deferTensorManifest || expected.tensorManifestSHA256 == actual.tensorManifestSHA256),
        ] where !matches {
            throw QwenMTPArtifactPreflightError.identityMismatch(role: role, field: field)
        }
    }

    private static func decodeConfig(
        _ data: Data,
        role: QwenMTPArtifactRole
    ) throws -> ConfigWire {
        do {
            return try JSONDecoder().decode(ConfigWire.self, from: data)
        } catch {
            throw QwenMTPArtifactPreflightError.malformedConfig(role: role)
        }
    }

    private static func validateTextModelType(
        _ config: ConfigWire,
        role: QwenMTPArtifactRole
    ) throws {
        guard config.textConfig.modelType == "qwen3_5_text" else {
            throw QwenMTPArtifactPreflightError.unsupportedTextModelType(
                role: role, actual: config.textConfig.modelType)
        }
    }

    private static func requireArchitecture(
        expected: QwenMTPArchitecture,
        actual: QwenMTPArchitecture,
        role: QwenMTPArtifactRole
    ) throws {
        _ = role
        for (field, matches) in [
            ("hidden_size", expected.hiddenSize == actual.hiddenSize),
            ("intermediate_size", expected.intermediateSize == actual.intermediateSize),
            ("vocab_size", expected.vocabularySize == actual.vocabularySize),
            ("num_hidden_layers", expected.targetLayerCount == actual.targetLayerCount),
            ("full_attention_interval",
                expected.fullAttentionInterval == actual.fullAttentionInterval),
            ("num_attention_heads", expected.attentionHeadCount == actual.attentionHeadCount),
            ("num_key_value_heads", expected.keyValueHeadCount == actual.keyValueHeadCount),
            ("head_dim", expected.headDimension == actual.headDimension),
            ("mtp_use_dedicated_embeddings",
                expected.usesDedicatedMTPEmbeddings == actual.usesDedicatedMTPEmbeddings),
        ] where !matches {
            throw QwenMTPArtifactPreflightError.architectureMismatch(field: field)
        }
    }

    private static func tensorMap(
        _ tensors: [QwenMTPTensorDescriptor]
    ) throws -> [String: QwenMTPTensorDescriptor] {
        var result: [String: QwenMTPTensorDescriptor] = [:]
        for tensor in tensors {
            guard !tensor.name.isEmpty,
                !tensor.name.contains("\0"),
                !tensor.name.contains("\n"),
                !tensor.dtype.isEmpty,
                !tensor.dtype.contains("\0"),
                !tensor.dtype.contains("\n"),
                !tensor.shape.isEmpty,
                tensor.shape.allSatisfy({ $0 > 0 })
            else {
                throw QwenMTPArtifactPreflightError.tensorDescriptorMismatch(tensor.name)
            }
            guard result.updateValue(tensor, forKey: tensor.name) == nil else {
                throw QwenMTPArtifactPreflightError.duplicateTensor(tensor.name)
            }
        }
        return result
    }

    private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.count == count
            && value.utf8.allSatisfy {
                (48...57).contains($0) || (97...102).contains($0)
            }
    }
}

private struct ConfigWire: Decodable {
    let blockSize: Int?
    let modelType: String
    let quantization: QwenMTPQuantization
    let textConfig: TextConfigWire

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case modelType = "model_type"
        case quantization
        case textConfig = "text_config"
    }
}

extension QwenMTPQuantization: Decodable {
    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

private struct TextConfigWire: Decodable {
    let modelType: String
    let hiddenSize: Int
    let intermediateSize: Int
    let vocabularySize: Int
    let targetLayerCount: Int
    let fullAttentionInterval: Int
    let attentionHeadCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let mtpNumHiddenLayers: Int
    let mtpUseDedicatedEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case targetLayerCount = "num_hidden_layers"
        case fullAttentionInterval = "full_attention_interval"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimension = "head_dim"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
    }

    var architecture: QwenMTPArchitecture {
        .init(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabularySize: vocabularySize,
            targetLayerCount: targetLayerCount,
            fullAttentionInterval: fullAttentionInterval,
            attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            usesDedicatedMTPEmbeddings: mtpUseDedicatedEmbeddings)
    }
}
