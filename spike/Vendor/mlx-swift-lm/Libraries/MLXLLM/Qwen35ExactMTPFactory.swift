// Copyright © 2026 Apple Inc.

import CryptoKit
import Foundation
import MLXLMCommon

public enum Qwen35ExactMTPArtifactRole: String, Equatable, Sendable {
    case target
    case drafter
}

public struct Qwen35ExactMTPArtifactIdentity: Equatable, Sendable {
    public let modelID: String
    public let revision: String
    public let configSHA256: String
    public let tokenizerSHA256: String
    public let tensorManifestSHA256: String

    package init(
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

public struct Qwen35ExactMTPTensorDescriptor: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: String

    package init(name: String, shape: [Int], dtype: String) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
    }
}

public struct Qwen35ExactMTPQuantization: Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    package init(bits: Int, groupSize: Int, mode: String) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }
}

public struct Qwen35ExactMTPArchitecture: Equatable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let targetLayerCount: Int
    public let fullAttentionInterval: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let headDimension: Int
    public let mtpDepth: Int
    public let usesDedicatedMTPEmbeddings: Bool

    package init(
        hiddenSize: Int,
        intermediateSize: Int,
        vocabularySize: Int,
        targetLayerCount: Int,
        fullAttentionInterval: Int,
        attentionHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        mtpDepth: Int,
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
        self.mtpDepth = mtpDepth
        self.usesDedicatedMTPEmbeddings = usesDedicatedMTPEmbeddings
    }
}

public struct Qwen35ExactMTPArtifactLock: Equatable, Sendable {
    public let sourceRevision: String
    public let target: Qwen35ExactMTPArtifactIdentity
    public let drafter: Qwen35ExactMTPArtifactIdentity
    public let architecture: Qwen35ExactMTPArchitecture
    public let targetQuantization: Qwen35ExactMTPQuantization
    public let drafterQuantization: Qwen35ExactMTPQuantization
    public let drafterTensors: [Qwen35ExactMTPTensorDescriptor]
    public let runtimeBlockSize: Int
    public let maximumAcceptedDraftTokens: Int

    package init(
        sourceRevision: String,
        target: Qwen35ExactMTPArtifactIdentity,
        drafter: Qwen35ExactMTPArtifactIdentity,
        architecture: Qwen35ExactMTPArchitecture,
        targetQuantization: Qwen35ExactMTPQuantization,
        drafterQuantization: Qwen35ExactMTPQuantization,
        drafterTensors: [Qwen35ExactMTPTensorDescriptor],
        runtimeBlockSize: Int = 3,
        maximumAcceptedDraftTokens: Int = 2
    ) {
        self.sourceRevision = sourceRevision
        self.target = target
        self.drafter = drafter
        self.architecture = architecture
        self.targetQuantization = targetQuantization
        self.drafterQuantization = drafterQuantization
        self.drafterTensors = drafterTensors
        self.runtimeBlockSize = runtimeBlockSize
        self.maximumAcceptedDraftTokens = maximumAcceptedDraftTokens
    }
}

public enum Qwen35ExactMTPKnownArtifactLocks {
    // gitleaks:allow -- canonical public token-to-id vocabulary SHA-256, not a credential.
    private static let qwen35TokenizerVocabularySHA256 =
        "38eaf282b2679a1ede9fb3ee9418fc72f656f3fabfb9c33e16d34239ca88ddc1"
    // gitleaks:allow -- canonical public token-to-id vocabulary SHA-256, not a credential.
    private static let qwen38TokenizerVocabularySHA256 =
        "38eaf282b2679a1ede9fb3ee9418fc72f656f3fabfb9c33e16d34239ca88ddc1"

    public static let qwen35_9BDepth1 = Qwen35ExactMTPArtifactLock(
        sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
        target: .init(
            modelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            configSHA256: "a96942cb6a8a1d3f1d17514d81a1925d04362a6a3233b389d13012211baaa9f8",
            tokenizerSHA256: qwen35TokenizerVocabularySHA256,
            tensorManifestSHA256: "74435498cbb5085d6bc9b9cd175c80ddf010296471f5ce18d6109b2d93fb3a51"),
        drafter: .init(
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
            headDimension: 256,
            mtpDepth: 1),
        targetQuantization: .init(bits: 4, groupSize: 64, mode: "affine"),
        drafterQuantization: .init(bits: 5, groupSize: 64, mode: "affine"),
        drafterTensors: qwen35_9BMTP5BitTensors)

    /// Authenticated Qwen3.8 27B MXFP8 target and native MTP pair.
    ///
    /// This row is selectable only through the exact factory SPI. It does not register the MTP model
    /// type or grant serving authority; live token/cache equivalence remains a separate gate.
    public static let qwen38_27BMXFP8Depth1 = Qwen35ExactMTPArtifactLock(
        sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
        target: .init(
            modelID: "mlx-community/Qwen3.8-27B-mxfp8",
            revision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            configSHA256: "ce016401438761ac53dcc6df48eb897036ed5c0eadd735002a35fd253701cfbf",
            tokenizerSHA256: qwen38TokenizerVocabularySHA256,
            tensorManifestSHA256: "7a5e32297470983aa8dafe03a094f59a79787fee09c25760e82abaa09fe2e7b3"),
        drafter: .init(
            modelID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            revision: "a50634460045613f166b09b13519466e801c6568",
            configSHA256: "be0048271c09a95620762f32cac1e487d4a798368ac25f42c7c35d9a9f1b4827",
            tokenizerSHA256: qwen38TokenizerVocabularySHA256,
            tensorManifestSHA256: "32ee1b818a5c6ce7191863910131b172ac3fa82f99ce8c23521ddac27cc1fcb7"),
        architecture: .init(
            hiddenSize: 5120,
            intermediateSize: 17408,
            vocabularySize: 248_320,
            targetLayerCount: 64,
            fullAttentionInterval: 4,
            attentionHeadCount: 24,
            keyValueHeadCount: 4,
            headDimension: 256,
            mtpDepth: 1),
        targetQuantization: .init(bits: 8, groupSize: 32, mode: "mxfp8"),
        drafterQuantization: .init(bits: 8, groupSize: 32, mode: "mxfp8"),
        drafterTensors: qwen38_27BMXFP8MTPTensors)

    /// Authenticated Qwen3.8 27B affine-4-bit target paired with the same MXFP8 native MTP
    /// drafter. The drafter is mxfp8-matched to the mxfp8 target, so acceptance on this row is a
    /// floor, not a ceiling; target-side verification still guarantees exactness.
    ///
    /// This row is selectable only through the exact factory SPI. It does not register the MTP model
    /// type or grant serving authority; live token/cache equivalence remains a separate gate.
    public static let qwen38_27B4BitDepth1 = Qwen35ExactMTPArtifactLock(
        sourceRevision: "01472a78fca830689ff78246a82c6d31ab111a78",
        target: .init(
            modelID: "mlx-community/Qwen3.8-27B-4bit",
            revision: "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
            configSHA256: "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
            tokenizerSHA256: qwen38TokenizerVocabularySHA256,
            tensorManifestSHA256: "c0a71d953c6e2177681c46e1c7ad19406e09a387f3036b698eabad1030ccd350"),
        drafter: .init(
            modelID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            revision: "a50634460045613f166b09b13519466e801c6568",
            configSHA256: "be0048271c09a95620762f32cac1e487d4a798368ac25f42c7c35d9a9f1b4827",
            tokenizerSHA256: qwen38TokenizerVocabularySHA256,
            tensorManifestSHA256: "32ee1b818a5c6ce7191863910131b172ac3fa82f99ce8c23521ddac27cc1fcb7"),
        architecture: .init(
            hiddenSize: 5120,
            intermediateSize: 17408,
            vocabularySize: 248_320,
            targetLayerCount: 64,
            fullAttentionInterval: 4,
            attentionHeadCount: 24,
            keyValueHeadCount: 4,
            headDimension: 256,
            mtpDepth: 1),
        targetQuantization: .init(bits: 4, groupSize: 64, mode: "affine"),
        drafterQuantization: .init(bits: 8, groupSize: 32, mode: "mxfp8"),
        drafterTensors: qwen38_27BMXFP8MTPTensors)

    private static let qwen35_9BMTP5BitTensors: [Qwen35ExactMTPTensorDescriptor] = [
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

    private static let qwen38_27BMXFP8MTPTensors: [Qwen35ExactMTPTensorDescriptor] = [
        .init(name: "fc.scales", shape: [5120, 320], dtype: "U8"),
        .init(name: "fc.weight", shape: [5120, 2560], dtype: "U32"),
        .init(name: "layers.0.input_layernorm.weight", shape: [5120], dtype: "BF16"),
        .init(name: "layers.0.mlp.down_proj.scales", shape: [5120, 544], dtype: "U8"),
        .init(name: "layers.0.mlp.down_proj.weight", shape: [5120, 4352], dtype: "U32"),
        .init(name: "layers.0.mlp.gate_proj.scales", shape: [17408, 160], dtype: "U8"),
        .init(name: "layers.0.mlp.gate_proj.weight", shape: [17408, 1280], dtype: "U32"),
        .init(name: "layers.0.mlp.up_proj.scales", shape: [17408, 160], dtype: "U8"),
        .init(name: "layers.0.mlp.up_proj.weight", shape: [17408, 1280], dtype: "U32"),
        .init(name: "layers.0.post_attention_layernorm.weight", shape: [5120], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_norm.weight", shape: [256], dtype: "BF16"),
        .init(name: "layers.0.self_attn.k_proj.scales", shape: [1024, 160], dtype: "U8"),
        .init(name: "layers.0.self_attn.k_proj.weight", shape: [1024, 1280], dtype: "U32"),
        .init(name: "layers.0.self_attn.o_proj.scales", shape: [5120, 192], dtype: "U8"),
        .init(name: "layers.0.self_attn.o_proj.weight", shape: [5120, 1536], dtype: "U32"),
        .init(name: "layers.0.self_attn.q_norm.weight", shape: [256], dtype: "BF16"),
        .init(name: "layers.0.self_attn.q_proj.scales", shape: [12288, 160], dtype: "U8"),
        .init(name: "layers.0.self_attn.q_proj.weight", shape: [12288, 1280], dtype: "U32"),
        .init(name: "layers.0.self_attn.v_proj.scales", shape: [1024, 160], dtype: "U8"),
        .init(name: "layers.0.self_attn.v_proj.weight", shape: [1024, 1280], dtype: "U32"),
        .init(name: "norm.weight", shape: [5120], dtype: "BF16"),
        .init(name: "pre_fc_norm_embedding.weight", shape: [5120], dtype: "BF16"),
        .init(name: "pre_fc_norm_hidden.weight", shape: [5120], dtype: "BF16"),
    ]
}

public struct Qwen35ExactMTPArtifactCandidate: Equatable, Sendable {
    public let identity: Qwen35ExactMTPArtifactIdentity
    public let configJSON: Data
    public let tensors: [Qwen35ExactMTPTensorDescriptor]

    package init(
        identity: Qwen35ExactMTPArtifactIdentity,
        configJSON: Data,
        tensors: [Qwen35ExactMTPTensorDescriptor]
    ) {
        self.identity = identity
        self.configJSON = configJSON
        self.tensors = tensors
    }
}

public struct Qwen35ExactMTPPreflightEvidence: Equatable, Sendable {
    public let lock: Qwen35ExactMTPArtifactLock
    public let target: Qwen35ExactMTPArtifactCandidate
    public let drafter: Qwen35ExactMTPArtifactCandidate
    public let binding: Qwen35ExactMTPBinding
}

public struct Qwen35ExactMTPBinding: Equatable, Sendable {
    public let targetModelID: String
    public let drafterModelID: String
    public let targetRevision: String
    public let drafterRevision: String
    public let sourceRevision: String
    public let architecture: Qwen35ExactMTPArchitecture
    public let runtimeBlockSize: Int
    public let maximumAcceptedDraftTokens: Int
}

public enum Qwen35ExactMTPAdmissionError: Error, Equatable, Sendable {
    case invalidSourceRevision
    case invalidIdentity(role: Qwen35ExactMTPArtifactRole, field: String)
    case identityMismatch(role: Qwen35ExactMTPArtifactRole, field: String)
    case missingFile(role: Qwen35ExactMTPArtifactRole, name: String)
    case configDigestMismatch(role: Qwen35ExactMTPArtifactRole)
    case tokenizerDigestMismatch(role: Qwen35ExactMTPArtifactRole)
    case malformedConfig(role: Qwen35ExactMTPArtifactRole)
    case unsupportedModelType(role: Qwen35ExactMTPArtifactRole, actual: String)
    case unsupportedTextModelType(role: Qwen35ExactMTPArtifactRole, actual: String)
    case unsupportedMTPDepth(role: Qwen35ExactMTPArtifactRole, actual: Int)
    case unsupportedBlockSize(Int)
    case architectureMismatch(role: Qwen35ExactMTPArtifactRole, field: String)
    case quantizationMismatch(role: Qwen35ExactMTPArtifactRole)
    case tokenizerMismatch
    case noSafetensors(role: Qwen35ExactMTPArtifactRole)
    case malformedSafetensorHeader(role: Qwen35ExactMTPArtifactRole, file: String)
    case incompleteTensorManifest(role: Qwen35ExactMTPArtifactRole)
    case duplicateTensor(role: Qwen35ExactMTPArtifactRole, name: String)
    case missingTensor(String)
    case unexpectedTensor(String)
    case tensorDescriptorMismatch(String)
    case tensorManifestDigestMismatch(role: Qwen35ExactMTPArtifactRole)
}

public struct Qwen35ExactMTPLoadedPair {
    public let target: ModelContext
    public let drafter: MTPDrafterContext
    public let binding: Qwen35ExactMTPBinding
}

@_spi(FastMLXExactMTP)
public enum Qwen35ExactMTPArtifactSelection: Equatable, Sendable {
    case qwen35_9BDepth1
    case qwen38_27BMXFP8Depth1
    case qwen38_27B4BitDepth1
}

@_spi(FastMLXExactMTP)
public enum Qwen35ExactMTPFactory {
    public static func loadDepth1Pair(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        authorizePreflight: @Sendable (Qwen35ExactMTPPreflightEvidence) async throws -> Void,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending Qwen35ExactMTPLoadedPair {
        try await loadDepth1Pair(
            selection: .qwen35_9BDepth1,
            from: downloader,
            using: tokenizerLoader,
            authorizePreflight: authorizePreflight,
            progressHandler: progressHandler)
    }

    public static func loadDepth1Pair(
        selection: Qwen35ExactMTPArtifactSelection,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        authorizePreflight: @Sendable (Qwen35ExactMTPPreflightEvidence) async throws -> Void,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending Qwen35ExactMTPLoadedPair {
        let lock = selectedLock(selection)
        let targetConfiguration = ModelConfiguration(
            id: lock.target.modelID,
            revision: lock.target.revision)
        let drafterConfiguration = ModelConfiguration(
            id: lock.drafter.modelID,
            revision: lock.drafter.revision)
        let targetResolved = try await resolve(
            configuration: targetConfiguration,
            from: downloader,
            useLatest: false,
            progressHandler: progressHandler)
        let drafterResolved = try await resolve(
            configuration: drafterConfiguration,
            from: downloader,
            useLatest: false,
            progressHandler: { _ in })
        return try await loadResolvedDepth1Pair(
            lock: lock,
            target: targetResolved,
            drafter: drafterResolved,
            tokenizerLoader: tokenizerLoader,
            authorizePreflight: authorizePreflight)
    }

    @_spi(FastMLXExactMTP)
    public static func admitSourceLockedDepth1Pair(
        selection: Qwen35ExactMTPArtifactSelection,
        targetDirectory: URL,
        drafterDirectory: URL
    ) throws -> Qwen35ExactMTPPreflightEvidence {
        let lock = selectedLock(selection)
        return try admitResolvedDepth1Pair(
            lock: lock,
            targetDirectory: targetDirectory,
            targetTokenizerDirectory: targetDirectory,
            drafterDirectory: drafterDirectory,
            drafterTokenizerDirectory: drafterDirectory)
    }

    private static func selectedLock(
        _ selection: Qwen35ExactMTPArtifactSelection
    ) -> Qwen35ExactMTPArtifactLock {
        switch selection {
        case .qwen35_9BDepth1:
            Qwen35ExactMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            Qwen35ExactMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        case .qwen38_27B4BitDepth1:
            Qwen35ExactMTPKnownArtifactLocks.qwen38_27B4BitDepth1
        }
    }

    package static func loadResolvedDepth1Pair(
        lock: Qwen35ExactMTPArtifactLock,
        target targetResolved: ResolvedModelConfiguration,
        drafter drafterResolved: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader,
        authorizePreflight: @Sendable (Qwen35ExactMTPPreflightEvidence) async throws -> Void
    ) async throws -> Qwen35ExactMTPLoadedPair {
        let evidence = try admitResolvedDepth1Pair(
            lock: lock,
            targetDirectory: targetResolved.modelDirectory,
            targetTokenizerDirectory: targetResolved.tokenizerDirectory,
            drafterDirectory: drafterResolved.modelDirectory,
            drafterTokenizerDirectory: drafterResolved.tokenizerDirectory)
        try await authorizePreflight(evidence)

        let target = try await LLMModelFactory.shared._load(
            configuration: targetResolved,
            tokenizerLoader: tokenizerLoader)

        let drafterConfigData = evidence.drafter.configJSON
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: drafterConfigData)
        let drafterConfig = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: drafterConfigData)
        let drafter = Qwen35MTPDraftModel(drafterConfig, preconvertedNorms: true)
        try loadWeights(
            modelDirectory: drafterResolved.modelDirectory,
            model: drafter,
            perLayerQuantization: baseConfig.perLayerQuantization)

        return Qwen35ExactMTPLoadedPair(
            target: target,
            drafter: MTPDrafterContext(
                configuration: ModelConfiguration(
                    directory: drafterResolved.modelDirectory,
                    defaultPrompt: ""),
                model: drafter),
            binding: evidence.binding)
    }

    package static func admitResolvedDepth1Pair(
        lock: Qwen35ExactMTPArtifactLock,
        targetDirectory: URL,
        targetTokenizerDirectory: URL,
        drafterDirectory: URL,
        drafterTokenizerDirectory: URL
    ) throws -> Qwen35ExactMTPPreflightEvidence {
        let targetCandidate = try artifactCandidate(
            identity: lock.target,
            directory: targetDirectory,
            tokenizerDirectory: targetTokenizerDirectory,
            role: .target)
        let drafterCandidate = try artifactCandidate(
            identity: lock.drafter,
            directory: drafterDirectory,
            tokenizerDirectory: drafterTokenizerDirectory,
            role: .drafter)
        let binding = try Qwen35ExactMTPAdmission.validate(
            lock: lock,
            target: targetCandidate,
            drafter: drafterCandidate)
        return Qwen35ExactMTPPreflightEvidence(
            lock: lock,
            target: targetCandidate,
            drafter: drafterCandidate,
            binding: binding)
    }

    private static func artifactCandidate(
        identity expectedIdentity: Qwen35ExactMTPArtifactIdentity,
        directory: URL,
        tokenizerDirectory: URL,
        role: Qwen35ExactMTPArtifactRole
    ) throws -> Qwen35ExactMTPArtifactCandidate {
        let configURL = directory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configURL)
        } catch {
            throw Qwen35ExactMTPAdmissionError.missingFile(role: role, name: "config.json")
        }
        let tokenizerDigest = try Qwen35ExactMTPAdmission.canonicalTokenizerVocabularySHA256(
            in: tokenizerDirectory,
            role: role)
        let tensors = try Qwen35ExactMTPAdmission.safetensorDescriptors(
            in: directory,
            role: role)
        let identity = Qwen35ExactMTPArtifactIdentity(
            modelID: expectedIdentity.modelID,
            revision: expectedIdentity.revision,
            configSHA256: sha256Hex(configData),
            tokenizerSHA256: tokenizerDigest,
            tensorManifestSHA256: Qwen35ExactMTPAdmission.tensorManifestSHA256(tensors))
        return .init(identity: identity, configJSON: configData, tensors: tensors)
    }
}

package enum Qwen35ExactMTPAdmission {
    package static func validate(
        lock: Qwen35ExactMTPArtifactLock,
        target: Qwen35ExactMTPArtifactCandidate,
        drafter: Qwen35ExactMTPArtifactCandidate
    ) throws -> Qwen35ExactMTPBinding {
        guard isLowercaseHex(lock.sourceRevision, count: 40) else {
            throw Qwen35ExactMTPAdmissionError.invalidSourceRevision
        }

        try validateIdentity(lock.target, role: .target)
        try validateIdentity(lock.drafter, role: .drafter)
        try validateIdentity(target.identity, role: .target)
        try validateIdentity(drafter.identity, role: .drafter)
        try requireIdentity(expected: lock.target, actual: target.identity, role: .target)
        try requireIdentity(expected: lock.drafter, actual: drafter.identity, role: .drafter)

        guard sha256Hex(target.configJSON) == target.identity.configSHA256 else {
            throw Qwen35ExactMTPAdmissionError.configDigestMismatch(role: .target)
        }
        guard sha256Hex(drafter.configJSON) == drafter.identity.configSHA256 else {
            throw Qwen35ExactMTPAdmissionError.configDigestMismatch(role: .drafter)
        }

        let targetConfig = try decodeConfig(target.configJSON, role: .target)
        let drafterConfig = try decodeConfig(drafter.configJSON, role: .drafter)
        guard targetConfig.modelType == "qwen3_5" else {
            throw Qwen35ExactMTPAdmissionError.unsupportedModelType(
                role: .target, actual: targetConfig.modelType)
        }
        guard drafterConfig.modelType == "qwen3_5_mtp" else {
            throw Qwen35ExactMTPAdmissionError.unsupportedModelType(
                role: .drafter, actual: drafterConfig.modelType)
        }
        try validateTextModelType(targetConfig, role: .target)
        try validateTextModelType(drafterConfig, role: .drafter)

        guard targetConfig.textConfig.mtpNumHiddenLayers == lock.architecture.mtpDepth else {
            throw Qwen35ExactMTPAdmissionError.unsupportedMTPDepth(
                role: .target, actual: targetConfig.textConfig.mtpNumHiddenLayers)
        }
        guard drafterConfig.textConfig.mtpNumHiddenLayers == lock.architecture.mtpDepth else {
            throw Qwen35ExactMTPAdmissionError.unsupportedMTPDepth(
                role: .drafter, actual: drafterConfig.textConfig.mtpNumHiddenLayers)
        }
        guard lock.runtimeBlockSize == 3, lock.maximumAcceptedDraftTokens == 2 else {
            throw Qwen35ExactMTPAdmissionError.unsupportedBlockSize(lock.runtimeBlockSize)
        }
        guard let blockSize = drafterConfig.blockSize, blockSize == lock.runtimeBlockSize else {
            throw Qwen35ExactMTPAdmissionError.unsupportedBlockSize(
                drafterConfig.blockSize ?? 0)
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
            throw Qwen35ExactMTPAdmissionError.quantizationMismatch(role: .target)
        }
        guard drafterConfig.quantization == lock.drafterQuantization else {
            throw Qwen35ExactMTPAdmissionError.quantizationMismatch(role: .drafter)
        }
        guard target.identity.tokenizerSHA256 == lock.target.tokenizerSHA256 else {
            throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: .target)
        }
        guard drafter.identity.tokenizerSHA256 == lock.drafter.tokenizerSHA256 else {
            throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: .drafter)
        }
        guard target.identity.tokenizerSHA256 == drafter.identity.tokenizerSHA256 else {
            throw Qwen35ExactMTPAdmissionError.tokenizerMismatch
        }

        _ = try tensorMap(target.tensors, role: .target)
        let targetManifest = tensorManifestSHA256(target.tensors)
        guard targetManifest == target.identity.tensorManifestSHA256,
            targetManifest == lock.target.tensorManifestSHA256
        else {
            throw Qwen35ExactMTPAdmissionError.tensorManifestDigestMismatch(role: .target)
        }

        let expectedByName = try tensorMap(lock.drafterTensors, role: .drafter)
        let actualByName = try tensorMap(drafter.tensors, role: .drafter)
        for name in expectedByName.keys.sorted() where actualByName[name] == nil {
            throw Qwen35ExactMTPAdmissionError.missingTensor(name)
        }
        for name in actualByName.keys.sorted() where expectedByName[name] == nil {
            throw Qwen35ExactMTPAdmissionError.unexpectedTensor(name)
        }
        for name in expectedByName.keys.sorted()
        where expectedByName[name] != actualByName[name] {
            throw Qwen35ExactMTPAdmissionError.tensorDescriptorMismatch(name)
        }

        let expectedManifest = tensorManifestSHA256(lock.drafterTensors)
        guard expectedManifest == lock.drafter.tensorManifestSHA256 else {
            throw Qwen35ExactMTPAdmissionError.tensorManifestDigestMismatch(role: .drafter)
        }
        let actualManifest = tensorManifestSHA256(drafter.tensors)
        guard actualManifest == drafter.identity.tensorManifestSHA256,
            actualManifest == lock.drafter.tensorManifestSHA256
        else {
            throw Qwen35ExactMTPAdmissionError.tensorManifestDigestMismatch(role: .drafter)
        }

        return Qwen35ExactMTPBinding(
            targetModelID: target.identity.modelID,
            drafterModelID: drafter.identity.modelID,
            targetRevision: target.identity.revision,
            drafterRevision: drafter.identity.revision,
            sourceRevision: lock.sourceRevision,
            architecture: lock.architecture,
            runtimeBlockSize: lock.runtimeBlockSize,
            maximumAcceptedDraftTokens: lock.maximumAcceptedDraftTokens)
    }

    package static func tensorManifestSHA256(
        _ tensors: [Qwen35ExactMTPTensorDescriptor]
    ) -> String {
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

    package static func canonicalTokenizerVocabularySHA256(
        in directory: URL,
        role: Qwen35ExactMTPArtifactRole
    ) throws -> String {
        let tokenizerJSON = directory.appending(component: "tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerJSON.path) {
            let data = try Data(contentsOf: tokenizerJSON)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let vocab = tokenizerJSONVocabulary(json)
            else {
                throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: role)
            }
            return try vocabularySHA256(vocab, role: role)
        }

        let vocabJSON = directory.appending(component: "vocab.json")
        if FileManager.default.fileExists(atPath: vocabJSON.path) {
            let data = try Data(contentsOf: vocabJSON)
            guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
                throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: role)
            }
            return try vocabularySHA256(vocab, role: role)
        }

        throw Qwen35ExactMTPAdmissionError.missingFile(role: role, name: "tokenizer.json")
    }

    package static func safetensorDescriptors(
        in directory: URL,
        role: Qwen35ExactMTPArtifactRole
    ) throws -> [Qwen35ExactMTPTensorDescriptor] {
        let files = try safetensorFiles(in: directory)
        guard !files.isEmpty else {
            throw Qwen35ExactMTPAdmissionError.noSafetensors(role: role)
        }

        var tensors: [Qwen35ExactMTPTensorDescriptor] = []
        var tensorFiles: [String: String] = [:]
        for file in files {
            let relative = relativePath(file, under: directory)
            let descriptors = try safetensorDescriptors(file: file, role: role)
            for descriptor in descriptors {
                if tensorFiles.updateValue(relative, forKey: descriptor.name) != nil {
                    throw Qwen35ExactMTPAdmissionError.duplicateTensor(
                        role: role, name: descriptor.name)
                }
            }
            tensors.append(contentsOf: descriptors)
        }
        try validateIndexManifest(
            directory: directory,
            tensorFiles: tensorFiles,
            role: role)
        return tensors.sorted(by: { $0.name < $1.name })
    }

    private static func validateIdentity(
        _ identity: Qwen35ExactMTPArtifactIdentity,
        role: Qwen35ExactMTPArtifactRole
    ) throws {
        guard !identity.modelID.isEmpty,
            identity.modelID.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value != 0x7f })
        else {
            throw Qwen35ExactMTPAdmissionError.invalidIdentity(role: role, field: "modelID")
        }
        guard isLowercaseHex(identity.revision, count: 40) else {
            throw Qwen35ExactMTPAdmissionError.invalidIdentity(role: role, field: "revision")
        }
        for (field, value) in [
            ("configSHA256", identity.configSHA256),
            ("tokenizerSHA256", identity.tokenizerSHA256),
            ("tensorManifestSHA256", identity.tensorManifestSHA256),
        ] where !isLowercaseHex(value, count: 64) {
            throw Qwen35ExactMTPAdmissionError.invalidIdentity(role: role, field: field)
        }
    }

    private static func requireIdentity(
        expected: Qwen35ExactMTPArtifactIdentity,
        actual: Qwen35ExactMTPArtifactIdentity,
        role: Qwen35ExactMTPArtifactRole
    ) throws {
        for (field, matches) in [
            ("modelID", expected.modelID == actual.modelID),
            ("revision", expected.revision == actual.revision),
            ("configSHA256", expected.configSHA256 == actual.configSHA256),
            ("tokenizerSHA256", expected.tokenizerSHA256 == actual.tokenizerSHA256),
            ("tensorManifestSHA256",
                expected.tensorManifestSHA256 == actual.tensorManifestSHA256),
        ] where !matches {
            throw Qwen35ExactMTPAdmissionError.identityMismatch(role: role, field: field)
        }
    }

    private static func decodeConfig(
        _ data: Data,
        role: Qwen35ExactMTPArtifactRole
    ) throws -> Qwen35ExactMTPConfigWire {
        do {
            return try JSONDecoder.json5().decode(Qwen35ExactMTPConfigWire.self, from: data)
        } catch {
            throw Qwen35ExactMTPAdmissionError.malformedConfig(role: role)
        }
    }

    private static func validateTextModelType(
        _ config: Qwen35ExactMTPConfigWire,
        role: Qwen35ExactMTPArtifactRole
    ) throws {
        guard config.textConfig.modelType == "qwen3_5_text" else {
            throw Qwen35ExactMTPAdmissionError.unsupportedTextModelType(
                role: role, actual: config.textConfig.modelType)
        }
    }

    private static func requireArchitecture(
        expected: Qwen35ExactMTPArchitecture,
        actual: Qwen35ExactMTPArchitecture,
        role: Qwen35ExactMTPArtifactRole
    ) throws {
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
            throw Qwen35ExactMTPAdmissionError.architectureMismatch(role: role, field: field)
        }
    }

    private static func tensorMap(
        _ tensors: [Qwen35ExactMTPTensorDescriptor],
        role: Qwen35ExactMTPArtifactRole
    ) throws -> [String: Qwen35ExactMTPTensorDescriptor] {
        var result: [String: Qwen35ExactMTPTensorDescriptor] = [:]
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
                throw Qwen35ExactMTPAdmissionError.tensorDescriptorMismatch(tensor.name)
            }
            guard result.updateValue(tensor, forKey: tensor.name) == nil else {
                throw Qwen35ExactMTPAdmissionError.duplicateTensor(role: role, name: tensor.name)
            }
        }
        return result
    }

    private static func tokenizerJSONVocabulary(_ json: [String: Any]) -> [String: Int]? {
        if let model = json["model"] as? [String: Any],
            let vocab = model["vocab"] as? [String: Int]
        {
            return vocab
        }
        return json["vocab"] as? [String: Int]
    }

    private static func vocabularySHA256(
        _ vocab: [String: Int],
        role: Qwen35ExactMTPArtifactRole
    ) throws -> String {
        // The reviewed lock was produced by `jq -S -c '.model.vocab'`. Foundation's
        // `.sortedKeys` uses a different ordering for some non-ASCII keys, and its default JSON
        // encoding escapes slashes, so it is not byte-compatible with that provenance command.
        // jq orders object keys by UTF-8 bytes and emits unescaped slashes in compact JSON.
        var data = Data("{".utf8)
        do {
            for (index, token) in vocab.keys.sorted(by: {
                $0.utf8.lexicographicallyPrecedes($1.utf8)
            }).enumerated() {
                if index > 0 {
                    data.append(44)
                }
                let encoded = try JSONSerialization.data(
                    withJSONObject: [token], options: [.withoutEscapingSlashes])
                guard encoded.count >= 2 else {
                    throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: role)
                }
                data.append(encoded.dropFirst().dropLast())
                guard let tokenID = vocab[token] else {
                    throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: role)
                }
                data.append(58)
                data.append(Data(String(tokenID).utf8))
            }
        } catch let error as Qwen35ExactMTPAdmissionError {
            throw error
        } catch {
            throw Qwen35ExactMTPAdmissionError.tokenizerDigestMismatch(role: role)
        }
        data.append(125)
        data.append(10)
        return sha256Hex(data)
    }

    private static func safetensorFiles(in directory: URL) throws -> [URL] {
        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey])!
        var files: [URL] = []
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            files.append(url)
        }
        return files.sorted { relativePath($0, under: directory) < relativePath($1, under: directory) }
    }

    private static func safetensorDescriptors(
        file: URL,
        role: Qwen35ExactMTPArtifactRole
    ) throws -> [Qwen35ExactMTPTensorDescriptor] {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: file)
        } catch {
            throw Qwen35ExactMTPAdmissionError.malformedSafetensorHeader(
                role: role, file: file.lastPathComponent)
        }
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw Qwen35ExactMTPAdmissionError.malformedSafetensorHeader(
                role: role, file: file.lastPathComponent)
        }
        var headerLength: UInt64 = 0
        for (offset, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << UInt64(offset * 8)
        }
        guard headerLength > 0, headerLength <= UInt64(Int.max),
            let headerData = try handle.read(upToCount: Int(headerLength)),
            headerData.count == Int(headerLength),
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw Qwen35ExactMTPAdmissionError.malformedSafetensorHeader(
                role: role, file: file.lastPathComponent)
        }

        var result: [Qwen35ExactMTPTensorDescriptor] = []
        for key in json.keys.sorted() where key != "__metadata__" {
            guard let entry = json[key] as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                entry["data_offsets"] is [Int]
            else {
                throw Qwen35ExactMTPAdmissionError.malformedSafetensorHeader(
                    role: role, file: file.lastPathComponent)
            }
            result.append(.init(name: key, shape: shape, dtype: dtype))
        }
        return result
    }

    private static func validateIndexManifest(
        directory: URL,
        tensorFiles: [String: String],
        role: Qwen35ExactMTPArtifactRole
    ) throws {
        let indexURL = directory.appending(component: "model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            return
        }
        let data = try Data(contentsOf: indexURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String],
            !weightMap.isEmpty
        else {
            throw Qwen35ExactMTPAdmissionError.incompleteTensorManifest(role: role)
        }
        let expectedTensorNames = Set(weightMap.keys)
        guard expectedTensorNames == Set(tensorFiles.keys) else {
            throw Qwen35ExactMTPAdmissionError.incompleteTensorManifest(role: role)
        }
        for (tensor, file) in weightMap {
            guard tensorFiles[tensor] == file,
                FileManager.default.fileExists(
                    atPath: directory.appending(component: file).path)
            else {
                throw Qwen35ExactMTPAdmissionError.incompleteTensorManifest(role: role)
            }
        }
    }

    private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.count == count
            && value.utf8.allSatisfy {
                (48...57).contains($0) || (97...102).contains($0)
            }
    }
}

private struct Qwen35ExactMTPConfigWire: Decodable {
    let blockSize: Int?
    let modelType: String
    let quantization: Qwen35ExactMTPQuantization
    let textConfig: Qwen35ExactMTPTextConfigWire

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case modelType = "model_type"
        case quantization
        case textConfig = "text_config"
    }
}

extension Qwen35ExactMTPQuantization: Decodable {
    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

private struct Qwen35ExactMTPTextConfigWire: Decodable {
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

    var architecture: Qwen35ExactMTPArchitecture {
        .init(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabularySize: vocabularySize,
            targetLayerCount: targetLayerCount,
            fullAttentionInterval: fullAttentionInterval,
            attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            mtpDepth: mtpNumHiddenLayers,
            usesDedicatedMTPEmbeddings: mtpUseDedicatedEmbeddings)
    }
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func relativePath(_ url: URL, under directory: URL) -> String {
    let base = directory.standardizedFileURL.path
    let path = url.standardizedFileURL.path
    guard path.hasPrefix(base + "/") else {
        return url.lastPathComponent
    }
    return String(path.dropFirst(base.count + 1))
}
