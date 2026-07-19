import Foundation

/// Explicit request for how an affine-backed KV tier reaches attention. The absence of a
/// request preserves the previously qualified materialize-then-attend behavior.
public enum CompressedKVAttentionRequest:
    String, Codable, CaseIterable, Equatable, Hashable, Sendable
{
    case materialize
    case splitAffineQuantizedMM = "split-affine-quantized-mm"
}

/// Operation actually reported by the cache after an attention forward. This lives in
/// HarnessCore so durable evidence does not depend on the MLX-coupled SpikeCore enum.
public enum CompressedKVAttentionObservedOperation:
    String, Codable, Equatable, Hashable, Sendable
{
    case materializedKV = "materialized-kv"
    case splitQuantizedMM = "split-quantized-mm"
}

public enum CompressedKVAttentionModelFamily:
    String, Codable, Equatable, Hashable, Sendable
{
    case qwen3
    case llama
}

public enum CompressedKVAttentionRuntimeAdmissionError:
    Error, Equatable, Sendable
{
    case invalidSourceIdentity
    case checkpointContentIdentityMismatch
    case sourceIdentityChangedDuringModelLoad
    case malformedModelConfig
    case unsupportedModelType(String)
    case unsupportedArchitecture(String)
    case unsupportedAttentionFeature(String)
    case invalidGeometry(String)
    case unsupportedGroupGeometry(
        headDimension: Int,
        keyGroupSize: Int,
        valueGroupSize: Int)
    case referenceModelIdentityMismatch
    case scheduleIdentityMismatch
    case attentionOperationMismatch
}

/// Exact, path-free source identity sampled on both sides of model loading. The checkpoint
/// manifest and tokenizer digest are included because a frozen KVTuner schedule binds all three
/// sources, not only `config.json`.
public struct CompressedKVAttentionRuntimeSourceSnapshot:
    Equatable, Sendable
{
    public let exactModelConfigData: Data
    public let checkpointManifestHash: String
    /// Domain-separated SHA-256 over the exact config/index bytes, logical shard names,
    /// resolved regular-file sizes, and every shard content byte. The cheaper manifest hash is
    /// retained for compatibility with frozen KVTuner schedules; runtime qualification binds
    /// both identities so a same-size weight replacement cannot inherit an admission receipt.
    public let checkpointContentSHA256: String
    public let tokenizerSHA256: String

    private init(
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String
    ) {
        self.exactModelConfigData = exactModelConfigData
        self.checkpointManifestHash = checkpointManifestHash
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
    }

    public static func load(
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String
    ) throws -> Self {
        guard !exactModelConfigData.isEmpty,
            isLowercaseHex(checkpointManifestHash, lengths: [16, 64]),
            isLowercaseHex(checkpointContentSHA256, lengths: [64]),
            isLowercaseHex(tokenizerSHA256, lengths: [64])
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidSourceIdentity
        }
        return Self(
            exactModelConfigData: exactModelConfigData,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
    }

    public static func validateUnchanged(
        before: Self,
        after: Self
    ) throws -> Self {
        guard before == after else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .sourceIdentityChangedDuringModelLoad
        }
        return after
    }

    private static func isLowercaseHex(
        _ value: String,
        lengths: Set<Int>
    ) -> Bool {
        lengths.contains(value.utf8.count)
            && value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }
    }
}

/// Authenticated capability token for the initial shared-router dense-GQA gate. This is an
/// explicit qualification registry, not the product's model-support boundary: other models keep
/// their existing materialized route until their attention/cache contract is independently
/// qualified.
public struct CompressedKVAttentionRuntimeAdmission:
    Codable, Equatable, Hashable, Sendable
{
    public let family: CompressedKVAttentionModelFamily
    public let modelType: String
    public let architecture: String
    public let modelConfigHash: String
    public let modelConfigSHA256: String
    public let checkpointManifestHash: String
    public let checkpointContentSHA256: String
    public let tokenizerSHA256: String
    public let layerCount: Int
    public let queryHeadCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let maxPositionEmbeddings: Int

    private struct ModelConfiguration: Decodable {
        let modelType: String
        let architectures: [String]
        let hiddenSize: Int
        let layerCount: Int
        let queryHeadCount: Int
        let kvHeadCount: Int
        let headDimension: Int?
        let maxPositionEmbeddings: Int
        let useSlidingWindow: Bool?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case architectures
            case hiddenSize = "hidden_size"
            case layerCount = "num_hidden_layers"
            case queryHeadCount = "num_attention_heads"
            case kvHeadCount = "num_key_value_heads"
            case headDimension = "head_dim"
            case maxPositionEmbeddings = "max_position_embeddings"
            case useSlidingWindow = "use_sliding_window"
        }
    }

    /// The compressed-attention KL contract measures one checkpoint with two cache paths.
    /// A distinct reference checkpoint would confound weight and cache loss, and the current
    /// evidence schema has no independently predeclared full-content digest for that reference.
    /// Reject it before either model is loaded or teacher-forced scoring begins.
    public static func validateKLReferenceModel(
        isSameResolvedModel: Bool,
        request: CompressedKVAttentionRequest?
    ) throws {
        guard request == nil || isSameResolvedModel else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .referenceModelIdentityMismatch
        }
    }

    public static func load(
        sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot
    ) throws -> Self {
        let data = sourceSnapshot.exactModelConfigData
        let root: [String: Any]
        let configuration: ModelConfiguration
        do {
            guard let decodedRoot = try JSONSerialization.jsonObject(with: data)
                as? [String: Any]
            else {
                throw CompressedKVAttentionRuntimeAdmissionError
                    .malformedModelConfig
            }
            root = decodedRoot
            configuration = try JSONDecoder().decode(
                ModelConfiguration.self, from: data)
        } catch let error as CompressedKVAttentionRuntimeAdmissionError {
            throw error
        } catch {
            throw CompressedKVAttentionRuntimeAdmissionError
                .malformedModelConfig
        }

        for key in [
            "sliding_window", "attention_sinks", "attention_sink_size",
            "num_sink_tokens", "sinks", "text_config",
        ] where root[key].map({ !($0 is NSNull) }) == true {
            throw CompressedKVAttentionRuntimeAdmissionError
                .unsupportedAttentionFeature(key)
        }
        if configuration.useSlidingWindow == true {
            throw CompressedKVAttentionRuntimeAdmissionError
                .unsupportedAttentionFeature("use_sliding_window")
        }

        let family: CompressedKVAttentionModelFamily
        let expectedArchitecture: String
        switch configuration.modelType {
        case "qwen3":
            family = .qwen3
            expectedArchitecture = "Qwen3ForCausalLM"
        case "llama":
            family = .llama
            expectedArchitecture = "LlamaForCausalLM"
        default:
            throw CompressedKVAttentionRuntimeAdmissionError
                .unsupportedModelType(configuration.modelType)
        }
        guard configuration.architectures.count == 1,
            let architecture = configuration.architectures.first
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .malformedModelConfig
        }
        guard architecture == expectedArchitecture else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .unsupportedArchitecture(architecture)
        }

        let headDimension: Int
        if let explicit = configuration.headDimension {
            headDimension = explicit
        } else {
            guard configuration.hiddenSize > 0,
                configuration.queryHeadCount > 0,
                configuration.hiddenSize.isMultiple(
                    of: configuration.queryHeadCount)
            else {
                throw CompressedKVAttentionRuntimeAdmissionError
                    .invalidGeometry("headDimension")
            }
            headDimension = configuration.hiddenSize
                / configuration.queryHeadCount
        }

        guard configuration.hiddenSize > 0,
            configuration.layerCount > 0,
            configuration.layerCount <= Int(Int32.max),
            configuration.queryHeadCount > 0,
            configuration.queryHeadCount <= Int(Int32.max),
            configuration.kvHeadCount > 0,
            configuration.kvHeadCount <= configuration.queryHeadCount,
            configuration.queryHeadCount.isMultiple(
                of: configuration.kvHeadCount),
            headDimension > 0,
            headDimension <= Int(Int32.max),
            configuration.maxPositionEmbeddings > 0,
            configuration.maxPositionEmbeddings <= Int(Int32.max)
        else {
            let field = configuration.queryHeadCount > 0
                    && configuration.kvHeadCount > 0
                    && !configuration.queryHeadCount.isMultiple(
                        of: configuration.kvHeadCount)
                ? "queryHeadCount" : "denseGQAGeometry"
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidGeometry(field)
        }

        return Self(
            family: family,
            modelType: configuration.modelType,
            architecture: architecture,
            modelConfigHash: fnv1a64(data),
            modelConfigSHA256: sha256Hex(data),
            checkpointManifestHash: sourceSnapshot.checkpointManifestHash,
            checkpointContentSHA256:
                sourceSnapshot.checkpointContentSHA256,
            tokenizerSHA256: sourceSnapshot.tokenizerSHA256,
            layerCount: configuration.layerCount,
            queryHeadCount: configuration.queryHeadCount,
            kvHeadCount: configuration.kvHeadCount,
            headDimension: headDimension,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings)
    }

    public func validateAffineGeometry(
        keyGroupSize: Int,
        valueGroupSize: Int
    ) throws {
        guard keyGroupSize > 0,
            valueGroupSize > 0,
            headDimension.isMultiple(of: keyGroupSize),
            headDimension.isMultiple(of: valueGroupSize)
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .unsupportedGroupGeometry(
                    headDimension: headDimension,
                    keyGroupSize: keyGroupSize,
                    valueGroupSize: valueGroupSize)
        }
    }

    public func validateCheckpointContentIdentity(
        _ expectedSHA256: String
    ) throws {
        guard expectedSHA256.utf8.count == 64,
            expectedSHA256.utf8.allSatisfy({
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }),
            checkpointContentSHA256 == expectedSHA256
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .checkpointContentIdentityMismatch
        }
    }

    public func validateScheduleIdentity(
        modelConfigHash: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        layerCount: Int,
        groupSize: Int
    ) throws {
        guard self.modelConfigHash == modelConfigHash,
            self.checkpointManifestHash == checkpointManifestHash,
            self.tokenizerSHA256 == tokenizerSHA256,
            self.layerCount == layerCount
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .scheduleIdentityMismatch
        }
        do {
            try validateAffineGeometry(
                keyGroupSize: groupSize,
                valueGroupSize: groupSize)
        } catch {
            throw CompressedKVAttentionRuntimeAdmissionError
                .scheduleIdentityMismatch
        }
    }

    /// Revalidate the scalar admission after `Codable` decoding. The live loader proves the
    /// hashes from exact source bytes; this method prevents a durable row from bypassing the
    /// loader's format/family/geometry invariants by relying on synthesized decoding alone.
    @discardableResult
    public func validatedForEvidence() throws -> Self {
        func isLowercaseHex(_ value: String, lengths: Set<Int>) -> Bool {
            lengths.contains(value.utf8.count)
                && value.utf8.allSatisfy {
                    (48 ... 57).contains($0) || (97 ... 102).contains($0)
                }
        }
        let expectedIdentity: (
            family: CompressedKVAttentionModelFamily,
            modelType: String,
            architecture: String
        )
        switch family {
        case .qwen3:
            expectedIdentity = (.qwen3, "qwen3", "Qwen3ForCausalLM")
        case .llama:
            expectedIdentity = (.llama, "llama", "LlamaForCausalLM")
        }
        guard family == expectedIdentity.family,
            modelType == expectedIdentity.modelType,
            architecture == expectedIdentity.architecture,
            isLowercaseHex(modelConfigHash, lengths: [16]),
            isLowercaseHex(modelConfigSHA256, lengths: [64]),
            isLowercaseHex(checkpointManifestHash, lengths: [16, 64]),
            isLowercaseHex(checkpointContentSHA256, lengths: [64]),
            isLowercaseHex(tokenizerSHA256, lengths: [64]),
            layerCount > 0,
            queryHeadCount > 0,
            kvHeadCount > 0,
            kvHeadCount <= queryHeadCount,
            queryHeadCount.isMultiple(of: kvHeadCount),
            headDimension > 0,
            maxPositionEmbeddings > 0
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidSourceIdentity
        }
        return self
    }
}

/// Path-free, durable proof that an explicit request executed the matching attention operation
/// against one authenticated checkpoint configuration. A requested operation is never accepted
/// as evidence of itself; callers must construct this from post-forward cache telemetry.
public struct CompressedKVAttentionRuntimeBinding:
    Codable, Equatable, Hashable, Sendable
{
    public let request: CompressedKVAttentionRequest
    public let observedOperation:
        CompressedKVAttentionObservedOperation
    public let admission: CompressedKVAttentionRuntimeAdmission

    public init(
        request: CompressedKVAttentionRequest,
        observedOperation:
            CompressedKVAttentionObservedOperation,
        admission: CompressedKVAttentionRuntimeAdmission
    ) throws {
        let expected: CompressedKVAttentionObservedOperation
        switch request {
        case .materialize:
            expected = .materializedKV
        case .splitAffineQuantizedMM:
            expected = .splitQuantizedMM
        }
        guard observedOperation == expected else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .attentionOperationMismatch
        }
        self.request = request
        self.observedOperation = observedOperation
        self.admission = admission
    }

    /// Re-run the constructor invariant after decoding durable evidence. `Codable` synthesis
    /// necessarily bypasses the throwing initializer, so promotion validators must call this
    /// before trusting request/operation agreement from an artifact.
    @discardableResult
    public func validated() throws -> Self {
        try admission.validatedForEvidence()
        _ = try Self(
            request: request,
            observedOperation: observedOperation,
            admission: admission)
        return self
    }
}
