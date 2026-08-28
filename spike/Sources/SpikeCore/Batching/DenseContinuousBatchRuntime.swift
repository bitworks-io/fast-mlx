import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Unforgeable outside SpikeCore: created only by reading the model's checked-in config.
public struct DenseContinuousBatchModelProof: Sendable, Equatable {
    fileprivate let modelFamily: CompressedKVAttentionModelFamily
    fileprivate let modelConfigHash: String
    fileprivate let modelConfigSHA256: String
    fileprivate let checkpointManifestHash: String?
    fileprivate let checkpointContentSHA256: String?
    fileprivate let tokenizerSHA256: String?
    fileprivate let maxPositionEmbeddings: Int
    fileprivate let vocabularySize: Int
    fileprivate let layerCount: Int
    fileprivate let keyValueHeadCount: Int
    fileprivate let headDimension: Int
    fileprivate let elementBytes: Int
    /// The heterogeneous per-layer cache geometry for the hybrid-linear family (`.qwen35`); `nil` for the
    /// uniform-KV family (`.qwen3`). Carried in the proof so the config-hash-pinned derivation is proven
    /// once and never re-derived outside the unforgeable boundary.
    fileprivate let hybridGeometry: HybridCacheGeometry?

    public var verifiedModelFamily: CompressedKVAttentionModelFamily {
        modelFamily
    }

    /// The hybrid per-layer cache geometry, present iff `verifiedModelFamily == .qwen35`.
    public var verifiedHybridGeometry: HybridCacheGeometry? {
        hybridGeometry
    }

    public var modelConfigurationSHA256: String {
        modelConfigSHA256
    }

    public var maximumContextTokens: Int {
        maxPositionEmbeddings
    }

    public var verifiedLayerCount: Int {
        layerCount
    }

    public var verifiedKeyValueHeadCount: Int {
        keyValueHeadCount
    }

    public var verifiedHeadDimension: Int {
        headDimension
    }

    /// Minimal lenient decode of just the family tag, so an unsupported family is
    /// identified before the strict geometry `Configuration` decode (which fails on
    /// VL-wrapped hybrids whose fields live under `text_config`).
    private struct ModelTypeProbe: Decodable {
        let modelType: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    private struct Configuration: Decodable {
        let modelType: String
        let maxPositionEmbeddings: Int
        let vocabularySize: Int
        let layerCount: Int?
        let keyValueHeadCount: Int?
        let headDimension: Int?
        let torchDtype: String?
        let dtype: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabularySize = "vocab_size"
            case layerCount = "num_hidden_layers"
            case keyValueHeadCount = "num_key_value_heads"
            case headDimension = "head_dim"
            case torchDtype = "torch_dtype"
            case dtype
        }
    }

    private init(
        modelFamily: CompressedKVAttentionModelFamily,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String?,
        checkpointContentSHA256: String?,
        tokenizerSHA256: String?,
        maxPositionEmbeddings: Int,
        vocabularySize: Int,
        layerCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        elementBytes: Int,
        hybridGeometry: HybridCacheGeometry? = nil
    ) {
        self.modelFamily = modelFamily
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.vocabularySize = vocabularySize
        self.layerCount = layerCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        self.elementBytes = elementBytes
        self.hybridGeometry = hybridGeometry
    }

    public static func verifying(
        modelDirectory: URL,
        stableCompressedSource: CompressedKVAttentionRuntimeSourceSnapshot? = nil,
        allowHybridQwen35: Bool = false
    ) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        // Determine the family BEFORE the strict geometry decode. A valid config for an
        // unsupported family must fail as unsupportedModelFamily, not invalidModelConfiguration:
        // hybrid families (e.g. qwen3_5) ship VL-wrapped with the geometry under `text_config`,
        // so the strict Configuration decode fails on absent top-level fields first and would
        // mask the family — defeating the continuous→scalar serving fallback that keys on
        // unsupportedModelFamily. A config with no readable `model_type` is genuinely broken and
        // falls through to invalidModelConfiguration below.
        if let probe = try? JSONDecoder().decode(ModelTypeProbe.self, from: data),
            probe.modelType != "qwen3"
        {
            // Flag-gated hybrid-linear (qwen3_5) admission (design §2.4). OFF by default everywhere, so
            // the scalar fallback that keys on `unsupportedModelFamily` keeps working; ON, the proof
            // carries the config-hash-pinned `HybridCacheGeometry`. Any other family still throws here.
            if probe.modelType == "qwen3_5", allowHybridQwen35 {
                return try hybridQwen35Proof(
                    data: data, stableCompressedSource: stableCompressedSource)
            }
            throw DenseContinuousBatchRuntimeError.unsupportedModelFamily(probe.modelType)
        }
        let configuration: Configuration
        do {
            configuration = try JSONDecoder().decode(
                Configuration.self, from: data)
        } catch {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        if let stableCompressedSource,
            stableCompressedSource.exactModelConfigData != data
        {
            throw DenseContinuousBatchRuntimeError.compressedBatchSourceProofMismatch
        }
        // Redundant after the family probe above, kept as defense in depth.
        guard configuration.modelType == "qwen3" else {
            throw DenseContinuousBatchRuntimeError.unsupportedModelFamily(
                configuration.modelType)
        }
        guard let layerCount = configuration.layerCount,
            let keyValueHeadCount = configuration.keyValueHeadCount,
            let headDimension = configuration.headDimension
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        let elementBytes: Int
        switch configuration.torchDtype ?? configuration.dtype {
        case "float16", "bfloat16": elementBytes = 2
        case "float32": elementBytes = 4
        default: throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        guard configuration.maxPositionEmbeddings > 0,
            configuration.maxPositionEmbeddings <= Int(Int32.max),
            configuration.vocabularySize > 0,
            configuration.vocabularySize <= Int(Int32.max),
            layerCount > 0,
            keyValueHeadCount > 0,
            headDimension > 0
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        return Self(
            modelFamily: .qwen3,
            modelConfigHash: fnv1a64(data),
            modelConfigSHA256: sha256Hex(data),
            checkpointManifestHash: stableCompressedSource?.checkpointManifestHash,
            checkpointContentSHA256: stableCompressedSource?.checkpointContentSHA256,
            tokenizerSHA256: stableCompressedSource?.tokenizerSHA256,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings,
            vocabularySize: configuration.vocabularySize,
            layerCount: layerCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            elementBytes: elementBytes)
    }

    /// Build the flag-gated hybrid-linear (`qwen3_5`) proof. Derivation is admission-grade fail-closed
    /// (any config defect → `invalidModelConfiguration`); the config bytes are hash-pinned exactly as the
    /// qwen3 path, and the max-context/vocab guards mirror it. The scalar fields (layerCount, kvHeads,
    /// headDim, elementBytes) come from the derived geometry so there is one authority for them.
    private static func hybridQwen35Proof(
        data: Data,
        stableCompressedSource: CompressedKVAttentionRuntimeSourceSnapshot?
    ) throws -> Self {
        let geometry: HybridCacheGeometry
        do {
            geometry = try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data)
        } catch {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        // NOTE: the real gated-delta Metal kernel requires the linear key head dim (Dk) to be a
        // multiple of 32 (GatedDelta.swift:29, `n_per_t = Dk / 32`), but that check is DELIBERATELY NOT
        // here. This proof is shared with the toy hybrid runtime tests, which serve a fp32 in-Swift toy
        // model (no Metal kernel) at Dk=1. The kernel-viability guard therefore lives in the REAL serving
        // adapter (`loadContinuousServingModel`), which the toy runtime construction bypasses.
        // max_position_embeddings / vocab_size live under text_config on the VL wrapper (root fallback).
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        let textScope = (root["text_config"] as? [String: Any]) ?? [:]
        func scalar(_ key: String) -> Int? {
            func asInt(_ v: Any?) -> Int? {
                if let i = v as? Int { return i }
                if let n = v as? NSNumber { return Int(exactly: n.doubleValue) }
                if let d = v as? Double { return Int(exactly: d) }
                return nil
            }
            return asInt(textScope[key]) ?? asInt(root[key])
        }
        guard let maxPositionEmbeddings = scalar("max_position_embeddings"),
            let vocabularySize = scalar("vocab_size"),
            maxPositionEmbeddings > 0, maxPositionEmbeddings <= Int(Int32.max),
            vocabularySize > 0, vocabularySize <= Int(Int32.max)
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        if let stableCompressedSource,
            stableCompressedSource.exactModelConfigData != data
        {
            throw DenseContinuousBatchRuntimeError.compressedBatchSourceProofMismatch
        }
        return Self(
            modelFamily: .qwen35,
            modelConfigHash: fnv1a64(data),
            modelConfigSHA256: sha256Hex(data),
            checkpointManifestHash: stableCompressedSource?.checkpointManifestHash,
            checkpointContentSHA256: stableCompressedSource?.checkpointContentSHA256,
            tokenizerSHA256: stableCompressedSource?.tokenizerSHA256,
            maxPositionEmbeddings: maxPositionEmbeddings,
            vocabularySize: vocabularySize,
            layerCount: geometry.map.layerCount,
            keyValueHeadCount: geometry.dense.kvHeads,
            headDimension: geometry.dense.headDim,
            elementBytes: geometry.dense.elementBytes,
            hybridGeometry: geometry)
    }

    static func testing(
        maxPositionEmbeddings: Int,
        vocabularySize: Int,
        modelConfigHash: String = "0000000000000000",
        modelConfigSHA256: String = String(repeating: "0", count: 64),
        checkpointManifestHash: String? = nil,
        checkpointContentSHA256: String? = nil,
        tokenizerSHA256: String? = nil,
        layerCount: Int = 1,
        keyValueHeadCount: Int = 1,
        headDimension: Int = 1,
        elementBytes: Int = 4
    ) -> Self {
        Self(
            modelFamily: .qwen3,
            modelConfigHash: modelConfigHash,
            modelConfigSHA256: modelConfigSHA256,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256,
            maxPositionEmbeddings: maxPositionEmbeddings,
            vocabularySize: vocabularySize,
            layerCount: layerCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            elementBytes: elementBytes)
    }
}

public enum DenseContinuousBatchRuntimeError: Error, Equatable {
    case unsupportedModelFamily(String)
    case invalidModelConfiguration
    case invalidAllocationChunk(Int)
    case invalidContextLimit(Int)
    case invalidAggregateContextLimit(Int)
    case invalidAggregateKVByteLimit(Int)
    case invalidInitialDecodeReserve(Int)
    case noCacheLayers
    case unknownRequest(BatchRequestID)
    case outOfOrderPrefill(BatchRequestID, expected: Int, actual: Int)
    case invalidPrefillMetadata(BatchRequestID)
    case prefillAfterFinalization(BatchRequestID)
    case decodeBeforeFinalPrefill(BatchRequestID)
    case drainWithoutPendingLookahead(BatchRequestID)
    case pendingLookaheadInBatch(BatchRequestID)
    case speculativeDrainRequired(BatchRequestID)
    case speculationStateMismatch(BatchRequestID)
    case invalidBatchMembership([BatchRequestID])
    case incompatibleDecodeCohort([BatchRequestID])
    case speculationUnsupported
    case cacheLayerMismatch(BatchRequestID, expected: Int, actual: Int)
    case cacheLengthMismatch(BatchRequestID, expected: Int, actual: Int)
    case contextLimitExceeded(BatchRequestID, requested: Int, limit: Int)
    case requestReservedContextLimitExceeded(
        BatchRequestID,
        requested: Int,
        limit: Int)
    case requestReservedKVByteLimitExceeded(
        BatchRequestID,
        requested: Int,
        limit: Int)
    case aggregateContextLimitExceeded(requested: Int, limit: Int)
    case aggregateKVByteLimitExceeded(requested: Int, limit: Int)
    case kvByteAccountingOverflow
    case modelLayerCountMismatch(expected: Int, actual: Int)
    case cacheGeometryMismatch(expectedHeads: Int, expectedDimension: Int, actual: [Int])
    case cacheElementSizeMismatch(expected: Int, actual: Int)
    case compressedBatchAdmissionRequired
    case compressedBatchAdmissionMismatch
    case compressedBatchSourceProofMismatch
    case unsupportedCompressedBatchCache
    case invalidTokenID(BatchRequestID, Int)
    case positionOverflow(BatchRequestID)
    case speculationTelemetryOverflow
}

struct DenseContinuousBatchRuntimeDiagnostics: Equatable {
    let batchTraceCount: Int
    let batchMembership: [BatchRequestID]
    let batchCapacity: Int?
    let batchPhysicalWrittenEnd: Int?
    let kvBytesPerToken: Int
    let reservedKVBytes: Int
    let maxReservedKVBytes: Int
}

/// Actor-confined dense MLX executor for `ContinuousBatchCoordinator`.
///
/// This type is intentionally not `Sendable`. The coordinator's `sending` initializer
/// transfers the runtime, model, cache arrays, and compiled closures into one actor isolation
/// region. Only value-typed request work and decode results cross that boundary.
public final class DenseContinuousBatchRuntime: ContinuousBatchRuntime {
    private typealias Step = ([MLXArray]) -> [MLXArray]

    private final class TraceCounter {
        var count = 0
    }

    private struct Slot {
        let incarnation: UUID
        let totalPromptTokens: Int
        let maxOutputTokens: Int
        var processedTokens: Int
        var cachedTokens: Int
        var emittedTokens: Int
        var prefillComplete: Bool
        let decodeCohort: BatchDecodeCohort
        var soloPipelineState: BatchSoloPipelineState
        var pldSession: IncrementalPLDSession?
        // Kind-agnostic rows (S3): dense layers hold `ContinuousScalarKVCache` rows, recurrent
        // layers hold the vendored `MambaCache` itself. Dense-only members (capacity/truncate/
        // geometry) are reached by keying off the geometry map's dense layer indices — layer 0
        // is NOT dense on qwen3_5.
        var scalarCaches: [any ContinuousScalarRowCache]?
        var stagedToken: MLXArray?
        var scalarStep: Step?
        var scalarVerifySteps: [Int: Step]
        var scalarTraceCounter: TraceCounter?
    }

    private struct MaterializedSlot {
        let caches: [any ContinuousScalarRowCache]
        let stagedToken: MLXArray
    }

    private struct BatchState {
        let ids: [BatchRequestID]
        let incarnations: [UUID]
        let caches: [any ContinuousBatchedRowCache]
        var stagedTokens: MLXArray
        var step: Step?
        let traceCounter: TraceCounter
    }

    private let model: any LanguageModel
    private let allocationChunk: Int
    private let maxContextTokens: Int
    private let maxReservedContextTokens: Int
    private let initialDecodeReserve: Int
    private let vocabularySize: Int
    private let layerCount: Int
    private let kvBytePlan: any ContinuousKVByteAdmissionPlanning
    private let cacheFamily: ContinuousBatchKVCacheFamily
    private let expectedKVHeads: Int
    private let expectedKVHeadDimension: Int
    private let expectedKVElementBytes: Int
    private let maxReservedKVBytes: Int
    private let soloPLDConfiguration: SpecDecodeConfig?
    private var slots: [BatchRequestID: Slot] = [:]
    private var speculativePromptTokens: [BatchRequestID: [Int]] = [:]
    private var contextReservations: [BatchRequestID: Int] = [:]
    private var kvCapacityReservations: [BatchRequestID: Int] = [:]
    private var reservedKVBytes = 0
    private var cacheGeometryCalibrated = false
    private var batch: BatchState?
    private var speculationRequestedRequests = 0
    private var speculationDraftedTokens = 0
    private var speculationAcceptedDraftTokens = 0
    private var speculationVerificationRounds = 0
    private var speculationFallbackRounds = 0

    package init(
        model: any LanguageModel,
        verifiedBy proof: DenseContinuousBatchModelProof,
        allocationChunk: Int = 256,
        maxContextTokens: Int? = nil,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil,
        kvCacheKind: KVCacheKind = .fp16,
        affineAttentionMode: AffineKVAttentionMode = .materialize,
        compressedKVAttentionAdmission: CompressedKVAttentionRuntimeAdmission? = nil,
        soloPLDConfiguration: SpecDecodeConfig? = nil
    ) throws {
        guard allocationChunk > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAllocationChunk(allocationChunk)
        }
        let resolvedContextLimit = maxContextTokens ?? proof.maxPositionEmbeddings
        guard resolvedContextLimit > 0,
            resolvedContextLimit <= proof.maxPositionEmbeddings,
            resolvedContextLimit <= Int(Int32.max)
        else {
            throw DenseContinuousBatchRuntimeError.invalidContextLimit(resolvedContextLimit)
        }
        guard allocationChunk <= resolvedContextLimit else {
            throw DenseContinuousBatchRuntimeError.invalidAllocationChunk(allocationChunk)
        }
        guard initialDecodeReserve > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidInitialDecodeReserve(
                initialDecodeReserve)
        }
        guard soloPLDConfiguration == nil || kvCacheKind.supportsSpecDecode else {
            throw DenseContinuousBatchRuntimeError.speculationUnsupported
        }
        let resolvedReservationLimit = maxReservedContextTokens ?? resolvedContextLimit
        guard resolvedReservationLimit > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAggregateContextLimit(
                resolvedReservationLimit)
        }
        // Hybrid (heterogeneous dense/recurrent) construction guards. Placed BEFORE the
        // compressed-admission validation so a compressed kind on a hybrid proof reports the
        // hybrid-incompatibility identity (`.unsupportedCompressedBatchCache`), not the
        // dense-path `.compressedBatchAdmissionRequired`. Hybrid runs fp16-only (the recurrent
        // state has no compressed representation), and speculation is physically incompatible
        // with fixed-size recurrent state (dense KV rows can truncate/rollback; SSM state cannot
        // rewind), so both are rejected fail-closed at construction.
        let hybridGeometry = proof.verifiedHybridGeometry
        if hybridGeometry != nil {
            guard kvCacheKind == .fp16 else {
                throw DenseContinuousBatchRuntimeError.unsupportedCompressedBatchCache
            }
            guard soloPLDConfiguration == nil else {
                throw DenseContinuousBatchRuntimeError.speculationUnsupported
            }
        }
        try Self.validateCompressedBatchAdmission(
            cacheKind: kvCacheKind,
            proof: proof,
            resolvedContextLimit: resolvedContextLimit,
            admission: compressedKVAttentionAdmission)
        let layerCount = model.newCache(parameters: nil).count
        guard layerCount > 0 else {
            throw DenseContinuousBatchRuntimeError.noCacheLayers
        }
        guard layerCount == proof.layerCount else {
            throw DenseContinuousBatchRuntimeError.modelLayerCountMismatch(
                expected: proof.layerCount, actual: layerCount)
        }
        let cacheFamily: ContinuousBatchKVCacheFamily
        let kvBytePlan: any ContinuousKVByteAdmissionPlanning
        if let hybridGeometry {
            // Hybrid path: the family is selected from the verified geometry (there is no hybrid
            // `KVCacheKind`), and admission is charged dense-layers-only per token plus a fixed
            // per-row recurrent term — never the uniform dense plan, which would over-charge every
            // token as if all layers grew a KV buffer and starve admission (see HybridKVByteAdmission).
            cacheFamily = ContinuousBatchKVCacheFamily(hybridGeometry: hybridGeometry)
            do {
                kvBytePlan = try HybridKVByteAdmissionPlan(
                    geometry: hybridGeometry,
                    allocationChunk: allocationChunk,
                    maxContextTokens: resolvedContextLimit)
            } catch {
                throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
            }
        } else {
            do {
                cacheFamily = try ContinuousBatchKVCacheFamily(
                    cacheKind: kvCacheKind,
                    layerCount: layerCount,
                    affineAttentionMode: affineAttentionMode)
            } catch {
                throw DenseContinuousBatchRuntimeError.unsupportedCompressedBatchCache
            }
            do {
                if let configurations = cacheFamily.affineConfigurations {
                    kvBytePlan = try AffineKVByteAdmissionPlan(
                        configurations: configurations,
                        keyValueHeadCount: proof.keyValueHeadCount,
                        headDimension: proof.headDimension,
                        metadataScalarBytes: proof.elementBytes,
                        allocationChunk: allocationChunk,
                        maxContextTokens: resolvedContextLimit)
                } else {
                    kvBytePlan = try DenseKVByteAdmissionPlan(
                        layerCount: proof.layerCount,
                        keyValueHeadCount: proof.keyValueHeadCount,
                        headDimension: proof.headDimension,
                        elementBytes: proof.elementBytes,
                        allocationChunk: allocationChunk,
                        maxContextTokens: resolvedContextLimit)
                }
            } catch {
                throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
            }
        }
        let resolvedKVByteLimit = maxReservedKVBytes ?? Int.max
        guard resolvedKVByteLimit > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAggregateKVByteLimit(
                resolvedKVByteLimit)
        }
        self.model = model
        self.allocationChunk = allocationChunk
        self.maxContextTokens = resolvedContextLimit
        self.maxReservedContextTokens = resolvedReservationLimit
        self.initialDecodeReserve = initialDecodeReserve
        self.vocabularySize = proof.vocabularySize
        self.layerCount = layerCount
        self.kvBytePlan = kvBytePlan
        self.cacheFamily = cacheFamily
        self.expectedKVHeads = proof.keyValueHeadCount
        self.expectedKVHeadDimension = proof.headDimension
        self.expectedKVElementBytes = proof.elementBytes
        self.maxReservedKVBytes = resolvedKVByteLimit
        self.soloPLDConfiguration = soloPLDConfiguration
    }

    convenience init(
        testing model: any LanguageModel,
        allocationChunk: Int = 256,
        maxContextTokens: Int = 32_768,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil,
        kvCacheKind: KVCacheKind = .fp16,
        affineAttentionMode: AffineKVAttentionMode = .materialize,
        compressedKVAttentionAdmission: CompressedKVAttentionRuntimeAdmission? = nil,
        soloPLDConfiguration: SpecDecodeConfig? = nil,
        layerCount: Int = 1,
        keyValueHeadCount: Int = 1,
        headDimension: Int = 1,
        elementBytes: Int = 4
    ) throws {
        try self.init(
            model: model,
            verifiedBy: .testing(
                maxPositionEmbeddings: maxContextTokens,
                vocabularySize: 2_048,
                modelConfigHash: compressedKVAttentionAdmission?.modelConfigHash
                    ?? "0000000000000000",
                modelConfigSHA256: compressedKVAttentionAdmission?.modelConfigSHA256
                    ?? String(repeating: "0", count: 64),
                checkpointManifestHash:
                    compressedKVAttentionAdmission?.checkpointManifestHash,
                checkpointContentSHA256:
                    compressedKVAttentionAdmission?.checkpointContentSHA256,
                tokenizerSHA256: compressedKVAttentionAdmission?.tokenizerSHA256,
                layerCount: layerCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension,
                elementBytes: elementBytes),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            maxReservedContextTokens: maxReservedContextTokens,
            initialDecodeReserve: initialDecodeReserve,
            maxReservedKVBytes: maxReservedKVBytes,
            kvCacheKind: kvCacheKind,
            affineAttentionMode: affineAttentionMode,
            compressedKVAttentionAdmission: compressedKVAttentionAdmission,
            soloPLDConfiguration: soloPLDConfiguration)
    }

    private static func validateCompressedBatchAdmission(
        cacheKind: KVCacheKind,
        proof: DenseContinuousBatchModelProof,
        resolvedContextLimit: Int,
        admission: CompressedKVAttentionRuntimeAdmission?
    ) throws {
        switch cacheKind {
        case .fp16:
            return
        case .turboQuant, .kvarn, .kvtunerCandidate:
            throw DenseContinuousBatchRuntimeError.unsupportedCompressedBatchCache
        case .affine, .kvtuner:
            break
        }

        guard let admission else {
            throw DenseContinuousBatchRuntimeError.compressedBatchAdmissionRequired
        }
        do {
            try admission.validatedForEvidence()
        } catch {
            throw DenseContinuousBatchRuntimeError.compressedBatchAdmissionMismatch
        }
        guard admission.family == proof.modelFamily,
            admission.modelConfigHash == proof.modelConfigHash,
            admission.modelConfigSHA256 == proof.modelConfigSHA256,
            admission.checkpointManifestHash == proof.checkpointManifestHash,
            admission.checkpointContentSHA256 == proof.checkpointContentSHA256,
            admission.tokenizerSHA256 == proof.tokenizerSHA256,
            admission.layerCount == proof.layerCount,
            admission.kvHeadCount == proof.keyValueHeadCount,
            admission.headDimension == proof.headDimension,
            admission.maxPositionEmbeddings == proof.maxPositionEmbeddings,
            resolvedContextLimit <= admission.maxPositionEmbeddings
        else {
            throw DenseContinuousBatchRuntimeError.compressedBatchAdmissionMismatch
        }

        do {
            switch cacheKind {
            case .affine(let tier):
                try admission.validateAffineGeometry(
                    keyGroupSize: tier.configuration.keyGroupSize,
                    valueGroupSize: tier.configuration.valueGroupSize)
            case .kvtuner(let selection):
                try admission.validateScheduleIdentity(
                    modelConfigHash: selection.modelConfigHash,
                    modelConfigSHA256: selection.modelConfigSHA256,
                    checkpointManifestHash: selection.checkpointManifestHash,
                    checkpointContentSHA256:
                        selection.checkpointContentSHA256,
                    tokenizerSHA256: selection.tokenizerSHA256,
                    layerCount: selection.layers.count,
                    groupSize: selection.groupSize)
            case .fp16, .turboQuant, .kvarn, .kvtunerCandidate:
                preconditionFailure("compressed batch admission switch changed")
            }
        } catch {
            throw DenseContinuousBatchRuntimeError.compressedBatchAdmissionMismatch
        }
    }

    public func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        var additions: [(BatchRequestID, tokens: Int, capacity: Int)] = []
        var speculativeAdditions: [BatchRequestID: [Int]] = [:]
        additions.reserveCapacity(admissions.count)
        for admission in admissions {
            let submission = admission.submission
            let validated = try validateAdmission(admission)
            if submission.requestsSpeculation {
                speculativeAdditions[admission.id] = submission.promptTokens
            }
            additions.append(
                (
                    admission.id,
                    validated.requestedTokens,
                    validated.capacity
                ))
        }

        if !additions.isEmpty {
            try calibrateCacheGeometryIfNeeded()
        }

        var requested = contextReservations.values.reduce(0, +)
        for (_, tokens, _) in additions {
            let (next, overflow) = requested.addingReportingOverflow(tokens)
            guard !overflow, next <= maxReservedContextTokens else {
                throw DenseContinuousBatchRuntimeError.aggregateContextLimitExceeded(
                    requested: overflow ? Int.max : next,
                    limit: maxReservedContextTokens)
            }
            requested = next
        }
        var candidateCapacities = kvCapacityReservations
        for (id, _, capacity) in additions { candidateCapacities[id] = capacity }
        let requestedKVBytes: Int
        do {
            requestedKVBytes = try kvBytePlan.transitionEnvelopeBytes(
                capacities: Array(candidateCapacities.values))
        } catch {
            throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
        }
        guard requestedKVBytes <= maxReservedKVBytes else {
            throw DenseContinuousBatchRuntimeError.aggregateKVByteLimitExceeded(
                requested: requestedKVBytes, limit: maxReservedKVBytes)
        }
        let (nextSpeculationRequests, speculationRequestOverflow) =
            speculationRequestedRequests.addingReportingOverflow(
                speculativeAdditions.count)
        guard !speculationRequestOverflow else {
            throw DenseContinuousBatchRuntimeError.speculationTelemetryOverflow
        }
        for (id, tokens, capacity) in additions {
            contextReservations[id] = tokens
            kvCapacityReservations[id] = capacity
        }
        for (id, promptTokens) in speculativeAdditions {
            speculativePromptTokens[id] = promptTokens
        }
        speculationRequestedRequests = nextSpeculationRequests
        reservedKVBytes = requestedKVBytes
    }

    public func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        let submission = admission.submission
        _ = try validateAdmission(admission)
        return try decodeCohort(
            id: admission.id,
            promptTokens: submission.promptTokens.count,
            maxOutputTokens: submission.maxOutputTokens)
    }

    public func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        guard !work.tokens.isEmpty,
            work.startToken >= 0,
            work.totalPromptTokens > 0,
            work.maxOutputTokens > 0,
            work.startToken <= work.totalPromptTokens,
            work.tokens.count <= work.totalPromptTokens - work.startToken
        else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
        }
        try validateRequest(
            id: work.id,
            promptTokens: work.tokens,
            totalPromptTokens: work.totalPromptTokens,
            maxOutputTokens: work.maxOutputTokens)
        let endToken = work.startToken + work.tokens.count
        guard work.isFinal == (endToken == work.totalPromptTokens) else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
        }
        if let expectedPrompt = speculativePromptTokens[work.id] {
            guard expectedPrompt.count == work.totalPromptTokens,
                Array(expectedPrompt[work.startToken ..< endToken]) == work.tokens
            else {
                throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
            }
        }

        var slot: Slot
        if let existing = slots[work.id] {
            guard !existing.prefillComplete else {
                throw DenseContinuousBatchRuntimeError.prefillAfterFinalization(work.id)
            }
            guard existing.totalPromptTokens == work.totalPromptTokens,
                existing.maxOutputTokens == work.maxOutputTokens
            else {
                throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
            }
            slot = existing
        } else {
            guard work.startToken == 0 else {
                throw DenseContinuousBatchRuntimeError.outOfOrderPrefill(
                    work.id, expected: 0, actual: work.startToken)
            }
            let capacity = try roundedCapacity(
                promptTokens: work.totalPromptTokens,
                outputTokens: min(work.maxOutputTokens, initialDecodeReserve),
                id: work.id)
            let decodeCohort = try decodeCohort(
                id: work.id,
                promptTokens: work.totalPromptTokens,
                maxOutputTokens: work.maxOutputTokens)
            slot = Slot(
                incarnation: UUID(),
                totalPromptTokens: work.totalPromptTokens,
                maxOutputTokens: work.maxOutputTokens,
                processedTokens: 0,
                cachedTokens: 0,
                emittedTokens: 0,
                prefillComplete: false,
                decodeCohort: decodeCohort,
                soloPipelineState: .canonical,
                pldSession: nil,
                scalarCaches: cacheFamily.makeScalarRows(
                    layerCount: layerCount,
                    capacity: capacity),
                stagedToken: nil,
                scalarStep: nil,
                scalarVerifySteps: [:],
                scalarTraceCounter: nil)
        }

        guard work.startToken == slot.processedTokens else {
            throw DenseContinuousBatchRuntimeError.outOfOrderPrefill(
                work.id, expected: slot.processedTokens, actual: work.startToken)
        }
        guard let caches = slot.scalarCaches else {
            throw DenseContinuousBatchRuntimeError.prefillAfterFinalization(work.id)
        }
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.cacheLayerMismatch(
                work.id, expected: layerCount, actual: caches.count)
        }

        let ids = MLXArray(work.tokens.map(Int32.init)).reshaped([1, work.tokens.count])
        let modelCaches: [any KVCache] = caches.map { $0 }
        let logits = model(ids, cache: modelCaches)
        let staged = CompiledMLXDecoder.greedyTokenOrInvalidSentinel(
            logits[0..., -1, 0...])
        eval([staged] + caches.flatMap { $0.innerState() })
        _ = try validatedToken(staged, id: work.id)

        // Recurrent-row offsets are DRIVER-maintained: the vendored `ArraysCache.advance` adjusts
        // only transient lengths/leftPadding and never `BaseKVCache.offset` (S3 finding 3). Prefill
        // is a commit point, so record the committed count here; downstream merging/extraction
        // reads it as the row's authoritative logical length. Dense offsets advance inside the
        // cache update itself and are VALIDATED (not written) by the loop below.
        commitRecurrentRowOffsets(caches, committed: endToken)

        try validatePhysicalCacheGeometry(caches)

        for cache in caches {
            let actual = cache.continuousLogicalOffset
            guard actual == endToken else {
                throw DenseContinuousBatchRuntimeError.cacheLengthMismatch(
                    work.id, expected: endToken, actual: actual)
            }
        }
        slot.processedTokens = endToken
        slot.cachedTokens = endToken
        slot.prefillComplete = work.isFinal
        slot.stagedToken = work.isFinal ? staged : nil
        if work.isFinal,
            let promptTokens = speculativePromptTokens[work.id],
            let configuration = soloPLDConfiguration
        {
            slot.pldSession = try IncrementalPLDSession(
                promptTokens: promptTokens,
                drafter: configuration.drafter,
                maxDraft: configuration.maxDraft,
                lookback: configuration.lookback,
                gate: configuration.gate)
        }
        slots[work.id] = slot
        if work.isFinal {
            speculativePromptTokens[work.id] = nil
        }
    }

    public func decode(_ action: BatchDecodeAction) throws
        -> [ContinuousBatchRuntimeDecodeResult]
    {
        switch action {
        case .solo(let id, let speculationAllowed):
            try ensureScalar(id)
            if speculationAllowed {
                return [try decodeSpeculativeSolo(id)]
            }
            guard slots[id]?.soloPipelineState != .speculative else {
                throw DenseContinuousBatchRuntimeError.speculativeDrainRequired(id)
            }
            let result = try decodeScalar(
                id,
                soloPipelineState: .pipelinedLookahead,
                synchronously: false)
            try recordCanonicalPLDToken(id: id, token: result.tokens[0])
            return [result]

        case .drainSoloPipeline(let id):
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard slot.soloPipelineState.requiresDrain else {
                throw DenseContinuousBatchRuntimeError.drainWithoutPendingLookahead(id)
            }
            try ensureScalar(id)
            if slot.soloPipelineState == .speculative {
                return [try drainSpeculativeSolo(id)]
            }
            let result = try decodeScalar(
                id,
                soloPipelineState: .canonical,
                synchronously: true)
            try recordCanonicalPLDToken(id: id, token: result.tokens[0])
            return [result]

        case .batch(let ids, let speculationAllowed):
            guard !speculationAllowed else {
                throw DenseContinuousBatchRuntimeError.speculationUnsupported
            }
            return try decodeBatch(ids)
        }
    }

    public func remove(_ id: BatchRequestID) {
        contextReservations[id] = nil
        speculativePromptTokens[id] = nil
        slots[id] = nil

        guard let batch else {
            kvCapacityReservations[id] = nil
            recomputeReservedKVBytes()
            return
        }
        let removedBatchedRow = batch.ids.contains(id)
        let hasLiveBatchedRow = batch.ids.enumerated().contains {
            row, physicalID in
            guard batch.incarnations.indices.contains(row),
                let currentSlot = slots[physicalID]
            else {
                return false
            }
            return batch.incarnations[row] == currentSlot.incarnation
        }
        if !hasLiveBatchedRow {
            self.batch = nil
            pruneKVReservationsToLiveSlots()
        } else if !removedBatchedRow {
            kvCapacityReservations[id] = nil
            recomputeReservedKVBytes()
        }
        // A removed row of a still-live batch remains physically allocated until the next
        // membership rebuild. Keep its KV byte reservation until ensureBatch/ensureScalar
        // releases the old batch arrays.
    }

    func diagnostics() -> DenseContinuousBatchRuntimeDiagnostics {
        // Capacity/frontier are dense-KV concepts: key them off the first DENSE layer (layer 0 on
        // uniform families, the geometry map's first dense index on hybrid — never plain `.first`).
        let denseBatchCache = batch.flatMap { try? denseBatchedCache(in: $0.caches) }
        return DenseContinuousBatchRuntimeDiagnostics(
            batchTraceCount: batch?.traceCounter.count ?? 0,
            batchMembership: batch?.ids ?? [],
            batchCapacity: denseBatchCache?.capacity,
            batchPhysicalWrittenEnd: denseBatchCache?.continuousPhysicalWrittenEnd,
            kvBytesPerToken: kvBytePlan.bytesPerToken,
            reservedKVBytes: reservedKVBytes,
            maxReservedKVBytes: maxReservedKVBytes)
    }

    func diagnosticScalarLogicalOffset(
        for id: BatchRequestID
    ) -> Int? {
        slots[id]?.scalarCaches?.first?.continuousLogicalOffset
    }

    public func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: kvBytePlan.bytesPerToken,
            reservedKVBytes: reservedKVBytes,
            maxReservedKVBytes: maxReservedKVBytes,
            speculation: soloPLDConfiguration == nil
                ? nil
                : ContinuousBatchRuntimeSpeculationSnapshot(
                    requestedRequests: speculationRequestedRequests,
                    activeSessions: slots.values.reduce(into: 0) { count, slot in
                        if slot.pldSession != nil { count += 1 }
                    } + speculativePromptTokens.count,
                    draftedTokens: speculationDraftedTokens,
                    acceptedDraftTokens: speculationAcceptedDraftTokens,
                    verificationRounds: speculationVerificationRounds,
                    fallbackRounds: speculationFallbackRounds))
    }

    private func decodeSpeculativeSolo(
        _ id: BatchRequestID
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard let configuration = soloPLDConfiguration,
            let slot = slots[id],
            let session = slot.pldSession
        else {
            throw DenseContinuousBatchRuntimeError.speculationUnsupported
        }

        switch session.cacheInvariant {
        case .awaitingFirstToken:
            guard slot.soloPipelineState == .canonical else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }
            let result = try decodeScalar(
                id,
                soloPipelineState: .pipelinedLookahead,
                synchronously: false)
            try recordCanonicalPLDToken(id: id, token: result.tokens[0])
            return result

        case .pipelined:
            guard slot.soloPipelineState != .speculative else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }

        case .speculative:
            guard slot.soloPipelineState == .speculative else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }
        }

        let remaining = slot.maxOutputTokens - slot.emittedTokens
        guard remaining > 0 else {
            throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                id,
                requested: slot.totalPromptTokens + slot.emittedTokens + 1,
                limit: slot.totalPromptTokens + slot.maxOutputTokens)
        }
        let maximumDraftTokens = max(0, remaining - 1)
        let fixedDraftWidth = configuration.compiledVerify
            && maximumDraftTokens > 0
            ? min(configuration.maxDraft, maximumDraftTokens)
            : nil
        let plan = try session.makeRoundPlan(
            maximumDraftTokens: maximumDraftTokens,
            fixedDraftWidth: fixedDraftWidth)

        switch plan {
        case .plainFromPipeline:
            let gateWasEnabled = session.isGateEnabled
            let result = try decodeScalar(
                id,
                soloPipelineState: .pipelinedLookahead,
                synchronously: false)
            try commitPLDFallback(
                id: id,
                plan: plan,
                emittedToken: result.tokens[0])
            return resultWithObservation(
                result,
                observation: pldLowYieldObservation(
                    gateWasEnabled: gateWasEnabled,
                    session: slots[id]?.pldSession))

        case .fallbackFromSpeculative:
            return try decodeSpeculativeFallback(id: id, plan: plan)

        case .verifyPipelined, .verifySpeculative:
            return try decodeSpeculativeVerification(
                id: id,
                plan: plan,
                compiled: configuration.compiledVerify)
        }
    }

    private func decodeSpeculativeFallback(
        id: BatchRequestID,
        plan: IncrementalPLDRoundPlan
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard var slot = slots[id],
            var session = slot.pldSession,
            let current = slot.stagedToken,
            let caches = slot.scalarCaches,
            case .fallbackFromSpeculative(let plannedLast) = plan,
            slot.soloPipelineState == .speculative,
            session.cacheInvariant == .speculative
        else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        let forwardedLast = try validatedToken(current, id: id)
        guard forwardedLast == plannedLast else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        let gateWasEnabled = session.isGateEnabled
        // Speculative truncate/rollback is a dense-KV-only operation; hybrid rejects speculation
        // at construction (recurrent state cannot rewind), so this narrow is defense in depth.
        let denseCaches = caches.compactMap { $0 as? any ContinuousScalarKVCache }
        guard denseCaches.count == caches.count else {
            throw DenseContinuousBatchRuntimeError.speculationUnsupported
        }

        let previousCachedTokens = slot.cachedTokens
        try ensureScalarCapacity(for: id, slot: &slot, additionalTokens: 2)
        if slot.scalarStep == nil {
            let traceCounter = TraceCounter()
            slot.scalarTraceCounter = traceCounter
            slot.scalarStep = makeScalarStep(
                caches: caches,
                counter: traceCounter)
        }
        var rollbackFallback = true
        defer {
            if rollbackFallback {
                for cache in denseCaches {
                    cache.truncate(to: previousCachedTokens)
                }
            }
        }
        let next = slot.scalarStep!([current])[0]
        let emitted = try validatedToken(next, id: id)
        let following = slot.scalarStep!([next])[0]
        asyncEval(following)

        do {
            try session.commitFallback(plan, emittedToken: emitted)
        } catch {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        try recordSpeculationTelemetry(fallbackRounds: 1)
        slot.cachedTokens += 2
        slot.emittedTokens += 1
        slot.stagedToken = following
        slot.soloPipelineState = .pipelinedLookahead
        slot.pldSession = session
        slots[id] = slot
        rollbackFallback = false
        return ContinuousBatchRuntimeDecodeResult(
            id: id,
            tokens: [emitted],
            finished: false,
            soloPipelineState: .pipelinedLookahead,
            observation: pldLowYieldObservation(
                gateWasEnabled: gateWasEnabled,
                session: session))
    }

    private func decodeSpeculativeVerification(
        id: BatchRequestID,
        plan: IncrementalPLDRoundPlan,
        compiled: Bool
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard var slot = slots[id],
            var session = slot.pldSession,
            let current = slot.stagedToken,
            let caches = slot.scalarCaches
        else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        // Speculative truncate/rollback is a dense-KV-only operation; hybrid rejects speculation
        // at construction (recurrent state cannot rewind), so this narrow is defense in depth.
        let denseCaches = caches.compactMap { $0 as? any ContinuousScalarKVCache }
        guard denseCaches.count == caches.count else {
            throw DenseContinuousBatchRuntimeError.speculationUnsupported
        }

        let draft: [Int]
        let verifyInput: [Int]
        let prefetchedToken: Int?
        switch plan {
        case .verifyPipelined(let proposed):
            guard session.cacheInvariant == .pipelined,
                slot.soloPipelineState != .speculative
            else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }
            let prefetched = try validatedToken(current, id: id)
            draft = proposed
            verifyInput = proposed
            prefetchedToken = prefetched

        case .verifySpeculative(let plannedLast, let proposed):
            guard session.cacheInvariant == .speculative,
                slot.soloPipelineState == .speculative
            else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }
            let forwardedLast = try validatedToken(current, id: id)
            guard forwardedLast == plannedLast else {
                throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
            }
            draft = proposed
            verifyInput = [plannedLast] + proposed
            prefetchedToken = nil

        case .plainFromPipeline, .fallbackFromSpeculative:
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        try validateRuntimeTokens(draft, id: id)
        guard !verifyInput.isEmpty else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        let gateWasEnabled = session.isGateEnabled

        let previousCachedTokens = slot.cachedTokens
        try ensureScalarCapacity(
            for: id,
            slot: &slot,
            additionalTokens: verifyInput.count)
        let ids = MLXArray(verifyInput.map(Int32.init))
            .reshaped([1, verifyInput.count])
        let verifyArgmax: MLXArray
        if compiled {
            if slot.scalarVerifySteps[verifyInput.count] == nil {
                slot.scalarVerifySteps[verifyInput.count] = makeScalarVerifyStep(
                    caches: caches,
                    inputWidth: verifyInput.count)
            }
            verifyArgmax = slot.scalarVerifySteps[verifyInput.count]!([ids])[0]
        } else {
            let modelCaches: [any KVCache] = caches.map { $0 }
            let logits = model(ids, cache: modelCaches)
            verifyArgmax = CompiledMLXDecoder.greedyTokenOrInvalidSentinel(
                logits[0])
        }
        var rollbackVerification = true
        defer {
            if rollbackVerification {
                for cache in denseCaches {
                    cache.truncate(to: previousCachedTokens)
                }
            }
        }
        let picks = verifyArgmax.asType(.int32)
            .asArray(Int32.self).map(Int.init)
        try validateRuntimeTokens(picks, id: id)

        let commit: IncrementalPLDCommit
        do {
            commit = try session.commitVerification(
                plan,
                prefetchedToken: prefetchedToken,
                verifyArgmax: picks)
        } catch {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        slot.cachedTokens += verifyInput.count
        try recordSpeculationTelemetry(
            draftedTokens: draft.count,
            acceptedDraftTokens: commit.acceptedDraftTokens,
            verificationRounds: 1)
        let keep: Int
        let bonusArray: MLXArray
        switch plan {
        case .verifyPipelined:
            keep = previousCachedTokens + commit.acceptedDraftTokens
            bonusArray = commit.acceptedDraftTokens == 0
                ? current
                : verifyArgmax[commit.acceptedDraftTokens - 1].reshaped([1])

        case .verifySpeculative:
            keep = previousCachedTokens + 1 + commit.acceptedDraftTokens
            bonusArray = verifyArgmax[commit.acceptedDraftTokens].reshaped([1])

        case .plainFromPipeline, .fallbackFromSpeculative:
            preconditionFailure("verification plan changed during commit")
        }
        if keep < slot.cachedTokens {
            for cache in denseCaches { cache.truncate(to: keep) }
            slot.cachedTokens = keep
        }
        slot.emittedTokens += commit.emittedTokens.count
        slot.stagedToken = bonusArray
        slot.soloPipelineState = .speculative
        slot.pldSession = session
        slots[id] = slot
        rollbackVerification = false
        return ContinuousBatchRuntimeDecodeResult(
            id: id,
            tokens: commit.emittedTokens,
            finished: false,
            soloPipelineState: .speculative,
            observation: pldLowYieldObservation(
                gateWasEnabled: gateWasEnabled,
                session: session))
    }

    private func pldLowYieldObservation(
        gateWasEnabled: Bool,
        session: IncrementalPLDSession?
    ) -> BatchDecodeObservation {
        guard gateWasEnabled, let session, !session.isGateEnabled else {
            return .none
        }
        return .pldLowYieldDisabled
    }

    private func resultWithObservation(
        _ result: ContinuousBatchRuntimeDecodeResult,
        observation: BatchDecodeObservation
    ) -> ContinuousBatchRuntimeDecodeResult {
        ContinuousBatchRuntimeDecodeResult(
            id: result.id,
            tokens: result.tokens,
            finished: result.finished,
            soloPipelineState: result.soloPipelineState,
            observation: observation)
    }

    private func drainSpeculativeSolo(
        _ id: BatchRequestID
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard var slot = slots[id],
            var session = slot.pldSession,
            let current = slot.stagedToken,
            let caches = slot.scalarCaches,
            slot.soloPipelineState == .speculative,
            session.cacheInvariant == .speculative
        else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        let forwarded = try validatedToken(current, id: id)
        do {
            try session.recordCanonicalDrain(forwardedToken: forwarded)
        } catch {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }

        try ensureScalarCapacity(for: id, slot: &slot, additionalTokens: 1)
        if slot.scalarStep == nil {
            let traceCounter = TraceCounter()
            slot.scalarTraceCounter = traceCounter
            slot.scalarStep = makeScalarStep(
                caches: caches,
                counter: traceCounter)
        }
        let following = slot.scalarStep!([current])[0]
        eval([following] + caches.flatMap { $0.innerState() })

        slot.cachedTokens += 1
        slot.stagedToken = following
        slot.soloPipelineState = .canonical
        slot.pldSession = session
        slots[id] = slot
        return ContinuousBatchRuntimeDecodeResult(
            id: id,
            tokens: [],
            finished: false,
            soloPipelineState: .canonical)
    }

    private func recordCanonicalPLDToken(
        id: BatchRequestID,
        token: Int
    ) throws {
        guard var slot = slots[id], var session = slot.pldSession else {
            return
        }
        do {
            try session.recordCanonicalToken(token)
        } catch {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        slot.pldSession = session
        slots[id] = slot
    }

    private func commitPLDFallback(
        id: BatchRequestID,
        plan: IncrementalPLDRoundPlan,
        emittedToken: Int
    ) throws {
        guard var slot = slots[id], var session = slot.pldSession else {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        do {
            try session.commitFallback(plan, emittedToken: emittedToken)
        } catch {
            throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
        }
        try recordSpeculationTelemetry(fallbackRounds: 1)
        slot.pldSession = session
        slots[id] = slot
    }

    private func recordSpeculationTelemetry(
        draftedTokens: Int = 0,
        acceptedDraftTokens: Int = 0,
        verificationRounds: Int = 0,
        fallbackRounds: Int = 0
    ) throws {
        let updates = [
            (speculationDraftedTokens, draftedTokens),
            (speculationAcceptedDraftTokens, acceptedDraftTokens),
            (speculationVerificationRounds, verificationRounds),
            (speculationFallbackRounds, fallbackRounds),
        ]
        var next: [Int] = []
        next.reserveCapacity(updates.count)
        for (current, delta) in updates {
            let (value, overflow) = current.addingReportingOverflow(delta)
            guard !overflow else {
                throw DenseContinuousBatchRuntimeError.speculationTelemetryOverflow
            }
            next.append(value)
        }
        speculationDraftedTokens = next[0]
        speculationAcceptedDraftTokens = next[1]
        speculationVerificationRounds = next[2]
        speculationFallbackRounds = next[3]
    }

    private func decodeScalar(
        _ id: BatchRequestID,
        soloPipelineState: BatchSoloPipelineState,
        synchronously: Bool
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard var slot = slots[id] else {
            throw DenseContinuousBatchRuntimeError.unknownRequest(id)
        }
        guard slot.prefillComplete,
            slot.scalarCaches != nil,
            let current = slot.stagedToken
        else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        try ensureScalarCapacity(for: id, slot: &slot, additionalTokens: 1)

        if slot.scalarStep == nil {
            let traceCounter = TraceCounter()
            slot.scalarTraceCounter = traceCounter
            slot.scalarStep = makeScalarStep(caches: slot.scalarCaches!, counter: traceCounter)
        }
        let emitted = try validatedToken(current, id: id)
        let following = slot.scalarStep!([current])[0]
        if synchronously {
            eval([following] + slot.scalarCaches!.flatMap { $0.innerState() })
        } else {
            asyncEval(following)
        }

        slot.cachedTokens += 1
        slot.emittedTokens += 1
        slot.stagedToken = following
        slot.soloPipelineState = soloPipelineState
        // Scalar-side of S3 finding 3: dense rows advance their offset in-graph inside `update`, but
        // a recurrent row's offset is driver-maintained. Commit it from the new `cachedTokens` so a
        // later rejoin (scalar → batch) validates the row at its true logical length instead of the
        // frozen prefill offset. No-op for uniform dense families (guarded inside the helper).
        commitRecurrentRowOffsets(slot.scalarCaches!, committed: slot.cachedTokens)
        slots[id] = slot
        return ContinuousBatchRuntimeDecodeResult(
            id: id,
            tokens: [emitted],
            finished: false,
            soloPipelineState: soloPipelineState)
    }

    private func decodeBatch(_ ids: [BatchRequestID]) throws
        -> [ContinuousBatchRuntimeDecodeResult]
    {
        guard ids.count >= 2, Set(ids).count == ids.count else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership(ids)
        }
        for id in ids {
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard slot.prefillComplete else {
                throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
            }
            guard !slot.soloPipelineState.requiresDrain else {
                throw DenseContinuousBatchRuntimeError.pendingLookaheadInBatch(id)
            }
        }
        let cohorts = ids.compactMap { slots[$0]?.decodeCohort }
        guard cohorts.count == ids.count,
            Set(cohorts).count == 1,
            ifCaseFixedKVCapacity(cohorts[0]) != nil
        else {
            throw DenseContinuousBatchRuntimeError.incompatibleDecodeCohort(ids)
        }

        try ensureBatch(ids)
        guard var batch else {
            preconditionFailure("ensureBatch returned without a batch")
        }
        try ensureBatchCapacity(&batch, additionalTokens: 1)
        if batch.step == nil {
            batch.step = makeBatchStep(
                caches: batch.caches,
                batchSize: ids.count,
                counter: batch.traceCounter)
        }

        let current = batch.stagedTokens
        let emitted = current.asType(.int32).asArray(Int32.self).map(Int.init)
        guard emitted.count == ids.count else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership(ids)
        }
        for (id, token) in zip(ids, emitted) {
            guard token >= 0, token < vocabularySize else {
                throw DenseContinuousBatchRuntimeError.invalidTokenID(id, token)
            }
        }
        let following = batch.step!([current])[0]
        asyncEval(following)

        for (id, token) in zip(ids, emitted) {
            guard var slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            slot.cachedTokens += 1
            slot.emittedTokens += 1
            slot.soloPipelineState = .canonical
            if var session = slot.pldSession {
                do {
                    try session.recordCanonicalToken(token)
                } catch {
                    throw DenseContinuousBatchRuntimeError.speculationStateMismatch(id)
                }
                slot.pldSession = session
            }
            slots[id] = slot
        }
        // Batched-side of S3 finding 3: every row advanced by exactly one committed token in
        // lockstep this step. Dense batched offsets advance in-graph; the recurrent batched rows'
        // logical offsets are driver-maintained, so advance them here to keep them equal to each
        // row's `cachedTokens`. Without this, extract at spill/re-merge restores a stale offset and
        // `validateCacheLengths` throws. No-op for uniform dense families (guarded in the helper).
        advanceRecurrentBatchedRowOffsets(batch.caches, by: 1)
        batch.stagedTokens = following
        self.batch = batch
        return zip(ids, emitted).map { id, token in
            ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                soloPipelineState: .canonical)
        }
    }

    private func ensureScalar(_ id: BatchRequestID) throws {
        guard var slot = slots[id] else {
            throw DenseContinuousBatchRuntimeError.unknownRequest(id)
        }
        guard slot.prefillComplete else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        if slot.scalarCaches != nil, slot.stagedToken != nil {
            return
        }
        guard let batch, let row = batch.ids.firstIndex(of: id) else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        guard batch.incarnations.indices.contains(row),
            batch.incarnations[row] == slot.incarnation
        else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        let otherLiveRows = batch.ids.enumerated().compactMap {
            physicalRow, physicalID -> BatchRequestID? in
            guard physicalID != id,
                batch.incarnations.indices.contains(physicalRow),
                let currentSlot = slots[physicalID],
                batch.incarnations[physicalRow] == currentSlot.incarnation
            else {
                return nil
            }
            return physicalID
        }
        guard otherLiveRows.isEmpty else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership([id] + otherLiveRows)
        }

        let caches = try batch.caches.map { try $0.extractContinuousRow(slot: row) }
        let staged = batch.stagedTokens[row].reshaped([1])
        eval([staged] + caches.flatMap { $0.innerState() })
        try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)

        slot.scalarCaches = caches
        slot.stagedToken = staged
        slot.scalarStep = nil
        slot.scalarVerifySteps = [:]
        slot.scalarTraceCounter = nil
        slots[id] = slot
        self.batch = nil
        recordPhysicalKVCapacity(try denseScalarCache(in: caches).capacity, for: [id])
        pruneKVReservationsToLiveSlots()
    }

    private func ensureBatch(_ ids: [BatchRequestID]) throws {
        if let batch,
            batch.ids == ids,
            batch.incarnations.count == ids.count,
            zip(ids, batch.incarnations).allSatisfy({
                slots[$0.0]?.incarnation == $0.1
            })
        {
            return
        }

        var materialized: [BatchRequestID: MaterializedSlot] = [:]
        var spilledRows: [BatchRequestID: MaterializedSlot] = [:]
        if let existingBatch = batch {
            for (row, id) in existingBatch.ids.enumerated() {
                guard let slot = slots[id],
                    existingBatch.incarnations.indices.contains(row),
                    existingBatch.incarnations[row] == slot.incarnation
                else { continue }
                let caches = try existingBatch.caches.map {
                    try $0.extractContinuousRow(slot: row)
                }
                let staged = existingBatch.stagedTokens[row].reshaped([1])
                eval([staged] + caches.flatMap { $0.innerState() })
                try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)
                let spilled = MaterializedSlot(
                    caches: caches,
                    stagedToken: staged)
                spilledRows[id] = spilled
                if ids.contains(id) {
                    materialized[id] = spilled
                }
            }
        }

        for id in ids where materialized[id] == nil {
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard let caches = slot.scalarCaches, let staged = slot.stagedToken else {
                throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
            }
            try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)
            materialized[id] = MaterializedSlot(caches: caches, stagedToken: staged)
        }

        let rows = try ids.map { id -> MaterializedSlot in
            guard let row = materialized[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            return row
        }
        // The fixed-capacity cohort requirement is a dense-KV property: only the growing dense
        // layers carry a capacity, so check exactly those (recurrent rows are fixed-size by
        // construction). A wrong-kind row at a dense index fails the check (nil != capacity).
        guard let expectedCapacity = ifCaseFixedKVCapacity(
            slots[ids[0]]!.decodeCohort),
            rows.allSatisfy({ row in
                denseLayerIndices.allSatisfy { layer in
                    (row.caches[layer] as? any ContinuousScalarKVCache)?.capacity
                        == expectedCapacity
                }
            })
        else {
            throw DenseContinuousBatchRuntimeError.incompatibleDecodeCohort(ids)
        }
        let lengths = ids.map { slots[$0]!.cachedTokens }
        var caches: [any ContinuousBatchedRowCache] = []
        caches.reserveCapacity(layerCount)
        for layer in 0 ..< layerCount {
            caches.append(
                try cacheFamily.mergeRow(
                    layer: layer,
                    rows: rows.map { $0.caches[layer] },
                    lengths: lengths))
        }
        let staged = concatenated(rows.map(\.stagedToken), axis: 0)
        eval([staged] + caches.flatMap { $0.modelCache.innerState() })

        let traceCounter = TraceCounter()
        let nextBatch = BatchState(
            ids: ids,
            incarnations: ids.map { slots[$0]!.incarnation },
            caches: caches,
            stagedTokens: staged,
            step: nil,
            traceCounter: traceCounter)
        for (id, spilled) in spilledRows where !ids.contains(id) {
            guard var slot = slots[id] else { continue }
            slot.scalarCaches = spilled.caches
            slot.stagedToken = spilled.stagedToken
            slot.scalarStep = nil
            slot.scalarVerifySteps = [:]
            slot.scalarTraceCounter = nil
            slot.soloPipelineState = .canonical
            slots[id] = slot
        }
        for id in ids {
            var slot = slots[id]!
            slot.scalarCaches = nil
            slot.stagedToken = nil
            slot.scalarStep = nil
            slot.scalarVerifySteps = [:]
            slot.scalarTraceCounter = nil
            slot.soloPipelineState = .canonical
            slots[id] = slot
        }
        batch = nextBatch
        recordPhysicalKVCapacity(try denseBatchedCache(in: caches).capacity, for: ids)
        pruneKVReservationsToLiveSlots()
    }

    private func calibrateCacheGeometryIfNeeded() throws {
        guard !cacheGeometryCalibrated else { return }

        let caches = cacheFamily.makeScalarRows(
            layerCount: layerCount,
            capacity: 1)
        let modelCaches: [any KVCache] = caches.map { $0 }
        let token = MLXArray([Int32(0)]).reshaped([1, 1])
        let logits = model(token, cache: modelCaches)
        eval([logits] + caches.flatMap { $0.innerState() })
        try validatePhysicalCacheGeometry(caches)
        cacheGeometryCalibrated = true
    }

    /// Layer indices carrying a growing dense KV cache: every layer for the uniform fp16/affine
    /// families, the verified geometry map's dense set for hybrid. All capacity/geometry
    /// accounting keys off these — on qwen3_5 (interval 4) layer 0 is RECURRENT, so any
    /// `caches[0]`/`caches.first` capacity assumption is wrong there.
    private var denseLayerIndices: [Int] {
        cacheFamily.hybridGeometry?.map.denseLayerIndices ?? Array(0 ..< layerCount)
    }

    /// First dense layer index (0 for uniform families). `nil` only for a pure-recurrent map,
    /// which the hybrid geometry's failable init already rejects — callers still fail closed.
    private var firstDenseLayerIndex: Int? {
        guard let hybrid = cacheFamily.hybridGeometry else { return 0 }
        return hybrid.map.firstDenseLayerIndex
    }

    /// The first dense scalar row — the authority for per-request KV capacity. Throws (never
    /// defaults to layer 0) when the dense layer is absent or of the wrong kind.
    private func denseScalarCache(
        in caches: [any ContinuousScalarRowCache]
    ) throws -> any ContinuousScalarKVCache {
        guard let index = firstDenseLayerIndex,
            caches.indices.contains(index),
            let dense = caches[index] as? any ContinuousScalarKVCache
        else {
            throw DenseContinuousBatchRuntimeError.noCacheLayers
        }
        return dense
    }

    /// The first dense batched cache — the authority for batch KV capacity/frontier. Throws
    /// (never defaults to layer 0) when the dense layer is absent or of the wrong kind.
    private func denseBatchedCache(
        in caches: [any ContinuousBatchedRowCache]
    ) throws -> any ContinuousBatchedKVCache {
        guard let index = firstDenseLayerIndex,
            caches.indices.contains(index),
            let dense = caches[index] as? any ContinuousBatchedKVCache
        else {
            throw DenseContinuousBatchRuntimeError.noCacheLayers
        }
        return dense
    }

    /// Driver-side maintenance of recurrent-row offsets (S3 finding 3): nothing in the vendored
    /// model ever advances a `MambaCache`'s `offset`, so the runtime records each commit point
    /// itself. No-op for uniform dense families.
    private func commitRecurrentRowOffsets(
        _ caches: [any ContinuousScalarRowCache], committed: Int
    ) {
        guard cacheFamily.hybridGeometry != nil else { return }
        for cache in caches {
            if let recurrent = cache as? RecurrentScalarRowCache {
                recurrent.offset = committed
            }
        }
    }

    /// Driver-side maintenance of BATCHED recurrent-row offsets (S3 finding 3). `decodeBatch`
    /// commits exactly `delta` tokens per row per step in lockstep; the batched `MambaCache`'s
    /// logical offsets are never advanced by the model, so the runtime advances them so a later
    /// extract (spill or churn re-merge) restores each row at its true logical length. No-op for
    /// uniform dense families.
    private func advanceRecurrentBatchedRowOffsets(
        _ caches: [any ContinuousBatchedRowCache], by delta: Int
    ) {
        guard cacheFamily.hybridGeometry != nil else { return }
        for cache in caches {
            (cache as? BatchedRecurrentStateCache)?.advanceOffsets(by: delta)
        }
    }

    private func validatePhysicalCacheGeometry(
        _ caches: [any ContinuousScalarRowCache]
    ) throws {
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.modelLayerCountMismatch(
                expected: layerCount, actual: caches.count)
        }
        let hybridMap = cacheFamily.hybridGeometry?.map
        for (layer, cache) in caches.enumerated() {
            if let hybridMap, hybridMap.kind(atLayer: layer).isRecurrentState {
                // A recurrent layer's row must literally BE the vendored `MambaCache`: the model
                // consumes it via a concrete `cache as? MambaCache` downcast (Qwen35.swift:487,537)
                // and any other object would run the layer silently stateless, not fail closed.
                guard cache is RecurrentScalarRowCache else {
                    throw DenseContinuousBatchRuntimeError.cacheGeometryMismatch(
                        expectedHeads: expectedKVHeads,
                        expectedDimension: expectedKVHeadDimension,
                        actual: [])
                }
                continue
            }
            guard let dense = cache as? any ContinuousScalarKVCache,
                let geometry = dense.continuousKVGeometry
            else {
                throw DenseContinuousBatchRuntimeError.cacheGeometryMismatch(
                    expectedHeads: expectedKVHeads,
                    expectedDimension: expectedKVHeadDimension,
                    actual: [])
            }
            guard geometry.keyValueHeadCount == expectedKVHeads,
                geometry.keyHeadDimension == expectedKVHeadDimension,
                geometry.valueHeadDimension == expectedKVHeadDimension
            else {
                throw DenseContinuousBatchRuntimeError.cacheGeometryMismatch(
                    expectedHeads: expectedKVHeads,
                    expectedDimension: expectedKVHeadDimension,
                    actual: [
                        geometry.keyValueHeadCount,
                        geometry.keyHeadDimension,
                        geometry.valueHeadDimension,
                    ])
            }
            guard geometry.keyElementBytes == expectedKVElementBytes,
                geometry.valueElementBytes == expectedKVElementBytes
            else {
                throw DenseContinuousBatchRuntimeError.cacheElementSizeMismatch(
                    expected: expectedKVElementBytes,
                    actual: geometry.keyElementBytes == expectedKVElementBytes
                        ? geometry.valueElementBytes : geometry.keyElementBytes)
            }
        }
    }

    private func pruneKVReservationsToLiveSlots() {
        kvCapacityReservations = kvCapacityReservations.filter { slots[$0.key] != nil }
        recomputeReservedKVBytes()
    }

    private func recomputeReservedKVBytes() {
        reservedKVBytes = (try? kvBytePlan.transitionEnvelopeBytes(
            capacities: Array(kvCapacityReservations.values))) ?? maxReservedKVBytes
    }

    private func recordPhysicalKVCapacity(
        _ capacity: Int,
        for ids: [BatchRequestID]
    ) {
        var changed = false
        for id in ids {
            guard let reserved = kvCapacityReservations[id], capacity > reserved else { continue }
            kvCapacityReservations[id] = capacity
            changed = true
        }
        if changed { recomputeReservedKVBytes() }
    }

    private func makeScalarStep(
        caches: [any ContinuousScalarRowCache], counter: TraceCounter
    ) -> Step {
        let model = self.model
        // A scalar row IS the model cache: a dense row is the `CompiledCache` the model updates,
        // a recurrent row is the vendored `MambaCache` the model consumes via `cache as? MambaCache`
        // — so the row maps straight to `any KVCache` (no `.modelCache` indirection, unlike the
        // batched seam whose rows wrap an inner cache).
        let modelCaches: [any KVCache] = caches.map { $0 }
        let state: [any Updatable] = caches.map { $0 }
        let step: Step = { arguments in
            counter.count += 1
            let input = arguments[0].reshaped([1, 1])
            let logits = model(input, cache: modelCaches)
            return [
                CompiledMLXDecoder.greedyTokenOrInvalidSentinel(
                    logits[0..., -1, 0...]),
            ]
        }
        return compile(inputs: state, outputs: state, step)
    }

    private func makeScalarVerifyStep(
        caches: [any ContinuousScalarRowCache],
        inputWidth: Int
    ) -> Step {
        let model = self.model
        // Scalar rows are model caches directly — see `makeScalarStep`.
        let modelCaches: [any KVCache] = caches.map { $0 }
        let state: [any Updatable] = caches.map { $0 }
        let step: Step = { arguments in
            let input = arguments[0].reshaped([1, inputWidth])
            let logits = model(input, cache: modelCaches)
            return [
                CompiledMLXDecoder.greedyTokenOrInvalidSentinel(logits[0]),
            ]
        }
        return compile(inputs: state, outputs: state, step)
    }

    private func makeBatchStep(
        caches: [any ContinuousBatchedRowCache],
        batchSize: Int,
        counter: TraceCounter
    ) -> Step {
        let model = self.model
        // A batched row WRAPS its model cache (dense → self, recurrent → the inner `MambaCache`),
        // so the model forward and its `Updatable` state must go through `.modelCache`. Mapping the
        // row directly would hand the model the wrapper — the silent-stateless trap in batch mode —
        // and does not even conform to `KVCache`.
        let modelCaches: [any KVCache] = caches.map { $0.modelCache }
        let state: [any Updatable] = caches.map { $0.modelCache }
        let step: Step = { arguments in
            counter.count += 1
            let input = arguments[0].reshaped([batchSize, 1])
            let logits = model(input, cache: modelCaches)
            return [
                CompiledMLXDecoder.greedyTokenOrInvalidSentinel(
                    logits[0..., -1, 0...]),
            ]
        }
        return compile(inputs: state, outputs: state, step)
    }

    private func ensureScalarCapacity(
        for id: BatchRequestID, slot: inout Slot, additionalTokens: Int
    ) throws {
        guard let caches = slot.scalarCaches else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        // Capacity is a dense-KV concept: read it from the first DENSE row, never `caches.first`
        // (layer 0 is recurrent on qwen3_5). Growth still loops every row — `grow(by:)` is a
        // proven no-op on the fixed-size recurrent state.
        let capacity = try denseScalarCache(in: caches).capacity
        let required = slot.cachedTokens + additionalTokens
        let requestLimit = min(
            maxContextTokens,
            slot.totalPromptTokens + slot.maxOutputTokens)
        guard required <= requestLimit else {
            throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                id, requested: required, limit: requestLimit)
        }
        guard required > capacity else { return }

        if let staged = slot.stagedToken {
            eval([staged] + caches.flatMap { $0.innerState() })
        }
        let growth = roundedGrowth(required: required, capacity: capacity)
        for cache in caches { cache.grow(by: growth) }
        recordPhysicalKVCapacity(try denseScalarCache(in: caches).capacity, for: [id])
        slot.scalarStep = nil
        slot.scalarVerifySteps = [:]
        slot.scalarTraceCounter = nil
    }

    private func ensureBatchCapacity(
        _ batch: inout BatchState, additionalTokens: Int
    ) throws {
        for id in batch.ids {
            guard let slot = slots[id] else { continue }
            let required = slot.cachedTokens + additionalTokens
            let requestLimit = min(
                maxContextTokens,
                slot.totalPromptTokens + slot.maxOutputTokens)
            guard required <= requestLimit else {
                throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                    id, requested: required, limit: requestLimit)
            }
        }
        // Frontier and capacity are dense-KV concepts — read them from the first DENSE batched
        // layer, never `batch.caches.first` (recurrent on qwen3_5). This mirrors the same
        // dense-keying already applied at the diagnostics site; a recurrent row's frontier is not
        // a growing-KV write position. Growth loops every row (`grow(by:)` no-ops the fixed-size
        // recurrent state); state eval goes through `.modelCache` because a batched row wraps its
        // model cache rather than being one.
        let denseCache = try denseBatchedCache(in: batch.caches)
        let (required, overflow) = denseCache.continuousPhysicalWrittenEnd
            .addingReportingOverflow(additionalTokens)
        guard !overflow, required <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(batch.ids[0])
        }
        let capacity = denseCache.capacity
        guard required > capacity else { return }

        eval([batch.stagedTokens] + batch.caches.flatMap { $0.modelCache.innerState() })
        let growth = roundedGrowth(required: required, capacity: capacity)
        for cache in batch.caches { cache.grow(by: growth) }
        recordPhysicalKVCapacity(try denseBatchedCache(in: batch.caches).capacity, for: batch.ids)
        batch.step = nil
    }

    private func validateCacheLengths(
        _ caches: [any ContinuousScalarRowCache],
        id: BatchRequestID,
        expected: Int
    ) throws {
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.cacheLayerMismatch(
                id, expected: layerCount, actual: caches.count)
        }
        for cache in caches {
            let actual = cache.continuousLogicalOffset
            guard actual == expected else {
                throw DenseContinuousBatchRuntimeError.cacheLengthMismatch(
                    id, expected: expected, actual: actual)
            }
        }
    }

    private func validateRequest(
        id: BatchRequestID,
        promptTokens: [Int],
        totalPromptTokens: Int,
        maxOutputTokens: Int
    ) throws {
        guard totalPromptTokens > 0, maxOutputTokens > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(id)
        }
        let (requested, overflow) = totalPromptTokens.addingReportingOverflow(maxOutputTokens)
        guard !overflow, requested <= maxContextTokens else {
            throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                id, requested: overflow ? Int.max : requested, limit: maxContextTokens)
        }
        for token in promptTokens where token < 0 || token >= vocabularySize {
            throw DenseContinuousBatchRuntimeError.invalidTokenID(id, token)
        }
    }

    private func validateAdmission(
        _ admission: ContinuousBatchRuntimeAdmission
    ) throws -> (requestedTokens: Int, capacity: Int) {
        let submission = admission.submission
        if submission.requestsSpeculation, soloPLDConfiguration == nil {
            throw DenseContinuousBatchRuntimeError.speculationUnsupported
        }
        try validateRequest(
            id: admission.id,
            promptTokens: submission.promptTokens,
            totalPromptTokens: submission.promptTokens.count,
            maxOutputTokens: submission.maxOutputTokens)
        let (requestedTokens, overflow) =
            submission.promptTokens.count.addingReportingOverflow(
                submission.maxOutputTokens)
        guard !overflow, requestedTokens <= maxReservedContextTokens else {
            throw DenseContinuousBatchRuntimeError
                .requestReservedContextLimitExceeded(
                    admission.id,
                    requested: overflow ? Int.max : requestedTokens,
                    limit: maxReservedContextTokens)
        }
        let capacity: Int
        do {
            capacity = try kvBytePlan.reservedCapacity(
                promptTokens: submission.promptTokens.count,
                outputTokens: submission.maxOutputTokens)
        } catch {
            throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
        }
        let requestKVBytes: Int
        do {
            requestKVBytes = try kvBytePlan.transitionEnvelopeBytes(
                capacities: [capacity])
        } catch {
            throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
        }
        guard requestKVBytes <= maxReservedKVBytes else {
            throw DenseContinuousBatchRuntimeError
                .requestReservedKVByteLimitExceeded(
                    admission.id,
                    requested: requestKVBytes,
                    limit: maxReservedKVBytes)
        }
        return (requestedTokens, capacity)
    }

    private func validatedToken(
        _ token: MLXArray,
        id: BatchRequestID
    ) throws -> Int {
        let value = token.item(Int.self)
        guard value >= 0, value < vocabularySize else {
            throw DenseContinuousBatchRuntimeError.invalidTokenID(id, value)
        }
        return value
    }

    private func validateRuntimeTokens(
        _ tokens: [Int],
        id: BatchRequestID
    ) throws {
        for token in tokens where
            token < 0 || token >= vocabularySize || Int32(exactly: token) == nil
        {
            throw DenseContinuousBatchRuntimeError.invalidTokenID(id, token)
        }
    }

    private func decodeCohort(
        id: BatchRequestID,
        promptTokens: Int,
        maxOutputTokens: Int
    ) throws -> BatchDecodeCohort {
        guard maxOutputTokens <= initialDecodeReserve else {
            return .isolated(id)
        }
        return .fixedKVCapacity(
            try roundedCapacity(
                promptTokens: promptTokens,
                outputTokens: maxOutputTokens,
                id: id))
    }

    private func ifCaseFixedKVCapacity(
        _ cohort: BatchDecodeCohort
    ) -> Int? {
        guard case .fixedKVCapacity(let capacity) = cohort else {
            return nil
        }
        return capacity
    }

    private func roundedCapacity(
        promptTokens: Int, outputTokens: Int, id: BatchRequestID
    ) throws -> Int {
        let (total, overflow) = promptTokens.addingReportingOverflow(outputTokens)
        guard !overflow, total > 0, total <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        let (withRounding, roundingOverflow) = total.addingReportingOverflow(allocationChunk - 1)
        guard !roundingOverflow else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        let rounded = min(
            maxContextTokens,
            max(allocationChunk, (withRounding / allocationChunk) * allocationChunk))
        guard rounded <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        return rounded
    }

    private func roundedGrowth(required: Int, capacity: Int) -> Int {
        let missing = required - capacity
        let chunked = ((missing + allocationChunk - 1) / allocationChunk) * allocationChunk
        return min(chunked, maxContextTokens - capacity)
    }
}
