import Foundation
import HarnessCore
import MLX
import MLXLMCommon

public enum AffineKVCacheConfigurationError: Error, Equatable, Sendable {
    case unsupportedBitWidth(Int)
    case unsupportedGroupSize(Int)
}

/// How a shared attention helper is allowed to consume an affine cache.
///
/// The default remains the previously qualified materialize-then-attend behavior. The packed
/// route is an explicit experimental opt-in so adding the shared router cannot silently activate
/// it for an unqualified model architecture.
public enum AffineKVAttentionMode:
    String, Codable, Equatable, Hashable, Sendable
{
    case materialize
    case splitQuantizedMM = "split-quantized-mm"
}

/// The cache read path most recently built into an MLX graph.
///
/// This is intentionally distinct from the requested mode: a requested packed route is not
/// evidence of engagement until an attention forward actually calls it.
public enum AffineKVAttentionOperation: String, Codable, Equatable, Sendable {
    case materializedKV = "materialized-kv"
    case splitQuantizedMM = "split-quantized-mm"
}

/// Native MLX affine-packing controls for one KV cache.
///
/// K and V are deliberately independent: asymmetric cells such as K4V2 are first-class
/// configurations rather than aliases that silently coerce one side. The supported values
/// are the native MLX formats exercised by fast-mlx's storage accountant and bench matrix.
public struct AffineKVCacheConfiguration: Equatable, Hashable, Sendable {
    public let keyBits: Int
    public let valueBits: Int
    public let keyGroupSize: Int
    public let valueGroupSize: Int

    public init(
        keyBits: Int, valueBits: Int,
        keyGroupSize: Int, valueGroupSize: Int
    ) throws {
        for bits in [keyBits, valueBits] where !Self.supportedBits.contains(bits) {
            throw AffineKVCacheConfigurationError.unsupportedBitWidth(bits)
        }
        for groupSize in [keyGroupSize, valueGroupSize]
        where !Self.supportedGroupSizes.contains(groupSize) {
            throw AffineKVCacheConfigurationError.unsupportedGroupSize(groupSize)
        }
        self.init(
            validatedKeyBits: keyBits, valueBits: valueBits,
            keyGroupSize: keyGroupSize, valueGroupSize: valueGroupSize)
    }

    fileprivate init(
        validatedKeyBits keyBits: Int, valueBits: Int,
        keyGroupSize: Int, valueGroupSize: Int
    ) {
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.keyGroupSize = keyGroupSize
        self.valueGroupSize = valueGroupSize
    }

    private static let supportedBits: Set<Int> = [2, 4, 8]
    private static let supportedGroupSizes: Set<Int> = [32, 64, 128]
}

/// Named affine cells admitted to the storage-quality matrix.
///
/// The raw value is the only accepted CLI/evidence spelling. Keeping this set closed makes a
/// typo or an unmeasured geometry fail instead of being coerced to a nearby tier or fp16.
public enum AffineKVTier: String, CaseIterable, Sendable, Hashable {
    case k4v2G64 = "affine-k4v2-g64"
    case k4v2G128 = "affine-k4v2-g128"
    case k8v2G64 = "affine-k8v2-g64"
    case k8v2G128 = "affine-k8v2-g128"
    case k4v4G128 = "affine-k4v4-g128"

    public var keyBits: Int {
        switch self {
        case .k8v2G64, .k8v2G128: 8
        case .k4v2G64, .k4v2G128, .k4v4G128: 4
        }
    }

    public var valueBits: Int {
        switch self {
        case .k4v4G128: 4
        case .k4v2G64, .k4v2G128, .k8v2G64, .k8v2G128: 2
        }
    }

    public var groupSize: Int {
        switch self {
        case .k4v2G64, .k8v2G64: 64
        case .k4v2G128, .k8v2G128, .k4v4G128: 128
        }
    }

    public var configuration: AffineKVCacheConfiguration {
        AffineKVCacheConfiguration(
            validatedKeyBits: keyBits, valueBits: valueBits,
            keyGroupSize: groupSize, valueGroupSize: groupSize)
    }
}

/// Actual MLX-array geometry, persistent bytes, and logical materialization workspace owned or
/// produced by an affine cache after allocation.
///
/// `dataArrayBytes` reconciles directly with `KVStorageFormat.affine`; the in-graph offset
/// is reported separately as control state so capacity claims never hide implementation
/// bytes that are not part of the packed data layout.
public struct AffineKVCacheStorageSnapshot: Equatable, Sendable {
    public let capacityTokens: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int
    /// Exact logical bytes of the full-precision K/V pair returned to attention by the most
    /// recent update. Transformer layers consume these sequentially, so this is one layer's
    /// pair rather than the sum across every persistent layer cache.
    public let materializationWorkspaceBytes: Int
    /// Logical score plus softmax-weight arrays owned by the explicit split-attention graph.
    /// This is zero on the materialized route. Raw allocator peak remains a separate bench
    /// receipt because MLX may fuse or reuse buffers internally.
    public let attentionWorkspaceBytes: Int
    /// Route-specific logical workspace. The two components are mutually exclusive today, but
    /// retaining their sum makes future composed paths fail visibly instead of dropping a term.
    public let workspaceBytes: Int
    /// Actual cache read operation observed while building the most recent attention graph.
    public let attentionOperation: AffineKVAttentionOperation

    public var dataArrayBytes: Int { payloadBytes + metadataBytes }
    public var totalPersistentBytes: Int { dataArrayBytes + controlBytes }
}

/// Actor-safe scalar telemetry aggregated from every affine layer cache after a run.
/// MLX arrays never leave their owner; only their evaluated geometry and byte counts do.
public struct AffineKVCacheTelemetry: Equatable, Sendable {
    public let tier: AffineKVTier
    public let cachedTokens: Int
    public let layerCount: Int
    public let capacityTokens: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int
    public let attentionWorkspaceBytes: Int
    public let workspaceBytes: Int
    public let attentionOperation: AffineKVAttentionOperation

    public var dataArrayBytes: Int { payloadBytes + metadataBytes }
    public var totalPersistentBytes: Int { dataArrayBytes + controlBytes }

    /// Capture must run inside the cache owner's inference actor. It synchronizes only the
    /// in-graph offsets and is therefore intended for post-run evidence, never the hot loop.
    public static func capture(
        tier: AffineKVTier, caches: [AffineKVCache]
    ) -> AffineKVCacheTelemetry {
        precondition(!caches.isEmpty, "affine telemetry requires at least one layer cache")
        let snapshots = caches.map { cache -> AffineKVCacheStorageSnapshot in
            precondition(
                cache.configuration == tier.configuration,
                "affine cache configuration does not match the requested tier")
            guard let snapshot = cache.storageSnapshot() else {
                preconditionFailure("affine cache did not allocate before telemetry capture")
            }
            return snapshot
        }
        let first = snapshots[0]
        precondition(
            first.keyHeadDimension == first.valueHeadDimension,
            "K/V head dimensions differ")
        precondition(
            snapshots.dropFirst().allSatisfy {
                $0.capacityTokens == first.capacityTokens
                    && $0.sequences == first.sequences
                    && $0.kvHeadCount == first.kvHeadCount
                    && $0.keyHeadDimension == first.keyHeadDimension
                    && $0.valueHeadDimension == first.valueHeadDimension
                    && $0.metadataScalarBytes == first.metadataScalarBytes
                    && $0.materializationWorkspaceBytes
                        == first.materializationWorkspaceBytes
                    && $0.attentionWorkspaceBytes
                        == first.attentionWorkspaceBytes
                    && $0.workspaceBytes == first.workspaceBytes
                    && $0.attentionOperation == first.attentionOperation
            },
            "affine layer-cache geometry is inconsistent")

        let cachedTokens = Int(caches[0].offsetArr.item(Int32.self))
        precondition(
            caches.dropFirst().allSatisfy {
                Int($0.offsetArr.item(Int32.self)) == cachedTokens
            },
            "affine layer-cache offsets are inconsistent")

        func sum(_ values: [Int]) -> Int {
            values.reduce(into: 0) { result, value in
                let (next, overflow) = result.addingReportingOverflow(value)
                precondition(!overflow, "affine telemetry byte count overflow")
                result = next
            }
        }
        return AffineKVCacheTelemetry(
            tier: tier,
            cachedTokens: cachedTokens,
            layerCount: caches.count,
            capacityTokens: first.capacityTokens,
            sequences: first.sequences,
            kvHeadCount: first.kvHeadCount,
            headDimension: first.keyHeadDimension,
            metadataScalarBytes: first.metadataScalarBytes,
            payloadBytes: sum(snapshots.map(\.payloadBytes)),
            metadataBytes: sum(snapshots.map(\.metadataBytes)),
            controlBytes: sum(snapshots.map(\.controlBytes)),
            materializationWorkspaceBytes: first.materializationWorkspaceBytes,
            attentionWorkspaceBytes: first.attentionWorkspaceBytes,
            workspaceBytes: first.workspaceBytes,
            attentionOperation: first.attentionOperation)
    }
}

public enum KVTunerKVCacheTelemetryError: Error, Equatable, Sendable {
    case layerCountMismatch(expected: Int, actual: Int)
    case configurationMismatch(layer: Int)
    case unallocatedLayer(Int)
    case inconsistentGeometry(layer: Int)
    case inconsistentOffset(layer: Int)
    case byteCountOverflow
}

/// Shared scalar reconciliation for qualified and preselection KVTuner policies. This helper is
/// intentionally private: callers receive only the policy-specific authenticated envelope, never
/// a policy-free byte snapshot that could be detached from the schedule which produced it.
private struct KVTunerAffineCacheAggregate {
    let cachedTokens: Int
    let layerCount: Int
    let capacityTokens: Int
    let sequences: Int
    let kvHeadCount: Int
    let headDimension: Int
    let metadataScalarBytes: Int
    let payloadBytes: Int
    let metadataBytes: Int
    let controlBytes: Int
    let materializationWorkspaceBytes: Int
    let attentionWorkspaceBytes: Int
    let workspaceBytes: Int
    let attentionOperation: AffineKVAttentionOperation
    let totalPersistentBytes: Int
    let totalBytes: Int

    static func capture(
        layers: [KVTunerRuntimeLayerPolicy],
        groupSize: Int,
        caches: [AffineKVCache]
    ) throws -> KVTunerAffineCacheAggregate {
        guard caches.count == layers.count else {
            throw KVTunerKVCacheTelemetryError.layerCountMismatch(
                expected: layers.count, actual: caches.count)
        }

        var snapshots: [AffineKVCacheStorageSnapshot] = []
        snapshots.reserveCapacity(caches.count)
        for (position, pair) in zip(caches, layers).enumerated() {
            let (cache, policy) = pair
            let expected: AffineKVCacheConfiguration
            do {
                expected = try AffineKVCacheConfiguration(
                    keyBits: policy.keyBits,
                    valueBits: policy.valueBits,
                    keyGroupSize: groupSize,
                    valueGroupSize: groupSize)
            } catch {
                throw KVTunerKVCacheTelemetryError.configurationMismatch(
                    layer: position)
            }
            guard cache.configuration == expected else {
                throw KVTunerKVCacheTelemetryError.configurationMismatch(
                    layer: position)
            }
            guard let snapshot = cache.storageSnapshot() else {
                throw KVTunerKVCacheTelemetryError.unallocatedLayer(position)
            }
            snapshots.append(snapshot)
        }
        guard let first = snapshots.first, let firstCache = caches.first else {
            throw KVTunerKVCacheTelemetryError.layerCountMismatch(
                expected: layers.count, actual: 0)
        }
        guard first.keyHeadDimension == first.valueHeadDimension else {
            throw KVTunerKVCacheTelemetryError.inconsistentGeometry(layer: 0)
        }

        let cachedTokens = Int(firstCache.offsetArr.item(Int32.self))
        for position in snapshots.indices.dropFirst() {
            let snapshot = snapshots[position]
            guard snapshot.capacityTokens == first.capacityTokens,
                snapshot.sequences == first.sequences,
                snapshot.kvHeadCount == first.kvHeadCount,
                snapshot.keyHeadDimension == first.keyHeadDimension,
                snapshot.valueHeadDimension == first.valueHeadDimension,
                snapshot.metadataScalarBytes == first.metadataScalarBytes
                    && snapshot.materializationWorkspaceBytes
                        == first.materializationWorkspaceBytes
                    && snapshot.attentionWorkspaceBytes
                        == first.attentionWorkspaceBytes
                    && snapshot.workspaceBytes == first.workspaceBytes
                    && snapshot.attentionOperation == first.attentionOperation
            else {
                throw KVTunerKVCacheTelemetryError.inconsistentGeometry(
                    layer: position)
            }
            guard Int(caches[position].offsetArr.item(Int32.self))
                == cachedTokens
            else {
                throw KVTunerKVCacheTelemetryError.inconsistentOffset(
                    layer: position)
            }
        }

        func checkedSum(_ values: [Int]) throws -> Int {
            var result = 0
            for value in values {
                let (next, overflow) = result.addingReportingOverflow(value)
                guard !overflow else {
                    throw KVTunerKVCacheTelemetryError.byteCountOverflow
                }
                result = next
            }
            return result
        }

        let payloadBytes = try checkedSum(snapshots.map(\.payloadBytes))
        let metadataBytes = try checkedSum(snapshots.map(\.metadataBytes))
        let controlBytes = try checkedSum(snapshots.map(\.controlBytes))
        let materializationWorkspaceBytes = snapshots.map(
            \.materializationWorkspaceBytes)
            .max() ?? 0
        let attentionWorkspaceBytes = snapshots.map(\.attentionWorkspaceBytes)
            .max() ?? 0
        let workspaceBytes = snapshots.map(\.workspaceBytes).max() ?? 0
        let totalPersistentBytes = try checkedSum([
            payloadBytes, metadataBytes, controlBytes,
        ])
        let totalBytes = try checkedSum([
            totalPersistentBytes, workspaceBytes,
        ])
        return KVTunerAffineCacheAggregate(
            cachedTokens: cachedTokens,
            layerCount: caches.count,
            capacityTokens: first.capacityTokens,
            sequences: first.sequences,
            kvHeadCount: first.kvHeadCount,
            headDimension: first.keyHeadDimension,
            metadataScalarBytes: first.metadataScalarBytes,
            payloadBytes: payloadBytes,
            metadataBytes: metadataBytes,
            controlBytes: controlBytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            attentionWorkspaceBytes: attentionWorkspaceBytes,
            workspaceBytes: workspaceBytes,
            attentionOperation: first.attentionOperation,
            totalPersistentBytes: totalPersistentBytes,
            totalBytes: totalBytes)
    }
}

/// Actor-safe scalar evidence for one authenticated heterogeneous affine policy. The exact
/// schedule digest and frozen layer decisions travel with the actual MLX-array byte totals;
/// no MLX array crosses the inference actor boundary.
public struct KVTunerKVCacheTelemetry: Equatable, Sendable {
    public let artifactSHA256: String
    public let matrixID: String
    public let cellID: String
    public let groupSize: Int
    public let layers: [KVTunerRuntimeLayerPolicy]
    public let cachedTokens: Int
    public let layerCount: Int
    public let capacityTokens: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int
    public let attentionWorkspaceBytes: Int
    public let workspaceBytes: Int
    public let attentionOperation: AffineKVAttentionOperation
    public let totalPersistentBytes: Int
    public let totalBytes: Int

    /// Capture must run in the cache owner's inference actor after allocation. Persistent
    /// bytes sum every layer; workspace is the largest one-layer route-specific workspace,
    /// because transformer layers materialize/attend and consume their temporaries sequentially.
    public static func capture(
        selection: KVTunerRuntimeSelection,
        caches: [AffineKVCache]
    ) throws -> KVTunerKVCacheTelemetry {
        let aggregate = try KVTunerAffineCacheAggregate.capture(
            layers: selection.layers,
            groupSize: selection.groupSize,
            caches: caches)
        return KVTunerKVCacheTelemetry(
            artifactSHA256: selection.artifactSHA256,
            matrixID: selection.matrixID,
            cellID: selection.cellID,
            groupSize: selection.groupSize,
            layers: selection.layers,
            cachedTokens: aggregate.cachedTokens,
            layerCount: aggregate.layerCount,
            capacityTokens: aggregate.capacityTokens,
            sequences: aggregate.sequences,
            kvHeadCount: aggregate.kvHeadCount,
            headDimension: aggregate.headDimension,
            metadataScalarBytes: aggregate.metadataScalarBytes,
            payloadBytes: aggregate.payloadBytes,
            metadataBytes: aggregate.metadataBytes,
            controlBytes: aggregate.controlBytes,
            materializationWorkspaceBytes:
                aggregate.materializationWorkspaceBytes,
            attentionWorkspaceBytes:
                aggregate.attentionWorkspaceBytes,
            workspaceBytes: aggregate.workspaceBytes,
            attentionOperation: aggregate.attentionOperation,
            totalPersistentBytes: aggregate.totalPersistentBytes,
            totalBytes: aggregate.totalBytes)
    }
}

/// Actor-safe execution receipt for an authenticated preselection candidate. The policy digest,
/// complete candidate-set identity, exact model identities, frozen layer decisions, and observed
/// MLX allocation all travel together so a search row cannot self-report cache engagement.
public struct KVTunerCandidateKVCacheTelemetry: Equatable, Sendable {
    public let runtimePolicySHA256: String
    public let calibrationManifestSHA256: String
    public let sourceSensitivityArtifactSHA256: String
    public let candidateListSHA256: String
    public let candidateSHA256: String
    public let matrixID: String
    public let modelConfigHash: String
    public let modelConfigSHA256: String
    public let checkpointManifestHash: String
    public let tokenizerSHA256: String
    public let groupSize: Int
    public let targetPairBitTotal: Int
    public let candidateCount: Int
    public let candidateOrdinal: Int
    public let layers: [KVTunerRuntimeLayerPolicy]
    public let cachedTokens: Int
    public let layerCount: Int
    public let capacityTokens: Int
    public let sequences: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let metadataScalarBytes: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int
    public let materializationWorkspaceBytes: Int
    public let attentionOperation: AffineKVAttentionOperation
    public let totalPersistentBytes: Int
    public let totalBytes: Int

    /// Schema-2 scalar receipt derived directly from observed cache telemetry. Keeping this
    /// conversion beside the source values prevents CLI orchestration from transposing geometry
    /// or omitting a byte class while constructing durable candidate evidence.
    public var evidenceReceipt: KVTunerCandidateRuntimeReceipt {
        KVTunerCandidateRuntimeReceipt(
            runtimePolicySHA256: runtimePolicySHA256,
            cachedTokens: cachedTokens,
            capacityTokens: capacityTokens,
            layers: layers,
            geometry: KVTunerCandidateRuntimeGeometry(
                layerCount: layerCount,
                kvHeadCount: kvHeadCount,
                headDimension: headDimension),
            groupSize: groupSize,
            sequenceCount: sequences,
            metadataScalarBytes: metadataScalarBytes,
            actualPayloadBytes: payloadBytes,
            actualMetadataBytes: metadataBytes,
            actualControlBytes: controlBytes,
            actualWorkspaceBytes: materializationWorkspaceBytes,
            actualTotalPersistentBytes: totalPersistentBytes,
            actualTotalBytes: totalBytes)
    }

    public static func capture(
        policy: KVTunerCandidateRuntimePolicy,
        caches: [AffineKVCache]
    ) throws -> KVTunerCandidateKVCacheTelemetry {
        let aggregate = try KVTunerAffineCacheAggregate.capture(
            layers: policy.layers,
            groupSize: policy.groupSize,
            caches: caches)
        return KVTunerCandidateKVCacheTelemetry(
            runtimePolicySHA256: policy.runtimePolicySHA256,
            calibrationManifestSHA256:
                policy.calibrationManifestSHA256,
            sourceSensitivityArtifactSHA256:
                policy.sourceSensitivityArtifactSHA256,
            candidateListSHA256: policy.candidateListSHA256,
            candidateSHA256: policy.candidateSHA256,
            matrixID: policy.matrixID,
            modelConfigHash: policy.modelConfigHash,
            modelConfigSHA256: policy.modelConfigSHA256,
            checkpointManifestHash: policy.checkpointManifestHash,
            tokenizerSHA256: policy.tokenizerSHA256,
            groupSize: policy.groupSize,
            targetPairBitTotal: policy.targetPairBitTotal,
            candidateCount: policy.candidateCount,
            candidateOrdinal: policy.candidateOrdinal,
            layers: policy.layers,
            cachedTokens: aggregate.cachedTokens,
            layerCount: aggregate.layerCount,
            capacityTokens: aggregate.capacityTokens,
            sequences: aggregate.sequences,
            kvHeadCount: aggregate.kvHeadCount,
            headDimension: aggregate.headDimension,
            metadataScalarBytes: aggregate.metadataScalarBytes,
            payloadBytes: aggregate.payloadBytes,
            metadataBytes: aggregate.metadataBytes,
            controlBytes: aggregate.controlBytes,
            materializationWorkspaceBytes:
                aggregate.materializationWorkspaceBytes,
            attentionOperation: aggregate.attentionOperation,
            totalPersistentBytes: aggregate.totalPersistentBytes,
            totalBytes: aggregate.totalBytes)
    }
}

/// Fixed-capacity, compile-capturable KV cache backed by MLX's native affine packing.
///
/// Incoming K and V rows are independently packed and scattered into fixed-shape buffers.
/// The default read dequantizes the full buffers before the existing fused attention path.
/// Qualified callers can explicitly opt into independent packed K/V matrix multiplies. Both
/// routes preserve the same mask, RoPE, growth, truncation, and in-place reset contract as
/// `CompiledKVCache`.
///
/// This class intentionally does not conform to `Sendable`: every instance is confined to
/// the inference actor because its MLX state is non-Sendable.
public final class AffineKVCache: AttentionKVCacheProtocol, Updatable {
    /// Actor-confined packed state used only while rebuilding continuous-batch membership.
    /// MLX arrays remain inside SpikeCore and never cross a concurrency boundary.
    struct PackedBatchState {
        let capacity: Int
        let configuration: AffineKVCacheConfiguration
        let attentionMode: AffineKVAttentionMode
        let logicalOffset: Int
        let kPayload: MLXArray
        let kScales: MLXArray
        let kBiases: MLXArray
        let vPayload: MLXArray
        let vScales: MLXArray
        let vBiases: MLXArray
        let keyDimension: Int
        let valueDimension: Int
        let keyOutputDType: DType
        let valueOutputDType: DType
        let materializationWorkspaceBytes: Int?
        let attentionWorkspaceBytes: Int?
        let attentionOperation: AffineKVAttentionOperation?
    }

    public private(set) var capacity: Int
    public let configuration: AffineKVCacheConfiguration
    public let attentionMode: AffineKVAttentionMode

    // Lazily allocated from the first model K/V tensors so batch, head count, head dimension,
    // and metadata dtype match the active model. Internal visibility supports on-box layout
    // reconciliation tests through @testable import.
    var kPayload: MLXArray?
    var kScales: MLXArray?
    var kBiases: MLXArray?
    var vPayload: MLXArray?
    var vScales: MLXArray?
    var vBiases: MLXArray?
    var offsetArr: MLXArray = MLXArray([Int32(0)])

    private var keyDimension: Int?
    private var valueDimension: Int?
    private var keyOutputDType: DType?
    private var valueOutputDType: DType?
    private var materializationWorkspaceBytes: Int?
    private var attentionWorkspaceBytes: Int?
    private var attentionOperation: AffineKVAttentionOperation?

    /// Host-side mirror used only by uncompiled prefill/control code. Compiled replays update
    /// `offsetArr` in graph, matching `CompiledKVCache`'s documented contract.
    public private(set) var offset: Int = 0

    public init(
        capacity: Int,
        configuration: AffineKVCacheConfiguration,
        attentionMode: AffineKVAttentionMode = .materialize
    ) {
        precondition(capacity > 0, "capacity must be positive")
        self.capacity = capacity
        self.configuration = configuration
        self.attentionMode = attentionMode
    }

    public var maxSize: Int? { nil }

    /// Snapshot initialized packed arrays for an actor-confined membership transition.
    /// The authoritative logical position is read from graph state, not the host mirror.
    func packedBatchState() -> PackedBatchState? {
        guard let kPayload, let kScales, let kBiases,
            let vPayload, let vScales, let vBiases,
            let keyDimension, let valueDimension,
            let keyOutputDType, let valueOutputDType
        else { return nil }
        return PackedBatchState(
            capacity: capacity,
            configuration: configuration,
            attentionMode: attentionMode,
            logicalOffset: Int(offsetArr.item(Int32.self)),
            kPayload: kPayload,
            kScales: kScales,
            kBiases: kBiases,
            vPayload: vPayload,
            vScales: vScales,
            vBiases: vBiases,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            keyOutputDType: keyOutputDType,
            valueOutputDType: valueOutputDType,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            attentionWorkspaceBytes: attentionWorkspaceBytes,
            attentionOperation: attentionOperation)
    }

    /// Restore one independent scalar cache from a validated packed batch row.
    static func restoringPackedBatchState(_ state: PackedBatchState) -> AffineKVCache {
        let cache = AffineKVCache(
            capacity: state.capacity,
            configuration: state.configuration,
            attentionMode: state.attentionMode)
        cache.kPayload = state.kPayload
        cache.kScales = state.kScales
        cache.kBiases = state.kBiases
        cache.vPayload = state.vPayload
        cache.vScales = state.vScales
        cache.vBiases = state.vBiases
        cache.offsetArr = MLXArray([Int32(state.logicalOffset)])
        cache.keyDimension = state.keyDimension
        cache.valueDimension = state.valueDimension
        cache.keyOutputDType = state.keyOutputDType
        cache.valueOutputDType = state.valueOutputDType
        cache.materializationWorkspaceBytes = state.materializationWorkspaceBytes
        cache.attentionWorkspaceBytes = state.attentionWorkspaceBytes
        cache.attentionOperation = state.attentionOperation
        cache.offset = state.logicalOffset
        return cache
    }

    public func innerState() -> [MLXArray] {
        [kPayload, kScales, kBiases, vPayload, vScales, vBiases].compactMap { $0 }
            + [offsetArr]
    }

    public var ropeOffset: RoPEOffset { .batch(offsetArr) }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        store(keys: keys, values: values)

        let materializedKeys = materialize(
            payload: kPayload!, scales: kScales!, biases: kBiases!,
            dimension: keyDimension!, bits: configuration.keyBits,
            groupSize: configuration.keyGroupSize, dtype: keyOutputDType!)
        let materializedValues = materialize(
            payload: vPayload!, scales: vScales!, biases: vBiases!,
            dimension: valueDimension!, bits: configuration.valueBits,
            groupSize: configuration.valueGroupSize, dtype: valueOutputDType!)
        let (workspaceBytes, overflow) = materializedKeys.nbytes.addingReportingOverflow(
            materializedValues.nbytes)
        precondition(!overflow, "affine materialization workspace byte count overflow")
        materializationWorkspaceBytes = workspaceBytes
        attentionWorkspaceBytes = 0
        attentionOperation = .materializedKV
        return (materializedKeys, materializedValues)
    }

    /// Update the fixed-shape packed cache and attend through two independently configured
    /// quantized matrix multiplies. No dense K/V tensor is reconstructed. This is not fused SDPA:
    /// the score/weight tensor remains an explicit temporary, especially for prefill.
    public func updateAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        precondition(scale.isFinite && scale > 0, "attention scale must be finite and positive")
        switch attentionMode {
        case .materialize:
            let (cachedKeys, cachedValues) = update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask)
        case .splitQuantizedMM:
            store(keys: keys, values: values)
            materializationWorkspaceBytes = 0
            attentionOperation = .splitQuantizedMM
            return packedAttention(queries: queries, scale: scale, mask: mask)
        }
    }

    private func store(keys: MLXArray, values: MLXArray) {
        validateInput(keys: keys, values: values)
        let tokenCount = keys.dim(2)
        let keyCode = encode(
            keys, bits: configuration.keyBits, groupSize: configuration.keyGroupSize)
        let valueCode = encode(
            values, bits: configuration.valueBits, groupSize: configuration.valueGroupSize)

        if kPayload == nil {
            allocate(keyCode: keyCode, valueCode: valueCode)
            keyDimension = keys.dim(3)
            valueDimension = values.dim(3)
            keyOutputDType = keys.dtype
            valueOutputDType = values.dtype
        }

        let positions = offsetArr + MLXArray(Int32(0) ..< Int32(tokenCount))
        let indices = positions.reshaped([1, 1, tokenCount, 1])
        kPayload = putAlong(kPayload!, indices, values: keyCode.payload, axis: 2)
        kScales = putAlong(kScales!, indices, values: keyCode.scales, axis: 2)
        kBiases = putAlong(kBiases!, indices, values: keyCode.biases, axis: 2)
        vPayload = putAlong(vPayload!, indices, values: valueCode.payload, axis: 2)
        vScales = putAlong(vScales!, indices, values: valueCode.scales, axis: 2)
        vBiases = putAlong(vBiases!, indices, values: valueCode.biases, axis: 2)
        offsetArr = offsetArr + MLXArray([Int32(tokenCount)])
        offset += tokenCount
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        precondition(windowSize == nil, "sliding window not supported by AffineKVCache")
        let keyPositions = MLXArray(Int32(0) ..< Int32(capacity))
            .reshaped([1, 1, 1, capacity])
        let queryPositions = (offsetArr + MLXArray(Int32(0) ..< Int32(n)))
            .reshaped([1, 1, n, 1])
        return .array(keyPositions .<= queryPositions)
    }

    public func grow(by chunk: Int) {
        precondition(chunk > 0, "growth chunk must be positive")
        guard kPayload != nil else {
            capacity += chunk
            return
        }

        func grown(_ buffer: MLXArray) -> MLXArray {
            let padding = MLXArray.zeros(
                [buffer.dim(0), buffer.dim(1), chunk, buffer.dim(3)], dtype: buffer.dtype)
            return concatenated([buffer, padding], axis: 2)
        }

        kPayload = grown(kPayload!)
        kScales = grown(kScales!)
        kBiases = grown(kBiases!)
        vPayload = grown(vPayload!)
        vScales = grown(vScales!)
        vBiases = grown(vBiases!)
        capacity += chunk
    }

    public func truncate(to newLength: Int) {
        precondition(newLength >= 0 && newLength <= capacity, "truncate target outside the buffer")
        offsetArr._updateInternal(MLXArray([Int32(newLength)]))
        offset = newLength
    }

    public func resetInPlace() {
        for buffer in [kPayload, kScales, kBiases, vPayload, vScales, vBiases] {
            if let buffer {
                buffer._updateInternal(MLXArray.zeros(buffer.shape, dtype: buffer.dtype))
            }
        }
        offsetArr._updateInternal(MLXArray([Int32(0)]))
        offset = 0
        // Workspace is a per-run high-water mark. The operation stays populated so the
        // identity-preserving cache remains inspectable immediately after reset; the next
        // uncompiled prefill re-observes the fixed route and rebuilds both byte counters before
        // any evidence is captured. Zeroing (rather than nil-ing) also preserves legacy
        // KVTuner lifecycle telemetry while preventing a long prior run from inflating a short
        // measured run.
        materializationWorkspaceBytes = 0
        attentionWorkspaceBytes = 0
    }

    public func storageSnapshot() -> AffineKVCacheStorageSnapshot? {
        guard let kPayload, let kScales, let kBiases,
            let vPayload, let vScales, let vBiases,
            let materializationWorkspaceBytes, let attentionWorkspaceBytes,
            let attentionOperation
        else { return nil }
        let (workspaceBytes, workspaceOverflow) =
            materializationWorkspaceBytes.addingReportingOverflow(
                attentionWorkspaceBytes)
        precondition(
            !workspaceOverflow,
            "affine total workspace byte count overflow")
        precondition(
            [kScales, kBiases, vScales, vBiases].allSatisfy {
                $0.itemSize == kScales.itemSize
            },
            "affine metadata scalar dtypes differ")
        return AffineKVCacheStorageSnapshot(
            capacityTokens: capacity,
            sequences: kPayload.dim(0),
            kvHeadCount: kPayload.dim(1),
            keyHeadDimension: keyDimension!,
            valueHeadDimension: valueDimension!,
            metadataScalarBytes: kScales.itemSize,
            payloadBytes: kPayload.nbytes + vPayload.nbytes,
            metadataBytes: kScales.nbytes + kBiases.nbytes + vScales.nbytes + vBiases.nbytes,
            controlBytes: offsetArr.nbytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            attentionWorkspaceBytes: attentionWorkspaceBytes,
            workspaceBytes: workspaceBytes,
            attentionOperation: attentionOperation)
    }

    private struct StoredCode {
        let payload: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private func validateInput(keys: MLXArray, values: MLXArray) {
        precondition(keys.shape.count == 4 && values.shape.count == 4, "K/V must be rank 4")
        precondition(
            keys.dim(0) == values.dim(0) && keys.dim(1) == values.dim(1)
                && keys.dim(2) == values.dim(2),
            "K/V batch, head, and token dimensions must match")
        precondition(keys.dim(2) > 0, "K/V update must contain at least one token")
        precondition(offset + keys.dim(2) <= capacity, "K/V update exceeds cache capacity")
        validateDimension(
            keys.dim(3), expected: keyDimension,
            bits: configuration.keyBits, groupSize: configuration.keyGroupSize)
        validateDimension(
            values.dim(3), expected: valueDimension,
            bits: configuration.valueBits, groupSize: configuration.valueGroupSize)
    }

    private func validateDimension(
        _ dimension: Int, expected: Int?, bits: Int, groupSize: Int
    ) {
        precondition(expected == nil || dimension == expected, "KV head dimension changed")
        precondition(
            dimension.isMultiple(of: groupSize),
            "head dimension must be divisible by affine group size")
        precondition(
            (dimension * bits).isMultiple(of: 32),
            "packed affine row must contain a whole number of uint32 words")
    }

    private func encode(_ input: MLXArray, bits: Int, groupSize: Int) -> StoredCode {
        let batch = input.dim(0)
        let heads = input.dim(1)
        let tokens = input.dim(2)
        let dimension = input.dim(3)
        let code = quantized(
            input.reshaped([-1, dimension]),
            groupSize: groupSize, bits: bits, mode: .affine)
        guard let biases = code.biases else {
            preconditionFailure("native affine quantization did not return biases")
        }
        return StoredCode(
            payload: code.wq.reshaped([batch, heads, tokens, code.wq.dim(1)]),
            scales: code.scales.reshaped([batch, heads, tokens, code.scales.dim(1)]),
            biases: biases.reshaped([batch, heads, tokens, biases.dim(1)]))
    }

    private func allocate(keyCode: StoredCode, valueCode: StoredCode) {
        func storage(like array: MLXArray) -> MLXArray {
            MLXArray.zeros(
                [array.dim(0), array.dim(1), capacity, array.dim(3)], dtype: array.dtype)
        }
        kPayload = storage(like: keyCode.payload)
        kScales = storage(like: keyCode.scales)
        kBiases = storage(like: keyCode.biases)
        vPayload = storage(like: valueCode.payload)
        vScales = storage(like: valueCode.scales)
        vBiases = storage(like: valueCode.biases)
    }

    private func materialize(
        payload: MLXArray, scales: MLXArray, biases: MLXArray,
        dimension: Int, bits: Int, groupSize: Int, dtype: DType
    ) -> MLXArray {
        let batch = payload.dim(0)
        let heads = payload.dim(1)
        let tokens = payload.dim(2)
        let packedWidth = payload.dim(3)
        let metadataWidth = scales.dim(3)
        return dequantized(
            payload.reshaped([-1, packedWidth]),
            scales: scales.reshaped([-1, metadataWidth]),
            biases: biases.reshaped([-1, metadataWidth]),
            groupSize: groupSize, bits: bits, mode: .affine, dtype: dtype
        ).reshaped([batch, heads, tokens, dimension])
    }

    private func packedAttention(
        queries: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        precondition(queries.ndim == 4, "queries must be rank 4")
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryTokens = queries.dim(2)
        let dimension = queries.dim(3)
        let kvHeads = kPayload!.dim(1)
        precondition(
            batch == kPayload!.dim(0) && kvHeads > 0
                && queryHeads.isMultiple(of: kvHeads),
            "query and packed-cache batch/head geometry is incompatible")
        precondition(
            dimension == keyDimension,
            "query dimension does not match packed key dimension")

        let repeats = queryHeads / kvHeads
        var scaledQueries = queries * MLXArray(scale).asType(queries.dtype)
        var keyWeights = kPayload!
        var keyScales = kScales!
        var keyBiases: MLXArray? = kBiases!
        var valueWeights = vPayload!
        var valueScales = vScales!
        var valueBiases: MLXArray? = vBiases!
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped([
                batch, kvHeads, repeats, queryTokens, dimension,
            ])
            keyWeights = keyWeights.expandedDimensions(axis: -3)
            keyScales = keyScales.expandedDimensions(axis: -3)
            keyBiases = keyBiases?.expandedDimensions(axis: -3)
            valueWeights = valueWeights.expandedDimensions(axis: -3)
            valueScales = valueScales.expandedDimensions(axis: -3)
            valueBiases = valueBiases?.expandedDimensions(axis: -3)
        }

        var scores = quantizedMM(
            scaledQueries,
            keyWeights,
            scales: keyScales,
            biases: keyBiases,
            transpose: true,
            groupSize: configuration.keyGroupSize,
            bits: configuration.keyBits,
            mode: .affine)
        scores = apply(mask: mask, to: scores, groupedQueryRepeats: repeats)
        let weights = softmax(scores, axis: -1, precise: true)
        let (workspaceBytes, workspaceOverflow) = scores.nbytes
            .addingReportingOverflow(weights.nbytes)
        precondition(
            !workspaceOverflow,
            "split attention workspace byte count overflow")
        attentionWorkspaceBytes = max(
            attentionWorkspaceBytes ?? 0,
            workspaceBytes)
        var output = quantizedMM(
            weights,
            valueWeights,
            scales: valueScales,
            biases: valueBiases,
            transpose: false,
            groupSize: configuration.valueGroupSize,
            bits: configuration.valueBits,
            mode: .affine)
        if repeats > 1 {
            output = output.reshaped([
                batch, queryHeads, queryTokens, valueDimension!,
            ])
        }
        return output
    }

    private func apply(
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        to scores: MLXArray,
        groupedQueryRepeats: Int
    ) -> MLXArray {
        switch mask {
        case .none:
            return maskUnwrittenCacheRows(in: scores)
        case .causal:
            let queryTokens = scores.dim(-2)
            let keyTokens = scores.dim(-1)
            let queryPositions = MLXArray(0 ..< queryTokens)
                + offsetArr - MLXArray([Int32(queryTokens)])
            let keyPositions = MLXArray(0 ..< keyTokens)
            let causallyAllowed = greaterEqual(
                queryPositions.expandedDimensions(axis: -1),
                keyPositions.expandedDimensions(axis: -2))
            let written = keyPositions .< offsetArr
            return MLX.where(
                causallyAllowed & written,
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        case .array(let array):
            return maskUnwrittenCacheRows(in: apply(
                array: array,
                to: scores,
                groupedQueryRepeats: groupedQueryRepeats))
        case .arrays(let arrays):
            precondition(arrays.count <= 1, "only one attention mask array is supported")
            guard let array = arrays.first else {
                return maskUnwrittenCacheRows(in: scores)
            }
            return maskUnwrittenCacheRows(in: apply(
                array: array,
                to: scores,
                groupedQueryRepeats: groupedQueryRepeats))
        }
    }

    private func maskUnwrittenCacheRows(in scores: MLXArray) -> MLXArray {
        let keyPositions = MLXArray(0 ..< scores.dim(-1))
        let written = keyPositions .< offsetArr
        return MLX.where(
            written,
            scores,
            MLXArray(-Float.infinity).asType(scores.dtype))
    }

    private func apply(
        array: MLXArray,
        to scores: MLXArray,
        groupedQueryRepeats: Int
    ) -> MLXArray {
        var array = array
        if groupedQueryRepeats > 1 && array.ndim == 4 {
            array = array.expandedDimensions(axis: 2)
        }
        let validRanks = groupedQueryRepeats > 1 ? [2, 5] : [2, 4]
        precondition(validRanks.contains(array.ndim), "unsupported attention-mask rank")
        if array.dtype == .bool {
            return MLX.where(
                array,
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype))
        }
        return scores + array.asType(scores.dtype)
    }

    // MARK: - KVCache protocol surface unused by fast-mlx's decode path

    public var state: [MLXArray] {
        get { innerState() }
        set { fatalError("AffineKVCache state restore not supported in the spike") }
    }

    public var metaState: [String] {
        get { [""] }
        set {}
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("AffineKVCache.copy() not supported in the spike")
    }
}
