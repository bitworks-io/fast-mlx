import Foundation
import HarnessCore
import MLX
import MLXLMCommon

public enum AffineKVCacheConfigurationError: Error, Equatable, Sendable {
    case unsupportedBitWidth(Int)
    case unsupportedGroupSize(Int)
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
            materializationWorkspaceBytes: first.materializationWorkspaceBytes)
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
    public let totalPersistentBytes: Int
    public let totalBytes: Int

    /// Capture must run in the cache owner's inference actor after allocation. Persistent
    /// bytes sum every layer; workspace is the largest one-layer full-precision K/V pair,
    /// because transformer layers materialize and consume their caches sequentially.
    public static func capture(
        selection: KVTunerRuntimeSelection,
        caches: [AffineKVCache]
    ) throws -> KVTunerKVCacheTelemetry {
        guard caches.count == selection.layers.count else {
            throw KVTunerKVCacheTelemetryError.layerCountMismatch(
                expected: selection.layers.count, actual: caches.count)
        }

        var snapshots: [AffineKVCacheStorageSnapshot] = []
        snapshots.reserveCapacity(caches.count)
        for (position, pair) in zip(caches, selection.layers).enumerated() {
            let (cache, policy) = pair
            let expected: AffineKVCacheConfiguration
            do {
                expected = try AffineKVCacheConfiguration(
                    keyBits: policy.keyBits,
                    valueBits: policy.valueBits,
                    keyGroupSize: selection.groupSize,
                    valueGroupSize: selection.groupSize)
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
                expected: selection.layers.count, actual: 0)
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
        let workspaceBytes = snapshots.map(\.materializationWorkspaceBytes)
            .max() ?? 0
        let totalPersistentBytes = try checkedSum([
            payloadBytes, metadataBytes, controlBytes,
        ])
        let totalBytes = try checkedSum([
            totalPersistentBytes, workspaceBytes,
        ])

        return KVTunerKVCacheTelemetry(
            artifactSHA256: selection.artifactSHA256,
            matrixID: selection.matrixID,
            cellID: selection.cellID,
            groupSize: selection.groupSize,
            layers: selection.layers,
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
            materializationWorkspaceBytes: workspaceBytes,
            totalPersistentBytes: totalPersistentBytes,
            totalBytes: totalBytes)
    }
}

/// Fixed-capacity, compile-capturable KV cache backed by MLX's native affine packing.
///
/// Incoming K and V rows are independently packed and scattered into fixed-shape buffers.
/// Reads dequantize the full buffers before the existing fused attention path, preserving
/// the same mask, RoPE, growth, truncation, and in-place reset contract as
/// `CompiledKVCache`. This is a correctness-first storage path; compressed-domain attention
/// remains a separate optimization.
///
/// This class intentionally does not conform to `Sendable`: every instance is confined to
/// the inference actor because its MLX state is non-Sendable.
public final class AffineKVCache: KVCache, Updatable {
    public private(set) var capacity: Int
    public let configuration: AffineKVCacheConfiguration

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

    /// Host-side mirror used only by uncompiled prefill/control code. Compiled replays update
    /// `offsetArr` in graph, matching `CompiledKVCache`'s documented contract.
    public private(set) var offset: Int = 0

    public init(capacity: Int, configuration: AffineKVCacheConfiguration) {
        precondition(capacity > 0, "capacity must be positive")
        self.capacity = capacity
        self.configuration = configuration
    }

    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] {
        [kPayload, kScales, kBiases, vPayload, vScales, vBiases].compactMap { $0 }
            + [offsetArr]
    }

    public var ropeOffset: RoPEOffset { .batch(offsetArr) }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
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
        return (materializedKeys, materializedValues)
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
    }

    public func storageSnapshot() -> AffineKVCacheStorageSnapshot? {
        guard let kPayload, let kScales, let kBiases,
            let vPayload, let vScales, let vBiases,
            let materializationWorkspaceBytes
        else { return nil }
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
            materializationWorkspaceBytes: materializationWorkspaceBytes)
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
