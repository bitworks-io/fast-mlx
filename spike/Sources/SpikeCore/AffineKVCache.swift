import Foundation
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
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.keyGroupSize = keyGroupSize
        self.valueGroupSize = valueGroupSize
    }

    private static let supportedBits: Set<Int> = [2, 4, 8]
    private static let supportedGroupSizes: Set<Int> = [32, 64, 128]
}

/// Actual persistent MLX-array bytes owned by an affine cache after allocation.
///
/// `dataArrayBytes` reconciles directly with `KVStorageFormat.affine`; the in-graph offset
/// is reported separately as control state so capacity claims never hide implementation
/// bytes that are not part of the packed data layout.
public struct AffineKVCacheStorageSnapshot: Equatable, Sendable {
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int

    public var dataArrayBytes: Int { payloadBytes + metadataBytes }
    public var totalPersistentBytes: Int { dataArrayBytes + controlBytes }
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

        return (
            materialize(
                payload: kPayload!, scales: kScales!, biases: kBiases!,
                dimension: keyDimension!, bits: configuration.keyBits,
                groupSize: configuration.keyGroupSize, dtype: keyOutputDType!),
            materialize(
                payload: vPayload!, scales: vScales!, biases: vBiases!,
                dimension: valueDimension!, bits: configuration.valueBits,
                groupSize: configuration.valueGroupSize, dtype: valueOutputDType!)
        )
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
            let vPayload, let vScales, let vBiases
        else { return nil }
        return AffineKVCacheStorageSnapshot(
            payloadBytes: kPayload.nbytes + vPayload.nbytes,
            metadataBytes: kScales.nbytes + kBiases.nbytes + vScales.nbytes + vBiases.nbytes,
            controlBytes: offsetArr.nbytes)
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
