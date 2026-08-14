import Foundation

public enum KVStorageFormatError: Error, Equatable, Sendable {
    case invalidGeometry
    case invalidFormat
    case invalidAllocation
    case invalidLayerPolicy
    case layerCountMismatch(expected: Int, actual: Int)
    case arithmeticOverflow
}

/// Dense KV geometry shared by the pure accountant and MLX allocation reconciliation.
public struct KVStorageGeometry: Equatable, Sendable {
    public let layerCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int

    public init(layerCount: Int, kvHeadCount: Int, headDimension: Int) {
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
    }
}

/// Bytes for the format's smallest independently encoded unit: one token/head for fp16 and
/// per-token affine, or one full token tile/head for KVarN.
public struct KVStorageUnitLayout: Equatable, Sendable {
    public let tokenCount: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let totalBytes: Int

    public var bytesPerHeadToken: Double { Double(totalBytes) / Double(tokenCount) }
    public var effectiveBitsPerElement: Double {
        (Double(totalBytes) * 8)
            / (Double(tokenCount) * 2 * Double(_headDimension))
    }

    fileprivate init(
        tokenCount: Int, headDimension: Int, payloadBytes: Int, metadataBytes: Int,
        alignmentPaddingBytes: Int, totalBytes: Int
    ) {
        self.tokenCount = tokenCount
        self.payloadBytes = payloadBytes
        self.metadataBytes = metadataBytes
        self.alignmentPaddingBytes = alignmentPaddingBytes
        self.totalBytes = totalBytes
        // A KV token contains `2 * headDimension` scalar elements (one K and one V).
        self._headDimension = headDimension
    }

    private let _headDimension: Int
}

/// Persistent terms for fast-mlx's tight sequential-cache layout. This is not the upstream
/// vLLM block allocator: transient materialization/attention memory and any implementation pool
/// headroom are measured separately through the explicit `workspaceBytes` term.
public struct KVStorageAllocation: Equatable, Sendable {
    public let packedTileSlotsPerSequence: Int
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let workspaceBytes: Int
    public let totalBytes: Int
}

/// Persistent native-affine storage for one complete KVTuner layer policy plus the logical
/// materialization workspace used by the existing sequential materialize-then-attend path.
/// The workspace is the maximum for any one layer, never the sum across persistent layer caches.
/// This is a storage prediction only; it does not imply compressed-domain attention or a speedup.
public struct KVTunerStorageAllocation: Equatable, Sendable {
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let controlBytes: Int
    public let workspaceBytes: Int
    public let totalPersistentBytes: Int
    public let totalBytes: Int
}

/// Format-aware, integer byte accounting. Unlike `KVQuantTier.bytesPerElement`, this is suitable
/// for fast-mlx evidence: it counts distinct K/V widths, metadata, caller-selected local record
/// alignment, and the local KVarN design's fixed fp16 state. Actual MLX array bytes must reconcile
/// with this prediction before a format is measured.
public struct KVStorageFormat: Equatable, Sendable {
    private enum Kind: Equatable, Sendable {
        case fp16
        case affine
        case kvarn
    }

    private let kind: Kind
    private let keyBits: Int
    private let valueBits: Int
    private let groupSize: Int
    private let sinkTokens: Int
    private let metadataScalarBytes: Int
    private let alignment: Int

    public static let fp16 = KVStorageFormat(
        kind: .fp16, keyBits: 16, valueBits: 16, groupSize: 1, sinkTokens: 0,
        metadataScalarBytes: 0, alignment: 1)

    public static func affine(
        keyBits: Int, valueBits: Int, groupSize: Int, metadataScalarBytes: Int
    ) -> KVStorageFormat {
        KVStorageFormat(
            kind: .affine, keyBits: keyBits, valueBits: valueBits,
            groupSize: groupSize, sinkTokens: 0,
            metadataScalarBytes: metadataScalarBytes, alignment: 1)
    }

    public static func kvarn(
        keyBits: Int, valueBits: Int, groupSize: Int, sinkTokens: Int,
        metadataScalarBytes: Int, alignment: Int
    ) -> KVStorageFormat {
        KVStorageFormat(
            kind: .kvarn, keyBits: keyBits, valueBits: valueBits,
            groupSize: groupSize, sinkTokens: sinkTokens,
            metadataScalarBytes: metadataScalarBytes, alignment: alignment)
    }

    /// Predicts the current native-affine layout for a complete, canonically ordered KVTuner
    /// policy. The caller supplies the observed scale/bias scalar width because native MLX
    /// preserves the cache input dtype; each persistent layer owns one Int32 offset. Passing
    /// runtime telemetry here makes the pure prediction reconcile without assuming fp16.
    public static func kvtunerAllocation(
        layerPolicy: [KVLayerPrecision],
        groupSize: Int,
        geometry: KVStorageGeometry,
        capacityTokens: Int,
        sequences: Int,
        metadataScalarBytes: Int,
        maximumLayerWorkspaceBytes: Int
    ) throws -> KVTunerStorageAllocation {
        guard geometry.layerCount > 0, geometry.kvHeadCount > 0,
            geometry.headDimension > 0
        else { throw KVStorageFormatError.invalidGeometry }
        guard capacityTokens > 0, sequences > 0, metadataScalarBytes > 0,
            maximumLayerWorkspaceBytes >= 0
        else {
            throw KVStorageFormatError.invalidAllocation
        }
        guard layerPolicy.count == geometry.layerCount else {
            throw KVStorageFormatError.layerCountMismatch(
                expected: geometry.layerCount, actual: layerPolicy.count)
        }
        guard kvtunerGroupSizes.contains(groupSize) else {
            throw KVStorageFormatError.invalidFormat
        }

        let perLayerGeometry = KVStorageGeometry(
            layerCount: 1,
            kvHeadCount: geometry.kvHeadCount,
            headDimension: geometry.headDimension)
        var payloadBytes = 0
        var metadataBytes = 0
        for (position, precision) in layerPolicy.enumerated() {
            guard precision.layer == position else {
                throw KVStorageFormatError.invalidLayerPolicy
            }
            guard kvtunerPrecisionPairs.contains(
                PrecisionPair(keyBits: precision.keyBits, valueBits: precision.valueBits))
            else { throw KVStorageFormatError.invalidFormat }

            let allocation = try affine(
                keyBits: precision.keyBits,
                valueBits: precision.valueBits,
                groupSize: groupSize,
                metadataScalarBytes: metadataScalarBytes
            ).allocation(
                geometry: perLayerGeometry,
                capacityTokens: capacityTokens,
                sequences: sequences,
                workspaceBytes: 0)
            payloadBytes = try sum([payloadBytes, allocation.payloadBytes])
            metadataBytes = try sum([metadataBytes, allocation.metadataBytes])
        }

        let controlBytes = try product([
            layerPolicy.count, kvtunerControlBytesPerLayer,
        ])
        let totalPersistentBytes = try sum([
            payloadBytes, metadataBytes, controlBytes,
        ])
        let totalBytes = try sum([
            totalPersistentBytes, maximumLayerWorkspaceBytes,
        ])
        return KVTunerStorageAllocation(
            payloadBytes: payloadBytes,
            metadataBytes: metadataBytes,
            controlBytes: controlBytes,
            workspaceBytes: maximumLayerWorkspaceBytes,
            totalPersistentBytes: totalPersistentBytes,
            totalBytes: totalBytes)
    }

    public func unitLayout(geometry: KVStorageGeometry) throws -> KVStorageUnitLayout {
        try validate(geometry: geometry)
        let d = geometry.headDimension
        switch kind {
        case .fp16:
            let payload = try Self.product([d, 2, 2])
            return KVStorageUnitLayout(
                tokenCount: 1, headDimension: d,
                payloadBytes: payload, metadataBytes: 0, alignmentPaddingBytes: 0,
                totalBytes: payload)

        case .affine:
            let keyPayload = try Self.packedBytes(elements: d, bits: keyBits)
            let valuePayload = try Self.packedBytes(elements: d, bits: valueBits)
            let payload = try Self.sum([keyPayload, valuePayload])
            let groups = d / groupSize
            // Each K and V group has one scale + one bias.
            let metadata = try Self.product([groups, 2, 2, metadataScalarBytes])
            let total = try Self.sum([payload, metadata])
            return KVStorageUnitLayout(
                tokenCount: 1, headDimension: d, payloadBytes: payload,
                metadataBytes: metadata, alignmentPaddingBytes: 0,
                totalBytes: total)

        case .kvarn:
            let elements = try Self.product([d, groupSize])
            let keyPayload = try Self.packedBytes(elements: elements, bits: keyBits)
            let valuePayload = try Self.packedBytes(elements: elements, bits: valueBits)
            let payload = try Self.sum([keyPayload, valuePayload])
            // K: 2D+G fp16 values. V: D+2G fp16 values.
            let metadataValues = try Self.sum([
                try Self.sum([try Self.product([2, d]), groupSize]),
                try Self.sum([d, try Self.product([2, groupSize])]),
            ])
            let metadata = try Self.product([metadataValues, metadataScalarBytes])
            let raw = try Self.sum([payload, metadata])
            let aligned = try Self.alignUp(raw, to: alignment)
            return KVStorageUnitLayout(
                tokenCount: groupSize, headDimension: d, payloadBytes: payload,
                metadataBytes: metadata, alignmentPaddingBytes: aligned - raw,
                totalBytes: aligned)
        }
    }

    public func allocation(
        geometry: KVStorageGeometry, capacityTokens: Int, sequences: Int,
        workspaceBytes: Int
    ) throws -> KVStorageAllocation {
        guard capacityTokens > 0, sequences > 0, workspaceBytes >= 0 else {
            throw KVStorageFormatError.invalidAllocation
        }
        let unit = try unitLayout(geometry: geometry)
        let layerHeadSequences = try Self.product([
            geometry.layerCount, geometry.kvHeadCount, sequences,
        ])

        let slots: Int
        let unitCount: Int
        let sinkBytes: Int
        let tailBytes: Int
        switch kind {
        case .fp16, .affine:
            slots = 0
            unitCount = try Self.product([capacityTokens, layerHeadSequences])
            sinkBytes = 0
            tailBytes = 0

        case .kvarn:
            let remaining = max(0, capacityTokens - sinkTokens)
            slots = try Self.ceilDiv(remaining, by: groupSize)
            unitCount = try Self.product([slots, layerHeadSequences])
            let fp16BytesPerTokenHead = try Self.product([geometry.headDimension, 2, 2])
            sinkBytes = sinkTokens == 0 ? 0 : try Self.product([
                sinkTokens, fp16BytesPerTokenHead, layerHeadSequences,
            ])
            // The local correctness-first cache reserves one full fp16 in-progress tail per
            // active sequence, independent of how many positions are currently occupied.
            tailBytes = try Self.product([
                groupSize, fp16BytesPerTokenHead, layerHeadSequences,
            ])
        }

        let payload = try Self.product([unit.payloadBytes, unitCount])
        let metadata = try Self.product([unit.metadataBytes, unitCount])
        let padding = try Self.product([unit.alignmentPaddingBytes, unitCount])
        let total = try Self.sum([
            payload, metadata, padding, sinkBytes, tailBytes, workspaceBytes,
        ])
        return KVStorageAllocation(
            packedTileSlotsPerSequence: slots,
            payloadBytes: payload,
            metadataBytes: metadata,
            alignmentPaddingBytes: padding,
            fp16SinkBytes: sinkBytes,
            fp16TailBytes: tailBytes,
            workspaceBytes: workspaceBytes,
            totalBytes: total)
    }

    private func validate(geometry: KVStorageGeometry) throws {
        guard geometry.layerCount > 0, geometry.kvHeadCount > 0,
            geometry.headDimension > 0
        else { throw KVStorageFormatError.invalidGeometry }
        switch kind {
        case .fp16:
            return
        case .affine:
            guard Self.supportedBits.contains(keyBits),
                Self.supportedBits.contains(valueBits), groupSize > 0,
                groupSize.isPowerOfTwo, geometry.headDimension.isMultiple(of: groupSize),
                metadataScalarBytes > 0
            else { throw KVStorageFormatError.invalidFormat }
            let keyBitCount = try Self.product([geometry.headDimension, keyBits])
            let valueBitCount = try Self.product([geometry.headDimension, valueBits])
            guard keyBitCount.isMultiple(of: 32), valueBitCount.isMultiple(of: 32)
            else { throw KVStorageFormatError.invalidFormat }
        case .kvarn:
            // Keep the evidence accountant narrower than the generic math: these are the
            // configurations exposed by the pinned runnable backend and covered by this gate.
            guard keyBits == 4, [2, 4].contains(valueBits),
                [64, 128].contains(groupSize),
                [128, 256, 512].contains(geometry.headDimension),
                sinkTokens == groupSize, metadataScalarBytes == 2,
                alignment > 0, alignment.isPowerOfTwo,
                groupSize.isMultiple(of: 8 / keyBits),
                geometry.headDimension.isMultiple(of: 8 / valueBits)
            else { throw KVStorageFormatError.invalidFormat }
        }
    }

    private static let supportedBits: Set<Int> = [2, 4, 8]

    private struct PrecisionPair: Hashable {
        let keyBits: Int
        let valueBits: Int
    }

    private static let kvtunerPrecisionPairs: Set<PrecisionPair> = [
        PrecisionPair(keyBits: 8, valueBits: 4),
        PrecisionPair(keyBits: 8, valueBits: 2),
        PrecisionPair(keyBits: 4, valueBits: 2),
    ]
    private static let kvtunerGroupSizes: Set<Int> = [64, 128]
    private static let kvtunerControlBytesPerLayer = MemoryLayout<Int32>.size

    private static func packedBytes(elements: Int, bits: Int) throws -> Int {
        let bitCount = try product([elements, bits])
        guard bitCount.isMultiple(of: 8) else { throw KVStorageFormatError.invalidFormat }
        return bitCount / 8
    }

    private static func product(_ values: [Int]) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else { throw KVStorageFormatError.arithmeticOverflow }
            result = next
        }
        return result
    }

    private static func sum(_ values: [Int]) throws -> Int {
        var result = 0
        for value in values {
            let (next, overflow) = result.addingReportingOverflow(value)
            guard !overflow else { throw KVStorageFormatError.arithmeticOverflow }
            result = next
        }
        return result
    }

    private static func alignUp(_ value: Int, to alignment: Int) throws -> Int {
        let remainder = value % alignment
        guard remainder != 0 else { return value }
        return try sum([value, alignment - remainder])
    }

    private static func ceilDiv(_ value: Int, by divisor: Int) throws -> Int {
        guard value >= 0, divisor > 0 else { throw KVStorageFormatError.invalidAllocation }
        guard value > 0 else { return 0 }
        return try sum([value, divisor - 1]) / divisor
    }
}

private extension Int {
    var isPowerOfTwo: Bool { self > 0 && (self & (self - 1)) == 0 }
}
