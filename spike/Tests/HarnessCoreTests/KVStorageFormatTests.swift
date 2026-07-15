import XCTest
@testable import HarnessCore

final class KVStorageFormatTests: XCTestCase {
    private let d128 = KVStorageGeometry(layerCount: 1, kvHeadCount: 1, headDimension: 128)

    func testKVarNK4V2G128LayoutIncludesEveryMetadataByte() throws {
        let format = KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8)

        let layout = try format.unitLayout(geometry: d128)

        XCTAssertEqual(layout.tokenCount, 128)
        XCTAssertEqual(layout.payloadBytes, 12_288)
        XCTAssertEqual(layout.metadataBytes, 1_536)
        XCTAssertEqual(layout.alignmentPaddingBytes, 0)
        XCTAssertEqual(layout.totalBytes, 13_824)
        XCTAssertEqual(layout.bytesPerHeadToken, 108.0, accuracy: 0)
        XCTAssertEqual(layout.effectiveBitsPerElement, 3.375, accuracy: 0)
    }

    func testKVarNG64PaysMoreMetadataPerToken() throws {
        let format = KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 64, sinkTokens: 64,
            metadataScalarBytes: 2, alignment: 8)

        let layout = try format.unitLayout(geometry: d128)

        XCTAssertEqual(layout.payloadBytes, 6_144)
        XCTAssertEqual(layout.metadataBytes, 1_152)
        XCTAssertEqual(layout.totalBytes, 7_296)
        XCTAssertEqual(layout.bytesPerHeadToken, 114.0, accuracy: 0)
        XCTAssertEqual(layout.effectiveBitsPerElement, 3.5625, accuracy: 0)
    }

    func testAffineK4V2CountsNativeScaleAndBiasForBothTensors() throws {
        let g128 = try KVStorageFormat.affine(
            keyBits: 4, valueBits: 2, groupSize: 128, metadataScalarBytes: 2
        ).unitLayout(geometry: d128)
        let g64 = try KVStorageFormat.affine(
            keyBits: 4, valueBits: 2, groupSize: 64, metadataScalarBytes: 2
        ).unitLayout(geometry: d128)

        XCTAssertEqual(g128.tokenCount, 1)
        XCTAssertEqual(g128.payloadBytes, 96)
        XCTAssertEqual(g128.metadataBytes, 8)
        XCTAssertEqual(g128.totalBytes, 104)
        XCTAssertEqual(g64.payloadBytes, 96)
        XCTAssertEqual(g64.metadataBytes, 16)
        XCTAssertEqual(g64.totalBytes, 112)
    }

    func testKVTunerAllocationSumsMixedPersistentLayersButKeepsOneWorkspace() throws {
        let geometry = KVStorageGeometry(
            layerCount: 3, kvHeadCount: 2, headDimension: 128)
        let policy = [
            KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            KVLayerPrecision(layer: 1, keyBits: 8, valueBits: 2),
            KVLayerPrecision(layer: 2, keyBits: 4, valueBits: 2),
        ]

        let allocation = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: policy,
            groupSize: 64,
            geometry: geometry,
            capacityTokens: 10,
            sequences: 2,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 20_480)

        XCTAssertEqual(allocation.payloadBytes, 17_920)
        XCTAssertEqual(allocation.metadataBytes, 1_920)
        XCTAssertEqual(allocation.controlBytes, 12)
        XCTAssertEqual(allocation.workspaceBytes, 20_480)
        XCTAssertEqual(allocation.totalPersistentBytes, 19_852)
        XCTAssertEqual(allocation.totalBytes, 40_332)

        let float32Metadata = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: policy,
            groupSize: 64,
            geometry: geometry,
            capacityTokens: 10,
            sequences: 2,
            metadataScalarBytes: 4,
            maximumLayerWorkspaceBytes: 40_960)
        XCTAssertEqual(float32Metadata.payloadBytes, allocation.payloadBytes)
        XCTAssertEqual(float32Metadata.metadataBytes, allocation.metadataBytes * 2)
        XCTAssertEqual(float32Metadata.controlBytes, allocation.controlBytes)
        XCTAssertEqual(float32Metadata.workspaceBytes, 40_960)
    }

    func testKVTunerAllocationRejectsIncompleteGeometryAndLayerCountMismatch() {
        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [],
            groupSize: 64,
            geometry: KVStorageGeometry(
                layerCount: 0, kvHeadCount: 1, headDimension: 128),
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .invalidGeometry)
        }

        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: KVStorageGeometry(
                layerCount: 2, kvHeadCount: 1, headDimension: 128),
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(
                error as? KVStorageFormatError,
                .layerCountMismatch(expected: 2, actual: 1))
        }

        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: d128,
            capacityTokens: 1,
            sequences: 0,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .invalidAllocation)
        }

        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: d128,
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 0,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .invalidAllocation)
        }
    }

    func testKVTunerAllocationRejectsNoncanonicalAndUnsupportedPolicy() {
        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 1, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: d128,
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .invalidLayerPolicy)
        }

        for (policy, groupSize) in [
            ([KVLayerPrecision(layer: 0, keyBits: 4, valueBits: 4)], 64),
            ([KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4)], 32),
        ] {
            XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
                layerPolicy: policy,
                groupSize: groupSize,
                geometry: d128,
                capacityTokens: 1,
                sequences: 1,
                metadataScalarBytes: 2,
                maximumLayerWorkspaceBytes: 0
            )) { error in
                XCTAssertEqual(error as? KVStorageFormatError, .invalidFormat)
            }
        }
    }

    func testKVTunerAllocationOverflowFailsClosedWithoutTrapping() {
        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: KVStorageGeometry(
                layerCount: 1, kvHeadCount: Int.max, headDimension: 128),
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: 0
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .arithmeticOverflow)
        }

        XCTAssertThrowsError(try KVStorageFormat.kvtunerAllocation(
            layerPolicy: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ],
            groupSize: 64,
            geometry: d128,
            capacityTokens: 1,
            sequences: 1,
            metadataScalarBytes: 2,
            maximumLayerWorkspaceBytes: Int.max
        )) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .arithmeticOverflow)
        }
    }

    func testKVarNAllocationIncludesSinkTailPackedSlotsAndWorkspace() throws {
        let geometry = KVStorageGeometry(layerCount: 2, kvHeadCount: 3, headDimension: 128)
        let format = KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8)

        let allocation = try format.allocation(
            geometry: geometry, capacityTokens: 384, sequences: 2, workspaceBytes: 4_096)

        XCTAssertEqual(allocation.packedTileSlotsPerSequence, 2)
        XCTAssertEqual(allocation.payloadBytes, 294_912)
        XCTAssertEqual(allocation.metadataBytes, 36_864)
        XCTAssertEqual(allocation.alignmentPaddingBytes, 0)
        XCTAssertEqual(allocation.fp16SinkBytes, 786_432)
        XCTAssertEqual(allocation.fp16TailBytes, 786_432)
        XCTAssertEqual(allocation.workspaceBytes, 4_096)
        XCTAssertEqual(allocation.totalBytes, 1_908_736)
    }

    func testPartialCapacityStillReservesExplicitTailBlock() throws {
        let format = KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8)

        let allocation = try format.allocation(
            geometry: d128, capacityTokens: 129, sequences: 1, workspaceBytes: 0)

        XCTAssertEqual(allocation.packedTileSlotsPerSequence, 1)
        XCTAssertEqual(allocation.fp16SinkBytes, 65_536)
        XCTAssertEqual(allocation.fp16TailBytes, 65_536)
        XCTAssertEqual(allocation.totalBytes, 144_896)
    }

    func testFp16AndAffineDoNotInventSinkOrTailStorage() throws {
        let fp16 = try KVStorageFormat.fp16.allocation(
            geometry: d128, capacityTokens: 10, sequences: 1, workspaceBytes: 0)
        let affine = try KVStorageFormat.affine(
            keyBits: 4, valueBits: 2, groupSize: 128, metadataScalarBytes: 2
        ).allocation(geometry: d128, capacityTokens: 10, sequences: 1, workspaceBytes: 0)

        XCTAssertEqual(fp16.payloadBytes, 5_120)
        XCTAssertEqual(fp16.totalBytes, 5_120)
        XCTAssertEqual(affine.payloadBytes, 960)
        XCTAssertEqual(affine.metadataBytes, 80)
        XCTAssertEqual(affine.totalBytes, 1_040)
        XCTAssertEqual(affine.fp16SinkBytes, 0)
        XCTAssertEqual(affine.fp16TailBytes, 0)
    }

    func testInvalidGeometryPackingAndOverflowFailClosed() {
        XCTAssertThrowsError(try KVStorageFormat.affine(
            keyBits: 3, valueBits: 2, groupSize: 64, metadataScalarBytes: 2
        ).unitLayout(geometry: d128))
        XCTAssertThrowsError(try KVStorageFormat.affine(
            keyBits: 4, valueBits: 2, groupSize: 96, metadataScalarBytes: 2
        ).unitLayout(geometry: d128))
        XCTAssertThrowsError(try KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 127, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8
        ).unitLayout(geometry: d128))
        XCTAssertThrowsError(try KVStorageFormat.kvarn(
            keyBits: 2, valueBits: 2, groupSize: 2, sinkTokens: 2,
            metadataScalarBytes: 2, alignment: 8
        ).unitLayout(geometry: KVStorageGeometry(
            layerCount: 1, kvHeadCount: 1, headDimension: 2)))
        XCTAssertThrowsError(try KVStorageFormat.fp16.allocation(
            geometry: KVStorageGeometry(
                layerCount: Int.max, kvHeadCount: Int.max, headDimension: Int.max),
            capacityTokens: Int.max, sequences: Int.max, workspaceBytes: 0))
    }

    func testKVarNAccountantRejectsUnrunnablePresetGeometry() {
        let unsupported: [KVStorageFormat] = [
            .kvarn(
                keyBits: 8, valueBits: 2, groupSize: 128, sinkTokens: 128,
                metadataScalarBytes: 2, alignment: 8),
            .kvarn(
                keyBits: 4, valueBits: 8, groupSize: 128, sinkTokens: 128,
                metadataScalarBytes: 2, alignment: 8),
            .kvarn(
                keyBits: 4, valueBits: 2, groupSize: 32, sinkTokens: 32,
                metadataScalarBytes: 2, alignment: 8),
            .kvarn(
                keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 256,
                metadataScalarBytes: 2, alignment: 8),
            .kvarn(
                keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
                metadataScalarBytes: 4, alignment: 8),
        ]

        for format in unsupported {
            XCTAssertThrowsError(try format.unitLayout(geometry: d128))
        }
        XCTAssertThrowsError(try KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128, sinkTokens: 128,
            metadataScalarBytes: 2, alignment: 8
        ).unitLayout(geometry: KVStorageGeometry(
            layerCount: 1, kvHeadCount: 1, headDimension: 64)))
    }

    func testAggregateLayoutOverflowFailsClosed() {
        let headDimension = (Int.max / 5 / 4) * 4
        let geometry = KVStorageGeometry(
            layerCount: 1, kvHeadCount: 1, headDimension: headDimension)
        let format = KVStorageFormat.affine(
            keyBits: 8, valueBits: 8, groupSize: 1, metadataScalarBytes: 1)

        XCTAssertThrowsError(try format.unitLayout(geometry: geometry)) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .arithmeticOverflow)
        }
    }

    func testAffineBitCountOverflowFailsClosedWithoutTrapping() {
        let geometry = KVStorageGeometry(
            layerCount: 1, kvHeadCount: 1, headDimension: Int.max / 2)
        let format = KVStorageFormat.affine(
            keyBits: 8, valueBits: 8, groupSize: 1, metadataScalarBytes: 1)

        XCTAssertThrowsError(try format.unitLayout(geometry: geometry)) { error in
            XCTAssertEqual(error as? KVStorageFormatError, .arithmeticOverflow)
        }
    }

    func testEffectiveBitsAvoidsOverflowingIntegerIntermediates() throws {
        let geometry = KVStorageGeometry(
            layerCount: 1, kvHeadCount: 1, headDimension: Int.max / 16)

        let layout = try KVStorageFormat.fp16.unitLayout(geometry: geometry)

        XCTAssertEqual(layout.effectiveBitsPerElement, 16.0, accuracy: 1e-12)
    }
}
