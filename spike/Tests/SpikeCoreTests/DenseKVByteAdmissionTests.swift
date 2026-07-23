import XCTest

@testable import SpikeCore

final class DenseKVByteAdmissionTests: XCTestCase {
    func testQwen32BGeometryUsesRoundedFullRequestCapacityAndTransitionEnvelope() throws {
        let plan = try DenseKVByteAdmissionPlan(
            layerCount: 64,
            keyValueHeadCount: 8,
            headDimension: 128,
            elementBytes: 2,
            allocationChunk: 256,
            maxContextTokens: 40_960)

        XCTAssertEqual(plan.bytesPerToken, 262_144)
        XCTAssertEqual(plan.metadataBytesPerRow, 260)
        let first = try plan.reservedCapacity(promptTokens: 17, outputTokens: 16)
        let second = try plan.reservedCapacity(promptTokens: 11, outputTokens: 4)
        XCTAssertEqual(first, 256)
        XCTAssertEqual(second, 256)
        XCTAssertEqual(
            try plan.transitionEnvelopeBytes(capacities: [first, second]),
            671_091_240)
    }

    func testTransitionEnvelopeAccountsForPaddingToLargestCapacity() throws {
        let plan = try DenseKVByteAdmissionPlan(
            layerCount: 1,
            keyValueHeadCount: 1,
            headDimension: 1,
            elementBytes: 2,
            allocationChunk: 4,
            maxContextTokens: 32)

        // bytes/token = K+V = 4. The lazy rebuild graph can retain old batch, extracted
        // scalars, tail-zero inputs, padded rows, and final batch at once.
        XCTAssertEqual(
            try plan.transitionEnvelopeBytes(capacities: [4, 8]),
            (5 * 2 * 8) * 4 + 2 * 5 * 8)
        XCTAssertEqual(
            try DenseKVByteAdmissionPlan(
                layerCount: 1,
                keyValueHeadCount: 1,
                headDimension: 1,
                elementBytes: 2,
                allocationChunk: 4,
                maxContextTokens: 10
            ).reservedCapacity(promptTokens: 9, outputTokens: 1),
            10)
    }

    func testInvalidGeometryAndArithmeticOverflowFailClosed() {
        XCTAssertThrowsError(
            try DenseKVByteAdmissionPlan(
                layerCount: 0,
                keyValueHeadCount: 8,
                headDimension: 128,
                elementBytes: 2,
                allocationChunk: 256,
                maxContextTokens: 40_960))

        XCTAssertThrowsError(
            try DenseKVByteAdmissionPlan(
                layerCount: Int.max,
                keyValueHeadCount: Int.max,
                headDimension: 128,
                elementBytes: 2,
                allocationChunk: 256,
                maxContextTokens: 40_960))
    }

    func testAffineAdmissionCountsIndependentPackedKVMetadataAndPhysicalEnd() throws {
        let plan = try AffineKVByteAdmissionPlan(
            configurations: [AffineKVTier.k4v2G64.configuration],
            keyValueHeadCount: 2,
            headDimension: 128,
            metadataScalarBytes: 2,
            allocationChunk: 4,
            maxContextTokens: 32)

        XCTAssertEqual(plan.bytesPerToken, 224)
        XCTAssertEqual(
            try plan.transitionEnvelopeBytes(capacities: [4, 8]),
            18_020)
    }

    func testHeterogeneousAffineAdmissionSumsEveryLayerConfiguration() throws {
        let plan = try AffineKVByteAdmissionPlan(
            configurations: [
                try AffineKVCacheConfiguration(
                    keyBits: 8, valueBits: 4,
                    keyGroupSize: 128, valueGroupSize: 128),
                try AffineKVCacheConfiguration(
                    keyBits: 4, valueBits: 2,
                    keyGroupSize: 128, valueGroupSize: 128),
            ],
            keyValueHeadCount: 2,
            headDimension: 128,
            metadataScalarBytes: 2,
            allocationChunk: 4,
            maxContextTokens: 32)

        XCTAssertEqual(plan.bytesPerToken, 608)
        XCTAssertEqual(
            try plan.transitionEnvelopeBytes(capacities: [4, 8]),
            48_800)
    }

    func testAffinePackedBitArithmeticOverflowFailsClosedWithoutTrapping() {
        XCTAssertThrowsError(
            try AffineKVByteAdmissionPlan(
                configurations: [AffineKVTier.k8v2G64.configuration],
                keyValueHeadCount: 1,
                headDimension: Int.max - 63,
                metadataScalarBytes: 2,
                allocationChunk: 1,
                maxContextTokens: 1)
        ) { error in
            XCTAssertEqual(
                error as? DenseKVByteAdmissionPlanError,
                .arithmeticOverflow)
        }
    }
}
