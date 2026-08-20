import XCTest

import HarnessCore

@testable import SpikeCore

final class HybridKVByteAdmissionTests: XCTestCase {
    /// Toy heterogeneous geometry: 4 layers, recurrent at 0/2, dense at 1/3
    /// (denseCount = 2, recurrentCount = 2).
    private func makeGeometry(
        kvHeads: Int = 2, headDim: Int = 4, elementBytes: Int = 2
    ) -> HybridCacheGeometry {
        let map = HybridLayerKindMap(kinds: [
            .recurrentState, .denseAttention, .recurrentState, .denseAttention,
        ])
        let dense = DenseKVGeometry(kvHeads: kvHeads, headDim: headDim, elementBytes: elementBytes)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: 3, convDim: 4, valueHeads: 2, valueHeadDim: 2, keyHeadDim: 2,
            convElementBytes: 2)
        return HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent)!
    }

    func testBytesPerTokenIsDenseLayersOnlyAndStrictlyLessThanNaiveAllLayers() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        let denseCount = geometry.map.denseLayerIndices.count
        let expectedDenseOnly =
            denseCount * 2 * geometry.dense.kvHeads * geometry.dense.headDim
            * geometry.dense.elementBytes
        XCTAssertEqual(plan.bytesPerToken, expectedDenseOnly)
        XCTAssertEqual(plan.bytesPerToken, 64)

        let naiveAllLayers =
            geometry.map.kinds.count * 2 * geometry.dense.kvHeads * geometry.dense.headDim
            * geometry.dense.elementBytes
        XCTAssertLessThan(plan.bytesPerToken, naiveAllLayers)
    }

    func testRecurrentBytesPerRowPinnedToGeometryAuthority() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        XCTAssertEqual(plan.recurrentBytesPerRow, geometry.recurrentBytesTotal)
        XCTAssertEqual(plan.recurrentBytesPerRow, 96)
    }

    func testReservedCapacityRoundsUpToChunkAndCapsAtMaxContextTokens() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        XCTAssertEqual(try plan.reservedCapacity(promptTokens: 3, outputTokens: 2), 8)
        XCTAssertEqual(try plan.reservedCapacity(promptTokens: 1, outputTokens: 1), 4)

        let cappingPlan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 10)
        XCTAssertEqual(try cappingPlan.reservedCapacity(promptTokens: 9, outputTokens: 1), 10)
    }

    func testTransitionEnvelopeIncludesFixedRecurrentTermNeverDropped() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        // bytesPerToken = 64, metadataBytesPerRow = (2+1)*4 = 12, recurrentBytesPerRow = 96.
        let capacities = [4, 8]
        let maxCapacity = 8
        let count = 2
        let denseData = maxCapacity * count * 5 * plan.bytesPerToken
        let denseMetadata = count * 5 * plan.metadataBytesPerRow
        let recurrentEnvelope = count * 5 * plan.recurrentBytesPerRow
        let expected = denseData + denseMetadata + recurrentEnvelope

        XCTAssertEqual(try plan.transitionEnvelopeBytes(capacities: capacities), expected)
        XCTAssertEqual(expected, 6200)

        // The recurrent term is never dropped: a dense-only envelope of the same shape must be
        // strictly smaller.
        let denseOnlyEnvelope = denseData + denseMetadata
        XCTAssertLessThan(
            denseOnlyEnvelope, try plan.transitionEnvelopeBytes(capacities: capacities))
    }

    func testTransitionEnvelopeEmptyCapacitiesReturnsZero() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        XCTAssertEqual(try plan.transitionEnvelopeBytes(capacities: []), 0)
    }

    func testInvalidCapacityThrowsFailClosed() throws {
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)

        XCTAssertThrowsError(try plan.transitionEnvelopeBytes(capacities: [0])) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidCapacity(0))
        }
        XCTAssertThrowsError(try plan.transitionEnvelopeBytes(capacities: [33])) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidCapacity(33))
        }
    }

    func testAllocationChunkGreaterThanMaxContextTokensThrowsInvalidGeometry() throws {
        let geometry = makeGeometry()
        XCTAssertThrowsError(
            try HybridKVByteAdmissionPlan(
                geometry: geometry, allocationChunk: 64, maxContextTokens: 32)
        ) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidGeometry)
        }
    }

    func testAllDenseOrAllRecurrentGeometryStillGuardsBothCounts() throws {
        let map = HybridLayerKindMap(kinds: [.recurrentState, .recurrentState])
        let dense = DenseKVGeometry(kvHeads: 2, headDim: 4, elementBytes: 2)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: 3, convDim: 4, valueHeads: 2, valueHeadDim: 2, keyHeadDim: 2,
            convElementBytes: 2)
        let geometry = HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent)!

        XCTAssertThrowsError(
            try HybridKVByteAdmissionPlan(
                geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        ) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidGeometry)
        }
    }

    func testArithmeticOverflowInBytesPerTokenFailsClosed() throws {
        let map = HybridLayerKindMap(kinds: [
            .recurrentState, .denseAttention, .recurrentState, .denseAttention,
        ])
        let dense = DenseKVGeometry(kvHeads: Int.max, headDim: 128, elementBytes: 2)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: 3, convDim: 4, valueHeads: 2, valueHeadDim: 2, keyHeadDim: 2,
            convElementBytes: 2)
        let geometry = HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent)!

        XCTAssertThrowsError(
            try HybridKVByteAdmissionPlan(
                geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        ) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .arithmeticOverflow)
        }
    }
}
