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

    // MARK: QSA sparse-indexer aux term — closes
    // docs/task-inbox/2026-09-05-hybrid-admission-aux-term-latent-trap.md. Every expected value below
    // is an independently hand-derived literal (arithmetic shown in comments), never recomputed
    // through `HybridKVByteAdmissionPlan`/`DenseKVGeometry` itself — see the DECISION record's
    // "CORRECTION" section on why self-consistency tests are not evidence.

    /// Flash Next attention-layer shape: 12 dense layers of 48, kvHeads 2, headDim 256, elementBytes 2,
    /// auxPerLayerKeyDim 128 (`indexer_head_dim`). Recurrent side is a minimal filler (not under test
    /// here); only the dense+aux `bytesPerToken` term is exercised.
    private func flashNextShapedGeometry(elementBytes: Int = 2, auxPerLayerKeyDim: Int?) -> HybridCacheGeometry {
        var kinds: [LayerCacheKind] = []
        for _ in 0..<12 { kinds += [.recurrentState, .recurrentState, .recurrentState, .denseAttention] }
        let map = HybridLayerKindMap(kinds: kinds)
        let dense = DenseKVGeometry(
            kvHeads: 2, headDim: 256, elementBytes: elementBytes, auxPerLayerKeyDim: auxPerLayerKeyDim)
        let recurrent = RecurrentStateGeometry(
            convKernelSize: 3, convDim: 4, valueHeads: 2, valueHeadDim: 2, keyHeadDim: 2,
            convElementBytes: 2)
        return HybridCacheGeometry(map: map, dense: dense, recurrent: recurrent)!
    }

    func testFlashNextShape_auxTerm_chargedInBytesPerToken_pinsLiteral() throws {
        // 12 * (2*2*256*2 + 128*2) = 12 * (2048 + 256) = 12 * 2304 = 27_648.
        let geometry = flashNextShapedGeometry(auxPerLayerKeyDim: 128)
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        XCTAssertEqual(plan.bytesPerToken, 27_648)
    }

    func testFlashNextShape_auxNil_regressesToDenseOnly_andDeltaIsExactlyTheAuxTerm() throws {
        // 12 * (2*2*256*2) = 12 * 2048 = 24_576. Delta from the aux-charged plan above must be exactly
        // 12 * 128 * 2 = 3_072 — the regression proof that the aux term is actually charged, not just
        // present in the type.
        let geometryNoAux = flashNextShapedGeometry(auxPerLayerKeyDim: nil)
        let planNoAux = try HybridKVByteAdmissionPlan(
            geometry: geometryNoAux, allocationChunk: 4, maxContextTokens: 32)
        XCTAssertEqual(planNoAux.bytesPerToken, 24_576)

        let geometryWithAux = flashNextShapedGeometry(auxPerLayerKeyDim: 128)
        let planWithAux = try HybridKVByteAdmissionPlan(
            geometry: geometryWithAux, allocationChunk: 4, maxContextTokens: 32)
        XCTAssertEqual(planWithAux.bytesPerToken - planNoAux.bytesPerToken, 3_072)
    }

    func testFlashNextShape_auxTerm_doesNotScaleWithElementBytes_int8KVTier() throws {
        // Dense K+V halves under int8 (elementBytes 1): 12*2*2*256*1 = 12_288. The aux term stays fixed
        // at model dtype regardless of the KV-quant tier: 12*128*2 = 3_072 (NOT 12*128*1 = 1_536).
        // Total: 12_288 + 3_072 = 15_360. This pins the fail-closed rule from
        // docs/task-inbox/2026-09-05-qwen4exp-fit-check-qsa-indexer-term-DECISION.md item 4.
        let geometry = flashNextShapedGeometry(elementBytes: 1, auxPerLayerKeyDim: 128)
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        XCTAssertEqual(plan.bytesPerToken, 15_360)
    }

    func testQwen35Shape_auxAbsent_bytesPerTokenUnchangedFromTodaysBehavior() throws {
        // Pins the pre-existing `testBytesPerTokenIsDenseLayersOnlyAndStrictlyLessThanNaiveAllLayers`
        // expected value (64) using the default `makeGeometry()` (no aux term at all, matching every
        // hybrid family before qwen4_exp) — proves the structural refactor changes nothing for
        // families with no aux term.
        let geometry = makeGeometry()
        let plan = try HybridKVByteAdmissionPlan(
            geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        XCTAssertEqual(plan.bytesPerToken, 64)
    }

    func testAuxOverflow_kvHeadsAndHeadDimNearIntMax_throwsArithmeticOverflowWithoutTrapping() throws {
        let map = HybridLayerKindMap(kinds: [
            .recurrentState, .denseAttention, .recurrentState, .denseAttention,
        ])
        let dense = DenseKVGeometry(
            kvHeads: Int.max / 2, headDim: Int.max / 2, elementBytes: 2, auxPerLayerKeyDim: 128)
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

    func testAuxPerLayerKeyDimZero_throwsInvalidGeometry() throws {
        let geometry = flashNextShapedGeometry(auxPerLayerKeyDim: 0)
        XCTAssertThrowsError(
            try HybridKVByteAdmissionPlan(
                geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        ) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidGeometry)
        }
    }

    func testAuxPerLayerKeyDimNegative_throwsInvalidGeometry() throws {
        let geometry = flashNextShapedGeometry(auxPerLayerKeyDim: -1)
        XCTAssertThrowsError(
            try HybridKVByteAdmissionPlan(
                geometry: geometry, allocationChunk: 4, maxContextTokens: 32)
        ) { error in
            XCTAssertEqual(error as? DenseKVByteAdmissionPlanError, .invalidGeometry)
        }
    }
}
