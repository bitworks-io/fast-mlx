import XCTest
@testable import HarnessCore

/// S1 of the hybrid continuous-batching build (design of record: the continuous-batching
/// heterogeneous-cache design). Pins the
/// structural per-layer cache-kind map: which layers are dense-attention vs recurrent, and the
/// `firstDenseLayerIndex` landmine (layer 0 is recurrent on qwen3_5, so KV accounting must not key
/// off layer 0). No MLX, no model load — pure value type.
final class HybridLayerKindMapTests: XCTestCase {

    // MARK: qwen3_5 interval derivation (mirrors the vendored (i+1) % interval formula)

    func testQwen35Interval4EightLayersYieldsThreeRecurrentThenDensePattern() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 8, fullAttentionInterval: 4))
        // (i+1) % 4 == 0 → dense at i = 3, 7; recurrent elsewhere → [R,R,R,D,R,R,R,D].
        XCTAssertEqual(map.kinds, [
            .recurrentState, .recurrentState, .recurrentState, .denseAttention,
            .recurrentState, .recurrentState, .recurrentState, .denseAttention,
        ])
        XCTAssertEqual(map.denseLayerIndices, [3, 7])
        XCTAssertEqual(map.recurrentLayerIndices, [0, 1, 2, 4, 5, 6])
    }

    // The load-bearing landmine: layer 0 is RECURRENT on qwen3_5, so capacity/geometry accounting must
    // key off firstDenseLayerIndex, never a hardcoded layer 0.
    func testFirstDenseLayerIndexIsNotZeroForQwen35() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 8))
        XCTAssertEqual(map.firstDenseLayerIndex, 3, "layer 0 is a linear layer; the first dense layer is 3")
        XCTAssertEqual(map.kind(atLayer: 0), .recurrentState)
        XCTAssertEqual(map.kind(atLayer: 3), .denseAttention)
    }

    func testQwen35IsHeterogeneous() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 8))
        XCTAssertTrue(map.isHeterogeneous, "an interleaved hybrid genuinely needs the heterogeneous path")
    }

    func testQwen35DefaultIntervalIsFour() throws {
        let explicit = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 12, fullAttentionInterval: 4))
        let defaulted = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 12))
        XCTAssertEqual(explicit, defaulted)
        XCTAssertEqual(defaulted.denseLayerIndices, [3, 7, 11])
    }

    func testQwen35IntervalOfOneIsAllDense() throws {
        // interval 1 → (i+1) % 1 == 0 for all i → every layer dense (degenerate but well-defined).
        let map = try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 5, fullAttentionInterval: 1))
        XCTAssertEqual(map.denseLayerIndices, [0, 1, 2, 3, 4])
        XCTAssertEqual(map.firstDenseLayerIndex, 0)
        XCTAssertFalse(map.isHeterogeneous, "all-dense is not heterogeneous; route through the uniform path")
    }

    // MARK: fail-closed derivation

    func testQwen35FailsClosedOnNonPositiveInputs() {
        XCTAssertNil(HybridLayerKindMap.qwen35(layerCount: 0))
        XCTAssertNil(HybridLayerKindMap.qwen35(layerCount: -1))
        XCTAssertNil(HybridLayerKindMap.qwen35(layerCount: 8, fullAttentionInterval: 0))
        XCTAssertNil(HybridLayerKindMap.qwen35(layerCount: 8, fullAttentionInterval: -4))
    }

    // MARK: explicit attention-index factory (the general decoder-integration seam)

    func testFromAttentionIndicesBuildsTheComplementAsRecurrent() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.from(attentionLayerIndices: [3, 7], layerCount: 8))
        XCTAssertEqual(map.denseLayerIndices, [3, 7])
        XCTAssertEqual(map.recurrentLayerIndices, [0, 1, 2, 4, 5, 6])
        // Agrees with the qwen3_5 interval-4 derivation for the same attention set.
        XCTAssertEqual(map, try XCTUnwrap(HybridLayerKindMap.qwen35(layerCount: 8, fullAttentionInterval: 4)))
    }

    func testFromAttentionIndicesFailsClosedOnOutOfRangeIndex() {
        XCTAssertNil(HybridLayerKindMap.from(attentionLayerIndices: [3, 8], layerCount: 8),
                     "an attention index at/after layerCount is malformed → fail closed")
        XCTAssertNil(HybridLayerKindMap.from(attentionLayerIndices: [-1], layerCount: 8))
        XCTAssertNil(HybridLayerKindMap.from(attentionLayerIndices: [0], layerCount: 0))
    }

    func testFromAttentionIndicesEmptySetIsAllRecurrent() throws {
        let map = try XCTUnwrap(HybridLayerKindMap.from(attentionLayerIndices: [], layerCount: 4))
        XCTAssertEqual(map.denseLayerIndices, [])
        XCTAssertNil(map.firstDenseLayerIndex, "a pure-recurrent model has no dense layer to key on")
        XCTAssertTrue(map.isHeterogeneous)
    }

    // MARK: value semantics

    func testEquatableAndLayerCount() {
        let a = HybridLayerKindMap(kinds: [.recurrentState, .denseAttention])
        let b = HybridLayerKindMap(kinds: [.recurrentState, .denseAttention])
        let c = HybridLayerKindMap(kinds: [.denseAttention, .recurrentState])
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertEqual(a.layerCount, 2)
    }
}
