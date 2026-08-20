import XCTest
@testable import HarnessCore

/// Drift-guard for the `.dualGeometrySWA` catalog entries (Baichuan-M1-14B, MiMo-V2-Flash): the
/// decoder's geometry for these two real configs is pinned in `ModelConfigDecoderTests`
/// (`testBaichuanM1_dualGeometrySWA_realConfig`, `testMimoV2Flash_dualGeometrySWA_asymmetricKV_realConfig`).
/// This test asserts the CURATED `ModelArchProfile.catalog` entries decode-match those pinned real
/// configs field-for-field, so the catalog can never silently drift from what the live decoder
/// produces for the same checkpoints, and cross-checks `CapacityModel` against the same hand-derived
/// KV numbers on the CATALOG entry (not just the freshly-decoded one).
final class DualGeometrySWACatalogTests: XCTestCase {
    private func data(_ s: String) -> Data { Data(s.utf8) }

    private func catalogEntry(_ id: String) -> ModelArchProfile? {
        ModelArchProfile.catalog.first(where: { $0.id == id })
    }

    // MARK: - Baichuan-M1-14B

    private func baichuanConfigJSON() -> String {
        """
        {
          "model_type": "baichuan_m1",
          "num_hidden_layers": 40,
          "hidden_size": 5120,
          "num_attention_heads": 20,
          "num_key_value_heads": 2,
          "num_swa_attention_heads": 40,
          "num_swa_key_value_heads": 8,
          "sliding_window": 8192,
          "sliding_window_layers": [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39],
          "conv_window": 2,
          "max_position_embeddings": 32768
        }
        """
    }

    func testBaichuanM1_catalogEntryPresent() throws {
        let entry = try XCTUnwrap(
            catalogEntry("Baichuan-M1-14B"),
            "expected a curated .dualGeometrySWA catalog entry for Baichuan-M1-14B")
        let decoded = try ModelConfigDecoder.decode(
            configJSON: data(baichuanConfigJSON()), safetensorsBytes: 1, id: "Baichuan-M1-14B"
        ).profile

        XCTAssertEqual(entry.modelType, decoded.modelType)
        XCTAssertEqual(entry.nLayers, decoded.nLayers)
        XCTAssertEqual(entry.nAttnLayers, decoded.nAttnLayers)
        XCTAssertEqual(entry.nKVHeads, decoded.nKVHeads)
        XCTAssertEqual(entry.headDim, decoded.headDim)
        XCTAssertEqual(entry.swaKVHeads, decoded.swaKVHeads)
        XCTAssertEqual(entry.swaHeadDim, decoded.swaHeadDim)
        XCTAssertEqual(entry.vHeadDim, decoded.vHeadDim)
        XCTAssertEqual(entry.swaVHeadDim, decoded.swaVHeadDim)
        XCTAssertEqual(entry.slidingWindow, decoded.slidingWindow)
        XCTAssertEqual(entry.fixedStateBytes, decoded.fixedStateBytes)
        XCTAssertEqual(entry.nativeMaxContext, decoded.nativeMaxContext)
    }

    func testBaichuanM1_catalogEntry_capacityModelCrossCheck() throws {
        let entry = try XCTUnwrap(catalogEntry("Baichuan-M1-14B"))

        XCTAssertTrue(entry.isKVDerivable)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(entry, kvQuant: .fp16), 40_960)
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(entry, kvQuant: .fp16), 671_088_640)
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(entry, context: 32_768, kvQuant: .fp16, concurrency: 1),
            2_013_388_800)
    }

    // MARK: - MiMo-V2-Flash

    private func mimoHybridLayerPattern() -> String {
        let globalIdx: Set<Int> = [0, 5, 11, 17, 23, 29, 35, 41, 47]
        return "[" + (0..<48).map { globalIdx.contains($0) ? "0" : "1" }.joined(separator: ",") + "]"
    }

    private func mimoV2FlashConfigJSON() -> String {
        """
        {
          "model_type": "mimo_v2_flash",
          "num_hidden_layers": 48,
          "hidden_size": 4096,
          "num_attention_heads": 64,
          "num_key_value_heads": 4,
          "head_dim": 192,
          "v_head_dim": 128,
          "swa_num_attention_heads": 64,
          "swa_num_key_value_heads": 8,
          "swa_head_dim": 192,
          "swa_v_head_dim": 128,
          "sliding_window": 128,
          "sliding_window_size": 128,
          "max_position_embeddings": 262144,
          "hybrid_layer_pattern": \(mimoHybridLayerPattern())
        }
        """
    }

    func testMimoV2Flash_catalogEntryPresent() throws {
        let entry = try XCTUnwrap(
            catalogEntry("MiMo-V2-Flash"),
            "expected a curated .dualGeometrySWA catalog entry for MiMo-V2-Flash")
        let decoded = try ModelConfigDecoder.decode(
            configJSON: data(mimoV2FlashConfigJSON()), safetensorsBytes: 1, id: "MiMo-V2-Flash"
        ).profile

        XCTAssertEqual(entry.modelType, decoded.modelType)
        XCTAssertEqual(entry.nLayers, decoded.nLayers)
        XCTAssertEqual(entry.nAttnLayers, decoded.nAttnLayers)
        XCTAssertEqual(entry.nKVHeads, decoded.nKVHeads)
        XCTAssertEqual(entry.headDim, decoded.headDim)
        XCTAssertEqual(entry.swaKVHeads, decoded.swaKVHeads)
        XCTAssertEqual(entry.swaHeadDim, decoded.swaHeadDim)
        XCTAssertEqual(entry.vHeadDim, decoded.vHeadDim)
        XCTAssertEqual(entry.swaVHeadDim, decoded.swaVHeadDim)
        XCTAssertEqual(entry.slidingWindow, decoded.slidingWindow)
        XCTAssertEqual(entry.fixedStateBytes, decoded.fixedStateBytes)
        XCTAssertEqual(entry.nativeMaxContext, decoded.nativeMaxContext)
    }

    func testMimoV2Flash_catalogEntry_capacityModelCrossCheck() throws {
        let entry = try XCTUnwrap(catalogEntry("MiMo-V2-Flash"))

        XCTAssertTrue(entry.isKVDerivable)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(entry, kvQuant: .fp16), 23_040)
        XCTAssertEqual(CapacityModel.swaFixedLocalBytes(entry, kvQuant: .fp16), 25_559_040)
        XCTAssertEqual(
            CapacityModel.kvBytesForContext(entry, context: 32_768, kvQuant: .fp16, concurrency: 1),
            780_533_760)
    }
}
