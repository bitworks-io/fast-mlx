import XCTest
@testable import SpikeCore
import HarnessCore

/// S1-continued of the hybrid continuous-batching build (design of record: the continuous-batching
/// heterogeneous-cache design, §2.4). Pins the
/// FLAG-GATED qwen3_5 arm of the unforgeable `DenseContinuousBatchModelProof`: with the opt-in OFF (the
/// default everywhere), qwen3_5 is rejected exactly as today so the scalar fallback keeps working; with
/// it ON, the proof carries the derived `HybridCacheGeometry` and reports family `.qwen35`.
final class HybridQwen35ProofArmTests: XCTestCase {

    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    /// VL-wrapped qwen3_5: model_type at root, ALL text geometry (incl. dtype, max_position_embeddings,
    /// vocab_size) under text_config. 48 layers, interval 4 → dense at {3,7,...,47}, layer 0 recurrent.
    private func qwen35VLConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    // MARK: default OFF — unchanged behavior

    func testQwen35RejectedByDefaultEvenWithFullGeometry() throws {
        let dir = try writeConfig(qwen35VLConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try DenseContinuousBatchModelProof.verifying(modelDirectory: dir)) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, .unsupportedModelFamily("qwen3_5"))
        }
    }

    // MARK: flag ON — hybrid proof

    func testFlaggedQwen35ProofCarriesGeometryAndFamily() throws {
        let dir = try writeConfig(qwen35VLConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)

        XCTAssertEqual(proof.verifiedModelFamily, .qwen35)
        XCTAssertEqual(proof.maximumContextTokens, 262144)
        XCTAssertEqual(proof.verifiedLayerCount, 48)
        XCTAssertEqual(proof.verifiedKeyValueHeadCount, 8)
        // dtype lived ONLY under text_config → elementBytes must still resolve to 2 (bfloat16).
        let geo = try XCTUnwrap(proof.verifiedHybridGeometry)
        XCTAssertEqual(geo.dense.elementBytes, 2)
        XCTAssertEqual(geo.dense.headDim, 128)
        // The layer-0 landmine, re-pinned at proof level: layer 0 is recurrent; first dense is 3.
        XCTAssertEqual(geo.map.firstDenseLayerIndex, 3)
        XCTAssertEqual(geo.map.denseLayerIndices.count, 12)   // 48 / 4
        XCTAssertEqual(geo.map.recurrentLayerIndices.count, 36)
        // Geometry matches the standalone HarnessCore derivation over the same bytes.
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        XCTAssertEqual(geo, try ModelConfigDecoder.qwen35HybridGeometry(configJSON: data))
        // Config identity pinned + deterministic: re-verifying the same bytes yields an equal proof.
        XCTAssertEqual(proof.modelConfigurationSHA256.count, 64)
        XCTAssertEqual(proof, try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true))
    }

    func testFlaggedQwen35FailsClosedOnMissingLinearKey() throws {
        // Drop linear_num_value_heads — admission-grade derivation must throw, NOT degrade to a
        // structure-only proof.
        let json = #"""
        {"model_type":"qwen3_5","text_config":{"max_position_embeddings":262144,"vocab_size":248320,
          "num_hidden_layers":48,"full_attention_interval":4,"num_key_value_heads":8,"head_dim":128,
          "torch_dtype":"bfloat16","linear_num_key_heads":16,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
        let dir = try writeConfig(json)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, .invalidModelConfiguration)
        }
    }

    /// Boundary pin: the PROOF deliberately does NOT enforce the gated-delta kernel's Dk%32==0
    /// constraint (GatedDelta.swift:29). A Dk=48 config is a valid positive-integer geometry, so the
    /// proof ADMITS it and carries keyHeadDim=48 — because this proof is shared with the fp32 toy
    /// runtime tests (Dk=1, no Metal kernel). The kernel-viability guard lives in the REAL serving
    /// adapter (`loadContinuousServingModel`, see MLXContinuousServingTests), not here. This test
    /// prevents anyone re-adding the guard to the proof (which would break the toy runtime suite).
    func testFlaggedQwen35ProofAdmitsUnalignedDk_kernelGuardIsAdapterLevel() throws {
        let json = #"""
        {"model_type":"qwen3_5","text_config":{"max_position_embeddings":262144,"vocab_size":248320,
          "num_hidden_layers":48,"full_attention_interval":4,"num_key_value_heads":8,"head_dim":128,
          "torch_dtype":"bfloat16","linear_num_key_heads":16,"linear_num_value_heads":32,
          "linear_key_head_dim":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
        let dir = try writeConfig(json)
        defer { try? FileManager.default.removeItem(at: dir) }
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
        XCTAssertEqual(try XCTUnwrap(proof.verifiedHybridGeometry).recurrent.keyHeadDim, 48)
    }

    func testFlaggedQwen35FailsClosedOnMissingContextField() throws {
        // No max_position_embeddings anywhere → invalidModelConfiguration (mirrors the qwen3 path).
        let json = #"""
        {"model_type":"qwen3_5","text_config":{"vocab_size":248320,
          "num_hidden_layers":48,"full_attention_interval":4,"num_key_value_heads":8,"head_dim":128,
          "torch_dtype":"bfloat16","linear_num_key_heads":16,"linear_num_value_heads":32,
          "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
        let dir = try writeConfig(json)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, .invalidModelConfiguration)
        }
    }

    // MARK: qwen3 golden — the flag never changes the supported path

    func testFlagDoesNotAlterQwen3Proof() throws {
        let json = #"""
        {"model_type":"qwen3","max_position_embeddings":32768,"vocab_size":151936,
         "num_hidden_layers":36,"num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16"}
        """#
        let dir = try writeConfig(json)
        defer { try? FileManager.default.removeItem(at: dir) }
        let off = try DenseContinuousBatchModelProof.verifying(modelDirectory: dir)
        let on = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
        XCTAssertEqual(off, on, "the hybrid opt-in must not perturb a qwen3 proof")
        XCTAssertEqual(on.verifiedModelFamily, .qwen3)
        XCTAssertNil(on.verifiedHybridGeometry, "a qwen3 proof carries no hybrid geometry")
    }
}
