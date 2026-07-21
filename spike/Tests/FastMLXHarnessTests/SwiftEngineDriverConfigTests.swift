import Foundation
import XCTest

@testable import HarnessCore
@testable import SpikeCore
@testable import fastmlx_harness

final class SwiftEngineDriverConfigTests: XCTestCase {
    func testExplicitSplitRequiresAdmissionAndReturnsDistinctDecoderMode() throws {
        let admission = try makeAdmission()
        let config = RunConfig(
            kvQuant: "affine-k4v2-g64",
            compressedKVAttention: .splitAffineQuantizedMM,
            compressedKVAttentionExpectedCheckpointContentSHA256:
                admission.checkpointContentSHA256)

        let selection = try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: admission)

        XCTAssertEqual(selection.kind, .affine(.k4v2G64))
        XCTAssertEqual(selection.affineAttentionMode, .splitQuantizedMM)

        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: nil))
    }

    func testDefaultAndExplicitMaterializePreserveLegacyRoute() throws {
        for request in [nil, CompressedKVAttentionRequest.materialize] {
            let selection = try resolveSwiftEngineCacheSelection(
                config: RunConfig(
                    kvQuant: "affine-k4v2-g64",
                    compressedKVAttention: request,
                    compressedKVAttentionExpectedCheckpointContentSHA256:
                        request == nil
                            ? nil : try makeAdmission()
                                .checkpointContentSHA256),
                compressedKVAttentionAdmission:
                    request == nil ? nil : try makeAdmission())
            XCTAssertEqual(selection.kind, .affine(.k4v2G64))
            XCTAssertEqual(selection.affineAttentionMode, .materialize)
        }
    }

    func testSplitRejectsNonAffineCacheBeforeActorExecution() throws {
        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: RunConfig(
                kvQuant: nil,
                compressedKVAttention: .splitAffineQuantizedMM,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    try makeAdmission().checkpointContentSHA256),
            compressedKVAttentionAdmission: try makeAdmission()))
    }

    func testKVarNDirectRequestSelectsAuthenticatedKVarNCacheAndAttentionMode()
        throws
    {
        let admission = try makeAdmission()
        let selection = try resolveSwiftEngineCacheSelection(
            config: RunConfig(
                kvQuant: "kvarn-k4v2-g128",
                compressedKVAttention: .splitKVarNQuantizedMM,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    admission.checkpointContentSHA256),
            compressedKVAttentionAdmission: admission)

        XCTAssertEqual(selection.kind, .kvarn(.k4v2G128I8))
        XCTAssertEqual(selection.kvarnAttentionMode, .splitQuantizedMM)
    }

    func testKVTunerAlwaysBindsFrozenScheduleToLoadedSourceIdentity() throws {
        let admission = try makeAdmission(
            layerCount: 3,
            checkpointManifestHash: "fedcba9876543210",
            tokenizerSHA256: String(repeating: "c", count: 64))
        let selection = try makeSelection(admission: admission)
        let config = RunConfig(
            kvQuant: selection.cellID,
            kvtunerSelection: selection,
            compressedKVAttention: .splitAffineQuantizedMM,
            compressedKVAttentionExpectedCheckpointContentSHA256:
                admission.checkpointContentSHA256)

        let resolved = try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: admission)
        XCTAssertEqual(resolved.kind, .kvtuner(selection))
        XCTAssertEqual(
            resolved.affineAttentionMode, .splitQuantizedMM)

        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: nil)) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .invalidSourceIdentity)
        }
        let wrongCheckpoint = try makeAdmission(
            layerCount: 3,
            checkpointManifestHash: "0000000000000000",
            tokenizerSHA256: String(repeating: "c", count: 64))
        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: wrongCheckpoint)) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .scheduleIdentityMismatch)
        }
        let wrongContent = try makeAdmission(
            layerCount: 3,
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: String(repeating: "c", count: 64))
        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission: wrongContent)) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .checkpointContentIdentityMismatch)
        }

        let materializedConfig = RunConfig(
            kvQuant: selection.cellID,
            kvtunerSelection: selection)
        XCTAssertThrowsError(try resolveSwiftEngineCacheSelection(
            config: materializedConfig,
            compressedKVAttentionAdmission: wrongContent)) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .scheduleIdentityMismatch)
        }
    }

    private func makeAdmission(
        layerCount: Int = 64,
        checkpointManifestHash: String = "0123456789abcdef",
        checkpointContentSHA256: String = String(repeating: "d", count: 64),
        tokenizerSHA256: String = String(repeating: "a", count: 64)
    ) throws
        -> CompressedKVAttentionRuntimeAdmission
    {
        let config = Data(
            """
            {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":5120,"num_hidden_layers":\(layerCount),"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false}
            """.utf8)
        let snapshot = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: config,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: snapshot)
    }

    private func makeSelection(
        admission: CompressedKVAttentionRuntimeAdmission
    ) throws -> KVTunerRuntimeSelection {
        let average = String(Double(14) / 3)
        let schedule = KVTunerSchedule(
            schemaVersion: 4,
            matrixID: "kvarn-qwen3-32b-v2",
            cellID: "kvtuner-g128-b\(average)",
            modelConfigHash: admission.modelConfigHash,
            modelConfigSHA256: admission.modelConfigSHA256,
            checkpointManifestHash:
                admission.checkpointManifestHash,
            checkpointContentSHA256:
                admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
            groupSize: 128,
            calibrationCorpusID: "kvtuner-calibration-v1",
            calibrationCorpusHash: "1111111111111111",
            calibrationEntryHashes: [
                "2222222222222222",
                "3333333333333333",
            ],
            calibrationSourceItemDigests: (0..<200).map {
                sha256Hex(Data("source-\($0)".utf8))
            }.sorted(),
            seed: 1234,
            objective: "maximize-gsm8k-accuracy-at-b\(average)",
            nominalAverageBits: Double(14) / 3,
            sourceSensitivityArtifactSHA256:
                String(repeating: "a", count: 64),
            sourceSearchArtifactSHA256:
                String(repeating: "b", count: 64),
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
                KVLayerPrecision(layer: 2, keyBits: 8, valueBits: 2),
            ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try KVTunerRuntimeSelection.loadForTesting(
            artifactData: encoder.encode(schedule),
            expectedLayerCount: 3,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedModelConfigHash: admission.modelConfigHash,
            expectedModelConfigSHA256: admission.modelConfigSHA256,
            expectedCheckpointManifestHash:
                admission.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                admission.checkpointContentSHA256,
            evaluationCorpora: [
                try KVTunerEvaluationCorpusIdentity(
                    id: "measurement-corpus-v2",
                    aggregateDigest: "4444444444444444",
                    canonicalEntryDigests: [
                        "5555555555555555",
                    ],
                    canonicalSourceItemDigests: [
                        sha256Hex(Data(
                            "runtime-source".utf8))
                    ]),
            ])
    }
}
