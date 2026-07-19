import Foundation
import XCTest

@testable import HarnessCore

final class CompressedKVAttentionRuntimeAdmissionTests: XCTestCase {
    func testKVTunerDefaultsToMaterializedAttentionWithoutChangingOtherDefaults() {
        XCTAssertEqual(
            CompressedKVAttentionRequest.effective(
                explicit: nil,
                hasAuthenticatedKVTunerSchedule: true),
            .materialize)
        XCTAssertEqual(
            CompressedKVAttentionRequest.effective(
                explicit: .splitAffineQuantizedMM,
                hasAuthenticatedKVTunerSchedule: true),
            .splitAffineQuantizedMM)
        XCTAssertNil(CompressedKVAttentionRequest.effective(
            explicit: nil,
            hasAuthenticatedKVTunerSchedule: false))
    }

    func testExplicitCompressedAttentionKLRequiresTheSameResolvedReferenceModel() throws {
        XCTAssertNoThrow(
            try CompressedKVAttentionRuntimeAdmission
                .validateKLReferenceModel(
                    isSameResolvedModel: false,
                    request: nil))
        XCTAssertNoThrow(
            try CompressedKVAttentionRuntimeAdmission
                .validateKLReferenceModel(
                    isSameResolvedModel: true,
                    request: .splitAffineQuantizedMM))
        XCTAssertThrowsError(
            try CompressedKVAttentionRuntimeAdmission
                .validateKLReferenceModel(
                    isSameResolvedModel: false,
                    request: .materialize)
        ) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .referenceModelIdentityMismatch)
        }
    }

    private let checkpointManifestHash = "0123456789abcdef"
    private let checkpointContentSHA256 = String(repeating: "d", count: 64)
    private let tokenizerSHA256 = String(repeating: "a", count: 64)

    func testQwen3DenseGQAAdmissionBindsExactConfigAndGeometry() throws {
        let data = qwenConfig()
        let snapshot = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: data,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)

        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: snapshot)

        XCTAssertEqual(admission.family, .qwen3)
        XCTAssertEqual(admission.modelType, "qwen3")
        XCTAssertEqual(admission.architecture, "Qwen3ForCausalLM")
        XCTAssertEqual(admission.layerCount, 64)
        XCTAssertEqual(admission.queryHeadCount, 64)
        XCTAssertEqual(admission.kvHeadCount, 8)
        XCTAssertEqual(admission.headDimension, 128)
        XCTAssertEqual(admission.maxPositionEmbeddings, 40_960)
        XCTAssertEqual(admission.modelConfigHash, fnv1a64(data))
        XCTAssertEqual(admission.modelConfigSHA256, sha256Hex(data))
        XCTAssertEqual(admission.checkpointManifestHash, checkpointManifestHash)
        XCTAssertEqual(
            admission.checkpointContentSHA256,
            checkpointContentSHA256)
        XCTAssertEqual(admission.tokenizerSHA256, tokenizerSHA256)
        XCTAssertNoThrow(try admission.validateAffineGeometry(
            keyGroupSize: 64, valueGroupSize: 64))
    }

    func testLlamaDenseGQAAdmissionAllowsAuthenticatedRopeScaling() throws {
        let admission = try admission(for: llamaConfig(headDimension: nil))

        XCTAssertEqual(admission.family, .llama)
        XCTAssertEqual(admission.architecture, "LlamaForCausalLM")
        XCTAssertEqual(admission.layerCount, 80)
        XCTAssertEqual(admission.maxPositionEmbeddings, 131_072)
        XCTAssertNoThrow(try admission.validateAffineGeometry(
            keyGroupSize: 128, valueGroupSize: 128))
    }

    func testExplicitRuntimeRequestBindsThePredeclaredCheckpointContent()
        throws
    {
        let admission = try admission(for: qwenConfig())
        XCTAssertNoThrow(try admission.validateCheckpointContentIdentity(
            checkpointContentSHA256))
        XCTAssertThrowsError(try admission.validateCheckpointContentIdentity(
            String(repeating: "e", count: 64)
        )) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .checkpointContentIdentityMismatch)
        }
    }

    func testAdmissionRejectsUnknownOrMismatchedArchitectures() throws {
        XCTAssertThrowsError(try admission(for: config(
            modelType: "mistral",
            architecture: "MistralForCausalLM"
        ))) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .unsupportedModelType("mistral"))
        }

        XCTAssertThrowsError(try admission(for: config(
            modelType: "qwen3",
            architecture: "CustomQwenForCausalLM"
        ))) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .unsupportedArchitecture("CustomQwenForCausalLM"))
        }
    }

    func testAdmissionRejectsWindowSinkAndNestedCustomAttention() throws {
        for (name, extra) in [
            ("sliding_window", "\"sliding_window\":4096"),
            ("use_sliding_window", "\"use_sliding_window\":true"),
            ("attention_sinks", "\"attention_sinks\":[0.1]"),
            ("text_config", "\"text_config\":{\"model_type\":\"qwen3\"}"),
        ] {
            XCTAssertThrowsError(try admission(for: config(extra: extra)), name) {
                error in
                XCTAssertEqual(
                    error as? CompressedKVAttentionRuntimeAdmissionError,
                    .unsupportedAttentionFeature(name))
            }
        }
    }

    func testAdmissionRejectsMalformedOrUnsupportedGeometry() throws {
        XCTAssertThrowsError(try admission(for: Data("{}".utf8))) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .malformedModelConfig)
        }

        XCTAssertThrowsError(try admission(for: config(
            queryHeads: 63, kvHeads: 8
        ))) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .invalidGeometry("queryHeadCount"))
        }

        let admitted = try admission(for: qwenConfig())
        XCTAssertThrowsError(try admitted.validateAffineGeometry(
            keyGroupSize: 96, valueGroupSize: 64
        )) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .unsupportedGroupGeometry(
                    headDimension: 128,
                    keyGroupSize: 96,
                    valueGroupSize: 64))
        }
    }

    func testScheduleIdentityMustMatchLoadedSourceBeforeExecution() throws {
        let admission = try admission(for: qwenConfig())
        XCTAssertNoThrow(try admission.validateScheduleIdentity(
            modelConfigHash: admission.modelConfigHash,
            modelConfigSHA256: admission.modelConfigSHA256,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: admission.checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256,
            layerCount: 64,
            groupSize: 128))

        let mutations: [(
            modelConfigHash: String,
            modelConfigSHA256: String,
            checkpointManifestHash: String,
            checkpointContentSHA256: String,
            tokenizerSHA256: String,
            layerCount: Int,
            groupSize: Int
        )] = [
            (
                "ffffffffffffffff", admission.modelConfigSHA256,
                checkpointManifestHash,
                admission.checkpointContentSHA256,
                tokenizerSHA256, 64, 128),
            (
                admission.modelConfigHash,
                String(repeating: "e", count: 64),
                checkpointManifestHash,
                admission.checkpointContentSHA256,
                tokenizerSHA256, 64, 128),
            (
                admission.modelConfigHash, admission.modelConfigSHA256,
                "ffffffffffffffff",
                admission.checkpointContentSHA256,
                tokenizerSHA256, 64, 128),
            (
                admission.modelConfigHash, admission.modelConfigSHA256,
                checkpointManifestHash,
                String(repeating: "e", count: 64),
                tokenizerSHA256, 64, 128),
            (
                admission.modelConfigHash, admission.modelConfigSHA256,
                checkpointManifestHash,
                admission.checkpointContentSHA256,
                String(repeating: "b", count: 64), 64, 128),
            (
                admission.modelConfigHash, admission.modelConfigSHA256,
                checkpointManifestHash,
                admission.checkpointContentSHA256,
                tokenizerSHA256, 63, 128),
            (
                admission.modelConfigHash, admission.modelConfigSHA256,
                checkpointManifestHash,
                admission.checkpointContentSHA256,
                tokenizerSHA256, 64, 96),
        ]
        for mutation in mutations {
            XCTAssertThrowsError(try admission.validateScheduleIdentity(
                modelConfigHash: mutation.modelConfigHash,
                modelConfigSHA256: mutation.modelConfigSHA256,
                checkpointManifestHash:
                    mutation.checkpointManifestHash,
                checkpointContentSHA256:
                    mutation.checkpointContentSHA256,
                tokenizerSHA256: mutation.tokenizerSHA256,
                layerCount: mutation.layerCount,
                groupSize: mutation.groupSize
            )) { error in
                XCTAssertEqual(
                    error as? CompressedKVAttentionRuntimeAdmissionError,
                    .scheduleIdentityMismatch)
            }
        }
    }

    func testSourceSnapshotRejectsMutationAcrossModelLoad() throws {
        let before = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: qwenConfig(),
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
        let same = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: qwenConfig(),
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
        XCTAssertEqual(
            try CompressedKVAttentionRuntimeSourceSnapshot.validateUnchanged(
                before: before, after: same),
            same)

        let changed = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: llamaConfig(),
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
        XCTAssertThrowsError(
            try CompressedKVAttentionRuntimeSourceSnapshot.validateUnchanged(
                before: before, after: changed)
        ) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .sourceIdentityChangedDuringModelLoad)
        }

        let sameMetadataDifferentWeights = try
            CompressedKVAttentionRuntimeSourceSnapshot.load(
                exactModelConfigData: qwenConfig(),
                checkpointManifestHash: checkpointManifestHash,
                checkpointContentSHA256:
                    String(repeating: "e", count: 64),
                tokenizerSHA256: tokenizerSHA256)
        XCTAssertThrowsError(
            try CompressedKVAttentionRuntimeSourceSnapshot
                .validateUnchanged(
                    before: before,
                    after: sameMetadataDifferentWeights)
        ) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .sourceIdentityChangedDuringModelLoad)
        }
    }

    func testRunConfigDefaultsToLegacyMaterializationAndCarriesExplicitSplitRequest() {
        XCTAssertNil(RunConfig().compressedKVAttention)
        XCTAssertNil(
            RunConfig()
                .compressedKVAttentionExpectedCheckpointContentSHA256)
        XCTAssertEqual(
            RunConfig(
                kvQuant: "affine-k4v2-g64",
                compressedKVAttention: .splitAffineQuantizedMM,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    checkpointContentSHA256)
                .compressedKVAttention,
            .splitAffineQuantizedMM)
    }

    func testRuntimeBindingAuthenticatesRequestedAndObservedOperation() throws {
        let admitted = try admission(for: qwenConfig())
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admitted)

        XCTAssertEqual(binding.request, .splitAffineQuantizedMM)
        XCTAssertEqual(binding.observedOperation, .splitQuantizedMM)
        XCTAssertEqual(binding.admission, admitted)

        XCTAssertThrowsError(try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .materializedKV,
            admission: admitted)) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .attentionOperationMismatch)
        }

        var forged = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(binding)) as? [String: Any])
        forged["observedOperation"] = "materialized-kv"
        let decodedForgery = try JSONDecoder().decode(
            CompressedKVAttentionRuntimeBinding.self,
            from: JSONSerialization.data(withJSONObject: forged))
        XCTAssertThrowsError(try decodedForgery.validated()) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .attentionOperationMismatch)
        }

        forged = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(binding)) as? [String: Any])
        var forgedAdmission = try XCTUnwrap(
            forged["admission"] as? [String: Any])
        forgedAdmission["checkpointContentSHA256"] = "not-a-digest"
        forged["admission"] = forgedAdmission
        let malformedIdentity = try JSONDecoder().decode(
            CompressedKVAttentionRuntimeBinding.self,
            from: JSONSerialization.data(withJSONObject: forged))
        XCTAssertThrowsError(try malformedIdentity.validated()) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .invalidSourceIdentity)
        }
    }

    func testKVarNDirectRequestHasADistinctDurableOperationReceipt() throws {
        XCTAssertEqual(
            CompressedKVAttentionRequest.splitKVarNQuantizedMM.rawValue,
            "split-kvarn-quantized-mm")
        XCTAssertEqual(
            CompressedKVAttentionObservedOperation.splitKVarNQuantizedMM
                .rawValue,
            "split-kvarn-quantized-mm")

        let admitted = try admission(for: qwenConfig())
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .splitKVarNQuantizedMM,
            observedOperation: .splitKVarNQuantizedMM,
            admission: admitted)

        XCTAssertEqual(binding.request, .splitKVarNQuantizedMM)
        XCTAssertEqual(
            binding.observedOperation,
            .splitKVarNQuantizedMM)
        XCTAssertThrowsError(try CompressedKVAttentionRuntimeBinding(
            request: .splitKVarNQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admitted
        )) {
            XCTAssertEqual(
                $0 as? CompressedKVAttentionRuntimeAdmissionError,
                .attentionOperationMismatch)
        }
    }

    private func admission(
        for data: Data
    ) throws -> CompressedKVAttentionRuntimeAdmission {
        try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: data,
                checkpointManifestHash: checkpointManifestHash,
                checkpointContentSHA256: checkpointContentSHA256,
                tokenizerSHA256: tokenizerSHA256))
    }

    private func qwenConfig() -> Data { config() }

    private func llamaConfig(headDimension: Int? = 128) -> Data {
        config(
            modelType: "llama",
            architecture: "LlamaForCausalLM",
            hiddenSize: 8192,
            layerCount: 80,
            headDimension: headDimension,
            maxPositionEmbeddings: 131_072,
            extra: "\"rope_scaling\":{\"factor\":8.0,\"rope_type\":\"llama3\"}")
    }

    private func config(
        modelType: String = "qwen3",
        architecture: String = "Qwen3ForCausalLM",
        hiddenSize: Int = 5120,
        layerCount: Int = 64,
        queryHeads: Int = 64,
        kvHeads: Int = 8,
        headDimension: Int? = 128,
        maxPositionEmbeddings: Int = 40_960,
        extra: String? = nil
    ) -> Data {
        let extraField = extra.map { ",\($0)" } ?? ""
        let windowField = extra?.contains("\"use_sliding_window\"") == true
            ? "" : ",\"use_sliding_window\":false"
        let headDimensionField = headDimension.map {
            ",\"head_dim\":\($0)"
        } ?? ""
        return Data("""
        {
          "model_type":"\(modelType)",
          "architectures":["\(architecture)"],
          "hidden_size":\(hiddenSize),
          "num_hidden_layers":\(layerCount),
          "num_attention_heads":\(queryHeads),
          "num_key_value_heads":\(kvHeads),
          "max_position_embeddings":\(maxPositionEmbeddings)
          \(headDimensionField)
          \(windowField)
          \(extraField)
        }
        """.utf8)
    }
}
