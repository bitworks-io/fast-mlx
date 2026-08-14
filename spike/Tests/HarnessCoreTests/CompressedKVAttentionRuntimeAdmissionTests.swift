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
        XCTAssertEqual(admission.modelNativeDType, .bfloat16)
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

    func testPhi3MiniAdmissionBindsExactSourceLockedInertLongRopeGeometry()
        throws
    {
        let data = phi3MiniSourceLockedConfig()
        let snapshot = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: data,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)

        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: snapshot)

        XCTAssertEqual(admission.family, .phi3)
        XCTAssertEqual(admission.modelType, "phi3")
        XCTAssertEqual(admission.architecture, "Phi3ForCausalLM")
        XCTAssertEqual(admission.layerCount, 32)
        XCTAssertEqual(admission.queryHeadCount, 24)
        XCTAssertEqual(admission.kvHeadCount, 8)
        XCTAssertEqual(admission.headDimension, 128)
        XCTAssertEqual(admission.maxPositionEmbeddings, 131_072)
        XCTAssertEqual(admission.modelNativeDType, .bfloat16)
        XCTAssertEqual(admission.modelConfigHash, fnv1a64(data))
        XCTAssertEqual(admission.modelConfigSHA256, sha256Hex(data))
        XCTAssertEqual(admission.checkpointManifestHash, checkpointManifestHash)
        XCTAssertEqual(
            admission.checkpointContentSHA256,
            checkpointContentSHA256)
        XCTAssertEqual(admission.tokenizerSHA256, tokenizerSHA256)
        XCTAssertNoThrow(try admission.validateAffineGeometry(
            keyGroupSize: 128, valueGroupSize: 128))
    }

    func testPhi3MiniEvidenceRevalidationRejectsForgedSourceLockedScalars()
        throws
    {
        let admitted = try admission(for: phi3MiniSourceLockedConfig())
        let encoded = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(admitted)) as? [String: Any])
        let mutations: [(String, Any)] = [
            ("layerCount", 31),
            ("queryHeadCount", 32),
            ("kvHeadCount", 6),
            ("headDimension", 96),
            ("maxPositionEmbeddings", 65_536),
            ("modelNativeDType", "float16"),
        ]

        for (field, value) in mutations {
            var forged = encoded
            forged[field] = value
            let decoded = try JSONDecoder().decode(
                CompressedKVAttentionRuntimeAdmission.self,
                from: JSONSerialization.data(withJSONObject: forged))

            XCTAssertThrowsError(try decoded.validatedForEvidence(), field) {
                error in
                XCTAssertEqual(
                    error as? CompressedKVAttentionRuntimeAdmissionError,
                    .invalidSourceIdentity)
            }
        }
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

    func testAdmissionCarriesOnlyDeclaredModelNativeScalarDTypes() throws {
        XCTAssertEqual(
            try admission(for: config(modelNativeDType: "float16"))
                .modelNativeDType,
            .float16)
        XCTAssertEqual(
            try admission(for: config(modelNativeDType: "bfloat16"))
                .modelNativeDType,
            .bfloat16)
        XCTAssertEqual(
            try admission(for: config(modelNativeDType: "float32"))
                .modelNativeDType,
            .float32)
        XCTAssertNil(
            try admission(for: config(modelNativeDType: nil))
                .modelNativeDType,
            "legacy evidence remains decodable, but a KVarN runtime must reject the absent policy")
        XCTAssertNil(
            try admission(for: config(modelNativeDType: "float8_e4m3fn"))
                .modelNativeDType,
            "unsupported config dtypes must never be guessed as fp16")
    }

    func testAdmissionUsesDTypeWhenTorchDTypeIsAbsent() throws {
        let admission = try admission(for: config(
            modelNativeDType: nil,
            extra: "\"dtype\":\"float16\""))

        XCTAssertEqual(admission.modelNativeDType, .float16)
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

    func testPhi3MiniAdmissionPreservesGenericSlidingWindowRejection()
        throws
    {
        for window in [131_072, 4_096] {
            let data = try phi3MiniSourceLockedConfig { root in
                root["sliding_window"] = window
            }
            XCTAssertThrowsError(try admission(for: data), "\(window)") {
                error in
                XCTAssertEqual(
                    error as? CompressedKVAttentionRuntimeAdmissionError,
                    .unsupportedAttentionFeature("sliding_window"))
            }
        }
    }

    func testPhi3MiniAdmissionRejectsPinnedWindowForNonPhiFamilies()
        throws
    {
        let cases: [(String, Data)] = [
            ("qwen3", config(extra: "\"sliding_window\":262144")),
            ("llama", config(
                modelType: "llama",
                architecture: "LlamaForCausalLM",
                hiddenSize: 8192,
                layerCount: 80,
                maxPositionEmbeddings: 131_072,
                extra: """
                "rope_scaling":{"factor":8.0,"rope_type":"llama3"},
                "sliding_window":262144
                """)),
            ("mistral", config(
                modelType: "mistral",
                architecture: "MistralForCausalLM",
                hiddenSize: 4096,
                layerCount: 32,
                queryHeads: 32,
                maxPositionEmbeddings: 131_072,
                extra: "\"sliding_window\":262144")),
        ]
        for (name, data) in cases {
            XCTAssertThrowsError(try admission(for: data), name) { error in
                XCTAssertEqual(
                    error as? CompressedKVAttentionRuntimeAdmissionError,
                    .unsupportedAttentionFeature("sliding_window"))
            }
        }
    }

    func testPhi3MiniAdmissionRejectsBadArchitectureAndGeometry() throws {
        let badArchitecture = try phi3MiniSourceLockedConfig { root in
            root["architectures"] = ["PhiForCausalLM"]
        }
        XCTAssertThrowsError(try admission(for: badArchitecture)) { error in
            XCTAssertEqual(
                error as? CompressedKVAttentionRuntimeAdmissionError,
                .unsupportedArchitecture("PhiForCausalLM"))
        }

        let mutations: [(String, (inout [String: Any]) -> Void)] = [
            ("hiddenSize", { (root: inout [String: Any]) in
                root["hidden_size"] = 4_096
            }),
            ("layerCount", { (root: inout [String: Any]) in
                root["num_hidden_layers"] = 31
            }),
            ("queryHeadCount", { (root: inout [String: Any]) in
                root["num_attention_heads"] = 32
            }),
            ("kvHeadCount", { (root: inout [String: Any]) in
                root["num_key_value_heads"] = 6
            }),
            ("headDimension", { (root: inout [String: Any]) in
                root["head_dim"] = 96
            }),
            ("maxPositionEmbeddings", { (root: inout [String: Any]) in
                root["max_position_embeddings"] = 65_536
            }),
            ("modelNativeDType", { (root: inout [String: Any]) in
                root["torch_dtype"] = "float16"
            }),
            ("stillInertButNotSourceLockedWindow", {
                (root: inout [String: Any]) in
                root["sliding_window"] = 262_145
            }),
        ]
        for (name, mutation) in mutations {
            let data = try phi3MiniSourceLockedConfig(mutatingRoot: mutation)
            XCTAssertThrowsError(try admission(for: data), name)
        }
    }

    func testPhi3MiniAdmissionRequiresLongRopePartialRotarySemantics()
        throws
    {
        let mutations: [(String, (inout [String: Any]) -> Void)] = [
            ("missingSlidingWindow", { (root: inout [String: Any]) in
                root.removeValue(forKey: "sliding_window")
            }),
            ("explicitUseSlidingWindowFalse", {
                (root: inout [String: Any]) in
                root["use_sliding_window"] = false
            }),
            ("missingOriginalMaxPositionEmbeddings", {
                (root: inout [String: Any]) in
                root.removeValue(forKey: "original_max_position_embeddings")
            }),
            ("wrongOriginalMaxPositionEmbeddings", {
                (root: inout [String: Any]) in
                root["original_max_position_embeddings"] = 8_192
            }),
            ("missingPartialRotaryFactor", {
                (root: inout [String: Any]) in
                root.removeValue(forKey: "partial_rotary_factor")
            }),
            ("wrongPartialRotaryFactor", { (root: inout [String: Any]) in
                root["partial_rotary_factor"] = 1.0
            }),
            ("wrongRopeTheta", { (root: inout [String: Any]) in
                root["rope_theta"] = 500_000.0
            }),
            ("missingRopeScaling", { (root: inout [String: Any]) in
                root.removeValue(forKey: "rope_scaling")
            }),
            ("wrongRopeScalingType", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                ropeScaling["type"] = "linear"
                root["rope_scaling"] = ropeScaling
            }),
            ("missingLongFactor", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                ropeScaling.removeValue(forKey: "long_factor")
                root["rope_scaling"] = ropeScaling
            }),
            ("missingShortFactor", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                ropeScaling.removeValue(forKey: "short_factor")
                root["rope_scaling"] = ropeScaling
            }),
            ("wrongLongFactorLength", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                var longFactor = ropeScaling["long_factor"] as! [Any]
                longFactor.removeLast()
                ropeScaling["long_factor"] = longFactor
                root["rope_scaling"] = ropeScaling
            }),
            ("wrongShortFactorLength", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                var shortFactor = ropeScaling["short_factor"] as! [Any]
                shortFactor.append(1.0)
                ropeScaling["short_factor"] = shortFactor
                root["rope_scaling"] = ropeScaling
            }),
            ("wrongLongFactorValue", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                var longFactor = ropeScaling["long_factor"] as! [Any]
                longFactor[1] = 1.118320673
                ropeScaling["long_factor"] = longFactor
                root["rope_scaling"] = ropeScaling
            }),
            ("wrongShortFactorValue", { (root: inout [String: Any]) in
                var ropeScaling = root["rope_scaling"] as! [String: Any]
                var shortFactor = ropeScaling["short_factor"] as! [Any]
                shortFactor[0] = 0.5
                ropeScaling["short_factor"] = shortFactor
                root["rope_scaling"] = ropeScaling
            }),
            ("missingFullAttentionMod", { (root: inout [String: Any]) in
                root.removeValue(forKey: "full_attn_mod")
            }),
            ("wrongFullAttentionMod", { (root: inout [String: Any]) in
                root["full_attn_mod"] = 2
            }),
        ]
        for (name, mutation) in mutations {
            let data = try phi3MiniSourceLockedConfig(mutatingRoot: mutation)
            XCTAssertThrowsError(try admission(for: data), name)
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

    private func phi3MiniSourceLockedConfig(
        mutatingRoot mutation: (inout [String: Any]) -> Void
    ) throws -> Data {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: phi3MiniSourceLockedConfig())
                as? [String: Any])
        mutation(&root)
        return try JSONSerialization.data(
            withJSONObject: root,
            options: [.sortedKeys])
    }

    private func phi3MiniSourceLockedConfig() -> Data {
        Data(#"""
        {
            "architectures": [
                "Phi3ForCausalLM"
            ],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_phi3.Phi3Config",
                "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
                "AutoTokenizer": "Xenova/gpt-4o"
            },
            "bos_token_id": 199999,
            "embd_pdrop": 0.0,
            "eos_token_id": 200020,
            "full_attn_mod": 1,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "interpolate_factor": 1,
            "lm_head_bias": false,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "phi3",
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "original_max_position_embeddings": 4096,
            "pad_token_id": 199999,
            "partial_rotary_factor": 0.75,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "quantization_config": {
                "group_size": 64,
                "bits": 4
            },
            "resid_pdrop": 0.0,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "long_factor": [
                    1,
                    1.118320672,
                    1.250641126,
                    1.398617824,
                    1.564103225,
                    1.74916897,
                    1.956131817,
                    2.187582649,
                    2.446418898,
                    2.735880826,
                    3.059592084,
                    3.421605075,
                    3.826451687,
                    4.279200023,
                    4.785517845,
                    5.351743533,
                    5.984965424,
                    6.693110555,
                    7.485043894,
                    8.370679318,
                    9.36110372,
                    10.4687158,
                    11.70738129,
                    13.09260651,
                    14.64173252,
                    16.37415215,
                    18.31155283,
                    20.47818807,
                    22.90118105,
                    25.61086418,
                    28.64115884,
                    32.03,
                    32.1,
                    32.13,
                    32.23,
                    32.6,
                    32.61,
                    32.64,
                    32.66,
                    32.7,
                    32.71,
                    32.93,
                    32.97,
                    33.28,
                    33.49,
                    33.5,
                    44.16,
                    47.77
                ],
                "short_factor": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                "type": "longrope"
            },
            "rope_theta": 10000.0,
            "sliding_window": 262144,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.45.0",
            "use_cache": true,
            "vocab_size": 200064
        }
        """#.utf8)
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
        modelNativeDType: String? = "bfloat16",
        extra: String? = nil
    ) -> Data {
        let extraField = extra.map { ",\($0)" } ?? ""
        let windowField = extra?.contains("\"use_sliding_window\"") == true
            ? "" : ",\"use_sliding_window\":false"
        let headDimensionField = headDimension.map {
            ",\"head_dim\":\($0)"
        } ?? ""
        let modelNativeDTypeField = modelNativeDType.map {
            ",\"torch_dtype\":\"\($0)\""
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
          \(modelNativeDTypeField)
          \(windowField)
          \(extraField)
        }
        """.utf8)
    }
}
