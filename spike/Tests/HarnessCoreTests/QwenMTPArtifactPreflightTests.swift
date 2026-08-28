import Foundation
import XCTest
@testable import HarnessCore

final class QwenMTPArtifactPreflightTests: XCTestCase {
    func testKnownQwen38_27BMXFP8LockMatchesReviewedArtifacts() throws {
        let lock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        let targetConfig = try fixtureData(named: "qwen38-27b-target-config")
        let drafterConfig = try fixtureData(named: "qwen38-27b-mtp-config")
        let targetTensors = try tensorFixture(named: "qwen38-27b-target-tensors")

        XCTAssertEqual(lock.sourceRevision, "01472a78fca830689ff78246a82c6d31ab111a78")
        XCTAssertEqual(sha256Hex(targetConfig), lock.targetIdentity.configSHA256)
        XCTAssertEqual(sha256Hex(drafterConfig), lock.drafterIdentity.configSHA256)
        XCTAssertEqual(lock.targetIdentity.tokenizerSHA256,
            lock.drafterIdentity.tokenizerSHA256)
        XCTAssertEqual(lock.drafterTensors.count, 23)
        XCTAssertEqual(
            QwenMTPArtifactPreflight.tensorManifestSHA256(lock.drafterTensors),
            lock.drafterIdentity.tensorManifestSHA256)
        XCTAssertEqual(targetTensors.count, 1_682)
        XCTAssertEqual(
            QwenMTPArtifactPreflight.tensorManifestSHA256(targetTensors),
            lock.targetIdentity.tensorManifestSHA256)

        let result = try QwenMTPArtifactPreflight.validate(
            lock: lock,
            target: .init(
                identity: lock.targetIdentity,
                configJSON: targetConfig,
                tensors: targetTensors),
            drafter: .init(
                identity: lock.drafterIdentity,
                configJSON: drafterConfig,
                tensors: lock.drafterTensors))

        XCTAssertEqual(result.targetModelID, "mlx-community/Qwen3.8-27B-mxfp8")
        XCTAssertEqual(result.drafterModelID, "mlx-community/Qwen3.8-27B-MTP-mxfp8")
        XCTAssertEqual(result.targetRevision, "d48d163bcdf24acaf656474854ab88ea17d65bd1")
        XCTAssertEqual(result.drafterRevision, "a50634460045613f166b09b13519466e801c6568")
        XCTAssertEqual(result.sourceRevision, lock.sourceRevision)
        XCTAssertEqual(result.runtimeBlockSize, 3)
        XCTAssertEqual(result.maximumAcceptedDraftTokens, 2)
    }

    func testKnownQwen38_27BMXFP8TargetAndDrafterManifestDriftFailsClosed() throws {
        let lock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        let targetConfig = try fixtureData(named: "qwen38-27b-target-config")
        let drafterConfig = try fixtureData(named: "qwen38-27b-mtp-config")
        let targetTensors = try tensorFixture(named: "qwen38-27b-target-tensors")
        let target = QwenMTPArtifactCandidate(
            identity: lock.targetIdentity,
            configJSON: targetConfig,
            tensors: targetTensors)
        let drafter = QwenMTPArtifactCandidate(
            identity: lock.drafterIdentity,
            configJSON: drafterConfig,
            tensors: lock.drafterTensors)

        let badTargetIdentity = lock.targetIdentity.withTensorManifestSHA256(
            String(repeating: "7", count: 64))
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: lock.withTargetIdentity(badTargetIdentity),
            target: target.withIdentity(badTargetIdentity),
            drafter: drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorManifestDigestMismatch(role: .target))
        }

        let badDrafterIdentity = lock.drafterIdentity.withTensorManifestSHA256(
            String(repeating: "8", count: 64))
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: lock.withDrafterIdentity(badDrafterIdentity),
            target: target,
            drafter: drafter.withIdentity(badDrafterIdentity)
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorManifestDigestMismatch(role: .drafter))
        }
    }

    func testKnownQwen35_9BLockMatchesReviewedArtifacts() throws {
        let lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        let targetConfig = try fixtureData(named: "qwen35-9b-target-config")
        let drafterConfig = try fixtureData(named: "qwen35-9b-mtp-config")
        let targetTensors = try tensorFixture(named: "qwen35-9b-target-tensors")

        XCTAssertEqual(sha256Hex(targetConfig), lock.targetIdentity.configSHA256)
        XCTAssertEqual(sha256Hex(drafterConfig), lock.drafterIdentity.configSHA256)
        XCTAssertEqual(lock.targetIdentity.tokenizerSHA256,
            lock.drafterIdentity.tokenizerSHA256)
        XCTAssertEqual(lock.drafterTensors.count, 31)
        XCTAssertEqual(
            QwenMTPArtifactPreflight.tensorManifestSHA256(lock.drafterTensors),
            lock.drafterIdentity.tensorManifestSHA256)
        XCTAssertEqual(targetTensors.count, 1_260)
        XCTAssertEqual(
            QwenMTPArtifactPreflight.tensorManifestSHA256(targetTensors),
            lock.targetIdentity.tensorManifestSHA256)

        let result = try QwenMTPArtifactPreflight.validate(
            lock: lock,
            target: .init(
                identity: lock.targetIdentity,
                configJSON: targetConfig,
                tensors: targetTensors),
            drafter: .init(
                identity: lock.drafterIdentity,
                configJSON: drafterConfig,
                tensors: lock.drafterTensors))

        XCTAssertEqual(result.targetModelID, "mlx-community/Qwen3.5-9B-MLX-4bit")
        XCTAssertEqual(result.drafterModelID, "mlx-community/Qwen3.5-9B-MTP-5bit")
        XCTAssertEqual(result.runtimeBlockSize, 3)
        XCTAssertEqual(result.maximumAcceptedDraftTokens, 2)
    }

    func testExactLockedPairPassesForDepthOne() throws {
        let fixture = Fixture()

        let result = try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock,
            target: fixture.target,
            drafter: fixture.drafter)

        XCTAssertEqual(result.targetModelID, fixture.target.identity.modelID)
        XCTAssertEqual(result.drafterModelID, fixture.drafter.identity.modelID)
        XCTAssertEqual(result.runtimeBlockSize, 3)
        XCTAssertEqual(result.maximumAcceptedDraftTokens, 2)
        XCTAssertEqual(result.architecture, fixture.architecture)
    }

    func testExactLockedPairRejectsUnsupportedBlockSizes() throws {
        let fixture = Fixture()

        for blockSize in [2, 4] {
            let config = Fixture.drafterConfig(blockSize: blockSize)
            let identity = fixture.drafter.identity.withConfigSHA256(sha256Hex(config))
            XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
                lock: fixture.lock.withDrafterIdentity(identity),
                target: fixture.target,
                drafter: fixture.drafter.withIdentity(identity).withConfig(config)
            )) { error in
                XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                    .unsupportedBlockSize(blockSize))
            }
        }
    }

    func testIdentityAndConfigDigestsFailClosed() throws {
        let fixture = Fixture()

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock,
            target: fixture.target.withIdentity(
                fixture.target.identity.withRevision(String(repeating: "f", count: 40))),
            drafter: fixture.drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .identityMismatch(role: .target, field: "revision"))
        }

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock,
            target: fixture.target.withConfig(Data("{}".utf8)),
            drafter: fixture.drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .configDigestMismatch(role: .target))
        }
    }

    func testTokenizerMismatchFailsBeforeModelUse() throws {
        let fixture = Fixture()
        let mismatched = fixture.drafter.withIdentity(
            fixture.drafter.identity.withTokenizerSHA256(String(repeating: "9", count: 64)))

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(mismatched.identity),
            target: fixture.target,
            drafter: mismatched
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError, .tokenizerMismatch)
        }
    }

    func testArchitectureMismatchFailsClosed() throws {
        let fixture = Fixture()
        let changedConfig = Fixture.drafterConfig(hiddenSize: 8192)
        let changedIdentity = fixture.drafter.identity.withConfigSHA256(sha256Hex(changedConfig))
        let changed = fixture.drafter
            .withIdentity(changedIdentity)
            .withConfig(changedConfig)

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(changedIdentity),
            target: fixture.target,
            drafter: changed
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .architectureMismatch(field: "hidden_size"))
        }
    }

    func testUnsupportedModelTypeAndDepthFailClosed() throws {
        let fixture = Fixture()
        let wrongType = Fixture.drafterConfig(modelType: "qwen3_5")
        let wrongIdentity = fixture.drafter.identity.withConfigSHA256(sha256Hex(wrongType))

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(wrongIdentity),
            target: fixture.target,
            drafter: fixture.drafter.withIdentity(wrongIdentity).withConfig(wrongType)
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .unsupportedModelType(role: .drafter, actual: "qwen3_5"))
        }

        let depthTwo = Fixture.drafterConfig(mtpLayers: 2)
        let depthIdentity = fixture.drafter.identity.withConfigSHA256(sha256Hex(depthTwo))
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(depthIdentity),
            target: fixture.target,
            drafter: fixture.drafter.withIdentity(depthIdentity).withConfig(depthTwo)
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .unsupportedMTPDepth(2))
        }
    }

    func testQuantizationMismatchFailsClosed() throws {
        let fixture = Fixture()
        let changedConfig = Fixture.drafterConfig(bits: 4)
        let changedIdentity = fixture.drafter.identity.withConfigSHA256(sha256Hex(changedConfig))

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(changedIdentity),
            target: fixture.target,
            drafter: fixture.drafter.withIdentity(changedIdentity).withConfig(changedConfig)
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .quantizationMismatch(role: .drafter))
        }
    }

    func testMissingExtraAndMismatchedTensorsFailClosed() throws {
        let fixture = Fixture()
        let missing = Array(fixture.drafter.tensors.dropLast())
        XCTAssertThrowsError(try fixture.validate(drafterTensors: missing)) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .missingTensor("pre_fc_norm_hidden.weight"))
        }

        let extra = fixture.drafter.tensors + [
            .init(name: "unexpected.weight", shape: [1], dtype: "BF16")
        ]
        XCTAssertThrowsError(try fixture.validate(drafterTensors: extra)) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .unexpectedTensor("unexpected.weight"))
        }

        var mismatched = fixture.drafter.tensors
        mismatched[0] = .init(name: mismatched[0].name, shape: [99], dtype: mismatched[0].dtype)
        XCTAssertThrowsError(try fixture.validate(drafterTensors: mismatched)) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorDescriptorMismatch("fc.biases"))
        }
    }

    func testDuplicateTensorNamesAndManifestDigestMismatchFailClosed() throws {
        let fixture = Fixture()
        let duplicate = fixture.drafter.tensors + [fixture.drafter.tensors[0]]
        XCTAssertThrowsError(try fixture.validate(drafterTensors: duplicate)) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .duplicateTensor("fc.biases"))
        }

        let badIdentity = fixture.drafter.identity.withTensorManifestSHA256(
            String(repeating: "7", count: 64))
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withDrafterIdentity(badIdentity),
            target: fixture.target,
            drafter: fixture.drafter.withIdentity(badIdentity)
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorManifestDigestMismatch(role: .drafter))
        }
    }

    func testTargetTensorManifestIsRequiredAndAuthenticated() throws {
        let fixture = Fixture()

        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock,
            target: fixture.target.withTensors([]),
            drafter: fixture.drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorManifestDigestMismatch(role: .target))
        }

        let changed = [QwenMTPTensorDescriptor(
            name: "model.embed_tokens.weight", shape: [2, 2], dtype: "BF16")]
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock,
            target: fixture.target.withTensors(changed),
            drafter: fixture.drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .tensorManifestDigestMismatch(role: .target))
        }
    }

    func testMalformedIdentityDigestFailsClosed() throws {
        let fixture = Fixture()
        let malformed = fixture.target.identity.withConfigSHA256("not-a-sha")
        XCTAssertThrowsError(try QwenMTPArtifactPreflight.validate(
            lock: fixture.lock.withTargetIdentity(malformed),
            target: fixture.target.withIdentity(malformed),
            drafter: fixture.drafter
        )) { error in
            XCTAssertEqual(error as? QwenMTPArtifactPreflightError,
                .invalidIdentity(role: .target, field: "configSHA256"))
        }
    }
}

private func fixtureData(named name: String) throws -> Data {
    let url = try XCTUnwrap(Bundle.module.url(
        forResource: name,
        withExtension: "json"))
    var data = try Data(contentsOf: url)
    if data.last == 0x0a {
        data.removeLast()
    }
    return data
}

private struct TensorFixture: Decodable {
    let name: String
    let shape: [Int]
    let dtype: String
}

private func tensorFixture(named name: String) throws -> [QwenMTPTensorDescriptor] {
    let data = try fixtureData(named: name)
    return try JSONDecoder().decode([TensorFixture].self, from: data).map {
        .init(name: $0.name, shape: $0.shape, dtype: $0.dtype)
    }
}

private struct Fixture {
    let architecture = QwenMTPArchitecture(
        hiddenSize: 4096,
        intermediateSize: 12288,
        vocabularySize: 248_320,
        targetLayerCount: 32,
        fullAttentionInterval: 4,
        attentionHeadCount: 16,
        keyValueHeadCount: 4,
        headDimension: 256)

    let target: QwenMTPArtifactCandidate
    let drafter: QwenMTPArtifactCandidate
    let lock: QwenMTPArtifactLock

    init() {
        let targetConfig = Self.targetConfig()
        let drafterConfig = Self.drafterConfig()
        let targetTensors = Self.targetTensors
        let tensors = Self.drafterTensors
        let tokenizer = String(repeating: "a", count: 64)
        let targetIdentity = QwenMTPArtifactIdentity(
            modelID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: String(repeating: "1", count: 40),
            configSHA256: sha256Hex(targetConfig),
            tokenizerSHA256: tokenizer,
            tensorManifestSHA256:
                QwenMTPArtifactPreflight.tensorManifestSHA256(targetTensors))
        let drafterIdentity = QwenMTPArtifactIdentity(
            modelID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            revision: String(repeating: "2", count: 40),
            configSHA256: sha256Hex(drafterConfig),
            tokenizerSHA256: tokenizer,
            tensorManifestSHA256: QwenMTPArtifactPreflight.tensorManifestSHA256(tensors))

        target = .init(
            identity: targetIdentity, configJSON: targetConfig, tensors: targetTensors)
        drafter = .init(identity: drafterIdentity, configJSON: drafterConfig, tensors: tensors)
        lock = .init(
            sourceRevision: String(repeating: "3", count: 40),
            targetIdentity: targetIdentity,
            drafterIdentity: drafterIdentity,
            architecture: architecture,
            targetQuantization: .init(bits: 4, groupSize: 64, mode: "affine"),
            drafterQuantization: .init(bits: 5, groupSize: 64, mode: "affine"),
            drafterTensors: tensors)
    }

    func validate(drafterTensors: [QwenMTPTensorDescriptor]) throws
        -> QwenMTPArtifactBinding
    {
        let identity = drafter.identity.withTensorManifestSHA256(
            QwenMTPArtifactPreflight.tensorManifestSHA256(drafterTensors))
        return try QwenMTPArtifactPreflight.validate(
            lock: lock,
            target: target,
            drafter: drafter.withIdentity(identity).withTensors(drafterTensors))
    }

    static let drafterTensors: [QwenMTPTensorDescriptor] = [
        .init(name: "fc.biases", shape: [4096], dtype: "BF16"),
        .init(name: "fc.scales", shape: [4096, 128], dtype: "BF16"),
        .init(name: "fc.weight", shape: [4096, 1024], dtype: "U32"),
        .init(name: "layers.0.self_attn.q_proj.weight", shape: [4096, 512], dtype: "U32"),
        .init(name: "norm.weight", shape: [4096], dtype: "BF16"),
        .init(name: "pre_fc_norm_embedding.weight", shape: [4096], dtype: "BF16"),
        .init(name: "pre_fc_norm_hidden.weight", shape: [4096], dtype: "BF16"),
    ]

    static let targetTensors: [QwenMTPTensorDescriptor] = [
        .init(name: "model.embed_tokens.weight", shape: [248_320, 4_096], dtype: "BF16")
    ]

    static func targetConfig() -> Data {
        Data("""
        {"model_type":"qwen3_5","quantization":{"bits":4,"group_size":64,"mode":"affine"},"text_config":\(textConfig())}
        """.utf8)
    }

    static func drafterConfig(
        blockSize: Int = 3,
        modelType: String = "qwen3_5_mtp",
        hiddenSize: Int = 4096,
        mtpLayers: Int = 1,
        bits: Int = 5
    ) -> Data {
        Data("""
        {"block_size":\(blockSize),"model_type":"\(modelType)","quantization":{"bits":\(bits),"group_size":64,"mode":"affine"},"text_config":\(textConfig(hiddenSize: hiddenSize, mtpLayers: mtpLayers))}
        """.utf8)
    }

    static func textConfig(hiddenSize: Int = 4096, mtpLayers: Int = 1) -> String {
        """
        {"model_type":"qwen3_5_text","hidden_size":\(hiddenSize),"intermediate_size":12288,"vocab_size":248320,"num_hidden_layers":32,"full_attention_interval":4,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"mtp_num_hidden_layers":\(mtpLayers),"mtp_use_dedicated_embeddings":false}
        """
    }
}

private extension QwenMTPArtifactIdentity {
    func withRevision(_ value: String) -> Self {
        .init(modelID: modelID, revision: value, configSHA256: configSHA256,
            tokenizerSHA256: tokenizerSHA256, tensorManifestSHA256: tensorManifestSHA256)
    }

    func withConfigSHA256(_ value: String) -> Self {
        .init(modelID: modelID, revision: revision, configSHA256: value,
            tokenizerSHA256: tokenizerSHA256, tensorManifestSHA256: tensorManifestSHA256)
    }

    func withTokenizerSHA256(_ value: String) -> Self {
        .init(modelID: modelID, revision: revision, configSHA256: configSHA256,
            tokenizerSHA256: value, tensorManifestSHA256: tensorManifestSHA256)
    }

    func withTensorManifestSHA256(_ value: String) -> Self {
        .init(modelID: modelID, revision: revision, configSHA256: configSHA256,
            tokenizerSHA256: tokenizerSHA256, tensorManifestSHA256: value)
    }
}

private extension QwenMTPArtifactCandidate {
    func withIdentity(_ value: QwenMTPArtifactIdentity) -> Self {
        .init(identity: value, configJSON: configJSON, tensors: tensors)
    }

    func withConfig(_ value: Data) -> Self {
        .init(identity: identity, configJSON: value, tensors: tensors)
    }

    func withTensors(_ value: [QwenMTPTensorDescriptor]) -> Self {
        .init(identity: identity, configJSON: configJSON, tensors: value)
    }
}

private extension QwenMTPArtifactLock {
    func withTargetIdentity(_ value: QwenMTPArtifactIdentity) -> Self {
        .init(sourceRevision: sourceRevision, targetIdentity: value,
            drafterIdentity: drafterIdentity, architecture: architecture,
            targetQuantization: targetQuantization, drafterQuantization: drafterQuantization,
            drafterTensors: drafterTensors)
    }

    func withDrafterIdentity(_ value: QwenMTPArtifactIdentity) -> Self {
        .init(sourceRevision: sourceRevision, targetIdentity: targetIdentity,
            drafterIdentity: value, architecture: architecture,
            targetQuantization: targetQuantization, drafterQuantization: drafterQuantization,
            drafterTensors: drafterTensors)
    }
}
