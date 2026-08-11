import HarnessCore
import MLX
import MLXLLM
import XCTest

@testable import fastmlx_harness

final class KVTunerSensitivityActorTests: XCTestCase {
    private let checkpointHash = "fedcba9876543210"
    private let tokenizerSHA256 = String(repeating: "c", count: 64)

    private func configData() -> Data {
        Data(
            """
            {
              "eos_token_id": 63,
              "head_dim": 64,
              "hidden_size": 256,
              "intermediate_size": 512,
              "model_type": "qwen3",
              "num_attention_heads": 4,
              "num_hidden_layers": 2,
              "num_key_value_heads": 2,
              "rms_norm_eps": 0.000001,
              "rope_theta": 1000000,
              "tie_word_embeddings": false,
              "torch_dtype": "float32",
              "vocab_size": 64
            }
            """.utf8)
    }

    private func runtimeIdentity(
        checkpointHash: String? = nil
    ) throws -> KVTunerCandidateRuntimeIdentity {
        let data = configData()
        return try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: data,
            checkpointManifestHash: checkpointHash ?? self.checkpointHash,
            checkpointContentSHA256: String(repeating: "d", count: 64),
            tokenizerSHA256: tokenizerSHA256,
            eosTokenID: 63)
    }

    private func makeDriver() throws -> SwiftEngineDriver {
        let data = configData()
        let configuration = try JSONDecoder().decode(
            Qwen3Configuration.self, from: data)
        let model = Qwen3Model(configuration)
        eval(model)
        return SwiftEngineDriver(
            engine: HarnessEngineActor(
                model: model,
                kvtunerRuntimeIdentity: try runtimeIdentity()),
            eos: 63)
    }

    func testActorCapturesWithItsAuthenticatedModelConfig() async throws {
        let driver = try makeDriver()

        let samples = try await driver.engine.captureKVTunerSensitivity(
            promptTokenIDs: [[1, 2, 3]],
            groupSize: 64,
            expectedRuntimeIdentity: try runtimeIdentity())

        XCTAssertEqual(samples.count, 6)
        XCTAssertEqual(Set(samples.map(\.promptIndex)), [0])
        XCTAssertEqual(Set(samples.map(\.layer)), [0, 1])
    }

    func testActorRejectsCaptureWithoutAuthenticatedRuntimeIdentity() async throws {
        let data = configData()
        let configuration = try JSONDecoder().decode(
            Qwen3Configuration.self, from: data)
        let model = Qwen3Model(configuration)
        eval(model)
        let actor = HarnessEngineActor(model: model)

        do {
            _ = try await actor.captureKVTunerSensitivity(
                promptTokenIDs: [[1, 2, 3]],
                groupSize: 64,
                expectedRuntimeIdentity: try runtimeIdentity())
            XCTFail("expected missing live runtime identity")
        } catch {
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimeIdentityError,
                .missingRuntimeIdentity)
        }
    }

    func testActorRejectsCaptureForDifferentExpectedSource() async throws {
        let driver = try makeDriver()

        do {
            _ = try await driver.engine.captureKVTunerSensitivity(
                promptTokenIDs: [[1, 2, 3]],
                groupSize: 64,
                expectedRuntimeIdentity: try runtimeIdentity(
                    checkpointHash: "0123456789abcdef"))
            XCTFail("expected manifest-to-actor identity mismatch")
        } catch {
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimeIdentityError,
                .sourceIdentityChangedDuringModelLoad)
        }
    }

    func testDriverRejectsNoncanonicalPromptCountAndGroupSize() async throws {
        let driver = try makeDriver()

        do {
            _ = try await driver.captureKVTunerSensitivity(
                prompts: [[1]],
                groupSize: 64,
                expectedRuntimeIdentity: try runtimeIdentity())
            XCTFail("expected canonical prompt-count rejection")
        } catch {
            XCTAssertTrue(
                String(describing: error).contains("prompt count=1"))
        }

        do {
            _ = try await driver.captureKVTunerSensitivity(
                prompts: Array(repeating: [1], count: 20),
                groupSize: 32,
                expectedRuntimeIdentity: try runtimeIdentity())
            XCTFail("expected group-size rejection")
        } catch {
            XCTAssertTrue(
                String(describing: error).contains("groupSize=32"))
        }
    }
}
