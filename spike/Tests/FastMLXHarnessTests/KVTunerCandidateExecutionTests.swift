import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import HarnessCore
@testable import SpikeCore
@testable import fastmlx_harness

private final class TinyCandidateModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1, 1, 1]
    private let vocabularySize = 32

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        guard let cache, cache.count == kvHeads.count else {
            preconditionFailure("tiny candidate model requires three caches")
        }
        let scalar = inputs.asType(.float16).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let kv = broadcast(
            scalar, to: [inputs.dim(0), 1, inputs.dim(1), 128])
        for layerCache in cache {
            _ = layerCache.update(keys: kv, values: kv)
        }
        let target = inputs.asType(.int32).reshaped([
            inputs.dim(0), inputs.dim(1), 1,
        ])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class KVTunerCandidateExecutionTests: XCTestCase {
    private let checkpointHash = "fedcba9876543210"
    private let checkpointContentSHA256 = String(repeating: "d", count: 64)
    private let tokenizerSHA256 = String(repeating: "c", count: 64)
    private let configData = Data(
        #"{"eos_token_id":2,"head_dim":128,"hidden_size":128,"model_type":"qwen3","num_attention_heads":1,"num_hidden_layers":3,"num_key_value_heads":1,"torch_dtype":"float16"}"#.utf8)

    private func policy(
        checkpointHash: String? = nil
    ) throws -> KVTunerCandidateRuntimePolicy {
        try KVTunerCandidateRuntimePolicy.loadForTesting(
            candidate: KVTunerScheduleCandidate(
                ordinal: 0,
                analysisSHA256: String(repeating: "a", count: 64),
                totalPairBits: 28,
                meanAttentionOutputError: 0.125,
                layers: [
                    KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                    KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
                    KVLayerPrecision(layer: 2, keyBits: 8, valueBits: 2),
                ]),
            matrixID: "kvarn-qwen3-32b-v2",
            modelConfigHash: fnv1a64(configData),
            modelConfigSHA256: sha256Hex(configData),
            checkpointManifestHash: checkpointHash ?? self.checkpointHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256,
            groupSize: 128)
    }

    private func identity() throws -> KVTunerCandidateRuntimeIdentity {
        try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: configData,
            checkpointManifestHash: checkpointHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256,
            eosTokenID: 2)
    }

    func testActorRejectsCandidateWhenLiveIdentityWasNotRequested() async throws {
        let actor = HarnessEngineActor(model: TinyCandidateModel())

        do {
            _ = try await actor.evaluateKVTunerCandidateCohort(
                prompts: [[1]],
                maxTokens: 3,
                policy: try policy())
            XCTFail("expected missing live runtime identity")
        } catch {
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimeIdentityError,
                .missingRuntimeIdentity)
        }
    }

    func testActorRejectsPolicyForDifferentLiveCheckpoint() async throws {
        let actor = HarnessEngineActor(
            model: TinyCandidateModel(),
            kvtunerRuntimeIdentity: try identity())

        do {
            _ = try await actor.evaluateKVTunerCandidateCohort(
                prompts: [[1]],
                maxTokens: 3,
                policy: try policy(checkpointHash: "different-checkpoint"))
            XCTFail("expected live checkpoint mismatch")
        } catch {
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimeIdentityError,
                .checkpointIdentityMismatch)
        }
    }

    func testCohortClassifiesTerminationAndRetryStartsFresh() async throws {
        let actor = HarnessEngineActor(
            model: TinyCandidateModel(),
            kvtunerRuntimeIdentity: try identity())
        let runtimePolicy = try policy()
        let largePrompt = Array(repeating: 1, count: 200)

        let first = try await actor.evaluateKVTunerCandidateCohort(
            prompts: [largePrompt, [1, 2], [1]],
            maxTokens: 3,
            policy: runtimePolicy)
        XCTAssertEqual(first.count, 3)
        XCTAssertEqual(first[0].finishReason, .generationBudgetExhausted)
        XCTAssertEqual(first[1].finishReason, .endOfSequence)
        XCTAssertEqual(first[2].finishReason, .generationBudgetExhausted)
        XCTAssertEqual(first.map(\.promptOrdinal), [0, 1, 2])
        XCTAssertEqual(
            first.map(\.promptTokenIDsSHA256),
            [largePrompt, [1, 2], [1]].map(taskTokenIDsSHA256))
        XCTAssertEqual(first.map(\.telemetry.capacityTokens), [768, 768, 768])

        let retry = try await actor.evaluateKVTunerCandidateCohort(
            prompts: [[1]],
            maxTokens: 3,
            policy: runtimePolicy)
        XCTAssertEqual(retry.map(\.promptOrdinal), [0])
        XCTAssertEqual(
            retry.map(\.promptTokenIDsSHA256),
            [taskTokenIDsSHA256([1])])
        XCTAssertEqual(retry.map(\.telemetry.capacityTokens), [512])
    }
}
