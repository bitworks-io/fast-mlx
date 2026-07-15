import HarnessCore
import MLX
import MLXLLM
import XCTest

@testable import SpikeCore

final class KVTunerSensitivityCaptureTests: XCTestCase {
    private func configData(modelType: String = "qwen3") -> Data {
        Data(
            """
            {
              "eos_token_id": 63,
              "head_dim": 64,
              "hidden_size": 256,
              "intermediate_size": 512,
              "model_type": "\(modelType)",
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

    func testQwen3CaptureProducesCanonicalFiniteRowsAndReplaysModel() throws {
        let data = configData()
        let configuration = try JSONDecoder().decode(
            Qwen3Configuration.self, from: data)
        let model = Qwen3Model(configuration)
        eval(model)

        let samples = try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData: data,
            promptTokenIDs: [[1, 2, 3], [4, 5]],
            groupSize: 64,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs)
        let repeatedSamples = try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData: data,
            promptTokenIDs: [[1, 2, 3], [4, 5]],
            groupSize: 64,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs)

        XCTAssertEqual(samples.count, 12)
        XCTAssertEqual(repeatedSamples, samples)
        XCTAssertEqual(
            samples.map { [$0.promptIndex, $0.layer, $0.keyBits, $0.valueBits] },
            [
                [0, 0, 8, 4], [0, 0, 8, 2], [0, 0, 4, 2],
                [0, 1, 8, 4], [0, 1, 8, 2], [0, 1, 4, 2],
                [1, 0, 8, 4], [1, 0, 8, 2], [1, 0, 4, 2],
                [1, 1, 8, 4], [1, 1, 8, 2], [1, 1, 4, 2],
            ])
        for sample in samples {
            for metric in [
                sample.relativeKeyError,
                sample.relativeValueError,
                sample.attentionScoreError,
                sample.relativeAttentionOutputError,
            ] {
                XCTAssertTrue(metric.isFinite)
                XCTAssertGreaterThanOrEqual(metric, 0)
            }
        }
    }

    func testElementwiseRelativeErrorUsesPerElementDenominatorFloor() {
        let reference = MLXArray([1 as Float, 100])
        let candidate = MLXArray([0 as Float, 99])

        let error = KVTunerSensitivityMetrics.elementwiseRelativeMeanAbsoluteError(
            reference: reference,
            candidate: candidate,
            denominatorEpsilon: 1e-8)

        XCTAssertEqual(error, 0.505, accuracy: 1e-6)
    }

    func testCaptureRejectsUnsupportedModelAndEmptyPrompt() throws {
        let validData = configData()
        let configuration = try JSONDecoder().decode(
            Qwen3Configuration.self, from: validData)
        let model = Qwen3Model(configuration)
        eval(model)

        XCTAssertThrowsError(try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData: configData(modelType: "qwen2"),
            promptTokenIDs: [[1]],
            groupSize: 4,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs))
        XCTAssertThrowsError(try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData: validData,
            promptTokenIDs: [[]],
            groupSize: 64,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs))
        XCTAssertThrowsError(try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData: validData,
            promptTokenIDs: [[1]],
            groupSize: 32,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs))
    }
}
