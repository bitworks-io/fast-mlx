import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerSensitivityTests: XCTestCase {
    private let pairs = KVTunerSensitivityArtifact.canonicalPrecisionPairs

    private func artifact(
        groupSize: Int = 64,
        layerOutputs: [[Double]] = [
            [0.01, 0.04, 0.08],
            [0.02, 0.04, 0.08],
            [0.40, 0.40, 0.40],
            [0.45, 0.45, 0.45],
            [0.80, 0.80, 0.80],
        ]
    ) -> KVTunerSensitivityArtifact {
        let promptDigests = (0..<20).map {
            KVTunerPromptDigest.exactText("prompt \($0)")
        }
        var samples: [KVTunerSensitivitySample] = []
        for promptIndex in promptDigests.indices {
            for layer in layerOutputs.indices {
                for (pairIndex, pair) in pairs.enumerated() {
                    let output = layerOutputs[layer][pairIndex]
                    samples.append(KVTunerSensitivitySample(
                        promptIndex: promptIndex,
                        layer: layer,
                        keyBits: pair.keyBits,
                        valueBits: pair.valueBits,
                        relativeKeyError: output + 0.001,
                        relativeValueError: output + 0.002,
                        attentionScoreError: output + 0.003,
                        relativeAttentionOutputError: output))
                }
            }
        }
        return KVTunerSensitivityArtifact(
            schemaVersion: 1,
            matrixID: "kvarn-qwen3-32b-v1",
            modelConfigHash: "0123456789abcdef",
            modelConfigSHA256: String(repeating: "c", count: 64),
            checkpointManifestHash: "fedcba9876543210",
            tokenizerSHA256: String(repeating: "a", count: 64),
            calibrationCorpusID: "gsm8k-kvtuner-calibration-v1",
            calibrationCorpusHash: "1111111111111111",
            promptManifestSHA256: String(repeating: "b", count: 64),
            promptDigests: promptDigests,
            quantizerID: "mlx-affine-asymmetric-v1",
            captureMode: "single-prefill-no-error-propagation-v1",
            groupSize: groupSize,
            layerCount: layerOutputs.count,
            precisionPairs: pairs,
            metricProtocolID:
                "kvtuner-v5-elementwise-mean-absolute-v1",
            metricAccumulationDType: "float32",
            denominatorEpsilon: 1e-8,
            aggregationID: "ordered-incremental-mean-v1",
            dbscanEpsilon: 0.05,
            dbscanMinSamples: 2,
            samples: samples)
    }

    func testCompleteCanonicalArtifactValidatesAndRoundTrips() throws {
        let validated = try artifact().validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(validated)
        let decoded = try JSONDecoder().decode(
            KVTunerSensitivityArtifact.self, from: data)

        XCTAssertEqual(try decoded.validated(), validated)
        XCTAssertEqual(validated.samples.count, 300)

        var missingStrongConfigIdentity = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        missingStrongConfigIdentity.removeValue(forKey: "modelConfigSHA256")
        let missingData = try JSONSerialization.data(
            withJSONObject: missingStrongConfigIdentity)
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerSensitivityArtifact.self, from: missingData))
    }

    func testValidationRejectsIncompleteDuplicateAndNonCanonicalSamples() {
        var incomplete = artifact()
        incomplete.samples.removeLast()
        XCTAssertThrowsError(try incomplete.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .incompleteSamples(expected: 300, actual: 299))
        }

        var nonCanonical = artifact()
        nonCanonical.samples.swapAt(0, 1)
        XCTAssertThrowsError(try nonCanonical.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .nonCanonicalSample(position: 0))
        }

        var duplicatePrompt = artifact()
        duplicatePrompt.promptDigests[1] = duplicatePrompt.promptDigests[0]
        XCTAssertThrowsError(try duplicatePrompt.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidPromptDigests)
        }

        var shortProtocol = artifact()
        shortProtocol.promptDigests.removeLast()
        XCTAssertThrowsError(try shortProtocol.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidPromptCount(expected: 20, actual: 19))
        }
    }

    func testValidationRejectsNonFiniteNegativeAndUndeclaredMetrics() {
        var nonFinite = artifact()
        nonFinite.samples[0].relativeAttentionOutputError = .nan
        XCTAssertThrowsError(try nonFinite.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidMetric(sample: 0))
        }

        var negative = artifact()
        negative.samples[4].attentionScoreError = -0.001
        XCTAssertThrowsError(try negative.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidMetric(sample: 4))
        }

        var wrongPairOrder = artifact()
        wrongPairOrder.precisionPairs.swapAt(0, 1)
        XCTAssertThrowsError(try wrongPairOrder.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidPrecisionPairs)
        }
    }

    func testValidationRequiresCanonicalExactModelConfigSHA256() {
        var uppercase = artifact()
        uppercase.modelConfigSHA256 = String(repeating: "C", count: 64)
        XCTAssertThrowsError(try uppercase.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidDigest(uppercase.modelConfigSHA256))
        }

        var short = artifact()
        short.modelConfigSHA256 = String(repeating: "c", count: 63)
        XCTAssertThrowsError(try short.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidDigest(short.modelConfigSHA256))
        }
    }

    func testValidationPinsIndependentGroupAndProtocolSemantics() throws {
        XCTAssertNoThrow(try artifact(groupSize: 64).validated())
        XCTAssertNoThrow(try artifact(groupSize: 128).validated())

        var unsupported = artifact(groupSize: 32)
        XCTAssertThrowsError(try unsupported.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .unsupportedGroupSize(32))
        }

        unsupported = artifact()
        unsupported.captureMode = "propagated-quantization-error"
        XCTAssertThrowsError(try unsupported.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                    .invalidProtocol("captureMode"))
        }

        unsupported = artifact()
        unsupported.metricProtocolID = "relative-frobenius-norm"
        XCTAssertThrowsError(try unsupported.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidProtocol("metricProtocolID"))
        }

        unsupported = artifact()
        unsupported.metricAccumulationDType = "float16"
        XCTAssertThrowsError(try unsupported.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidProtocol("metricAccumulationDType"))
        }

        unsupported = artifact()
        unsupported.dbscanEpsilon = 0.051
        XCTAssertThrowsError(try unsupported.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .invalidProtocol("dbscanEpsilon"))
        }
    }

    func testAnalysisAggregatesInPromptOrderParetoPrunesAndClusters() throws {
        let analysis = try artifact().analyzed()

        XCTAssertEqual(analysis.layers.count, 5)
        for (actual, expected) in zip(
            analysis.layers[0].aggregates.map(
                \.relativeAttentionOutputError),
            [0.01, 0.04, 0.08])
        {
            XCTAssertEqual(actual, expected, accuracy: 1e-15)
        }
        XCTAssertEqual(analysis.layers[0].paretoPairs, pairs)
        XCTAssertEqual(analysis.layers[2].paretoPairs, [pairs[2]])

        XCTAssertEqual(analysis.groups.count, 3)
        XCTAssertEqual(analysis.groups[0].id, 0)
        XCTAssertEqual(analysis.groups[0].layers, [0, 1])
        XCTAssertEqual(analysis.groups[0].allowedPairs, pairs)
        XCTAssertEqual(analysis.groups[1].layers, [2, 3])
        XCTAssertEqual(analysis.groups[1].allowedPairs, [pairs[2]])
        XCTAssertEqual(analysis.groups[2].layers, [4])
        XCTAssertEqual(analysis.groups[2].allowedPairs, [pairs[2]])
    }

    func testDBSCANIncludesExactEpsilonBoundaryAndCanonicalizesNoise() throws {
        let clusters = try KVTunerDeterministicDBSCAN.cluster(
            points: [
                KVTunerDBSCANPoint(layer: 8, features: [0.0]),
                KVTunerDBSCANPoint(layer: 3, features: [0.05]),
                KVTunerDBSCANPoint(layer: 5, features: [0.30]),
            ],
            epsilon: 0.05,
            minSamples: 2)

        XCTAssertEqual(clusters, [[3, 8], [5]])
    }

    func testDBSCANExpandsDensityReachableNeighborsDeterministically() throws {
        let clusters = try KVTunerDeterministicDBSCAN.cluster(
            points: [
                KVTunerDBSCANPoint(layer: 12, features: [0.08]),
                KVTunerDBSCANPoint(layer: 4, features: [0.00]),
                KVTunerDBSCANPoint(layer: 9, features: [0.04]),
            ],
            epsilon: 0.05,
            minSamples: 2)

        XCTAssertEqual(clusters, [[4, 9, 12]])
    }

    func testAnalysisIsInvariantToPromptValuesThatPreserveTheOrderedMean() throws {
        var shifted = artifact()
        for index in shifted.samples.indices {
            let delta = shifted.samples[index].promptIndex < 10
                ? -0.001 : 0.001
            shifted.samples[index].relativeKeyError += delta
            shifted.samples[index].relativeValueError += delta
            shifted.samples[index].attentionScoreError += delta
            shifted.samples[index].relativeAttentionOutputError += delta
        }

        let shiftedAnalysis = try shifted.analyzed()
        let baseline = try artifact().analyzed()
        XCTAssertEqual(shiftedAnalysis.groups, baseline.groups)
        for layer in baseline.layers.indices {
            for pair in baseline.layers[layer].aggregates.indices {
                XCTAssertEqual(
                    shiftedAnalysis.layers[layer].aggregates[pair]
                        .relativeAttentionOutputError,
                    baseline.layers[layer].aggregates[pair]
                        .relativeAttentionOutputError,
                    accuracy: 1e-15)
            }
        }
    }
}
