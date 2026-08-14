import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerCandidateRuntimePolicyTests: XCTestCase {
    private enum TokenizerStubError: Error {
        case unknownPrompt
    }

    private struct Inputs {
        let configData: Data
        let manifest: KVTunerCalibrationManifest
        let manifestData: Data
        let sensitivity: KVTunerSensitivityArtifact
        let sensitivityData: Data
        let candidates: [KVTunerScheduleCandidate]
    }

    private func inputs(
        modelLayerCount: Int = 4,
        modelConfigSHA256Override: String? = nil
    ) throws -> Inputs {
        let configData = Data(
            "{\"model_type\":\"qwen3\",\"num_hidden_layers\":\(modelLayerCount)}"
                .utf8)
        var manifest = try KVTunerTestFixtures.calibrationManifest()
        manifest.modelConfigHash = fnv1a64(configData)
        manifest.modelConfigSHA256 = modelConfigSHA256Override
            ?? sha256Hex(configData)
        manifest = try manifest.validated()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let manifestData = try encoder.encode(manifest)
        let manifestSHA256 = sha256Hex(manifestData)
        let pairs = KVTunerSensitivityArtifact.canonicalPrecisionPairs
        let layerOutputs = [
            [0.01, 0.05, 0.20],
            [0.01, 0.05, 0.20],
            [0.02, 0.06, 0.10],
            [0.02, 0.06, 0.10],
        ]
        var samples: [KVTunerSensitivitySample] = []
        for promptIndex in manifest.sensitivityPrompts.indices {
            for layer in layerOutputs.indices {
                for (pairIndex, pair) in pairs.enumerated() {
                    let output = layerOutputs[layer][pairIndex]
                    samples.append(KVTunerSensitivitySample(
                        promptIndex: promptIndex,
                        layer: layer,
                        keyBits: pair.keyBits,
                        valueBits: pair.valueBits,
                        relativeKeyError: output,
                        relativeValueError: output,
                        attentionScoreError: output,
                        relativeAttentionOutputError: output))
                }
            }
        }
        let sensitivity = try KVTunerSensitivityArtifact(
            schemaVersion: 3,
            matrixID: "kvarn-qwen3-32b-v1",
            modelConfigHash: manifest.modelConfigHash,
            modelConfigSHA256: manifest.modelConfigSHA256,
            checkpointManifestHash: manifest.checkpointManifestHash,
            checkpointContentSHA256: manifest.checkpointContentSHA256,
            tokenizerSHA256: manifest.tokenizerSHA256,
            calibrationCorpusID: manifest.corpusID,
            calibrationCorpusHash: manifestSHA256,
            promptManifestSHA256: manifestSHA256,
            promptDigests: manifest.sensitivityPrompts.map(\.promptDigest),
            quantizerID: "mlx-affine-asymmetric-v1",
            captureMode: "single-prefill-no-error-propagation-v1",
            groupSize: 64,
            layerCount: layerOutputs.count,
            precisionPairs: pairs,
            metricProtocolID:
                "kvtuner-v5-elementwise-mean-absolute-v1",
            metricAccumulationDType: "float32",
            denominatorEpsilon: 1e-8,
            aggregationID: "ordered-incremental-mean-v1",
            dbscanEpsilon: 0.05,
            dbscanMinSamples: 2,
            captureEnvironment:
                KVTunerTestFixtures.sensitivityCaptureEnvironment(),
            samples: samples).validated(
                calibrationManifest: manifest,
                exactCalibrationManifestData: manifestData)
        let sensitivityData = try encoder.encode(sensitivity)
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: sensitivity.analyzed(),
            targetPairBitTotal: 36,
            maxCandidates: 10)
        return Inputs(
            configData: configData,
            manifest: manifest,
            manifestData: manifestData,
            sensitivity: sensitivity,
            sensitivityData: sensitivityData,
            candidates: candidates)
    }

    private func tokenizer(
        for manifest: KVTunerCalibrationManifest
    ) -> (String) throws -> [Int] {
        let rows = manifest.sensitivityPrompts + manifest.searchPrompts
        let tokenIDsByPrompt = Dictionary(uniqueKeysWithValues: rows.map {
            (String(decoding: $0.promptUTF8, as: UTF8.self), $0.tokenIDs)
        })
        return { prompt in
            guard let tokenIDs = tokenIDsByPrompt[prompt] else {
                throw TokenizerStubError.unknownPrompt
            }
            return tokenIDs
        }
    }

    private func load(
        _ inputs: Inputs,
        candidateOrdinal: Int = 0,
        maxCandidates: Int = 10,
        tokenizePrompt: ((String) throws -> [Int])? = nil
    ) throws -> KVTunerCandidateRuntimePolicy {
        try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: maxCandidates,
            candidateOrdinal: candidateOrdinal,
            tokenizePrompt: tokenizePrompt ?? tokenizer(for: inputs.manifest))
    }

    func testLoadAuthenticatesExactInputsAndDerivesImmutablePolicy() throws {
        let inputs = try inputs()
        let expected = inputs.candidates[0]
        let policy = try load(inputs)

        XCTAssertEqual(policy.matrixID, inputs.sensitivity.matrixID)
        XCTAssertEqual(
            policy.calibrationManifestSHA256,
            sha256Hex(inputs.manifestData))
        XCTAssertEqual(
            policy.sourceSensitivityArtifactSHA256,
            sha256Hex(inputs.sensitivityData))
        XCTAssertEqual(
            policy.candidateListSHA256,
            try KVTunerScheduleSearch.candidateListSHA256(inputs.candidates))
        XCTAssertEqual(policy.candidateCount, inputs.candidates.count)
        XCTAssertEqual(policy.candidateOrdinal, expected.ordinal)
        XCTAssertEqual(
            policy.candidateSHA256,
            try KVTunerScheduleSearch.candidateSHA256(expected))
        XCTAssertEqual(policy.targetPairBitTotal, 36)
        XCTAssertEqual(policy.groupSize, inputs.sensitivity.groupSize)
        XCTAssertEqual(policy.modelConfigHash, inputs.manifest.modelConfigHash)
        XCTAssertEqual(
            policy.modelConfigSHA256,
            sha256Hex(inputs.configData))
        XCTAssertEqual(
            policy.checkpointManifestHash,
            inputs.manifest.checkpointManifestHash)
        XCTAssertEqual(
            policy.tokenizerSHA256,
            inputs.manifest.tokenizerSHA256)
        XCTAssertEqual(
            policy.layers,
            expected.layers.map {
                KVTunerRuntimeLayerPolicy(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            })
        XCTAssertEqual(policy.runtimePolicySHA256.count, 64)
        XCTAssertTrue(policy.runtimePolicySHA256.allSatisfy {
            $0.isHexDigit && !$0.isUppercase
        })
        XCTAssertEqual(
            policy.runtimePolicySHA256,
            "20f3606b16285d1f560c3e8c81b9ac563cfc3adbd39d392077639f0ee1195b52")
        XCTAssertEqual(policy, try load(inputs))
    }

    func testLoadRejectsMalformedAndMismatchedExactSources() throws {
        let inputs = try inputs()

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: Data("{}".utf8),
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .invalidCalibrationManifest)
            }

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: Data("{}".utf8),
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .invalidSensitivityArtifact)
            }

        let differentConfig = Data(
            "{\"model_type\":\"qwen3\",\"num_hidden_layers\":4,\"changed\":true}"
                .utf8)
        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: Data("{".utf8),
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .invalidModelConfig(.malformedConfig))
            }

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: differentConfig,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .modelConfigIdentityMismatch)
            }

        let wrongLayers = try self.inputs(modelLayerCount: 5)
        XCTAssertThrowsError(try load(wrongLayers)) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .layerCountMismatch(expected: 4, actual: 5))
        }

        let wrongExactSHA = try self.inputs(
            modelConfigSHA256Override: String(repeating: "d", count: 64))
        XCTAssertThrowsError(try load(wrongExactSHA)) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .modelConfigIdentityMismatch)
        }
    }

    func testLoadRejectsCheckpointTokenizerAndLiveTokenizationMismatch() throws {
        let inputs = try inputs()

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash: "0000000000000000",
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .checkpointIdentityMismatch)
            }

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                String(repeating: "f", count: 64),
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .checkpointIdentityMismatch)
            }

        XCTAssertThrowsError(try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: String(repeating: "0", count: 64),
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: 0,
            tokenizePrompt: tokenizer(for: inputs.manifest))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .tokenizerIdentityMismatch)
            }

        var calls = 0
        XCTAssertThrowsError(try load(inputs, tokenizePrompt: { prompt in
            defer { calls += 1 }
            let expected = try self.tokenizer(for: inputs.manifest)(prompt)
            return calls == 7 ? expected + [999] : expected
        })) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .promptTokenizationMismatch(
                    phase: "sensitivity", position: 7))
        }

        calls = 0
        XCTAssertThrowsError(try load(inputs, tokenizePrompt: { prompt in
            defer { calls += 1 }
            let expected = try self.tokenizer(for: inputs.manifest)(prompt)
            return calls == KVTunerCalibrationManifest.requiredSensitivityCount
                + 3 ? expected + [999] : expected
        })) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .promptTokenizationMismatch(
                    phase: "search", position: 3))
        }
    }

    func testLoadRejectsTruncatedEnumerationAndOutOfRangeCandidate() throws {
        let inputs = try inputs()
        XCTAssertGreaterThan(inputs.candidates.count, 1)

        XCTAssertThrowsError(try load(inputs, maxCandidates: 1)) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .candidateEnumerationTruncated(limit: 1))
        }
        XCTAssertThrowsError(try load(
            inputs, candidateOrdinal: inputs.candidates.count)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimePolicyError,
                    .candidateOrdinalOutOfRange(inputs.candidates.count))
            }
    }

    func testRuntimePolicyDigestBindsCandidateAndExactSensitivityBytes() throws {
        let inputs = try inputs()
        let first = try load(inputs, candidateOrdinal: 0)
        let second = try load(inputs, candidateOrdinal: 1)
        XCTAssertNotEqual(first.candidateSHA256, second.candidateSHA256)
        XCTAssertNotEqual(first.runtimePolicySHA256, second.runtimePolicySHA256)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let reformattedSensitivityData = try encoder.encode(inputs.sensitivity)
        XCTAssertNotEqual(
            reformattedSensitivityData,
            inputs.sensitivityData)
        let reformattedInputs = Inputs(
            configData: inputs.configData,
            manifest: inputs.manifest,
            manifestData: inputs.manifestData,
            sensitivity: inputs.sensitivity,
            sensitivityData: reformattedSensitivityData,
            candidates: inputs.candidates)
        let reformatted = try load(reformattedInputs)

        XCTAssertEqual(first.candidateSHA256, reformatted.candidateSHA256)
        XCTAssertEqual(
            first.candidateListSHA256,
            reformatted.candidateListSHA256)
        XCTAssertNotEqual(
            first.sourceSensitivityArtifactSHA256,
            reformatted.sourceSensitivityArtifactSHA256)
        XCTAssertNotEqual(
            first.runtimePolicySHA256,
            reformatted.runtimePolicySHA256)
    }
}
