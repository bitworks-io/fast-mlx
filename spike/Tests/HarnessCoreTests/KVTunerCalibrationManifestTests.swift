import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerCalibrationManifestTests: XCTestCase {
    func testPinnedManifestValidatesRoundTripsAndExposesCompleteLeakageSet() throws {
        let manifest = try KVTunerTestFixtures.calibrationManifest().validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(manifest)
        let decoded = try JSONDecoder().decode(
            KVTunerCalibrationManifest.self, from: data)

        XCTAssertEqual(try decoded.validated(), manifest)
        XCTAssertEqual(manifest.sensitivityPrompts.count, 20)
        XCTAssertEqual(manifest.searchPrompts.count, 200)
        XCTAssertEqual(manifest.calibrationEntryDigests.count, 220)
        XCTAssertEqual(manifest.calibrationSourceItemDigests.count, 200)
        XCTAssertEqual(
            KVTunerSourceItemDigest.gsm8kTestItem(
                repository: manifest.datasetSourceRepository,
                commit: manifest.datasetSourceCommit,
                testDataSHA256: manifest.testDataSHA256,
                testIndex: 0),
            "e6c09c41a6b0a16d24d8a7703822530b8a8b808e9c4dda09407cf910d36d5d1c")
        XCTAssertEqual(
            manifest.calibrationEntryDigests,
            manifest.calibrationEntryDigests.sorted())

        var missingStrongConfigIdentity = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        missingStrongConfigIdentity.removeValue(forKey: "modelConfigSHA256")
        let missingData = try JSONSerialization.data(
            withJSONObject: missingStrongConfigIdentity)
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerCalibrationManifest.self, from: missingData))
    }

    func testManifestRejectsMissingReorderedOrDuplicatePromptIdentity() throws {
        var missing = try KVTunerTestFixtures.calibrationManifest()
        missing.searchPrompts.removeLast()
        XCTAssertThrowsError(try missing.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidPromptCount(
                    phase: "search", expected: 200, actual: 199))
        }

        var reordered = try KVTunerTestFixtures.calibrationManifest()
        reordered.sensitivityPrompts.swapAt(0, 1)
        XCTAssertThrowsError(try reordered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .nonCanonicalPrompt(phase: "sensitivity", position: 0))
        }

        var duplicate = try KVTunerTestFixtures.calibrationManifest()
        duplicate.searchPrompts[1] = duplicate.searchPrompts[0]
        XCTAssertThrowsError(try duplicate.validated())
    }

    func testManifestRejectsPromptHashListThatDoesNotMatchOfficialSource() throws {
        var tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.sensitivityPrompts[0].promptSHA256 =
            String(repeating: "f", count: 64)
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidPromptIdentity(
                    phase: "sensitivity", position: 0))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.searchPrompts[199].promptDigest =
            "0000000000000000"
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidPromptIdentity(phase: "search", position: 199))
        }
    }

    func testManifestPinsDatasetToolingAndFewShotProtocol() throws {
        var tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.datasetSourceCommit = String(repeating: "0", count: 40)
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("datasetSourceCommit"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.fewShotSeed = 7
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("fewShotSeed"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.fewShotIndexTableSHA256 = String(repeating: "0", count: 64)
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("fewShotIndexTableSHA256"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.lmEvalSourceRepository = "EleutherAI/lm-evaluation-harness"
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("lmEvalSourceRepository"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.promptListEncodingID = "newline-delimited-hashes"
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("promptListEncodingID"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.tokenizationProtocolID = "tokenizer-defaults"
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidProtocol("tokenizationProtocolID"))
        }
    }

    func testManifestRejectsMalformedModelAndTokenizerIdentity() throws {
        var tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.modelConfigHash = "unknown"
        XCTAssertThrowsError(try tampered.validated())

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.modelConfigSHA256 = String(repeating: "C", count: 64)
        XCTAssertThrowsError(try tampered.validated()) { error in
            XCTAssertEqual(
                error as? KVTunerCalibrationManifestError,
                .invalidIdentity("modelConfigSHA256"))
        }

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.checkpointManifestHash = "ABCDEF0123456789"
        XCTAssertThrowsError(try tampered.validated())

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.tokenizerSHA256 = String(repeating: "g", count: 64)
        XCTAssertThrowsError(try tampered.validated())

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.searchPrompts[0].tokenIDs[0] += 1
        XCTAssertThrowsError(try tampered.validated())

        tampered = try KVTunerTestFixtures.calibrationManifest()
        tampered.searchPrompts[0].promptUTF8 = Data("substituted".utf8)
        XCTAssertThrowsError(try tampered.validated())
    }

    func testSensitivityArtifactBindsExactManifestBytesAndSensitivityPrompts() throws {
        let manifest = try KVTunerTestFixtures.calibrationManifest().validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let manifestData = try encoder.encode(manifest)
        let manifestSHA = sha256Hex(manifestData)
        let pairs = KVTunerSensitivityArtifact.canonicalPrecisionPairs
        var samples: [KVTunerSensitivitySample] = []
        for promptIndex in manifest.sensitivityPrompts.indices {
            for pair in pairs {
                samples.append(KVTunerSensitivitySample(
                    promptIndex: promptIndex,
                    layer: 0,
                    keyBits: pair.keyBits,
                    valueBits: pair.valueBits,
                    relativeKeyError: 0.1,
                    relativeValueError: 0.1,
                    attentionScoreError: 0.1,
                    relativeAttentionOutputError: 0.1))
            }
        }
        var sensitivity = KVTunerSensitivityArtifact(
            schemaVersion: 2,
            matrixID: "kvarn-qwen3-32b-v1",
            modelConfigHash: manifest.modelConfigHash,
            modelConfigSHA256: manifest.modelConfigSHA256,
            checkpointManifestHash: manifest.checkpointManifestHash,
            tokenizerSHA256: manifest.tokenizerSHA256,
            calibrationCorpusID: manifest.corpusID,
            calibrationCorpusHash: manifestSHA,
            promptManifestSHA256: manifestSHA,
            promptDigests: manifest.sensitivityPrompts.map(\.promptDigest),
            quantizerID: "mlx-affine-asymmetric-v1",
            captureMode: "single-prefill-no-error-propagation-v1",
            groupSize: 64,
            layerCount: 1,
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
            samples: samples)

        XCTAssertNoThrow(try sensitivity.validated(
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData))

        sensitivity.modelConfigSHA256 = String(repeating: "d", count: 64)
        XCTAssertThrowsError(try sensitivity.validated(
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData)) { error in
                XCTAssertEqual(
                    error as? KVTunerSensitivityError,
                    .calibrationManifestMismatch)
            }
        sensitivity.modelConfigSHA256 = manifest.modelConfigSHA256

        sensitivity.promptDigests.swapAt(0, 1)
        XCTAssertThrowsError(try sensitivity.validated(
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData)) { error in
                XCTAssertEqual(
                    error as? KVTunerSensitivityError,
                    .calibrationManifestMismatch)
            }

        sensitivity.promptDigests.swapAt(0, 1)
        XCTAssertThrowsError(try sensitivity.validated(
            calibrationManifest: manifest,
            exactCalibrationManifestData: Data("{}".utf8))) { error in
                XCTAssertEqual(
                    error as? KVTunerSensitivityError,
                    .calibrationManifestMismatch)
            }
    }
}
