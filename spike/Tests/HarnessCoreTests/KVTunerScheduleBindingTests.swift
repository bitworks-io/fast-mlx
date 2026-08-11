import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerScheduleBindingTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v1"
    private let cellID = "kvtuner-g128-b4.5"
    private let configHash = "0123456789abcdef"
    private let configSHA256 = String(repeating: "d", count: 64)
    private let checkpointHash = "fedcba9876543210"
    private let checkpointContentSHA256 = String(repeating: "e", count: 64)

    private var evaluation: KVTunerEvaluationCorpusIdentity {
        get throws {
            try KVTunerEvaluationCorpusIdentity(
                id: "evaluation-v1",
                aggregateDigest: "4444444444444444",
                canonicalEntryDigests: [
                    "5555555555555555",
                    "6666666666666666",
                ],
                canonicalSourceItemDigests: [
                    sha256Hex(Data("evaluation-source".utf8))
                ])
        }
    }

    private func selection(
        modelConfigSHA256: String? = nil,
        checkpointContentSHA256: String? = nil,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]? = nil
    ) throws -> KVTunerRuntimeSelection {
        let selectedModelConfigSHA256 =
            modelConfigSHA256 ?? configSHA256
        let selectedCheckpointContentSHA256 =
            checkpointContentSHA256 ?? self.checkpointContentSHA256
        let schedule = KVTunerSchedule(
            schemaVersion: 4,
            matrixID: matrixID,
            cellID: cellID,
            modelConfigHash: configHash,
            modelConfigSHA256: selectedModelConfigSHA256,
            checkpointManifestHash: checkpointHash,
            checkpointContentSHA256:
                selectedCheckpointContentSHA256,
            tokenizerSHA256: String(repeating: "c", count: 64),
            groupSize: 128,
            calibrationCorpusID: "calibration-v1",
            calibrationCorpusHash: "1111111111111111",
            calibrationEntryHashes: [
                "2222222222222222",
                "3333333333333333",
            ],
            calibrationSourceItemDigests: (0..<200).map {
                sha256Hex(Data("source-\($0)".utf8))
            }.sorted(),
            seed: 1234,
            objective: "maximize-gsm8k-accuracy-at-b4.5",
            nominalAverageBits: 4.5,
            sourceSensitivityArtifactSHA256: String(
                repeating: "a", count: 64),
            sourceSearchArtifactSHA256: String(
                repeating: "b", count: 64),
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
            ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try KVTunerRuntimeSelection.loadForTesting(
            artifactData: encoder.encode(schedule),
            expectedLayerCount: 2,
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256:
                selectedModelConfigSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                selectedCheckpointContentSHA256,
            evaluationCorpora: evaluationCorpora ?? [try evaluation])
    }

    private func load(
        _ data: Data,
        expectedMatrixID: String? = nil,
        expectedCellID: String? = nil,
        expectedModelConfigHash: String? = nil,
        expectedCheckpointManifestHash: String? = nil,
        expectedLayerCount: Int = 2,
        requiredEvaluationCorpus: KVTunerEvaluationCorpusIdentity? = nil
    ) throws -> KVTunerScheduleBinding {
        try KVTunerScheduleBinding.load(
            from: data,
            expectedMatrixID: expectedMatrixID ?? matrixID,
            expectedCellID: expectedCellID ?? cellID,
            expectedModelConfigHash: expectedModelConfigHash ?? configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash:
                expectedCheckpointManifestHash ?? checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: expectedLayerCount,
            requiredEvaluationCorpus:
                requiredEvaluationCorpus ?? (try evaluation))
    }

    private func mutatedBinding(
        _ binding: KVTunerScheduleBinding,
        mutate: (inout [String: Any]) -> Void
    ) throws -> Data {
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(binding)) as? [String: Any])
        mutate(&object)
        return try JSONSerialization.data(withJSONObject: object)
    }

    func testConstructsFromAuthenticatedSelectionAndRoundTrips() throws {
        let selection = try selection()
        let binding = KVTunerScheduleBinding(selection: selection)
        let data = try JSONEncoder().encode(binding)
        let loaded = try load(data)

        XCTAssertEqual(loaded, binding)
        XCTAssertEqual(loaded.schemaVersion, 4)
        XCTAssertEqual(loaded.scheduleSchemaVersion, 4)
        XCTAssertEqual(loaded.artifactSHA256, selection.artifactSHA256)
        XCTAssertEqual(
            loaded.qualificationBundleSHA256,
            selection.qualificationBundleSHA256)
        XCTAssertEqual(loaded.matrixID, matrixID)
        XCTAssertEqual(loaded.cellID, cellID)
        XCTAssertEqual(loaded.modelConfigHash, configHash)
        XCTAssertEqual(loaded.modelConfigSHA256, configSHA256)
        XCTAssertEqual(loaded.checkpointManifestHash, checkpointHash)
        XCTAssertEqual(
            loaded.checkpointContentSHA256,
            checkpointContentSHA256)
        XCTAssertEqual(loaded.groupSize, 128)
        XCTAssertEqual(
            loaded.promptDigestAlgorithm,
            KVTunerPromptDigest.algorithm)
        XCTAssertEqual(loaded.calibrationCorpusID, "calibration-v1")
        XCTAssertEqual(
            loaded.calibrationCorpusHash,
            "1111111111111111")
        XCTAssertEqual(
            loaded.calibrationEntryDigests,
            ["2222222222222222", "3333333333333333"])
        XCTAssertEqual(loaded.seed, 1234)
        XCTAssertEqual(
            loaded.objective,
            "maximize-gsm8k-accuracy-at-b4.5")
        XCTAssertEqual(loaded.nominalAverageBits, 4.5, accuracy: 0)
        XCTAssertEqual(
            loaded.sourceSensitivityArtifactSHA256,
            String(repeating: "a", count: 64))
        XCTAssertEqual(
            loaded.sourceSearchArtifactSHA256,
            String(repeating: "b", count: 64))
        XCTAssertEqual(loaded.evaluationCorpora, [try evaluation])
        XCTAssertEqual(loaded.layers, selection.layers)
        requireSendable(loaded)
    }

    func testGenuineV2BindingShapeIsClassifiedBeforeV3FieldsDecode() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())
        let data = try mutatedBinding(binding) { object in
            object["schemaVersion"] = 2
            for field in [
                "tokenizerSHA256", "seed", "objective",
                "nominalAverageBits", "sourceSensitivityArtifactSHA256",
                "sourceSearchArtifactSHA256",
            ] {
                object.removeValue(forKey: field)
            }
        }

        XCTAssertThrowsError(try load(data)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleBindingError,
                .unsupportedBindingSchema(2))
        }
    }

    func testGenuineV3BindingShapeIsClassifiedBeforeV4FieldsDecode() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())
        let data = try mutatedBinding(binding) { object in
            object["schemaVersion"] = 3
            object.removeValue(forKey: "modelConfigSHA256")
            object.removeValue(forKey: "checkpointContentSHA256")
        }

        XCTAssertThrowsError(try load(data)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleBindingError,
                .unsupportedBindingSchema(3))
        }
    }

    func testValidationRequiresEveryExternalIdentityAndExactEvaluation() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())
        let evaluation = try evaluation

        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: "another-matrix",
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation))
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: "kvtuner-g64-b4.5",
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation))
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: "aaaaaaaaaaaaaaaa",
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation))
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: String(repeating: "0", count: 64),
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleBindingError,
                    .invalidSchedule(.modelConfigSHA256Mismatch))
            }
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: "bbbbbbbbbbbbbbbb",
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation))
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                String(repeating: "0", count: 64),
            expectedLayerCount: 2,
            requiredEvaluationCorpus: evaluation)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleBindingError,
                    .invalidSchedule(.checkpointContentSHA256Mismatch))
            }
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 3,
            requiredEvaluationCorpus: evaluation))

        let sameNameButDifferentPrompt = try KVTunerEvaluationCorpusIdentity(
            id: evaluation.id,
            aggregateDigest: evaluation.aggregateDigest,
            canonicalEntryDigests: ["7777777777777777"],
            canonicalSourceItemDigests: [
                sha256Hex(Data("different-prompt-source".utf8))
            ])
        XCTAssertThrowsError(try binding.validated(
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedLayerCount: 2,
            requiredEvaluationCorpus: sameNameButDifferentPrompt))
    }

    func testDecodedBindingRejectsInvalidArtifactDigestLayerOrderAndPair() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())

        let invalidDigest = try mutatedBinding(binding) {
            $0["artifactSHA256"] = "not-a-digest"
        }
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                KVTunerScheduleBinding.self, from: invalidDigest))

        let invalidOrder = try mutatedBinding(binding) { object in
            var layers = object["layers"] as! [[String: Any]]
            layers.swapAt(0, 1)
            object["layers"] = layers
        }
        XCTAssertThrowsError(try load(invalidOrder))

        let invalidPair = try mutatedBinding(binding) { object in
            var layers = object["layers"] as! [[String: Any]]
            layers[0]["keyBits"] = 2
            layers[0]["valueBits"] = 2
            object["layers"] = layers
        }
        XCTAssertThrowsError(try load(invalidPair))
    }

    func testDecodedBindingRequiresSearchProtocolAndBothSourceArtifacts() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())

        for field in [
            "modelConfigSHA256",
            "checkpointContentSHA256",
            "seed",
            "objective",
            "sourceSensitivityArtifactSHA256",
            "sourceSearchArtifactSHA256",
        ] {
            let missing = try mutatedBinding(binding) {
                $0.removeValue(forKey: field)
            }
            XCTAssertThrowsError(
                try JSONDecoder().decode(
                    KVTunerScheduleBinding.self,
                    from: missing),
                "missing \(field) must fail closed")
        }

        let invalidSearchDigest = try mutatedBinding(binding) {
            $0["sourceSearchArtifactSHA256"] = String(
                repeating: "c", count: 63)
        }
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerScheduleBinding.self,
            from: invalidSearchDigest))
    }

    func testDecodedBindingRevalidatesCalibrationEvaluationDisjointness() throws {
        let binding = KVTunerScheduleBinding(selection: try selection())
        let leakedEvaluation = try mutatedBinding(binding) { object in
            var corpora = object["evaluationCorpora"] as! [[String: Any]]
            corpora[0]["canonicalEntryDigests"] = [
                "2222222222222222",
                "6666666666666666",
            ]
            object["evaluationCorpora"] = corpora
        }

        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerScheduleBinding.self,
            from: leakedEvaluation))
    }

    func testSameScheduleIgnoresOnlyEvaluationCorpusList() throws {
        let first = KVTunerScheduleBinding(selection: try selection())
        let otherEvaluation = try KVTunerEvaluationCorpusIdentity(
            id: "task-evaluation-v1",
            aggregateDigest: "7777777777777777",
            canonicalEntryDigests: ["8888888888888888"],
            canonicalSourceItemDigests: [
                sha256Hex(Data("other-evaluation-source".utf8))
            ])
        let second = KVTunerScheduleBinding(
            selection: try selection(evaluationCorpora: [otherEvaluation]))

        XCTAssertNotEqual(first, second)
        XCTAssertTrue(first.sameSchedule(as: second))

        let changedData = try mutatedBinding(second) { object in
            object["artifactSHA256"] = String(repeating: "b", count: 64)
        }
        let changed = try JSONDecoder().decode(
            KVTunerScheduleBinding.self, from: changedData)
        XCTAssertFalse(first.sameSchedule(as: changed))

        let changedStrongIdentity = KVTunerScheduleBinding(
            selection: try selection(
                checkpointContentSHA256:
                    String(repeating: "f", count: 64)))
        XCTAssertFalse(first.sameSchedule(as: changedStrongIdentity))
    }
}

private func requireSendable<T: Sendable>(_: T) {}
