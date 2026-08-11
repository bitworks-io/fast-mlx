import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerScheduleTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v1"
    private let cellID = "kvtuner-g128-b4.5"
    private let configHash = "0123456789abcdef"
    private let configSHA256 = String(repeating: "d", count: 64)
    private let checkpointHash = "fedcba9876543210"
    private let checkpointContentSHA256 = String(repeating: "e", count: 64)

    private func validSchedule() -> KVTunerSchedule {
        KVTunerSchedule(
            schemaVersion: 4,
            matrixID: matrixID,
            cellID: cellID,
            modelConfigHash: configHash,
            modelConfigSHA256: configSHA256,
            checkpointManifestHash: checkpointHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: String(repeating: "c", count: 64),
            groupSize: 128,
            calibrationCorpusID: "kvtuner-calibration-v1",
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
            sourceSensitivityArtifactSHA256: String(repeating: "a", count: 64),
            sourceSearchArtifactSHA256: String(repeating: "b", count: 64),
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
            ])
    }

    private func validate(
        _ schedule: KVTunerSchedule,
        expectedCellID: String? = nil
    ) throws -> KVTunerSchedule {
        try schedule.validated(
            expectedLayerCount: 2,
            expectedMatrixID: matrixID,
            expectedCellID: expectedCellID ?? cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256)
    }

    func testCompletePinnedV4ScheduleValidatesAndRoundTripsJSON() throws {
        let validated = try validate(validSchedule())
        let encoded = try JSONEncoder().encode(validated)
        let decoded = try JSONDecoder().decode(KVTunerSchedule.self, from: encoded)

        XCTAssertEqual(validated.computedNominalAverageBits, 4.5, accuracy: 0)
        XCTAssertEqual(decoded, validated)
    }

    func testPriorSchemasCannotQualifyEvenWhenAllV3FieldsArePresent() {
        for schema in [1, 2] {
            var legacy = validSchedule()
            legacy.schemaVersion = schema

            XCTAssertThrowsError(try validate(legacy)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .unsupportedSchema(schema))
            }
        }
    }

    func testGenuineV2ShapeIsClassifiedAsUnsupportedBeforeV3FieldsDecode() throws {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(validSchedule())) as? [String: Any])
        object["schemaVersion"] = 2
        object.removeValue(forKey: "tokenizerSHA256")
        object.removeValue(forKey: "sourceSearchArtifactSHA256")
        let data = try JSONSerialization.data(withJSONObject: object)

        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerSchedule.self, from: data)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .unsupportedSchema(2))
            }
    }

    func testMissingV3SourceFieldCannotDecodeAsAQualifyingSchedule() throws {
        for field in [
            "sourceSensitivityArtifactSHA256",
            "sourceSearchArtifactSHA256",
        ] {
            let encoded = try JSONEncoder().encode(validSchedule())
            var object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: encoded)
                    as? [String: Any])
            object.removeValue(forKey: field)

            XCTAssertThrowsError(try JSONDecoder().decode(
                KVTunerSchedule.self,
                from: JSONSerialization.data(withJSONObject: object)))
        }
    }

    func testModelConfigAndCheckpointMustMatchExpectedRuntime() {
        XCTAssertThrowsError(try validSchedule().validated(
            expectedLayerCount: 2,
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: "aaaaaaaaaaaaaaaa",
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .modelConfigHashMismatch)
            }
        XCTAssertThrowsError(try validSchedule().validated(
            expectedLayerCount: 2,
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: "bbbbbbbbbbbbbbbb",
            expectedCheckpointContentSHA256:
                checkpointContentSHA256)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .checkpointManifestHashMismatch)
            }
    }

    func testRewrappedPromptFromSameSourceItemFailsLeakageGate() throws {
        let schedule = try validate(validSchedule())
        XCTAssertThrowsError(try schedule.validateEvaluationCorpus(
            id: "rewrapped-gsm8k-v1",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [schedule.calibrationSourceItemDigests[0]]
        )) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .evaluationCorpusLeaksCalibration)
        }
    }

    func testMatrixIDMustMatchRequestedMatrix() {
        var mismatched = validSchedule()
        mismatched.matrixID = "another-declared-matrix"

        XCTAssertThrowsError(try validate(mismatched)) { error in
            XCTAssertEqual(error as? KVTunerScheduleError, .matrixIDMismatch)
        }
    }

    func testCellIDMustMatchRequestedCell() {
        var mismatched = validSchedule()
        mismatched.cellID = "another-declared-cell"

        XCTAssertThrowsError(try validate(mismatched)) { error in
            XCTAssertEqual(error as? KVTunerScheduleError, .cellIDMismatch)
        }
    }

    func testCellGroupMustMatchScheduleGeometry() {
        var mismatched = validSchedule()
        mismatched.groupSize = 64

        XCTAssertThrowsError(try validate(mismatched)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .cellGroupSizeMismatch(cell: 128, schedule: 64))
        }
    }

    func testCellBudgetMustMatchScheduleNominalAverage() {
        var mismatched = validSchedule()
        mismatched.layers[1] = KVLayerPrecision(
            layer: 1, keyBits: 8, valueBits: 2)
        mismatched.nominalAverageBits = mismatched.computedNominalAverageBits

        XCTAssertEqual(mismatched.nominalAverageBits, 5.5, accuracy: 0)
        XCTAssertThrowsError(try validate(mismatched)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .cellNominalAverageBitsMismatch(cell: 4.5, schedule: 5.5))
        }
    }

    func testCellDescriptorMustUseCanonicalNumericSpellings() {
        for cellID in [
            "kvtuner-g0128-b4.5",
            "kvtuner-g128-b4.50",
        ] {
            var invalid = validSchedule()
            invalid.cellID = cellID

            XCTAssertThrowsError(try validate(
                invalid, expectedCellID: cellID)) { error in
                    XCTAssertEqual(
                        error as? KVTunerScheduleError,
                        .invalidCellDescriptor(cellID))
                }
        }
    }

    func testIdentifiersAndDigestsFailClosed() {
        var invalid = validSchedule()
        invalid.matrixID = " matrix"
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.cellID = "unknown"
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.objective = "has spaces"
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.modelConfigHash = "not-a-digest"
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.checkpointManifestHash = String(repeating: "A", count: 16)
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.calibrationCorpusHash = "1234"
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.sourceSensitivityArtifactSHA256 = String(repeating: "g", count: 64)
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.sourceSearchArtifactSHA256 = String(repeating: "G", count: 64)
        XCTAssertThrowsError(try validate(invalid))
    }

    func testSearchSeedAndObjectiveArePartOfTheRuntimeContract() {
        var invalid = validSchedule()
        invalid.seed = 7
        XCTAssertThrowsError(try validate(invalid)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .invalidSearchProtocol("seed"))
        }

        invalid = validSchedule()
        invalid.objective = "minimize-attention-error"
        XCTAssertThrowsError(try validate(invalid)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .invalidSearchProtocol("objective"))
        }
    }

    func testCalibrationEntryHashesAreNonemptyUniqueAndCanonical() {
        var invalid = validSchedule()
        invalid.calibrationEntryHashes = []
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.calibrationEntryHashes = [
            "2222222222222222", "2222222222222222",
        ]
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.calibrationEntryHashes.reverse()
        XCTAssertThrowsError(try validate(invalid))

        invalid = validSchedule()
        invalid.calibrationEntryHashes[0] = "not-a-digest"
        XCTAssertThrowsError(try validate(invalid))
    }

    func testCalibrationAndEvaluationEntryDigestsPinFNV1a64TextIdentity() {
        let reusedPrompt = "the exact prompt must not cross calibration"
        let evaluationFNV = fnv1a64(reusedPrompt.utf8)
        let calibrationSHA = sha256Hex(Data(reusedPrompt.utf8))

        var ambiguousCalibration = validSchedule()
        ambiguousCalibration.calibrationEntryHashes = [calibrationSHA]
        XCTAssertThrowsError(try validate(ambiguousCalibration)) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleError,
                .invalidDigest(calibrationSHA))
        }

        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "renamed-evaluation",
            hash: "4444444444444444",
            entryHashes: [calibrationSHA],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)])) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .invalidDigest(calibrationSHA))
            }

        var pinnedCalibration = validSchedule()
        pinnedCalibration.calibrationEntryHashes = [evaluationFNV]
        XCTAssertThrowsError(try pinnedCalibration.validateEvaluationCorpus(
            id: "renamed-evaluation",
            hash: "4444444444444444",
            entryHashes: [evaluationFNV],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)])) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .evaluationCorpusLeaksCalibration)
            }
    }

    func testGroupSizeMustBeOneOfTheDeclaredScheduleGeometries() {
        for groupSize in [0, 32, 256] {
            var invalid = validSchedule()
            invalid.groupSize = groupSize
            XCTAssertThrowsError(try validate(invalid)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .unsupportedGroupSize(groupSize))
            }
        }

        var g64 = validSchedule()
        g64.groupSize = 64
        g64.cellID = "kvtuner-g64-b4.5"
        XCTAssertNoThrow(try validate(
            g64, expectedCellID: g64.cellID))
    }

    func testLayersMustBeCanonicalExactZeroThroughLayerCountMinusOne() {
        var missing = validSchedule()
        missing.layers.removeLast()
        XCTAssertThrowsError(try validate(missing))

        var duplicate = validSchedule()
        duplicate.layers[1].layer = 0
        XCTAssertThrowsError(try validate(duplicate))

        var reversed = validSchedule()
        reversed.layers.reverse()
        XCTAssertThrowsError(try validate(reversed))

        var outOfRange = validSchedule()
        outOfRange.layers[1].layer = 2
        XCTAssertThrowsError(try validate(outOfRange))
    }

    func testOnlyDeclaredAsymmetricPerLayerPairsQualify() {
        for (keyBits, valueBits) in [
            (16, 2), (8, 8), (4, 4), (2, 2), (2, 4), (3, 2),
        ] {
            var invalid = validSchedule()
            invalid.layers[1].keyBits = keyBits
            invalid.layers[1].valueBits = valueBits
            XCTAssertThrowsError(try validate(invalid)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleError,
                    .unsupportedPrecision(
                        layer: 1, keyBits: keyBits, valueBits: valueBits))
            }
        }

        for (keyBits, valueBits, cellID) in [
            (8, 4, "kvtuner-g128-b6.0"),
            (8, 2, "kvtuner-g128-b5.5"),
            (4, 2, "kvtuner-g128-b4.5"),
        ] {
            var runnable = validSchedule()
            runnable.layers[1].keyBits = keyBits
            runnable.layers[1].valueBits = valueBits
            runnable.nominalAverageBits =
                runnable.computedNominalAverageBits
            runnable.cellID = cellID
            runnable.objective =
                "maximize-gsm8k-accuracy-at-b\(runnable.nominalAverageBits)"
            XCTAssertNoThrow(try validate(
                runnable, expectedCellID: cellID))
        }
    }

    func testNominalAverageMustBeFinitePositiveAndExact() {
        for value in [Double.nan, .infinity, -.infinity, 0, -1] {
            var invalid = validSchedule()
            invalid.nominalAverageBits = value
            XCTAssertThrowsError(try validate(invalid))
        }

        var wrongAverage = validSchedule()
        wrongAverage.nominalAverageBits = 4.500_000_000_01
        XCTAssertThrowsError(try validate(wrongAverage))
    }

    func testNominalAverageComputationDoesNotTrapOnUntrustedIntegers() {
        var schedule = validSchedule()
        schedule.layers = [
            KVLayerPrecision(
                layer: 0, keyBits: Int.max, valueBits: Int.max),
        ]

        XCTAssertTrue(schedule.computedNominalAverageBits.isFinite)
        XCTAssertEqual(
            schedule.computedNominalAverageBits,
            Double(Int.max),
            accuracy: 0)
        XCTAssertThrowsError(try schedule.validated(
            expectedLayerCount: 1,
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: configHash,
            expectedModelConfigSHA256: configSHA256,
            expectedCheckpointManifestHash: checkpointHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256))
    }

    func testEvaluationCorpusRejectsAggregateOrEntryLevelCalibrationLeakage() {
        let schedule = validSchedule()
        XCTAssertNoThrow(try schedule.validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555", "6666666666666666"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))

        XCTAssertThrowsError(try schedule.validateEvaluationCorpus(
            id: schedule.calibrationCorpusID,
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
        XCTAssertThrowsError(try schedule.validateEvaluationCorpus(
            id: "renamed-corpus",
            hash: schedule.calibrationCorpusHash,
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
        XCTAssertThrowsError(try schedule.validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["2222222222222222", "5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
    }

    func testMalformedEvaluationIdentityFailsBeforeLeakageComparison() {
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "evaluation corpus",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "not-a-digest",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: [],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["not-a-digest"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
    }

    func testEvaluationLeakageCheckAlsoRejectsMalformedStoredCalibrationIdentity() {
        var invalid = validSchedule()
        invalid.calibrationCorpusID = "calibration corpus"
        XCTAssertThrowsError(try invalid.validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))

        invalid = validSchedule()
        invalid.calibrationCorpusHash = "not-a-digest"
        XCTAssertThrowsError(try invalid.validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))

        invalid = validSchedule()
        invalid.calibrationEntryHashes = []
        XCTAssertThrowsError(try invalid.validateEvaluationCorpus(
            id: "measurement-corpus-v2",
            hash: "4444444444444444",
            entryHashes: ["5555555555555555"],
            sourceProvenance: .canonicalSourceItems,
            sourceItemHashes: [String(repeating: "f", count: 64)]))
    }
}
