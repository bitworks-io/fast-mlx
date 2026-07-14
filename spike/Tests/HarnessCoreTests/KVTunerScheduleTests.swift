import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerScheduleTests: XCTestCase {
    private func validSchedule() -> KVTunerSchedule {
        KVTunerSchedule(
            schemaVersion: 1,
            modelConfigHash: "model-config-abc",
            calibrationCorpusID: "kvtuner-calibration-v1",
            calibrationCorpusHash: "calibration-hash-123",
            seed: 7,
            objective: "minimize-attention-error-at-4.5-average-bits",
            nominalAverageBits: 4.5,
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
            ])
    }

    func testCompletePinnedScheduleValidatesAndRoundTripsJSON() throws {
        let schedule = validSchedule()

        let validated = try schedule.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc")
        let encoded = try JSONEncoder().encode(validated)
        let decoded = try JSONDecoder().decode(KVTunerSchedule.self, from: encoded)

        XCTAssertEqual(validated.computedNominalAverageBits, 4.5, accuracy: 0)
        XCTAssertEqual(decoded, validated)
    }

    func testMissingDuplicateOrOutOfRangeLayerFailsClosed() {
        var missing = validSchedule()
        missing.layers = [KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4)]
        XCTAssertThrowsError(try missing.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc"))

        var duplicate = validSchedule()
        duplicate.layers = [
            KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            KVLayerPrecision(layer: 0, keyBits: 4, valueBits: 2),
        ]
        XCTAssertThrowsError(try duplicate.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc"))

        var outOfRange = validSchedule()
        outOfRange.layers[1] = KVLayerPrecision(layer: 2, keyBits: 4, valueBits: 2)
        XCTAssertThrowsError(try outOfRange.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc"))
    }

    func testModelHashUnsupportedBitsAndAverageMismatchFailClosed() {
        XCTAssertThrowsError(try validSchedule().validated(
            expectedLayerCount: 2, expectedModelConfigHash: "different-model"))

        var unsupported = validSchedule()
        unsupported.layers[1] = KVLayerPrecision(layer: 1, keyBits: 3, valueBits: 2)
        XCTAssertThrowsError(try unsupported.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc"))

        var wrongAverage = validSchedule()
        wrongAverage.nominalAverageBits = 4.0
        XCTAssertThrowsError(try wrongAverage.validated(
            expectedLayerCount: 2, expectedModelConfigHash: "model-config-abc"))
    }

    func testCalibrationAndEvaluationCorpusMustBeDistinct() {
        XCTAssertNoThrow(try validSchedule().validateEvaluationCorpus(
            id: "measurement-corpus-v2", hash: "different-hash"))
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "kvtuner-calibration-v1", hash: "calibration-hash-123"))
        XCTAssertThrowsError(try validSchedule().validateEvaluationCorpus(
            id: "renamed-corpus", hash: "calibration-hash-123"))
    }

    func testNominalAverageComputationDoesNotTrapOnUntrustedIntegers() {
        let schedule = KVTunerSchedule(
            schemaVersion: 1,
            modelConfigHash: "model-config-abc",
            calibrationCorpusID: "calibration",
            calibrationCorpusHash: "calibration-hash",
            seed: 7,
            objective: "test",
            nominalAverageBits: 0,
            layers: [KVLayerPrecision(layer: 0, keyBits: Int.max, valueBits: Int.max)])

        XCTAssertEqual(schedule.computedNominalAverageBits, Double(Int.max), accuracy: 0)
    }
}
