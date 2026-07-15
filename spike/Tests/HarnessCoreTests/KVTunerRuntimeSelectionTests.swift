import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerRuntimeSelectionTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v1"
    private let cellID = "kvtuner-g128-b4.5"
    private let configHash = "0123456789abcdef"
    private let checkpointHash = "fedcba9876543210"

    private func validSchedule() -> KVTunerSchedule {
        KVTunerSchedule(
            schemaVersion: 3,
            matrixID: matrixID,
            cellID: cellID,
            modelConfigHash: configHash,
            checkpointManifestHash: checkpointHash,
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

    private func artifactData(
        _ schedule: KVTunerSchedule? = nil
    ) throws -> Data {
        try JSONEncoder().encode(schedule ?? validSchedule())
    }

    private func sourceRows(_ label: String) -> [String] {
        [sha256Hex(Data("test-source-\(label)".utf8))]
    }

    private func evaluationCorpora() throws
        -> [KVTunerEvaluationCorpusIdentity]
    {
        [
            try KVTunerEvaluationCorpusIdentity(
                id: "measurement-short-v2",
                aggregateDigest: "4444444444444444",
                canonicalEntryDigests: [
                    "5555555555555555",
                    "6666666666666666",
                ],
                canonicalSourceItemDigests: sourceRows("short")),
            try KVTunerEvaluationCorpusIdentity(
                id: "measurement-long-v2",
                aggregateDigest: "7777777777777777",
                canonicalEntryDigests: [
                    "8888888888888888",
                    "9999999999999999",
                ],
                canonicalSourceItemDigests: sourceRows("long")),
        ]
    }

    private func load(
        _ data: Data? = nil,
        expectedLayerCount: Int = 2,
        expectedMatrixID: String? = nil,
        expectedCellID: String? = nil,
        expectedModelConfigHash: String? = nil,
        expectedCheckpointManifestHash: String? = nil,
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]? = nil
    ) throws -> KVTunerRuntimeSelection {
        let artifact: Data
        if let data {
            artifact = data
        } else {
            artifact = try artifactData()
        }
        return try KVTunerRuntimeSelection.loadForTesting(
            artifactData: artifact,
            expectedLayerCount: expectedLayerCount,
            expectedMatrixID: expectedMatrixID ?? matrixID,
            expectedCellID: expectedCellID ?? cellID,
            expectedModelConfigHash: expectedModelConfigHash ?? configHash,
            expectedCheckpointManifestHash:
                expectedCheckpointManifestHash ?? checkpointHash,
            evaluationCorpora:
                evaluationCorpora ?? (try self.evaluationCorpora()))
    }

    private func assertSelectionError(
        _ expected: KVTunerRuntimeSelectionError,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ operation: () throws -> Void
    ) {
        XCTAssertThrowsError(
            try operation(),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? KVTunerRuntimeSelectionError,
                expected,
                file: file,
                line: line)
        }
    }

    func testLoadsExactBytesValidatesMultipleCorporaAndFreezesRuntimePolicy() throws {
        let data = try artifactData()
        let corpora = try evaluationCorpora()
        let selection = try load(data, evaluationCorpora: corpora)

        XCTAssertEqual(selection.artifactSHA256, sha256Hex(data))
        XCTAssertEqual(
            selection.qualificationBundleSHA256,
            sha256Hex(data))
        XCTAssertEqual(selection.matrixID, matrixID)
        XCTAssertEqual(selection.cellID, cellID)
        XCTAssertEqual(selection.modelConfigHash, configHash)
        XCTAssertEqual(selection.checkpointManifestHash, checkpointHash)
        XCTAssertEqual(selection.groupSize, 128)
        XCTAssertEqual(selection.schemaVersion, 3)
        XCTAssertEqual(selection.seed, 1234)
        XCTAssertEqual(
            selection.objective,
            "maximize-gsm8k-accuracy-at-b4.5")
        XCTAssertEqual(selection.nominalAverageBits, 4.5, accuracy: 0)
        XCTAssertEqual(
            selection.sourceSensitivityArtifactSHA256,
            String(repeating: "a", count: 64))
        XCTAssertEqual(
            selection.sourceSearchArtifactSHA256,
            String(repeating: "b", count: 64))
        XCTAssertEqual(selection.evaluationCorpora, corpora)
        XCTAssertEqual(selection.layers, [
            KVTunerRuntimeLayerPolicy(layer: 0, keyBits: 8, valueBits: 4),
            KVTunerRuntimeLayerPolicy(layer: 1, keyBits: 4, valueBits: 2),
        ])

        requireHashable(selection)
        requireSendable(selection)
        requireHashable(selection.layers[0])
        requireSendable(selection.layers[0])
        XCTAssertEqual(Set([selection, selection]).count, 1)
    }

    func testSameCellWithSemanticallyIdenticalAlteredBytesHasDifferentIdentity() throws {
        let data = try artifactData()
        var alteredData = data
        alteredData.append(0x0a)

        let original = try load(data)
        let altered = try load(alteredData)

        XCTAssertEqual(original.matrixID, altered.matrixID)
        XCTAssertEqual(original.cellID, altered.cellID)
        XCTAssertEqual(original.layers, altered.layers)
        XCTAssertNotEqual(original.artifactSHA256, altered.artifactSHA256)
        XCTAssertNotEqual(original, altered)
    }

    func testRuntimeLayersDoNotChangeWhenSourceScheduleIsMutatedAfterLoad() throws {
        var schedule = validSchedule()
        let selection = try load(try artifactData(schedule))
        let frozenLayers = selection.layers

        schedule.layers[0] = KVLayerPrecision(
            layer: 0, keyBits: 4, valueBits: 2)
        schedule.layers.removeLast()
        schedule.groupSize = 64

        XCTAssertEqual(selection.layers, frozenLayers)
        XCTAssertEqual(selection.groupSize, 128)
        XCTAssertEqual(selection.layers[0].keyBits, 8)
        XCTAssertEqual(selection.layers[1].valueBits, 2)
    }

    func testEveryExpectedRuntimeIdentityAndLayerCountMustMatch() {
        assertSelectionError(.invalidSchedule(.invalidLayerCount)) {
            _ = try load(expectedLayerCount: 3)
        }
        assertSelectionError(.invalidSchedule(.matrixIDMismatch)) {
            _ = try load(expectedMatrixID: "another-matrix")
        }
        assertSelectionError(.invalidSchedule(.cellIDMismatch)) {
            _ = try load(expectedCellID: "kvtuner-g64-b4.5")
        }
        assertSelectionError(.invalidSchedule(.modelConfigHashMismatch)) {
            _ = try load(
                expectedModelConfigHash: "aaaaaaaaaaaaaaaa")
        }
        assertSelectionError(
            .invalidSchedule(.checkpointManifestHashMismatch)
        ) {
            _ = try load(
                expectedCheckpointManifestHash: "bbbbbbbbbbbbbbbb")
        }
    }

    func testMalformedOrStructurallyInvalidJSONFailsClosed() throws {
        assertSelectionError(.malformedArtifact) {
            _ = try load(Data("{".utf8))
        }

        let encoded = try artifactData()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "layers")
        let missingLayers = try JSONSerialization.data(withJSONObject: object)
        assertSelectionError(.malformedArtifact) {
            _ = try load(missingLayers)
        }
    }

    func testDecodableButInvalidScheduleFailsClosed() throws {
        var schedule = validSchedule()
        schedule.schemaVersion = 1

        assertSelectionError(.invalidSchedule(.unsupportedSchema(1))) {
            _ = try load(try artifactData(schedule))
        }
    }

    func testAtLeastOneEvaluationCorpusIsRequired() {
        assertSelectionError(.missingEvaluationCorpus) {
            _ = try load(evaluationCorpora: [])
        }
    }

    func testEvaluationIdentityAggregateAndEntryLeakageFailClosedAtTheirIndex() throws {
        let leakingID = try KVTunerEvaluationCorpusIdentity(
            id: "kvtuner-calibration-v1",
            aggregateDigest: "4444444444444444",
            canonicalEntryDigests: ["5555555555555555"],
            canonicalSourceItemDigests: sourceRows("leaking-id"))
        assertSelectionError(
            .invalidEvaluationCorpus(
                index: 0,
                reason: .evaluationCorpusLeaksCalibration)
        ) {
            _ = try load(evaluationCorpora: [leakingID])
        }

        let leakingAggregate = try KVTunerEvaluationCorpusIdentity(
            id: "measurement-short-v2",
            aggregateDigest: "1111111111111111",
            canonicalEntryDigests: ["5555555555555555"],
            canonicalSourceItemDigests: sourceRows("leaking-aggregate"))
        assertSelectionError(
            .invalidEvaluationCorpus(
                index: 0,
                reason: .evaluationCorpusLeaksCalibration)
        ) {
            _ = try load(evaluationCorpora: [leakingAggregate])
        }

        let leakingEntry = try KVTunerEvaluationCorpusIdentity(
            id: "measurement-short-v2",
            aggregateDigest: "4444444444444444",
            canonicalEntryDigests: [
                "2222222222222222",
                "5555555555555555",
            ],
            canonicalSourceItemDigests: sourceRows("leaking-entry"))
        assertSelectionError(
            .invalidEvaluationCorpus(
                index: 1,
                reason: .evaluationCorpusLeaksCalibration)
        ) {
            _ = try load(evaluationCorpora: [
                try evaluationCorpora()[0],
                leakingEntry,
            ])
        }
    }

    func testMalformedEvaluationIdentityFailsClosedThroughExistingGate() throws {
        let malformed = try KVTunerEvaluationCorpusIdentity(
            id: "measurement corpus",
            aggregateDigest: "not-a-digest",
            canonicalEntryDigests: [],
            canonicalSourceItemDigests: sourceRows("malformed"))

        assertSelectionError(
            .invalidEvaluationCorpus(
                index: 0,
                reason: .invalidIdentifier("measurement corpus"))
        ) {
            _ = try load(evaluationCorpora: [malformed])
        }
    }
}

private func requireHashable<T: Hashable>(_: T) {}
private func requireSendable<T: Sendable>(_: T) {}
