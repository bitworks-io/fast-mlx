import Foundation
import HarnessCore
import MLX
import XCTest

@testable import SpikeCore

final class KVTunerKVCacheSelectionTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v2"
    private let configHash = "0123456789abcdef"
    private let checkpointHash = "fedcba9876543210"

    private func selection(objective: String = "minimize-attention-error") throws
        -> KVTunerRuntimeSelection
    {
        let average = String(Double(14) / 3)
        let schedule = KVTunerSchedule(
            schemaVersion: 2,
            matrixID: matrixID,
            cellID: "kvtuner-g128-b\(average)",
            modelConfigHash: configHash,
            checkpointManifestHash: checkpointHash,
            groupSize: 128,
            calibrationCorpusID: "kvtuner-calibration-v1",
            calibrationCorpusHash: "1111111111111111",
            calibrationEntryHashes: [
                "2222222222222222",
                "3333333333333333",
            ],
            seed: 7,
            objective: objective,
            nominalAverageBits: Double(14) / 3,
            sourceSensitivityArtifactSHA256: String(repeating: "a", count: 64),
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
                KVLayerPrecision(layer: 2, keyBits: 8, valueBits: 2),
            ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try KVTunerRuntimeSelection.load(
            artifactData: encoder.encode(schedule),
            expectedLayerCount: 3,
            expectedMatrixID: matrixID,
            expectedCellID: schedule.cellID,
            expectedModelConfigHash: configHash,
            expectedCheckpointManifestHash: checkpointHash,
            evaluationCorpora: [
                KVTunerEvaluationCorpusIdentity(
                    id: "measurement-corpus-v2",
                    aggregateDigest: "4444444444444444",
                    canonicalEntryDigests: ["5555555555555555"]),
            ])
    }

    private func randomKV(
        seed: Int, dtype: DType = .float16
    ) -> (MLXArray, MLXArray) {
        let values = (0 ..< 128).map { index in
            Float((index + seed) % 29 - 14) / 8
        }
        let array = MLXArray(values).reshaped([1, 1, 1, 128])
            .asType(dtype)
        return (array, array)
    }

    func testSelectionBuildsLayerSpecificAffineCachesInCanonicalOrder() throws {
        let selection = try selection()
        let kind = KVCacheKind.kvtuner(selection)
        let caches = try kind.makeCaches(layerCount: 3, capacity: 7)
        let affine = caches.compactMap { $0 as? AffineKVCache }

        XCTAssertEqual(affine.count, 3)
        XCTAssertEqual(
            affine.map(\.configuration),
            [
                try AffineKVCacheConfiguration(
                    keyBits: 8, valueBits: 4,
                    keyGroupSize: 128, valueGroupSize: 128),
                try AffineKVCacheConfiguration(
                    keyBits: 4, valueBits: 2,
                    keyGroupSize: 128, valueGroupSize: 128),
                try AffineKVCacheConfiguration(
                    keyBits: 8, valueBits: 2,
                    keyGroupSize: 128, valueGroupSize: 128),
            ])
        XCTAssertEqual(
            kind.executionMode(requestingCompilation: true), .compiled)
        XCTAssertFalse(kind.supportsSpecDecode)
    }

    func testRuntimeFactoryRejectsLayerCountMismatchWithoutFallback() throws {
        let kind = KVCacheKind.kvtuner(try selection())

        XCTAssertThrowsError(
            try kind.makeCaches(layerCount: 2, capacity: 7)
        ) { error in
            XCTAssertEqual(
                error as? KVCacheKindError,
                .layerCountMismatch(expected: 3, actual: 2))
        }
    }

    func testExactArtifactDigestParticipatesInDecoderKeyIdentity() throws {
        let first = try selection(objective: "minimize-attention-error")
        let second = try selection(objective: "minimize-task-loss")

        XCTAssertNotEqual(first.artifactSHA256, second.artifactSHA256)
        XCTAssertNotEqual(
            KVCacheKind.kvtuner(first), KVCacheKind.kvtuner(second))
    }

    func testMixedTelemetryAuthenticatesPolicyAndAggregatesActualArrays() throws {
        let selection = try selection()
        let caches = try KVCacheKind.kvtuner(selection).makeCaches(
            layerCount: 3, capacity: 5)
        let affine = try XCTUnwrap(caches as? [AffineKVCache])
        for (index, cache) in affine.enumerated() {
            let (keys, values) = randomKV(seed: index)
            _ = cache.update(keys: keys, values: values)
        }

        let telemetry = try KVTunerKVCacheTelemetry.capture(
            selection: selection, caches: affine)
        let snapshots = try affine.map { try XCTUnwrap($0.storageSnapshot()) }

        XCTAssertEqual(telemetry.artifactSHA256, selection.artifactSHA256)
        XCTAssertEqual(telemetry.cellID, selection.cellID)
        XCTAssertEqual(telemetry.cachedTokens, 1)
        XCTAssertEqual(telemetry.layerCount, 3)
        XCTAssertEqual(
            telemetry.payloadBytes,
            snapshots.reduce(0) { $0 + $1.payloadBytes })
        XCTAssertEqual(
            telemetry.metadataBytes,
            snapshots.reduce(0) { $0 + $1.metadataBytes })
        XCTAssertEqual(
            telemetry.controlBytes,
            snapshots.reduce(0) { $0 + $1.controlBytes })
        XCTAssertEqual(
            telemetry.materializationWorkspaceBytes,
            snapshots.map(\.materializationWorkspaceBytes).max())

        let predicted = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: selection.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: selection.groupSize,
            geometry: KVStorageGeometry(
                layerCount: telemetry.layerCount,
                kvHeadCount: telemetry.kvHeadCount,
                headDimension: telemetry.headDimension),
            capacityTokens: telemetry.capacityTokens,
            sequences: telemetry.sequences,
            metadataScalarBytes: telemetry.metadataScalarBytes,
            maximumLayerWorkspaceBytes:
                telemetry.materializationWorkspaceBytes)
        XCTAssertEqual(predicted.payloadBytes, telemetry.payloadBytes)
        XCTAssertEqual(predicted.metadataBytes, telemetry.metadataBytes)
        XCTAssertEqual(predicted.controlBytes, telemetry.controlBytes)
        XCTAssertEqual(
            predicted.totalPersistentBytes, telemetry.totalPersistentBytes)
        XCTAssertEqual(predicted.totalBytes, telemetry.totalBytes)
    }

    func testFloat32MetadataWidthReconcilesWithoutUnderstatement() throws {
        let selection = try selection()
        let caches = try KVCacheKind.kvtuner(selection).makeCaches(
            layerCount: 3, capacity: 2)
        let affine = try XCTUnwrap(caches as? [AffineKVCache])
        for (index, cache) in affine.enumerated() {
            let (keys, values) = randomKV(seed: index, dtype: .float32)
            _ = cache.update(keys: keys, values: values)
        }

        let telemetry = try KVTunerKVCacheTelemetry.capture(
            selection: selection, caches: affine)
        XCTAssertEqual(telemetry.metadataScalarBytes, 4)

        let predicted = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: selection.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: selection.groupSize,
            geometry: KVStorageGeometry(
                layerCount: telemetry.layerCount,
                kvHeadCount: telemetry.kvHeadCount,
                headDimension: telemetry.headDimension),
            capacityTokens: telemetry.capacityTokens,
            sequences: telemetry.sequences,
            metadataScalarBytes: telemetry.metadataScalarBytes,
            maximumLayerWorkspaceBytes:
                telemetry.materializationWorkspaceBytes)
        XCTAssertEqual(predicted.metadataBytes, telemetry.metadataBytes)
        XCTAssertEqual(predicted.totalBytes, telemetry.totalBytes)
    }

    func testMixedTelemetryRejectsOneSubstitutedLayerConfiguration() throws {
        let selection = try selection()
        let caches: [AffineKVCache] = [
            AffineKVCache(
                capacity: 5,
                configuration: try AffineKVCacheConfiguration(
                    keyBits: 8, valueBits: 4,
                    keyGroupSize: 128, valueGroupSize: 128)),
            AffineKVCache(
                capacity: 5,
                configuration: AffineKVTier.k8v2G128.configuration),
            AffineKVCache(
                capacity: 5,
                configuration: AffineKVTier.k8v2G128.configuration),
        ]
        for (index, cache) in caches.enumerated() {
            let (keys, values) = randomKV(seed: index)
            _ = cache.update(keys: keys, values: values)
        }

        XCTAssertThrowsError(try KVTunerKVCacheTelemetry.capture(
            selection: selection, caches: caches)) { error in
                XCTAssertEqual(
                    error as? KVTunerKVCacheTelemetryError,
                    .configurationMismatch(layer: 1))
            }
    }
}
