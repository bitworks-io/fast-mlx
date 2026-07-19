import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import HarnessCore
@testable import SpikeCore

private final class TinyKVTunerCandidateModel:
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
            preconditionFailure("tiny KVTuner model requires three caches")
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

final class KVTunerKVCacheSelectionTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v2"
    private let configHash = "0123456789abcdef"
    private let checkpointHash = "fedcba9876543210"

    private func selection(
        sourceSearchArtifactSHA256: String = String(repeating: "b", count: 64)
    ) throws
        -> KVTunerRuntimeSelection
    {
        let average = String(Double(14) / 3)
        let schedule = KVTunerSchedule(
            schemaVersion: 3,
            matrixID: matrixID,
            cellID: "kvtuner-g128-b\(average)",
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
            objective: "maximize-gsm8k-accuracy-at-b\(average)",
            nominalAverageBits: Double(14) / 3,
            sourceSensitivityArtifactSHA256: String(repeating: "a", count: 64),
            sourceSearchArtifactSHA256: sourceSearchArtifactSHA256,
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
                KVLayerPrecision(layer: 1, keyBits: 4, valueBits: 2),
                KVLayerPrecision(layer: 2, keyBits: 8, valueBits: 2),
            ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try KVTunerRuntimeSelection.loadForTesting(
            artifactData: encoder.encode(schedule),
            expectedLayerCount: 3,
            expectedMatrixID: matrixID,
            expectedCellID: schedule.cellID,
            expectedModelConfigHash: configHash,
            expectedCheckpointManifestHash: checkpointHash,
            evaluationCorpora: [
                try KVTunerEvaluationCorpusIdentity(
                    id: "measurement-corpus-v2",
                    aggregateDigest: "4444444444444444",
                    canonicalEntryDigests: ["5555555555555555"],
                    canonicalSourceItemDigests: [
                        sha256Hex(Data("spike-evaluation-source".utf8))
                    ]),
            ])
    }

    private func candidatePolicy() throws -> KVTunerCandidateRuntimePolicy {
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
            matrixID: matrixID,
            modelConfigHash: configHash,
            modelConfigSHA256: String(repeating: "b", count: 64),
            checkpointManifestHash: checkpointHash,
            tokenizerSHA256: String(repeating: "c", count: 64),
            groupSize: 128)
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

    func testSelectionFactoryThreadsExplicitSplitModeToEveryScheduledLayer() throws {
        let kind = KVCacheKind.kvtuner(try selection())
        let caches = try kind.makeCaches(
            layerCount: 3,
            capacity: 7,
            affineAttentionMode: .splitQuantizedMM)
        let affine = try XCTUnwrap(caches as? [AffineKVCache])

        XCTAssertEqual(
            affine.map(\.attentionMode),
            Array(repeating: .splitQuantizedMM, count: 3))
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

    func testCandidatePolicyBuildsTheExactHeterogeneousCachesButHasNoTierParserRoute() throws {
        let policy = try candidatePolicy()
        let kind = KVCacheKind.kvtunerCandidate(policy)
        let caches = try kind.makeCaches(layerCount: 3, capacity: 7)
        let affine = caches.compactMap { $0 as? AffineKVCache }

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
        XCTAssertNil(KVCacheKind(kvQuant: "kvtuner-candidate"))
        XCTAssertNil(KVCacheKind(kvQuant: "kvtuner-g128-b4.666"))
        XCTAssertEqual(
            kind.executionMode(requestingCompilation: true), .compiled)
        XCTAssertFalse(kind.supportsSpecDecode)
    }

    func testCandidateTelemetryAuthenticatesPolicyAndActualArrays() throws {
        let policy = try candidatePolicy()
        let caches = try KVCacheKind.kvtunerCandidate(policy).makeCaches(
            layerCount: 3, capacity: 5)
        let affine = try XCTUnwrap(caches as? [AffineKVCache])
        for (index, cache) in affine.enumerated() {
            let (keys, values) = randomKV(seed: index)
            _ = cache.update(keys: keys, values: values)
        }

        let telemetry = try KVTunerCandidateKVCacheTelemetry.capture(
            policy: policy, caches: affine)
        XCTAssertEqual(
            telemetry.runtimePolicySHA256,
            policy.runtimePolicySHA256)
        XCTAssertEqual(telemetry.candidateSHA256, policy.candidateSHA256)
        XCTAssertEqual(
            telemetry.candidateListSHA256,
            policy.candidateListSHA256)
        XCTAssertEqual(telemetry.cachedTokens, 1)
        XCTAssertEqual(telemetry.layerCount, 3)
        let receipt = telemetry.evidenceReceipt
        XCTAssertEqual(
            receipt.runtimePolicySHA256,
            policy.runtimePolicySHA256)
        XCTAssertEqual(receipt.cachedTokens, telemetry.cachedTokens)
        XCTAssertEqual(receipt.capacityTokens, telemetry.capacityTokens)
        XCTAssertEqual(receipt.layers, policy.layers)
        XCTAssertEqual(receipt.geometry.layerCount, telemetry.layerCount)
        XCTAssertEqual(receipt.geometry.kvHeadCount, telemetry.kvHeadCount)
        XCTAssertEqual(
            receipt.geometry.headDimension,
            telemetry.headDimension)
        XCTAssertEqual(receipt.actualTotalBytes, telemetry.totalBytes)

        let predicted = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: policy.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: policy.groupSize,
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
        XCTAssertEqual(predicted.totalBytes, telemetry.totalBytes)
    }

    func testCandidateDecoderReceiptIncludesTheSubmittedLookaheadToken() throws {
        let policy = try candidatePolicy()
        var decoder = CompiledMLXDecoder(
            model: TinyKVTunerCandidateModel(),
            kvCache: .kvtunerCandidate(policy))

        let first = decoder.prefill([1, 2])
        XCTAssertEqual(first, 2)
        var telemetry = try XCTUnwrap(
            decoder.kvtunerCandidateKVTelemetry())
        XCTAssertEqual(telemetry.cachedTokens, 3)
        XCTAssertEqual(telemetry.capacityTokens, 512)

        let second = decoder.step(last: first)
        XCTAssertEqual(second, 2)
        telemetry = try XCTUnwrap(
            decoder.kvtunerCandidateKVTelemetry())
        XCTAssertEqual(telemetry.cachedTokens, 4)

        decoder.reset()
        telemetry = try XCTUnwrap(
            decoder.kvtunerCandidateKVTelemetry())
        XCTAssertEqual(telemetry.cachedTokens, 0)
        XCTAssertEqual(telemetry.capacityTokens, 512)
    }

    func testExactArtifactDigestParticipatesInDecoderKeyIdentity() throws {
        let first = try selection()
        let second = try selection(
            sourceSearchArtifactSHA256: String(repeating: "c", count: 64))

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
        XCTAssertEqual(telemetry.attentionWorkspaceBytes, 0)
        XCTAssertEqual(
            telemetry.workspaceBytes,
            telemetry.materializationWorkspaceBytes)

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
