import Foundation
import XCTest
@testable import HarnessCore

final class KVFrontierEvidenceTests: XCTestCase {
    private func identity(_ suffix: String = "same") -> KVModelEvidenceIdentity {
        KVModelEvidenceIdentity(
            configHash: "config-\(suffix)",
            checkpointManifestHash: "checkpoint-\(suffix)")
    }

    private func geometry(
        tier: String = "affine-k4v2-g128",
        capacityTokens: Int = 24_192
    ) -> KVFormatGeometryEvidence {
        KVFormatGeometryEvidence(
            kind: .affine, tier: tier, keyBits: 4, valueBits: 2, groupSize: 128,
            sinkTokens: 0, layerCount: 64, kvHeadCount: 8, headDimension: 128,
            capacityTokens: capacityTokens, sequences: 1, metadataScalarBytes: 2,
            recordAlignment: 1)
    }

    private func breakdown(
        capacityTokens: Int = 24_192,
        workspaceBytes: Int = 0,
        total: Int? = nil
    ) -> KVStorageBreakdownEvidence {
        let tokenHeads = capacityTokens * 64 * 8
        let payloadBytes = tokenHeads * 96
        let metadataBytes = tokenHeads * 8
        return KVStorageBreakdownEvidence(
            payloadBytes: payloadBytes, metadataBytes: metadataBytes,
            alignmentPaddingBytes: 0, fp16SinkBytes: 0, fp16TailBytes: 0,
            workspaceBytes: workspaceBytes,
            totalBytes: total ?? payloadBytes + metadataBytes + workspaceBytes)
    }

    private func frontier(
        schemaVersion: Int? = nil,
        sameWeights: Bool = true,
        baseline: KVComparisonBaseline = .sameWeightsFP16KV,
        candidate: KVModelEvidenceIdentity? = nil,
        reference: KVModelEvidenceIdentity? = nil,
        matrixID: String = "kvarn-qwen3-32b-v1",
        cellID: String = "affine-k4v2-g128",
        format: KVFormatGeometryEvidence? = nil,
        storage: KVStorageEvidence? = nil,
        controlBytes: Int? = 256,
        executionMode: String? = nil,
        codecIterations: Int? = nil,
        memoryGate: KVarNMemoryGateEvidence? = nil,
        kvtunerSchedule: KVTunerScheduleBinding? = nil,
        compressedKVAttention:
            CompressedKVAttentionRuntimeBinding? = nil,
        materializationWorkspaceBytes: Int? = nil,
        attentionWorkspaceBytes: Int? = nil,
        autoBindKVTuner: Bool = true
    ) -> KVFrontierEvidence {
        let same = identity()
        let bytes = breakdown()
        let resolvedStorage = storage
            ?? KVStorageEvidence(predicted: bytes, actual: bytes)
        let resolvedSchemaVersion = schemaVersion
            ?? (kvtunerSchedule == nil ? 1 : 2)
        func resolvedModel(
            _ supplied: KVModelEvidenceIdentity?
        ) -> KVModelEvidenceIdentity {
            guard let schedule = kvtunerSchedule else {
                return supplied ?? same
            }
            if let supplied,
                supplied.configHash != schedule.modelConfigHash
                    || supplied.checkpointManifestHash
                        != schedule.checkpointManifestHash
            {
                return supplied
            }
            return KVModelEvidenceIdentity(
                configHash: schedule.modelConfigHash,
                checkpointManifestHash: schedule.checkpointManifestHash,
                checkpointContentSHA256:
                    supplied?.checkpointContentSHA256
                        ?? schedule.checkpointContentSHA256)
        }
        let automaticBinding: CompressedKVAttentionRuntimeBinding?
        if autoBindKVTuner, kvtunerSchedule != nil {
            automaticBinding = try! CompressedKVAttentionRuntimeBinding(
                request: .materialize,
                observedOperation: .materializedKV,
                admission: compressedAttentionAdmission())
        } else {
            automaticBinding = nil
        }
        return KVFrontierEvidence(
            schemaVersion: resolvedSchemaVersion,
            matrixID: matrixID, cellID: cellID,
            sameWeights: sameWeights, comparisonBaseline: baseline,
            referenceKVQuantTier: "fp16",
            candidateModel: resolvedModel(candidate),
            referenceModel: resolvedModel(reference),
            candidateFormat: format ?? geometry(),
            storage: resolvedStorage,
            actualControlBytes: controlBytes,
            candidateExecutionMode: executionMode,
            candidateCodecIterations: codecIterations,
            candidateMemoryGate: memoryGate,
            candidateKVTunerSchedule: kvtunerSchedule,
            candidateCompressedKVAttention:
                compressedKVAttention ?? automaticBinding,
            candidateMaterializationWorkspaceBytes:
                materializationWorkspaceBytes
                    ?? (automaticBinding == nil
                        ? nil : resolvedStorage.actual.workspaceBytes),
            candidateAttentionWorkspaceBytes:
                attentionWorkspaceBytes
                    ?? (automaticBinding == nil ? nil : 0))
    }

    private func compressedAttentionAdmission() throws
        -> CompressedKVAttentionRuntimeAdmission
    {
        let config = Data(
            #"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":8192,"num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false}"#.utf8)
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: config,
                checkpointManifestHash: "0123456789abcdef",
                checkpointContentSHA256: String(repeating: "d", count: 64),
                tokenizerSHA256: String(repeating: "c", count: 64)))
    }

    private func kvarnMemoryGate(
        tier: String = "kvarn-k4v2-g128",
        iterations: Int = 8,
        harnessSHA: String = String(repeating: "a", count: 40),
        maximumCapacityTokens: Int = 24_192
    ) -> KVarNMemoryGateEvidence {
        KVarNMemoryGateEvidence(
            schemaVersion: 1,
            artifactSHA256: String(repeating: "b", count: 64),
            harnessGitSHA: harnessSHA,
            mlxSwiftVersion: "0.31.6",
            hardwareChip: "Apple M3 Ultra",
            hardwareOS: "macOS 15.5",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            runtimeTier: tier, codecIterations: iterations,
            cacheBoundaryMaximumCapacityTokens: maximumCapacityTokens,
            encodeSampleCount: 3, decodeSampleCount: 3,
            cacheBoundarySampleCount: 9,
            encodeTransientPeakBytes: 28_147_712,
            decodeTransientPeakBytes: 2_310_144,
            cacheBoundaryTransientPeakBytes: 48_037_888,
            maximumPeakActiveBytes: 1_000_000_000)
    }

    private func memoryProbeRow(
        phase: String, capacity: Int, run: Int,
        iterations: Int = 8, peak: Int = 100_000_000,
        transient: Int = 10_000_000,
        evaluatedArrayCount: Int? = nil,
        expectedEvaluatedArrayCount: Int? = nil
    ) -> KVarNMemoryProbeArtifactRow {
        let phaseArrayCount: Int
        let startArrayCount: Int
        let endArrayCount: Int
        let persistentLogicalBytes: Int
        let materializationLogicalBytes: Int
        let controlLogicalBytes: Int
        let startLogicalBytes: Int
        let endMinimumLogicalBytes: Int
        switch phase {
        case "encode":
            phaseArrayCount = 8
            startArrayCount = 2
            endArrayCount = 10
            persistentLogicalBytes = 16_384
            materializationLogicalBytes = 0
            controlLogicalBytes = 0
            startLogicalBytes = 8_192
            endMinimumLogicalBytes = 24_576
        case "decode":
            phaseArrayCount = 2
            startArrayCount = 8
            endArrayCount = 10
            persistentLogicalBytes = 32_768
            materializationLogicalBytes = 32_768
            controlLogicalBytes = 0
            startLogicalBytes = persistentLogicalBytes
            endMinimumLogicalBytes = 65_536
        case "cache-boundary":
            phaseArrayCount = 15
            startArrayCount = 15
            endArrayCount = 17
            let slots = (capacity - 128 + 127) / 128
            persistentLogicalBytes = slots * 8 * 13_824
                + 2 * 2 * 128 * 8 * 128 * 2
            materializationLogicalBytes =
                2 * 2 * 8 * capacity * 128
            controlLogicalBytes = 4
            startLogicalBytes = persistentLogicalBytes
                + controlLogicalBytes + 2 * 2 * 8 * 128
            endMinimumLogicalBytes = startLogicalBytes
                + materializationLogicalBytes
        default:
            phaseArrayCount = 0
            startArrayCount = 0
            endArrayCount = 0
            persistentLogicalBytes = 0
            materializationLogicalBytes = 0
            controlLogicalBytes = 0
            startLogicalBytes = 0
            endMinimumLogicalBytes = 0
        }
        let pageBytes = 4_096
        let baselineBytes = 128
        let allocatorBytes: (Int) -> Int = { logical in
            logical <= pageBytes
                ? logical
                : ((logical + pageBytes - 1) / pageBytes) * pageBytes
        }
        let startExpectedAllocatorBytes = allocatorBytes(startLogicalBytes)
        let startActiveBytes = startExpectedAllocatorBytes + baselineBytes
        let retainedActiveBytes = max(
            startActiveBytes, peak - transient)
        let postDetachExpectedAllocatorBytes = allocatorBytes(
            endMinimumLogicalBytes)
        let postDetachActiveBytes =
            postDetachExpectedAllocatorBytes + baselineBytes
        let emptyBaseline = KVarNMemoryProbeArtifactCounters(
            activeBytes: baselineBytes, cacheBytes: 0, peakActiveBytes: 0)
        let startReconciliation = KVarNMemoryProbeArtifactReconciliation(
            logicalBytes: startLogicalBytes,
            expectedAllocatorBytes: startExpectedAllocatorBytes,
            activeBytes: startActiveBytes,
            runtimeBaselineBytes: baselineBytes,
            arrayCount: startArrayCount,
            allocatorPageBytes: pageBytes,
            maximumActiveBytes: startExpectedAllocatorBytes + baselineBytes,
            activeAboveExpectedAllocatorBytes: baselineBytes)
        let endRetained = KVarNMemoryProbeArtifactRetainedAccounting(
            minimumLogicalBytes: endMinimumLogicalBytes,
            activeBytes: retainedActiveBytes,
            arrayCount: endArrayCount,
            activeAboveMinimumLogicalBytes:
                retainedActiveBytes - endMinimumLogicalBytes)
        let postDetachCounters = KVarNMemoryProbeArtifactCounters(
            activeBytes: postDetachActiveBytes,
            cacheBytes: 0, peakActiveBytes: 0)
        let postDetachReconciliation =
            KVarNMemoryProbeArtifactReconciliation(
                logicalBytes: endMinimumLogicalBytes,
                expectedAllocatorBytes: postDetachExpectedAllocatorBytes,
                activeBytes: postDetachActiveBytes,
                runtimeBaselineBytes: baselineBytes,
                arrayCount: endArrayCount,
                allocatorPageBytes: pageBytes,
                maximumActiveBytes:
                    postDetachExpectedAllocatorBytes + baselineBytes,
                activeAboveExpectedAllocatorBytes: baselineBytes)
        let startCounters = KVarNMemoryProbeArtifactCounters(
            activeBytes: startActiveBytes,
            cacheBytes: 0, peakActiveBytes: 0)
        let endCounters = KVarNMemoryProbeArtifactCounters(
            activeBytes: retainedActiveBytes,
            cacheBytes: 0, peakActiveBytes: peak)

        let structuralMemory: KVarNMemoryProbeArtifactStructuralMemory?
        if phase == "cache-boundary" {
            let materialized = 2 * 8 * capacity * 128
            let materializedAllocator = allocatorBytes(materialized)
            let reconstructed = 2 * 8 * 128 * 128
            let reconstructedAllocator = allocatorBytes(reconstructed)
            let tiles = (capacity - 128) / 128
            let minimumIncrement = materializedAllocator
                + tiles * reconstructedAllocator
            let minimumPeak = startActiveBytes + minimumIncrement
            let dualIncrement = 2 * minimumIncrement
            let dualPeak = startActiveBytes + dualIncrement
            structuralMemory = KVarNMemoryProbeArtifactStructuralMemory(
                completedTileCount: tiles,
                materializedOutputArrayBytes: materialized,
                materializedOutputArrayAllocatorBytes:
                    materializedAllocator,
                reconstructedTileArrayBytes: reconstructed,
                reconstructedTileArrayAllocatorBytes:
                    reconstructedAllocator,
                minimumConcatIncrementBytes: minimumIncrement,
                minimumStructuralPeakActiveBytes: minimumPeak,
                dualConcatReferenceIncrementBytes: dualIncrement,
                dualConcatReferencePeakActiveBytes: dualPeak,
                observedPeakActiveBytes: peak,
                observedPeakAboveMinimumStructuralBytes:
                    peak - minimumPeak,
                observedPeakDeltaFromDualConcatReferenceBytes:
                    peak - dualPeak)
        } else {
            structuralMemory = nil
        }
        return KVarNMemoryProbeArtifactRow(
            schemaVersion: 3,
            harnessSHA: String(repeating: "a", count: 40),
            mlxSwiftVersion: "0.31.6",
            hardwareChip: "Apple M3 Ultra",
            hardwareOS: "macOS 15.5",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            configuration: KVarNMemoryProbeArtifactConfiguration(
                phase: phase, heads: 8, headDimension: 128,
                groupSize: 128, iterations: iterations,
                capacity: capacity, cacheLimitBytes: 0, run: run),
            persistentLogicalBytes: persistentLogicalBytes,
            materializationLogicalBytes: materializationLogicalBytes,
            controlLogicalBytes: controlLogicalBytes,
            evaluatedArrayCount: evaluatedArrayCount ?? phaseArrayCount,
            expectedEvaluatedArrayCount:
                expectedEvaluatedArrayCount ?? phaseArrayCount,
            valuesFinite: true,
            emptyBaseline: emptyBaseline,
            startReconciliation: startReconciliation,
            endRetainedAccounting: endRetained,
            postDetachCounters: postDetachCounters,
            postDetachReconciliation: postDetachReconciliation,
            cacheBoundaryStructuralMemory: structuralMemory,
            highWater: KVarNMemoryProbeArtifactHighWater(
                start: startCounters, end: endCounters,
                observedPeakActiveBytes: peak,
                retainedActiveBytes: retainedActiveBytes,
                transientActiveAboveRetainedBytes: transient,
                incrementalPeakActiveBytes: peak - startActiveBytes),
            status: "PASS")
    }

    private func memoryProbeRows(iterations: Int = 8)
        -> [KVarNMemoryProbeArtifactRow]
    {
        var rows: [KVarNMemoryProbeArtifactRow] = []
        for run in 1 ... 3 {
            rows.append(memoryProbeRow(
                phase: "encode", capacity: 256, run: run,
                iterations: iterations, transient: 20_000_000 + run))
            rows.append(memoryProbeRow(
                phase: "decode", capacity: 256, run: run,
                iterations: iterations, transient: 2_000_000 + run))
            for capacity in [256, 4_096, 24_192] {
                rows.append(memoryProbeRow(
                    phase: "cache-boundary", capacity: capacity,
                    run: run, iterations: iterations,
                    peak: capacity * 100_000,
                    transient: capacity * 1_000 + run))
            }
        }
        return rows
    }

    private func kvarnFormatAndStorage(
        tier: String = "kvarn-k4v2-g128",
        capacityTokens: Int = 24_192,
        kvHeadCount: Int = 8,
        headDimension: Int = 128,
        layerCount: Int = 2
    ) throws -> (KVFormatGeometryEvidence, KVStorageEvidence) {
        let format = KVFormatGeometryEvidence(
            kind: .kvarn, tier: tier,
            keyBits: 4, valueBits: 2, groupSize: 128,
            sinkTokens: 128, layerCount: layerCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            capacityTokens: capacityTokens, sequences: 1,
            metadataScalarBytes: 2, recordAlignment: 8)
        let workspaceBytes = capacityTokens * kvHeadCount * headDimension * 4
        let allocation = try KVStorageFormat.kvarn(
            keyBits: 4, valueBits: 2, groupSize: 128,
            sinkTokens: 128, metadataScalarBytes: 2, alignment: 8
        ).allocation(
            geometry: KVStorageGeometry(
                layerCount: layerCount, kvHeadCount: kvHeadCount,
                headDimension: headDimension),
            capacityTokens: capacityTokens, sequences: 1,
            workspaceBytes: workspaceBytes)
        let actual = KVStorageBreakdownEvidence(
            payloadBytes: allocation.payloadBytes,
            metadataBytes: allocation.metadataBytes,
            alignmentPaddingBytes: allocation.alignmentPaddingBytes,
            fp16SinkBytes: allocation.fp16SinkBytes,
            fp16TailBytes: allocation.fp16TailBytes,
            workspaceBytes: allocation.workspaceBytes,
            totalBytes: allocation.totalBytes)
        return (format, try format.storageEvidence(actual: actual))
    }

    private func kvtunerEvaluation(
        id: String = "measurement-corpus-v2",
        aggregateDigest: String = "4444444444444444"
    ) throws -> KVTunerEvaluationCorpusIdentity {
        try KVTunerEvaluationCorpusIdentity(
            id: id,
            aggregateDigest: aggregateDigest,
            canonicalEntryDigests: [
                "5555555555555555",
                "6666666666666666",
            ],
            canonicalSourceItemDigests: [
                sha256Hex(Data("frontier-evaluation-source".utf8))
            ])
    }

    private func kvtunerBinding(
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]? = nil
    ) throws -> KVTunerScheduleBinding {
        let matrixID = "kvarn-qwen3-32b-v1"
        let cellID = "kvtuner-g128-b4.5"
        let admission = try compressedAttentionAdmission()
        let schedule = KVTunerSchedule(
            schemaVersion: 4,
            matrixID: matrixID,
            cellID: cellID,
            modelConfigHash: admission.modelConfigHash,
            modelConfigSHA256: admission.modelConfigSHA256,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
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
            layers: (0 ..< admission.layerCount).map {
                KVLayerPrecision(
                    layer: $0,
                    keyBits: $0.isMultiple(of: 2) ? 8 : 4,
                    valueBits: $0.isMultiple(of: 2) ? 4 : 2)
            })
        let selection = try KVTunerRuntimeSelection.loadForTesting(
            artifactData: JSONEncoder().encode(schedule),
            expectedLayerCount: admission.layerCount,
            expectedMatrixID: matrixID,
            expectedCellID: cellID,
            expectedModelConfigHash: admission.modelConfigHash,
            expectedModelConfigSHA256:
                admission.modelConfigSHA256,
            expectedCheckpointManifestHash:
                admission.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                admission.checkpointContentSHA256,
            evaluationCorpora:
                evaluationCorpora ?? [try kvtunerEvaluation()])
        return KVTunerScheduleBinding(selection: selection)
    }

    private func kvtunerFormatAndStorage(
        binding: KVTunerScheduleBinding,
        capacityTokens: Int = 24_192
    ) throws -> (
        format: KVFormatGeometryEvidence,
        storage: KVStorageEvidence,
        allocation: KVTunerStorageAllocation
    ) {
        let format = KVFormatGeometryEvidence(
            kind: .kvtuner, tier: binding.cellID,
            keyBits: 0, valueBits: 0, groupSize: binding.groupSize,
            sinkTokens: 0, layerCount: binding.layers.count,
            kvHeadCount: 8, headDimension: 128,
            capacityTokens: capacityTokens, sequences: 1,
            metadataScalarBytes: 4, recordAlignment: 1)
        let workspaceBytes = capacityTokens * 8 * 128 * 4
        let allocation = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: binding.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: binding.groupSize,
            geometry: KVStorageGeometry(
                layerCount: binding.layers.count,
                kvHeadCount: 8,
                headDimension: 128),
            capacityTokens: capacityTokens,
            sequences: 1,
            metadataScalarBytes: 4,
            maximumLayerWorkspaceBytes: workspaceBytes)
        let actual = KVStorageBreakdownEvidence(
            payloadBytes: allocation.payloadBytes,
            metadataBytes: allocation.metadataBytes,
            alignmentPaddingBytes: 0,
            fp16SinkBytes: 0,
            fp16TailBytes: 0,
            workspaceBytes: allocation.workspaceBytes,
            totalBytes: allocation.totalBytes - allocation.controlBytes)
        return (
            format,
            try format.storageEvidence(
                actual: actual, kvtunerSchedule: binding),
            allocation)
    }

    private func payload(
        kvQuantTier: String = "affine-k4v2-g128",
        frontier: KVFrontierEvidence? = nil
    ) -> KLPayload {
        KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: 0.04, klLongContextTailP95Nats: 0.3,
            klPooledMedianNats: 0.05, klPooledP95Nats: 0.4,
            pplCandidate: 10.2, pplReference: 10.0, pplDeltaPct: 2.0,
            totalPositions: 200, entryCount: 4,
            teacherForcedTop1AgreementCount: 160,
            teacherForcedTop1ScoredPositions: 200,
            teacherForcedTop1AgreementRate: 0.8,
            frontier: frontier ?? self.frontier(),
            shortEntryCount: 3, shortScoredPositions: 72,
            longContextEntryCount: 1, longContextScoredPositions: 128,
            shortEntryScoring: [
                KVEntryScoringEvidence(entryID: "short-0", scoredPositions: 24),
                KVEntryScoringEvidence(entryID: "short-1", scoredPositions: 24),
                KVEntryScoringEvidence(entryID: "short-2", scoredPositions: 24),
            ],
            longContextEntryScoring: [
                KVEntryScoringEvidence(entryID: "long-0", scoredPositions: 128),
            ],
            longContextMaxDocumentTokens: 24_151,
            longContextMaxScoredContextTokens: 24_150)
    }

    func testCompleteSameWeightsCellValidatesAndRoundTrips() throws {
        let validated = try payload().validatedForPromotion()
        let data = try JSONEncoder().encode(validated)
        let decoded = try JSONDecoder().decode(KLPayload.self, from: data)

        XCTAssertEqual(decoded, validated)
        XCTAssertEqual(decoded.frontier?.matrixID, "kvarn-qwen3-32b-v1")
        XCTAssertEqual(
            decoded.frontier?.storage?.actual.totalBytes,
            breakdown().totalBytes)
    }

    func testCompressedAttentionFrontierAuthenticatesObservedOperationAndIdentity() throws {
        let admission = try compressedAttentionAdmission()
        let model = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: admission.checkpointContentSHA256)
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admission)
        let splitBytes = breakdown(workspaceBytes: 4_096)
        let row = payload(frontier: frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            storage: KVStorageEvidence(
                predicted: splitBytes, actual: splitBytes),
            compressedKVAttention: binding,
            materializationWorkspaceBytes: 0,
            attentionWorkspaceBytes: 4_096))

        XCTAssertNoThrow(try row.validatedForPromotion())

        let wrongContentModel = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try payload(frontier: frontier(
            schemaVersion: 2,
            candidate: wrongContentModel,
            reference: wrongContentModel,
            storage: KVStorageEvidence(
                predicted: splitBytes, actual: splitBytes),
            compressedKVAttention: binding,
            materializationWorkspaceBytes: 0,
            attentionWorkspaceBytes: 4_096)).validatedForPromotion()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }

        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(row)) as? [String: Any])
        var frontierObject = try XCTUnwrap(
            object["frontier"] as? [String: Any])
        var bindingObject = try XCTUnwrap(
            frontierObject["candidateCompressedKVAttention"]
                as? [String: Any])
        bindingObject["observedOperation"] = "materialized-kv"
        frontierObject["candidateCompressedKVAttention"] = bindingObject
        object["frontier"] = frontierObject
        let forged = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONSerialization.data(withJSONObject: object))
        XCTAssertThrowsError(try forged.validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }

        frontierObject.removeValue(
            forKey: "candidateCompressedKVAttention")
        object["frontier"] = frontierObject
        let missingBinding = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONSerialization.data(withJSONObject: object))
        XCTAssertThrowsError(try missingBinding.validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }

        frontierObject = try XCTUnwrap(object["frontier"] as? [String: Any])
        frontierObject["candidateCompressedKVAttention"] = bindingObject
        frontierObject.removeValue(
            forKey: "candidateAttentionWorkspaceBytes")
        object["frontier"] = frontierObject
        let missingWorkspace = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONSerialization.data(withJSONObject: object))
        XCTAssertThrowsError(try missingWorkspace.validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }
    }

    func testHistoricalSchemaOneFrontierForbidsCompressedAttentionEvidence() throws {
        let admission = try compressedAttentionAdmission()
        let model = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: admission.checkpointContentSHA256)
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .materialize,
            observedOperation: .materializedKV,
            admission: admission)

        XCTAssertNoThrow(try payload().validatedForPromotion())
        XCTAssertThrowsError(try payload(frontier: frontier(
            candidate: model,
            reference: model,
            compressedKVAttention: binding,
            materializationWorkspaceBytes: 4_096,
            attentionWorkspaceBytes: 0)).validatedForPromotion()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }
    }

    func testCompressedAttentionSchemaTwoReconcilesWorkspaceComponents() throws {
        let admission = try compressedAttentionAdmission()
        let model = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256: admission.checkpointContentSHA256)
        let materializedBinding = try CompressedKVAttentionRuntimeBinding(
            request: .materialize,
            observedOperation: .materializedKV,
            admission: admission)
        let materializedBytes = breakdown(workspaceBytes: 4_096)

        XCTAssertNoThrow(try payload(frontier: frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            storage: KVStorageEvidence(
                predicted: materializedBytes, actual: materializedBytes),
            compressedKVAttention: materializedBinding,
            materializationWorkspaceBytes: 4_096,
            attentionWorkspaceBytes: 0)).validatedForPromotion())

        XCTAssertThrowsError(try payload(frontier: frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            storage: KVStorageEvidence(
                predicted: materializedBytes, actual: materializedBytes),
            compressedKVAttention: materializedBinding,
            materializationWorkspaceBytes: 0,
            attentionWorkspaceBytes: 4_096)).validatedForPromotion()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }

        let splitBinding = try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admission)
        XCTAssertThrowsError(try payload(frontier: frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            storage: KVStorageEvidence(
                predicted: materializedBytes, actual: materializedBytes),
            compressedKVAttention: splitBinding,
            materializationWorkspaceBytes: 1,
            attentionWorkspaceBytes: 4_095)).validatedForPromotion()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }

        let mismatchedStorage = breakdown(workspaceBytes: 4_097)
        XCTAssertThrowsError(try payload(frontier: frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            storage: KVStorageEvidence(
                predicted: mismatchedStorage, actual: mismatchedStorage),
            compressedKVAttention: splitBinding,
            materializationWorkspaceBytes: 0,
            attentionWorkspaceBytes: 4_096)).validatedForPromotion()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }
    }

    func testKVTunerSchemaTwoRequiresCompressedAttentionBinding() throws {
        let schedule = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: schedule)

        XCTAssertThrowsError(try payload(
            kvQuantTier: schedule.cellID,
            frontier: frontier(
                schemaVersion: 2,
                candidate: KVModelEvidenceIdentity(
                    configHash: schedule.modelConfigHash,
                    checkpointManifestHash: schedule.checkpointManifestHash),
                reference: KVModelEvidenceIdentity(
                    configHash: schedule.modelConfigHash,
                    checkpointManifestHash: schedule.checkpointManifestHash),
                cellID: schedule.cellID,
                format: measured.format,
                storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: schedule,
                materializationWorkspaceBytes:
                    measured.storage.actual.workspaceBytes,
                attentionWorkspaceBytes: 0,
                autoBindKVTuner: false)).validatedForPromotion()) {
                    XCTAssertEqual(
                        $0 as? KVFrontierEvidenceError,
                        .invalidKVTunerSchedule)
                }
    }

    func testHistoricalFrontierWithoutControlBytesStillDecodesButCannotPromote() throws {
        let encoded = try JSONEncoder().encode(payload())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var frontier = try XCTUnwrap(object["frontier"] as? [String: Any])
        frontier.removeValue(forKey: "actualControlBytes")
        object["frontier"] = frontier

        let historical = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(KLPayload.self, from: historical)

        XCTAssertNil(decoded.frontier?.actualControlBytes)
        XCTAssertNil(decoded.frontier?.candidateKVTunerSchedule)
        XCTAssertNoThrow(try decoded.validatedForRecord())
        XCTAssertThrowsError(try decoded.validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingControlStorage)
        }
    }

    func testFormatBuildsPredictionAroundMeasuredRuntimeArrays() throws {
        let actual = breakdown()
        let evidence = try geometry().storageEvidence(actual: actual)

        XCTAssertEqual(evidence.actual, actual)
        XCTAssertEqual(evidence.predicted, actual)
    }

    func testKVTunerFrontierRoundTripsAndReconcilesHeterogeneousStorage() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)
        let evidence = frontier(
            candidate: model,
            reference: model,
            matrixID: binding.matrixID,
            cellID: binding.cellID,
            format: measured.format,
            storage: measured.storage,
            controlBytes: measured.allocation.controlBytes,
            kvtunerSchedule: binding)
        let candidate = payload(
            kvQuantTier: binding.cellID,
            frontier: evidence)

        XCTAssertNoThrow(try candidate.validatedForPromotion())
        XCTAssertEqual(
            measured.storage.predicted.payloadBytes,
            measured.allocation.payloadBytes)
        XCTAssertEqual(
            measured.storage.predicted.metadataBytes,
            measured.allocation.metadataBytes)
        XCTAssertEqual(
            measured.storage.predicted.totalBytes,
            measured.allocation.totalBytes
                - measured.allocation.controlBytes)

        let decoded = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONEncoder().encode(candidate))
        XCTAssertEqual(decoded, candidate)
        XCTAssertEqual(
            decoded.frontier?.candidateKVTunerSchedule,
            binding)

        let record = ResultRecord(
            subcommand: "kl",
            provenance: provenance(
                configHash: binding.modelConfigHash,
                manifestHash: binding.checkpointManifestHash,
                corpusHash: try kvtunerEvaluation().aggregateDigest),
            payload: candidate)
        XCTAssertNoThrow(try record.validatedForPromotionEvidence())
    }

    func testKVTunerRequiresScheduleExclusivelyAndExactControlStorage() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)

        let missingSchedule = frontier(
            candidate: model,
            reference: model,
            matrixID: binding.matrixID,
            cellID: binding.cellID,
            format: measured.format,
            storage: measured.storage,
            controlBytes: measured.allocation.controlBytes)
        XCTAssertThrowsError(try payload(
            kvQuantTier: binding.cellID,
            frontier: missingSchedule).validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidKVTunerSchedule)
        }

        let unexpectedSchedule = frontier(
            kvtunerSchedule: binding)
        XCTAssertThrowsError(try payload(
            frontier: unexpectedSchedule).validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidKVTunerSchedule)
        }

        let wrongControl = frontier(
            candidate: model,
            reference: model,
            matrixID: binding.matrixID,
            cellID: binding.cellID,
            format: measured.format,
            storage: measured.storage,
            controlBytes: measured.allocation.controlBytes - 1,
            kvtunerSchedule: binding)
        XCTAssertThrowsError(try payload(
            kvQuantTier: binding.cellID,
            frontier: wrongControl).validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidControlStorage)
        }
    }

    func testKVTunerForbidsKVarNOnlyRuntimeEvidence() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)

        let invalidRows = [
            frontier(
                candidate: model,
                reference: model,
                matrixID: binding.matrixID,
                cellID: binding.cellID,
                format: measured.format,
                storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                executionMode: "uncompiled-correctness",
                codecIterations: 8,
                kvtunerSchedule: binding),
            frontier(
                candidate: model,
                reference: model,
                matrixID: binding.matrixID,
                cellID: binding.cellID,
                format: measured.format,
                storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                memoryGate: kvarnMemoryGate(),
                kvtunerSchedule: binding),
        ]
        for invalid in invalidRows {
            XCTAssertThrowsError(try payload(
                kvQuantTier: binding.cellID,
                frontier: invalid).validatedForRecord()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidRuntimeEvidence)
            }
        }
    }

    func testKVTunerScheduleBindsMatrixCellModelLayerAndGroupIdentity() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)
        let invalidFormats = [
            KVFormatGeometryEvidence(
                kind: .kvtuner, tier: binding.cellID,
                keyBits: 0, valueBits: 0, groupSize: binding.groupSize,
                sinkTokens: 0, layerCount: binding.layers.count + 1,
                kvHeadCount: 8, headDimension: 128,
                capacityTokens: 24_192, sequences: 1,
                metadataScalarBytes: 4, recordAlignment: 1),
            KVFormatGeometryEvidence(
                kind: .kvtuner, tier: binding.cellID,
                keyBits: 0, valueBits: 0, groupSize: 64,
                sinkTokens: 0, layerCount: binding.layers.count,
                kvHeadCount: 8, headDimension: 128,
                capacityTokens: 24_192, sequences: 1,
                metadataScalarBytes: 4, recordAlignment: 1),
        ]
        let invalidRows = [
            frontier(
                candidate: model, reference: model,
                matrixID: "other-matrix", cellID: binding.cellID,
                format: measured.format, storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding),
            frontier(
                candidate: model, reference: model,
                matrixID: binding.matrixID, cellID: "other-cell",
                format: measured.format, storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding),
            frontier(
                candidate: identity("different"),
                reference: identity("different"),
                matrixID: binding.matrixID, cellID: binding.cellID,
                format: measured.format, storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding),
        ] + invalidFormats.map { format in
            frontier(
                candidate: model, reference: model,
                matrixID: binding.matrixID, cellID: binding.cellID,
                format: format, storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding)
        }
        for invalid in invalidRows {
            XCTAssertThrowsError(try payload(
                kvQuantTier: binding.cellID,
                frontier: invalid).validatedForRecord()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidKVTunerSchedule)
            }
        }
    }

    func testKVTunerEvaluationCorpusMustMatchKLProvenance() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)
        let candidate = payload(
            kvQuantTier: binding.cellID,
            frontier: frontier(
                candidate: model,
                reference: model,
                matrixID: binding.matrixID,
                cellID: binding.cellID,
                format: measured.format,
                storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding))

        for invalidProvenance in [
            provenance(
                configHash: binding.modelConfigHash,
                manifestHash: binding.checkpointManifestHash,
                corpusHash: "7777777777777777"),
            provenance(
                configHash: binding.modelConfigHash,
                manifestHash: binding.checkpointManifestHash,
                corpusHash: try kvtunerEvaluation().aggregateDigest,
                corpusID: "other-evaluation"),
        ] {
            XCTAssertThrowsError(try ResultRecord(
                subcommand: "kl",
                provenance: invalidProvenance,
                payload: candidate
            ).validatedForRecordEvidence()) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .kvtunerEvaluationProvenanceMismatch)
            }
        }
    }

    func testKVTunerKLRejectsMutatedScheduleEntryDigests() throws {
        let binding = try kvtunerBinding()
        let measured = try kvtunerFormatAndStorage(binding: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash)
        let candidate = payload(
            kvQuantTier: binding.cellID,
            frontier: frontier(
                candidate: model,
                reference: model,
                matrixID: binding.matrixID,
                cellID: binding.cellID,
                format: measured.format,
                storage: measured.storage,
                controlBytes: measured.allocation.controlBytes,
                kvtunerSchedule: binding))
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(candidate)) as? [String: Any])
        var frontierObject = try XCTUnwrap(
            object["frontier"] as? [String: Any])
        var scheduleObject = try XCTUnwrap(
            frontierObject["candidateKVTunerSchedule"] as? [String: Any])
        var corpora = try XCTUnwrap(
            scheduleObject["evaluationCorpora"] as? [[String: Any]])
        corpora[0]["canonicalEntryDigests"] = [
            "5555555555555555",
            "7777777777777777",
        ]
        corpora[0]["canonicalSourceItemDigests"] =
            KVTunerEvaluationSourceProvenance.auditedSourceItemDigests(
                entryDigests: [
                    "5555555555555555",
                    "7777777777777777",
                ])
        scheduleObject["evaluationCorpora"] = corpora
        frontierObject["candidateKVTunerSchedule"] = scheduleObject
        frontierObject["candidateKVTunerEvaluationCorpus"] =
            try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(try kvtunerEvaluation()))
                    as? [String: Any])
        object["frontier"] = frontierObject
        let mutated = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONSerialization.data(withJSONObject: object))

        XCTAssertThrowsError(try mutated.validatedForPromotion()) { error in
            XCTAssertEqual(
                error as? KVFrontierEvidenceError,
                .invalidKVTunerSchedule)
        }
    }

    func testKVarNPromotionRequiresMeasuredExecutionModeAndIterationCell() throws {
        let (format, storage) = try kvarnFormatAndStorage()
        let complete = frontier(
            cellID: "kvarn-k4v2-g128-i8",
            format: format, storage: storage, controlBytes: 8,
            executionMode: "uncompiled-correctness", codecIterations: 8,
            memoryGate: kvarnMemoryGate())

        let completePayload = payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: complete)
        XCTAssertNoThrow(try completePayload.validatedForPromotion())
        XCTAssertNoThrow(try ResultRecord(
            subcommand: "kl", provenance: provenance(), payload: completePayload
        ).validatedForPromotionEvidence())

        let wrongMemorySHA = frontier(
            cellID: "kvarn-k4v2-g128-i8",
            format: format, storage: storage, controlBytes: 8,
            executionMode: "uncompiled-correctness", codecIterations: 8,
            memoryGate: kvarnMemoryGate(
                harnessSHA: String(repeating: "c", count: 40)))
        XCTAssertThrowsError(try ResultRecord(
            subcommand: "kl", provenance: provenance(),
            payload: payload(
                kvQuantTier: "kvarn-k4v2-g128", frontier: wrongMemorySHA)
        ).validatedForPromotionEvidence()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .memoryGateProvenanceMismatch)
        }

        let (i16Format, i16Storage) = try kvarnFormatAndStorage(
            tier: "kvarn-k4v2-g128-i16")
        XCTAssertNoThrow(try payload(
            kvQuantTier: "kvarn-k4v2-g128-i16",
            frontier: frontier(
                cellID: "kvarn-k4v2-g128-i16",
                format: i16Format, storage: i16Storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 16,
                memoryGate: kvarnMemoryGate(
                    tier: "kvarn-k4v2-g128-i16", iterations: 16))
        ).validatedForPromotion())

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 8)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingMemoryGateEvidence)
        }

        for invalid in [
            frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: nil, codecIterations: 8),
            frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "compiled", codecIterations: 8),
            frontier(
                cellID: "kvarn-k4v2-g128-i4",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 4),
            frontier(
                cellID: "kvarn-k4v2-g128-i16",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 8,
                memoryGate: kvarnMemoryGate()),
        ] {
            XCTAssertThrowsError(try payload(
                kvQuantTier: "kvarn-k4v2-g128",
                frontier: invalid).validatedForPromotion()
            ) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
            }
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 0,
                executionMode: "uncompiled-correctness", codecIterations: 8,
                memoryGate: kvarnMemoryGate())
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidControlStorage)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 8,
                memoryGate: kvarnMemoryGate(
                    tier: "kvarn-k4v2-g128-i16", iterations: 16))
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness", codecIterations: 8,
                memoryGate: kvarnMemoryGate(
                    maximumCapacityTokens: 24_193))
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }
    }

    func testKVarNDirectAttentionPromotionRequiresExactI8BindingAndWorkspace()
        throws
    {
        let admission = try compressedAttentionAdmission()
        let model = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256:
                admission.checkpointContentSHA256)
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .splitKVarNQuantizedMM,
            observedOperation: .splitKVarNQuantizedMM,
            admission: admission)
        let (format, storage) = try kvarnFormatAndStorage(
            layerCount: admission.layerCount)
        let direct = frontier(
            schemaVersion: 2,
            candidate: model,
            reference: model,
            cellID: "kvarn-k4v2-g128-i8",
            format: format,
            storage: storage,
            controlBytes: admission.layerCount * MemoryLayout<Int32>.size,
            executionMode: "uncompiled-correctness",
            codecIterations: 8,
            memoryGate: kvarnMemoryGate(),
            compressedKVAttention: binding,
            materializationWorkspaceBytes: 0,
            attentionWorkspaceBytes: storage.actual.workspaceBytes)

        XCTAssertNoThrow(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: direct).validatedForPromotion())

        let roundTripped = try JSONDecoder().decode(
            KLPayload.self,
            from: JSONEncoder().encode(payload(
                kvQuantTier: "kvarn-k4v2-g128",
                frontier: direct)))
        XCTAssertNoThrow(try roundTripped.validatedForPromotion())

        let (i16Format, i16Storage) = try kvarnFormatAndStorage(
            tier: "kvarn-k4v2-g128-i16",
            layerCount: admission.layerCount)
        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128-i16",
            frontier: frontier(
                schemaVersion: 2,
                candidate: model,
                reference: model,
                cellID: "kvarn-k4v2-g128-i16",
                format: i16Format,
                storage: i16Storage,
                controlBytes:
                    admission.layerCount * MemoryLayout<Int32>.size,
                executionMode: "uncompiled-correctness",
                codecIterations: 16,
                memoryGate: kvarnMemoryGate(
                    tier: "kvarn-k4v2-g128-i16", iterations: 16),
                compressedKVAttention: binding,
                materializationWorkspaceBytes: 0,
                attentionWorkspaceBytes:
                    i16Storage.actual.workspaceBytes)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }
    }

    func testKVarNDirectAttentionPromotionRejectsWrongRouteAndWorkspace()
        throws
    {
        let admission = try compressedAttentionAdmission()
        let model = KVModelEvidenceIdentity(
            configHash: admission.modelConfigHash,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256:
                admission.checkpointContentSHA256)
        let kvarnBinding = try CompressedKVAttentionRuntimeBinding(
            request: .splitKVarNQuantizedMM,
            observedOperation: .splitKVarNQuantizedMM,
            admission: admission)
        let affineBinding = try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admission)
        let (format, storage) = try kvarnFormatAndStorage(
            layerCount: admission.layerCount)
        let controlBytes = admission.layerCount
            * MemoryLayout<Int32>.size

        func direct(
            binding: CompressedKVAttentionRuntimeBinding = kvarnBinding,
            memoryGate: KVarNMemoryGateEvidence? = kvarnMemoryGate(),
            materializationWorkspaceBytes: Int = 0,
            attentionWorkspaceBytes: Int? = nil
        ) -> KVFrontierEvidence {
            frontier(
                schemaVersion: 2,
                candidate: model,
                reference: model,
                cellID: "kvarn-k4v2-g128-i8",
                format: format,
                storage: storage,
                controlBytes: controlBytes,
                executionMode: "uncompiled-correctness",
                codecIterations: 8,
                memoryGate: memoryGate,
                compressedKVAttention: binding,
                materializationWorkspaceBytes:
                    materializationWorkspaceBytes,
                attentionWorkspaceBytes: attentionWorkspaceBytes
                    ?? storage.actual.workspaceBytes)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: direct(binding: affineBinding)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: direct(
                materializationWorkspaceBytes: 1,
                attentionWorkspaceBytes:
                    storage.actual.workspaceBytes - 1)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: direct(
                materializationWorkspaceBytes:
                    storage.actual.workspaceBytes,
                attentionWorkspaceBytes: 0)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: direct(memoryGate: nil)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingMemoryGateEvidence)
        }

        let affineWorkspace = breakdown(workspaceBytes: 4_096)
        XCTAssertThrowsError(try payload(
            kvQuantTier: "affine-k4v2-g128",
            frontier: frontier(
                schemaVersion: 2,
                candidate: model,
                reference: model,
                format: geometry(),
                storage: KVStorageEvidence(
                    predicted: affineWorkspace,
                    actual: affineWorkspace),
                controlBytes: 256,
                compressedKVAttention: kvarnBinding,
                materializationWorkspaceBytes: 0,
                attentionWorkspaceBytes: 4_096)
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidRuntimeEvidence)
        }
    }

    func testKVarNPromotionRejectsGeometryWithoutAnAdmittedRuntimeCell() throws {
        let format = KVFormatGeometryEvidence(
            kind: .kvarn, tier: "kvarn-k4v4-g128",
            keyBits: 4, valueBits: 4, groupSize: 128,
            sinkTokens: 128, layerCount: 2, kvHeadCount: 1,
            headDimension: 128, capacityTokens: 257, sequences: 1,
            metadataScalarBytes: 2, recordAlignment: 8)
        let actual = KVStorageBreakdownEvidence(
            payloadBytes: 65_536, metadataBytes: 6_144,
            alignmentPaddingBytes: 0,
            fp16SinkBytes: 131_072, fp16TailBytes: 131_072,
            workspaceBytes: 131_584, totalBytes: 465_408)
        XCTAssertThrowsError(try format.storageEvidence(actual: actual)) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .invalidGeometry)
        }

        for (sequences, capacity) in [(2, 257), (1, Int(Int32.max) + 1)] {
            let unavailable = KVFormatGeometryEvidence(
                kind: .kvarn, tier: "kvarn-k4v2-g128",
                keyBits: 4, valueBits: 2, groupSize: 128,
                sinkTokens: 128, layerCount: 2, kvHeadCount: 1,
                headDimension: 128, capacityTokens: capacity,
                sequences: sequences, metadataScalarBytes: 2,
                recordAlignment: 8)
            XCTAssertThrowsError(try unavailable.storageEvidence(actual: actual)) {
                XCTAssertEqual($0 as? KVFrontierEvidenceError, .invalidGeometry)
            }
        }
    }

    func testKVarNMemoryGateIsDerivedFromTheExactRawProbeMatrix() throws {
        let rows = memoryProbeRows()
        let gate = try KVarNMemoryGateEvidence.derived(
            from: rows,
            artifactSHA256: String(repeating: "b", count: 64),
            runtimeTier: "kvarn-k4v2-g128")

        XCTAssertEqual(gate.encodeSampleCount, 3)
        XCTAssertEqual(gate.decodeSampleCount, 3)
        XCTAssertEqual(gate.cacheBoundarySampleCount, 9)
        XCTAssertEqual(gate.cacheBoundaryMaximumCapacityTokens, 24_192)
        XCTAssertEqual(gate.encodeTransientPeakBytes, 20_000_003)
        XCTAssertEqual(gate.decodeTransientPeakBytes, 2_000_003)
        XCTAssertEqual(gate.cacheBoundaryTransientPeakBytes, 24_192_003)
        XCTAssertEqual(gate.maximumPeakActiveBytes, 2_419_200_000)

        let i16Rows = memoryProbeRows(iterations: 16)
        for artifactRows in [i16Rows, rows + i16Rows] {
            let i16Gate = try KVarNMemoryGateEvidence.derived(
                from: artifactRows,
                artifactSHA256: String(repeating: "c", count: 64),
                runtimeTier: "kvarn-k4v2-g128-i16")
            XCTAssertEqual(i16Gate.runtimeTier, "kvarn-k4v2-g128-i16")
            XCTAssertEqual(i16Gate.codecIterations, 16)
            XCTAssertEqual(i16Gate.cacheBoundaryMaximumCapacityTokens, 24_192)
        }

        let falseCompleteCounts = rows.enumerated().map { index, row in
            index == 0
                ? memoryProbeRow(
                    phase: row.configuration.phase,
                    capacity: row.configuration.capacity,
                    run: row.configuration.run,
                    evaluatedArrayCount: 1,
                    expectedEvaluatedArrayCount: 1)
                : row
        }

        for invalid in [
            Array(rows.dropLast()),
            rows + [rows[0]],
            rows.map { row in
                row.configuration.capacity == 24_192
                    ? memoryProbeRow(
                        phase: row.configuration.phase,
                        capacity: 4_096, run: row.configuration.run)
                    : row
            },
            falseCompleteCounts,
            rows + i16Rows.dropLast(),
        ] {
            XCTAssertThrowsError(try KVarNMemoryGateEvidence.derived(
                from: invalid,
                artifactSHA256: String(repeating: "b", count: 64),
                runtimeTier: "kvarn-k4v2-g128")) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidMemoryGateEvidence)
            }
        }
    }

    func testKVarNMemoryGateAllowsZeroTransientRepeatsButRequiresAPositivePhasePeak() throws {
        let rows = memoryProbeRows()
        let oneZeroRepeat = rows.map { row in
            row.configuration.phase == "cache-boundary"
                && row.configuration.capacity == 4_096
                && row.configuration.run == 1
                ? memoryProbeRow(
                    phase: row.configuration.phase,
                    capacity: row.configuration.capacity,
                    run: row.configuration.run,
                    iterations: row.configuration.iterations,
                    peak: row.highWater.observedPeakActiveBytes,
                    transient: 0)
                : row
        }
        XCTAssertNoThrow(try KVarNMemoryGateEvidence.derived(
            from: oneZeroRepeat,
            artifactSHA256: String(repeating: "b", count: 64),
            runtimeTier: "kvarn-k4v2-g128").validated(
                candidateTier: "kvarn-k4v2-g128",
                candidateIterations: 8))

        for invalidRows in [
            rows.map { row in
                row.configuration.phase == "encode"
                    ? memoryProbeRow(
                        phase: row.configuration.phase,
                        capacity: row.configuration.capacity,
                        run: row.configuration.run,
                        iterations: row.configuration.iterations,
                        peak: row.highWater.observedPeakActiveBytes,
                        transient: 0)
                    : row
            },
            rows.enumerated().map { index, row in
                index == 0
                    ? memoryProbeRow(
                        phase: row.configuration.phase,
                        capacity: row.configuration.capacity,
                        run: row.configuration.run,
                        iterations: row.configuration.iterations,
                        peak: row.highWater.observedPeakActiveBytes,
                        transient: -1)
                    : row
            },
        ] {
            XCTAssertThrowsError(try KVarNMemoryGateEvidence.derived(
                from: invalidRows,
                artifactSHA256: String(repeating: "b", count: 64),
                runtimeTier: "kvarn-k4v2-g128").validated(
                    candidateTier: "kvarn-k4v2-g128",
                    candidateIterations: 8)) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidMemoryGateEvidence)
            }
        }
    }

    func testKVarNRawMemoryArtifactJSONLRejectsEmptyRows() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let rows = Array(memoryProbeRows().prefix(2))
        let lines = try rows.map {
            String(decoding: try encoder.encode($0), as: UTF8.self)
        }
        let valid = Data((lines.joined(separator: "\n") + "\n").utf8)
        XCTAssertEqual(
            try KVarNMemoryProbeArtifact.decodeJSONL(valid),
            rows)

        let invalidArtifacts = [
            Data(),
            Data("\n".utf8),
            Data((lines.joined(separator: "\n") + "\n\n").utf8),
            Data((lines[0] + "\n\n" + lines[1]).utf8),
        ]
        for artifact in invalidArtifacts {
            XCTAssertThrowsError(
                try KVarNMemoryProbeArtifact.decodeJSONL(artifact)
            ) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidMemoryGateEvidence)
            }
        }
    }

    func testKVarNSchema3ArtifactRejectsRowsMissingReconciliationProof() throws {
        let unauthenticated = Data(
            """
            {"configuration":{"capacity":256,"cacheLimitBytes":0,"groupSize":128,"headDimension":128,"heads":8,"iterations":8,"phase":"encode","run":1},"evaluatedArrayCount":8,"expectedEvaluatedArrayCount":8,"hardwareChip":"Apple M3 Ultra","hardwareOS":"macOS 15.5","hardwareRAMBytes":274877906944,"harnessSHA":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","highWater":{"observedPeakActiveBytes":100000000,"transientActiveAboveRetainedBytes":20000001},"mlxSwiftVersion":"0.31.6","schemaVersion":3,"status":"PASS","valuesFinite":true}
            """.utf8)

        XCTAssertThrowsError(
            try KVarNMemoryProbeArtifact.decodeJSONL(unauthenticated)
        ) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }
    }

    func testKVarNMemoryGateRejectsContradictoryReconciliationProof() throws {
        let encoder = JSONEncoder()
        let original = try encoder.encode(memoryProbeRows()[0])
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: original)
                as? [String: Any])
        var reconciliation = try XCTUnwrap(
            object["startReconciliation"] as? [String: Any])
        reconciliation["activeAboveExpectedAllocatorBytes"] = 999
        object["startReconciliation"] = reconciliation
        let tampered = try JSONSerialization.data(withJSONObject: object)
        let row = try XCTUnwrap(
            KVarNMemoryProbeArtifact.decodeJSONL(tampered).first)
        var rows = memoryProbeRows()
        rows[0] = row

        XCTAssertThrowsError(try KVarNMemoryGateEvidence.derived(
            from: rows,
            artifactSHA256: String(repeating: "b", count: 64),
            runtimeTier: "kvarn-k4v2-g128"
        )) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }
    }

    func testKVarNMemoryGateBindsCacheBoundaryLogicalBytesToGeometry() throws {
        let encoder = JSONEncoder()
        let source = memoryProbeRow(
            phase: "cache-boundary", capacity: 256, run: 1)
        let original = try encoder.encode(source)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: original)
                as? [String: Any])
        object["controlLogicalBytes"] = 64
        let tampered = try JSONSerialization.data(withJSONObject: object)
        let row = try XCTUnwrap(
            KVarNMemoryProbeArtifact.decodeJSONL(tampered).first)
        var rows = memoryProbeRows()
        let index = try XCTUnwrap(rows.firstIndex(where: {
            $0.configuration.phase == "cache-boundary"
                && $0.configuration.capacity == 256
                && $0.configuration.run == 1
        }))
        rows[index] = row

        XCTAssertThrowsError(try KVarNMemoryGateEvidence.derived(
            from: rows,
            artifactSHA256: String(repeating: "b", count: 64),
            runtimeTier: "kvarn-k4v2-g128"
        )) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }
    }

    func testKVarNPromotionRejectsCapacityBeyondMeasuredBoundary() throws {
        let (format, storage) = try kvarnFormatAndStorage(
            capacityTokens: 24_193)
        let evidence = frontier(
            cellID: "kvarn-k4v2-g128-i8",
            format: format,
            storage: storage,
            controlBytes: 8,
            executionMode: "uncompiled-correctness",
            codecIterations: 8,
            memoryGate: kvarnMemoryGate())

        XCTAssertThrowsError(try payload(
            kvQuantTier: "kvarn-k4v2-g128",
            frontier: evidence).validatedForPromotion()
        ) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMemoryGateEvidence)
        }
    }

    func testPromotionBindsFormatCapacityToDeepestScoredContext() throws {
        let format = geometry(capacityTokens: 4_096)
        let actual = breakdown(capacityTokens: 4_096)
        let evidence = frontier(
            format: format,
            storage: KVStorageEvidence(predicted: actual, actual: actual))

        XCTAssertThrowsError(try payload(
            frontier: evidence).validatedForPromotion()
        ) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .invalidGeometry)
        }
    }

    func testKVarNMemoryGateRejectsUnmeasuredHeadGeometry() throws {
        for (heads, dimension) in [(4, 128), (8, 256)] {
            let (format, storage) = try kvarnFormatAndStorage(
                kvHeadCount: heads, headDimension: dimension)
            let evidence = frontier(
                cellID: "kvarn-k4v2-g128-i8",
                format: format, storage: storage, controlBytes: 8,
                executionMode: "uncompiled-correctness",
                codecIterations: 8,
                memoryGate: kvarnMemoryGate())

            XCTAssertThrowsError(try payload(
                kvQuantTier: "kvarn-k4v2-g128",
                frontier: evidence).validatedForPromotion()
            ) {
                XCTAssertEqual(
                    $0 as? KVFrontierEvidenceError,
                    .invalidMemoryGateEvidence)
            }
        }
    }

    func testPromotionRejectsMissingFrontierAndLongContextTail() {
        XCTAssertThrowsError(try payload(frontier: nil).withoutFrontier().validatedForPromotion())
        XCTAssertThrowsError(try KLPayload(
            kvQuantTier: "affine-k4v2-g128",
            klMedianNats: 0.04, klLongContextTailP95Nats: nil,
            klPooledMedianNats: 0.05, klPooledP95Nats: 0.4,
            pplCandidate: 10.2, pplReference: 10.0, pplDeltaPct: 2.0,
            totalPositions: 100, entryCount: 4,
            teacherForcedTop1AgreementCount: 83,
            teacherForcedTop1ScoredPositions: 100,
            teacherForcedTop1AgreementRate: 0.83,
            frontier: frontier(),
            shortEntryCount: 3, shortScoredPositions: 72,
            longContextEntryCount: 1,
            longContextScoredPositions: 28).validatedForPromotion())
    }

    func testExploratoryRecordMayOmitStorageButPromotionFailsClosed() throws {
        let full = frontier()
        let exploratory = KVFrontierEvidence(
            schemaVersion: full.schemaVersion,
            matrixID: full.matrixID, cellID: full.cellID,
            sameWeights: full.sameWeights,
            comparisonBaseline: full.comparisonBaseline,
            referenceKVQuantTier: full.referenceKVQuantTier,
            candidateModel: full.candidateModel,
            referenceModel: full.referenceModel,
            candidateFormat: nil, storage: nil)
        let row = payload(frontier: exploratory)

        XCTAssertNoThrow(try row.validatedForRecord())
        XCTAssertThrowsError(try row.validatedForPromotion()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .missingFormat)
        }
    }

    func testPromotionRejectsDifferentWeightsOrInconsistentBaseline() {
        XCTAssertThrowsError(try payload(frontier: frontier(
            sameWeights: false,
            baseline: .differentWeightsFP16KV,
            reference: identity("different"))).validatedForPromotion())
        XCTAssertThrowsError(try payload(frontier: frontier(
            sameWeights: true,
            baseline: .differentWeightsFP16KV)).validatedForRecord())
    }

    func testPromotionRejectsMissingIdentityAndTierMismatch() {
        XCTAssertThrowsError(try payload(frontier: frontier(
            candidate: KVModelEvidenceIdentity(
                configHash: "unknown", checkpointManifestHash: "checkpoint-same")
        )).validatedForRecord())
        XCTAssertThrowsError(try payload(frontier: frontier(
            format: geometry(tier: "wrong-tier"))).validatedForPromotion())
    }

    func testPromotionRejectsStorageMismatchAndInvalidBreakdown() {
        let predicted = breakdown()
        let differentActual = KVStorageBreakdownEvidence(
            payloadBytes: predicted.payloadBytes, metadataBytes: predicted.metadataBytes,
            alignmentPaddingBytes: 0, fp16SinkBytes: 0, fp16TailBytes: 0,
            workspaceBytes: 1, totalBytes: predicted.totalBytes + 1)
        XCTAssertThrowsError(try payload(frontier: frontier(
            storage: KVStorageEvidence(
                predicted: predicted, actual: differentActual))).validatedForPromotion())

        let invalid = breakdown(total: 1)
        XCTAssertThrowsError(try payload(frontier: frontier(
            storage: KVStorageEvidence(predicted: invalid, actual: invalid)
        )).validatedForRecord())
    }

    func testPromotionRequiresExplicitImplementationControlBytes() {
        XCTAssertThrowsError(try payload(frontier: frontier(
            controlBytes: nil
        )).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingControlStorage)
        }
        XCTAssertThrowsError(try payload(frontier: frontier(
            controlBytes: -1
        )).validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidControlStorage)
        }
    }

    func testPromotionRejectsEqualButFabricatedStorageAndTierGeometryMismatch() {
        let fabricated = KVStorageBreakdownEvidence(
            payloadBytes: 201_326_591, metadataBytes: 16_777_217,
            alignmentPaddingBytes: 0, fp16SinkBytes: 0, fp16TailBytes: 0,
            workspaceBytes: 0, totalBytes: 218_103_808)
        XCTAssertThrowsError(try payload(frontier: frontier(
            storage: KVStorageEvidence(
                predicted: fabricated, actual: fabricated))).validatedForPromotion()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .storagePredictionMismatch)
        }

        XCTAssertThrowsError(try payload(frontier: frontier(
            format: KVFormatGeometryEvidence(
                kind: .kvarn, tier: "affine-k4v2-g128",
                keyBits: 4, valueBits: 2, groupSize: 128,
                sinkTokens: 128, layerCount: 64, kvHeadCount: 8,
                headDimension: 128, capacityTokens: 4_096, sequences: 1,
                metadataScalarBytes: 2, recordAlignment: 1)
        )).validatedForRecord())
    }

    func testMetricsRejectNonFiniteAndInconsistentTop1Evidence() {
        XCTAssertThrowsError(try KLPayload(
            kvQuantTier: "affine-k4v2-g128",
            klMedianNats: .nan, klLongContextTailP95Nats: 0.3,
            klPooledMedianNats: 0.05, klPooledP95Nats: 0.4,
            pplCandidate: 10.2, pplReference: 10.0, pplDeltaPct: 2.0,
            totalPositions: 100, entryCount: 4,
            teacherForcedTop1AgreementCount: 83,
            teacherForcedTop1ScoredPositions: 100,
            teacherForcedTop1AgreementRate: 0.83,
            frontier: frontier(),
            shortEntryCount: 3, shortScoredPositions: 72,
            longContextEntryCount: 1,
            longContextScoredPositions: 28).validatedForRecord())
        XCTAssertThrowsError(try KLPayload(
            kvQuantTier: "affine-k4v2-g128",
            klMedianNats: 0.04, klLongContextTailP95Nats: 0.3,
            klPooledMedianNats: 0.05, klPooledP95Nats: 0.4,
            pplCandidate: 10.2, pplReference: 10.0, pplDeltaPct: 2.0,
            totalPositions: 100, entryCount: 4,
            teacherForcedTop1AgreementCount: 83,
            teacherForcedTop1ScoredPositions: 100,
            teacherForcedTop1AgreementRate: 0.5,
            frontier: frontier(),
            shortEntryCount: 3, shortScoredPositions: 72,
            longContextEntryCount: 1,
            longContextScoredPositions: 28).validatedForRecord())
    }

    func testPromotionEnforcesAvailableHardCoherenceFloorPredicates() {
        XCTAssertThrowsError(try payload().withQuality(
            pplCandidate: 20, pplReference: 10, pplDeltaPct: 100
        ).validatedForPromotion()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .qualityFloorFailed("perplexity"))
        }
        XCTAssertThrowsError(try payload().withQuality(
            longTail: 5
        ).validatedForPromotion()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .qualityFloorFailed("longContextTailP95"))
        }
        XCTAssertThrowsError(try payload().withQuality(
            top1Matches: 98, top1Rate: 0.49
        ).validatedForPromotion()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .qualityFloorFailed("teacherForcedTop1"))
        }
    }

    func testPromotionRequiresBothShortAndLongContextCohorts() {
        XCTAssertThrowsError(try payload().withCohorts(
            shortEntries: 0, shortPositions: 0,
            longEntries: 4, longPositions: 100
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingRequiredCohort("short"))
        }
        XCTAssertThrowsError(try payload().withCohorts(
            shortEntries: 4, shortPositions: 100,
            longEntries: 0, longPositions: 0
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingRequiredCohort("longContext"))
        }
    }

    func testPromotionRequiresPerEntrySamplingDetail() {
        XCTAssertThrowsError(try payload().withoutEntryScoring(
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingCohortEntryEvidence)
        }
        XCTAssertThrowsError(try payload().withEntryScoring(
            short: [23, 24, 25], longContext: [128]
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .insufficientEntryPositions(
                    cohort: "short", entryID: "short-0", got: 23, required: 24))
        }
        XCTAssertThrowsError(try payload().withEntryScoring(
            short: [24, 24, 24], longContext: [127]
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .insufficientEntryPositions(
                    cohort: "longContext", entryID: "long-0", got: 127, required: 128))
        }
    }

    func testRecordRejectsMalformedPerEntrySamplingDetail() throws {
        let encoded = try JSONEncoder().encode(payload())

        func decoded(after mutate: (inout [String: Any]) -> Void) throws -> KLPayload {
            var object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            mutate(&object)
            return try JSONDecoder().decode(
                KLPayload.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        let halfPresent = try decoded {
            $0.removeValue(forKey: "longContextEntryScoring")
        }
        XCTAssertThrowsError(try halfPresent.validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingCohortEntryEvidence)
        }

        let duplicateID = try decoded {
            var long = $0["longContextEntryScoring"] as! [[String: Any]]
            long[0]["entryID"] = "short-0"
            $0["longContextEntryScoring"] = long
        }
        XCTAssertThrowsError(try duplicateID.validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMetric("cohortEntryEvidence"))
        }

        let mismatchedAggregate = try decoded {
            var short = $0["shortEntryScoring"] as! [[String: Any]]
            short[0]["scoredPositions"] = 25
            $0["shortEntryScoring"] = short
        }
        XCTAssertThrowsError(try mismatchedAggregate.validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMetric("cohortEntryEvidence"))
        }
    }

    func testPromotionRequiresAContextLockedScoreAt24KDepth() {
        XCTAssertThrowsError(try payload().withLongContextDepth(
            documentTokens: 1_024, scoredContextTokens: 1_023
        ).validatedForPromotion()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .insufficientLongContextDepth(got: 1_023, required: 24_000))
        }
        XCTAssertThrowsError(try payload().withLongContextDepth(
            documentTokens: 1_024, scoredContextTokens: 1_025
        ).validatedForRecord()) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .invalidMetric("longContextDepth"))
        }
    }

    func testPromotionEnvelopeBindsCleanProvenanceCorpusAndCandidateConfig() throws {
        let valid = ResultRecord(
            subcommand: "kl", provenance: provenance(), payload: payload())
        XCTAssertNoThrow(try valid.validatedForPromotionEvidence())

        let dirty = ResultRecord(
            subcommand: "kl", provenance: provenance(gitSHA: "deadbeef-dirty"),
            payload: payload())
        XCTAssertThrowsError(try dirty.validatedForPromotionEvidence())

        let missingCorpus = ResultRecord(
            subcommand: "kl", provenance: provenance(corpusHash: nil), payload: payload())
        XCTAssertThrowsError(try missingCorpus.validatedForPromotionEvidence())

        let mismatch = ResultRecord(
            subcommand: "kl", provenance: provenance(configHash: "other-config"),
            payload: payload())
        XCTAssertThrowsError(try mismatch.validatedForPromotionEvidence()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .candidateProvenanceMismatch)
        }

        let checkpointMismatch = ResultRecord(
            subcommand: "kl",
            provenance: provenance(manifestHash: "other-checkpoint"),
            payload: payload())
        XCTAssertThrowsError(try checkpointMismatch.validatedForPromotionEvidence()) {
            XCTAssertEqual($0 as? KVFrontierEvidenceError, .candidateProvenanceMismatch)
        }
    }

    func testRequiredKLWriterDoesNotCreateEvidenceForInvalidPromotion() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("promotion.jsonl")
        let invalid = ResultRecord(
            subcommand: "kl", provenance: provenance(gitSHA: "unknown"), payload: payload())

        XCTAssertThrowsError(try RequiredKLEvidenceWriter.append(
            invalid, to: url, promotion: true))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testRequiredKLWriterRejectsNewExploratoryEvidenceWithoutEntryDetail() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("exploratory.jsonl")
        let incomplete = ResultRecord(
            subcommand: "kl", provenance: provenance(),
            payload: payload().withoutEntryScoring())

        XCTAssertThrowsError(try RequiredKLEvidenceWriter.append(
            incomplete, to: url, promotion: false)) {
            XCTAssertEqual(
                $0 as? KVFrontierEvidenceError,
                .missingCohortEntryEvidence)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testRequiredKLWriterValidatesAndPersistsCompletePromotion() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("promotion.jsonl")
        let record = ResultRecord(
            subcommand: "kl", provenance: provenance(), payload: payload())

        try RequiredKLEvidenceWriter.append(record, to: url, promotion: true)

        let data = try Data(contentsOf: url)
        XCTAssertTrue(data.last == 0x0a)
        let line = data.dropLast()
        let decoded = try JSONDecoder().decode(
            ResultRecord<KLPayload>.self, from: Data(line))
        XCTAssertEqual(decoded.payload, payload())
    }

    func testHistoricalPayloadWithoutOptionalEvidenceStillDecodes() throws {
        let legacy = Data("""
        {
          "kvQuantTier":"fp16",
          "klMedianNats":0.01,
          "klLongContextTailP95Nats":0.02,
          "klPooledMedianNats":0.01,
          "klPooledP95Nats":0.02,
          "pplCandidate":10.0,
          "pplReference":10.0,
          "pplDeltaPct":0.0,
          "totalPositions":24,
          "entryCount":1
        }
        """.utf8)

        let decoded = try JSONDecoder().decode(KLPayload.self, from: legacy)
        XCTAssertNil(decoded.teacherForcedTop1AgreementCount)
        XCTAssertNil(decoded.teacherForcedTop1ScoredPositions)
        XCTAssertNil(decoded.teacherForcedTop1AgreementRate)
        XCTAssertNil(decoded.frontier)
        XCTAssertNil(decoded.shortEntryCount)
        XCTAssertNil(decoded.shortScoredPositions)
        XCTAssertNil(decoded.longContextEntryCount)
        XCTAssertNil(decoded.longContextScoredPositions)
        XCTAssertNil(decoded.shortEntryScoring)
        XCTAssertNil(decoded.longContextEntryScoring)
        XCTAssertNil(decoded.longContextMaxDocumentTokens)
        XCTAssertNil(decoded.longContextMaxScoredContextTokens)
    }

    private func provenance(
        gitSHA: String = String(repeating: "a", count: 40),
        configHash: String = "config-same",
        manifestHash: String? = "checkpoint-same",
        corpusHash: String? = "corpus-hash",
        corpusID: String? = "measurement-corpus-v2"
    ) -> Provenance {
        Provenance(
            date: "2026-07-14T00:00:00Z", hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            hardwareOS: "macOS 15.5", harnessGitSHA: gitSHA,
            mlxSwiftVersion: "0.31.6", referenceMLXVersion: "0.30.1",
            referenceMLXLMVersion: "0.28.0", modelPath: "/models/qwen3-32b",
            modelConfigHash: configHash,
            modelCheckpointManifestHash: manifestHash,
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: corpusHash == nil ? nil : corpusID,
            corpusContentHash: corpusHash, nonce: "evidence-nonce")
    }
}

private extension KLPayload {
    func withoutFrontier() -> KLPayload {
        return KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate,
            pplReference: pplReference,
            pplDeltaPct: pplDeltaPct,
            totalPositions: totalPositions,
            entryCount: entryCount,
            teacherForcedTop1AgreementCount: teacherForcedTop1AgreementCount,
            teacherForcedTop1ScoredPositions: teacherForcedTop1ScoredPositions,
            teacherForcedTop1AgreementRate: teacherForcedTop1AgreementRate,
            frontier: nil,
            shortEntryCount: shortEntryCount,
            shortScoredPositions: shortScoredPositions,
            longContextEntryCount: longContextEntryCount,
            longContextScoredPositions: longContextScoredPositions,
            shortEntryScoring: shortEntryScoring,
            longContextEntryScoring: longContextEntryScoring,
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
    }

    func withQuality(
        longTail: Double? = nil,
        pplCandidate: Double? = nil,
        pplReference: Double? = nil,
        pplDeltaPct: Double? = nil,
        top1Matches: Int? = nil,
        top1Rate: Double? = nil
    ) -> KLPayload {
        KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: longTail ?? klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate ?? self.pplCandidate,
            pplReference: pplReference ?? self.pplReference,
            pplDeltaPct: pplDeltaPct ?? self.pplDeltaPct,
            totalPositions: totalPositions,
            entryCount: entryCount,
            teacherForcedTop1AgreementCount:
                top1Matches ?? teacherForcedTop1AgreementCount,
            teacherForcedTop1ScoredPositions: teacherForcedTop1ScoredPositions,
            teacherForcedTop1AgreementRate: top1Rate ?? teacherForcedTop1AgreementRate,
            frontier: frontier,
            shortEntryCount: shortEntryCount,
            shortScoredPositions: shortScoredPositions,
            longContextEntryCount: longContextEntryCount,
            longContextScoredPositions: longContextScoredPositions,
            shortEntryScoring: shortEntryScoring,
            longContextEntryScoring: longContextEntryScoring,
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
    }

    func withCohorts(
        shortEntries: Int, shortPositions: Int,
        longEntries: Int, longPositions: Int
    ) -> KLPayload {
        let positions = shortPositions + longPositions
        let matches = positions * 4 / 5
        return KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate,
            pplReference: pplReference,
            pplDeltaPct: pplDeltaPct,
            totalPositions: positions,
            entryCount: entryCount,
            teacherForcedTop1AgreementCount: matches,
            teacherForcedTop1ScoredPositions: positions,
            teacherForcedTop1AgreementRate: Double(matches) / Double(positions),
            frontier: frontier,
            shortEntryCount: shortEntries,
            shortScoredPositions: shortPositions,
            longContextEntryCount: longEntries,
            longContextScoredPositions: longPositions,
            shortEntryScoring: (0 ..< shortEntries).map {
                KVEntryScoringEvidence(
                    entryID: "short-\($0)",
                    scoredPositions: shortPositions / max(shortEntries, 1)
                        + ($0 < shortPositions % max(shortEntries, 1) ? 1 : 0))
            },
            longContextEntryScoring: (0 ..< longEntries).map {
                KVEntryScoringEvidence(
                    entryID: "long-\($0)",
                    scoredPositions: longPositions / max(longEntries, 1)
                        + ($0 < longPositions % max(longEntries, 1) ? 1 : 0))
            },
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
    }

    func withLongContextDepth(
        documentTokens: Int, scoredContextTokens: Int
    ) -> KLPayload {
        KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate,
            pplReference: pplReference,
            pplDeltaPct: pplDeltaPct,
            totalPositions: totalPositions,
            entryCount: entryCount,
            teacherForcedTop1AgreementCount: teacherForcedTop1AgreementCount,
            teacherForcedTop1ScoredPositions: teacherForcedTop1ScoredPositions,
            teacherForcedTop1AgreementRate: teacherForcedTop1AgreementRate,
            frontier: frontier,
            shortEntryCount: shortEntryCount,
            shortScoredPositions: shortScoredPositions,
            longContextEntryCount: longContextEntryCount,
            longContextScoredPositions: longContextScoredPositions,
            shortEntryScoring: shortEntryScoring,
            longContextEntryScoring: longContextEntryScoring,
            longContextMaxDocumentTokens: documentTokens,
            longContextMaxScoredContextTokens: scoredContextTokens)
    }

    func withoutEntryScoring() -> KLPayload {
        KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate,
            pplReference: pplReference,
            pplDeltaPct: pplDeltaPct,
            totalPositions: totalPositions,
            entryCount: entryCount,
            teacherForcedTop1AgreementCount: teacherForcedTop1AgreementCount,
            teacherForcedTop1ScoredPositions: teacherForcedTop1ScoredPositions,
            teacherForcedTop1AgreementRate: teacherForcedTop1AgreementRate,
            frontier: frontier,
            shortEntryCount: shortEntryCount,
            shortScoredPositions: shortScoredPositions,
            longContextEntryCount: longContextEntryCount,
            longContextScoredPositions: longContextScoredPositions,
            shortEntryScoring: nil,
            longContextEntryScoring: nil,
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
    }

    func withEntryScoring(
        short: [Int], longContext: [Int]
    ) -> KLPayload {
        let shortTotal = short.reduce(0, +)
        let longTotal = longContext.reduce(0, +)
        let positions = shortTotal + longTotal
        let matches = positions * 4 / 5
        return KLPayload(
            kvQuantTier: kvQuantTier,
            klMedianNats: klMedianNats,
            klLongContextTailP95Nats: klLongContextTailP95Nats,
            klPooledMedianNats: klPooledMedianNats,
            klPooledP95Nats: klPooledP95Nats,
            pplCandidate: pplCandidate,
            pplReference: pplReference,
            pplDeltaPct: pplDeltaPct,
            totalPositions: positions,
            entryCount: short.count + longContext.count,
            teacherForcedTop1AgreementCount: matches,
            teacherForcedTop1ScoredPositions: positions,
            teacherForcedTop1AgreementRate: Double(matches) / Double(positions),
            frontier: frontier,
            shortEntryCount: short.count,
            shortScoredPositions: shortTotal,
            longContextEntryCount: longContext.count,
            longContextScoredPositions: longTotal,
            shortEntryScoring: short.enumerated().map {
                KVEntryScoringEvidence(
                    entryID: "short-\($0.offset)", scoredPositions: $0.element)
            },
            longContextEntryScoring: longContext.enumerated().map {
                KVEntryScoringEvidence(
                    entryID: "long-\($0.offset)", scoredPositions: $0.element)
            },
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
    }
}
