import Foundation
import XCTest
@testable import HarnessCore

final class TaskCoherenceEvidenceTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v1"
    private let candidateCellID = "affine-k4v2-g128"
    private let kvtunerCellID = "kvtuner-g128-b4.5"
    private let modelConfigHash = "0123456789abcdef"
    private let checkpointManifestHash = "fedcba9876543210"
    private let cleanSHA = String(repeating: "a", count: 40)

    func testQualificationV3RecordsStructuredChatTemplateAndDecodesLegacyV2() throws {
        let v3 = TaskCoherenceRunConfiguration.qualificationV3(
            structuredToolMaxTokens: 96)

        XCTAssertEqual(v3.restrictedChoicePromptFormat, .rawV1)
        XCTAssertEqual(
            v3.structuredToolPromptFormat,
            .checkpointChatTemplateGenerationPromptThinkingDisabledV1)
        XCTAssertNoThrow(try v3.validated())

        let encodedV3 = try JSONEncoder().encode(v3)
        let v3Object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encodedV3)
                as? [String: Any])
        XCTAssertEqual(
            v3Object["restrictedChoicePromptFormat"] as? String,
            "raw-v1")
        XCTAssertEqual(
            v3Object["structuredToolPromptFormat"] as? String,
            "checkpoint-chat-template-generation-prompt-thinking-disabled-v1")

        let v2 = TaskCoherenceRunConfiguration.qualificationV2(
            structuredToolMaxTokens: 96)
        var legacyObject = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(v2)) as? [String: Any])
        legacyObject.removeValue(forKey: "restrictedChoicePromptFormat")
        legacyObject.removeValue(forKey: "structuredToolPromptFormat")
        let legacyData = try JSONSerialization.data(
            withJSONObject: legacyObject)

        XCTAssertEqual(
            try JSONDecoder().decode(
                TaskCoherenceRunConfiguration.self, from: legacyData),
            v2)
    }

    func testRunnableTaskTierCellMappingIsClosed() {
        XCTAssertEqual(
            TaskCoherenceArtifact.expectedCellID(forTier: "fp16"), "fp16")
        XCTAssertEqual(
            TaskCoherenceArtifact.expectedCellID(
                forTier: "affine-k8v2-g128"),
            "affine-k8v2-g128")
        XCTAssertEqual(
            TaskCoherenceArtifact.expectedCellID(
                forTier: "kvarn-k4v2-g128"),
            "kvarn-k4v2-g128-i8")
        XCTAssertEqual(
            TaskCoherenceArtifact.expectedCellID(
                forTier: kvtunerCellID),
            kvtunerCellID)
        XCTAssertNil(TaskCoherenceArtifact.expectedCellID(forTier: "tq2.5"))
        for invalid in [
            "kvtuner-unbound",
            "kvtuner-g0128-b4.5",
            "kvtuner-g128-b4.50",
            "kvtuner-g32-b4.5",
        ] {
            XCTAssertNil(TaskCoherenceArtifact.expectedCellID(
                forTier: invalid))
        }
    }

    func testReferenceModelIdentityRequiresExactContentWhenCandidateDeclaresIt() {
        let contentSHA256 = String(repeating: "d", count: 64)
        let contentlessReference = TaskCoherenceRunIdentity(
            corpusID: "kvarn-task-coherence-v2",
            corpusContentHash: "1740d0d07f586def",
            modelConfigHash: modelConfigHash,
            modelCheckpointManifestHash: checkpointManifestHash,
            modelCheckpointContentSHA256: nil,
            kvQuantTier: "fp16")
        let exactReference = TaskCoherenceRunIdentity(
            corpusID: contentlessReference.corpusID,
            corpusContentHash: contentlessReference.corpusContentHash,
            modelConfigHash: contentlessReference.modelConfigHash,
            modelCheckpointManifestHash:
                contentlessReference.modelCheckpointManifestHash,
            modelCheckpointContentSHA256: contentSHA256,
            kvQuantTier: "fp16")
        let historicalCandidate = KVModelEvidenceIdentity(
            configHash: modelConfigHash,
            checkpointManifestHash: checkpointManifestHash)
        let exactCandidate = KVModelEvidenceIdentity(
            configHash: modelConfigHash,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: contentSHA256)

        XCTAssertTrue(TaskCoherenceArtifact.referenceModelIdentityMatches(
            reference: contentlessReference,
            candidate: historicalCandidate))
        XCTAssertFalse(TaskCoherenceArtifact.referenceModelIdentityMatches(
            reference: contentlessReference,
            candidate: exactCandidate))
        XCTAssertTrue(TaskCoherenceArtifact.referenceModelIdentityMatches(
            reference: exactReference,
            candidate: exactCandidate))

        let substitutedCandidate = KVModelEvidenceIdentity(
            configHash: modelConfigHash,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: String(repeating: "e", count: 64))
        XCTAssertFalse(TaskCoherenceArtifact.referenceModelIdentityMatches(
            reference: exactReference,
            candidate: substitutedCandidate))
    }

    func testSchemaThreeFP16CarriesExactCheckpointWithoutAttentionBinding() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let contentSHA256 = String(repeating: "d", count: 64)
        let rows = try makeRows(
            corpus: corpus,
            tier: "fp16",
            cellID: "fp16",
            referenceArtifactSHA256: nil,
            modelCheckpointContentSHA256Override: contentSHA256,
            schemaVersion:
                TaskCoherenceArtifact.compressedAttentionSchemaVersion)

        let summary = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)
        XCTAssertEqual(
            summary.identity.modelCheckpointContentSHA256,
            contentSHA256)

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus,
                tier: "fp16",
                cellID: "fp16",
                referenceArtifactSHA256: nil,
                modelCheckpointContentSHA256Override: contentSHA256,
                schemaVersion: TaskCoherenceArtifact.schemaVersion)),
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .invalidRuntimeEvidence("compressed-attention-schema"))
            }
    }

    func testTaskCompressedAttentionBindingAuthenticatesModelTokenizerAndOperation() throws {
        let admission = try compressedAttentionAdmission()
        let identity = TaskCoherenceRunIdentity(
            corpusID: "kvarn-task-coherence-v2",
            corpusContentHash: "1740d0d07f586def",
            modelConfigHash: admission.modelConfigHash,
            modelCheckpointManifestHash:
                admission.checkpointManifestHash,
            modelCheckpointContentSHA256:
                admission.checkpointContentSHA256,
            kvQuantTier: "affine-k4v2-g128")
        let tokenization = TaskCoherenceTokenizationEvidence(
            tokenizerManifestSHA256: admission.tokenizerSHA256,
            promptTokenIDsSHA256: String(repeating: "c", count: 64),
            restrictedChoiceLabelTokenIDs: nil)
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .splitAffineQuantizedMM,
            observedOperation: .splitQuantizedMM,
            admission: admission)

        XCTAssertNoThrow(try TaskCoherenceArtifact
            .validateCompressedKVAttention(
                binding,
                identity: identity,
                tokenization: tokenization))

        var forgedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(binding)) as? [String: Any])
        forgedObject["observedOperation"] = "materialized-kv"
        let forged = try JSONDecoder().decode(
            CompressedKVAttentionRuntimeBinding.self,
            from: JSONSerialization.data(withJSONObject: forgedObject))
        XCTAssertThrowsError(try TaskCoherenceArtifact
            .validateCompressedKVAttention(
                forged,
                identity: identity,
                tokenization: tokenization)) {
            XCTAssertEqual(
                $0 as? TaskCoherenceEvidenceError,
                .invalidRuntimeEvidence("compressed-attention"))
        }

        let mismatchedIdentity = TaskCoherenceRunIdentity(
            corpusID: identity.corpusID,
            corpusContentHash: identity.corpusContentHash,
            modelConfigHash: "ffffffffffffffff",
            modelCheckpointManifestHash:
                identity.modelCheckpointManifestHash,
            modelCheckpointContentSHA256:
                identity.modelCheckpointContentSHA256,
            kvQuantTier: identity.kvQuantTier)
        XCTAssertThrowsError(try TaskCoherenceArtifact
            .validateCompressedKVAttention(
                binding,
                identity: mismatchedIdentity,
                tokenization: tokenization))

        let mismatchedContentIdentity = TaskCoherenceRunIdentity(
            corpusID: identity.corpusID,
            corpusContentHash: identity.corpusContentHash,
            modelConfigHash: identity.modelConfigHash,
            modelCheckpointManifestHash:
                identity.modelCheckpointManifestHash,
            modelCheckpointContentSHA256:
                String(repeating: "0", count: 64),
            kvQuantTier: identity.kvQuantTier)
        XCTAssertThrowsError(try TaskCoherenceArtifact
            .validateCompressedKVAttention(
                binding,
                identity: mismatchedContentIdentity,
                tokenization: tokenization)) {
            XCTAssertEqual(
                $0 as? TaskCoherenceEvidenceError,
                .invalidRuntimeEvidence("compressed-attention"))
        }
    }

    func testTaskCompressedAttentionSchemaThreeIsRequiredForAffineBindings() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let admission = try compressedAttentionAdmission()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil,
            modelConfigHashOverride: admission.modelConfigHash,
            modelCheckpointManifestHashOverride:
                admission.checkpointManifestHash)
        let binding = try compressedAttentionBinding()

        XCTAssertNoThrow(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: candidateCellID,
                cellID: candidateCellID,
                referenceArtifactSHA256: reference.artifactSHA256,
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: binding)),
            corpus: corpus))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: candidateCellID,
                cellID: candidateCellID,
                referenceArtifactSHA256: reference.artifactSHA256,
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: nil)),
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .invalidRuntimeEvidence("compressed-attention-schema"))
            }

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: candidateCellID,
                cellID: candidateCellID,
                referenceArtifactSHA256: reference.artifactSHA256,
                schemaVersion: TaskCoherenceArtifact.schemaVersion,
                compressedKVAttention: binding)),
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .invalidRuntimeEvidence("compressed-attention-schema"))
            }
    }

    func testTaskCompressedAttentionSchemaThreeForbidsNonAffineBindings() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let binding = try compressedAttentionBinding()

        XCTAssertNoThrow(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: "fp16", cellID: "fp16",
                referenceArtifactSHA256: nil,
                modelCheckpointContentSHA256Override:
                    String(repeating: "d", count: 64),
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: nil)),
            corpus: corpus))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: "fp16", cellID: "fp16",
                referenceArtifactSHA256: nil,
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: binding)),
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .invalidRuntimeEvidence("compressed-attention-schema"))
            }
    }

    func testTaskCompressedAttentionSchemaThreeCoversKVTunerBindings() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        let admission = try compressedAttentionAdmission()
        let schedule = try kvtunerBinding(
            corpus: corpus,
            modelConfigHash: admission.modelConfigHash,
            modelConfigSHA256: admission.modelConfigSHA256,
            checkpointManifestHash: admission.checkpointManifestHash,
            checkpointContentSHA256:
                admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
            layerCount: admission.layerCount)
        let binding = try CompressedKVAttentionRuntimeBinding(
            request: .materialize,
            observedOperation: .materializedKV,
            admission: admission)

        XCTAssertNoThrow(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: kvtunerCellID,
                cellID: kvtunerCellID,
                referenceArtifactSHA256: reference.artifactSHA256,
                kvtunerSchedule: schedule,
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: binding)),
            corpus: corpus))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: kvtunerCellID,
                cellID: kvtunerCellID,
                referenceArtifactSHA256: reference.artifactSHA256,
                kvtunerSchedule: schedule,
                schemaVersion:
                    TaskCoherenceArtifact.compressedAttentionSchemaVersion,
                compressedKVAttention: nil)),
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .invalidRuntimeEvidence("kvtuner-schedule"))
            }
    }

    func testPromptLayoutRequiresMaterialInsideCompletedCompressedTiles() {
        XCTAssertNoThrow(try TaskCoherencePromptLayoutEvidence(
            promptTokens: 512,
            materialStartToken: 128,
            materialEndToken: 256,
            compressedRegionEndToken: 512,
            minimumCompletedTileCount: 1).validated())

        for invalid in [
            TaskCoherencePromptLayoutEvidence(
                promptTokens: 512, materialStartToken: 127,
                materialEndToken: 256, compressedRegionEndToken: 512,
                minimumCompletedTileCount: 1),
            TaskCoherencePromptLayoutEvidence(
                promptTokens: 511, materialStartToken: 128,
                materialEndToken: 384, compressedRegionEndToken: 511,
                minimumCompletedTileCount: 2),
            TaskCoherencePromptLayoutEvidence(
                promptTokens: 512, materialStartToken: 128,
                materialEndToken: 257, compressedRegionEndToken: 512,
                minimumCompletedTileCount: 1),
        ] {
            XCTAssertThrowsError(try invalid.validated())
        }
    }

    func testPromptLayoutDerivesConservativeTokenizerBoundarySpan() throws {
        let stablePrefix = Array(0 ..< 140)
        let stableMaterial = stablePrefix + Array(200 ..< 300)
        let stablePrompt = stableMaterial + Array(400 ..< 700)
        let stable = try TaskCoherencePromptLayoutEvidence.derive(
            prefixTokenIDs: stablePrefix,
            prefixAndMaterialTokenIDs: stableMaterial,
            suffixAndQueryTokenIDs: Array(400 ..< 700),
            promptTokenIDs: stablePrompt)
        XCTAssertEqual(stable.materialStartToken, 140)
        XCTAssertEqual(stable.materialEndToken, 240)

        var mergedPrefix = stablePrefix
        mergedPrefix[139] = 9_999
        var mergedMaterial = stableMaterial
        mergedMaterial[239] = 8_888
        let merged = try TaskCoherencePromptLayoutEvidence.derive(
            prefixTokenIDs: mergedPrefix,
            prefixAndMaterialTokenIDs: mergedMaterial,
            suffixAndQueryTokenIDs: Array(400 ..< 700),
            promptTokenIDs: stablePrompt)
        XCTAssertEqual(merged.materialStartToken, 139)
        XCTAssertEqual(merged.materialEndToken, 240)
        XCTAssertEqual(merged.compressedRegionEndToken, 512)
        XCTAssertEqual(merged.minimumCompletedTileCount, 1)
    }

    func testPromptLayoutIncludesEveryTokenResegmentedAtMaterialSuffixBoundary() throws {
        let prefix = Array(0 ..< 140)
        let prefixAndMaterial = prefix + Array(200 ..< 300)
        let stableSuffix = Array(402 ..< 700)
        let prompt = Array(prefixAndMaterial.dropLast(2))
            + [9_900, 9_901, 9_902]
            + stableSuffix

        let layout = try TaskCoherencePromptLayoutEvidence.derive(
            prefixTokenIDs: prefix,
            prefixAndMaterialTokenIDs: prefixAndMaterial,
            suffixAndQueryTokenIDs: [8_000, 8_001] + stableSuffix,
            promptTokenIDs: prompt)

        // The three replacement tokens all belong to the uncertain material/suffix boundary.
        // A single-divergent-token heuristic returns 239 and understates compressed coverage.
        XCTAssertEqual(layout.materialEndToken, 241)
    }

    func testStrictJSONLRejectsEmptyInternalRowsAndMalformedRecords() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let line = try rows[0].jsonLine()

        XCTAssertThrowsError(try TaskCoherenceArtifact.decodeJSONL(
            Data("\(line)\n\n\(line)\n".utf8)))
        XCTAssertThrowsError(try TaskCoherenceArtifact.decodeJSONL(
            Data("{not-json}\n".utf8)))
    }

    func testLegacyTaskSchemaCannotQualifyWithV2Fields() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        var rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: 1,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: first.payload.promptContentHash,
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: first.payload.scoredOutput,
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: first.payload.engagement))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)) { error in
                XCTAssertEqual(
                    error as? TaskCoherenceEvidenceError,
                    .unsupportedSchema(1))
            }
    }

    func testSummaryRequiresEveryFrozenCaseExactlyOnce() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(Array(rows.dropLast())), corpus: corpus))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(Array(rows.dropLast()) + [rows[0]]),
            corpus: corpus))
    }

    func testSummaryBindsFrozenPromptAndRunProvenance() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        var rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: first.payload.schemaVersion,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: "forged-prompt",
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: first.payload.scoredOutput,
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: first.payload.engagement))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))

        rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let mismatched = rows[0]
        rows[0] = ResultRecord(
            subcommand: mismatched.subcommand,
            provenance: provenance(
                corpus: corpus, tier: "fp16", nonce: "other-run"),
            payload: mismatched.payload)
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testKVarNSummaryRequiresRealCompletedTileEngagement() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        XCTAssertEqual(
            Set(reference.cases.map {
                $0.tokenization.tokenizerManifestSHA256
            }).count,
            1)
        var rows = try makeRows(
            corpus: corpus, tier: "kvarn-k4v2-g128",
            cellID: "kvarn-k4v2-g128-i8",
            referenceArtifactSHA256: reference.artifactSHA256)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: first.payload.schemaVersion,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: first.payload.promptContentHash,
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: first.payload.scoredOutput,
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: 513, affineTokens: nil,
                    kvarnCompletedTileCount: 0,
                    kvarnCompressedTokens: 0,
                    kvarnCodecIterations: 8,
                    kvarnExecutionMode: "uncompiled-correctness")))

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testKVTunerSummaryBindsScheduleAndExactGenerationAndScoringEngagement()
        throws
    {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        let binding = try kvtunerBinding(corpus: corpus)
        let compressedBinding = try compressedAttentionBinding(
            request: .materialize,
            observedOperation: .materializedKV)
        let rows = try makeRows(
            corpus: corpus, tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceArtifactSHA256: reference.artifactSHA256,
            kvtunerSchedule: binding,
            schemaVersion:
                TaskCoherenceArtifact.compressedAttentionSchemaVersion,
            compressedKVAttention: compressedBinding)

        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)
        XCTAssertTrue(try XCTUnwrap(
            candidate.identity.kvtunerSchedule).sameSchedule(as: binding))
        XCTAssertTrue(candidate.cases.allSatisfy {
            $0.engagement.kvtunerTokens == $0.engagement.cachedTokens
                && $0.engagement.kvtunerLayerCount == binding.layers.count
        })
        XCTAssertTrue(candidate.cases.allSatisfy { payload in
            if payload.score.domain == .structuredTool {
                return payload.engagement.scoringCachedTokens == nil
                    && payload.engagement.scoringKVTunerLayerCount == nil
            }
            return payload.engagement.scoringCachedTokens
                    == payload.layout.promptTokens
                && payload.engagement.scoringKVTunerLayerCount
                    == binding.layers.count
        })

        let missingBinding = try makeRows(
            corpus: corpus, tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceArtifactSHA256: reference.artifactSHA256,
            kvtunerSchedule: nil)
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(missingBinding), corpus: corpus))

        var wrongLayers = rows
        let first = wrongLayers[0]
        wrongLayers[0] = replacing(
            first,
            engagement: TaskCoherenceCacheEngagementEvidence(
                cachedTokens: first.payload.engagement.cachedTokens,
                affineTokens: nil,
                kvtunerTokens: first.payload.engagement.kvtunerTokens,
                kvtunerLayerCount: binding.layers.count + 1,
                kvarnCompletedTileCount: nil,
                kvarnCompressedTokens: nil,
                kvarnCodecIterations: nil,
                kvarnExecutionMode: nil,
                scoringCachedTokens:
                    first.payload.engagement.scoringCachedTokens,
                scoringKVTunerLayerCount:
                    first.payload.engagement.scoringKVTunerLayerCount))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(wrongLayers), corpus: corpus))

        var wrongScoringLayers = rows
        wrongScoringLayers[0] = replacing(
            first,
            engagement: TaskCoherenceCacheEngagementEvidence(
                cachedTokens: first.payload.engagement.cachedTokens,
                affineTokens: nil,
                kvtunerTokens: first.payload.engagement.kvtunerTokens,
                kvtunerLayerCount: binding.layers.count,
                kvarnCompletedTileCount: nil,
                kvarnCompressedTokens: nil,
                kvarnCodecIterations: nil,
                kvarnExecutionMode: nil,
                scoringCachedTokens:
                    first.payload.engagement.scoringCachedTokens,
                scoringKVTunerLayerCount: binding.layers.count + 1))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(wrongScoringLayers), corpus: corpus))

        var wrongTokens = rows
        wrongTokens[0] = replacing(
            first,
            engagement: TaskCoherenceCacheEngagementEvidence(
                cachedTokens: first.payload.engagement.cachedTokens,
                affineTokens: nil,
                kvtunerTokens:
                    (first.payload.engagement.kvtunerTokens ?? 0) - 1,
                kvtunerLayerCount: binding.layers.count,
                kvarnCompletedTileCount: nil,
                kvarnCompressedTokens: nil,
                kvarnCodecIterations: nil,
                kvarnExecutionMode: nil,
                scoringCachedTokens:
                    first.payload.engagement.scoringCachedTokens,
                scoringKVTunerLayerCount:
                    first.payload.engagement.scoringKVTunerLayerCount))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(wrongTokens), corpus: corpus))
    }

    func testNonKVTunerTaskRowsRejectAnExtraScheduleBinding() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        let binding = try kvtunerBinding(corpus: corpus)
        let rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256,
            kvtunerSchedule: binding)

        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testCandidateAssessmentBindsExactFP16Artifact() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        let candidate = try summary(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceDigest: reference.artifactSHA256)

        let evidence = try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus)
        XCTAssertTrue(evidence.assessment.hardFloorPassed)
        XCTAssertEqual(
            evidence.reference.artifactSHA256,
            reference.artifactSHA256)

        let wrongReference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil, nonce: "alternate-fp16-run")
        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: wrongReference,
            corpus: corpus))
    }

    func testContentBoundPromotionRejectsContentlessOrSubstitutedFP16Reference() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let admission = try compressedAttentionAdmission()
        let taskBinding = try kvtunerBinding(corpus: corpus)
        let contentlessReference = try summary(
            corpus: corpus,
            tier: "fp16",
            cellID: "fp16",
            referenceDigest: nil,
            modelConfigHashOverride: admission.modelConfigHash,
            modelCheckpointManifestHashOverride:
                admission.checkpointManifestHash)
        let contentlessCandidate = try summary(
            corpus: corpus,
            tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceDigest: contentlessReference.artifactSHA256,
            kvtunerSchedule: taskBinding)

        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: contentlessCandidate,
            reference: contentlessReference,
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .referenceArtifactMismatch)
            }

        let substitutedReference = try summary(
            corpus: corpus,
            tier: "fp16",
            cellID: "fp16",
            referenceDigest: nil,
            modelConfigHashOverride: admission.modelConfigHash,
            modelCheckpointManifestHashOverride:
                admission.checkpointManifestHash,
            modelCheckpointContentSHA256Override:
                String(repeating: "e", count: 64))
        let substitutedCandidate = try summary(
            corpus: corpus,
            tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceDigest: substitutedReference.artifactSHA256,
            kvtunerSchedule: taskBinding)
        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: substitutedCandidate,
            reference: substitutedReference,
            corpus: corpus)) {
                XCTAssertEqual(
                    $0 as? TaskCoherenceEvidenceError,
                    .referenceArtifactMismatch)
            }

        let exactReference = try summary(
            corpus: corpus,
            tier: "fp16",
            cellID: "fp16",
            referenceDigest: nil,
            modelConfigHashOverride: admission.modelConfigHash,
            modelCheckpointManifestHashOverride:
                admission.checkpointManifestHash,
            modelCheckpointContentSHA256Override:
                admission.checkpointContentSHA256)
        let exactCandidate = try summary(
            corpus: corpus,
            tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceDigest: exactReference.artifactSHA256,
            kvtunerSchedule: taskBinding)
        XCTAssertNoThrow(try TaskCoherencePromotionEvidence.derive(
            candidate: exactCandidate,
            reference: exactReference,
            corpus: corpus))
    }

    func testPromotionBindsTaskArtifactToKLMatrixCellAndCleanRuntime() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        let candidate = try summary(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceDigest: reference.artifactSHA256)
        let evidence = try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus)

        let kl = try validKLRecord()
        XCTAssertNoThrow(try evidence.validated(
            with: kl, corpus: corpus))

        let wrongCell = try validKLRecord(cellID: "affine-k4v2-g64")
        XCTAssertThrowsError(try evidence.validated(
            with: wrongCell, corpus: corpus))
    }

    func testKVTunerPromotionPairsTheExactTaskAndKLSchedule() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let admission = try compressedAttentionAdmission()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil,
            modelConfigHashOverride: admission.modelConfigHash,
            modelCheckpointManifestHashOverride:
                admission.checkpointManifestHash,
            modelCheckpointContentSHA256Override:
                admission.checkpointContentSHA256)
        let taskBinding = try kvtunerBinding(corpus: corpus)
        let candidate = try summary(
            corpus: corpus, tier: kvtunerCellID,
            cellID: kvtunerCellID,
            referenceDigest: reference.artifactSHA256,
            kvtunerSchedule: taskBinding)
        let promotion = try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus)

        let measurement = try kvtunerKLEvaluationCorpus()
        let matchingKLBinding = try kvtunerBinding(
            corpus: corpus,
            evaluationCorpora: [measurement])
        XCTAssertTrue(taskBinding.sameSchedule(as: matchingKLBinding))
        XCTAssertNoThrow(try promotion.validated(
            with: validKVTunerKLRecord(binding: matchingKLBinding),
            corpus: corpus))

        let differentKLBinding = try kvtunerBinding(
            corpus: corpus,
            searchArtifactSHA256: String(repeating: "e", count: 64),
            evaluationCorpora: [measurement])
        XCTAssertFalse(taskBinding.sameSchedule(as: differentKLBinding))
        XCTAssertThrowsError(try promotion.validated(
            with: validKVTunerKLRecord(binding: differentKLBinding),
            corpus: corpus)) {
            XCTAssertEqual(
                $0 as? TaskCoherenceEvidenceError,
                .klEvidenceMismatch("kvtuner-schedule"))
        }
    }

    func testSummaryAuthenticatesCanonicalBytesAndEmbeddedCases() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let data = try artifactData(rows)
        let summary = try TaskCoherenceArtifact.summarize(
            data, corpus: corpus)
        XCTAssertEqual(summary.cases.count, corpus.items.count)
        XCTAssertEqual(summary.artifactSHA256, sha256Hex(data))

        var noncanonical = Data(data.dropLast())
        noncanonical.append(contentsOf: Data(" \n".utf8))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            noncanonical, corpus: corpus))

        let reference = summary
        let candidate = try self.summary(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceDigest: reference.artifactSHA256)
        var forgedCases = candidate.cases
        let first = forgedCases[0]
        forgedCases[0] = TaskCoherenceCasePayload(
            schemaVersion: first.schemaVersion,
            matrixID: first.matrixID, cellID: first.cellID,
            identity: first.identity,
            referenceArtifactSHA256: first.referenceArtifactSHA256,
            promptContentHash: first.promptContentHash,
            runConfiguration: first.runConfiguration,
            tokenization: first.tokenization,
            layout: first.layout,
            generatedTokenCount: first.generatedTokenCount,
            scoredOutput: first.scoredOutput,
            outputSHA256: first.outputSHA256,
            score: first.score,
            engagement: TaskCoherenceCacheEngagementEvidence(
                cachedTokens: 0, affineTokens: 0,
                kvarnCompletedTileCount: nil,
                kvarnCompressedTokens: nil,
                kvarnCodecIterations: nil,
                kvarnExecutionMode: nil))
        let forged = TaskCoherenceArtifactSummary(
            schemaVersion: candidate.schemaVersion,
            matrixID: candidate.matrixID, cellID: candidate.cellID,
            artifactSHA256: candidate.artifactSHA256,
            referenceArtifactSHA256:
                candidate.referenceArtifactSHA256,
            identity: candidate.identity,
            runConfiguration: candidate.runConfiguration,
            provenance: candidate.provenance,
            caseCount: candidate.caseCount,
            cases: forgedCases,
            scores: candidate.scores)
        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: forged, reference: reference, corpus: corpus))
    }

    func testOutputDigestAndScoreAreRecomputedFromPreservedOutput() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        var rows = try makeRows(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceArtifactSHA256: nil)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: first.payload.schemaVersion,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: first.payload.promptContentHash,
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: "tampered-output",
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: first.payload.engagement))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testEngagementMustCoverTheEntireGeneratedAnswer() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        let index = try XCTUnwrap(rows.firstIndex(where: {
            $0.payload.score.domain == .structuredTool
        }))
        let stale = rows[index]
        rows[index] = ResultRecord(
            subcommand: stale.subcommand,
            provenance: stale.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: stale.payload.schemaVersion,
                matrixID: stale.payload.matrixID,
                cellID: stale.payload.cellID,
                identity: stale.payload.identity,
                referenceArtifactSHA256:
                    stale.payload.referenceArtifactSHA256,
                promptContentHash: stale.payload.promptContentHash,
                runConfiguration: stale.payload.runConfiguration,
                tokenization: stale.payload.tokenization,
                layout: stale.payload.layout,
                generatedTokenCount: stale.payload.generatedTokenCount,
                scoredOutput: stale.payload.scoredOutput,
                outputSHA256: stale.payload.outputSHA256,
                score: stale.payload.score,
                engagement: TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: stale.payload.layout.promptTokens,
                    affineTokens: stale.payload.layout.promptTokens,
                    kvarnCompletedTileCount: nil,
                    kvarnCompressedTokens: nil,
                    kvarnCodecIterations: nil,
                    kvarnExecutionMode: nil)))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testRestrictedChoiceScoringRunMustAlsoEngageLossyCache() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: first.payload.schemaVersion,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: first.payload.promptContentHash,
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: first.payload.scoredOutput,
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: first.payload.engagement.cachedTokens,
                    affineTokens: first.payload.engagement.affineTokens,
                    kvarnCompletedTileCount: nil,
                    kvarnCompressedTokens: nil,
                    kvarnCodecIterations: nil,
                    kvarnExecutionMode: nil,
                    scoringCachedTokens:
                        first.payload.layout.promptTokens - 1,
                    scoringKVarNCompletedTileCount: nil,
                    scoringKVarNCompressedTokens: nil)))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testCandidateAssessmentBindsExactPromptAndLabelTokenization() throws {
        XCTAssertEqual(taskTokenIDsSHA256([7, 11, 13]),
            taskTokenIDsSHA256([7, 11, 13]))
        XCTAssertNotEqual(taskTokenIDsSHA256([7, 11, 13]),
            taskTokenIDsSHA256([7, 13, 11]))

        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        for index in rows.indices where
            rows[index].payload.score.domain != .structuredTool
        {
            let row = rows[index]
            rows[index] = ResultRecord(
                subcommand: row.subcommand,
                provenance: row.provenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: row.payload.schemaVersion,
                    matrixID: row.payload.matrixID,
                    cellID: row.payload.cellID,
                    identity: row.payload.identity,
                    referenceArtifactSHA256:
                        row.payload.referenceArtifactSHA256,
                    promptContentHash: row.payload.promptContentHash,
                    runConfiguration: row.payload.runConfiguration,
                    tokenization: TaskCoherenceTokenizationEvidence(
                        tokenizerManifestSHA256:
                            row.payload.tokenization.tokenizerManifestSHA256,
                        promptTokenIDsSHA256:
                            row.payload.tokenization.promptTokenIDsSHA256,
                        restrictedChoiceLabelTokenIDs: [
                            "A": 101, "B": 202, "C": 303, "D": 405,
                        ]),
                    layout: row.payload.layout,
                    generatedTokenCount: row.payload.generatedTokenCount,
                    scoredOutput: row.payload.scoredOutput,
                    outputSHA256: row.payload.outputSHA256,
                    score: row.payload.score,
                    engagement: row.payload.engagement))
        }
        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)

        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus))
    }

    func testCandidateAssessmentBindsTokenizerManifest() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        for index in rows.indices {
            let row = rows[index]
            rows[index] = ResultRecord(
                subcommand: row.subcommand,
                provenance: row.provenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: row.payload.schemaVersion,
                    matrixID: row.payload.matrixID,
                    cellID: row.payload.cellID,
                    identity: row.payload.identity,
                    referenceArtifactSHA256:
                        row.payload.referenceArtifactSHA256,
                    promptContentHash: row.payload.promptContentHash,
                    runConfiguration: row.payload.runConfiguration,
                    tokenization: TaskCoherenceTokenizationEvidence(
                        tokenizerManifestSHA256:
                            String(repeating: "c", count: 64),
                        promptTokenIDsSHA256:
                            row.payload.tokenization.promptTokenIDsSHA256,
                        restrictedChoiceLabelTokenIDs:
                            row.payload.tokenization
                                .restrictedChoiceLabelTokenIDs),
                    layout: row.payload.layout,
                    generatedTokenCount: row.payload.generatedTokenCount,
                    scoredOutput: row.payload.scoredOutput,
                    outputSHA256: row.payload.outputSHA256,
                    score: row.payload.score,
                    engagement: row.payload.engagement))
        }
        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)

        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus))
    }

    func testCandidateAssessmentBindsExactPromptLayout() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        for index in rows.indices {
            let row = rows[index]
            rows[index] = ResultRecord(
                subcommand: row.subcommand,
                provenance: row.provenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: row.payload.schemaVersion,
                    matrixID: row.payload.matrixID,
                    cellID: row.payload.cellID,
                    identity: row.payload.identity,
                    referenceArtifactSHA256:
                        row.payload.referenceArtifactSHA256,
                    promptContentHash: row.payload.promptContentHash,
                    runConfiguration: row.payload.runConfiguration,
                    tokenization: row.payload.tokenization,
                    layout: TaskCoherencePromptLayoutEvidence(
                        promptTokens: 512,
                        materialStartToken: 128,
                        materialEndToken: 384,
                        compressedRegionEndToken: 512,
                        minimumCompletedTileCount: 2),
                    generatedTokenCount: row.payload.generatedTokenCount,
                    scoredOutput: row.payload.scoredOutput,
                    outputSHA256: row.payload.outputSHA256,
                    score: row.payload.score,
                    engagement: row.payload.engagement))
        }
        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)

        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus))
    }

    func testCandidateAssessmentBindsExactRunConfiguration() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        for index in rows.indices {
            let row = rows[index]
            rows[index] = ResultRecord(
                subcommand: row.subcommand,
                provenance: row.provenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: row.payload.schemaVersion,
                    matrixID: row.payload.matrixID,
                    cellID: row.payload.cellID,
                    identity: row.payload.identity,
                    referenceArtifactSHA256:
                        row.payload.referenceArtifactSHA256,
                    promptContentHash: row.payload.promptContentHash,
                    runConfiguration:
                        TaskCoherenceRunConfiguration.qualificationV2(
                            structuredToolMaxTokens: 512),
                    tokenization: row.payload.tokenization,
                    layout: row.payload.layout,
                    generatedTokenCount: row.payload.generatedTokenCount,
                    scoredOutput: row.payload.scoredOutput,
                    outputSHA256: row.payload.outputSHA256,
                    score: row.payload.score,
                    engagement: row.payload.engagement))
        }
        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)

        XCTAssertThrowsError(try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus))
    }

    func testUnknownTierAndOverflowingKVarNEvidenceFailClosed() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        for tier in ["affine-invented", "kvtuner-unbound"] {
            let rows = try makeRows(
                corpus: corpus, tier: tier, cellID: tier,
                referenceArtifactSHA256: reference.artifactSHA256)
            XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
                artifactData(rows), corpus: corpus))
        }
        let wrongCell = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: "affine-k4v2-g128-invented",
            referenceArtifactSHA256: reference.artifactSHA256)
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(wrongCell), corpus: corpus))

        var rows = try makeRows(
            corpus: corpus, tier: "kvarn-k4v2-g128",
            cellID: "kvarn-k4v2-g128-i8",
            referenceArtifactSHA256: reference.artifactSHA256)
        let first = rows[0]
        rows[0] = ResultRecord(
            subcommand: first.subcommand,
            provenance: first.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: first.payload.schemaVersion,
                matrixID: first.payload.matrixID,
                cellID: first.payload.cellID,
                identity: first.payload.identity,
                referenceArtifactSHA256:
                    first.payload.referenceArtifactSHA256,
                promptContentHash: first.payload.promptContentHash,
                runConfiguration: first.payload.runConfiguration,
                tokenization: first.payload.tokenization,
                layout: first.payload.layout,
                generatedTokenCount: first.payload.generatedTokenCount,
                scoredOutput: first.payload.scoredOutput,
                outputSHA256: first.payload.outputSHA256,
                score: first.payload.score,
                engagement: TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: Int.max, affineTokens: nil,
                    kvarnCompletedTileCount: Int.max,
                    kvarnCompressedTokens: Int.max,
                    kvarnCodecIterations: 8,
                    kvarnExecutionMode: "uncompiled-correctness")))
        XCTAssertThrowsError(try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus))
    }

    func testEmptyStructuredOutputsRemainAdjudicableRejectionEvidence() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = try summary(
            corpus: corpus, tier: "fp16", cellID: "fp16",
            referenceDigest: nil)
        var rows = try makeRows(
            corpus: corpus, tier: candidateCellID,
            cellID: candidateCellID,
            referenceArtifactSHA256: reference.artifactSHA256)
        for index in rows.indices where
            rows[index].payload.score.domain == .structuredTool
        {
            let row = rows[index]
            rows[index] = ResultRecord(
                subcommand: row.subcommand,
                provenance: row.provenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: row.payload.schemaVersion,
                    matrixID: row.payload.matrixID,
                    cellID: row.payload.cellID,
                    identity: row.payload.identity,
                    referenceArtifactSHA256:
                        row.payload.referenceArtifactSHA256,
                    promptContentHash: row.payload.promptContentHash,
                    runConfiguration: row.payload.runConfiguration,
                    tokenization: row.payload.tokenization,
                    layout: row.payload.layout,
                    generatedTokenCount:
                        row.payload.generatedTokenCount,
                    scoredOutput: "",
                    outputSHA256: sha256Hex(Data()),
                    score: TaskItemScore(
                        itemID: row.payload.score.itemID,
                        domain: .structuredTool,
                        correct: false,
                        syntacticallyValid: false),
                    engagement: row.payload.engagement))
        }
        let candidate = try TaskCoherenceArtifact.summarize(
            artifactData(rows), corpus: corpus)
        let evidence = try TaskCoherencePromotionEvidence.derive(
            candidate: candidate, reference: reference, corpus: corpus)
        XCTAssertFalse(evidence.assessment.hardFloorPassed)
        XCTAssertEqual(evidence.assessment.structuredValidity.rate, 0)
    }

    private func summary(
        corpus: TaskCoherenceCorpus,
        tier: String,
        cellID: String,
        referenceDigest: String?,
        nonce: String? = nil,
        kvtunerSchedule: KVTunerScheduleBinding? = nil,
        modelConfigHashOverride: String? = nil,
        modelCheckpointManifestHashOverride: String? = nil,
        modelCheckpointContentSHA256Override: String? = nil
    ) throws -> TaskCoherenceArtifactSummary {
        let compressedBinding = try tier.hasPrefix("kvtuner-")
            && kvtunerSchedule != nil
            ? compressedAttentionBinding(
                request: .materialize,
                observedOperation: .materializedKV)
            : nil
        return try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: tier, cellID: cellID,
                referenceArtifactSHA256: referenceDigest,
                nonce: nonce,
                kvtunerSchedule: kvtunerSchedule,
                modelConfigHashOverride: modelConfigHashOverride,
                modelCheckpointManifestHashOverride:
                    modelCheckpointManifestHashOverride,
                modelCheckpointContentSHA256Override:
                    modelCheckpointContentSHA256Override,
                schemaVersion: compressedBinding == nil
                    && modelCheckpointContentSHA256Override == nil
                    ? TaskCoherenceArtifact.schemaVersion
                    : TaskCoherenceArtifact
                        .compressedAttentionSchemaVersion,
                compressedKVAttention: compressedBinding)),
            corpus: corpus)
    }

    private func artifactData(
        _ rows: [ResultRecord<TaskCoherenceCasePayload>]
    ) throws -> Data {
        Data((try rows.map { try $0.jsonLine() }
            .joined(separator: "\n") + "\n").utf8)
    }

    private func makeRows(
        corpus: TaskCoherenceCorpus,
        tier: String,
        cellID: String,
        referenceArtifactSHA256: String?,
        nonce: String? = nil,
        kvtunerSchedule: KVTunerScheduleBinding? = nil,
        modelConfigHashOverride: String? = nil,
        modelCheckpointManifestHashOverride: String? = nil,
        modelCheckpointContentSHA256Override: String? = nil,
        schemaVersion: Int = TaskCoherenceArtifact.schemaVersion,
        compressedKVAttention:
            CompressedKVAttentionRuntimeBinding? = nil
    ) throws -> [ResultRecord<TaskCoherenceCasePayload>] {
        let runModelConfigHash = compressedKVAttention?.admission.modelConfigHash
            ?? kvtunerSchedule?.modelConfigHash
            ?? modelConfigHashOverride
            ?? self.modelConfigHash
        let runCheckpointManifestHash = compressedKVAttention?
            .admission.checkpointManifestHash
            ?? kvtunerSchedule?.checkpointManifestHash
            ?? modelCheckpointManifestHashOverride
            ?? self.checkpointManifestHash
        let runTokenizerSHA256 = compressedKVAttention?.admission.tokenizerSHA256
            ?? kvtunerSchedule?.tokenizerSHA256
            ?? String(repeating: "b", count: 64)
        let identity = TaskCoherenceRunIdentity(
            corpusID: corpus.id,
            corpusContentHash: corpus.contentHash,
            modelConfigHash: runModelConfigHash,
            modelCheckpointManifestHash: runCheckpointManifestHash,
            modelCheckpointContentSHA256:
                modelCheckpointContentSHA256Override
                ?? compressedKVAttention?.admission.checkpointContentSHA256,
            kvQuantTier: tier,
            kvtunerSchedule: kvtunerSchedule)
        let runProvenance = provenance(
            corpus: corpus, tier: tier,
            nonce: nonce ?? "task-run-(cellID)",
            modelConfigHash: runModelConfigHash,
            modelCheckpointManifestHash: runCheckpointManifestHash)
        return corpus.items.map { item in
            let structured = item.domain == .structuredTool
            let generatedTokenCount = structured ? 24 : 1
            // The compiled decoder keeps one submitted lookahead in KV, so after emitting N
            // tokens its exact post-run cache offset is prompt + N.
            let expectedCachedTokens = 512 + generatedTokenCount
            let scoredOutput: String
            if let expectedTool = item.expectedTool {
                scoredOutput = String(
                    data: try! JSONEncoder().encode(expectedTool),
                    encoding: .utf8)!
            } else {
                scoredOutput = item.expectedChoice!
            }
            let engagement: TaskCoherenceCacheEngagementEvidence
            let restricted = item.scoringMode == .restrictedChoice
            if tier == "fp16" {
                engagement = TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: nil, affineTokens: nil,
                    kvarnCompletedTileCount: nil,
                    kvarnCompressedTokens: nil,
                    kvarnCodecIterations: nil,
                    kvarnExecutionMode: nil,
                    scoringCachedTokens: nil,
                    scoringKVarNCompletedTileCount: nil,
                    scoringKVarNCompressedTokens: nil)
            } else if tier.hasPrefix("kvarn-") {
                engagement = TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: expectedCachedTokens, affineTokens: nil,
                    kvarnCompletedTileCount: 3,
                    kvarnCompressedTokens: 384,
                    kvarnCodecIterations:
                        tier.hasSuffix("-i16") ? 16 : 8,
                    kvarnExecutionMode: "uncompiled-correctness",
                    scoringCachedTokens: restricted ? 512 : nil,
                    scoringKVarNCompletedTileCount: restricted ? 3 : nil,
                    scoringKVarNCompressedTokens: restricted ? 384 : nil)
            } else if tier.hasPrefix("kvtuner-") {
                let layerCount = kvtunerSchedule?.layers.count ?? 2
                engagement = TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: expectedCachedTokens,
                    affineTokens: nil,
                    kvtunerTokens: expectedCachedTokens,
                    kvtunerLayerCount: layerCount,
                    kvarnCompletedTileCount: nil,
                    kvarnCompressedTokens: nil,
                    kvarnCodecIterations: nil,
                    kvarnExecutionMode: nil,
                    scoringCachedTokens: restricted ? 512 : nil,
                    scoringKVTunerLayerCount:
                        restricted ? layerCount : nil,
                    scoringKVarNCompletedTileCount: nil,
                    scoringKVarNCompressedTokens: nil)
            } else {
                engagement = TaskCoherenceCacheEngagementEvidence(
                    cachedTokens: expectedCachedTokens,
                    affineTokens: expectedCachedTokens,
                    kvarnCompletedTileCount: nil,
                    kvarnCompressedTokens: nil,
                    kvarnCodecIterations: nil,
                    kvarnExecutionMode: nil,
                    scoringCachedTokens: restricted ? 512 : nil,
                    scoringKVarNCompletedTileCount: nil,
                    scoringKVarNCompressedTokens: nil)
            }
            return ResultRecord(
                subcommand: "task-coherence",
                provenance: runProvenance,
                payload: TaskCoherenceCasePayload(
                    schemaVersion: schemaVersion,
                    matrixID: matrixID,
                    cellID: cellID,
                    identity: identity,
                    referenceArtifactSHA256: referenceArtifactSHA256,
                    promptContentHash: fnv1a64(item.prompt.utf8),
                    runConfiguration:
                        TaskCoherenceRunConfiguration.qualificationV2(
                            structuredToolMaxTokens: 96),
                    tokenization: TaskCoherenceTokenizationEvidence(
                        tokenizerManifestSHA256: runTokenizerSHA256,
                        promptTokenIDsSHA256: taskTokenIDsSHA256(
                            Array(0 ..< 512)),
                        restrictedChoiceLabelTokenIDs: structured
                            ? nil
                            : ["A": 101, "B": 202, "C": 303, "D": 404]),
                    layout: TaskCoherencePromptLayoutEvidence(
                        promptTokens: 512,
                        materialStartToken: 128,
                        materialEndToken: 256,
                        compressedRegionEndToken: 512,
                        minimumCompletedTileCount: 1),
                    generatedTokenCount: generatedTokenCount,
                    scoredOutput: scoredOutput,
                    outputSHA256: sha256Hex(Data(scoredOutput.utf8)),
                    score: TaskItemScore(
                        itemID: item.id, domain: item.domain,
                        correct: true,
                        syntacticallyValid: structured ? true : nil),
                    engagement: engagement,
                    compressedKVAttention: compressedKVAttention))
        }
    }

    private func replacing(
        _ record: ResultRecord<TaskCoherenceCasePayload>,
        identity: TaskCoherenceRunIdentity? = nil,
        engagement: TaskCoherenceCacheEngagementEvidence? = nil
    ) -> ResultRecord<TaskCoherenceCasePayload> {
        let payload = record.payload
        return ResultRecord(
            subcommand: record.subcommand,
            provenance: record.provenance,
            payload: TaskCoherenceCasePayload(
                schemaVersion: payload.schemaVersion,
                matrixID: payload.matrixID,
                cellID: payload.cellID,
                identity: identity ?? payload.identity,
                referenceArtifactSHA256:
                    payload.referenceArtifactSHA256,
                promptContentHash: payload.promptContentHash,
                runConfiguration: payload.runConfiguration,
                tokenization: payload.tokenization,
                layout: payload.layout,
                generatedTokenCount: payload.generatedTokenCount,
                scoredOutput: payload.scoredOutput,
                outputSHA256: payload.outputSHA256,
                score: payload.score,
                engagement: engagement ?? payload.engagement,
                compressedKVAttention: payload.compressedKVAttention))
    }

    private func kvtunerEvaluationCorpus(
        _ corpus: TaskCoherenceCorpus
    ) throws -> KVTunerEvaluationCorpusIdentity {
        try KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(corpus)
    }

    private func compressedAttentionAdmission() throws
        -> CompressedKVAttentionRuntimeAdmission
    {
        let config = Data(
            #"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":8192,"num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false}"#.utf8)
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: config,
                checkpointManifestHash: checkpointManifestHash,
                checkpointContentSHA256: String(repeating: "d", count: 64),
                tokenizerSHA256: String(repeating: "b", count: 64)))
    }

    private func compressedAttentionBinding(
        request: CompressedKVAttentionRequest = .splitAffineQuantizedMM,
        observedOperation: CompressedKVAttentionObservedOperation =
            .splitQuantizedMM
    ) throws -> CompressedKVAttentionRuntimeBinding {
        try CompressedKVAttentionRuntimeBinding(
            request: request,
            observedOperation: observedOperation,
            admission: compressedAttentionAdmission())
    }

    private func kvtunerBinding(
        corpus: TaskCoherenceCorpus,
        modelConfigHash: String? = nil,
        modelConfigSHA256: String? = nil,
        checkpointManifestHash: String? = nil,
        checkpointContentSHA256: String? = nil,
        tokenizerSHA256: String? = nil,
        layerCount: Int = 64,
        searchArtifactSHA256: String = String(repeating: "d", count: 64),
        evaluationCorpora: [KVTunerEvaluationCorpusIdentity]? = nil
    ) throws -> KVTunerScheduleBinding {
        let evaluation = try kvtunerEvaluationCorpus(corpus)
        let admission = try compressedAttentionAdmission()
        let scheduleModelConfigHash = modelConfigHash
            ?? admission.modelConfigHash
        let scheduleModelConfigSHA256 = modelConfigSHA256
            ?? admission.modelConfigSHA256
        let scheduleCheckpointHash = checkpointManifestHash
            ?? admission.checkpointManifestHash
        let scheduleCheckpointContentSHA256 = checkpointContentSHA256
            ?? admission.checkpointContentSHA256
        let scheduleTokenizerSHA256 = tokenizerSHA256
            ?? admission.tokenizerSHA256
        let schedule = KVTunerSchedule(
            schemaVersion: 4,
            matrixID: matrixID,
            cellID: kvtunerCellID,
            modelConfigHash: scheduleModelConfigHash,
            modelConfigSHA256: scheduleModelConfigSHA256,
            checkpointManifestHash: scheduleCheckpointHash,
            checkpointContentSHA256:
                scheduleCheckpointContentSHA256,
            tokenizerSHA256: scheduleTokenizerSHA256,
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
                repeating: "c", count: 64),
            sourceSearchArtifactSHA256: searchArtifactSHA256,
            layers: (0 ..< layerCount).map {
                KVLayerPrecision(
                    layer: $0,
                    keyBits: $0.isMultiple(of: 2) ? 8 : 4,
                    valueBits: $0.isMultiple(of: 2) ? 4 : 2)
            })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let selection = try KVTunerRuntimeSelection.loadForTesting(
            artifactData: encoder.encode(schedule),
            expectedLayerCount: layerCount,
            expectedMatrixID: matrixID,
            expectedCellID: kvtunerCellID,
            expectedModelConfigHash: scheduleModelConfigHash,
            expectedModelConfigSHA256: scheduleModelConfigSHA256,
            expectedCheckpointManifestHash: scheduleCheckpointHash,
            expectedCheckpointContentSHA256:
                scheduleCheckpointContentSHA256,
            evaluationCorpora: evaluationCorpora ?? [evaluation])
        return KVTunerScheduleBinding(selection: selection)
    }

    private func kvtunerKLEvaluationCorpus()
        throws -> KVTunerEvaluationCorpusIdentity
    {
        try KVTunerEvaluationCorpusIdentity(
            id: "measurement-corpus-v2",
            aggregateDigest: "4444444444444444",
            canonicalEntryDigests: [
                "5555555555555555",
                "6666666666666666",
            ],
            canonicalSourceItemDigests: [
                sha256Hex(Data("kl-evaluation-source".utf8))
            ])
    }

    private func provenance(
        corpus: TaskCoherenceCorpus,
        tier: String,
        nonce: String,
        gitSHA: String? = nil,
        modelConfigHash: String? = nil,
        modelCheckpointManifestHash: String? = nil
    ) -> Provenance {
        Provenance(
            date: "2026-07-15T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            hardwareOS: "macOS 15.5",
            harnessGitSHA: gitSHA ?? cleanSHA,
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: "0.32.0",
            referenceMLXLMVersion: "0.29.0",
            modelPath: "/models/qwen3-32b",
            modelConfigHash: modelConfigHash ?? self.modelConfigHash,
            modelCheckpointManifestHash:
                modelCheckpointManifestHash ?? checkpointManifestHash,
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: corpus.id,
            corpusContentHash: corpus.contentHash,
            nonce: nonce)
    }

    private func validKLRecord(
        cellID: String? = nil
    ) throws -> ResultRecord<KLPayload> {
        let format = KVFormatGeometryEvidence(
            kind: .affine, tier: candidateCellID,
            keyBits: 4, valueBits: 2, groupSize: 128,
            sinkTokens: 0, layerCount: 64, kvHeadCount: 8,
            headDimension: 128, capacityTokens: 24_192,
            sequences: 1, metadataScalarBytes: 2,
            recordAlignment: 1)
        let tokenHeads = 24_192 * 64 * 8
        let actual = KVStorageBreakdownEvidence(
            payloadBytes: tokenHeads * 96,
            metadataBytes: tokenHeads * 8,
            alignmentPaddingBytes: 0, fp16SinkBytes: 0,
            fp16TailBytes: 0, workspaceBytes: 0,
            totalBytes: tokenHeads * 104)
        let frontier = KVFrontierEvidence(
            schemaVersion: 1, matrixID: matrixID,
            cellID: cellID ?? candidateCellID,
            sameWeights: true,
            comparisonBaseline: .sameWeightsFP16KV,
            referenceKVQuantTier: "fp16",
            candidateModel: KVModelEvidenceIdentity(
                configHash: modelConfigHash,
                checkpointManifestHash: checkpointManifestHash),
            referenceModel: KVModelEvidenceIdentity(
                configHash: modelConfigHash,
                checkpointManifestHash: checkpointManifestHash),
            candidateFormat: format,
            storage: KVStorageEvidence(predicted: actual, actual: actual),
            actualControlBytes: 256,
            candidateExecutionMode: nil,
            candidateCodecIterations: nil,
            candidateMemoryGate: nil)
        let payload = KLPayload(
            kvQuantTier: candidateCellID,
            klMedianNats: 0.04,
            klLongContextTailP95Nats: 0.3,
            klPooledMedianNats: 0.05,
            klPooledP95Nats: 0.4,
            pplCandidate: 10.2, pplReference: 10.0,
            pplDeltaPct: 2.0, totalPositions: 200,
            entryCount: 4,
            teacherForcedTop1AgreementCount: 160,
            teacherForcedTop1ScoredPositions: 200,
            teacherForcedTop1AgreementRate: 0.8,
            frontier: frontier,
            shortEntryCount: 3, shortScoredPositions: 72,
            longContextEntryCount: 1,
            longContextScoredPositions: 128,
            shortEntryScoring: [
                KVEntryScoringEvidence(
                    entryID: "short-0", scoredPositions: 24),
                KVEntryScoringEvidence(
                    entryID: "short-1", scoredPositions: 24),
                KVEntryScoringEvidence(
                    entryID: "short-2", scoredPositions: 24),
            ],
            longContextEntryScoring: [
                KVEntryScoringEvidence(
                    entryID: "long-0", scoredPositions: 128),
            ],
            longContextMaxDocumentTokens: 24_151,
            longContextMaxScoredContextTokens: 24_150)
        let measurementProvenance = Provenance(
            date: "2026-07-15T00:10:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            hardwareOS: "macOS 15.5", harnessGitSHA: cleanSHA,
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: "0.32.0",
            referenceMLXLMVersion: "0.29.0",
            modelPath: "/models/qwen3-32b",
            modelConfigHash: modelConfigHash,
            modelCheckpointManifestHash: checkpointManifestHash,
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: "measurement-corpus-v2",
            corpusContentHash: "measurement-corpus-hash",
            nonce: "kl-run")
        return ResultRecord(
            subcommand: "kl", provenance: measurementProvenance,
            payload: payload)
    }

    private func validKVTunerKLRecord(
        binding: KVTunerScheduleBinding
    ) throws -> ResultRecord<KLPayload> {
        let capacityTokens = 24_192
        let kvHeadCount = 8
        let headDimension = 128
        let metadataScalarBytes = 4
        let workspaceBytes = capacityTokens * kvHeadCount
            * headDimension * metadataScalarBytes
        let format = KVFormatGeometryEvidence(
            kind: .kvtuner,
            tier: binding.cellID,
            keyBits: 0,
            valueBits: 0,
            groupSize: binding.groupSize,
            sinkTokens: 0,
            layerCount: binding.layers.count,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            capacityTokens: capacityTokens,
            sequences: 1,
            metadataScalarBytes: metadataScalarBytes,
            recordAlignment: 1)
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
                kvHeadCount: kvHeadCount,
                headDimension: headDimension),
            capacityTokens: capacityTokens,
            sequences: 1,
            metadataScalarBytes: metadataScalarBytes,
            maximumLayerWorkspaceBytes: workspaceBytes)
        let actual = KVStorageBreakdownEvidence(
            payloadBytes: allocation.payloadBytes,
            metadataBytes: allocation.metadataBytes,
            alignmentPaddingBytes: 0,
            fp16SinkBytes: 0,
            fp16TailBytes: 0,
            workspaceBytes: allocation.workspaceBytes,
            totalBytes: allocation.totalBytes - allocation.controlBytes)
        let storage = try format.storageEvidence(
            actual: actual,
            kvtunerSchedule: binding)
        let model = KVModelEvidenceIdentity(
            configHash: binding.modelConfigHash,
            checkpointManifestHash: binding.checkpointManifestHash,
            checkpointContentSHA256:
                binding.checkpointContentSHA256)
        let compressedBinding = try CompressedKVAttentionRuntimeBinding(
            request: .materialize,
            observedOperation: .materializedKV,
            admission: compressedAttentionAdmission())
        let frontier = KVFrontierEvidence(
            schemaVersion: 2,
            matrixID: binding.matrixID,
            cellID: binding.cellID,
            sameWeights: true,
            comparisonBaseline: .sameWeightsFP16KV,
            referenceKVQuantTier: "fp16",
            candidateModel: model,
            referenceModel: model,
            candidateFormat: format,
            storage: storage,
            actualControlBytes: allocation.controlBytes,
            candidateExecutionMode: nil,
            candidateCodecIterations: nil,
            candidateMemoryGate: nil,
            candidateKVTunerSchedule: binding,
            candidateCompressedKVAttention: compressedBinding,
            candidateMaterializationWorkspaceBytes:
                allocation.workspaceBytes,
            candidateAttentionWorkspaceBytes: 0)
        let template = try validKLRecord().payload
        let payload = KLPayload(
            kvQuantTier: binding.cellID,
            klMedianNats: template.klMedianNats,
            klLongContextTailP95Nats:
                template.klLongContextTailP95Nats,
            klPooledMedianNats: template.klPooledMedianNats,
            klPooledP95Nats: template.klPooledP95Nats,
            pplCandidate: template.pplCandidate,
            pplReference: template.pplReference,
            pplDeltaPct: template.pplDeltaPct,
            totalPositions: template.totalPositions,
            entryCount: template.entryCount,
            teacherForcedTop1AgreementCount:
                template.teacherForcedTop1AgreementCount,
            teacherForcedTop1ScoredPositions:
                template.teacherForcedTop1ScoredPositions,
            teacherForcedTop1AgreementRate:
                template.teacherForcedTop1AgreementRate,
            frontier: frontier,
            shortEntryCount: template.shortEntryCount,
            shortScoredPositions: template.shortScoredPositions,
            longContextEntryCount: template.longContextEntryCount,
            longContextScoredPositions:
                template.longContextScoredPositions,
            shortEntryScoring: template.shortEntryScoring,
            longContextEntryScoring: template.longContextEntryScoring,
            longContextMaxDocumentTokens:
                template.longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens:
                template.longContextMaxScoredContextTokens)
        let evaluation = try kvtunerKLEvaluationCorpus()
        let provenance = Provenance(
            date: "2026-07-15T00:10:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 * 1_024 * 1_024 * 1_024,
            hardwareOS: "macOS 15.5",
            harnessGitSHA: cleanSHA,
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: "0.32.0",
            referenceMLXLMVersion: "0.29.0",
            modelPath: "/models/qwen3-32b",
            modelConfigHash: binding.modelConfigHash,
            modelCheckpointManifestHash:
                binding.checkpointManifestHash,
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: evaluation.id,
            corpusContentHash: evaluation.aggregateDigest,
            nonce: "kvtuner-kl-run")
        return ResultRecord(
            subcommand: "kl", provenance: provenance,
            payload: payload)
    }
}
