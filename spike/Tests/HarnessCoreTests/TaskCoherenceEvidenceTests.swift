import Foundation
import XCTest
@testable import HarnessCore

final class TaskCoherenceEvidenceTests: XCTestCase {
    private let matrixID = "kvarn-qwen3-32b-v1"
    private let candidateCellID = "affine-k4v2-g128"
    private let cleanSHA = String(repeating: "a", count: 40)

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
        XCTAssertNil(TaskCoherenceArtifact.expectedCellID(forTier: "tq2.5"))
        XCTAssertNil(TaskCoherenceArtifact.expectedCellID(
            forTier: "kvtuner-unbound"))
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
        nonce: String? = nil
    ) throws -> TaskCoherenceArtifactSummary {
        try TaskCoherenceArtifact.summarize(
            artifactData(try makeRows(
                corpus: corpus, tier: tier, cellID: cellID,
                referenceArtifactSHA256: referenceDigest,
                nonce: nonce)),
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
        nonce: String? = nil
    ) throws -> [ResultRecord<TaskCoherenceCasePayload>] {
        let identity = TaskCoherenceRunIdentity(
            corpusID: corpus.id,
            corpusContentHash: corpus.contentHash,
            modelConfigHash: "config-same",
            modelCheckpointManifestHash: "checkpoint-same",
            kvQuantTier: tier)
        let runProvenance = provenance(
            corpus: corpus, tier: tier,
            nonce: nonce ?? "task-run-(cellID)")
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
                    schemaVersion: TaskCoherenceArtifact.schemaVersion,
                    matrixID: matrixID,
                    cellID: cellID,
                    identity: identity,
                    referenceArtifactSHA256: referenceArtifactSHA256,
                    promptContentHash: fnv1a64(item.prompt.utf8),
                    runConfiguration:
                        TaskCoherenceRunConfiguration.qualificationV2(
                            structuredToolMaxTokens: 96),
                    tokenization: TaskCoherenceTokenizationEvidence(
                        tokenizerManifestSHA256:
                            String(repeating: "b", count: 64),
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
                    engagement: engagement))
        }
    }

    private func provenance(
        corpus: TaskCoherenceCorpus,
        tier: String,
        nonce: String,
        gitSHA: String? = nil
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
            modelConfigHash: "config-same",
            modelCheckpointManifestHash: "checkpoint-same",
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
                configHash: "config-same",
                checkpointManifestHash: "checkpoint-same"),
            referenceModel: KVModelEvidenceIdentity(
                configHash: "config-same",
                checkpointManifestHash: "checkpoint-same"),
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
            modelConfigHash: "config-same",
            modelCheckpointManifestHash: "checkpoint-same",
            modelQuant: ModelQuantInfo(bits: 4, groupSize: 64),
            corpusId: "measurement-corpus-v2",
            corpusContentHash: "measurement-corpus-hash",
            nonce: "kl-run")
        return ResultRecord(
            subcommand: "kl", provenance: measurementProvenance,
            payload: payload)
    }
}
