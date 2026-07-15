import Foundation
import XCTest
@testable import HarnessCore

final class KVFrontierEvidenceTests: XCTestCase {
    private func identity(_ suffix: String = "same") -> KVModelEvidenceIdentity {
        KVModelEvidenceIdentity(
            configHash: "config-\(suffix)",
            checkpointManifestHash: "checkpoint-\(suffix)")
    }

    private func geometry(tier: String = "affine-k4v2-g128") -> KVFormatGeometryEvidence {
        KVFormatGeometryEvidence(
            kind: .affine, tier: tier, keyBits: 4, valueBits: 2, groupSize: 128,
            sinkTokens: 0, layerCount: 64, kvHeadCount: 8, headDimension: 128,
            capacityTokens: 4_096, sequences: 1, metadataScalarBytes: 2,
            recordAlignment: 1)
    }

    private func breakdown(total: Int = 218_103_808) -> KVStorageBreakdownEvidence {
        KVStorageBreakdownEvidence(
            payloadBytes: 201_326_592, metadataBytes: 16_777_216,
            alignmentPaddingBytes: 0, fp16SinkBytes: 0, fp16TailBytes: 0,
            workspaceBytes: 0, totalBytes: total)
    }

    private func frontier(
        sameWeights: Bool = true,
        baseline: KVComparisonBaseline = .sameWeightsFP16KV,
        candidate: KVModelEvidenceIdentity? = nil,
        reference: KVModelEvidenceIdentity? = nil,
        matrixID: String = "kvarn-qwen3-32b-v1",
        cellID: String = "affine-k4v2-g128",
        format: KVFormatGeometryEvidence? = nil,
        storage: KVStorageEvidence? = nil,
        controlBytes: Int? = 256
    ) -> KVFrontierEvidence {
        let same = identity()
        let bytes = breakdown()
        return KVFrontierEvidence(
            schemaVersion: 1, matrixID: matrixID, cellID: cellID,
            sameWeights: sameWeights, comparisonBaseline: baseline,
            referenceKVQuantTier: "fp16",
            candidateModel: candidate ?? same,
            referenceModel: reference ?? same,
            candidateFormat: format ?? geometry(),
            storage: storage ?? KVStorageEvidence(predicted: bytes, actual: bytes),
            actualControlBytes: controlBytes)
    }

    private func payload(frontier: KVFrontierEvidence? = nil) -> KLPayload {
        KLPayload(
            kvQuantTier: "affine-k4v2-g128",
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
        XCTAssertEqual(decoded.frontier?.storage?.actual.totalBytes, 218_103_808)
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
        corpusHash: String? = "corpus-hash"
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
            corpusId: corpusHash == nil ? nil : "measurement-corpus-v2",
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
