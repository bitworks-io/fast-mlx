import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38MTPLiveExactnessGateTests: XCTestCase {
    private typealias Gate = Qwen38MTPLiveExactnessGate
    private typealias GateError = Qwen38MTPLiveExactnessGateError

    func testCanonicalTwoCaseRecordReturnsDerivedProofWithoutPromotingScorecardAuthority() throws {
        let proof = try Gate.validateJSONL(try recordData())

        XCTAssertEqual(proof.artifact, Qwen38MTPPerformanceScorecardGate.requiredArtifact)
        XCTAssertEqual(proof.artifactID, Gate.requiredArtifactID)
        XCTAssertEqual(proof.sourceID, Gate.requiredSourceIdentity.sourceID)
        XCTAssertTrue(proof.accepted)
        XCTAssertEqual(proof.evidenceID.count, 64)
        XCTAssertTrue(proof.evidenceID.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        XCTAssertNil(Qwen38MTPPerformanceScorecardGate.requiredAcceptedLiveExactnessProof)

        let second = try Gate.validateJSONL(try recordData())
        XCTAssertEqual(second, proof)

        let otherRun = try Gate.validateJSONL(try recordData(
            provenance: provenance(
                date: "2026-08-25T00:00:01Z",
                harnessGitSHA: hex40("f"))))
        XCTAssertNotEqual(
            otherRun.evidenceID,
            proof.evidenceID,
            "the proof identity must bind the complete canonical record, including provenance")
    }

    func testRejectsNonCanonicalMalformedWrongSubcommandAndWrongProvenanceJSONL() throws {
        XCTAssertThrowsError(try Gate.validateJSONL(Data())) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try evidenceRecord().jsonLine()).utf8))) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }
        XCTAssertThrowsError(try Gate.validateJSONL(try recordData() + recordData())) { error in
            XCTAssertEqual(error as? GateError, .invalidRecordCardinality(2))
        }

        var wrongSubcommand = evidenceRecord(subcommand: "qwen38-mtp-performance-scorecard")
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try wrongSubcommand.jsonLine() + "\n").utf8))) {
            error in
            XCTAssertEqual(error as? GateError, .wrongSubcommand("qwen38-mtp-performance-scorecard"))
        }

        let nonCanonical = Data(try JSONEncoder().encode(evidenceRecord())) + Data("\n".utf8)
        XCTAssertThrowsError(try Gate.validateJSONL(nonCanonical)) { error in
            XCTAssertEqual(error as? GateError, .nonCanonicalJSONL)
        }

        wrongSubcommand = evidenceRecord(
            provenance: provenance(modelPath: "sensitive-marker/operator/model"))
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try wrongSubcommand.jsonLine() + "\n").utf8))) {
            error in
            XCTAssertEqual(error as? GateError, .invalidProvenance("modelPath"))
        }
    }

    func testRejectsArtifactSourceCaseTokenDecodeDraftPassthroughAndCacheDrift() throws {
        var wrongArtifact = evidence()
        wrongArtifact.artifactID = hex("A")
        XCTAssertThrowsError(try validate(wrongArtifact)) { error in
            XCTAssertEqual(error as? GateError, .invalidArtifactIdentity)
        }

        var wrongSelection = evidence()
        wrongSelection.source.selection = "qwen35_9BDepth1"
        XCTAssertThrowsError(try validate(wrongSelection)) { error in
            XCTAssertEqual(error as? GateError, .invalidSourceIdentity)
        }

        var reordered = evidence()
        reordered.cases.swapAt(0, 1)
        XCTAssertThrowsError(try validate(reordered)) { error in
            XCTAssertEqual(error as? GateError, .invalidCaseOrder)
        }

        var duplicate = evidence()
        duplicate.cases[1].id = "numbers"
        XCTAssertThrowsError(try validate(duplicate)) { error in
            XCTAssertEqual(error as? GateError, .invalidCaseOrder)
        }

        var emptyTokens = evidence()
        emptyTokens.cases[0].mtpTokenIDs = []
        XCTAssertThrowsError(try validate(emptyTokens)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "tokens"))
        }

        var tokenDrift = evidence()
        tokenDrift.cases[0].mtpTokenIDs[0] += 1
        XCTAssertThrowsError(try validate(tokenDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "tokens"))
        }

        var decodedDrift = evidence()
        decodedDrift.cases[0].mtpDecodedUTF8Base64 = Data("different".utf8).base64EncodedString()
        XCTAssertThrowsError(try validate(decodedDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "decoded"))
        }

        var forgedDigest = evidence()
        forgedDigest.cases[0].decodedUTF8SHA256 = hex("B")
        XCTAssertThrowsError(try validate(forgedDigest)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "decoded"))
        }

        var noDraft = evidence()
        noDraft.cases[0].proposedDraftTokens = 0
        XCTAssertThrowsError(try validate(noDraft)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "drafts"))
        }

        var passthrough = evidence()
        passthrough.cases[0].passthroughReason = "private fallback"
        XCTAssertThrowsError(try validate(passthrough)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "passthrough"))
        }

        var cacheDrift = evidence()
        cacheDrift.cases[0].mtpCacheFingerprints[0].stateFingerprints[0].sha256 = hex("C")
        XCTAssertThrowsError(try validate(cacheDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var emptyCache = evidence()
        emptyCache.cases[0].scalarCacheFingerprints = []
        XCTAssertThrowsError(try validate(emptyCache)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var truncatedCacheState = evidence()
        truncatedCacheState.cases[0].scalarCacheFingerprints[0].stateFingerprints.removeLast()
        truncatedCacheState.cases[0].mtpCacheFingerprints[0].stateFingerprints.removeLast()
        XCTAssertThrowsError(try validate(truncatedCacheState)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongHybridTopology = evidence()
        wrongHybridTopology.cases[0].scalarCacheFingerprints[0].cacheType = "dense-attention"
        wrongHybridTopology.cases[0].mtpCacheFingerprints[0].cacheType = "dense-attention"
        XCTAssertThrowsError(try validate(wrongHybridTopology)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentOffset = evidence()
        wrongRecurrentOffset.cases[0].scalarCacheFingerprints[0].offset = 1
        wrongRecurrentOffset.cases[0].mtpCacheFingerprints[0].offset = 1
        XCTAssertThrowsError(try validate(wrongRecurrentOffset)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentShape = evidence()
        wrongRecurrentShape.cases[0].scalarCacheFingerprints[0]
            .stateFingerprints[0].shape = [1, 3, 10_239]
        wrongRecurrentShape.cases[0].mtpCacheFingerprints[0]
            .stateFingerprints[0].shape = [1, 3, 10_239]
        XCTAssertThrowsError(try validate(wrongRecurrentShape)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentDType = evidence()
        wrongRecurrentDType.cases[0].scalarCacheFingerprints[0]
            .stateFingerprints[1].dtype = "bfloat16"
        wrongRecurrentDType.cases[0].mtpCacheFingerprints[0]
            .stateFingerprints[1].dtype = "bfloat16"
        XCTAssertThrowsError(try validate(wrongRecurrentDType)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongDenseGeometry = evidence()
        wrongDenseGeometry.cases[0].scalarCacheFingerprints[3]
            .stateFingerprints[0].shape = [1, 4, 14, 255]
        wrongDenseGeometry.cases[0].mtpCacheFingerprints[3]
            .stateFingerprints[0].shape = [1, 4, 14, 255]
        XCTAssertThrowsError(try validate(wrongDenseGeometry)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongDenseByteCount = evidence()
        wrongDenseByteCount.cases[0].scalarCacheFingerprints[3]
            .stateFingerprints[0].byteCount -= 2
        wrongDenseByteCount.cases[0].mtpCacheFingerprints[3]
            .stateFingerprints[0].byteCount -= 2
        XCTAssertThrowsError(try validate(wrongDenseByteCount)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var promptDrift = evidence()
        promptDrift.cases[0].promptSHA256 = hex("e")
        XCTAssertThrowsError(try validate(promptDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "workload"))
        }
    }

    func testRejectsIncompleteOrUnboundRunProvenance() throws {
        let invalidRows: [Provenance] = [
            provenance(harnessGitSHA: String(repeating: "0", count: 40)),
            provenance(harnessGitSHA: hex40("e") + "-dirty"),
            provenance(hardwareRAMBytes: 24 * 1024 * 1024 * 1024),
            provenance(hardwareChip: ""),
            provenance(hardwareOS: ""),
            provenance(mlxSwiftVersion: "unknown"),
            provenance(date: "not-a-date"),
        ]
        for row in invalidRows {
            XCTAssertThrowsError(
                try Gate.validateJSONL(try recordData(provenance: row)),
                "invalid live-run provenance must never yield an accepted proof")
        }
    }

    private func validate(
        _ evidence: Qwen38MTPLiveExactnessEvidence
    ) throws -> Qwen38MTPPerformanceScorecardLiveExactnessProof {
        try Gate.validateJSONL(Data((try evidenceRecord(payload: evidence).jsonLine() + "\n").utf8))
    }

    private func recordData(provenance: Provenance? = nil) throws -> Data {
        Data((try evidenceRecord(provenance: provenance).jsonLine() + "\n").utf8)
    }

    private func evidenceRecord(
        subcommand: String = Gate.subcommand,
        provenance suppliedProvenance: Provenance? = nil,
        payload suppliedPayload: Qwen38MTPLiveExactnessEvidence? = nil
    ) -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        ResultRecord(
            subcommand: subcommand,
            provenance: suppliedProvenance ?? provenance(),
            payload: suppliedPayload ?? evidence())
    }

    private func evidence() -> Qwen38MTPLiveExactnessEvidence {
        Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Gate.requiredArtifactID,
            source: Gate.requiredSourceIdentity,
            cases: [
                caseEvidence(id: "numbers", text: " 13, 17", tokenBase: 100),
                caseEvidence(id: "sentence", text: " the automated one.", tokenBase: 200),
            ])
    }

    private func caseEvidence(
        id: String,
        text: String,
        tokenBase: Int
    ) -> Qwen38MTPLiveExactnessCaseEvidence {
        let bytes = Data(text.utf8)
        let cache = (0 ..< Gate.requiredCacheLayerCount).map {
            cacheFingerprint(layer: $0, digest: hex("1"))
        }
        return Qwen38MTPLiveExactnessCaseEvidence(
            id: id,
            promptSHA256: Gate.requiredCasesByID[id]!.promptSHA256,
            maxTokens: Gate.requiredCasesByID[id]!.maxTokens,
            scalarTokenIDs: [tokenBase, tokenBase + 1],
            mtpTokenIDs: [tokenBase, tokenBase + 1],
            scalarDecodedUTF8Base64: bytes.base64EncodedString(),
            mtpDecodedUTF8Base64: bytes.base64EncodedString(),
            decodedUTF8SHA256: Gate.sha256Hex(bytes),
            proposedDraftTokens: 3,
            acceptedDraftTokens: 2,
            passthroughReason: nil,
            scalarCacheFingerprints: cache,
            mtpCacheFingerprints: cache)
    }

    private func cacheFingerprint(
        layer: Int,
        digest: String
    ) -> Qwen38MTPLiveExactnessCacheFingerprint {
        let isDense = (layer + 1).isMultiple(of: 4)
        return Qwen38MTPLiveExactnessCacheFingerprint(
            layerIndex: layer,
            cacheType: isDense ? "dense-attention" : "recurrent-mamba",
            offset: isDense ? 14 : 0,
            metaStateSHA256: hex("d"),
            stateFingerprints: (0 ..< 2).map { state in
                let shape = isDense
                    ? [1, 4, 14, 256]
                    : (state == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                let dtype = isDense || state == 0 ? "bfloat16" : "float32"
                let byteCount = shape.reduce(1, *) * (dtype == "float32" ? 4 : 2)
                return Qwen38MTPLiveExactnessArrayFingerprint(
                    stateIndex: state,
                    shape: shape,
                    dtype: dtype,
                    byteCount: byteCount,
                    sha256: digest)
            })
    }

    private func provenance(
        modelPath: String = Qwen38MTPLiveExactnessGate.modelPathSentinel,
        date: String = "2026-08-25T00:00:00Z",
        hardwareChip: String = "Apple M3 Ultra",
        hardwareRAMBytes: UInt64 = Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
        hardwareOS: String = "macOS 26.0",
        harnessGitSHA: String = String(repeating: "e", count: 40),
        mlxSwiftVersion: String = "0.31.6"
    ) -> Provenance {
        Provenance(
            date: date,
            hardwareChip: hardwareChip,
            hardwareRAMBytes: hardwareRAMBytes,
            hardwareOS: hardwareOS,
            harnessGitSHA: harnessGitSHA,
            mlxSwiftVersion: mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: modelPath,
            modelConfigHash: Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func hex40(_ character: Character) -> String {
        String(repeating: String(character), count: 40)
    }
}
