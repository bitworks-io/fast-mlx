import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class Qwen38MTPLiveExactnessCLITests: XCTestCase {
    private typealias Gate = Qwen38MTPLiveExactnessGate
    private typealias CLIError = Qwen38MTPLiveExactnessCLIError

    func testStrictArgumentParsingRequiresOnlyEvidencePath() throws {
        let parsed = try parseQwen38MTPLiveExactnessValidationArguments([
            "--evidence", "live.jsonl",
        ])
        XCTAssertEqual(parsed.evidencePath, "live.jsonl")

        XCTAssertThrowsError(try parseQwen38MTPLiveExactnessValidationArguments([])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--evidence"))
        }
        XCTAssertThrowsError(try parseQwen38MTPLiveExactnessValidationArguments([
            "--evidence", "live.jsonl",
            "--evidence", "other.jsonl",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .duplicateFlag("--evidence"))
        }
        XCTAssertThrowsError(try parseQwen38MTPLiveExactnessValidationArguments([
            "--evidence", "live.jsonl",
            "--authority", "authority.json",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .unknownFlag)
            XCTAssertFalse(qwen38MTPLiveExactnessExternalDiagnostic(error).contains("authority.json"))
        }
        XCTAssertThrowsError(try parseQwen38MTPLiveExactnessValidationArguments([
            "sensitive-marker/operator/live.jsonl",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .unexpectedPositional)
            XCTAssertFalse(
                qwen38MTPLiveExactnessExternalDiagnostic(error).contains("sensitive-marker"))
        }
    }

    func testValidationPrintsOnlyCanonicalJSONProofOnSuccess() throws {
        let output = try validateQwen38MTPLiveExactness(
            arguments: ["--evidence", "live.jsonl"],
            readFile: { path in
                XCTAssertEqual(path, "live.jsonl")
                return try self.recordData()
            })
        let proof = try JSONDecoder().decode(
            Qwen38MTPPerformanceScorecardLiveExactnessProof.self,
            from: Data(output.utf8))

        XCTAssertTrue(proof.accepted)
        XCTAssertEqual(proof.artifactID, Gate.requiredArtifactID)
        XCTAssertEqual(output, try canonicalProofJSON(proof))
        XCTAssertFalse(output.contains("VALID"))
        XCTAssertFalse(output.contains("live.jsonl"))
        XCTAssertFalse(output.contains("sensitive-marker"))
    }

    func testReadAndGateFailuresAreRedacted() throws {
        XCTAssertThrowsError(
            try validateQwen38MTPLiveExactness(
                arguments: ["--evidence", "sensitive-marker/operator/live.jsonl"],
                readFile: { _ in throw CocoaError(.fileReadNoSuchFile) })) { error in
            XCTAssertEqual(error as? CLIError, .fileReadFailed("--evidence"))
            XCTAssertFalse(
                qwen38MTPLiveExactnessExternalDiagnostic(error).contains("sensitive-marker"))
        }

        XCTAssertThrowsError(
            try validateQwen38MTPLiveExactness(
                arguments: ["--evidence", "live.jsonl"],
                readFile: { _ in Data("{\"private\":\"payload\"}\n".utf8) })) { error in
            XCTAssertEqual(
                qwen38MTPLiveExactnessExternalDiagnostic(error),
                "live exactness evidence validation failed")
            XCTAssertFalse(qwen38MTPLiveExactnessExternalDiagnostic(error).contains("payload"))
        }
    }

    private func canonicalProofJSON(
        _ proof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    ) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try XCTUnwrap(String(data: try encoder.encode(proof), encoding: .utf8))
    }

    private func recordData() throws -> Data {
        let record = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: provenance(),
            payload: evidence())
        return Data((try record.jsonLine() + "\n").utf8)
    }

    private func evidence() -> Qwen38MTPLiveExactnessEvidence {
        Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Gate.requiredArtifactID,
            source: Gate.requiredSourceIdentity,
            gdnMode: .gdnOn,
            launchBinding: launchBinding(),
            processIsolation: processIsolation(),
            mlxMemoryBudget: .init(
                memoryLimitBytes: 220 * 1024 * 1024 * 1024,
                cacheLimitBytes: 48 * 1024 * 1024 * 1024),
            hostMemoryObservation: hostMemoryObservation(),
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
            let isDense = ($0 + 1).isMultiple(of: 4)
            return Qwen38MTPLiveExactnessCacheFingerprint(
                layerIndex: $0,
                cacheType: isDense ? "dense-attention" : "recurrent-mamba",
                offset: isDense ? 14 : 0,
                metaStateSHA256: hex("d"),
                stateFingerprints: (0 ..< 2).map { stateIndex in
                    let shape = isDense
                        ? [1, 4, 14, 256]
                        : (stateIndex == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                    let dtype = isDense || stateIndex == 0 ? "bfloat16" : "float32"
                    let byteCount = shape.reduce(1, *) * (dtype == "float32" ? 4 : 2)
                    return Qwen38MTPLiveExactnessArrayFingerprint(
                        stateIndex: stateIndex,
                        shape: shape,
                        dtype: dtype,
                        byteCount: byteCount,
                        sha256: hex("1"))
                })
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

    private func provenance() -> Provenance {
        Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
            hardwareOS: "macOS 26.0",
            harnessGitSHA: String(repeating: "e", count: 40),
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: Gate.modelPathSentinel,
            modelConfigHash: Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Gate.requiredSourceIdentity.sourceID)
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func processIsolation() -> Qwen38MTPLiveExactnessProcessIsolationEvidence {
        Qwen38MTPLiveExactnessProcessIsolationEvidence(
            processID: 44_001,
            parentProcessID: 44_000,
            processStartUptimeNanoseconds: 123_456_789,
            bootTimeUnixSeconds: 1_777_000_000,
            executableIdentitySource: .procPIDPath,
            executableSHA256: hex("6"),
            harnessGitSHA: String(repeating: "e", count: 40),
            sourceID: Gate.requiredSourceIdentity.sourceID,
            gdnMode: .gdnOn,
            observedEnv: .enabled)
    }

    private func launchBinding() -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let processIsolationEvidenceID = Gate.processIsolationEvidenceID(
            for: processIsolation())
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: .gdnOn,
            sourceDigest: Gate.requiredSourceIdentity.sourceID,
            observedEnv: .enabled,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: .gdnOn,
                sourceDigest: Gate.requiredSourceIdentity.sourceID,
                observedEnv: .enabled,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }

    private func hostMemoryObservation() -> Qwen38MTPLiveExactnessHostMemoryObservation {
        let gib = UInt64(1024 * 1024 * 1024)
        return Qwen38MTPLiveExactnessHostMemoryObservation(
            hostUse: "dedicated-serving",
            hostUseSource: "operator-assertion",
            hostUsePolicyVersion: Gate.requiredHostUsePolicyVersion,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 0,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib,
            memoryLimitBytes: 220 * gib,
            cacheLimitBytes: 48 * gib,
            reservedKVBytes: 40 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 4 * gib,
            osServiceReserveBytes: 8 * gib)
    }
}
