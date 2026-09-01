import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class Qwen38MTPPerformanceScorecardCLITests: XCTestCase {
    private typealias Gate = Qwen38MTPPerformanceScorecardGate
    private typealias GateError = Qwen38MTPPerformanceScorecardGateError
    private typealias CLIError = Qwen38MTPPerformanceScorecardCLIError

    func testStrictArgumentParsingRequiresEvidenceAndAuthorityExactlyOnce() throws {
        let parsed = try parseQwen38MTPPerformanceScorecardValidationArguments([
            "--evidence", "scorecard.jsonl",
            "--authority", "authority.json",
        ])

        XCTAssertEqual(parsed.evidencePath, "scorecard.jsonl")
        XCTAssertEqual(parsed.authorityPath, "authority.json")

        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardValidationArguments([
                "--authority", "authority.json",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--evidence"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardValidationArguments([
                "--evidence", "scorecard.jsonl",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--authority"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardValidationArguments([
                "--evidence", "scorecard.jsonl",
                "--authority", "authority.json",
                "--evidence", "other.jsonl",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .duplicateFlag("--evidence"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardValidationArguments([
                "--evidence", "scorecard.jsonl",
                "--authority", "authority.json",
                "--private-input/operator/unknown", "value",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .unknownFlag)
            XCTAssertFalse(
                qwen38MTPPerformanceScorecardExternalDiagnostic(error).contains("private-input"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardValidationArguments([
                "--evidence", "scorecard.jsonl",
                "private-input/operator/positional.jsonl",
                "--authority", "authority.json",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .unexpectedPositional)
            XCTAssertFalse(String(describing: error).contains("private-input"))
        }
    }

    func testDataValidationAcceptsExactlyOneQualifiedScorecardWithSeparateAuthority() throws {
        let evidence = makeEvidence()
        let output = try validateQwen38MTPPerformanceScorecard(
            arguments: [
                "--evidence", "scorecard.jsonl",
                "--authority", "authority.json",
            ],
            readFile: { path in
                switch path {
                case "scorecard.jsonl": return try self.evidenceData(evidence)
                case "authority.json": return try self.authorityData()
                default: throw CLIError.fileReadFailed(path)
                }
            })

        XCTAssertEqual(
            output,
            "qwen38-mtp-performance-scorecard: VALID qualified=true")
    }

    func testValidationRejectsMalformedAuthorityAndTrailingJSON() throws {
        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: try evidenceData(makeEvidence()),
                authorityData: Data("{".utf8))) { error in
            XCTAssertEqual(error as? CLIError, .malformedAuthority)
        }

        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: try evidenceData(makeEvidence()),
                authorityData: authorityData() + Data(" {}".utf8))) { error in
            XCTAssertEqual(error as? CLIError, .malformedAuthority)
        }
    }

    func testFileReadFailureNamesInputRoleWithoutEchoingPrivatePath() throws {
        let privateEvidencePath = "private-input/operator/scorecard.jsonl"
        let privateAuthorityPath = "private-input/operator/authority.json"

        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecard(
                arguments: [
                    "--evidence", privateEvidencePath,
                    "--authority", privateAuthorityPath,
                ],
                readFile: { _ in throw CocoaError(.fileReadNoSuchFile) })) { error in
            XCTAssertEqual(error as? CLIError, .fileReadFailed("--evidence"))
            XCTAssertFalse(String(describing: error).contains("private-input"))
        }

        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecard(
                arguments: [
                    "--evidence", privateEvidencePath,
                    "--authority", privateAuthorityPath,
                ],
                readFile: { path in
                    if path == privateEvidencePath { return Data() }
                    throw CocoaError(.fileReadNoSuchFile)
                })) { error in
            XCTAssertEqual(error as? CLIError, .fileReadFailed("--authority"))
            XCTAssertFalse(String(describing: error).contains("private-input"))
        }
    }

    func testValidationPropagatesEvidenceCardinalitySubcommandAndGateFailures() throws {
        let evidence = makeEvidence()
        let validLine = try evidenceRecord(evidence).jsonLine()
        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: Data(validLine.utf8),
                authorityData: authorityData())) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }

        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: Data((validLine + "\n" + validLine + "\n").utf8),
                authorityData: authorityData())) { error in
            XCTAssertEqual(error as? GateError, .invalidRecordCardinality(2))
        }

        let wrongSubcommand = ResultRecord(
            subcommand: "bench",
            provenance: makeProvenance(),
            payload: evidence)
        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: Data((try wrongSubcommand.jsonLine() + "\n").utf8),
                authorityData: authorityData())) { error in
            XCTAssertEqual(error as? GateError, .wrongSubcommand("bench"))
        }

        var drifted = evidence
        drifted.settings.penaltiesDisabled = false
        let driftedRecord = ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: drifted)
        XCTAssertThrowsError(
            try validateQwen38MTPPerformanceScorecardData(
                evidenceData: Data((try driftedRecord.jsonLine() + "\n").utf8),
                authorityData: authorityData())) { error in
            XCTAssertEqual(error as? GateError, .invalidGenerationSettings("penalties"))
        }
    }

    func testExternalDiagnosticRedactsAttackerControlledGatePayload() {
        let error = GateError.wrongSubcommand("private-input/operator/evidence.jsonl")

        XCTAssertEqual(
            qwen38MTPPerformanceScorecardExternalDiagnostic(error),
            "evidence or authority validation failed")
        XCTAssertFalse(
            qwen38MTPPerformanceScorecardExternalDiagnostic(error).contains("private-input"))
    }

    private var trustedProof: Qwen38MTPPerformanceScorecardLiveExactnessProof {
        Qwen38MTPPerformanceScorecardLiveExactnessProof(
            artifact: Gate.requiredArtifact,
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            evidenceID: hex("c"),
            accepted: true,
            gdnMode: .gdnOn,
            launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4")))
    }

    private var trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities {
        Qwen38MTPPerformanceScorecardTrustedEngineIdentities(
            candidate: Qwen38MTPPerformanceScorecardModel(
                label: "engine",
                executionMode: .exactMTP,
                artifact: Gate.requiredArtifact,
                executionDigest: Gate.promptSHA256("generic exact mtp execution identity"),
                sourceDigest: sharedSourceDigest,
                gdnMode: .gdnOn,
                launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4"))),
            reference: Qwen38MTPPerformanceScorecardModel(
                label: "engine",
                executionMode: .scalar,
                artifact: Gate.requiredArtifact,
                executionDigest: Gate.promptSHA256("generic scalar execution identity"),
                sourceDigest: sharedSourceDigest,
                gdnMode: .gdnOn,
                launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("5"))))
    }

    private var trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: Gate.measurementClass,
            hardwareChip: "generic-heavy-chip",
            hardwareRAMBytes: Gate.requiredRAMBytes,
            hardwareOSBuild: "generic-os-build-2026-08-24",
            hostIdentityDigest: Gate.promptSHA256("generic dedicated heavy host identity"),
            harnessGitSHA: String(repeating: "1", count: 40),
            candidateMLXSwiftVersion: "generic-mlx-swift-framework-1",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: Gate.modelArtifactLabel,
            modelConfigHash: Gate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: Gate.requiredWorkload.id,
            corpusContentHash: Gate.requiredWorkload.contentSHA256)
    }

    private var authority: Qwen38MTPPerformanceScorecardAuthorityBundle {
        Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: trustedProof,
            trustedEngineIdentities: trustedEngineIdentities,
            trustedRunIdentity: trustedRunIdentity)
    }

    private func makeEvidence(
        hostIdentityDigest: String? = nil
    ) -> Qwen38MTPPerformanceScorecardEvidence {
        var evidence = Qwen38MTPPerformanceScorecardEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Gate.requiredArtifact,
            candidate: trustedEngineIdentities.candidate,
            reference: trustedEngineIdentities.reference,
            liveExactnessProof: trustedProof,
            measurementClass: Gate.measurementClass,
            hardware: .init(
                className: Gate.measurementClass,
                chip: trustedRunIdentity.hardwareChip,
                ramBytes: Gate.requiredRAMBytes,
                osBuild: trustedRunIdentity.hardwareOSBuild,
                hostIdentityDigest: hostIdentityDigest ?? trustedRunIdentity.hostIdentityDigest),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            workload: Gate.requiredWorkload,
            settings: Gate.requiredSettings,
            runPlan: Gate.runPlan,
            pairs: makePairs(),
            metrics: .empty,
            verdict: .unqualified)
        guard hostIdentityDigest == nil else { return evidence }
        evidence.metrics = try! Gate.computeMetrics(evidence, authority: authority)
        evidence.verdict = try! Gate.evaluateCandidate(evidence, authority: authority)
        return evidence
    }

    private func makePairs() -> [Qwen38MTPPerformanceScorecardPair] {
        Gate.runPlan.schedules.enumerated().map { offset, schedule in
            Qwen38MTPPerformanceScorecardPair(
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                warmup: schedule.pairIndex < Gate.runPlan.droppedWarmupPairs,
                order: schedule.order,
                scheduledCaseIDs: schedule.caseIDs,
                scheduledBenchmarkCells: schedule.benchmarkCells,
                lane: schedule.lane,
                candidate: makeEngine(
                    identity: trustedEngineIdentities.candidate,
                    schedule: schedule,
                    offset: offset,
                    candidate: true),
                reference: makeEngine(
                    identity: trustedEngineIdentities.reference,
                    schedule: schedule,
                    offset: offset,
                    candidate: false))
        }
    }

    private func makeEngine(
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        offset: Int,
        candidate: Bool
    ) -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let e2e = candidate ? 9.0 : 10.0
        let decodeSeconds = candidate ? 0.90 : 1.0
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: identity,
            requests: schedule.caseIDs.enumerated().map { requestIndex, caseID in
                makeRequest(
                    caseID: caseID,
                    benchmarkCell: schedule.benchmarkCells[requestIndex],
                    requestIndex: requestIndex,
                    seed: offset + requestIndex,
                    e2eSeconds: e2e,
                    decodeSeconds: decodeSeconds)
            },
            wallSeconds: e2e,
            peakRSSBytes: 210_000_000_000,
            peakMetalBytes: 190_000_000_000,
            thermalBefore: "nominal",
            thermalAfter: "fair",
            proposalCount: candidate ? 12 * schedule.concurrency : 0,
            acceptedCount: candidate ? 8 * schedule.concurrency : 0,
            fallbackUsed: false,
            passthroughUsed: false)
    }

    private func makeRequest(
        caseID: String,
        benchmarkCell: Qwen38MTPPerformanceScorecardBenchmarkCellIdentity,
        requestIndex: Int,
        seed: Int,
        e2eSeconds: Double,
        decodeSeconds: Double
    ) -> Qwen38MTPPerformanceScorecardRequestMeasurement {
        Qwen38MTPPerformanceScorecardRequestMeasurement(
            caseID: caseID,
            benchmarkCell: benchmarkCell,
            requestIndex: requestIndex,
            promptSeconds: 0.25,
            prefillSeconds: 0.80,
            ttftSeconds: 1.00,
            decodeTokenCount: 100,
            decodeSeconds: decodeSeconds,
            e2eSeconds: e2eSeconds,
            outputDigest: digest(seed, "output"),
            cacheDigest: digest(seed, "cache"),
            outputProvenanceID: digest(seed, "output-provenance"),
            cacheProvenanceID: digest(seed, "cache-provenance"))
    }

    private func evidenceData(_ evidence: Qwen38MTPPerformanceScorecardEvidence) throws -> Data {
        Data((try evidenceRecord(evidence).jsonLine() + "\n").utf8)
    }

    private func evidenceRecord(
        _ evidence: Qwen38MTPPerformanceScorecardEvidence
    ) -> ResultRecord<Qwen38MTPPerformanceScorecardEvidence> {
        ResultRecord(
            subcommand: Gate.subcommand,
            provenance: makeProvenance(),
            payload: evidence)
    }

    private func authorityData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(authority)
    }

    private func makeProvenance() -> Provenance {
        Provenance(
            date: "2026-08-24T00:00:00Z",
            hardwareChip: trustedRunIdentity.hardwareChip,
            hardwareRAMBytes: trustedRunIdentity.hardwareRAMBytes,
            hardwareOS: trustedRunIdentity.hardwareOSBuild,
            harnessGitSHA: trustedRunIdentity.harnessGitSHA,
            mlxSwiftVersion: trustedRunIdentity.candidateMLXSwiftVersion,
            referenceMLXVersion: trustedRunIdentity.referenceMLXVersion,
            referenceMLXLMVersion: trustedRunIdentity.referenceMLXLMVersion,
            modelPath: trustedRunIdentity.modelLabel,
            modelConfigHash: trustedRunIdentity.modelConfigHash,
            modelCheckpointManifestHash: trustedRunIdentity.modelCheckpointManifestHash,
            modelQuant: trustedRunIdentity.modelQuant,
            corpusId: trustedRunIdentity.corpusID,
            corpusContentHash: trustedRunIdentity.corpusContentHash,
            nonce: "scorecard-fixture")
    }

    private func digest(_ seed: Int, _ salt: String) -> String {
        let bytes = Array("\(salt)-\(seed)".utf8)
        let alphabet = Array("0123456789abcdef")
        return String((0..<64).map { alphabet[Int(bytes[$0 % bytes.count]) % alphabet.count] })
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private var sharedSourceDigest: String {
        Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
    }

    private func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        processIsolationEvidenceID: String
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            mode == .gdnOn ? .enabled : .disabled
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: sharedSourceDigest,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: Gate.launchDigest(
                mode: mode,
                sourceDigest: sharedSourceDigest,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }
}
