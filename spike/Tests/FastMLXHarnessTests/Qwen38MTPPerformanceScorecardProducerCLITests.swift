import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class Qwen38MTPPerformanceScorecardProducerCLITests: XCTestCase {
    private typealias CLIError = Qwen38MTPPerformanceScorecardProducerCLIError
    private typealias Gate = Qwen38MTPPerformanceScorecardGate

    func testProducerArgumentsRequireAuthorityAndFreshOutputExactlyOnce() throws {
        let parsed = try parseQwen38MTPPerformanceScorecardProducerArguments([
            "--authority", "authority.json",
            "--output", "scorecard.jsonl",
        ])

        XCTAssertEqual(parsed.authorityPath, "authority.json")
        XCTAssertEqual(parsed.outputPath, "scorecard.jsonl")

        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardProducerArguments([
                "--output", "scorecard.jsonl",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--authority"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardProducerArguments([
                "--authority", "authority.json",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--output"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardProducerArguments([
                "--authority", "authority.json",
                "--output", "scorecard.jsonl",
                "--output", "other.jsonl",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .duplicateFlag("--output"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPPerformanceScorecardProducerArguments([
                "--authority", "authority.json",
                "--output", "scorecard.jsonl",
                "--model-path", "private-input/operator/model",
            ])) { error in
            XCTAssertEqual(error as? CLIError, .unknownFlag)
            XCTAssertFalse(
                qwen38MTPPerformanceScorecardProducerExternalDiagnostic(error)
                    .contains("private-input"))
        }
    }

    func testMissingAuthorityFailsBeforeReadingAnyPathOrCreatingProducer() async {
        let calls = CallCounts()

        do {
            _ = try await produceQwen38MTPPerformanceScorecard(
                arguments: ["--output", "private-input/operator/scorecard.jsonl"],
                readFile: { _ in
                    await calls.recordRead()
                    return Data()
                },
                makeRecord: { _ in
                    await calls.recordProduce()
                    throw CLIError.producerUnavailable
                },
                writeFresh: { _, _ in
                    await calls.recordWrite()
                })
            XCTFail("Expected missing authority to fail")
        } catch {
            XCTAssertEqual(error as? CLIError, .missingFlag("--authority"))
        }

        let snapshot = await calls.snapshot()
        XCTAssertEqual(snapshot, .init())
    }

    func testProductionDefaultStopsAfterAuthorityDecodeWithoutWritingOutput() async throws {
        let calls = CallCounts()
        let authorityData = try JSONEncoder().encode(Self.authority)

        do {
            _ = try await produceQwen38MTPPerformanceScorecard(
                arguments: [
                    "--authority", "private-input/operator/authority.json",
                    "--output", "private-input/operator/scorecard.jsonl",
                ],
                readFile: { _ in
                    await calls.recordRead()
                    return authorityData
                },
                makeRecord: nil,
                writeFresh: { _, _ in
                    await calls.recordWrite()
                })
            XCTFail("Expected unavailable production producer")
        } catch {
            XCTAssertEqual(error as? CLIError, .producerUnavailable)
            XCTAssertFalse(
                qwen38MTPPerformanceScorecardProducerExternalDiagnostic(error)
                    .contains("private-input"))
        }

        let snapshot = await calls.snapshot()
        XCTAssertEqual(snapshot.reads, 1)
        XCTAssertEqual(snapshot.productions, 0)
        XCTAssertEqual(snapshot.writes, 0)
    }

    func testRejectedRecordWithPrivateOrDriftedProvenanceNeverWrites() async throws {
        let calls = CallCounts()
        let authorityData = try JSONEncoder().encode(Self.authority)
        let record = ResultRecord(
            subcommand: Gate.rejectedSubcommand,
            provenance: Self.makeProvenance(modelPath: "private-input/operator/model"),
            payload: Self.makeUnqualifiedEnvelope())

        do {
            _ = try await produceQwen38MTPPerformanceScorecard(
                arguments: [
                    "--authority", "authority.json",
                    "--output", "scorecard.jsonl",
                ],
                readFile: { _ in authorityData },
                makeRecord: { _ in record },
                writeFresh: { _, _ in await calls.recordWrite() })
            XCTFail("Expected rejected record provenance drift to fail")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidProducerRecord)
            XCTAssertFalse(
                qwen38MTPPerformanceScorecardProducerExternalDiagnostic(error)
                    .contains("private-input"))
        }

        let snapshot = await calls.snapshot()
        XCTAssertEqual(snapshot.writes, 0)
    }

    func testAtomicFreshWriterPublishesOneCompleteFileAndRefusesOverwriteOrSymlink() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-scorecard-writer-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let output = directory.appendingPathComponent("scorecard.jsonl")
        let contents = Data("{\"qualified\":true}\n".utf8)
        try writeFreshQwen38MTPPerformanceScorecard(contents, output.path)

        XCTAssertEqual(try Data(contentsOf: output), contents)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(),
            ["scorecard.jsonl"])
        XCTAssertThrowsError(
            try writeFreshQwen38MTPPerformanceScorecard(Data("other\n".utf8), output.path)
        ) { error in
            XCTAssertEqual(error as? CLIError, .outputExists)
        }
        XCTAssertEqual(try Data(contentsOf: output), contents)

        let alias = directory.appendingPathComponent("alias.jsonl")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: output)
        XCTAssertThrowsError(
            try writeFreshQwen38MTPPerformanceScorecard(contents, alias.path)
        ) { error in
            XCTAssertEqual(error as? CLIError, .unsafeOutput)
        }
        XCTAssertEqual(try Data(contentsOf: output), contents)
    }

    private static var authority: Qwen38MTPPerformanceScorecardAuthorityBundle {
        Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: .init(
                artifact: Gate.requiredArtifact,
                artifactID: hex("a"),
                sourceID: hex("b"),
                evidenceID: hex("c"),
                accepted: true),
            trustedEngineIdentities: .init(
                candidate: .init(
                    label: "candidate",
                    artifact: Gate.requiredArtifact,
                    executionDigest: Gate.promptSHA256("generic candidate execution"),
                    sourceDigest: Gate.promptSHA256("generic candidate source")),
                reference: .init(
                    label: "reference",
                    artifact: Gate.requiredArtifact,
                    executionDigest: Gate.promptSHA256("generic reference execution"),
                    sourceDigest: Gate.promptSHA256("generic reference source"))),
            trustedRunIdentity: .init(
                measurementClass: Gate.measurementClass,
                hardwareChip: "generic-heavy-chip",
                hardwareRAMBytes: Gate.requiredRAMBytes,
                hardwareOSBuild: "generic-os-build",
                hostIdentityDigest: Gate.promptSHA256("generic host identity"),
                harnessGitSHA: String(repeating: "1", count: 40),
                candidateMLXSwiftVersion: "generic-mlx-swift",
                referenceMLXVersion: nil,
                referenceMLXLMVersion: nil,
                modelLabel: Gate.modelArtifactLabel,
                modelConfigHash: Gate.requiredArtifact.targetConfigSHA256,
                modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
                modelQuant: .init(bits: 8, groupSize: 32),
                corpusID: Gate.requiredWorkload.id,
                corpusContentHash: Gate.requiredWorkload.contentSHA256))
    }

    private static func makeUnqualifiedEnvelope() -> Qwen38MTPPerformanceScorecardEvidence {
        let runIdentity = authority.trustedRunIdentity
        var evidence = Qwen38MTPPerformanceScorecardEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Gate.requiredArtifact,
            candidate: authority.trustedEngineIdentities.candidate,
            reference: authority.trustedEngineIdentities.reference,
            liveExactnessProof: authority.acceptedLiveExactnessProof,
            measurementClass: Gate.measurementClass,
            hardware: .init(
                className: Gate.measurementClass,
                chip: runIdentity.hardwareChip,
                ramBytes: runIdentity.hardwareRAMBytes,
                osBuild: runIdentity.hardwareOSBuild,
                hostIdentityDigest: runIdentity.hostIdentityDigest),
            releaseBuildRequired: true,
            releaseBuildObserved: true,
            workload: Gate.requiredWorkload,
            settings: Gate.requiredSettings,
            runPlan: Gate.runPlan,
            pairs: Gate.runPlan.schedules.map { schedule in
                Qwen38MTPPerformanceScorecardPair(
                    concurrency: schedule.concurrency,
                    pairIndex: schedule.pairIndex,
                    warmup: schedule.pairIndex < Gate.runPlan.droppedWarmupPairs,
                    order: schedule.order,
                    scheduledCaseIDs: schedule.caseIDs,
                    candidate: makeEngine(
                        identity: authority.trustedEngineIdentities.candidate,
                        schedule: schedule,
                        candidate: true),
                    reference: makeEngine(
                        identity: authority.trustedEngineIdentities.reference,
                        schedule: schedule,
                        candidate: false))
            },
            metrics: .empty,
            verdict: .unqualified)
        evidence.metrics = try! Gate.computeMetrics(evidence, authority: authority)
        evidence.verdict = try! Gate.evaluateCandidate(evidence, authority: authority)
        precondition(!evidence.verdict.qualified)
        return evidence
    }

    private static func makeEngine(
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        candidate: Bool
    ) -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let e2eSeconds = candidate ? 11.0 : 10.0
        let decodeSeconds = candidate ? 1.1 : 1.0
        let requests = schedule.caseIDs.enumerated().map { requestIndex, caseID in
            let identitySeed = "\(schedule.concurrency)-\(schedule.pairIndex)-\(requestIndex)"
            return Qwen38MTPPerformanceScorecardRequestMeasurement(
                caseID: caseID,
                requestIndex: requestIndex,
                promptSeconds: 0.25,
                prefillSeconds: 0.8,
                ttftSeconds: 1.0,
                decodeTokenCount: 100,
                decodeSeconds: decodeSeconds,
                e2eSeconds: e2eSeconds,
                outputDigest: Gate.promptSHA256("output-\(identitySeed)"),
                cacheDigest: Gate.promptSHA256("cache-\(identitySeed)"),
                outputProvenanceID: Gate.promptSHA256("output-proof-\(identitySeed)"),
                cacheProvenanceID: Gate.promptSHA256("cache-proof-\(identitySeed)"))
        }
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: identity,
            requests: requests,
            wallSeconds: e2eSeconds,
            peakRSSBytes: 210_000_000_000,
            peakMetalBytes: 190_000_000_000,
            thermalBefore: "nominal",
            thermalAfter: "fair",
            proposalCount: candidate ? 12 * schedule.concurrency : 0,
            acceptedCount: candidate ? 8 * schedule.concurrency : 0,
            fallbackUsed: false,
            passthroughUsed: false)
    }

    private static func makeProvenance(modelPath: String) -> Provenance {
        let runIdentity = authority.trustedRunIdentity
        return Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: runIdentity.hardwareChip,
            hardwareRAMBytes: runIdentity.hardwareRAMBytes,
            hardwareOS: runIdentity.hardwareOSBuild,
            harnessGitSHA: runIdentity.harnessGitSHA,
            mlxSwiftVersion: runIdentity.candidateMLXSwiftVersion,
            referenceMLXVersion: runIdentity.referenceMLXVersion,
            referenceMLXLMVersion: runIdentity.referenceMLXLMVersion,
            modelPath: modelPath,
            modelConfigHash: runIdentity.modelConfigHash,
            modelCheckpointManifestHash: runIdentity.modelCheckpointManifestHash,
            modelQuant: runIdentity.modelQuant,
            corpusId: runIdentity.corpusID,
            corpusContentHash: runIdentity.corpusContentHash,
            nonce: "generic-producer-test")
    }

    private static func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}

private actor CallCounts {
    struct Snapshot: Equatable {
        var reads = 0
        var productions = 0
        var writes = 0
    }

    private var value = Snapshot()

    func recordRead() { value.reads += 1 }
    func recordProduce() { value.productions += 1 }
    func recordWrite() { value.writes += 1 }
    func snapshot() -> Snapshot { value }
}
