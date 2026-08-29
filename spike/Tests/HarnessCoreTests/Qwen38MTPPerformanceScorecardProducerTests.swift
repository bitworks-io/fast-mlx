import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38MTPPerformanceScorecardProducerTests: XCTestCase {
    private typealias Gate = Qwen38MTPPerformanceScorecardGate
    private typealias GateError = Qwen38MTPPerformanceScorecardGateError

    func testAuthorityPreflightRunsBeforeMeasurementClosure() async {
        let log = CallLog()
        var authority = Self.trustedAuthority
        authority.acceptedLiveExactnessProof.accepted = false
        let producer = Self.makeProducer(log: log)

        do {
            _ = try await producer.makeRecord(
                authority: authority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: true)
            XCTFail("Expected invalid authority to fail before measurement")
        } catch {
            XCTAssertEqual(error as? GateError, .invalidLiveExactnessProof)
        }

        let calls = await log.snapshot()
        XCTAssertEqual(calls, [])
    }

    func testProvenancePreflightRunsBeforeMeasurementClosure() async {
        let log = CallLog()
        let producer = Self.makeProducer(log: log)

        do {
            _ = try await producer.makeRecord(
                authority: Self.trustedAuthority,
                provenance: Self.makeProvenance(harnessGitSHA: String(repeating: "2", count: 40)),
                releaseBuildObserved: true)
            XCTFail("Expected invalid provenance to fail before measurement")
        } catch {
            XCTAssertEqual(error as? GateError, .invalidProvenance("record"))
        }

        let calls = await log.snapshot()
        XCTAssertEqual(calls, [])
    }

    func testReleaseBuildPreflightRunsBeforeMeasurementClosure() async {
        let log = CallLog()
        let producer = Self.makeProducer(log: log)

        do {
            _ = try await producer.makeRecord(
                authority: Self.trustedAuthority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: false)
            XCTFail("Expected debug build observation to fail before measurement")
        } catch {
            XCTAssertEqual(error as? GateError, .releaseBuildRequired)
        }

        let calls = await log.snapshot()
        XCTAssertEqual(calls, [])
    }

    func testUnsafeSameProcessStaticEnvironmentToggleFailsBeforeMeasurement() async {
        let log = CallLog()
        var authority = Self.trustedAuthority
        authority.trustedEngineIdentities.candidate.launchBinding =
            Self.launchBinding(mode: .gdnOn, processIsolationEvidenceID: Self.hex("4"))
        authority.trustedEngineIdentities.reference.launchBinding =
            Self.launchBinding(mode: .gdnOff, processIsolationEvidenceID: Self.hex("4"))
        let producer = Self.makeProducer(log: log)

        do {
            _ = try await producer.makeRecord(
                authority: authority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: true)
            XCTFail("Expected unsafe same-process static environment toggle to fail before measurement")
        } catch {
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        let calls = await log.snapshot()
        XCTAssertEqual(calls, [])
    }

    func testForgedLaunchDigestFailsBeforeMeasurement() async {
        let log = CallLog()
        var authority = Self.trustedAuthority
        authority.trustedEngineIdentities.candidate.launchBinding = Self.launchBinding(
            mode: .gdnOn,
            processIsolationEvidenceID: Self.hex("4"),
            launchDigest: Self.hex("8"))
        authority.acceptedLiveExactnessProof.launchBinding = authority.trustedEngineIdentities.candidate.launchBinding
        let producer = Self.makeProducer(log: log)

        do {
            _ = try await producer.makeRecord(
                authority: authority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: true)
            XCTFail("Expected forged launch digest to fail before measurement")
        } catch {
            XCTAssertEqual(error as? GateError, .invalidModelIdentity)
        }

        let calls = await log.snapshot()
        XCTAssertEqual(calls, [])
    }

    func testProducerMeasuresFrozenSchedulesInFrozenOrder() async throws {
        let log = CallLog()
        let producer = Self.makeProducer(log: log)

        let record = try await producer.makeRecord(
            authority: Self.trustedAuthority,
            provenance: Self.makeProvenance(),
            releaseBuildObserved: true)

        XCTAssertEqual(record.payload.pairs.count, 126)
        XCTAssertEqual(record.payload.runPlan.concurrencies, [1, 2, 4])
        XCTAssertEqual(record.payload.runPlan, Gate.runPlan)
        XCTAssertEqual(record.payload.workload, Gate.requiredWorkload)
        XCTAssertEqual(record.payload.settings, Gate.requiredSettings)
        XCTAssertEqual(record.subcommand, Gate.subcommand)
        XCTAssertTrue(record.payload.verdict.qualified)

        let expectedCalls = Gate.runPlan.schedules.flatMap { schedule in
            let candidate = CallEvent(
                role: .candidate,
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                caseIDs: schedule.caseIDs)
            let reference = CallEvent(
                role: .reference,
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                caseIDs: schedule.caseIDs)
            return schedule.order == .candidateThenReference
                ? [candidate, reference]
                : [reference, candidate]
        }
        let calls = await log.snapshot()
        XCTAssertEqual(calls.count, Gate.runPlan.budget.engineMeasurements)
        XCTAssertEqual(calls, expectedCalls)
    }

    func testQualifiedRecordValidatesWithExistingJSONLAuthorityPath() async throws {
        let producer = Self.makeProducer(log: CallLog())
        let record = try await producer.makeRecord(
            authority: Self.trustedAuthority,
            provenance: Self.makeProvenance(),
            releaseBuildObserved: true)

        let jsonl = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertEqual(
            try Gate.validateJSONL(jsonl, authority: Self.trustedAuthority),
            [record.payload.verdict])
    }

    func testRejectedVerdictUsesRejectedSubcommandWithoutFabricatingSuccess() async throws {
        let producer = Self.makeProducer(log: CallLog(), mode: .unqualifiedPerformance)

        let record = try await producer.makeRecord(
            authority: Self.trustedAuthority,
            provenance: Self.makeProvenance(),
            releaseBuildObserved: true)

        XCTAssertEqual(record.subcommand, Gate.rejectedSubcommand)
        XCTAssertFalse(record.payload.verdict.qualified)
    }

    func testInjectedScheduleAndOutputDriftFailClosed() async {
        do {
            _ = try await Self.makeProducer(
                log: CallLog(),
                mode: .scheduleDrift
            ).makeRecord(
                authority: Self.trustedAuthority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: true)
            XCTFail("Expected schedule drift to fail")
        } catch {
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: 0, reason: "request schedule"))
        }

        do {
            _ = try await Self.makeProducer(
                log: CallLog(),
                mode: .outputDrift
            ).makeRecord(
                authority: Self.trustedAuthority,
                provenance: Self.makeProvenance(),
                releaseBuildObserved: true)
            XCTFail("Expected output drift to fail")
        } catch {
            XCTAssertEqual(
                error as? GateError,
                .invalidPair(index: 0, reason: "output/cache provenance"))
        }
    }

    private static func makeProducer(
        log: CallLog,
        mode: StubMode = .qualified
    ) -> Qwen38MTPPerformanceScorecardProducer {
        Qwen38MTPPerformanceScorecardProducer { request in
            guard request.workload == Gate.requiredWorkload else {
                throw StubError.unexpectedInput
            }
            guard request.settings == Gate.requiredSettings else {
                throw StubError.unexpectedInput
            }
            await log.append(CallEvent(
                role: request.role,
                concurrency: request.schedule.concurrency,
                pairIndex: request.schedule.pairIndex,
                caseIDs: request.schedule.caseIDs))
            return makeEngine(
                identity: request.identity,
                schedule: request.schedule,
                role: request.role,
                mode: mode)
        }
    }

    private static var trustedAuthority: Qwen38MTPPerformanceScorecardAuthorityBundle {
        Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: Qwen38MTPPerformanceScorecardLiveExactnessProof(
                artifact: Gate.requiredArtifact,
                artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
                sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                evidenceID: hex("c"),
                accepted: true,
                gdnMode: .gdnOn,
                launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4"))),
            trustedEngineIdentities: Qwen38MTPPerformanceScorecardTrustedEngineIdentities(
                candidate: Qwen38MTPPerformanceScorecardModel(
                    label: "candidate",
                    artifact: Gate.requiredArtifact,
                    executionDigest: Gate.promptSHA256("generic candidate execution identity"),
                    sourceDigest: sharedSourceDigest,
                    gdnMode: .gdnOn,
                    launchBinding: launchBinding(mode: .gdnOn, processIsolationEvidenceID: hex("4"))),
                reference: Qwen38MTPPerformanceScorecardModel(
                    label: "reference",
                    artifact: Gate.requiredArtifact,
                    executionDigest: Gate.promptSHA256("generic reference execution identity"),
                    sourceDigest: sharedSourceDigest,
                    gdnMode: .gdnOff,
                    launchBinding: launchBinding(mode: .gdnOff, processIsolationEvidenceID: hex("5")))),
            trustedRunIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity(
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
                corpusContentHash: Gate.requiredWorkload.contentSHA256))
    }

    private static func makeProvenance(
        harnessGitSHA: String? = nil
    ) -> Provenance {
        let runIdentity = trustedAuthority.trustedRunIdentity
        return Provenance(
            date: "2026-08-24T00:00:00Z",
            hardwareChip: runIdentity.hardwareChip,
            hardwareRAMBytes: runIdentity.hardwareRAMBytes,
            hardwareOS: runIdentity.hardwareOSBuild,
            harnessGitSHA: harnessGitSHA ?? runIdentity.harnessGitSHA,
            mlxSwiftVersion: runIdentity.candidateMLXSwiftVersion,
            referenceMLXVersion: runIdentity.referenceMLXVersion,
            referenceMLXLMVersion: runIdentity.referenceMLXLMVersion,
            modelPath: runIdentity.modelLabel,
            modelConfigHash: runIdentity.modelConfigHash,
            modelCheckpointManifestHash: runIdentity.modelCheckpointManifestHash,
            modelQuant: runIdentity.modelQuant,
            corpusId: runIdentity.corpusID,
            corpusContentHash: runIdentity.corpusContentHash,
            nonce: "scorecard-producer-fixture")
    }

    private static func makeEngine(
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        role: Qwen38MTPPerformanceScorecardEngineRole,
        mode: StubMode
    ) -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let isCandidate = role == .candidate
        let e2e: Double
        let decodeSeconds: Double
        if isCandidate {
            e2e = mode == .unqualifiedPerformance ? 11.0 : 9.0
            decodeSeconds = mode == .unqualifiedPerformance ? 1.10 : 0.90
        } else {
            e2e = 10.0
            decodeSeconds = 1.0
        }
        var requests = schedule.caseIDs.enumerated().map { requestIndex, caseID in
            makeRequest(
                caseID: caseID,
                requestIndex: requestIndex,
                seed: schedule.concurrency * 1_000 + schedule.pairIndex * 10 + requestIndex,
                e2eSeconds: e2e,
                decodeSeconds: decodeSeconds)
        }
        if mode == .scheduleDrift && isCandidate && schedule.concurrency == 1
            && schedule.pairIndex == 0
        {
            requests[0].caseID = "case-06"
        }
        if mode == .outputDrift && !isCandidate && schedule.concurrency == 1
            && schedule.pairIndex == 0
        {
            requests[0].outputDigest = hex("f")
        }
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: identity,
            requests: requests,
            wallSeconds: e2e,
            peakRSSBytes: 210_000_000_000,
            peakMetalBytes: 190_000_000_000,
            thermalBefore: "nominal",
            thermalAfter: "fair",
            proposalCount: isCandidate ? 12 * schedule.concurrency : 0,
            acceptedCount: isCandidate ? 8 * schedule.concurrency : 0,
            fallbackUsed: false,
            passthroughUsed: false)
    }

    private static func makeRequest(
        caseID: String,
        requestIndex: Int,
        seed: Int,
        e2eSeconds: Double,
        decodeSeconds: Double
    ) -> Qwen38MTPPerformanceScorecardRequestMeasurement {
        Qwen38MTPPerformanceScorecardRequestMeasurement(
            caseID: caseID,
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

    private static func digest(_ seed: Int, _ salt: String) -> String {
        let bytes = Array("\(salt)-\(seed)".utf8)
        let alphabet = Array("0123456789abcdef")
        return String((0..<64).map { alphabet[Int(bytes[$0 % bytes.count]) % alphabet.count] })
    }

    private static func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private static var sharedSourceDigest: String {
        Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
    }

    private static func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv? = nil,
        processIsolationEvidenceID: String,
        launchDigest: String? = nil
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let observedEnv = observedEnv ?? (mode == .gdnOn ? .enabled : .disabled)
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: sharedSourceDigest,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: launchDigest ?? Gate.launchDigest(
                mode: mode,
                sourceDigest: sharedSourceDigest,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }
}

private actor CallLog {
    private var events: [CallEvent] = []

    func append(_ event: CallEvent) {
        events.append(event)
    }

    func snapshot() -> [CallEvent] {
        events
    }
}

private struct CallEvent: Equatable, Sendable {
    var role: Qwen38MTPPerformanceScorecardEngineRole
    var concurrency: Int
    var pairIndex: Int
    var caseIDs: [String]
}

private enum StubMode: Sendable {
    case qualified
    case unqualifiedPerformance
    case scheduleDrift
    case outputDrift
}

private enum StubError: Error {
    case unexpectedInput
}
