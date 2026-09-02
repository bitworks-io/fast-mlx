import Foundation
import HarnessCore
import XCTest

import ScorecardPairControl

/// `.fixedGDNAxis` is the one shipped preset and must accept exactly the
/// handshake shape production workers report today, and reject the same two
/// drift axes the coordinator's hard-coded checks used to reject before the
/// expectations were parameterized. Mirrors the fixture style of
/// `Tests/FastMLXHarnessTests/Qwen38MTPScorecardLiveAdapterTests.swift`.
final class Qwen38MTPScorecardPairExpectationsTests: XCTestCase {
    func testFixedGDNAxisAcceptsTodayShapedHandshakePair() throws {
        let coordinator = makeCoordinator()

        XCTAssertNoThrow(
            try coordinator.validateHandshake(
                handshake(role: .candidate, isolationSeed: "4"),
                expectedRole: .candidate))
        XCTAssertNoThrow(
            try coordinator.validateHandshake(
                handshake(role: .reference, isolationSeed: "5"),
                expectedRole: .reference))
    }

    func testFixedGDNAxisRejectsCandidateReportingGDNOffLaunch() throws {
        let coordinator = makeCoordinator()
        let candidate = handshake(
            role: .candidate,
            isolationSeed: "4",
            gdnMode: .gdnOff,
            observedEnv: .disabled)

        XCTAssertThrowsError(
            try coordinator.validateHandshake(candidate, expectedRole: .candidate)
        ) { error in
            XCTAssertEqual(
                error as? Qwen38MTPScorecardLiveAdapterError,
                .invalidHandshake(.candidate))
        }
    }

    func testFixedGDNAxisRejectsCandidateWithScalarExecutionMode() throws {
        let coordinator = makeCoordinator()
        let candidate = handshake(
            role: .candidate,
            isolationSeed: "4",
            executionMode: .scalar)

        XCTAssertThrowsError(
            try coordinator.validateHandshake(candidate, expectedRole: .candidate)
        ) { error in
            XCTAssertEqual(
                error as? Qwen38MTPScorecardLiveAdapterError,
                .invalidHandshake(.candidate))
        }
    }
}

private typealias Gate = Qwen38MTPPerformanceScorecardGate

private struct NoopScorecardWorkerClient: Qwen38MTPScorecardWorkerClient {
    let role: Qwen38MTPPerformanceScorecardEngineRole

    func start() async throws -> Qwen38MTPScorecardWorkerHandshake {
        fatalError("not exercised by expectations tests")
    }

    func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        fatalError("not exercised by expectations tests")
    }

    func assertReadyForDispatch(expected: Qwen38MTPScorecardWorkerHandshake) async throws {}

    func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        fatalError("not exercised by expectations tests")
    }

    func terminate() async {}
}

private func makeCoordinator() -> Qwen38MTPScorecardLiveCoordinator<
    NoopScorecardWorkerClient, NoopScorecardWorkerClient
> {
    Qwen38MTPScorecardLiveCoordinator(
        candidate: NoopScorecardWorkerClient(role: .candidate),
        reference: NoopScorecardWorkerClient(role: .reference),
        runIdentity: dummyRunIdentity,
        provenance: dummyProvenance,
        releaseBuildObserved: true,
        expectations: .fixedGDNAxis)
}

private let dummyRunIdentity = Qwen38MTPPerformanceScorecardTrustedRunIdentity(
    measurementClass: Gate.measurementClass,
    hardwareChip: "Apple M3 Ultra",
    hardwareRAMBytes: Gate.requiredRAMBytes,
    hardwareOSBuild: "macOS 26.0",
    hostIdentityDigest: hex("a"),
    harnessGitSHA: String(repeating: "e", count: 40),
    candidateMLXSwiftVersion: "0.31.6",
    referenceMLXVersion: nil,
    referenceMLXLMVersion: nil,
    modelLabel: Gate.modelArtifactLabel,
    modelConfigHash: Gate.requiredArtifact.targetConfigSHA256,
    modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
    modelQuant: .init(bits: 8, groupSize: 32),
    corpusID: Gate.requiredWorkload.id,
    corpusContentHash: Gate.requiredWorkload.contentSHA256)

private var dummyProvenance: Provenance {
    Provenance(
        date: "2026-08-25T00:00:00Z",
        hardwareChip: dummyRunIdentity.hardwareChip,
        hardwareRAMBytes: dummyRunIdentity.hardwareRAMBytes,
        hardwareOS: dummyRunIdentity.hardwareOSBuild,
        harnessGitSHA: dummyRunIdentity.harnessGitSHA,
        mlxSwiftVersion: dummyRunIdentity.candidateMLXSwiftVersion,
        referenceMLXVersion: nil,
        referenceMLXLMVersion: nil,
        modelPath: dummyRunIdentity.modelLabel,
        modelConfigHash: dummyRunIdentity.modelConfigHash,
        modelCheckpointManifestHash: dummyRunIdentity.modelCheckpointManifestHash,
        modelQuant: dummyRunIdentity.modelQuant,
        corpusId: dummyRunIdentity.corpusID,
        corpusContentHash: dummyRunIdentity.corpusContentHash,
        nonce: "expectations-test")
}

private func handshake(
    role: Qwen38MTPPerformanceScorecardEngineRole,
    isolationSeed: Character,
    gdnMode mode: Qwen38MTPPerformanceScorecardGDNMode = .gdnOn,
    observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv = .enabled,
    executionMode: Qwen38MTPPerformanceScorecardExecutionMode? = nil
) -> Qwen38MTPScorecardWorkerHandshake {
    let isolation = Qwen38MTPLiveExactnessProcessIsolationEvidence(
        processID: isolationSeed == "4" ? 44_004 : 44_005,
        parentProcessID: 44_000,
        processStartUptimeNanoseconds: isolationSeed == "4" ? 123_456_004 : 123_456_005,
        bootTimeUnixSeconds: 1_777_000_000,
        executableIdentitySource: .procPIDPath,
        executableSHA256: hex("6"),
        harnessGitSHA: String(repeating: "e", count: 40),
        sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        gdnMode: mode,
        observedEnv: observedEnv)
    let processID = Qwen38MTPLiveExactnessGate.processIsolationEvidenceID(for: isolation)
    let launchBinding = Qwen38MTPPerformanceScorecardLaunchBinding(
        mode: mode,
        sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        observedEnv: observedEnv,
        processIsolationEvidenceID: processID,
        launchDigest: Gate.launchDigest(
            mode: mode,
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processID))
    return Qwen38MTPScorecardWorkerHandshake(
        role: role,
        model: Qwen38MTPPerformanceScorecardModel(
            label: qwen38MTPScorecardSharedEngineLabel,
            executionMode: executionMode ?? (role == .candidate ? .exactMTP : .scalar),
            artifact: Gate.requiredArtifact,
            executionDigest: hex(role == .candidate ? "c" : "d"),
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            gdnMode: mode,
            launchBinding: launchBinding),
        processIsolation: isolation,
        launchBinding: launchBinding)
}

private func hex(_ character: Character) -> String {
    String(repeating: String(character), count: 64)
}
