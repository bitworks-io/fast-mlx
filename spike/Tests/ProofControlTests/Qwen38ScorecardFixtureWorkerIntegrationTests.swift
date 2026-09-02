import CryptoKit
import Darwin
import Foundation
import XCTest

@testable import HarnessCore
@testable import ProofControl
@testable import ScorecardPairControl

/// Slice 4b two-process integration test (design verdict P8): the first
/// REAL cross-process exercise of the launch-equality property and of the
/// pair transport against a live child.
///
/// The fixture child (`qwen38-scorecard-fixture-worker`, built beside the
/// test bundle) self-mints its process-isolation evidence with the
/// worker-side HarnessCore recipe over its OWN kernel facts, taking the
/// claim identity via argv (a seam acceptable ONLY in this fixture — the
/// production worker self-observes both fields). The runner side spawns
/// it through `spawnAndObserveAuthorized`, so the test proves over one
/// real child that the runner-minted evidence ID and the worker-self-
/// minted ID are EQUAL — the exact property `validateRunnerLaunchEquality`
/// consumes — and that any intermediary topology would diverge them by
/// construction (hazard C2).
///
/// Fixture keys are deterministic test seeds only (credential boundary
/// preserved). Fixtures mirror the resolver/authorized-spawn suites', kept
/// local per the chain's per-file self-containment convention.
final class Qwen38ScorecardFixtureWorkerIntegrationTests: XCTestCase {
    private static let rootSeed = Data(repeating: 0xC6, count: 32)
    private static let activeSeed = Data(repeating: 0xD7, count: 32)

    private static let claimHarnessGitSHA1 = String(repeating: "ab", count: 20)
    private static let claimSourceID = String(repeating: "5c", count: 32)

    private var caseRoot: URL!

    private static var productsDirectory: URL {
        Bundle(for: Qwen38ScorecardFixtureWorkerIntegrationTests.self)
            .bundleURL
            .deletingLastPathComponent()
    }

    private static var fixtureURL: URL {
        productsDirectory.appendingPathComponent(
            "qwen38-scorecard-fixture-worker"
        )
    }

    override func setUpWithError() throws {
        let canonicalTemporaryPath = try XCTUnwrap(
            Darwin.realpath(NSTemporaryDirectory(), nil)
        )
        defer { Darwin.free(canonicalTemporaryPath) }
        caseRoot = URL(
            fileURLWithPath: String(cString: canonicalTemporaryPath),
            isDirectory: true
        )
        .appendingPathComponent(
            "fast-mlx-qwen38-fixture-integration-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(
            at: caseRoot,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        if let caseRoot {
            try? FileManager.default.removeItem(at: caseRoot)
        }
    }

    private struct FixtureProtocolError: Error {}

    /// Local mirror of the fixture's report line; deliberately NOT an
    /// import of the fixture module (two executable modules in one test
    /// bundle would fight over `main`).
    private struct FixtureReport: Decodable {
        let role: String
        let processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
        let processIsolationEvidenceID: String
    }

    private struct SpawnedFixture {
        let worker: Qwen38ScorecardSpawnedWorker
        let stdin: Pipe
        let stdout: Pipe
        let stderr: Pipe
    }

    private func spawnFixture(
        role: String,
        extraArguments: [String] = []
    ) throws -> SpawnedFixture {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.fixtureURL.path),
            "fixture binary missing at \(Self.fixtureURL.path); "
                + "ProofControlTests must depend on the fixture target"
        )
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        let worker = try Qwen38ScorecardWorkerSpawner.spawnAndObserveAuthorized(
            authorization: try resolvedAuthorization(),
            executableURL: Self.fixtureURL,
            arguments: [
                "--role", role,
                "--harness-git-sha", Self.claimHarnessGitSHA1,
                "--source-id", Self.claimSourceID,
            ] + extraArguments,
            environment: ProcessInfo.processInfo.environment,
            standardInput: stdin,
            standardOutput: stdout,
            standardError: stderr,
            gdnMode: .on
        )
        return SpawnedFixture(
            worker: worker,
            stdin: stdin,
            stdout: stdout,
            stderr: stderr
        )
    }

    private func readReportLine(_ spawned: SpawnedFixture) throws -> FixtureReport {
        var buffer = Data()
        while buffer.firstIndex(of: 0x0a) == nil {
            let chunk = spawned.stdout.fileHandleForReading.availableData
            guard !chunk.isEmpty else {
                let diagnostics = String(
                    decoding: spawned.stderr.fileHandleForReading
                        .availableData,
                    as: UTF8.self
                )
                XCTFail("fixture exited before its report line: \(diagnostics)")
                throw FixtureProtocolError()
            }
            buffer.append(chunk)
        }
        let newline = try XCTUnwrap(buffer.firstIndex(of: 0x0a))
        return try JSONDecoder().decode(
            FixtureReport.self,
            from: Data(buffer[..<newline])
        )
    }

    // MARK: - The cross-recipe equality property

    /// The load-bearing Slice 4b proof: for one REAL spawned child, the
    /// worker-side self-minted evidence ID equals the runner-side
    /// kernel-observed evidence ID, and two children yield two distinct
    /// IDs — exactly what `validateRunnerLaunchEquality` requires from the
    /// launch observations.
    func testRunnerMintedAndSelfMintedEvidenceIDsAgreeOverRealChildren() throws {
        let candidate = try spawnFixture(role: "candidate")
        let reference = try spawnFixture(role: "reference")
        defer {
            try? candidate.stdin.fileHandleForWriting.close()
            try? reference.stdin.fileHandleForWriting.close()
            candidate.worker.process.waitUntilExit()
            reference.worker.process.waitUntilExit()
        }

        let candidateReport = try readReportLine(candidate)
        let referenceReport = try readReportLine(reference)

        XCTAssertEqual(candidateReport.role, "candidate")
        XCTAssertEqual(referenceReport.role, "reference")

        // Cross-recipe, cross-process equality per leg.
        XCTAssertEqual(
            candidateReport.processIsolationEvidenceID,
            candidate.worker.evidenceID
        )
        XCTAssertEqual(
            referenceReport.processIsolationEvidenceID,
            reference.worker.evidenceID
        )
        // Two distinct children ⇒ two distinct IDs.
        XCTAssertNotEqual(
            candidate.worker.evidenceID,
            reference.worker.evidenceID
        )

        // The fixture observed the FORCED GDN environment (authorized
        // spawn, gdn-on leg) and the runner's kernel-observed parentage.
        XCTAssertEqual(candidateReport.processIsolation.observedEnv, .enabled)
        XCTAssertEqual(candidateReport.processIsolation.gdnMode, .gdnOn)
        XCTAssertEqual(
            candidateReport.processIsolation.parentProcessID,
            Int(getpid())
        )
        XCTAssertEqual(
            candidateReport.processIsolation.harnessGitSHA,
            Self.claimHarnessGitSHA1
        )
        XCTAssertEqual(
            candidateReport.processIsolation.sourceID,
            Self.claimSourceID
        )
    }

    // MARK: - Transport against a real child

    /// Cooperative teardown: `terminate()` closes stdin first; the fixture
    /// exits on EOF; the child must be gone afterwards.
    func testPipesTransportTerminateEndsCooperativeFixture() async throws {
        let spawned = try spawnFixture(role: "candidate")
        _ = try readReportLine(spawned)

        let transport = Qwen38MTPScorecardProcessPipesTransport(
            role: .candidate,
            child: spawned.worker.process,
            stdin: spawned.stdin.fileHandleForWriting,
            stdout: spawned.stdout.fileHandleForReading,
            stderr: spawned.stderr.fileHandleForReading
        )
        await transport.terminate()
        spawned.worker.process.waitUntilExit()
        XCTAssertFalse(spawned.worker.process.isRunning)
    }

    /// Escalation teardown (design verdict P8 rider): a child that
    /// IGNORES both SIGTERM and SIGINT can only be ended by the
    /// transport's final SIGKILL rung — proven against a real process,
    /// not a fake. The transport is handed the NULL DEVICE as its stdin
    /// handle (not the fixture's real stdin), so the cooperative
    /// EOF-exit path stays closed and only escalation can end the child.
    func testPipesTransportTerminateEscalatesToSIGKILL() async throws {
        let spawned = try spawnFixture(
            role: "candidate",
            extraArguments: ["--ignore-termination"]
        )
        _ = try readReportLine(spawned)

        let transport = Qwen38MTPScorecardProcessPipesTransport(
            role: .candidate,
            child: spawned.worker.process,
            stdin: FileHandle.nullDevice,
            stdout: spawned.stdout.fileHandleForReading,
            stderr: spawned.stderr.fileHandleForReading
        )
        await transport.terminate()
        spawned.worker.process.waitUntilExit()
        XCTAssertFalse(spawned.worker.process.isRunning)
        // SIGKILL termination, not a clean exit.
        XCTAssertEqual(spawned.worker.process.terminationReason, .uncaughtSignal)
        try? spawned.stdin.fileHandleForWriting.close()
    }

    // MARK: - Fixtures (deterministic test seeds only)

    private func resolvedAuthorization() throws
        -> Qwen38ScorecardResolvedRunAuthorization
    {
        let root = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.rootSeed
        )
        let active = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let rootKeyID = Self.sha256Hex(root.publicKey.rawRepresentation)
        let policyBytes = try Qwen38ScorecardKeyPolicyVerifier.policyBytes(
            fields: Qwen38ScorecardKeyPolicyFields(
                rootKeyID: rootKeyID,
                policyGeneration: 3,
                validFromUnixSeconds: 1_700_000_000,
                validUntilUnixSeconds: 1_800_000_000,
                activeOperatorKeyID: Self.sha256Hex(
                    active.publicKey.rawRepresentation
                ),
                activeOperatorPublicKeyBase64:
                    active.publicKey.rawRepresentation.base64EncodedString(),
                activeOperatorScope: .scorecardRunClaim,
                allowedClaimSubject: .mtpScorecardResultPair,
                revokedOperatorKeyIDs: []
            )
        )
        let policyPath = caseRoot.appendingPathComponent("policy.txt")
        try policyBytes.write(to: policyPath)
        let policyFile = try AdmittedFile.capture(
            absolutePath: policyPath.path,
            maximumBytes: 4_096
        )
        let rootSignature = try root.signature(for: policyFile.bytes)
        let policy = try Qwen38ScorecardKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: rootSignature.base64EncodedString(),
            trustAnchor: Qwen38ScorecardKeyPolicyTrustAnchor(
                rootPublicKeyBase64:
                    root.publicKey.rawRepresentation.base64EncodedString(),
                rootKeyID: rootKeyID,
                expectedCurrentPolicySHA256: policyFile.sha256,
                minimumPolicyGeneration: 1,
                verificationUnixSeconds: 1_750_000_000
            )
        )

        let claimBytes = try Qwen38ScorecardRunClaimVerifier.claimBytes(
            fields: Qwen38ScorecardRunClaimFields(
                subject: .mtpScorecardResultPair,
                modelSHA256: String(repeating: "12", count: 32),
                tokenizerSHA256: String(repeating: "34", count: 32),
                tensorManifestSHA256: String(repeating: "56", count: 32),
                chatTemplateSHA256: String(repeating: "78", count: 32),
                quantizationIdentity: "mxfp8",
                target: Qwen38ScorecardModelIdentity(
                    modelID: "mlx-community/Qwen3.8-27B-mxfp8",
                    revision: "1a2b3c4"
                ),
                drafter: Qwen38ScorecardModelIdentity(
                    modelID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
                    revision: "5d6e7f8"
                ),
                sourceID: Self.claimSourceID,
                hostAdmissionID: String(repeating: "bc", count: 32),
                harnessGitSHA1: Self.claimHarnessGitSHA1,
                gdnOnMode: .on,
                gdnOffMode: .off,
                corpusID: "qwen38-27b-frozen-scorecard-workload-v2",
                corpusContentSHA256: String(repeating: "f0", count: 32),
                resultPairID: String(repeating: "1f", count: 32)
            )
        )
        let claimSignature = try active.signature(for: claimBytes)
        return try Qwen38ScorecardRunAuthorizationResolver.resolve(
            claimBytes: claimBytes,
            claimSignatureBase64: claimSignature.base64EncodedString(),
            policy: policy
        )
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
