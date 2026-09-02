import CryptoKit
import Darwin
import Foundation
import XCTest

@testable import ProofControl

/// Slice 4a: the authorized spawn entry point must derive the worker's
/// identity inputs from the resolved (operator-signed) claim — never from
/// the caller — and must force the child GDN environment to the requested
/// leg. Fixture keys are deterministic test seeds only (credential
/// boundary preserved). Fixtures mirror the resolver suite's, kept local
/// per the chain's per-file self-containment convention.
final class Qwen38ScorecardAuthorizedWorkerSpawnTests: XCTestCase {
    private static let rootSeed = Data(repeating: 0xA4, count: 32)
    private static let activeSeed = Data(repeating: 0xB5, count: 32)

    private static let claimHarnessGitSHA1 = String(repeating: "de", count: 20)
    private static let claimSourceID = String(repeating: "9a", count: 32)

    private var caseRoot: URL!

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
            "fast-mlx-qwen38-authorized-spawn-\(UUID().uuidString)"
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

    func testAuthorizedAccessorsExposeSignedClaimFields() throws {
        let resolved = try resolvedAuthorization()
        XCTAssertEqual(
            resolved.authorizedHarnessGitSHA1,
            Self.claimHarnessGitSHA1
        )
        XCTAssertEqual(resolved.authorizedSourceID, Self.claimSourceID)
    }

    /// The caller supplies a CONFLICTING environment (gdn variable "0" plus
    /// an unrelated variable) and free identity inputs do not exist on the
    /// API at all: the minted evidence must carry the SIGNED claim's
    /// harness/source identity and the forced-consistent environment
    /// observation.
    func testAuthorizedSpawnDerivesIdentityFromSignedClaimAndForcesEnv() throws {
        let resolved = try resolvedAuthorization()
        let stdinPipe = Pipe()
        let worker = try Qwen38ScorecardWorkerSpawner.spawnAndObserveAuthorized(
            authorization: resolved,
            executableURL: URL(fileURLWithPath: "/bin/cat"),
            arguments: [],
            environment: [
                Qwen38ScorecardWorkerSpawner.gdnEnvironmentKey: "0",
                "FASTMLX_TEST_UNRELATED": "x",
            ],
            standardInput: stdinPipe,
            gdnMode: .on
        )
        defer {
            try? stdinPipe.fileHandleForWriting.close()
            worker.process.waitUntilExit()
        }

        XCTAssertEqual(
            worker.evidence.harnessGitSHA,
            Self.claimHarnessGitSHA1
        )
        XCTAssertEqual(worker.evidence.sourceID, Self.claimSourceID)
        XCTAssertEqual(worker.evidence.gdnMode, .on)
        // Forced: gdn-on always observes the exact string "1" regardless
        // of the caller's conflicting value.
        XCTAssertEqual(worker.evidence.observedEnv, .enabled)
        XCTAssertEqual(worker.evidence.parentProcessID, Int(getpid()))

        try stdinPipe.fileHandleForWriting.close()
        worker.process.waitUntilExit()
        XCTAssertEqual(worker.process.terminationStatus, 0)
    }

    func testAuthorizedSpawnForcesGDNOffEnvironmentAbsent() throws {
        let resolved = try resolvedAuthorization()
        let stdinPipe = Pipe()
        let worker = try Qwen38ScorecardWorkerSpawner.spawnAndObserveAuthorized(
            authorization: resolved,
            executableURL: URL(fileURLWithPath: "/bin/cat"),
            arguments: [],
            environment: [
                Qwen38ScorecardWorkerSpawner.gdnEnvironmentKey: "1"
            ],
            standardInput: stdinPipe,
            gdnMode: .off
        )
        defer {
            try? stdinPipe.fileHandleForWriting.close()
            worker.process.waitUntilExit()
        }

        // Forced: gdn-off requires the variable ABSENT, so the observation
        // is disabled even though the caller tried to set it.
        XCTAssertEqual(worker.evidence.gdnMode, .off)
        XCTAssertEqual(worker.evidence.observedEnv, .disabled)
        XCTAssertEqual(
            worker.evidence.harnessGitSHA,
            Self.claimHarnessGitSHA1
        )
        XCTAssertEqual(worker.evidence.sourceID, Self.claimSourceID)

        try stdinPipe.fileHandleForWriting.close()
        worker.process.waitUntilExit()
        XCTAssertEqual(worker.process.terminationStatus, 0)
    }

    // MARK: - Fixtures

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
                modelSHA256: String(repeating: "21", count: 32),
                tokenizerSHA256: String(repeating: "43", count: 32),
                tensorManifestSHA256: String(repeating: "65", count: 32),
                chatTemplateSHA256: String(repeating: "87", count: 32),
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
                hostAdmissionID: String(repeating: "cb", count: 32),
                harnessGitSHA1: Self.claimHarnessGitSHA1,
                gdnOnMode: .on,
                gdnOffMode: .off,
                corpusID: "qwen38-27b-frozen-scorecard-workload-v2",
                corpusContentSHA256: String(repeating: "0f", count: 32),
                resultPairID: String(repeating: "f1", count: 32)
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
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
