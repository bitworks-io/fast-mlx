import CryptoKit
import Darwin
import Foundation
import XCTest

@testable import ProofControl

/// Slice 3: resolved run authorization — the REAL Ed25519 verification of
/// the operator's claim signature against the Slice 2-admitted active
/// operator key. Fixture keys are deterministic test seeds only; no real
/// operator credential exists anywhere in this suite (the chain stops at
/// the operator-signing credential boundary).
///
/// Unreachable-by-construction guards (subject/scope/revocation re-guards
/// in `resolve`) are deliberately NOT unit-forced: both input types pin
/// those invariants structurally and expose no init that could violate
/// them, so forcing the guards would require weakening the types. They
/// are defense-in-depth against future refactors, documented in the
/// implementation.
final class Qwen38ScorecardResolvedRunAuthorizationTests: XCTestCase {
    // Deterministic fixture seeds — DISTINCT from any real key material.
    private static let rootSeed = Data(repeating: 0xA1, count: 32)
    private static let activeSeed = Data(repeating: 0xB2, count: 32)
    private static let foreignSeed = Data(repeating: 0xC3, count: 32)

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
            "fast-mlx-qwen38-resolved-auth-\(UUID().uuidString)"
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

    // MARK: - Happy path

    func testResolveAcceptsOperatorSignedClaim() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let operatorKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let signature = try operatorKey.signature(for: claimBytes)

        let resolved = try Qwen38ScorecardRunAuthorizationResolver.resolve(
            claimBytes: claimBytes,
            claimSignatureBase64: signature.base64EncodedString(),
            policy: policy
        )

        XCTAssertEqual(resolved.policyAdmissionID, policy.admissionID)
        XCTAssertEqual(
            resolved.operatorKeyID,
            Self.sha256Hex(operatorKey.publicKey.rawRepresentation)
        )
        // Digests recomputed independently in the test (CryptoKit direct),
        // not read back from the code under test.
        XCTAssertEqual(resolved.claimSHA256, Self.sha256Hex(claimBytes))
        XCTAssertEqual(resolved.signatureSHA256, Self.sha256Hex(signature))
        XCTAssertEqual(resolved.claim.subject, .mtpScorecardResultPair)
        XCTAssertEqual(resolved.claim.claimSHA256, resolved.claimSHA256)

        // Independent recomputation of the authorization ID recipe.
        let expectedAuthorizationID = Self.sha256Hex(
            Data(
                ([
                    "fastmlx-qwen38-scorecard-resolved-run-authorization-id-v1",
                    "policy_admission_id=\(policy.admissionID.rawValue)",
                    "operator_key_id=\(resolved.operatorKeyID)",
                    "claim_sha256=\(resolved.claimSHA256)",
                    "signature_sha256=\(resolved.signatureSHA256)",
                ].joined(separator: "\n") + "\n").utf8
            )
        )
        XCTAssertEqual(resolved.authorizationID, expectedAuthorizationID)
    }

    func testAuthorizationIDDomainPinned() {
        XCTAssertEqual(
            Qwen38ScorecardRunAuthorizationResolver.authorizationIDDomain,
            "fastmlx-qwen38-scorecard-resolved-run-authorization-id-v1"
        )
    }

    /// Apple CryptoKit Ed25519 signatures are randomized: two signings of
    /// the same claim produce distinct signatures, and the authorization
    /// ID deliberately binds the SIGNATURE INSTANCE (a distinct signing
    /// event), not merely the claim content.
    func testDistinctSignatureInstancesYieldDistinctAuthorizationIDs() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let operatorKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let first = try operatorKey.signature(for: claimBytes)
        let second = try operatorKey.signature(for: claimBytes)
        XCTAssertNotEqual(first, second)

        let firstResolved = try Qwen38ScorecardRunAuthorizationResolver.resolve(
            claimBytes: claimBytes,
            claimSignatureBase64: first.base64EncodedString(),
            policy: policy
        )
        let secondResolved = try Qwen38ScorecardRunAuthorizationResolver.resolve(
            claimBytes: claimBytes,
            claimSignatureBase64: second.base64EncodedString(),
            policy: policy
        )

        XCTAssertEqual(firstResolved.claimSHA256, secondResolved.claimSHA256)
        XCTAssertNotEqual(
            firstResolved.signatureSHA256,
            secondResolved.signatureSHA256
        )
        XCTAssertNotEqual(
            firstResolved.authorizationID,
            secondResolved.authorizationID
        )
    }

    // MARK: - Signature rejection

    func testResolveRejectsSignatureByDifferentKey() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let foreignKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.foreignSeed
        )
        let signature = try foreignKey.signature(for: claimBytes)

        XCTAssertThrowsError(
            try Qwen38ScorecardRunAuthorizationResolver.resolve(
                claimBytes: claimBytes,
                claimSignatureBase64: signature.base64EncodedString(),
                policy: policy
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardRunAuthorizationError,
                .claimSignatureRejected
            )
        }
    }

    func testResolveRejectsSignatureOverDifferentBytes() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let operatorKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let signature = try operatorKey.signature(
            for: Data("not the claim".utf8)
        )

        XCTAssertThrowsError(
            try Qwen38ScorecardRunAuthorizationResolver.resolve(
                claimBytes: claimBytes,
                claimSignatureBase64: signature.base64EncodedString(),
                policy: policy
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardRunAuthorizationError,
                .claimSignatureRejected
            )
        }
    }

    /// A value-level tamper that keeps the claim canonically well-formed
    /// (hex digit substituted for another hex digit) must be caught by the
    /// SIGNATURE, not by the parser.
    func testResolveRejectsCanonicalValueTamper() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let operatorKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let signature = try operatorKey.signature(for: claimBytes)

        var tampered = claimBytes
        // Penultimate byte is the last hex digit of result_pair_id
        // (final byte is the trailing newline). Swap within the hex
        // alphabet so the claim stays structurally canonical.
        let index = tampered.count - 2
        tampered[index] = tampered[index] == UInt8(ascii: "a")
            ? UInt8(ascii: "b")
            : UInt8(ascii: "a")
        XCTAssertNotEqual(tampered, claimBytes)

        XCTAssertThrowsError(
            try Qwen38ScorecardRunAuthorizationResolver.resolve(
                claimBytes: tampered,
                claimSignatureBase64: signature.base64EncodedString(),
                policy: policy
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardRunAuthorizationError,
                .claimSignatureRejected
            )
        }
    }

    func testResolveSurfacesStructuralClaimTamper() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()
        let operatorKey = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let signature = try operatorKey.signature(for: claimBytes)

        // Dropping the trailing newline breaks the canonical form; the
        // Slice 1 structural parser must reject before any crypto runs.
        let structural = claimBytes.dropLast()
        XCTAssertThrowsError(
            try Qwen38ScorecardRunAuthorizationResolver.resolve(
                claimBytes: Data(structural),
                claimSignatureBase64: signature.base64EncodedString(),
                policy: policy
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardRunClaimError,
                .nonCanonicalClaim
            )
        }
    }

    func testResolveSurfacesStructuralSignatureEncodingErrors() throws {
        let policy = try admittedPolicy()
        let claimBytes = try Self.claimBytes()

        let rejected = [
            "not base64!",
            Data(repeating: 7, count: 63).base64EncodedString(),
            Data(repeating: 7, count: 65).base64EncodedString(),
        ]
        for signatureBase64 in rejected {
            XCTAssertThrowsError(
                try Qwen38ScorecardRunAuthorizationResolver.resolve(
                    claimBytes: claimBytes,
                    claimSignatureBase64: signatureBase64,
                    policy: policy
                ),
                signatureBase64
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardRunClaimError,
                    .invalidSignatureEncoding,
                    signatureBase64
                )
            }
        }
    }

    // MARK: - Fixtures

    private static func claimBytes() throws -> Data {
        try Qwen38ScorecardRunClaimVerifier.claimBytes(
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
                sourceID: String(repeating: "9a", count: 32),
                hostAdmissionID: String(repeating: "bc", count: 32),
                harnessGitSHA1: String(repeating: "de", count: 20),
                gdnOnMode: .on,
                gdnOffMode: .off,
                corpusID: "qwen38-27b-frozen-scorecard-workload-v2",
                corpusContentSHA256: String(repeating: "f0", count: 32),
                resultPairID: String(repeating: "1f", count: 32)
            )
        )
    }

    private func admittedPolicy() throws -> AdmittedQwen38ScorecardKeyPolicy {
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
                policyGeneration: 5,
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
        return try Qwen38ScorecardKeyPolicyVerifier.admit(
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
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
