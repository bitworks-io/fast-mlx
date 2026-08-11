import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class OperatorKeyPolicyTests: XCTestCase {
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
        .appendingPathComponent("fast-mlx-key-policy-\(UUID().uuidString)")
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

    func testCanonicalRootSignedPolicyAdmitsOnlyTypedCurrentRunClaimKey()
        throws
    {
        let policyFile = try capturePolicy(
            ProofControlKeyPolicyFixtures.policyBytes
        )

        XCTAssertEqual(
            try OperatorKeyPolicyVerifier.policyBytes(
                fields: ProofControlKeyPolicyFixtures.fields
            ),
            ProofControlKeyPolicyFixtures.policyBytes
        )

        let policy = try admit(policyFile)

        XCTAssertEqual(policy.file, policyFile)
        XCTAssertEqual(
            policy.policySHA256,
            ProofControlKeyPolicyFixtures.policySHA256
        )
        XCTAssertEqual(policy.policyGeneration, 7)
        XCTAssertEqual(policy.validFromUnixSeconds, 1_900_000_000)
        XCTAssertEqual(policy.validUntilUnixSeconds, 2_100_000_000)
        XCTAssertEqual(
            policy.rootKeyID,
            ProofControlKeyPolicyFixtures.rootKeyID
        )
        XCTAssertEqual(
            policy.activeOperatorKeyID,
            ProofControlKeyPolicyFixtures.activeKeyID
        )
        XCTAssertEqual(
            policy.activeOperatorPublicKeyBase64,
            ProofControlKeyPolicyFixtures.activePublicKeyBase64
        )
        XCTAssertEqual(policy.activeOperatorScope, .runClaim)
        XCTAssertEqual(
            policy.allowedClaimSubject,
            .absorbedMLALoadedResultPair
        )
        XCTAssertEqual(policy.revokedOperatorKeyIDs, [])
        XCTAssertEqual(
            policy.rootSignatureSHA256,
            ProofControlKeyPolicyFixtures.policySignatureSHA256
        )
        XCTAssertEqual(
            policy.admissionID.rawValue,
            ProofControlKeyPolicyFixtures.policyAdmissionID
        )
        XCTAssertNotEqual(
            policy.admissionID.rawValue,
            policy.policySHA256
        )
        XCTAssertNotEqual(
            policy.admissionID.rawValue,
            policy.activeOperatorKeyID
        )
    }

    func testRejectsNonCanonicalPolicyStructureAndScalars() throws {
        var reorderedLines = String(
            decoding: ProofControlKeyPolicyFixtures.policyBytes,
            as: UTF8.self
        )
        .split(separator: "\n", omittingEmptySubsequences: false)
        reorderedLines.swapAt(3, 4)

        let cases: [Data] = [
            replacing(
                "fast-mlx-proof-control-operator-key-policy-v1",
                with: "other-policy-v1"
            ),
            replacing(
                "subject=operator-run-signing-key-policy",
                with: "subject=other"
            ),
            replacing(
                "root_key_id=\(ProofControlKeyPolicyFixtures.rootKeyID)",
                with:
                    "root_key_id=" +
                    ProofControlKeyPolicyFixtures.rootKeyID.uppercased()
            ),
            replacing("policy_generation=7", with: "policy_generation=07"),
            replacing("policy_generation=7", with: "policy_generation=0"),
            replacing(
                "policy_generation=7",
                with: "policy_generation=18446744073709551616"
            ),
            replacing(
                "valid_from_unix_seconds=1900000000",
                with: "valid_from_unix_seconds=2100000001"
            ),
            replacing(
                "active_operator_key_id=" +
                    ProofControlKeyPolicyFixtures.activeKeyID,
                with:
                    "active_operator_key_id=" +
                    ProofControlKeyPolicyFixtures.activeKeyID.uppercased()
            ),
            replacing(
                "active_operator_public_key_base64=" +
                    ProofControlKeyPolicyFixtures.activePublicKeyBase64,
                with:
                    "active_operator_public_key_base64=" +
                    ProofControlKeyPolicyFixtures.activePublicKeyBase64 + "\n"
            ),
            replacing(
                "active_operator_scope=run-claim",
                with: "active_operator_scope=source-manifest"
            ),
            replacing(
                "allowed_claim_subject=absorbed-mla-loaded-result-pair",
                with: "allowed_claim_subject=other"
            ),
            ProofControlKeyPolicyFixtures.policyBytes + Data("extra=true\n".utf8),
            Data(ProofControlKeyPolicyFixtures.policyBytes.dropLast()),
            replacing(
                "policy_generation=7\n",
                with: ""
            ),
            replacing(
                "policy_generation=7\n",
                with: "policy_generation=7\npolicy_generation=7\n"
            ),
            Data(
                reorderedLines.map(String.init).joined(separator: "\n").utf8
            ),
            Data(
                String(
                    decoding: ProofControlKeyPolicyFixtures.policyBytes,
                    as: UTF8.self
                )
                .replacingOccurrences(of: "\n", with: "\r\n")
                .utf8
            ),
            Data([0xff]),
        ]

        for bytes in cases {
            let file = try capturePolicy(bytes)
            XCTAssertThrowsError(
                try admit(
                    file,
                    expectedPolicySHA256: file.sha256
                )
            ) { error in
                XCTAssertEqual(
                    error as? OperatorKeyPolicyError,
                    .nonCanonicalPolicy
                )
            }
        }
    }

    func testRejectsRevocationConfusionAndKeyIdentityMismatch() throws {
        let otherID =
            "0000000000000000000000000000000000000000000000000000000000000001"
        let laterID =
            "0000000000000000000000000000000000000000000000000000000000000002"
        let nonCanonicalRevocations: [
            (Data, OperatorKeyPolicyError)
        ] = [
            (
                replacing(
                    "revoked_operator_key_ids=none",
                    with: "revoked_operator_key_ids=\(laterID),\(otherID)"
                ),
                .nonCanonicalPolicy
            ),
            (
                replacing(
                    "revoked_operator_key_ids=none",
                    with: "revoked_operator_key_ids=\(otherID),\(otherID)"
                ),
                .nonCanonicalPolicy
            ),
            (
                replacing(
                    "revoked_operator_key_ids=none",
                    with:
                        "revoked_operator_key_ids=" +
                        ProofControlKeyPolicyFixtures.activeKeyID
                ),
                .activeOperatorKeyRevoked
            ),
            (
                replacing(
                    "revoked_operator_key_ids=none",
                    with:
                        "revoked_operator_key_ids=" +
                        ProofControlKeyPolicyFixtures.rootKeyID
                ),
                .rootKeyRevocationUnsupported
            ),
            (
                replacing(
                    replacing(
                        "active_operator_key_id=" +
                            ProofControlKeyPolicyFixtures.activeKeyID,
                        with:
                            "active_operator_key_id=" +
                            ProofControlKeyPolicyFixtures.rootKeyID
                    ),
                    source:
                        "active_operator_public_key_base64=" +
                        ProofControlKeyPolicyFixtures.activePublicKeyBase64,
                    with:
                        "active_operator_public_key_base64=" +
                        ProofControlKeyPolicyFixtures.rootPublicKeyBase64
                ),
                .rootAndActiveKeysMustDiffer
            ),
            (
                replacing(
                    "active_operator_key_id=" +
                        ProofControlKeyPolicyFixtures.activeKeyID,
                    with:
                        "active_operator_key_id=" +
                        ProofControlKeyPolicyFixtures.differentSHA256
                ),
                .activeOperatorKeyIDMismatch
            ),
        ]

        for (bytes, expectedError) in nonCanonicalRevocations {
            let file = try capturePolicy(bytes)
            XCTAssertThrowsError(
                try admit(
                    file,
                    expectedPolicySHA256: file.sha256
                )
            ) { error in
                XCTAssertEqual(
                    error as? OperatorKeyPolicyError,
                    expectedError
                )
            }
        }

        let canonicalRevokedFields = OperatorKeyPolicyFields(
            rootKeyID: ProofControlKeyPolicyFixtures.rootKeyID,
            policyGeneration: 8,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            activeOperatorKeyID: ProofControlKeyPolicyFixtures.activeKeyID,
            activeOperatorPublicKeyBase64:
                ProofControlKeyPolicyFixtures.activePublicKeyBase64,
            activeOperatorScope: .runClaim,
            allowedClaimSubject: .absorbedMLALoadedResultPair,
            revokedOperatorKeyIDs: [otherID, laterID]
        )
        let canonical = try OperatorKeyPolicyVerifier.policyBytes(
            fields: canonicalRevokedFields
        )
        XCTAssertTrue(
            String(decoding: canonical, as: UTF8.self).contains(
                "revoked_operator_key_ids=\(otherID),\(laterID)\n"
            )
        )
    }

    func testRejectsUntrustedCurrentPolicyRollbackAndInvalidTime()
        throws
    {
        let policyFile = try capturePolicy(
            ProofControlKeyPolicyFixtures.policyBytes
        )

        let contexts: [
            (OperatorKeyPolicyTrustAnchor, OperatorKeyPolicyError)
        ] = [
            (
                trustAnchor(
                    expectedPolicySHA256:
                        ProofControlKeyPolicyFixtures.differentSHA256
                ),
                .policyDigestMismatch
            ),
            (
                trustAnchor(minimumPolicyGeneration: 8),
                .policyGenerationRollback(minimum: 8, actual: 7)
            ),
            (
                trustAnchor(verificationUnixSeconds: 1_899_999_999),
                .policyNotYetValid
            ),
            (
                trustAnchor(verificationUnixSeconds: 2_100_000_001),
                .policyExpired
            ),
        ]

        for (anchor, expectedError) in contexts {
            XCTAssertThrowsError(
                try admit(policyFile, trustAnchor: anchor)
            ) { error in
                XCTAssertEqual(
                    error as? OperatorKeyPolicyError,
                    expectedError
                )
            }
        }

        for boundary in [UInt64(1_900_000_000), 2_100_000_000] {
            XCTAssertNoThrow(
                try admit(
                    policyFile,
                    trustAnchor: trustAnchor(
                        verificationUnixSeconds: boundary
                    )
                )
            )
        }

        let invalidRootID = trustAnchor(
            rootKeyID:
                ProofControlKeyPolicyFixtures.rootKeyID.uppercased()
        )
        XCTAssertThrowsError(
            try admit(policyFile, trustAnchor: invalidRootID)
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .invalidTrustAnchor(.rootKeyID)
            )
        }
        let invalidPolicyDigest = trustAnchor(
            expectedPolicySHA256:
                ProofControlKeyPolicyFixtures.policySHA256.uppercased()
        )
        XCTAssertThrowsError(
            try admit(policyFile, trustAnchor: invalidPolicyDigest)
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            )
        }

        let wrongRootID = trustAnchor(
            rootKeyID: ProofControlKeyPolicyFixtures.differentSHA256
        )
        XCTAssertThrowsError(
            try admit(policyFile, trustAnchor: wrongRootID)
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .rootPublicKeyIDMismatch
            )
        }

        let activeAsRoot = trustAnchor(
            rootPublicKeyBase64:
                ProofControlKeyPolicyFixtures.activePublicKeyBase64,
            rootKeyID: ProofControlKeyPolicyFixtures.activeKeyID
        )
        XCTAssertThrowsError(
            try admit(policyFile, trustAnchor: activeAsRoot)
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .rootKeyIDMismatch
            )
        }
    }

    func testRejectsMalformedRootAndDigestOrForeignSignatureReplay()
        throws
    {
        let policyFile = try capturePolicy(
            ProofControlKeyPolicyFixtures.policyBytes
        )
        let malformedRoot = trustAnchor(
            rootPublicKeyBase64:
                ProofControlKeyPolicyFixtures.rootPublicKeyBase64 + "\n"
        )
        XCTAssertThrowsError(
            try admit(policyFile, trustAnchor: malformedRoot)
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .invalidRootPublicKeyEncoding
            )
        }

        XCTAssertThrowsError(
            try admit(
                policyFile,
                rootSignatureBase64:
                    ProofControlKeyPolicyFixtures.policySignatureBase64 + "\n"
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .invalidRootSignatureEncoding
            )
        }
        XCTAssertThrowsError(
            try admit(
                policyFile,
                rootSignatureBase64:
                    ProofControlKeyPolicyFixtures
                        .policyRawDigestSignatureBase64
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .rootSignatureRejected
            )
        }
        XCTAssertThrowsError(
            try admit(
                policyFile,
                rootSignatureBase64:
                    ProofControlKeyPolicyFixtures.foreignSignatureBase64
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorKeyPolicyError,
                .rootSignatureRejected
            )
        }

        let rewrittenURL = caseRoot.appendingPathComponent("rewritten.policy")
        try ProofControlKeyPolicyFixtures.policyBytes.write(to: rewrittenURL)
        let snapshot = try AdmittedFile.capture(
            absolutePath: rewrittenURL.path,
            maximumBytes: 4_096
        )
        try Data("replaced\n".utf8).write(to: rewrittenURL)
        let admitted = try admit(snapshot)
        XCTAssertEqual(
            admitted.policySHA256,
            ProofControlKeyPolicyFixtures.policySHA256
        )
    }

    private func capturePolicy(_ bytes: Data) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent("\(UUID().uuidString).policy")
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 4_096
        )
    }

    private func admit(
        _ file: AdmittedFile,
        rootSignatureBase64: String =
            ProofControlKeyPolicyFixtures.policySignatureBase64,
        expectedPolicySHA256: String? = nil,
        trustAnchor: OperatorKeyPolicyTrustAnchor? = nil
    ) throws -> AdmittedOperatorKeyPolicy {
        try OperatorKeyPolicyVerifier.admit(
            policyFile: file,
            rootSignatureBase64: rootSignatureBase64,
            trustAnchor: trustAnchor ?? self.trustAnchor(
                expectedPolicySHA256:
                    expectedPolicySHA256 ??
                    ProofControlKeyPolicyFixtures.policySHA256
            )
        )
    }

    private func trustAnchor(
        rootPublicKeyBase64: String =
            ProofControlKeyPolicyFixtures.rootPublicKeyBase64,
        rootKeyID: String = ProofControlKeyPolicyFixtures.rootKeyID,
        expectedPolicySHA256: String =
            ProofControlKeyPolicyFixtures.policySHA256,
        minimumPolicyGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> OperatorKeyPolicyTrustAnchor {
        OperatorKeyPolicyTrustAnchor(
            rootPublicKeyBase64: rootPublicKeyBase64,
            rootKeyID: rootKeyID,
            expectedCurrentPolicySHA256: expectedPolicySHA256,
            minimumPolicyGeneration: minimumPolicyGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(
            decoding: ProofControlKeyPolicyFixtures.policyBytes,
            as: UTF8.self
        )
        return Data(
            text.replacingOccurrences(of: source, with: replacement).utf8
        )
    }

    private func replacing(
        _ bytes: Data,
        source: String,
        with replacement: String
    ) -> Data {
        let text = String(decoding: bytes, as: UTF8.self)
        return Data(
            text.replacingOccurrences(of: source, with: replacement).utf8
        )
    }
}

enum ProofControlKeyPolicyFixtures {
    static let rootPublicKeyBase64 =
        "luRNfpb0qNTRsACWkkndKnKIdMDgY7qTUdKV/C9TiNc="
    static let rootKeyID =
        "05bf822d8ee0fbdf448e7f553cc1e600c2089bf5f0e0a3710fcbda5f6eee18b1"
    static let activePublicKeyBase64 =
        "RVQ/DRmlz4JFKZol3QCNwrWeSTJyo2S6oP2D2Xa/oxM="
    static let activeKeyID =
        "78f493e824b8b0bc797304dbdb1948f71b9a290fbeec3288831a07f7786cf039"
    static let policySHA256 =
        "9a520930fc41223bbfe69713e394c1816152f5de8c91aabee7b12ec7636cd939"
    static let policySignatureBase64 =
        "TXoZjGZWA+Zq1OqV5t0s3bpRwTjFQLUG25OQz2oZbCveIL4GTrqefx/rC1Vhx32wan+PTHdOeRtXpCLJ1s6NBw=="
    static let policySignatureSHA256 =
        "c61b071e0af9c89929b45bc005b69edd2922d2756db5990e3e4266ab5d246fb0"
    static let policyAdmissionID =
        "06bbd754822d69e3dc91c01b08316f1bfff9a2f3f1bb41805e97ca66982e16b2"
    static let policyRawDigestSignatureBase64 =
        "CeJcT1An34xJAKfjoJeSrvmZ/9t79b3yU3P+Z4/V3NXwZkkaVLn5DXVRN1wBqHYV9iySl6S8XLCaLpErji9JBg=="
    static let foreignSignatureBase64 =
        "EZ/V8V4zIRVxMBqq2r+7vSEO1QVPVAZUbLwVzPjeAwoEAEaubl/UE+1P+gEWNv8n76m2pqkSpPtInNx4V4BeDA=="
    static let differentSHA256 =
        "0000000000000000000000000000000000000000000000000000000000000003"

    static let fields = OperatorKeyPolicyFields(
        rootKeyID: rootKeyID,
        policyGeneration: 7,
        validFromUnixSeconds: 1_900_000_000,
        validUntilUnixSeconds: 2_100_000_000,
        activeOperatorKeyID: activeKeyID,
        activeOperatorPublicKeyBase64: activePublicKeyBase64,
        activeOperatorScope: .runClaim,
        allowedClaimSubject: .absorbedMLALoadedResultPair,
        revokedOperatorKeyIDs: []
    )

    static let policyBytes = Data(
        """
        fast-mlx-proof-control-operator-key-policy-v1
        subject=operator-run-signing-key-policy
        root_key_id=\(rootKeyID)
        policy_generation=7
        valid_from_unix_seconds=1900000000
        valid_until_unix_seconds=2100000000
        active_operator_key_id=\(activeKeyID)
        active_operator_public_key_base64=\(activePublicKeyBase64)
        active_operator_scope=run-claim
        allowed_claim_subject=absorbed-mla-loaded-result-pair
        revoked_operator_key_ids=none

        """.utf8
    )
}
