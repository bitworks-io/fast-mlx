import CryptoKit
import Darwin
import Foundation
import XCTest

@testable import ProofControl

/// Chain Slice 2: root-signed operator key-policy admission for the Qwen38
/// scorecard domain. This is a NEW policy domain document — own domain,
/// subject, scope, and admitted-ID domain strings — that names the trusted
/// operator run-signing key for the Slice 1 claim domain. It deliberately
/// shares no code with the frozen absorbed-MLA `OperatorKeyPolicyVerifier`
/// (docs/task-inbox/2026-09-01-qwen38-proof-runner-scope-and-chain-design.md,
/// binding items 2 and 4). Unlike Slice 1's structural-only claim check,
/// admission here performs the REAL Ed25519 root-signature verification over
/// the exact policy bytes: a root-authenticated signer policy is the
/// prerequisite Slice 3 verifies claim signatures against.
///
/// The golden document digest below was computed OUT OF BAND with
/// `shasum -a 256` over the exact 574 canonical bytes, not with this
/// module's own hashing helpers. The two Ed25519 key constants were derived
/// once from the deterministic seeds used in this file and pinned; the
/// crypto-positive paths re-derive the same keys at runtime and sign
/// in-process (CryptoKit Ed25519 signatures are randomized, so signature
/// digests are asserted structurally, never pinned).
final class Qwen38ScorecardKeyPolicyTests: XCTestCase {
    // Deterministic Ed25519 seeds (test fixtures only, never operator
    // material). Derived public identities are pinned below.
    // gitleaks:allow
    private static let rootSeed = Data((0..<32).map { UInt8(1 + $0) })
    // gitleaks:allow
    private static let activeSeed = Data((0..<32).map { UInt8(65 + $0) })
    // gitleaks:allow
    private static let foreignRootSeed = Data((0..<32).map { UInt8(129 + $0) })

    private static let rootKeyID =
        "65b60673d6ed884bf01c2c222d82ada0740f29ac3355d6a925c81f17f47a27b8"
    private static let activePublicKeyBase64 =
        "rcFAEfgtHFbZVqpPnXPYhYNhpgYEhSXg0Ixjjcdd2Mc="
    private static let activeKeyID =
        "ba8112fa4ba3d6f934b2ad2aa06966023d56e1eeb0b2d5b74b5dd6ec152ba690"

    /// `shasum -a 256` over the exact golden bytes (574 bytes), computed
    /// outside this test target.
    private static let goldenPolicySHA256 =
        "5a2d5669a6999847655305a003b0cc0c63588297e5a329cdc1f8f9493c90c0e3"

    private static let goldenPolicyText = """
        fastmlx-qwen38-scorecard-operator-key-policy-v1
        subject=qwen38-scorecard-operator-run-signing-key-policy
        root_key_id=65b60673d6ed884bf01c2c222d82ada0740f29ac3355d6a925c81f17f47a27b8
        policy_generation=7
        valid_from_unix_seconds=1700000000
        valid_until_unix_seconds=1800000000
        active_operator_key_id=ba8112fa4ba3d6f934b2ad2aa06966023d56e1eeb0b2d5b74b5dd6ec152ba690
        active_operator_public_key_base64=rcFAEfgtHFbZVqpPnXPYhYNhpgYEhSXg0Ixjjcdd2Mc=
        active_operator_scope=qwen38-scorecard-run-claim
        allowed_claim_subject=qwen38-mtp-scorecard-result-pair
        revoked_operator_key_ids=none

        """

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
        .appendingPathComponent("fast-mlx-qwen38-key-policy-\(UUID().uuidString)")
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

    // MARK: - Golden bytes against the out-of-band oracle

    func testPinnedActiveKeyConstantsDeriveFromActiveSeed() throws {
        // Binds the pinned public identity to its seed so the golden
        // document's active key is provably the deterministic fixture key
        // (Slice 3's claim-signing tests will sign with this seed).
        let active = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        XCTAssertEqual(
            active.publicKey.rawRepresentation.base64EncodedString(),
            Self.activePublicKeyBase64
        )
        XCTAssertEqual(
            Self.sha256Hex(active.publicKey.rawRepresentation),
            Self.activeKeyID
        )
    }

    func testGoldenPolicyBytesMatchIndependentOracle() throws {
        let bytes = try Qwen38ScorecardKeyPolicyVerifier.policyBytes(
            fields: Self.goldenFields()
        )

        XCTAssertEqual(bytes, Data(Self.goldenPolicyText.utf8))
        XCTAssertEqual(bytes.count, 574)
        XCTAssertEqual(Self.sha256Hex(bytes), Self.goldenPolicySHA256)
    }

    // MARK: - Canonical root-signed admission

    func testCanonicalRootSignedPolicyAdmits() throws {
        let root = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.rootSeed
        )
        let policyFile = try capturePolicy(Data(Self.goldenPolicyText.utf8))
        let signature = try root.signature(for: policyFile.bytes)

        let admitted = try Qwen38ScorecardKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: signature.base64EncodedString(),
            trustAnchor: trustAnchor()
        )

        XCTAssertEqual(admitted.policySHA256, Self.goldenPolicySHA256)
        XCTAssertEqual(admitted.policyGeneration, 7)
        XCTAssertEqual(admitted.validFromUnixSeconds, 1_700_000_000)
        XCTAssertEqual(admitted.validUntilUnixSeconds, 1_800_000_000)
        XCTAssertEqual(admitted.rootKeyID, Self.rootKeyID)
        XCTAssertEqual(admitted.activeOperatorKeyID, Self.activeKeyID)
        XCTAssertEqual(
            admitted.activeOperatorPublicKeyBase64,
            Self.activePublicKeyBase64
        )
        XCTAssertEqual(admitted.activeOperatorScope, .scorecardRunClaim)
        XCTAssertEqual(admitted.allowedClaimSubject, .mtpScorecardResultPair)
        XCTAssertEqual(admitted.revokedOperatorKeyIDs, [])
        XCTAssertEqual(
            admitted.rootSignatureSHA256,
            Self.sha256Hex(signature)
        )

        // The admitted-ID format is pinned by a test-local recomputation of
        // the documented four-line construction under the Qwen38 ID domain.
        let expectedID = Self.sha256Hex(
            Data(
                ([
                    "fastmlx-qwen38-admitted-scorecard-operator-key-policy-id-v1",
                    "root_key_id=\(Self.rootKeyID)",
                    "policy_sha256=\(Self.goldenPolicySHA256)",
                    "signature_sha256=\(Self.sha256Hex(signature))",
                ].joined(separator: "\n") + "\n").utf8
            )
        )
        XCTAssertEqual(admitted.admissionID.rawValue, expectedID)
    }

    func testAdmitsAtInclusiveValidityBoundaries() throws {
        for verificationSeconds: UInt64 in [1_700_000_000, 1_800_000_000] {
            let (policyFile, signatureBase64) = try signedGoldenPolicy()
            XCTAssertNoThrow(
                try Qwen38ScorecardKeyPolicyVerifier.admit(
                    policyFile: policyFile,
                    rootSignatureBase64: signatureBase64,
                    trustAnchor: trustAnchor(
                        verificationUnixSeconds: verificationSeconds
                    )
                ),
                "verification at \(verificationSeconds) must be inside the inclusive window"
            )
        }
    }

    func testCanonicalRevocationListStillAdmitsAndIsPreserved() throws {
        let revokedA = String(repeating: "1", count: 64)
        let revokedB = String(repeating: "2", count: 64)
        let bytes = try Qwen38ScorecardKeyPolicyVerifier.policyBytes(
            fields: Self.goldenFields(
                revokedOperatorKeyIDs: [revokedA, revokedB]
            )
        )
        let (policyFile, signatureBase64) = try signedPolicy(bytes)

        let admitted = try Qwen38ScorecardKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: signatureBase64,
            trustAnchor: trustAnchor(
                expectedPolicySHA256: Self.sha256Hex(bytes)
            )
        )

        XCTAssertEqual(admitted.revokedOperatorKeyIDs, [revokedA, revokedB])
    }

    // MARK: - Non-canonical structure refusal

    func testNonCanonicalPolicyStructureRefused() throws {
        let golden = Self.goldenPolicyText
        let cases: [(String, Data)] = [
            ("empty", Data()),
            ("not utf8", Data([0xff, 0xfe, 0x00, 0x41])),
            ("missing trailing newline", Data(golden.dropLast().utf8)),
            ("extra trailing newline", Data((golden + "\n").utf8)),
            ("trailing junk line", Data((golden + "extra=1\n").utf8)),
            (
                "dropped generation line",
                replacing(golden, "policy_generation=7\n", with: "")
            ),
            (
                "reordered subject after root key",
                Data(
                    golden.split(
                        separator: "\n",
                        omittingEmptySubsequences: false
                    )
                    .enumerated()
                    .sorted { lhs, rhs in
                        let order = [0, 2, 1] + Array(3..<12)
                        return order[lhs.offset] < order[rhs.offset]
                    }
                    .map(\.element)
                    .joined(separator: "\n")
                    .utf8
                )
            ),
            (
                "uppercase root key hex",
                replacing(
                    golden,
                    "root_key_id=65b6",
                    with: "root_key_id=65B6"
                )
            ),
            (
                "leading-zero generation",
                replacing(
                    golden,
                    "policy_generation=7",
                    with: "policy_generation=07"
                )
            ),
            (
                "non-decimal validity",
                replacing(
                    golden,
                    "valid_from_unix_seconds=1700000000",
                    with: "valid_from_unix_seconds=17e8"
                )
            ),
            (
                "short active key id",
                replacing(
                    golden,
                    "active_operator_key_id=\(Self.activeKeyID)",
                    with: "active_operator_key_id=\(Self.activeKeyID.dropLast())"
                )
            ),
            (
                "inverted validity window",
                replacing(
                    golden,
                    "valid_until_unix_seconds=1800000000",
                    with: "valid_until_unix_seconds=1600000000"
                )
            ),
        ]

        for (label, bytes) in cases {
            let policyFile = try capturePolicy(bytes)
            XCTAssertThrowsError(
                try admitGoldenAnchored(
                    policyFile,
                    expectedPolicySHA256: Self.sha256Hex(bytes)
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    .nonCanonicalPolicy,
                    label
                )
            }
        }
    }

    func testNonBase64ActivePublicKeyRefusedWithTypedError() throws {
        let bytes = replacing(
            Self.goldenPolicyText,
            "active_operator_public_key_base64=\(Self.activePublicKeyBase64)",
            with: "active_operator_public_key_base64=!!notbase64!!"
        )
        let policyFile = try capturePolicy(bytes)

        XCTAssertThrowsError(
            try admitGoldenAnchored(
                policyFile,
                expectedPolicySHA256: Self.sha256Hex(bytes)
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardKeyPolicyError,
                .invalidActiveOperatorPublicKeyEncoding
            )
        }
    }

    // MARK: - Cross-domain confusion

    func testCanonicalAbsorbedMLAPolicyDocumentRefused() throws {
        // A fully canonical FROZEN-domain policy document (built with the
        // frozen public API and a real key) must never parse in the Qwen38
        // domain: the domain line differs on line 0.
        let frozenBytes = try OperatorKeyPolicyVerifier.policyBytes(
            fields: OperatorKeyPolicyFields(
                rootKeyID: Self.rootKeyID,
                policyGeneration: 7,
                validFromUnixSeconds: 1_700_000_000,
                validUntilUnixSeconds: 1_800_000_000,
                activeOperatorKeyID: Self.activeKeyID,
                activeOperatorPublicKeyBase64: Self.activePublicKeyBase64,
                activeOperatorScope: .runClaim,
                allowedClaimSubject: .absorbedMLALoadedResultPair,
                revokedOperatorKeyIDs: []
            )
        )
        let policyFile = try capturePolicy(frozenBytes)

        XCTAssertThrowsError(
            try admitGoldenAnchored(
                policyFile,
                expectedPolicySHA256: Self.sha256Hex(frozenBytes)
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardKeyPolicyError,
                .nonCanonicalPolicy
            )
        }
    }

    func testForeignScopeAndSubjectValuesRefused() throws {
        let cases: [(String, Data)] = [
            (
                "frozen scope literal",
                replacing(
                    Self.goldenPolicyText,
                    "active_operator_scope=qwen38-scorecard-run-claim",
                    with: "active_operator_scope=run-claim"
                )
            ),
            (
                "absorbed-mla claim subject",
                replacing(
                    Self.goldenPolicyText,
                    "allowed_claim_subject=qwen38-mtp-scorecard-result-pair",
                    with: "allowed_claim_subject=absorbed-mla-loaded-result-pair"
                )
            ),
            (
                "frozen policy subject",
                replacing(
                    Self.goldenPolicyText,
                    "subject=qwen38-scorecard-operator-run-signing-key-policy",
                    with: "subject=operator-run-signing-key-policy"
                )
            ),
        ]

        for (label, bytes) in cases {
            let policyFile = try capturePolicy(bytes)
            XCTAssertThrowsError(
                try admitGoldenAnchored(
                    policyFile,
                    expectedPolicySHA256: Self.sha256Hex(bytes)
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    .nonCanonicalPolicy,
                    label
                )
            }
        }
    }

    func testAdmittedIDDomainIsDisjointFromFrozenIDDomain() {
        let signatureSHA256 = String(repeating: "3", count: 64)
        func idDigest(_ domain: String) -> String {
            Self.sha256Hex(
                Data(
                    ([
                        domain,
                        "root_key_id=\(Self.rootKeyID)",
                        "policy_sha256=\(Self.goldenPolicySHA256)",
                        "signature_sha256=\(signatureSHA256)",
                    ].joined(separator: "\n") + "\n").utf8
                )
            )
        }

        XCTAssertNotEqual(
            idDigest(Qwen38ScorecardKeyPolicyVerifier.admittedPolicyIDDomain),
            idDigest(OperatorKeyPolicyVerifier.admittedPolicyIDDomain),
            "identical policy identities must mint different admission IDs per domain"
        )
    }

    // MARK: - Revocation confusion

    func testRevocationConfusionRefused() throws {
        let sortedPair = [
            String(repeating: "1", count: 64),
            String(repeating: "2", count: 64),
        ]
        let overCap = (0...256).map { index in
            Self.sha256Hex(Data("revoked-\(index)".utf8))
        }.sorted()

        let structural: [(String, [String])] = [
            ("unsorted list", sortedPair.reversed()),
            ("duplicate entries", [sortedPair[0], sortedPair[0]]),
            ("non-hex entry", [String(repeating: "z", count: 64)]),
            ("over revocation cap", overCap),
        ]
        for (label, revoked) in structural {
            XCTAssertThrowsError(
                try Qwen38ScorecardKeyPolicyVerifier.policyBytes(
                    fields: Self.goldenFields(revokedOperatorKeyIDs: revoked)
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    .nonCanonicalPolicy,
                    label
                )
            }
        }

        let identity: [(String, [String], Qwen38ScorecardKeyPolicyError)] = [
            (
                "root key revoked",
                [Self.rootKeyID],
                .rootKeyRevocationUnsupported
            ),
            (
                "active key revoked",
                [Self.activeKeyID],
                .activeOperatorKeyRevoked
            ),
        ]
        for (label, revoked, expected) in identity {
            let bytes = replacing(
                Self.goldenPolicyText,
                "revoked_operator_key_ids=none",
                with: "revoked_operator_key_ids=\(revoked.joined(separator: ","))"
            )
            let policyFile = try capturePolicy(bytes)
            XCTAssertThrowsError(
                try admitGoldenAnchored(
                    policyFile,
                    expectedPolicySHA256: Self.sha256Hex(bytes)
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    expected,
                    label
                )
            }
        }
    }

    // MARK: - Key identity confusion

    func testActiveKeyIdentityMismatchRefused() throws {
        let bytes = replacing(
            Self.goldenPolicyText,
            "active_operator_key_id=\(Self.activeKeyID)",
            with: "active_operator_key_id=\(String(repeating: "4", count: 64))"
        )
        let policyFile = try capturePolicy(bytes)

        XCTAssertThrowsError(
            try admitGoldenAnchored(
                policyFile,
                expectedPolicySHA256: Self.sha256Hex(bytes)
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardKeyPolicyError,
                .activeOperatorKeyIDMismatch
            )
        }
    }

    func testRootAndActiveKeysMustDiffer() throws {
        // Present the ACTIVE key as the root: policy root_key_id equals the
        // active key ID, and the trust anchor pins the active public key as
        // the root so every anchor/digest check passes first.
        let bytes = replacing(
            Self.goldenPolicyText,
            "root_key_id=\(Self.rootKeyID)",
            with: "root_key_id=\(Self.activeKeyID)"
        )
        let policyFile = try capturePolicy(bytes)

        XCTAssertThrowsError(
            try Qwen38ScorecardKeyPolicyVerifier.admit(
                policyFile: policyFile,
                rootSignatureBase64: Data(repeating: 0, count: 64)
                    .base64EncodedString(),
                trustAnchor: trustAnchor(
                    rootPublicKeyBase64: Self.activePublicKeyBase64,
                    rootKeyID: Self.activeKeyID,
                    expectedPolicySHA256: Self.sha256Hex(bytes)
                )
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardKeyPolicyError,
                .rootAndActiveKeysMustDiffer
            )
        }
    }

    // MARK: - Trust anchor, rollback, and window refusals

    func testTrustAnchorRollbackAndWindowRefusals() throws {
        let (policyFile, signatureBase64) = try signedGoldenPolicy()

        let cases: [(String, Qwen38ScorecardKeyPolicyTrustAnchor, Qwen38ScorecardKeyPolicyError)] = [
            (
                "non-hex anchor root key id",
                trustAnchor(rootKeyID: String(repeating: "G", count: 64)),
                .invalidTrustAnchor(.rootKeyID)
            ),
            (
                "non-hex expected policy digest",
                trustAnchor(
                    expectedPolicySHA256: String(repeating: "x", count: 63)
                ),
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            ),
            (
                "anchor root public key not base64",
                trustAnchor(rootPublicKeyBase64: "@@@"),
                .invalidRootPublicKeyEncoding
            ),
            (
                "anchor root public key does not match anchor id",
                trustAnchor(
                    rootPublicKeyBase64: Self.activePublicKeyBase64
                ),
                .rootPublicKeyIDMismatch
            ),
            (
                "pinned digest mismatch",
                trustAnchor(
                    expectedPolicySHA256: String(repeating: "5", count: 64)
                ),
                .policyDigestMismatch
            ),
            (
                "policy root key differs from anchor",
                trustAnchor(
                    rootPublicKeyBase64: Self.foreignRootPublicKeyBase64(),
                    rootKeyID: Self.foreignRootKeyID()
                ),
                .rootKeyIDMismatch
            ),
            (
                "generation rollback",
                trustAnchor(minimumPolicyGeneration: 8),
                .policyGenerationRollback(minimum: 8, actual: 7)
            ),
            (
                "not yet valid",
                trustAnchor(verificationUnixSeconds: 1_699_999_999),
                .policyNotYetValid
            ),
            (
                "expired",
                trustAnchor(verificationUnixSeconds: 1_800_000_001),
                .policyExpired
            ),
        ]

        for (label, anchor, expected) in cases {
            XCTAssertThrowsError(
                try Qwen38ScorecardKeyPolicyVerifier.admit(
                    policyFile: policyFile,
                    rootSignatureBase64: signatureBase64,
                    trustAnchor: anchor
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    expected,
                    label
                )
            }
        }
    }

    // MARK: - Root signature refusals

    func testRootSignatureRefusals() throws {
        let policyFile = try capturePolicy(Data(Self.goldenPolicyText.utf8))

        let encodings: [(String, String)] = [
            ("not base64", "!!!"),
            (
                "63 bytes",
                Data(repeating: 1, count: 63).base64EncodedString()
            ),
            (
                "65 bytes",
                Data(repeating: 1, count: 65).base64EncodedString()
            ),
        ]
        for (label, signatureBase64) in encodings {
            XCTAssertThrowsError(
                try admitGoldenAnchored(
                    policyFile,
                    rootSignatureBase64: signatureBase64
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    .invalidRootSignatureEncoding,
                    label
                )
            }
        }

        let foreignRoot = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.foreignRootSeed
        )
        let wrongKeySignature = try foreignRoot.signature(
            for: policyFile.bytes
        )
        let root = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.rootSeed
        )
        let wrongBytesSignature = try root.signature(
            for: Data("not the policy".utf8)
        )
        let rejected: [(String, Data)] = [
            ("signed by a different key", wrongKeySignature),
            ("signed over different bytes", wrongBytesSignature),
        ]
        for (label, signature) in rejected {
            XCTAssertThrowsError(
                try admitGoldenAnchored(
                    policyFile,
                    rootSignatureBase64: signature.base64EncodedString()
                ),
                label
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardKeyPolicyError,
                    .rootSignatureRejected,
                    label
                )
            }
        }
    }

    // MARK: - Helpers

    private static func goldenFields(
        revokedOperatorKeyIDs: [String] = []
    ) -> Qwen38ScorecardKeyPolicyFields {
        Qwen38ScorecardKeyPolicyFields(
            rootKeyID: rootKeyID,
            policyGeneration: 7,
            validFromUnixSeconds: 1_700_000_000,
            validUntilUnixSeconds: 1_800_000_000,
            activeOperatorKeyID: activeKeyID,
            activeOperatorPublicKeyBase64: activePublicKeyBase64,
            activeOperatorScope: .scorecardRunClaim,
            allowedClaimSubject: .mtpScorecardResultPair,
            revokedOperatorKeyIDs: revokedOperatorKeyIDs
        )
    }

    private static func foreignRootPublicKeyBase64() -> String {
        let key = try! Curve25519.Signing.PrivateKey(
            rawRepresentation: foreignRootSeed
        )
        return key.publicKey.rawRepresentation.base64EncodedString()
    }

    private static func foreignRootKeyID() -> String {
        let key = try! Curve25519.Signing.PrivateKey(
            rawRepresentation: foreignRootSeed
        )
        return sha256Hex(key.publicKey.rawRepresentation)
    }

    private func signedGoldenPolicy() throws -> (AdmittedFile, String) {
        try signedPolicy(Data(Self.goldenPolicyText.utf8))
    }

    private func signedPolicy(_ bytes: Data) throws -> (AdmittedFile, String) {
        let root = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.rootSeed
        )
        let policyFile = try capturePolicy(bytes)
        let signature = try root.signature(for: policyFile.bytes)
        return (policyFile, signature.base64EncodedString())
    }

    @discardableResult
    private func admitGoldenAnchored(
        _ policyFile: AdmittedFile,
        rootSignatureBase64: String? = nil,
        expectedPolicySHA256: String? = nil
    ) throws -> AdmittedQwen38ScorecardKeyPolicy {
        let signatureBase64: String
        if let rootSignatureBase64 {
            signatureBase64 = rootSignatureBase64
        } else {
            let root = try Curve25519.Signing.PrivateKey(
                rawRepresentation: Self.rootSeed
            )
            signatureBase64 = try root.signature(for: policyFile.bytes)
                .base64EncodedString()
        }
        return try Qwen38ScorecardKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: signatureBase64,
            trustAnchor: trustAnchor(
                expectedPolicySHA256:
                    expectedPolicySHA256 ?? Self.goldenPolicySHA256
            )
        )
    }

    private func trustAnchor(
        rootPublicKeyBase64: String =
            "ebVWLo/mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ=",
        rootKeyID: String =
            "65b60673d6ed884bf01c2c222d82ada0740f29ac3355d6a925c81f17f47a27b8",
        expectedPolicySHA256: String =
            "5a2d5669a6999847655305a003b0cc0c63588297e5a329cdc1f8f9493c90c0e3",
        minimumPolicyGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 1_750_000_000
    ) -> Qwen38ScorecardKeyPolicyTrustAnchor {
        Qwen38ScorecardKeyPolicyTrustAnchor(
            rootPublicKeyBase64: rootPublicKeyBase64,
            rootKeyID: rootKeyID,
            expectedCurrentPolicySHA256: expectedPolicySHA256,
            minimumPolicyGeneration: minimumPolicyGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func capturePolicy(_ bytes: Data) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).policy"
        )
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 65_536
        )
    }

    private func replacing(
        _ source: String,
        _ target: String,
        with replacement: String
    ) -> Data {
        Data(
            source.replacingOccurrences(of: target, with: replacement).utf8
        )
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
